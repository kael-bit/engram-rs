//! Audit sandbox: dry-run audit against an in-memory snapshot, then score
//! the proposed operations for quality. Nothing touches the real database.
//!
//! Flow:
//! 1. Snapshot current Core+Working into memory
//! 2. Run audit LLM call (same prompt as real audit)
//! 3. Parse ops but do NOT apply
//! 4. Score each op against heuristic rules + optional LLM judge
//! 5. Return a verdict with per-op grades and an overall quality score

use crate::ai::AiConfig;
use crate::consolidate::audit::AuditOp;
use crate::db::{Layer, Memory};
use crate::error::EngramError;
use crate::SharedDB;
use crate::util::truncate_chars;
use serde::Serialize;
use std::collections::HashMap;
use tracing::info;

/// Per-operation quality grade.
#[derive(Debug, Clone, Serialize)]
pub struct OpGrade {
    pub op: AuditOp,
    pub grade: Grade,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub enum Grade {
    /// Clearly correct
    Good,
    /// Acceptable but questionable
    Marginal,
    /// Wrong — would damage memory quality
    Bad,
}

/// Overall sandbox result.
#[derive(Debug, Serialize)]
pub struct SandboxResult {
    pub total_reviewed: usize,
    pub ops_proposed: usize,
    pub grades: Vec<OpGrade>,
    pub score: f64,           // 0.0 - 1.0
    pub safe_to_apply: bool,  // score >= threshold
    pub applied: usize,       // ops actually executed (0 in dry-run)
    pub skipped: usize,       // Bad ops skipped during auto-apply
    pub summary: String,
}

const SAFETY_THRESHOLD: f64 = 0.7;

/// Heuristic rules that catch obviously bad audit decisions.
pub struct RuleChecker<'a> {
    memories: HashMap<String, &'a Memory>,
    now_ms: i64,
}

impl<'a> RuleChecker<'a> {
    pub fn new(core: &'a [Memory], working: &'a [Memory]) -> Self {
        let mut memories = HashMap::new();
        for m in core.iter().chain(working.iter()) {
            memories.insert(m.id.clone(), m);
            // also store short id for lookup
            if m.id.len() >= 8 {
                memories.insert(crate::util::short_id(&m.id).to_string(), m);
            }
        }
        Self {
            memories,
            now_ms: crate::db::now_ms(),
        }
    }

    pub fn check(&self, op: &AuditOp) -> OpGrade {
        match op {
            AuditOp::Delete { id } => self.check_delete(id, op),
            AuditOp::Demote { id, to } => self.check_demote(id, *to, op),
            AuditOp::Promote { id, to } => self.check_promote(id, *to, op),
            AuditOp::Merge { ids, content, .. } => self.check_merge(ids, content, op),
        }
    }

    pub fn check_delete(&self, id: &str, op: &AuditOp) -> OpGrade {
        let Some(mem) = self.memories.get(id) else {
            return OpGrade { op: op.clone(), grade: Grade::Bad, reason: "target not found".into() };
        };

        // Rule: never delete identity memories
        if mem.tags.iter().any(|t| t == "identity" || t == "bootstrap") {
            return OpGrade {
                op: op.clone(), grade: Grade::Bad,
                reason: format!("deleting identity memory (tags: {:?})", mem.tags),
            };
        }

        // Rule: never delete lesson/audit/procedural memories (unless exact duplicates — handled by merge)
        if mem.tags.iter().any(|t| t == "lesson" || t == "audit" || t == "sandbox")
            || mem.kind == "procedural"
        {
            return OpGrade {
                op: op.clone(), grade: Grade::Bad,
                reason: "protected tag: deletion of lesson/audit memories requires manual review".into(),
            };
        }

        // Rule: never delete recently modified memories (24h)
        // Uses modified_at — immune to resume touch inflation on last_accessed.
        let mod_age_h = if mem.modified_at > 0 {
            (self.now_ms - mem.modified_at) as f64 / 3_600_000.0
        } else {
            (self.now_ms - mem.created_at) as f64 / 3_600_000.0
        };
        if mod_age_h < 24.0 {
            return OpGrade {
                op: op.clone(), grade: Grade::Bad,
                reason: format!("deleting memory modified {:.1}h ago (< 24h)", mod_age_h),
            };
        }

        // Rule: deleting high-access memories is suspicious
        if mem.access_count > 20 {
            return OpGrade {
                op: op.clone(), grade: Grade::Marginal,
                reason: format!("deleting memory with {} accesses — likely valuable", mem.access_count),
            };
        }

        // Rule: deleting Core memory is always risky
        if mem.layer == Layer::Core {
            return OpGrade {
                op: op.clone(), grade: Grade::Marginal,
                reason: "deleting Core memory — verify it's truly obsolete".into(),
            };
        }

        OpGrade { op: op.clone(), grade: Grade::Good, reason: "looks safe to delete".into() }
    }

    pub fn check_demote(&self, id: &str, to: u8, op: &AuditOp) -> OpGrade {
        let Some(mem) = self.memories.get(id) else {
            return OpGrade { op: op.clone(), grade: Grade::Bad, reason: "target not found".into() };
        };

        // Rule: same-layer or upward demote is a no-op
        if mem.layer as u8 <= to {
            return OpGrade {
                op: op.clone(), grade: Grade::Bad,
                reason: format!("same-layer or upward demote is a no-op (L{} → L{})", mem.layer as u8, to),
            };
        }

        // Rule: demoting identity/constraint is almost always wrong
        if mem.tags.iter().any(|t| t == "identity" || t == "constraint") {
            return OpGrade {
                op: op.clone(), grade: Grade::Bad,
                reason: "demoting identity/constraint memory".into(),
            };
        }

        // Rule: lessons and constraints must stay in Core — they ARE Core
        if mem.layer == Layer::Core && mem.tags.iter().any(|t| t == "lesson") {
            return OpGrade {
                op: op.clone(), grade: Grade::Bad,
                reason: format!("demoting Core lesson (ac={}) — lessons belong in Core", mem.access_count),
            };
        }

        // Rule: memories describing design mistakes / failures belong in Core
        if mem.layer == Layer::Core {
            let content_lower = mem.content.to_lowercase();
            let is_lesson_content = content_lower.contains("设计错误")
                || content_lower.contains("教训")
                || content_lower.contains("lesson")
                || content_lower.contains("原则")
                || content_lower.contains("principle")
                || content_lower.contains("严重失误")
                || content_lower.contains("must")
                || content_lower.contains("必须");
            if is_lesson_content {
                return OpGrade {
                    op: op.clone(), grade: Grade::Bad,
                    reason: "demoting Core memory containing lesson/principle language".into(),
                };
            }
        }

        // Rule: demoting high-ac Core memory
        if mem.layer == Layer::Core && mem.access_count > 50 {
            return OpGrade {
                op: op.clone(), grade: Grade::Bad,
                reason: format!("demoting Core memory with ac={} — heavily used", mem.access_count),
            };
        }

        // Rule: demoting to buffer (1) is aggressive
        if to == 1 {
            return OpGrade {
                op: op.clone(), grade: Grade::Marginal,
                reason: "demoting straight to Buffer — will expire soon".into(),
            };
        }

        OpGrade { op: op.clone(), grade: Grade::Good, reason: "reasonable demotion".into() }
    }

    pub fn check_promote(&self, id: &str, to: u8, op: &AuditOp) -> OpGrade {
        let Some(mem) = self.memories.get(id) else {
            return OpGrade { op: op.clone(), grade: Grade::Bad, reason: "target not found".into() };
        };

        // Rule: promoting session/distilled to Core is wrong
        if to == 3 && mem.tags.iter().any(|t| t == "session" || t == "distilled" || t == "auto-distilled") {
            return OpGrade {
                op: op.clone(), grade: Grade::Bad,
                reason: "promoting session/distilled memory to Core".into(),
            };
        }

        // Rule: promoting to Core with ac=0 is suspicious UNLESS recently created OR has lesson/constraint content
        if to == 3 && mem.access_count == 0 {
            let age_h = (self.now_ms - mem.created_at) as f64 / 3_600_000.0;
            let is_lesson_content = mem.tags.iter().any(|t| t == "lesson" || t == "constraint")
                || mem.content.to_lowercase().contains("lesson")
                || mem.content.to_lowercase().contains("教训")
                || mem.content.to_lowercase().contains("原则")
                || mem.content.to_lowercase().contains("严禁");
            if !(age_h < 48.0 || is_lesson_content) {
                return OpGrade {
                    op: op.clone(), grade: Grade::Marginal,
                    reason: "promoting to Core with zero accesses — unproven value".into(),
                };
            }
        }

        // Rule: promoting gate-rejected-final should not happen
        if to == 3 && mem.tags.iter().any(|t| t == "gate-rejected-final") {
            return OpGrade {
                op: op.clone(), grade: Grade::Bad,
                reason: "promoting gate-rejected-final to Core — gate decided 3x already".into(),
            };
        }

        OpGrade { op: op.clone(), grade: Grade::Good, reason: "reasonable promotion".into() }
    }

    pub fn check_merge(&self, ids: &[String], content: &str, op: &AuditOp) -> OpGrade {
        // Rule: merged content must not be shorter than the shortest input
        let min_len = ids.iter()
            .filter_map(|id| self.memories.get(id.as_str()))
            .map(|m| m.content.chars().count())
            .min()
            .unwrap_or(0);

        if content.chars().count() < min_len / 2 {
            return OpGrade {
                op: op.clone(), grade: Grade::Bad,
                reason: format!("merge output ({} chars) much shorter than inputs — likely losing information",
                    content.chars().count()),
            };
        }

        // Rule: merging identity with non-identity is wrong
        let has_identity = ids.iter().any(|id|
            self.memories.get(id.as_str())
                .map(|m| m.tags.iter().any(|t| t == "identity"))
                .unwrap_or(false)
        );
        let has_non_identity = ids.iter().any(|id|
            self.memories.get(id.as_str())
                .map(|m| !m.tags.iter().any(|t| t == "identity"))
                .unwrap_or(false)
        );
        if has_identity && has_non_identity {
            return OpGrade {
                op: op.clone(), grade: Grade::Bad,
                reason: "merging identity with non-identity memory".into(),
            };
        }

        // Rule: merging across layers (Core + Working) is risky
        let layers: Vec<Layer> = ids.iter()
            .filter_map(|id| self.memories.get(id.as_str()).map(|m| m.layer))
            .collect();
        if layers.contains(&Layer::Core) && layers.contains(&Layer::Working) {
            return OpGrade {
                op: op.clone(), grade: Grade::Marginal,
                reason: "merging across Core and Working layers".into(),
            };
        }

        // Rule: can't find all source memories
        let found = ids.iter().filter(|id| self.memories.contains_key(id.as_str())).count();
        if found < ids.len() {
            return OpGrade {
                op: op.clone(), grade: Grade::Bad,
                reason: format!("only {}/{} merge source IDs found", found, ids.len()),
            };
        }

        OpGrade { op: op.clone(), grade: Grade::Good, reason: "reasonable merge".into() }
    }
}

/// Run audit in sandbox mode: snapshot → audit → grade → optionally apply.
/// When `auto_apply` is true and score >= threshold, executes Good+Marginal ops,
/// skips Bad ops. When false, nothing is modified (dry-run).
pub async fn sandbox_audit(cfg: &AiConfig, db: &SharedDB, auto_apply: bool) -> Result<SandboxResult, EngramError> {
    let db2 = db.clone();
    let (core, working) = tokio::task::spawn_blocking(move || {
        let c = db2.list_by_layer_meta(Layer::Core, 500, 0).unwrap_or_default();
        let w = db2.list_by_layer_meta(Layer::Working, 500, 0).unwrap_or_default();
        (c, w)
    }).await.map_err(|e| EngramError::Internal(format!("spawn: {e}")))?;

    let total = core.len() + working.len();
    info!(core = core.len(), working = working.len(), "audit sandbox: snapshotted memories");

    // Build the same prompt the real audit uses
    let prompt = format_sandbox_prompt(&core, &working);

    // Call LLM with function calling (same model as gate — the audit model)
    let tcr = crate::ai::llm_tool_call::<super::audit::AuditToolResponse>(
        cfg, "audit", super::audit::AUDIT_SYSTEM_PUB, &prompt,
        "audit_operations",
        "Propose cleanup operations for the memory store. Return empty operations array if nothing needs changing.",
        super::audit::audit_tool_schema(),
    ).await?;
    if let Some(ref u) = tcr.usage {
        let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
        let _ = db.log_llm_call("sandbox_audit", &tcr.model, u.prompt_tokens, u.completion_tokens, cached, tcr.duration_ms);
    }

    // Resolve raw tool call ops without applying
    let ops = super::audit::resolve_audit_ops(tcr.value.operations, &core, &working);

    info!(ops = ops.len(), "audit sandbox: LLM proposed operations");

    // Grade each op
    let checker = RuleChecker::new(&core, &working);
    let grades: Vec<OpGrade> = ops.iter().map(|op| checker.check(op)).collect();

    // Compute score
    let (good, marginal, bad) = grades.iter().fold((0, 0, 0), |(g, m, b), og| {
        match og.grade {
            Grade::Good => (g + 1, m, b),
            Grade::Marginal => (g, m + 1, b),
            Grade::Bad => (g, m, b + 1),
        }
    });

    let score = if grades.is_empty() {
        1.0 // no ops = nothing to mess up
    } else {
        let total_ops = grades.len() as f64;
        (good as f64 + marginal as f64 * 0.5) / total_ops
    };

    let safe = score >= SAFETY_THRESHOLD;

    // Auto-apply: execute Good+Marginal ops, skip Bad
    let (mut applied, mut skipped) = (0usize, 0usize);
    if safe && auto_apply {
        let safe_ops: Vec<AuditOp> = grades.iter()
            .filter(|g| g.grade != Grade::Bad)
            .map(|g| g.op.clone())
            .collect();
        let bad_count = grades.iter().filter(|g| g.grade == Grade::Bad).count();
        skipped = bad_count;

        if !safe_ops.is_empty() {
            let db3 = db.clone();
            let op_count = safe_ops.len();
            let actual = tokio::task::spawn_blocking(move || {
                let mut r = super::audit::AuditResult::default();
                super::audit::apply_audit_ops_pub(&db3, safe_ops, &mut r);
                r
            }).await.map_err(|e| EngramError::Internal(format!("spawn: {e}")))?;
            applied = actual.promoted + actual.demoted + actual.deleted + actual.merged;
            info!(applied, skipped, op_count, "audit sandbox: auto-applied ops");
        }
    }

    let summary = format!(
        "Reviewed {} memories ({} Core + {} Working). \
         Audit proposed {} ops: {} good, {} marginal, {} bad. \
         Score: {:.0}% (threshold: {:.0}%). {}",
        total, core.len(), working.len(),
        grades.len(), good, marginal, bad,
        score * 100.0, SAFETY_THRESHOLD * 100.0,
        if !safe {
            "NOT safe — no changes applied.".to_string()
        } else if auto_apply {
            format!("Applied {} ops, skipped {} bad.", applied, skipped)
        } else {
            "Safe to apply (dry-run).".to_string()
        }
    );

    Ok(SandboxResult {
        total_reviewed: total,
        ops_proposed: grades.len(),
        grades,
        score,
        safe_to_apply: safe,
        applied,
        skipped,
        summary,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consolidate::audit::AuditOp;
    use crate::db::{Layer, Memory};

    /// Helper to create a Memory with sensible defaults. Caller overrides as needed.
    fn mem(id: &str, layer: Layer, tags: Vec<&str>, content: &str) -> Memory {
        let now = crate::db::now_ms();
        Memory {
            id: id.to_string(),
            content: content.to_string(),
            layer,
            importance: 0.5,
            created_at: now - 7 * 86_400_000,  // 7 days ago
            last_accessed: now - 86_400_000,
            access_count: 5,
            repetition_count: 0,
            decay_rate: layer.default_decay(),
            source: "test".into(),
            tags: tags.into_iter().map(String::from).collect(),
            namespace: "default".into(),
            embedding: None,
            kind: "semantic".into(),
            modified_at: now - 3 * 86_400_000,  // 3 days ago
        }
    }

    // ──────────────────────────────────────────────
    //  check_delete tests
    // ──────────────────────────────────────────────

    #[test]
    fn delete_identity_tagged_is_bad() {
        let core = [mem("id-001", Layer::Core, vec!["identity"], "I am atlas")];
        let working: [Memory; 0] = [];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Delete { id: "id-001".into() });
        assert_eq!(grade.grade, Grade::Bad);
        assert!(grade.reason.contains("identity"));
    }

    #[test]
    fn delete_bootstrap_tagged_is_bad() {
        let core = [mem("id-002", Layer::Core, vec!["bootstrap"], "bootstrap config")];
        let working: [Memory; 0] = [];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Delete { id: "id-002".into() });
        assert_eq!(grade.grade, Grade::Bad);
    }

    #[test]
    fn delete_lesson_tagged_is_bad() {
        let core = [mem("id-003", Layer::Core, vec!["lesson"], "never force-push")];
        let working: [Memory; 0] = [];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Delete { id: "id-003".into() });
        assert_eq!(grade.grade, Grade::Bad);
        assert!(grade.reason.contains("lesson"));
    }

    #[test]
    fn delete_procedural_kind_is_bad() {
        let mut m = mem("id-004", Layer::Working, vec!["deploy"], "step1 step2");
        m.kind = "procedural".into();
        let core: [Memory; 0] = [];
        let working = [m];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Delete { id: "id-004".into() });
        assert_eq!(grade.grade, Grade::Bad);
        assert!(grade.reason.contains("lesson") || grade.reason.contains("procedural") || grade.reason.contains("protected"));
    }

    #[test]
    fn delete_recently_modified_is_bad() {
        let now = crate::db::now_ms();
        let mut m = mem("id-005", Layer::Working, vec![], "recent stuff");
        m.modified_at = now - 3_600_000; // 1 hour ago
        let core: [Memory; 0] = [];
        let working = [m];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Delete { id: "id-005".into() });
        assert_eq!(grade.grade, Grade::Bad);
        assert!(grade.reason.contains("24h") || grade.reason.contains("modified"));
    }

    #[test]
    fn delete_high_access_count_is_marginal() {
        let mut m = mem("id-006", Layer::Working, vec![], "popular memory");
        m.access_count = 150;
        let core: [Memory; 0] = [];
        let working = [m];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Delete { id: "id-006".into() });
        assert_eq!(grade.grade, Grade::Marginal);
        assert!(grade.reason.contains("accesses") || grade.reason.contains("valuable"));
    }

    #[test]
    fn delete_core_memory_is_marginal() {
        let core = [mem("id-007", Layer::Core, vec![], "some core fact")];
        let working: [Memory; 0] = [];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Delete { id: "id-007".into() });
        assert_eq!(grade.grade, Grade::Marginal);
        assert!(grade.reason.contains("Core"));
    }

    #[test]
    fn delete_normal_working_memory_is_good() {
        let core: [Memory; 0] = [];
        let working = [mem("id-008", Layer::Working, vec![], "ephemeral note")];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Delete { id: "id-008".into() });
        assert_eq!(grade.grade, Grade::Good);
    }

    #[test]
    fn delete_not_found_is_bad() {
        let core: [Memory; 0] = [];
        let working: [Memory; 0] = [];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Delete { id: "nonexistent".into() });
        assert_eq!(grade.grade, Grade::Bad);
        assert!(grade.reason.contains("not found"));
    }

    // ──────────────────────────────────────────────
    //  check_demote tests
    // ──────────────────────────────────────────────

    #[test]
    fn demote_same_layer_is_bad() {
        let core: [Memory; 0] = [];
        let working = [mem("dm-001", Layer::Working, vec![], "some working memory")];
        let checker = RuleChecker::new(&core, &working);
        // Working = 2, demote to 2 is same-layer
        let grade = checker.check(&AuditOp::Demote { id: "dm-001".into(), to: 2 });
        assert_eq!(grade.grade, Grade::Bad);
        assert!(grade.reason.contains("same-layer") || grade.reason.contains("no-op"));
    }

    #[test]
    fn demote_upward_is_bad() {
        let core: [Memory; 0] = [];
        let working = [mem("dm-002", Layer::Working, vec![], "some working memory")];
        let checker = RuleChecker::new(&core, &working);
        // Working = 2, demote to 3 (Core) is upward
        let grade = checker.check(&AuditOp::Demote { id: "dm-002".into(), to: 3 });
        assert_eq!(grade.grade, Grade::Bad);
        assert!(grade.reason.contains("upward") || grade.reason.contains("no-op"));
    }

    #[test]
    fn demote_lesson_from_core_is_bad() {
        let core = [mem("dm-003", Layer::Core, vec!["lesson"], "never do X")];
        let working: [Memory; 0] = [];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Demote { id: "dm-003".into(), to: 2 });
        assert_eq!(grade.grade, Grade::Bad);
        assert!(grade.reason.contains("lesson") || grade.reason.contains("Core"));
    }

    #[test]
    fn demote_content_with_constraint_keywords_is_bad() {
        let core = [mem("dm-004", Layer::Core, vec![], "部署必须先跑测试")];
        let working: [Memory; 0] = [];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Demote { id: "dm-004".into(), to: 2 });
        assert_eq!(grade.grade, Grade::Bad);
        assert!(grade.reason.contains("lesson") || grade.reason.contains("principle"));
    }

    #[test]
    fn demote_content_with_principle_keyword_is_bad() {
        let core = [mem("dm-005", Layer::Core, vec![], "设计原则: 单一职责")];
        let working: [Memory; 0] = [];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Demote { id: "dm-005".into(), to: 2 });
        assert_eq!(grade.grade, Grade::Bad);
        assert!(grade.reason.contains("principle") || grade.reason.contains("lesson"));
    }

    #[test]
    fn demote_normal_core_to_working_is_good() {
        let core = [mem("dm-006", Layer::Core, vec![], "some outdated fact")];
        let working: [Memory; 0] = [];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Demote { id: "dm-006".into(), to: 2 });
        assert_eq!(grade.grade, Grade::Good);
    }

    #[test]
    fn demote_core_to_buffer_is_marginal() {
        let core = [mem("dm-007", Layer::Core, vec![], "some outdated fact")];
        let working: [Memory; 0] = [];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Demote { id: "dm-007".into(), to: 1 });
        assert_eq!(grade.grade, Grade::Marginal);
        assert!(grade.reason.contains("Buffer"));
    }

    #[test]
    fn demote_identity_is_bad() {
        let core = [mem("dm-008", Layer::Core, vec!["identity"], "I am atlas")];
        let working: [Memory; 0] = [];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Demote { id: "dm-008".into(), to: 2 });
        assert_eq!(grade.grade, Grade::Bad);
        assert!(grade.reason.contains("identity"));
    }

    #[test]
    fn demote_high_ac_core_is_bad() {
        let mut m = mem("dm-009", Layer::Core, vec![], "very popular");
        m.access_count = 100;
        let core = [m];
        let working: [Memory; 0] = [];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Demote { id: "dm-009".into(), to: 2 });
        assert_eq!(grade.grade, Grade::Bad);
        assert!(grade.reason.contains("heavily used") || grade.reason.contains("ac="));
    }

    // ──────────────────────────────────────────────
    //  check_promote tests
    // ──────────────────────────────────────────────

    #[test]
    fn promote_session_tagged_to_core_is_bad() {
        let core: [Memory; 0] = [];
        let working = [mem("pr-001", Layer::Working, vec!["session"], "did X today")];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Promote { id: "pr-001".into(), to: 3 });
        assert_eq!(grade.grade, Grade::Bad);
        assert!(grade.reason.contains("session"));
    }

    #[test]
    fn promote_distilled_tagged_to_core_is_bad() {
        let core: [Memory; 0] = [];
        let working = [mem("pr-002", Layer::Working, vec!["distilled"], "distilled summary")];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Promote { id: "pr-002".into(), to: 3 });
        assert_eq!(grade.grade, Grade::Bad);
        assert!(grade.reason.contains("distilled") || grade.reason.contains("session"));
    }

    #[test]
    fn promote_auto_distilled_tagged_to_core_is_bad() {
        let core: [Memory; 0] = [];
        let working = [mem("pr-003", Layer::Working, vec!["auto-distilled"], "auto summary")];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Promote { id: "pr-003".into(), to: 3 });
        assert_eq!(grade.grade, Grade::Bad);
    }

    #[test]
    fn promote_zero_ac_old_memory_to_core_is_marginal() {
        let now = crate::db::now_ms();
        let mut m = mem("pr-004", Layer::Working, vec![], "old untouched note");
        m.access_count = 0;
        m.created_at = now - 72 * 3_600_000; // 72h ago (> 48h)
        let core: [Memory; 0] = [];
        let working = [m];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Promote { id: "pr-004".into(), to: 3 });
        assert_eq!(grade.grade, Grade::Marginal);
        assert!(grade.reason.contains("zero accesses") || grade.reason.contains("unproven"));
    }

    #[test]
    fn promote_zero_ac_recent_memory_to_core_is_good() {
        let now = crate::db::now_ms();
        let mut m = mem("pr-005", Layer::Working, vec![], "fresh new insight");
        m.access_count = 0;
        m.created_at = now - 12 * 3_600_000; // 12h ago (< 48h)
        let core: [Memory; 0] = [];
        let working = [m];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Promote { id: "pr-005".into(), to: 3 });
        assert_eq!(grade.grade, Grade::Good);
    }

    #[test]
    fn promote_normal_memory_to_core_is_good() {
        let core: [Memory; 0] = [];
        let working = [mem("pr-006", Layer::Working, vec!["design"], "REST API pattern")];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Promote { id: "pr-006".into(), to: 3 });
        assert_eq!(grade.grade, Grade::Good);
    }

    #[test]
    fn promote_gate_rejected_final_is_bad() {
        let core: [Memory; 0] = [];
        let working = [mem("pr-007", Layer::Working, vec!["gate-rejected-final"], "rejected stuff")];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Promote { id: "pr-007".into(), to: 3 });
        assert_eq!(grade.grade, Grade::Bad);
        assert!(grade.reason.contains("gate"));
    }

    // ──────────────────────────────────────────────
    //  check_merge tests
    // ──────────────────────────────────────────────

    #[test]
    fn merge_content_too_short_is_bad() {
        let m1 = mem("mg-001", Layer::Working, vec![], "a moderately long piece of content that has useful information");
        let m2 = mem("mg-002", Layer::Working, vec![], "another piece of content with details");
        let core: [Memory; 0] = [];
        let working = [m1, m2];
        let checker = RuleChecker::new(&core, &working);
        // merged content is much shorter than the shortest input
        let grade = checker.check(&AuditOp::Merge {
            ids: vec!["mg-001".into(), "mg-002".into()],
            content: "short".into(),
            layer: 2,
            tags: vec![],
        });
        assert_eq!(grade.grade, Grade::Bad);
        assert!(grade.reason.contains("shorter") || grade.reason.contains("losing"));
    }

    #[test]
    fn merge_normal_is_good() {
        let m1 = mem("mg-003", Layer::Working, vec![], "fact A about deploy");
        let m2 = mem("mg-004", Layer::Working, vec![], "fact B about deploy");
        let core: [Memory; 0] = [];
        let working = [m1, m2];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Merge {
            ids: vec!["mg-003".into(), "mg-004".into()],
            content: "Combined facts A and B about deploy workflow and configuration".into(),
            layer: 2,
            tags: vec![],
        });
        assert_eq!(grade.grade, Grade::Good);
    }

    #[test]
    fn merge_identity_with_non_identity_is_bad() {
        let m1 = mem("mg-005", Layer::Core, vec!["identity"], "I am atlas");
        let m2 = mem("mg-006", Layer::Core, vec![], "some other fact");
        let core = [m1, m2];
        let working: [Memory; 0] = [];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Merge {
            ids: vec!["mg-005".into(), "mg-006".into()],
            content: "merged identity and fact".into(),
            layer: 3,
            tags: vec![],
        });
        assert_eq!(grade.grade, Grade::Bad);
        assert!(grade.reason.contains("identity"));
    }

    #[test]
    fn merge_cross_layer_is_marginal() {
        let core = [mem("mg-007", Layer::Core, vec![], "core fact")];
        let working = [mem("mg-008", Layer::Working, vec![], "working note")];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Merge {
            ids: vec!["mg-007".into(), "mg-008".into()],
            content: "combined core and working content into one merged memory".into(),
            layer: 3,
            tags: vec![],
        });
        assert_eq!(grade.grade, Grade::Marginal);
        assert!(grade.reason.contains("Core") && grade.reason.contains("Working"));
    }

    #[test]
    fn merge_missing_source_ids_is_bad() {
        let core: [Memory; 0] = [];
        let working = [mem("mg-009", Layer::Working, vec![], "existing note")];
        let checker = RuleChecker::new(&core, &working);
        let grade = checker.check(&AuditOp::Merge {
            ids: vec!["mg-009".into(), "mg-nonexist".into()],
            content: "merged something".into(),
            layer: 2,
            tags: vec![],
        });
        assert_eq!(grade.grade, Grade::Bad);
        assert!(grade.reason.contains("found") || grade.reason.contains("source"));
    }

    // ──────────────────────────────────────────────
    //  Score formula tests
    // ──────────────────────────────────────────────

    #[test]
    fn score_formula_all_good() {
        // 3 Good, 0 Marginal, 0 Bad → (3 + 0) / 3 = 1.0
        let score = (3.0_f64 + 0.0 * 0.5) / 3.0;
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn score_formula_mixed() {
        // 2 Good, 2 Marginal, 1 Bad → (2 + 2*0.5) / 5 = 3/5 = 0.6
        let good = 2;
        let marginal = 2;
        let total = 5;
        let score = (good as f64 + marginal as f64 * 0.5) / total as f64;
        assert!((score - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn score_formula_all_bad() {
        // 0 Good, 0 Marginal, 3 Bad → 0 / 3 = 0.0
        let score = (0.0_f64 + 0.0 * 0.5) / 3.0;
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn score_formula_empty_is_one() {
        // Edge case: no ops → score = 1.0
        let grades: Vec<OpGrade> = vec![];
        let score = if grades.is_empty() {
            1.0
        } else {
            let total_ops = grades.len() as f64;
            let (good, marginal, _bad) = grades.iter().fold((0, 0, 0), |(g, m, b), og| {
                match og.grade {
                    Grade::Good => (g + 1, m, b),
                    Grade::Marginal => (g, m + 1, b),
                    Grade::Bad => (g, m, b + 1),
                }
            });
            (good as f64 + marginal as f64 * 0.5) / total_ops
        };
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn score_formula_marginal_only() {
        // 0 Good, 4 Marginal, 0 Bad → (0 + 4*0.5) / 4 = 0.5
        let score: f64 = (0.0 + 4.0 * 0.5) / 4.0;
        assert!((score - 0.5).abs() < f64::EPSILON);
    }

    // ──────────────────────────────────────────────
    //  Integration: check() dispatches correctly
    // ──────────────────────────────────────────────

    #[test]
    fn check_dispatches_to_correct_method() {
        let core: [Memory; 0] = [];
        let working = [mem("disp-001", Layer::Working, vec![], "normal memory")];
        let checker = RuleChecker::new(&core, &working);

        // Delete
        let g = checker.check(&AuditOp::Delete { id: "disp-001".into() });
        assert_eq!(g.grade, Grade::Good);

        // Promote (Working → Core)
        let g = checker.check(&AuditOp::Promote { id: "disp-001".into(), to: 3 });
        assert_eq!(g.grade, Grade::Good);
    }
}

fn format_sandbox_prompt(core: &[Memory], working: &[Memory]) -> String {
    let now = crate::db::now_ms();
    let mut prompt = String::with_capacity(16_000);
    prompt.push_str("## Core Layer\n");
    for m in core {
        let tags = m.tags.join(",");
        let age_d = (now - m.created_at) as f64 / 86_400_000.0;
        let mod_d = if m.modified_at > 0 {
            (now - m.modified_at) as f64 / 86_400_000.0
        } else {
            age_d
        };
        let preview = truncate_chars(&m.content, 200);
        prompt.push_str(&format!("- [{}] [Layer: Core (3)] (imp={:.1}, ac={}, age={:.1}d, mod={:.1}d, kind={}, tags=[{}]) {}\n",
            crate::util::short_id(&m.id), m.importance,
            m.access_count, age_d, mod_d, m.kind, tags, preview));
    }
    prompt.push_str(&format!("\n## Working Layer ({} memories)\n", working.len()));
    for m in working {
        let tags = m.tags.join(",");
        let age_d = (now - m.created_at) as f64 / 86_400_000.0;
        let mod_d = if m.modified_at > 0 {
            (now - m.modified_at) as f64 / 86_400_000.0
        } else {
            age_d
        };
        let preview = truncate_chars(&m.content, 200);
        prompt.push_str(&format!("- [{}] [Layer: Working (2)] (imp={:.1}, ac={}, age={:.1}d, mod={:.1}d, kind={}, tags=[{}]) {}\n",
            crate::util::short_id(&m.id), m.importance,
            m.access_count, age_d, mod_d, m.kind, tags, preview));
    }
    prompt
}
