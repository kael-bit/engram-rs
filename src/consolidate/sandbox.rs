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
struct RuleChecker<'a> {
    memories: HashMap<String, &'a Memory>,
    now_ms: i64,
}

impl<'a> RuleChecker<'a> {
    fn new(core: &'a [Memory], working: &'a [Memory]) -> Self {
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

    fn check(&self, op: &AuditOp) -> OpGrade {
        match op {
            AuditOp::Delete { id } => self.check_delete(id, op),
            AuditOp::Demote { id, to } => self.check_demote(id, *to, op),
            AuditOp::Promote { id, to } => self.check_promote(id, *to, op),
            AuditOp::Merge { ids, content, .. } => self.check_merge(ids, content, op),
        }
    }

    fn check_delete(&self, id: &str, op: &AuditOp) -> OpGrade {
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

    fn check_demote(&self, id: &str, to: u8, op: &AuditOp) -> OpGrade {
        let Some(mem) = self.memories.get(id) else {
            return OpGrade { op: op.clone(), grade: Grade::Bad, reason: "target not found".into() };
        };

        // Rule: demoting identity/constraint is almost always wrong
        if mem.tags.iter().any(|t| t == "identity" || t == "constraint") {
            return OpGrade {
                op: op.clone(), grade: Grade::Bad,
                reason: "demoting identity/constraint memory".into(),
            };
        }

        // Rule: demoting a lesson from Core is suspicious
        if mem.layer == Layer::Core && mem.tags.iter().any(|t| t == "lesson") {
            return OpGrade {
                op: op.clone(), grade: Grade::Marginal,
                reason: format!("demoting Core lesson (ac={}) — verify it's stale", mem.access_count),
            };
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

    fn check_promote(&self, id: &str, to: u8, op: &AuditOp) -> OpGrade {
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

        // Rule: promoting to Core with ac=0 is suspicious (no proven value)
        if to == 3 && mem.access_count == 0 {
            return OpGrade {
                op: op.clone(), grade: Grade::Marginal,
                reason: "promoting to Core with zero accesses — unproven value".into(),
            };
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

    fn check_merge(&self, ids: &[String], content: &str, op: &AuditOp) -> OpGrade {
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

    // Call LLM (same model as gate — the audit model)
    let response = crate::ai::llm_chat_as(cfg, "gate", super::audit::AUDIT_SYSTEM_PUB, &prompt).await?;

    // Parse ops without applying
    let ops = super::audit::parse_audit_ops_pub(&response, &core, &working);

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
        prompt.push_str(&format!("- [{}] (imp={:.1}, ac={}, age={:.1}d, mod={:.1}d, kind={}, tags=[{}]) {}\n",
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
        prompt.push_str(&format!("- [{}] (imp={:.1}, ac={}, age={:.1}d, mod={:.1}d, kind={}, tags=[{}]) {}\n",
            crate::util::short_id(&m.id), m.importance,
            m.access_count, age_d, mod_d, m.kind, tags, preview));
    }
    prompt
}
