//! Audit sandbox: dry-run audit against an in-memory snapshot, then score
//! the proposed operations for quality. Nothing touches the real database.
//!
//! Flow:
//! 1. Snapshot current Core+Working into memory
//! 2. Cluster memories by semantic similarity
//! 3. Batch clusters into groups that fit prompt limits
//! 4. Run audit LLM call per batch with full content + similarity context
//! 5. Parse ops but do NOT apply (unless auto_apply=true)
//! 6. Score each op against simplified mechanical rules
//! 7. Return a verdict with per-op grades and an overall quality score

use crate::ai::AiConfig;
use crate::consolidate::audit::AuditOp;
use crate::consolidate::cluster::{batch_clusters, cluster_memories, MemoryCluster};
use crate::db::{Layer, Memory};
use crate::error::EngramError;
use crate::SharedDB;
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
    pub cluster_count: usize, // number of semantic clusters
    pub batch_count: usize,   // number of LLM batches
}

const SAFETY_THRESHOLD: f64 = crate::thresholds::SANDBOX_SAFETY_THRESHOLD;

/// Cosine+tag combined similarity threshold for clustering.
/// Higher than pure cosine 0.50 because tag_jaccard inflates scores.
const CLUSTER_THRESHOLD: f64 = 0.55;

/// Maximum characters per batch prompt.
const BATCH_MAX_CHARS: usize = 8_000;

/// Simplified mechanical rules that catch obviously bad audit decisions.
/// Only 3 categories of errors:
/// 1. Target not found
/// 2. Same-layer / upward demote (no-op)
/// 3. Merge losing information
pub struct RuleChecker<'a> {
    memories: HashMap<String, &'a Memory>,
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
        Self { memories }
    }

    pub fn check(&self, op: &AuditOp) -> OpGrade {
        match op {
            AuditOp::Delete { id } => {
                if !self.memories.contains_key(id) {
                    return bad(op, "target not found");
                }
                // Audit cannot delete — only demote. Lifecycle handles deletion.
                bad(op, "audit cannot delete memories — use demote instead, let lifecycle handle expiry")
            }
            AuditOp::Demote { id, to } => {
                if let Some(mem) = self.memories.get(id) {
                    if (mem.layer as u8) <= *to {
                        return bad(
                            op,
                            &format!(
                                "same-layer or upward demote is a no-op (L{} → L{})",
                                mem.layer as u8, to
                            ),
                        );
                    }
                } else {
                    return bad(op, "target not found");
                }
                good(op, "ok")
            }
            AuditOp::Promote { id, to: _ } => {
                if !self.memories.contains_key(id) {
                    return bad(op, "target not found");
                }
                good(op, "ok")
            }
            AuditOp::Merge { ids, content, .. } => {
                let found = ids
                    .iter()
                    .filter(|id| self.memories.contains_key(id.as_str()))
                    .count();
                if found < ids.len() {
                    return bad(
                        op,
                        &format!("only {}/{} merge source IDs found", found, ids.len()),
                    );
                }
                let min_len = ids
                    .iter()
                    .filter_map(|id| self.memories.get(id.as_str()))
                    .map(|m| m.content.chars().count())
                    .min()
                    .unwrap_or(0);
                if content.chars().count() < min_len / 2 {
                    return bad(
                        op,
                        "merge output much shorter than inputs — losing information",
                    );
                }
                good(op, "ok")
            }
        }
    }
}

fn good(op: &AuditOp, reason: &str) -> OpGrade {
    OpGrade {
        op: op.clone(),
        grade: Grade::Good,
        reason: reason.to_string(),
    }
}

fn bad(op: &AuditOp, reason: &str) -> OpGrade {
    OpGrade {
        op: op.clone(),
        grade: Grade::Bad,
        reason: reason.to_string(),
    }
}

/// Run audit in sandbox mode: snapshot → cluster → batch → audit → grade → optionally apply.
///
/// When `auto_apply` is true and score >= threshold, executes Good ops,
/// skips Bad ops. When false, nothing is modified (dry-run).
pub async fn sandbox_audit(
    cfg: &AiConfig,
    db: &SharedDB,
    auto_apply: bool,
) -> Result<SandboxResult, EngramError> {
    let db2 = db.clone();
    let (core, working) = tokio::task::spawn_blocking(move || {
        let c = db2
            .list_by_layer_meta(Layer::Core, 500, 0)
            .unwrap_or_default();
        let w = db2
            .list_by_layer_meta(Layer::Working, 500, 0)
            .unwrap_or_default();
        (c, w)
    })
    .await
    .map_err(|e| EngramError::Internal(format!("spawn: {e}")))?;

    let total = core.len() + working.len();
    info!(
        core = core.len(),
        working = working.len(),
        "audit sandbox: snapshotted memories"
    );

    // Get embeddings for clustering
    let all_ids: Vec<String> = core
        .iter()
        .chain(working.iter())
        .map(|m| m.id.clone())
        .collect();
    let db3 = db.clone();
    let embeddings = tokio::task::spawn_blocking(move || db3.get_embeddings_by_ids(&all_ids))
        .await
        .map_err(|e| EngramError::Internal(format!("spawn: {e}")))?;

    // Cluster memories by semantic similarity
    let all_memories: Vec<Memory> = core
        .iter()
        .chain(working.iter())
        .cloned()
        .collect();
    let clusters = cluster_memories(&all_memories, &embeddings, CLUSTER_THRESHOLD);
    let cluster_count = clusters.len();

    info!(
        clusters = cluster_count,
        memories = total,
        "audit sandbox: clustered memories"
    );

    // Batch clusters into groups that fit prompt limits
    let batch_indices = batch_clusters(&clusters, BATCH_MAX_CHARS);
    let batch_count = batch_indices.len();

    info!(
        batches = batch_count,
        "audit sandbox: batched clusters"
    );

    // Audit each batch
    let mut all_ops: Vec<AuditOp> = Vec::new();
    for (batch_num, indices) in batch_indices.iter().enumerate() {
        let batch_clusters: Vec<&MemoryCluster> =
            indices.iter().map(|&i| &clusters[i]).collect();
        let prompt = format_clustered_prompt(&batch_clusters, batch_num + 1, batch_count);

        let tcr = crate::ai::llm_tool_call::<super::audit::AuditToolResponse>(
            cfg,
            "audit",
            super::audit::AUDIT_SYSTEM_PUB,
            &prompt,
            "audit_operations",
            "Propose cleanup operations for the memory store. Return empty operations array if nothing needs changing.",
            super::audit::audit_tool_schema(),
        )
        .await?;

        if let Some(ref u) = tcr.usage {
            let cached = u
                .prompt_tokens_details
                .as_ref()
                .map_or(0, |d| d.cached_tokens);
            let _ = db.log_llm_call(
                "sandbox_audit",
                &tcr.model,
                u.prompt_tokens,
                u.completion_tokens,
                cached,
                tcr.duration_ms,
            );
        }

        let ops = super::audit::resolve_audit_ops(tcr.value.operations, &core, &working);
        info!(
            batch = batch_num + 1,
            ops = ops.len(),
            "audit sandbox: batch ops"
        );
        all_ops.extend(ops);
    }

    info!(
        ops = all_ops.len(),
        "audit sandbox: total LLM proposed operations"
    );

    // Grade each op with simplified mechanical checker
    let checker = RuleChecker::new(&core, &working);
    let grades: Vec<OpGrade> = all_ops.iter().map(|op| checker.check(op)).collect();

    // Compute score
    let (good_count, _marginal_count, bad_count) =
        grades.iter().fold((0, 0, 0), |(g, m, b), og| match og.grade {
            Grade::Good => (g + 1, m, b),
            Grade::Marginal => (g, m + 1, b),
            Grade::Bad => (g, m, b + 1),
        });

    let score = if grades.is_empty() {
        1.0 // no ops = nothing to mess up
    } else {
        let total_ops = grades.len() as f64;
        good_count as f64 / total_ops
    };

    let safe = score >= SAFETY_THRESHOLD;

    // Auto-apply: execute Good ops only, skip Bad
    let (mut applied, mut skipped) = (0usize, 0usize);
    if safe && auto_apply {
        let safe_ops: Vec<AuditOp> = grades
            .iter()
            .filter(|g| g.grade == Grade::Good)
            .map(|g| g.op.clone())
            .collect();
        let bad_ops = grades.iter().filter(|g| g.grade == Grade::Bad).count();
        skipped = bad_ops;

        if !safe_ops.is_empty() {
            let db4 = db.clone();
            let op_count = safe_ops.len();
            let actual = tokio::task::spawn_blocking(move || {
                let mut r = super::audit::AuditResult::default();
                super::audit::apply_audit_ops_pub(&db4, safe_ops, &mut r);
                r
            })
            .await
            .map_err(|e| EngramError::Internal(format!("spawn: {e}")))?;
            applied = actual.promoted + actual.demoted + actual.deleted + actual.merged;
            info!(applied, skipped, op_count, "audit sandbox: auto-applied ops");
        }
    }

    let summary = format!(
        "Reviewed {} memories ({} Core + {} Working) in {} clusters ({} batches). \
         Audit proposed {} ops: {} good, {} bad. \
         Score: {:.0}% (threshold: {:.0}%). {}",
        total,
        core.len(),
        working.len(),
        cluster_count,
        batch_count,
        grades.len(),
        good_count,
        bad_count,
        score * 100.0,
        SAFETY_THRESHOLD * 100.0,
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
        cluster_count,
        batch_count,
    })
}

/// Format a batch of clusters into the audit prompt.
/// Shows full content, layer, metadata, and pairwise similarity scores.
fn format_clustered_prompt(
    clusters: &[&MemoryCluster],
    batch_num: usize,
    total_batches: usize,
) -> String {
    let now = crate::db::now_ms();
    let mut prompt = String::with_capacity(10_000);

    if total_batches > 1 {
        prompt.push_str(&format!(
            "# Batch {}/{}\n\n",
            batch_num, total_batches
        ));
    }

    for (i, cluster) in clusters.iter().enumerate() {
        prompt.push_str(&format!(
            "## Cluster {}: {} ({} memor{})\n\n",
            i + 1,
            cluster.label,
            cluster.memories.len(),
            if cluster.memories.len() == 1 { "y" } else { "ies" }
        ));

        for m in &cluster.memories {
            let tags = m.tags.join(", ");
            let age_d = (now - m.created_at) as f64 / 86_400_000.0;
            let last_accessed_days = (now - m.last_accessed) as f64 / 86_400_000.0;
            let layer_name = match m.layer {
                Layer::Core => "Core",
                Layer::Working => "Working",
                Layer::Buffer => "Buffer",
            };
            // FULL content, no truncation
            prompt.push_str(&format!(
                "[{}] {} | imp={:.1} | ac={} | last_accessed={:.1}d ago | age={:.1}d | kind={} | tags=[{}]\n{}\n\n",
                crate::util::short_id(&m.id),
                layer_name,
                m.importance,
                m.access_count,
                last_accessed_days,
                age_d,
                m.kind,
                tags,
                m.content
            ));
        }

        // Show similarity pairs within cluster
        if !cluster.similarities.is_empty() {
            prompt.push_str("Similarities:\n");
            for (id1, id2, sim) in &cluster.similarities {
                prompt.push_str(&format!(
                    "  {}↔{} = {:.2}\n",
                    crate::util::short_id(id1),
                    crate::util::short_id(id2),
                    sim
                ));
            }
        }
        prompt.push('\n');
    }
    prompt
}
