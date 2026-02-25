use crate::ai::{self, AiConfig};
use crate::db::{Memory, MemoryDB};
use crate::error::EngramError;
use crate::SharedDB;
use crate::util::truncate_chars;
use serde::{Deserialize, Serialize};

/// Similarity range for audit merge hints (below auto-merge threshold).
use crate::thresholds::{AUDIT_MERGE_MIN_SIM, AUDIT_MERGE_MAX_SIM};

// Re-exported for sandbox use
pub(crate) const AUDIT_SYSTEM_PUB: &str = AUDIT_SYSTEM;

const AUDIT_SYSTEM: &str = r#"Review an AI agent's memory layers and propose cleanup operations.

Memories are presented in **semantic clusters** grouped by combined similarity (cosine + tag overlap).
Memories in the same cluster are semantically related — check for redundancy within clusters.

Metadata per memory:
- imp = importance (0-1)
- ac = access count (how many times recalled). ac=0 means never recalled — consider if it's still valuable
- last_accessed = days since last recall
- age = days since creation
- kind = memory type (semantic, procedural, episodic)
- tags = associated labels

Similarity scores between cluster members show how related two memories are:
- >0.80 = near-duplicate content, strongly consider merging
- 0.60-0.80 = related topic, check if they can be consolidated
- <0.60 = loosely related, probably separate concepts

Full content is shown — judge quality based on actual content, not just tags or metadata.

## Layer Principles

**Core** = stable principles, lessons from mistakes, identity constraints. Rarely changes.
If content references specific code/config versions, it's probably Working, not Core.

**Working** = active project context, current decisions. Becomes stale as projects evolve.

**Buffer** = temporary intake, expires naturally.

## What BELONGS in Core
- Lessons that prevent repeating mistakes
- Identity/preferences that shape behavior
- Hard constraints and rules
- Decision rationale (the WHY behind choices)

## What does NOT belong in Core
- Changelogs listing WHAT was done → demote or delete
- Implementation details already in code → demote
- Session logs and progress reports → delete
- Plans/TODOs that are stale → delete
- Config snapshots that go stale → demote

## Judgment Guidelines

- **Superseded memories:** if a newer memory in the same cluster covers the same knowledge, the older one should be removed or merged.
- **ac=0 + old age** = possibly forgotten. Judge if still relevant based on content, not just metrics.
- NEVER propose demoting a memory to the same layer it is already on. Check the layer metadata. L2→L2 or L3→L3 is a no-op bug.
- When memories in a cluster overlap heavily, prefer MERGE over DELETE to preserve information.
- Full content is shown — read it carefully before deciding.
- Propose no operations if nothing needs changing."#;

/// Tool response: a list of audit operations proposed by the LLM.
#[derive(Debug, Deserialize)]
pub struct AuditToolResponse {
    pub operations: Vec<RawAuditOp>,
}

/// Raw audit operation as returned by the LLM tool call.
/// Fields are optional because different op types use different fields.
#[derive(Debug, Deserialize)]
pub struct RawAuditOp {
    pub op: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub to: Option<u8>,
    #[serde(default)]
    pub ids: Option<Vec<String>>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub layer: Option<u8>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
}

/// Build the JSON schema for the audit_operations tool.
pub fn audit_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "description": "List of cleanup operations. Empty array if nothing needs changing.",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "enum": ["delete", "demote", "merge", "promote"],
                            "description": "Operation type"
                        },
                        "id": {
                            "type": "string",
                            "description": "8-char short ID of the target memory (for delete/demote/promote)"
                        },
                        "to": {
                            "type": "integer",
                            "enum": [2, 3],
                            "description": "Target layer: 2=Working, 3=Core (for demote/promote)"
                        },
                        "ids": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Short IDs of memories to merge (for merge op, minimum 2)"
                        },
                        "content": {
                            "type": "string",
                            "description": "Combined text for the merged memory (for merge op)"
                        },
                        "layer": {
                            "type": "integer",
                            "enum": [2, 3],
                            "description": "Layer for the merged memory (for merge op)"
                        },
                        "tags": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Tags for the merged memory (for merge op)"
                        }
                    },
                    "required": ["op"]
                }
            }
        },
        "required": ["operations"]
    })
}

/// Full audit: reviews Core+Working memories using the audit model via function calling.
/// Chunks automatically if total prompt would exceed ~100K chars.
pub async fn audit_memories(cfg: &AiConfig, db: &SharedDB) -> Result<AuditResult, EngramError> {
    let db2 = db.clone();
    let (core, working, merge_hints) = tokio::task::spawn_blocking(move || {
        let c = db2.list_by_layer_meta(crate::db::Layer::Core, 500, 0).unwrap_or_default();
        let w = db2.list_by_layer_meta(crate::db::Layer::Working, 500, 0).unwrap_or_default();
        let hints = find_merge_candidates(&db2, &w);
        (c, w, hints)
    }).await.map_err(|e| EngramError::Internal(format!("spawn failed: {e}")))?;

    let all: Vec<&Memory> = core.iter().chain(working.iter()).collect();

    // ~300 chars per formatted memory entry (200 content + metadata)
    const CHARS_PER_ENTRY: usize = 300;
    const MAX_PROMPT_CHARS: usize = 64_000;
    let max_per_batch = MAX_PROMPT_CHARS / CHARS_PER_ENTRY; // ~333

    let mut combined = AuditResult {
        total_reviewed: all.len(),
        ..Default::default()
    };

    // Core always goes in first batch (small, provides context)
    // Working gets chunked if needed
    if all.len() <= max_per_batch {
        // single batch
        let prompt = format_audit_prompt(&core, &working, &merge_hints);
        let tcr = call_audit_tool(cfg, &prompt).await?;
        if let Some(ref u) = tcr.usage {
            let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
            let _ = db.log_llm_call("audit", &tcr.model, u.prompt_tokens, u.completion_tokens, cached, tcr.duration_ms);
        }
        let ops = resolve_audit_ops(tcr.value.operations, &core, &working);
        let db3 = db.clone();
        let applied = tokio::task::spawn_blocking(move || {
            let mut r = AuditResult::default();
            apply_audit_ops(&db3, ops, &mut r);
            r
        }).await.unwrap_or_default();
        combined.promoted += applied.promoted;
        combined.demoted += applied.demoted;
        combined.deleted += applied.deleted;
        combined.merged += applied.merged;
        combined.ops.extend(applied.ops);
    } else {
        // chunked: Core summary + Working in batches
        let core_summary = format_layer_summary("Core", &core);
        for chunk in working.chunks(max_per_batch.saturating_sub(core.len())) {
            let mut prompt = core_summary.clone();
            prompt.push_str(&format!("\n## Working Layer (batch of {})\n", chunk.len()));
            for m in chunk {
                let tags = m.tags.join(",");
                let preview: String = truncate_chars(&m.content, 200);
                prompt.push_str(&format!("- [{}] [Layer: Working (2)] (imp={:.1}, acc={}, tags=[{}]) {}\n",
                    crate::util::short_id(&m.id), m.importance, m.access_count, tags, preview));
            }

            let tcr = call_audit_tool(cfg, &prompt).await?;
            if let Some(ref u) = tcr.usage {
                let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
                let _ = db.log_llm_call("audit", &tcr.model, u.prompt_tokens, u.completion_tokens, cached, tcr.duration_ms);
            }
            // for chunked mode, resolve against core + this chunk only
            let chunk_vec: Vec<Memory> = chunk.to_vec();
            let ops = resolve_audit_ops(tcr.value.operations, &core, &chunk_vec);
            let db4 = db.clone();
            let applied = tokio::task::spawn_blocking(move || {
                let mut r = AuditResult::default();
                apply_audit_ops(&db4, ops, &mut r);
                r
            }).await.unwrap_or_default();
            combined.promoted += applied.promoted;
            combined.demoted += applied.demoted;
            combined.deleted += applied.deleted;
            combined.merged += applied.merged;
            combined.ops.extend(applied.ops);
        }
    }

    Ok(combined)
}

/// Call the audit LLM with function calling, returning structured operations.
async fn call_audit_tool(cfg: &AiConfig, prompt: &str) -> Result<ai::ToolCallResult<AuditToolResponse>, EngramError> {
    ai::llm_tool_call::<AuditToolResponse>(
        cfg, "audit", AUDIT_SYSTEM, prompt,
        "audit_operations",
        "Propose cleanup operations for the memory store. Return empty operations array if nothing needs changing.",
        audit_tool_schema(),
    ).await
}

fn format_audit_prompt(core: &[Memory], working: &[Memory], merge_hints: &[(String, String, f64)]) -> String {
    let now = crate::db::now_ms();
    let mut prompt = String::with_capacity(16_000);
    prompt.push_str("## Core Layer (L3)\n");
    for m in core {
        let tags = m.tags.join(",");
        let age_d = (now - m.created_at) as f64 / 86_400_000.0;
        let mod_d = if m.modified_at > 0 {
            (now - m.modified_at) as f64 / 86_400_000.0
        } else {
            age_d
        };
        let preview: String = truncate_chars(&m.content, 200);
        prompt.push_str(&format!("- [{}] [Layer: Core (3)] (imp={:.1}, ac={}, age={:.1}d, mod={:.1}d, tags=[{}]) {}\n",
            crate::util::short_id(&m.id), m.importance, m.access_count, age_d, mod_d, tags, preview));
    }
    prompt.push_str(&format!("\n## Working Layer (L2, {} memories)\n", working.len()));
    for m in working {
        let tags = m.tags.join(",");
        let age_d = (now - m.created_at) as f64 / 86_400_000.0;
        let mod_d = if m.modified_at > 0 {
            (now - m.modified_at) as f64 / 86_400_000.0
        } else {
            age_d
        };
        let preview: String = truncate_chars(&m.content, 200);
        prompt.push_str(&format!("- [{}] [Layer: Working (2)] (imp={:.1}, ac={}, age={:.1}d, mod={:.1}d, tags=[{}]) {}\n",
            crate::util::short_id(&m.id), m.importance, m.access_count, age_d, mod_d, tags, preview));
    }

    if !merge_hints.is_empty() {
        prompt.push_str("\n## Merge Hints (pre-computed similarity)\n");
        prompt.push_str("These Working memories are semantically similar — review and merge if they cover the same topic:\n");
        for (a, b, sim) in merge_hints {
            prompt.push_str(&format!("- [{}] ↔ [{}] (similarity: {:.2})\n", a, b, sim));
        }
    }

    prompt
}

fn format_layer_summary(name: &str, memories: &[Memory]) -> String {
    let mut s = format!("## {} Layer ({} memories, shown for context — do NOT reorganize these)\n", name, memories.len());
    for m in memories {
        let preview: String = truncate_chars(&m.content, 80);
        s.push_str(&format!("- [{}] {}\n", crate::util::short_id(&m.id), preview));
    }
    s
}

/// Find Working memory pairs with high semantic similarity (merge candidates).
/// Returns short ID pairs + similarity score, sorted by similarity descending, max 10.
fn find_merge_candidates(db: &MemoryDB, working: &[Memory]) -> Vec<(String, String, f64)> {
    let ids: Vec<String> = working.iter().map(|m| m.id.clone()).collect();
    let embeddings = db.get_embeddings_by_ids(&ids);
    if embeddings.len() < 2 {
        return vec![];
    }

    let mut pairs: Vec<(String, String, f64)> = Vec::new();
    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            let sim = crate::ai::cosine_similarity(&embeddings[i].1, &embeddings[j].1);
            if sim > AUDIT_MERGE_MIN_SIM && sim < AUDIT_MERGE_MAX_SIM {
                // Related but not identical (merge sweet spot)
                // Higher similarity is handled by auto-merge in consolidation
                let a = crate::util::short_id(&embeddings[i].0);
                let b = crate::util::short_id(&embeddings[j].0);
                pairs.push((a.to_string(), b.to_string(), sim));
            }
        }
    }

    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(10);
    pairs
}

fn apply_audit_ops(db: &MemoryDB, ops: Vec<AuditOp>, result: &mut AuditResult) {
    for op in ops {
        match &op {
            AuditOp::Promote { id, to } => {
                if let Ok(Some(_)) = db.update_fields(id, None, Some(*to), None, None) {
                    result.promoted += 1;
                }
            }
            AuditOp::Demote { id, to } => {
                if let Ok(Some(_)) = db.update_fields(id, None, Some(*to), Some(0.5), None) {
                    result.demoted += 1;
                }
            }
            AuditOp::Delete { id } => {
                if db.delete(id).unwrap_or(false) {
                    result.deleted += 1;
                }
            }
            AuditOp::Merge { ids, content, layer, tags } => {
                let input = crate::db::MemoryInput {
                    content: content.clone(),
                    layer: Some(*layer),
                    tags: Some(tags.clone()),
                    supersedes: Some(ids.clone()),
                    source: Some("audit".into()),
                    ..Default::default()
                };
                if db.insert(input).is_ok() {
                    result.merged += 1;
                }
            }
        }
        result.ops.push(op);
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "op")]
pub enum AuditOp {
    #[serde(rename = "promote")]
    Promote { id: String, to: u8 },
    #[serde(rename = "demote")]
    Demote { id: String, to: u8 },
    #[serde(rename = "delete")]
    Delete { id: String },
    #[serde(rename = "merge")]
    Merge { ids: Vec<String>, content: String, layer: u8, tags: Vec<String> },
}

#[derive(Debug, Default, Serialize)]
pub struct AuditResult {
    pub total_reviewed: usize,
    pub promoted: usize,
    pub demoted: usize,
    pub deleted: usize,
    pub merged: usize,
    pub ops: Vec<AuditOp>,
}

/// Resolve raw LLM tool call operations into validated AuditOps.
/// Short IDs (8-char) are resolved to full UUIDs. Invalid ops are skipped.
pub fn resolve_audit_ops(
    raw_ops: Vec<RawAuditOp>,
    core: &[Memory],
    working: &[Memory],
) -> Vec<AuditOp> {
    let mut id_map = std::collections::HashMap::new();
    for m in core.iter().chain(working.iter()) {
        if m.id.len() >= 8 {
            id_map.insert(crate::util::short_id(&m.id).to_string(), m.id.clone());
        }
    }

    let resolve = |short: &str| -> Option<String> {
        if short.len() >= 32 {
            Some(short.to_string())
        } else {
            id_map.get(short).cloned()
        }
    };

    let mut ops = Vec::new();
    for item in raw_ops {
        match item.op.as_str() {
            "promote" => {
                if let (Some(id), Some(to)) = (
                    item.id.as_deref().and_then(&resolve),
                    item.to,
                ) {
                    if (1..=3).contains(&to) { ops.push(AuditOp::Promote { id, to }); }
                }
            }
            "demote" => {
                if let (Some(id), Some(to)) = (
                    item.id.as_deref().and_then(&resolve),
                    item.to,
                ) {
                    if (1..=3).contains(&to) { ops.push(AuditOp::Demote { id, to }); }
                }
            }
            "delete" => {
                if let Some(id) = item.id.as_deref().and_then(&resolve) {
                    ops.push(AuditOp::Delete { id });
                }
            }
            "merge" => {
                let ids: Vec<String> = item.ids.unwrap_or_default()
                    .iter()
                    .filter_map(|s| resolve(s))
                    .collect();
                let content = item.content.unwrap_or_default();
                let layer = item.layer.unwrap_or(2);
                let tags = item.tags.unwrap_or_default();
                if ids.len() >= 2 && !content.is_empty() {
                    ops.push(AuditOp::Merge { ids, content, layer, tags });
                }
            }
            _ => {}
        }
    }
    ops
}

pub fn apply_audit_ops_pub(db: &crate::SharedDB, ops: Vec<AuditOp>, result: &mut AuditResult) {
    apply_audit_ops(db, ops, result);
}

