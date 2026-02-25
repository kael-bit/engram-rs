use crate::ai::{self, AiConfig};
use crate::db::{Memory, MemoryDB};
use crate::error::EngramError;
use crate::SharedDB;
use crate::util::truncate_chars;
use serde::Serialize;

// Re-exported for sandbox use
pub(crate) const AUDIT_SYSTEM_PUB: &str = AUDIT_SYSTEM;

const AUDIT_SYSTEM: &str = r#"Review an AI agent's memory layers. Output a JSON array of operations to clean up.

Core = permanent, survives total context loss. Working = active context. Buffer = temporary.
Metadata: ac=access count, age=days old, mod=days since last edit, kind, tags.

## Operations
- {"op":"delete","id":"<8-char>"}
- {"op":"demote","id":"<8-char>","to":2}
- {"op":"merge","ids":["a","b"],"content":"combined text","layer":2,"tags":["t"]}
- {"op":"promote","id":"<8-char>","to":3}

## Rules
- mod < 1d → don't touch
- Never demote to buffer (to:1)
- Never delete identity or lesson-tagged memories
- NEVER propose demoting a memory to the same layer it is already on. Check the [Layer: ...] tag before proposing any demote. L2→L2 or L3→L3 is a no-op bug.
- When you see multiple memories covering the same topic or event, prefer MERGE over DELETE
- Look for memories with overlapping content — propose merging the less detailed into the more comprehensive one
- Propose at most 30% deletes. Focus on merges and promotions first. Deletions should be reserved for truly obsolete or garbage content.

## What BELONGS in Core (examples)
- "never force-push to main" — lesson, prevents mistakes
- "user prefers Chinese, hates verbose replies" — identity/preference
- "all public output must hide AI identity" — hard constraint
- "we chose SQLite for zero-dep deployment" — decision rationale (the WHY)

## What does NOT belong in Core (demote or delete)
- Changelogs: "Recall改进: 1) expansion 2) FTS gating 3) evidence" → DEMOTE (lists what was DONE)
- Implementation notes: "HNSW replaces brute-force search" → DEMOTE (already in the code)
- Bug reports: "BUG: triage didn't filter namespace" → DELETE if fixed
- Config snapshots: "cosine 0.78, TTL 24h, threshold 5" → DEMOTE (goes stale)
- Session logs: "Session: deployed v0.7, 143 tests pass" → DELETE
- Plans: "TODO: add compression, fix lifecycle" → DELETE (plans, not lessons)

## Decision test
For each memory ask: "If the agent loses all context and only has this memory, is it useful?"
- YES: lesson, constraint, identity, decision rationale → keep in Core
- NO: changelog, implementation detail, resolved bug, session log → demote or delete

Changelogs disguised as decisions are the #1 false positive. If it lists numbered changes
(1) did X 2) did Y), it's a changelog no matter how technical it sounds.

Be aggressive cleaning Working garbage. Output [] if nothing to do."#;

/// Full audit: reviews Core+Working memories using the gate model.
/// Chunks automatically if total prompt would exceed ~100K chars.
pub async fn audit_memories(cfg: &AiConfig, db: &SharedDB) -> Result<AuditResult, EngramError> {
    let db2 = db.clone();
    let (core, working) = tokio::task::spawn_blocking(move || {
        let c = db2.list_by_layer_meta(crate::db::Layer::Core, 500, 0).unwrap_or_default();
        let w = db2.list_by_layer_meta(crate::db::Layer::Working, 500, 0).unwrap_or_default();
        (c, w)
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
        let prompt = format_audit_prompt(&core, &working);
        let result = ai::llm_chat_as(cfg, "audit", AUDIT_SYSTEM, &prompt).await?;
        if let Some(ref u) = result.usage {
            let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
            let _ = db.log_llm_call("audit", &result.model, u.prompt_tokens, u.completion_tokens, cached, result.duration_ms);
        }
        let ops = parse_audit_ops(&result.content, &core, &working);
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

            let result = ai::llm_chat_as(cfg, "audit", AUDIT_SYSTEM, &prompt).await?;
            if let Some(ref u) = result.usage {
                let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
                let _ = db.log_llm_call("audit", &result.model, u.prompt_tokens, u.completion_tokens, cached, result.duration_ms);
            }
            // for chunked mode, resolve against core + this chunk only
            let chunk_vec: Vec<Memory> = chunk.to_vec();
            let ops = parse_audit_ops(&result.content, &core, &chunk_vec);
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

fn format_audit_prompt(core: &[Memory], working: &[Memory]) -> String {
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

// Re-exported for sandbox use
pub(crate) fn parse_audit_ops_pub(
    response: &str,
    core: &[Memory],
    working: &[Memory],
) -> Vec<AuditOp> {
    parse_audit_ops(response, core, working)
}

pub(crate) fn apply_audit_ops_pub(db: &crate::SharedDB, ops: Vec<AuditOp>, result: &mut AuditResult) {
    apply_audit_ops(db, ops, result);
}

fn parse_audit_ops(
    response: &str,
    core: &[Memory],
    working: &[Memory],
) -> Vec<AuditOp> {
    let json_str = response
        .find('[')
        .and_then(|start| response.rfind(']').map(|end| &response[start..=end]))
        .unwrap_or("[]");

    let arr: Vec<serde_json::Value> = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => return vec![],
    };

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
    for item in &arr {
        let op_type = item.get("op").and_then(|v| v.as_str()).unwrap_or("");
        match op_type {
            "promote" => {
                if let (Some(id), Some(to)) = (
                    item.get("id").and_then(|v| v.as_str()).and_then(&resolve),
                    item.get("to").and_then(serde_json::Value::as_u64).map(|n| n as u8),
                ) {
                    if (1..=3).contains(&to) { ops.push(AuditOp::Promote { id, to }); }
                }
            }
            "demote" => {
                if let (Some(id), Some(to)) = (
                    item.get("id").and_then(|v| v.as_str()).and_then(&resolve),
                    item.get("to").and_then(serde_json::Value::as_u64).map(|n| n as u8),
                ) {
                    if (1..=3).contains(&to) { ops.push(AuditOp::Demote { id, to }); }
                }
            }
            "delete" => {
                if let Some(id) = item.get("id").and_then(|v| v.as_str()).and_then(&resolve) {
                    ops.push(AuditOp::Delete { id });
                }
            }
            "merge" => {
                let ids: Vec<String> = item.get("ids")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_str().and_then(&resolve)).collect())
                    .unwrap_or_default();
                let content = item.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let layer = item.get("layer").and_then(serde_json::Value::as_u64).unwrap_or(2) as u8;
                let tags: Vec<String> = item.get("tags")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                if ids.len() >= 2 && !content.is_empty() {
                    ops.push(AuditOp::Merge { ids, content, layer, tags });
                }
            }
            _ => {}
        }
    }
    ops
}
