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
            if sim > 0.65 && sim < 0.78 {
                // 0.65-0.78 = related but not identical (merge sweet spot)
                // > 0.78 is handled by auto-merge in consolidation
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{Layer, Memory};

    /// Helper: create a Memory with the given id and layer, everything else zeroed/defaults.
    fn mem(id: &str, layer: Layer) -> Memory {
        Memory {
            id: id.to_string(),
            content: String::new(),
            layer,
            importance: 0.5,
            created_at: 0,
            last_accessed: 0,
            access_count: 0,
            repetition_count: 0,
            decay_rate: 0.1,
            source: String::new(),
            tags: vec![],
            namespace: "default".into(),
            embedding: None,
            kind: "episodic".into(),
            modified_at: 0,
        }
    }

    // ── 1. Valid JSON with all 4 op types ───────────────────────────────

    #[test]
    fn all_four_op_types() {
        let core = vec![
            mem("aaaaaaaa-1111-2222-3333-444444444444", Layer::Core),
            mem("bbbbbbbb-1111-2222-3333-444444444444", Layer::Core),
        ];
        let working = vec![
            mem("cccccccc-1111-2222-3333-444444444444", Layer::Working),
            mem("dddddddd-1111-2222-3333-444444444444", Layer::Working),
            mem("eeeeeeee-1111-2222-3333-444444444444", Layer::Working),
        ];

        let response = r#"[
            {"op":"promote","id":"cccccccc","to":3},
            {"op":"demote","id":"aaaaaaaa","to":2},
            {"op":"delete","id":"dddddddd"},
            {"op":"merge","ids":["bbbbbbbb","eeeeeeee"],"content":"merged content","layer":3,"tags":["t1","t2"]}
        ]"#;

        let ops = parse_audit_ops(response, &core, &working);
        assert_eq!(ops.len(), 4);

        match &ops[0] {
            AuditOp::Promote { id, to } => {
                assert_eq!(id, "cccccccc-1111-2222-3333-444444444444");
                assert_eq!(*to, 3);
            }
            other => panic!("expected Promote, got {:?}", other),
        }
        match &ops[1] {
            AuditOp::Demote { id, to } => {
                assert_eq!(id, "aaaaaaaa-1111-2222-3333-444444444444");
                assert_eq!(*to, 2);
            }
            other => panic!("expected Demote, got {:?}", other),
        }
        match &ops[2] {
            AuditOp::Delete { id } => {
                assert_eq!(id, "dddddddd-1111-2222-3333-444444444444");
            }
            other => panic!("expected Delete, got {:?}", other),
        }
        match &ops[3] {
            AuditOp::Merge { ids, content, layer, tags } => {
                assert_eq!(ids.len(), 2);
                assert_eq!(ids[0], "bbbbbbbb-1111-2222-3333-444444444444");
                assert_eq!(ids[1], "eeeeeeee-1111-2222-3333-444444444444");
                assert_eq!(content, "merged content");
                assert_eq!(*layer, 3);
                assert_eq!(tags, &["t1", "t2"]);
            }
            other => panic!("expected Merge, got {:?}", other),
        }
    }

    // ── 2. Short IDs resolved against memory list ──────────────────────

    #[test]
    fn short_ids_resolved() {
        let core = vec![mem("abcd1234-aaaa-bbbb-cccc-dddddddddddd", Layer::Core)];
        let working = vec![mem("ef567890-aaaa-bbbb-cccc-dddddddddddd", Layer::Working)];

        let response = r#"[{"op":"delete","id":"abcd1234"},{"op":"promote","id":"ef567890","to":3}]"#;
        let ops = parse_audit_ops(response, &core, &working);

        assert_eq!(ops.len(), 2);
        match &ops[0] {
            AuditOp::Delete { id } => assert_eq!(id, "abcd1234-aaaa-bbbb-cccc-dddddddddddd"),
            other => panic!("expected Delete, got {:?}", other),
        }
        match &ops[1] {
            AuditOp::Promote { id, to } => {
                assert_eq!(id, "ef567890-aaaa-bbbb-cccc-dddddddddddd");
                assert_eq!(*to, 3);
            }
            other => panic!("expected Promote, got {:?}", other),
        }
    }

    // ── 3. Full UUIDs passed through directly ──────────────────────────

    #[test]
    fn full_uuids_passed_through() {
        let full_id = "12345678-abcd-ef01-2345-6789abcdef01";
        // No memories in the list — full UUID should still work
        let response = format!(r#"[{{"op":"delete","id":"{full_id}"}}]"#);
        let ops = parse_audit_ops(&response, &[], &[]);

        assert_eq!(ops.len(), 1);
        match &ops[0] {
            AuditOp::Delete { id } => assert_eq!(id, full_id),
            other => panic!("expected Delete, got {:?}", other),
        }
    }

    // ── 4. Empty ops array → empty result ──────────────────────────────

    #[test]
    fn empty_ops_array() {
        let ops = parse_audit_ops("[]", &[], &[]);
        assert!(ops.is_empty());
    }

    // ── 5. Malformed JSON → empty result ───────────────────────────────

    #[test]
    fn malformed_json_returns_empty() {
        let ops = parse_audit_ops("this is not json at all", &[], &[]);
        assert!(ops.is_empty());
    }

    #[test]
    fn malformed_json_partial_bracket() {
        let ops = parse_audit_ops("[{broken", &[], &[]);
        assert!(ops.is_empty());
    }

    // ── 6. Missing required fields → op skipped ────────────────────────

    #[test]
    fn merge_without_source_ids_skipped() {
        // merge requires ids (>=2), content non-empty
        let response = r#"[{"op":"merge","content":"combined","layer":2,"tags":["x"]}]"#;
        let ops = parse_audit_ops(response, &[], &[]);
        assert!(ops.is_empty(), "merge without ids should be skipped");
    }

    #[test]
    fn merge_with_single_id_skipped() {
        let core = vec![mem("aaaaaaaa-1111-2222-3333-444444444444", Layer::Core)];
        let response = r#"[{"op":"merge","ids":["aaaaaaaa"],"content":"combined","layer":2,"tags":[]}]"#;
        let ops = parse_audit_ops(response, &core, &[]);
        assert!(ops.is_empty(), "merge with only 1 id should be skipped");
    }

    #[test]
    fn merge_with_empty_content_skipped() {
        let core = vec![
            mem("aaaaaaaa-1111-2222-3333-444444444444", Layer::Core),
            mem("bbbbbbbb-1111-2222-3333-444444444444", Layer::Core),
        ];
        let response = r#"[{"op":"merge","ids":["aaaaaaaa","bbbbbbbb"],"content":"","layer":2,"tags":[]}]"#;
        let ops = parse_audit_ops(response, &core, &[]);
        assert!(ops.is_empty(), "merge with empty content should be skipped");
    }

    #[test]
    fn promote_without_to_skipped() {
        let core = vec![mem("aaaaaaaa-1111-2222-3333-444444444444", Layer::Core)];
        let response = r#"[{"op":"promote","id":"aaaaaaaa"}]"#;
        let ops = parse_audit_ops(response, &core, &[]);
        assert!(ops.is_empty(), "promote without 'to' should be skipped");
    }

    #[test]
    fn delete_with_unresolvable_short_id_skipped() {
        // Short id not in any memory list
        let response = r#"[{"op":"delete","id":"zzzzzzzz"}]"#;
        let ops = parse_audit_ops(response, &[], &[]);
        assert!(ops.is_empty(), "unresolvable short id should be skipped");
    }

    // ── 7. Unknown op type → skipped ───────────────────────────────────

    #[test]
    fn unknown_op_type_skipped() {
        let response = r#"[{"op":"explode","id":"aaaaaaaa"}]"#;
        let ops = parse_audit_ops(response, &[], &[]);
        assert!(ops.is_empty());
    }

    // ── 8. Mix of valid and invalid ops → only valid returned ──────────

    #[test]
    fn mix_valid_and_invalid() {
        let core = vec![
            mem("aaaaaaaa-1111-2222-3333-444444444444", Layer::Core),
        ];
        let working = vec![
            mem("bbbbbbbb-1111-2222-3333-444444444444", Layer::Working),
        ];

        let response = r#"[
            {"op":"delete","id":"aaaaaaaa"},
            {"op":"unknown_type","id":"bbbbbbbb"},
            {"op":"promote","id":"bbbbbbbb","to":3},
            {"op":"promote","id":"nonexist"},
            {"op":"demote","id":"aaaaaaaa","to":2}
        ]"#;

        let ops = parse_audit_ops(response, &core, &working);
        // Valid: delete aaaaaaaa, promote bbbbbbbb to 3, demote aaaaaaaa to 2
        // Invalid: unknown_type (skipped), promote nonexist (no 'to' + unresolvable)
        assert_eq!(ops.len(), 3);
        assert!(matches!(&ops[0], AuditOp::Delete { .. }));
        assert!(matches!(&ops[1], AuditOp::Promote { to: 3, .. }));
        assert!(matches!(&ops[2], AuditOp::Demote { to: 2, .. }));
    }

    // ── Extra: JSON embedded in prose (LLM often wraps in markdown) ────

    #[test]
    fn json_embedded_in_prose() {
        let core = vec![mem("aaaaaaaa-1111-2222-3333-444444444444", Layer::Core)];
        let response = r#"Here are my suggestions:
```json
[{"op":"delete","id":"aaaaaaaa"}]
```
Let me know if you agree."#;

        let ops = parse_audit_ops(response, &core, &[]);
        assert_eq!(ops.len(), 1);
        assert!(matches!(&ops[0], AuditOp::Delete { .. }));
    }

    // ── Extra: `to` out of range (0 or 4) → skipped ───────────────────

    #[test]
    fn to_out_of_range_skipped() {
        let core = vec![mem("aaaaaaaa-1111-2222-3333-444444444444", Layer::Core)];
        let response = r#"[
            {"op":"promote","id":"aaaaaaaa","to":0},
            {"op":"demote","id":"aaaaaaaa","to":4}
        ]"#;
        let ops = parse_audit_ops(response, &core, &[]);
        assert!(ops.is_empty(), "to=0 and to=4 should both be skipped");
    }
}
