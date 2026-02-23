use crate::ai::{self, AiConfig, cosine_similarity};
use crate::db::{Layer, Memory, MemoryDB};
use crate::error::EngramError;
use crate::SharedDB;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

#[derive(Debug, Deserialize)]
pub struct ConsolidateRequest {
    pub promote_threshold: Option<i64>,
    pub promote_min_importance: Option<f64>,
    pub decay_drop_threshold: Option<f64>,
    /// Buffer entries older than this (seconds) get promoted or dropped.
    pub buffer_ttl_secs: Option<i64>,
    /// Working entries older than this (seconds) with decent importance auto-promote to core.
    pub working_age_promote_secs: Option<i64>,
    /// Whether to merge similar memories within each layer via LLM.
    pub merge: Option<bool>,
}

#[derive(Debug, Serialize, Default)]
pub struct ConsolidateResponse {
    pub promoted: usize,
    pub decayed: usize,
    pub demoted: usize,
    pub merged: usize,
    #[serde(skip_serializing_if = "is_zero")]
    pub importance_decayed: usize,
    #[serde(skip_serializing_if = "is_zero")]
    pub gate_rejected: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub promoted_ids: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub dropped_ids: Vec<String>,
    /// IDs of memories that absorbed others during merge (winners).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub merged_ids: Vec<String>,
    /// IDs that passed access/age thresholds but await LLM gate review.
    #[serde(skip)]
    pub(crate) promotion_candidates: Vec<(String, String)>, // (id, content)
}

fn is_zero(n: &usize) -> bool { *n == 0 }

pub async fn consolidate(
    db: SharedDB,
    req: Option<ConsolidateRequest>,
    ai: Option<AiConfig>,
) -> ConsolidateResponse {
    let do_merge = req.as_ref().and_then(|r| r.merge).unwrap_or(false);

    let db2 = db.clone();
    let mut result = tokio::task::spawn_blocking(move || {
        consolidate_sync(&db2, req.as_ref())
    })
    .await
    .unwrap_or_else(|e| {
        warn!(error = %e, "consolidate_sync task panicked");
        ConsolidateResponse::default()
    });

    // LLM gate: review promotion candidates before moving to Core.
    // Without AI config, promote all candidates directly (backward compat).
    let candidates = std::mem::take(&mut result.promotion_candidates);
    if !candidates.is_empty() {
        match &ai {
            Some(cfg) => {
                for (id, content) in &candidates {
                    match llm_promotion_gate(cfg, content).await {
                        Ok(true) => {
                            if db.promote(id, Layer::Core).is_ok() {
                                result.promoted_ids.push(id.clone());
                                result.promoted += 1;
                                debug!(id = %id, "LLM approved promotion to Core");
                            }
                        }
                        Ok(false) => {
                            result.gate_rejected += 1;
                            // Prevent repeated LLM calls: drop importance below promote threshold
                            let _ = db.update_fields(id, None, None, Some(0.4), None);
                            debug!(id = %id, "LLM rejected promotion, importance reduced to 0.4");
                        }
                        Err(e) => {
                            // LLM error → skip this round, don't promote blindly
                            warn!(id = %id, error = %e, "LLM gate failed, skipping");
                        }
                    }
                }
            }
            None => {
                // No AI configured — promote all (legacy behavior)
                for (id, _) in &candidates {
                    if db.promote(id, Layer::Core).is_ok() {
                        result.promoted_ids.push(id.clone());
                        result.promoted += 1;
                    }
                }
            }
        }
    }

    if do_merge {
        if let Some(cfg) = ai {
            let (count, ids) = merge_similar(&db, &cfg).await;
            result.merged = count;
            result.merged_ids = ids;
        }
    }

    // auto-repair FTS after merge (merge deletes can leave orphans)
    let db3 = db.clone();
    let _ = tokio::task::spawn_blocking(move || {
        if let Ok((orphans, rebuilt)) = db3.repair_fts() {
            if orphans > 0 || rebuilt > 0 {
                info!(orphans, rebuilt, "auto-repaired FTS index");
            }
        }
    }).await;

    result
}

pub(crate) fn consolidate_sync(db: &MemoryDB, req: Option<&ConsolidateRequest>) -> ConsolidateResponse {
    let promote_threshold = req.and_then(|r| r.promote_threshold).unwrap_or(3);
    let promote_min_imp = req.and_then(|r| r.promote_min_importance).unwrap_or(0.6);
    let decay_threshold = req.and_then(|r| r.decay_drop_threshold).unwrap_or(0.01);
    let buffer_ttl = req.and_then(|r| r.buffer_ttl_secs).unwrap_or(86400)
        .saturating_mul(1000);
    let working_age = req.and_then(|r| r.working_age_promote_secs).unwrap_or(7 * 86400)
        .saturating_mul(1000);

    let now = crate::db::now_ms();
    let mut promoted = 0_usize;
    let mut decayed = 0_usize;
    let mut promoted_ids = Vec::new();
    let mut dropped_ids = Vec::new();
    let mut promotion_candidates = Vec::new();

    // Demote session notes that shouldn't be in Core.
    // Session logs are episodic — they belong in Working at most.
    let mut demoted = 0_usize;
    for mem in db.list_by_layer_meta(Layer::Core, 10000, 0) {
        if (mem.source == "session" || mem.tags.iter().any(|t| t == "ephemeral"))
            && db.demote(&mem.id, Layer::Working).is_ok() {
                demoted += 1;
            }
    }

    // Working → Core: collect candidates for LLM gate review.
    // Reinforcement score = access_count + repetition_count * 2.5
    // Repetition weighs 2.5x because restating > incidental recall hit.
    // Session notes and ephemeral tags are always blocked.
    for mem in db.list_by_layer_meta(Layer::Working, 10000, 0) {
        if mem.source == "session" || mem.tags.iter().any(|t| t == "ephemeral") {
            continue;
        }
        let score = mem.access_count as f64 + mem.repetition_count as f64 * 2.5;
        let by_score = score >= promote_threshold as f64
            && mem.importance >= promote_min_imp;
        let by_age = (now - mem.created_at) > working_age
            && score > 0.0
            && mem.importance >= promote_min_imp;

        if by_score || by_age {
            promotion_candidates.push((mem.id.clone(), mem.content.clone()));
        }
    }

    // Buffer → Working: same weighted score, higher threshold.
    let buffer_threshold = promote_threshold.max(5) as f64;
    for mem in db.list_by_layer_meta(Layer::Buffer, 10000, 0) {
        let score = mem.access_count as f64 + mem.repetition_count as f64 * 2.5;
        if score >= buffer_threshold
            && db.promote(&mem.id, Layer::Working).is_ok() {
                promoted_ids.push(mem.id.clone());
                promoted += 1;
            }
    }

    // Buffer TTL — old L1 entries that weren't accessed enough get dropped.
    // Only rescue to Working if accessed at least half the promote threshold,
    // otherwise it wasn't important enough to keep.
    // Procedural memories are exempt — they persist indefinitely.
    for mem in db.list_by_layer_meta(Layer::Buffer, 10000, 0) {
        if promoted_ids.contains(&mem.id) || mem.kind == "procedural" {
            continue;
        }
        let age = now - mem.created_at;
        if age > buffer_ttl {
            let rescue_score = mem.access_count as f64 + mem.repetition_count as f64 * 2.5;
            if rescue_score >= buffer_threshold / 2.0 {
                // Enough combined signal to deserve a second life in Working
                if db.promote(&mem.id, Layer::Working).is_ok() {
                    promoted_ids.push(mem.id.clone());
                    promoted += 1;
                }
            } else if db.delete(&mem.id).unwrap_or(false) {
                dropped_ids.push(mem.id.clone());
                decayed += 1;
            }
        }
    }

    // Drop decayed Buffer/Working entries — but skip anything we just promoted
    // or procedural memories (they don't decay).
    for mem in db.get_decayed_meta(decay_threshold) {
        if promoted_ids.contains(&mem.id) || mem.kind == "procedural" {
            continue;
        }
        if db.delete(&mem.id).unwrap_or(false) {
            dropped_ids.push(mem.id.clone());
            decayed += 1;
        }
    }

    // Importance decay — memories not accessed in 24h lose a bit of importance.
    // Prevents everything from converging to 1.0 over time.
    // Floor of 0.3 ensures nothing becomes invisible.
    let importance_decayed = db.decay_importance(24.0, 0.05, 0.3).unwrap_or(0);

    if promoted > 0 || decayed > 0 || demoted > 0 || importance_decayed > 0 {
        info!(promoted, decayed, demoted, importance_decayed, "consolidation complete");
    } else {
        debug!("consolidation: nothing to do");
    }

    ConsolidateResponse {
        promoted,
        decayed,
        demoted,
        merged: 0,
        importance_decayed,
        gate_rejected: 0,
        promoted_ids,
        dropped_ids,
        merged_ids: vec![],
        promotion_candidates,
    }
}

const GATE_SYSTEM: &str = "You are a memory curator for an AI agent's long-term memory system.\n\
    The agent has a three-layer memory: Buffer (temporary), Working (useful), Core (permanent identity-level knowledge).\n\
    \n\
    A memory is being considered for promotion to Core (permanent). Decide if it belongs there.\n\
    \n\
    APPROVE for Core if it is:\n\
    - Identity: who the agent/user is, preferences, principles, personal constraints\n\
    - Strategic: long-term goals, key decisions\n\
    - Lessons learned: hard-won insights, mistakes to never repeat, behavioral patterns to change\n\
    - Relational: important facts about people, relationships\n\
    - Constraints: financial limits, payment rules, security policies, behavioral boundaries\n\
    - Architectural: fundamental design decisions that shape the project\n\
    \n\
    REJECT (keep in Working) if it is:\n\
    - Operational: bug fixes, code changes, version bumps, deployment logs\n\
    - Ephemeral: session summaries, daily progress, temporary status\n\
    - Technical detail: specific code patterns, API signatures, config values, tech stack specs\n\
    - Duplicate: restates something that's obviously already known\n\
    - Research notes: survey results, market analysis, feature lists\n\
    \n\
    Reply with ONLY one word: APPROVE or REJECT";

async fn llm_promotion_gate(cfg: &AiConfig, content: &str) -> Result<bool, String> {
    let truncated = if content.len() > 500 {
        &content[..content.char_indices().take(500).last()
            .map(|(i, c)| i + c.len_utf8()).unwrap_or(500)]
    } else {
        content
    };
    let response = ai::llm_chat_as(cfg, "gate", GATE_SYSTEM, truncated).await?;
    let decision = response.trim().to_uppercase();
    Ok(decision.starts_with("APPROVE"))
}

const MERGE_SYSTEM: &str = "Merge these related memory entries into a single concise note. Rules:\n\
    - Preserve ALL specific names, tools, libraries, versions, and technical terms.\n\
    - If one entry updates or supersedes the other, keep the latest state.\n\
    - Remove only truly redundant/repeated sentences.\n\
    - Names, numbers, versions, dates, tool names > vague summaries. Never drop specific terms.\n\
    - Keep it under 400 characters if possible.\n\
    - Same language as originals. Output only the merged text, nothing else.";

async fn merge_similar(db: &SharedDB, cfg: &AiConfig) -> (usize, Vec<String>) {
    let db2 = db.clone();
    let all = tokio::task::spawn_blocking(move || db2.get_all_with_embeddings())
        .await
        .unwrap_or_else(|e| {
            warn!(error = %e, "get_all_with_embeddings task failed");
            vec![]
        });

    if all.len() < 2 {
        return (0, vec![]);
    }

    let mut merged_total = 0;
    let mut merged_ids = Vec::new();

    for layer in [Layer::Buffer, Layer::Working, Layer::Core] {
        let layer_mems: Vec<&(Memory, Vec<f64>)> =
            all.iter().filter(|(m, _)| m.layer == layer).collect();

        if layer_mems.len() < 2 {
            continue;
        }

        // group by namespace so we never merge across agents
        let mut by_ns: std::collections::HashMap<&str, Vec<usize>> =
            std::collections::HashMap::new();
        for (i, (m, _)) in layer_mems.iter().enumerate() {
            by_ns.entry(&m.namespace).or_default().push(i);
        }

        for ns_indices in by_ns.values() {
            if ns_indices.len() < 2 {
                continue;
            }
            let ns_mems: Vec<&(Memory, Vec<f64>)> =
                ns_indices.iter().map(|&i| layer_mems[i]).collect();

        // text-embedding-3-small produces lower cosine scores for short CJK text,
        // but 0.68 was too aggressive — it merged related-but-distinct memories
        // (e.g. two v0.6.0 progress notes), destroying specific terms like "r2d2".
        // 0.78 limits merging to near-duplicates with high content overlap.
        let clusters = find_clusters(&ns_mems, 0.78);

        for cluster in clusters {
            if cluster.len() < 2 {
                continue;
            }

            let mut input = String::new();
            for (i, &idx) in cluster.iter().enumerate() {
                use std::fmt::Write;
                let _ = writeln!(input, "{}. {}", i + 1, ns_mems[idx].0.content);
            }

            let merged_content = match ai::llm_chat_as(cfg, "merge", MERGE_SYSTEM, &input).await {
                Ok(text) => text.trim().to_string(),
                Err(e) => {
                    warn!(error = %e, "LLM merge failed, skipping cluster");
                    continue;
                }
            };

            if merged_content.is_empty() {
                continue;
            }

            // Hard cap: if the LLM ignored the length instruction, truncate.
            // Also warn on outputs > 500 chars.
            let merged_len = merged_content.chars().count();
            let merged_content = if merged_len > 600 {
                warn!("merge output too long ({merged_len}), truncating to 600");
                merged_content.chars().take(600).collect::<String>()
            } else {
                merged_content
            };

            // Skip if merged output isn't shorter than total input — LLM failed to condense
            let total_input_len: usize = cluster.iter()
                .map(|&i| ns_mems[i].0.content.chars().count())
                .sum();
            if total_input_len == 0 || merged_content.chars().count() >= total_input_len {
                let preview: String = cluster.iter()
                    .map(|&i| ns_mems[i].0.content.chars().take(40).collect::<String>())
                    .collect::<Vec<_>>()
                    .join(" | ");
                warn!("merge produced longer output than inputs ({} >= {}), skipping: {}",
                    merged_content.chars().count(), total_input_len, preview);
                continue;
            }

            // keep the most recently created entry as the winner —
            // if two memories conflict, the newer one is more likely correct
            let Some(&best_idx) = cluster
                .iter()
                .max_by_key(|&&i| ns_mems[i].0.created_at)
            else {
                continue;
            };

            // take the highest importance from the cluster
            let max_importance = cluster
                .iter()
                .map(|&i| ns_mems[i].0.importance)
                .fold(0.0_f64, f64::max);

            // sum access counts — merged memory inherits all usage history
            let total_access: i64 = cluster
                .iter()
                .map(|&i| ns_mems[i].0.access_count)
                .sum();

            // merge all tags (cap at 20)
            let mut all_tags: Vec<String> = Vec::new();
            for &idx in &cluster {
                for tag in &ns_mems[idx].0.tags {
                    if !all_tags.contains(tag) {
                        all_tags.push(tag.clone());
                    }
                }
            }
            all_tags.truncate(20);

            // update the winner
            let best_id = ns_mems[best_idx].0.id.clone();
            {
                let db2 = db.clone();
                let id = best_id.clone();
                let content = merged_content.clone();
                let tags = all_tags;
                let imp = max_importance;
                let result = tokio::task::spawn_blocking(move || {
                    db2.update_fields(&id, Some(&content), None, Some(imp), Some(&tags))?;
                    // Carry over accumulated access history from all merged memories
                    db2.set_access_count(&id, total_access)?;
                    Ok::<_, EngramError>(())
                })
                .await;

                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        warn!(id = %best_id, error = %e, "merge update failed");
                        continue;
                    }
                    Err(e) => {
                        warn!(id = %best_id, error = %e, "merge task panicked");
                        continue;
                    }
                }
            }

            // regenerate embedding for merged content
            let embed_ok = if cfg.has_embed() {
                match ai::get_embeddings(cfg, &[merged_content]).await {
                    Ok(embs) if !embs.is_empty() => {
                        if let Some(emb) = embs.into_iter().next() {
                            let db2 = db.clone();
                            let id = best_id.clone();
                            let res = tokio::task::spawn_blocking(move || {
                                db2.set_embedding(&id, &emb)
                            }).await;
                            res.is_ok()
                        } else { true }
                    }
                    Err(e) => {
                        warn!(error = %e, "embedding for merged memory failed, skipping loser deletion");
                        false
                    }
                    _ => true
                }
            } else { true };

            if !embed_ok {
                continue; // don't delete losers without a valid winner embedding
            }

            // delete the rest
            let mut absorbed = Vec::new();
            for &idx in &cluster {
                if idx == best_idx {
                    continue;
                }
                let loser = &ns_mems[idx].0;
                absorbed.push(loser.content.chars().take(60).collect::<String>());
                let id = ns_mems[idx].0.id.clone();
                let db2 = db.clone();
                let _ = tokio::task::spawn_blocking(move || db2.delete(&id)).await;
            }

            let winner_preview: String = ns_mems[best_idx].0.content.chars().take(60).collect();
            info!(
                winner = %best_id,
                absorbed = ?absorbed,
                "merged {} memories: '{}'",
                cluster.len(), winner_preview,
            );

            merged_total += 1;
            merged_ids.push(best_id);
        }
        } // ns_indices
    }

    if merged_total > 0 {
        info!(merged = merged_total, "memory merge complete");
    }

    (merged_total, merged_ids)
}

fn find_clusters(mems: &[&(Memory, Vec<f64>)], threshold: f64) -> Vec<Vec<usize>> {
    let n = mems.len();
    let mut used = vec![false; n];
    let mut clusters = Vec::new();

    for i in 0..n {
        if used[i] {
            continue;
        }
        used[i] = true;
        let mut cluster = vec![i];

        for j in (i + 1)..n {
            if used[j] {
                continue;
            }
            if cosine_similarity(&mems[i].1, &mems[j].1) > threshold {
                cluster.push(j);
                used[j] = true;
            }
        }

        clusters.push(cluster);
    }

    clusters
}

// --- Memory Audit ---

const AUDIT_SYSTEM: &str = r#"You are reviewing an AI agent's memory store. You see ALL memories organized by layer.
Your job: reorganize them. Output a JSON array of operations.

## Layers
- Core (3): Permanent. Identity, values, key relationships, hard-won lessons, strategic goals.
- Working (2): Useful but not permanent. Project context, recent learnings, operational knowledge.
- Buffer (1): Temporary. Will auto-expire. Session logs, transient notes.

## Operations (output as JSON array)
- {"op":"promote","id":"<8-char-id>","to":3} — move to Core (only for truly permanent knowledge)
- {"op":"demote","id":"<8-char-id>","to":2} or {"op":"demote","id":"<8-char-id>","to":1}
- {"op":"merge","ids":["id1","id2"],"content":"merged text","layer":2,"tags":["tag1"]}
- {"op":"delete","id":"<8-char-id>"} — remove (duplicate, obsolete, or garbage)

## Guidelines
- Core should be SMALL: identity, values, lessons, key relationships, strategic constraints
- Technical details, bug fixes, version notes, config values → Working at most
- Near-duplicate memories → merge into one, keep the best content
- Session logs older than a few days with no unique insight → delete
- Be aggressive about merges. If two memories say similar things, merge them.
- Output ONLY a valid JSON array. Empty array [] if no changes needed."#;

/// Full audit: reviews Core+Working memories using the gate model.
/// Chunks automatically if total prompt would exceed ~100K chars.
pub async fn audit_memories(cfg: &AiConfig, db: &crate::db::MemoryDB) -> Result<AuditResult, String> {
    let core = db.list_by_layer_meta(crate::db::Layer::Core, 500, 0);
    let working = db.list_by_layer_meta(crate::db::Layer::Working, 500, 0);

    let all: Vec<&crate::db::Memory> = core.iter().chain(working.iter()).collect();

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
        let response = ai::llm_chat_as(cfg, "gate", AUDIT_SYSTEM, &prompt).await?;
        let ops = parse_audit_ops(&response, &core, &working);
        apply_audit_ops(db, ops, &mut combined);
    } else {
        // chunked: Core summary + Working in batches
        let core_summary = format_layer_summary("Core", &core);
        for chunk in working.chunks(max_per_batch.saturating_sub(core.len())) {
            let mut prompt = core_summary.clone();
            prompt.push_str(&format!("\n## Working Layer (batch of {})\n", chunk.len()));
            for m in chunk {
                let tags = m.tags.join(",");
                let preview: String = m.content.chars().take(200).collect();
                prompt.push_str(&format!("- [{}] (imp={:.1}, acc={}, tags=[{}]) {}\n",
                    &m.id[..8], m.importance, m.access_count, tags, preview));
            }

            let response = ai::llm_chat_as(cfg, "gate", AUDIT_SYSTEM, &prompt).await?;
            // for chunked mode, resolve against core + this chunk only
            let chunk_vec: Vec<crate::db::Memory> = chunk.to_vec();
            let ops = parse_audit_ops(&response, &core, &chunk_vec);
            apply_audit_ops(db, ops, &mut combined);
        }
    }

    Ok(combined)
}

fn format_audit_prompt(core: &[crate::db::Memory], working: &[crate::db::Memory]) -> String {
    let mut prompt = String::with_capacity(16_000);
    prompt.push_str("## Core Layer\n");
    for m in core {
        let tags = m.tags.join(",");
        let preview: String = m.content.chars().take(200).collect();
        prompt.push_str(&format!("- [{}] (imp={:.1}, acc={}, tags=[{}]) {}\n",
            &m.id[..8], m.importance, m.access_count, tags, preview));
    }
    prompt.push_str(&format!("\n## Working Layer ({} memories)\n", working.len()));
    for m in working {
        let tags = m.tags.join(",");
        let preview: String = m.content.chars().take(200).collect();
        prompt.push_str(&format!("- [{}] (imp={:.1}, acc={}, tags=[{}]) {}\n",
            &m.id[..8], m.importance, m.access_count, tags, preview));
    }
    prompt
}

fn format_layer_summary(name: &str, memories: &[crate::db::Memory]) -> String {
    let mut s = format!("## {} Layer ({} memories, shown for context — do NOT reorganize these)\n", name, memories.len());
    for m in memories {
        let preview: String = m.content.chars().take(80).collect();
        s.push_str(&format!("- [{}] {}\n", &m.id[..8], preview));
    }
    s
}

fn apply_audit_ops(db: &crate::db::MemoryDB, ops: Vec<AuditOp>, result: &mut AuditResult) {
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
                if db.delete(id).is_ok() {
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

#[derive(Debug, Clone, serde::Serialize)]
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

#[derive(Debug, Default, serde::Serialize)]
pub struct AuditResult {
    pub total_reviewed: usize,
    pub promoted: usize,
    pub demoted: usize,
    pub deleted: usize,
    pub merged: usize,
    pub ops: Vec<AuditOp>,
}

fn parse_audit_ops(
    response: &str,
    core: &[crate::db::Memory],
    working: &[crate::db::Memory],
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
            id_map.insert(m.id[..8].to_string(), m.id.clone());
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
                    item.get("to").and_then(|v| v.as_u64()).map(|n| n as u8),
                ) {
                    if (1..=3).contains(&to) { ops.push(AuditOp::Promote { id, to }); }
                }
            }
            "demote" => {
                if let (Some(id), Some(to)) = (
                    item.get("id").and_then(|v| v.as_str()).and_then(&resolve),
                    item.get("to").and_then(|v| v.as_u64()).map(|n| n as u8),
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
                let layer = item.get("layer").and_then(|v| v.as_u64()).unwrap_or(2) as u8;
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

    fn make_mem(id: &str, layer: Layer, importance: f64, emb: Vec<f64>) -> (Memory, Vec<f64>) {
        (
            Memory {
                id: id.into(),
                content: format!("memory {id}"),
                layer,
                importance,
                created_at: 0,
                last_accessed: 0,
                access_count: 0,
                repetition_count: 0,
                decay_rate: 1.0,
                source: "test".into(),
                tags: vec![],
                namespace: "default".into(),
                embedding: None,
                risk_score: 0.0,
                kind: "semantic".into(),
            },
            emb,
        )
    }

    #[test]
    fn cluster_similar_vectors() {
        let mems = [
            make_mem("a", Layer::Working, 0.5, vec![1.0, 0.0, 0.0]),
            make_mem("b", Layer::Working, 0.5, vec![0.999, 0.01, 0.0]),
            make_mem("c", Layer::Working, 0.5, vec![0.0, 1.0, 0.0]),
        ];
        let refs: Vec<&(Memory, Vec<f64>)> = mems.iter().collect();
        let clusters = find_clusters(&refs, 0.85);

        assert_eq!(clusters.len(), 2);
        let big = clusters.iter().find(|c| c.len() == 2).unwrap();
        assert!(big.contains(&0) && big.contains(&1));
    }

    #[test]
    fn cluster_all_different() {
        let mems = [
            make_mem("a", Layer::Working, 0.5, vec![1.0, 0.0, 0.0]),
            make_mem("b", Layer::Working, 0.5, vec![0.0, 1.0, 0.0]),
            make_mem("c", Layer::Working, 0.5, vec![0.0, 0.0, 1.0]),
        ];
        let refs: Vec<&(Memory, Vec<f64>)> = mems.iter().collect();
        let clusters = find_clusters(&refs, 0.85);

        assert_eq!(clusters.len(), 3);
        assert!(clusters.iter().all(|c| c.len() == 1));
    }

    #[test]
    fn cluster_all_identical() {
        let mems = [
            make_mem("a", Layer::Working, 0.5, vec![1.0, 0.0]),
            make_mem("b", Layer::Working, 0.5, vec![1.0, 0.0]),
            make_mem("c", Layer::Working, 0.5, vec![1.0, 0.0]),
        ];
        let refs: Vec<&(Memory, Vec<f64>)> = mems.iter().collect();
        let clusters = find_clusters(&refs, 0.85);

        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 3);
    }

    #[test]
    fn cluster_empty_input() {
        let mems: Vec<&(Memory, Vec<f64>)> = vec![];
        let clusters = find_clusters(&mems, 0.85);
        assert!(clusters.is_empty());
    }

    // --- consolidate_sync tests ---
    // These use import() to set up memories with specific timestamps.

    fn test_db() -> MemoryDB {
        MemoryDB::open(":memory:").expect("in-memory db")
    }

    fn mem_with_ts(
        id: &str,
        layer: Layer,
        importance: f64,
        access_count: i64,
        created_ms: i64,
        accessed_ms: i64,
    ) -> Memory {
        Memory {
            id: id.into(),
            content: format!("test memory {id}"),
            layer,
            importance,
            created_at: created_ms,
            last_accessed: accessed_ms,
            access_count,
            repetition_count: 0,
            decay_rate: layer.default_decay(),
            source: "test".into(),
            tags: vec![],
            namespace: "default".into(),
            embedding: None,
            risk_score: 0.0,
            kind: "semantic".into(),
        }
    }

    #[test]
    fn promote_high_access_working() {
        let db = test_db();
        let now = crate::db::now_ms();
        // working memory with enough accesses and importance → should be a candidate
        let good = mem_with_ts("promote-me", Layer::Working, 0.8, 5, now - 1000, now);
        // working memory with low access → should not be a candidate
        let meh = mem_with_ts("leave-me", Layer::Working, 0.8, 1, now - 1000, now);
        db.import(&[good, meh]).unwrap();

        let result = consolidate_sync(&db, None);
        // Without LLM, candidates are collected but not promoted
        assert_eq!(result.promotion_candidates.len(), 1);
        assert_eq!(result.promotion_candidates[0].0, "promote-me");

        // Simulate no-AI fallback: promote candidates directly
        for (id, _) in &result.promotion_candidates {
            db.promote(id, Layer::Core).unwrap();
        }
        let promoted = db.get("promote-me").unwrap().unwrap();
        assert_eq!(promoted.layer, Layer::Core);
        let stayed = db.get("leave-me").unwrap().unwrap();
        assert_eq!(stayed.layer, Layer::Working);
    }

    #[test]
    fn age_promote_old_working() {
        let db = test_db();
        let now = crate::db::now_ms();
        let eight_days_ago = now - 8 * 86_400_000;
        // old working memory with decent importance → should be a candidate by age
        let old = mem_with_ts("old-but-worthy", Layer::Working, 0.6, 1, eight_days_ago, now);
        // fresh working memory → should not be a candidate
        let fresh = mem_with_ts("too-young", Layer::Working, 0.6, 1, now - 1000, now);
        db.import(&[old, fresh]).unwrap();

        let result = consolidate_sync(&db, None);
        let candidate_ids: Vec<&str> = result.promotion_candidates.iter().map(|(id, _)| id.as_str()).collect();
        assert!(candidate_ids.contains(&"old-but-worthy"));
        assert!(!candidate_ids.contains(&"too-young"));
    }

    #[test]
    fn drop_expired_low_importance_buffer() {
        let db = test_db();
        let now = crate::db::now_ms();
        let two_days_ago = now - 2 * 86_400_000;
        // old buffer, never accessed → should be dropped after 24h TTL
        let expendable = mem_with_ts("bye", Layer::Buffer, 0.2, 0, two_days_ago, two_days_ago);
        // old buffer, accessed enough (≥ rescue threshold of 2) → rescued to working
        let valuable = mem_with_ts("save-me", Layer::Buffer, 0.5, 3, two_days_ago, two_days_ago);
        // old buffer, accessed once → below rescue threshold, dropped
        let barely = mem_with_ts("not-enough", Layer::Buffer, 0.5, 1, two_days_ago, two_days_ago);
        db.import(&[expendable, valuable, barely]).unwrap();

        let _result = consolidate_sync(&db, None);
        assert!(db.get("bye").unwrap().is_none(), "never-accessed buffer should be gone");
        assert!(db.get("not-enough").unwrap().is_none(), "barely-accessed buffer should be gone after TTL");
        let saved = db.get("save-me").unwrap();
        assert!(saved.is_some(), "well-accessed buffer should survive");
    }

    #[test]
    fn nothing_to_do() {
        let db = test_db();
        let now = crate::db::now_ms();
        // fresh core memory — nothing should happen
        let stable = mem_with_ts("stable", Layer::Core, 0.9, 10, now - 1000, now);
        db.import(&[stable]).unwrap();

        let r = consolidate_sync(&db, None);
        assert_eq!(r.promoted, 0);
        assert_eq!(r.decayed, 0);
    }

    #[test]
    fn buffer_promoted_by_access() {
        let db = test_db();
        let now = crate::db::now_ms();
        // Buffer memory with enough accesses (≥5) should promote to Working
        let accessed = mem_with_ts("recalled", Layer::Buffer, 0.1, 6, now - 1000, now);
        // Buffer with only 3 accesses — not enough, stays in Buffer
        let not_enough = mem_with_ts("still-young", Layer::Buffer, 0.1, 3, now - 1000, now);
        db.import(&[accessed, not_enough]).unwrap();

        let r = consolidate_sync(&db, None);
        assert_eq!(r.promoted, 1);
        let got = db.get("recalled").unwrap().unwrap();
        assert_eq!(got.layer, Layer::Working);
        let stayed = db.get("still-young").unwrap().unwrap();
        assert_eq!(stayed.layer, Layer::Buffer);
    }

    #[test]
    fn buffer_ttl_accessed_enough_promotes() {
        let db = test_db();
        let two_days_ago = crate::db::now_ms() - 2 * 86_400_000;
        // Old buffer with 3 accesses (≥ rescue threshold of 2) — should rescue to Working
        let accessed = mem_with_ts("rescued", Layer::Buffer, 0.1, 3, two_days_ago, two_days_ago);
        // Old buffer with 1 access — below rescue threshold, should be dropped
        let barely = mem_with_ts("barely", Layer::Buffer, 0.1, 1, two_days_ago, two_days_ago);
        db.import(&[accessed, barely]).unwrap();

        let r = consolidate_sync(&db, None);
        assert!(r.promoted >= 1);
        let got = db.get("rescued").unwrap().unwrap();
        assert_eq!(got.layer, Layer::Working);
        assert!(db.get("barely").unwrap().is_none(), "barely-accessed buffer should be dropped after TTL");
    }

    #[test]
    fn buffer_ttl_never_accessed_drops() {
        let db = test_db();
        let two_days_ago = crate::db::now_ms() - 2 * 86_400_000;
        // Old buffer with 0 accesses — should be dropped after TTL (24h)
        let unused = mem_with_ts("forgotten", Layer::Buffer, 0.1, 0, two_days_ago, two_days_ago);
        db.import(&[unused]).unwrap();

        let r = consolidate_sync(&db, None);
        assert!(r.decayed >= 1);
        assert!(db.get("forgotten").unwrap().is_none());
    }

    #[test]
    fn operational_tag_blocks_working_to_core_promotion() {
        let db = test_db();
        let now = crate::db::now_ms();

        // Session-tagged memory — high importance and access count,
        // but must NOT become a candidate (blocked before LLM gate).
        let mut session_mem = mem_with_ts("session-mem", Layer::Working, 0.9, 5, now - 1000, now);
        session_mem.source = "session".into();

        // Normal working memory with identical importance/access — SHOULD be a candidate.
        let normal = mem_with_ts("normal-mem", Layer::Working, 0.9, 5, now - 1000, now);

        // Ephemeral-tagged memory — also blocked.
        let mut ephemeral = mem_with_ts("ephemeral-mem", Layer::Working, 0.9, 5, now - 1000, now);
        ephemeral.tags = vec!["ephemeral".into()];

        db.import(&[session_mem, normal, ephemeral]).unwrap();

        let result = consolidate_sync(&db, None);
        let candidate_ids: Vec<&str> = result.promotion_candidates.iter().map(|(id, _)| id.as_str()).collect();

        assert!(candidate_ids.contains(&"normal-mem"), "normal memory should be a candidate");
        assert!(!candidate_ids.contains(&"session-mem"), "session memory must not be a candidate");
        assert!(!candidate_ids.contains(&"ephemeral-mem"), "ephemeral memory must not be a candidate");
    }
}
