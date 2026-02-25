use std::collections::HashSet;
use crate::ai::{self, AiConfig};
use crate::db::{Layer, Memory, MemoryDB};
use crate::error::EngramError;
use crate::prompts;
use crate::util::truncate_chars;
use crate::SharedDB;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Repetition carries more weight than incidental recall hits.
const REPETITION_WEIGHT: f64 = 2.5;

fn reinforcement_score(mem: &Memory) -> f64 {
    mem.access_count as f64 + mem.repetition_count as f64 * REPETITION_WEIGHT
}

mod audit;
mod cluster;
mod distill;
mod facts;
mod merge;
mod sandbox;
mod summary;
mod triage;

pub use audit::{audit_memories, AuditOp, AuditResult, RawAuditOp, AuditToolResponse, audit_tool_schema, resolve_audit_ops};
pub use sandbox::{sandbox_audit, SandboxResult, Grade, OpGrade, RuleChecker};
pub use merge::{find_clusters, reconcile_pair_key};
pub use triage::dedup_buffer;

use distill::distill_sessions;
use merge::{merge_similar, reconcile_updates};
use summary::update_core_summary;
use triage::{triage_buffer, heuristic_triage_buffer_sync};

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
    /// Number of memories updated via reconciliation (newer superseded older).
    #[serde(skip_serializing_if = "is_zero")]
    pub reconciled: usize,
    /// IDs of memories that were superseded (deleted) during reconciliation.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub reconciled_ids: Vec<String>,
    /// Number of duplicate buffer memories removed.
    #[serde(skip_serializing_if = "is_zero")]
    pub buffer_deduped: usize,
    /// Number of buffer memories promoted by LLM triage.
    #[serde(skip_serializing_if = "is_zero")]
    pub triaged: usize,
    /// Number of facts extracted from existing memories.
    #[serde(skip_serializing_if = "is_zero")]
    pub facts_extracted: usize,
    /// IDs that passed access/age thresholds but await LLM gate review.
    #[serde(skip)]
    pub promotion_candidates: Vec<(String, String, i64, i64, f64, Vec<String>)>, // (id, content, access_count, repetition_count, importance, tags)
    /// Number of session notes distilled into project context.
    #[serde(skip_serializing_if = "is_zero")]
    pub distilled: usize,
}

fn is_zero(n: &usize) -> bool { *n == 0 }

/// Session note check: source="session" OR tag="session". API-created session
/// memories often have source="api" + tag="session", so both must be checked.
fn is_session(mem: &Memory) -> bool {
    mem.source == "session" || mem.tags.iter().any(|t| t == "session")
}

pub async fn consolidate(
    db: SharedDB,
    req: Option<ConsolidateRequest>,
    ai: Option<AiConfig>,
) -> ConsolidateResponse {
    let do_merge = req.as_ref().and_then(|r| r.merge).unwrap_or(false);

    let llm_level = std::env::var("ENGRAM_LLM_LEVEL")
        .unwrap_or_else(|_| "auto".to_string())
        .to_lowercase();

    let db2 = db.clone();
    let llm_level_sync = llm_level.clone();
    let mut result = tokio::task::spawn_blocking(move || {
        consolidate_sync(&db2, req.as_ref(), &llm_level_sync)
    })
    .await
    .unwrap_or_else(|e| {
        warn!(error = %e, "consolidate_sync task panicked");
        ConsolidateResponse::default()
    });

    // LLM gate: review promotion candidates before moving to Core.
    // Behavior depends on ENGRAM_LLM_LEVEL:
    //   full  — always call LLM (original behavior)
    //   auto  — heuristic pre-filter, LLM for uncertain cases only
    //   off   — pure heuristic gate, no LLM calls
    // Without AI config, promote all candidates directly (backward compat).
    let candidates = std::mem::take(&mut result.promotion_candidates);
    if !candidates.is_empty() {
        match llm_level.as_str() {
            "off" => {
                // Pure heuristic gate: ac >= 3 AND importance >= 0.6
                for (id, _content, access_count, _rep_count, importance, _tags) in &candidates {
                    if *access_count >= 3 && *importance >= 0.6 {
                        let db2 = db.clone();
                        let id2 = id.clone();
                        let promoted = tokio::task::spawn_blocking(move || {
                            db2.promote(&id2, Layer::Core).is_ok()
                        }).await.unwrap_or(false);
                        if promoted {
                            info!(id = %id, "auto-approved {} for Core (heuristic: off-mode gate)", crate::util::short_id(id));
                            result.promoted_ids.push(id.clone());
                            result.promoted += 1;
                        }
                    }
                    // In off mode, don't tag-reject — memory stays in Working
                    // for proper LLM evaluation when mode changes back.
                }
            }
            "auto" => {
                // Pre-filter with heuristics, send uncertain to LLM.
                let mut uncertain = Vec::new();
                for cand in &candidates {
                    let (id, _content, ac, rep, imp, tags) = cand;
                    // Auto-approve: strong signal
                    if *ac >= 5
                        && (*rep >= 2 || tags.iter().any(|t| t == "lesson" || t == "procedural"))
                        && *imp >= 0.6
                    {
                        let db2 = db.clone();
                        let id2 = id.clone();
                        let promoted = tokio::task::spawn_blocking(move || {
                            db2.promote(&id2, Layer::Core).is_ok()
                        }).await.unwrap_or(false);
                        if promoted {
                            info!(id = %id, "auto-approved {} for Core (heuristic: strong candidate)", crate::util::short_id(id));
                            result.promoted_ids.push(id.clone());
                            result.promoted += 1;
                        }
                    } else {
                        uncertain.push(cand);
                    }
                }
                // Send uncertain to LLM
                if !uncertain.is_empty() {
                    match &ai {
                        Some(cfg) => {
                            for (id, content, access_count, rep_count, _importance, _tags) in uncertain {
                                match llm_promotion_gate(cfg, content, *access_count, *rep_count).await {
                                    Ok((true, kind, usage, model, duration_ms)) => {
                                        if let Some(ref u) = usage {
                                            let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
                                            let _ = db.log_llm_call("gate", &model, u.prompt_tokens, u.completion_tokens, cached, duration_ms);
                                        }
                                        let db2 = db.clone();
                                        let id2 = id.clone();
                                        let promoted = tokio::task::spawn_blocking(move || -> bool {
                                            if db2.promote(&id2, Layer::Core).is_ok() {
                                                if let Some(k) = kind {
                                                    if let Ok(Some(mem)) = db2.get(&id2) {
                                                        if mem.kind == "semantic" {
                                                            let _ = db2.update_kind(&id2, &k);
                                                        }
                                                    }
                                                }
                                                true
                                            } else {
                                                false
                                            }
                                        }).await.unwrap_or(false);
                                        if promoted {
                                            result.promoted_ids.push(id.clone());
                                            result.promoted += 1;
                                            debug!(id = %id, "promoted to Core");
                                        }
                                    }
                                    Ok((false, _, usage, model, duration_ms)) => {
                                        if let Some(ref u) = usage {
                                            let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
                                            let _ = db.log_llm_call("gate", &model, u.prompt_tokens, u.completion_tokens, cached, duration_ms);
                                        }
                                        result.gate_rejected += 1;
                                        let db2 = db.clone();
                                        let id2 = id.clone();
                                        let _ = tokio::task::spawn_blocking(move || {
                                            if let Ok(Some(mem)) = db2.get(&id2) {
                                                let mut tags: Vec<String> = mem.tags.iter()
                                                    .filter(|t| !t.starts_with("gate-rejected"))
                                                    .cloned().collect();
                                                let had_2 = mem.tags.iter().any(|t| t == "gate-rejected-2");
                                                let had_1 = mem.tags.iter().any(|t| t == "gate-rejected");
                                                let new_tag = if had_2 {
                                                    "gate-rejected-final"
                                                } else if had_1 {
                                                    "gate-rejected-2"
                                                } else {
                                                    "gate-rejected"
                                                };
                                                tags.push(new_tag.into());
                                                let _ = db2.update_fields(&id2, None, None, None, Some(&tags));
                                            }
                                        }).await;
                                        info!(id = %id, content = %truncate_chars(content, 50), "gate rejected promotion to Core");
                                    }
                                    Err(e) => {
                                        warn!(id = %id, error = %e, "LLM gate failed, skipping");
                                    }
                                }
                            }
                        }
                        None => {
                            // No AI config — promote all uncertain candidates (backward compat)
                            let db2 = db.clone();
                            let cands: Vec<String> = uncertain.iter().map(|(id, _, _, _, _, _)| id.clone()).collect();
                            let promoted_ids: Vec<String> = tokio::task::spawn_blocking(move || {
                                cands.into_iter().filter(|id| db2.promote(id, Layer::Core).is_ok()).collect::<Vec<_>>()
                            }).await.unwrap_or_default();
                            result.promoted += promoted_ids.len();
                            result.promoted_ids.extend(promoted_ids);
                        }
                    }
                }
            }
            _ => {
                // "full" mode — original behavior: always call LLM
                match &ai {
                    Some(cfg) => {
                        for (id, content, access_count, rep_count, _importance, _tags) in &candidates {
                            match llm_promotion_gate(cfg, content, *access_count, *rep_count).await {
                                Ok((true, kind, usage, model, duration_ms)) => {
                                    if let Some(ref u) = usage {
                                        let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
                                        let _ = db.log_llm_call("gate", &model, u.prompt_tokens, u.completion_tokens, cached, duration_ms);
                                    }
                                    let db2 = db.clone();
                                    let id2 = id.clone();
                                    let promoted = tokio::task::spawn_blocking(move || -> bool {
                                        if db2.promote(&id2, Layer::Core).is_ok() {
                                            if let Some(k) = kind {
                                                if let Ok(Some(mem)) = db2.get(&id2) {
                                                    if mem.kind == "semantic" {
                                                        let _ = db2.update_kind(&id2, &k);
                                                    }
                                                }
                                            }
                                            true
                                        } else {
                                            false
                                        }
                                    }).await.unwrap_or(false);
                                    if promoted {
                                        result.promoted_ids.push(id.clone());
                                        result.promoted += 1;
                                        debug!(id = %id, "promoted to Core");
                                    }
                                }
                                Ok((false, _, usage, model, duration_ms)) => {
                                    if let Some(ref u) = usage {
                                        let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
                                        let _ = db.log_llm_call("gate", &model, u.prompt_tokens, u.completion_tokens, cached, duration_ms);
                                    }
                                    result.gate_rejected += 1;
                                    let db2 = db.clone();
                                    let id2 = id.clone();
                                    let _ = tokio::task::spawn_blocking(move || {
                                        if let Ok(Some(mem)) = db2.get(&id2) {
                                            let mut tags: Vec<String> = mem.tags.iter()
                                                .filter(|t| !t.starts_with("gate-rejected"))
                                                .cloned().collect();
                                            let had_2 = mem.tags.iter().any(|t| t == "gate-rejected-2");
                                            let had_1 = mem.tags.iter().any(|t| t == "gate-rejected");
                                            let new_tag = if had_2 {
                                                "gate-rejected-final"
                                            } else if had_1 {
                                                "gate-rejected-2"
                                            } else {
                                                "gate-rejected"
                                            };
                                            tags.push(new_tag.into());
                                            let _ = db2.update_fields(&id2, None, None, None, Some(&tags));
                                        }
                                    }).await;
                                    info!(id = %id, content = %truncate_chars(content, 50), "gate rejected promotion to Core");
                                }
                                Err(e) => {
                                    warn!(id = %id, error = %e, "LLM gate failed, skipping");
                                }
                            }
                        }
                    }
                    None => {
                        let db2 = db.clone();
                        let cands: Vec<String> = candidates.iter().map(|(id, _, _, _, _, _)| id.clone()).collect();
                        let promoted_ids: Vec<String> = tokio::task::spawn_blocking(move || {
                            cands.into_iter().filter(|id| db2.promote(id, Layer::Core).is_ok()).collect::<Vec<_>>()
                        }).await.unwrap_or_default();
                        result.promoted += promoted_ids.len();
                        result.promoted_ids.extend(promoted_ids);
                    }
                }
            }
        }
    }

    // Reconcile: detect same-topic memories where newer one updates/supersedes older.
    // Runs every consolidation cycle (not just when merge is requested).
    // Skipped in `off` mode (requires LLM for content comparison).
    if llm_level != "off" {
        if let Some(ref cfg) = ai {
            let (count, ids) = reconcile_updates(&db, cfg).await;
            result.reconciled = count;
            result.reconciled_ids = ids;
        }
    }

    // Session distillation: turn accumulated session notes into project context.
    // Session notes can't promote to Core (by design), but the project knowledge
    // they contain is valuable. When 3+ session notes pile up, LLM distills them
    // into a single project-status memory that CAN promote normally.
    // Skipped in `off` mode (requires LLM for synthesis).
    if llm_level != "off" {
        if let Some(ref cfg) = ai {
            result.distilled = distill_sessions(&db, cfg).await;
        }
    }

    // Buffer dedup: remove near-duplicate entries within the buffer layer.
    // No LLM needed — purely cosine-based. Keeps the newest, accumulates access counts.
    {
        let db2 = db.clone();
        let deduped = tokio::task::spawn_blocking(move || dedup_buffer(&db2))
            .await
            .unwrap_or(0);
        result.buffer_deduped = deduped;
    }

    // Buffer triage: evaluates buffer memories for promotion.
    // In `off` mode: pure heuristic triage (no AI config needed).
    // In `auto` mode: heuristic pre-filter + LLM for uncertain cases.
    // In `full` mode: all candidates go to LLM.
    {
        let min_age = 600 * 1000; // 10 minutes — gives time for dedup to run first
        if llm_level == "off" {
            // Pure heuristic triage — no LLM needed
            let db2 = db.clone();
            let promoted_set = result.promoted_ids.clone();
            let (count, ids) = tokio::task::spawn_blocking(move || {
                heuristic_triage_buffer_sync(&db2, min_age, &promoted_set)
            }).await.unwrap_or((0, vec![]));
            result.triaged = count;
            for id in ids {
                result.promoted_ids.push(id);
            }
            result.promoted += count;
        } else if let Some(ref cfg) = ai {
            let (count, ids) = triage_buffer(&db, cfg, min_age, 20, &result.promoted_ids, &llm_level).await;
            result.triaged = count;
            for id in ids {
                result.promoted_ids.push(id);
            }
            result.promoted += count;
        }
    }

    // Extract fact triples from Working/Core memories that don't have any yet.
    // Facts auto-extraction disabled — the extracted triples were low quality
    // and duplicated information already in memories. Keep the API for manual use.
    // if let Some(ref cfg) = ai {
    //     result.facts_extracted = extract_facts_batch(&db, cfg, 5).await;
    // }

    // Merge: LLM combines near-duplicate memories. Skipped in `off` mode.
    if do_merge && llm_level != "off" {
        if let Some(ref cfg) = ai {
            let (count, ids) = merge_similar(&db, cfg).await;
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

    // Generate or update Core summary when Core is large enough to need
    // compression at resume time. The summary is cached in engram_meta and
    // invalidated by a hash of Core memory IDs + content lengths.
    // Skipped in `off` mode (requires LLM for summarization).
    if llm_level != "off" {
        if let Some(ref cfg) = ai {
            update_core_summary(&db, cfg).await;
        }
    }

    result
}

/// Strip stale process tags that should not persist:
/// - `gate-rejected` on Working memories where `last_accessed` is > 24 h ago
/// - `promotion` on anything already in Working or Core
pub fn cleanup_stale_tags(db: &MemoryDB) {
    let now = crate::db::now_ms();
    let stale_cutoff = 24 * 3600 * 1000_i64;
    let mut cleaned = 0_usize;

    for mem in db.list_by_layer_meta(Layer::Working, 10000, 0).unwrap_or_default() {
        if mem.tags.iter().any(|t| t == "gate-rejected")
            && (now - mem.last_accessed) > stale_cutoff
        {
            let new_tags: Vec<String> = mem.tags.iter()
                .filter(|t| t.as_str() != "gate-rejected")
                .cloned().collect();
            if db.update_fields(&mem.id, None, None, None, Some(&new_tags)).is_ok() {
                cleaned += 1;
            }
        }
    }

    for layer in [Layer::Working, Layer::Core] {
        for mem in db.list_by_layer_meta(layer, 10000, 0).unwrap_or_default() {
            if mem.tags.iter().any(|t| t == "promotion") {
                let new_tags: Vec<String> = mem.tags.iter()
                    .filter(|t| t.as_str() != "promotion")
                    .cloned().collect();
                if db.update_fields(&mem.id, None, None, None, Some(&new_tags)).is_ok() {
                    cleaned += 1;
                }
            }
        }
    }

    if cleaned > 0 {
        info!(cleaned, "stale tag cleanup");
    }
}

/// Fix buffer memories stuck by ultra-low decay rates or abandoned with no access.
pub fn fix_stuck_buffer(db: &MemoryDB) {
    let now = crate::db::now_ms();
    let two_days_ms = 48 * 3600 * 1000_i64;
    let seven_days_ms = 7 * 24 * 3600 * 1000_i64;
    let mut fixed = 0_usize;
    let mut cleaned = 0_usize;

    for mem in db.list_by_layer_meta(Layer::Buffer, 10000, 0).unwrap_or_default() {
        let age_ms = now - mem.created_at;

        // Reset stuck decay rates (< 1.0 after 48h means it'll never leave naturally)
        if age_ms > two_days_ms
            && mem.decay_rate < 1.0
            && db.update_decay_rate(&mem.id, 5.0).is_ok()
        {
            fixed += 1;
        }

        // Delete abandoned buffer entries (> 7 days, barely accessed, not protected)
        if age_ms > seven_days_ms
            && mem.access_count < 2
            && mem.kind != "procedural"
            && !mem.tags.iter().any(|t| t == "lesson")
            && db.delete(&mem.id).unwrap_or(false)
        {
            cleaned += 1;
        }
    }

    if fixed > 0 || cleaned > 0 {
        info!(decay_reset = fixed, expired_deleted = cleaned, "buffer maintenance");
    }
}

pub fn consolidate_sync(db: &MemoryDB, req: Option<&ConsolidateRequest>, llm_level: &str) -> ConsolidateResponse {
    cleanup_stale_tags(db);
    fix_stuck_buffer(db);

    let now = crate::db::now_ms();

    // --- Working layer quality rules (before main consolidation) ---

    // Rule 1: Demote auto-extracted Working memories to Buffer.
    // Proxy extraction is shut down; these are legacy noise.
    let mut auto_extract_demoted = 0usize;
    for m in db.list_by_layer_meta(Layer::Working, 10000, 0).unwrap_or_default() {
        if m.tags.iter().any(|t| t == "auto-extract") {
            if db.demote(&m.id, Layer::Buffer).is_ok() {
                auto_extract_demoted += 1;
                info!(id = %crate::util::short_id(&m.id), "demoted auto-extract from Working to Buffer");
            }
        }
    }

    // Rule 2: Delete gate-rejected Working memories older than 7 days (by modified_at).
    // The Core promotion gate already judged these unworthy; they shouldn't linger.
    let mut gate_rejected_cleaned = 0usize;
    {
        let seven_days_ms = 7 * 24 * 3600 * 1000_i64;
        for m in db.list_by_layer_meta(Layer::Working, 10000, 0).unwrap_or_default() {
            if m.tags.iter().any(|t| t == "gate-rejected")
                && (now - m.modified_at) > seven_days_ms
            {
                if db.delete(&m.id).unwrap_or(false) {
                    gate_rejected_cleaned += 1;
                    info!(id = %crate::util::short_id(&m.id), "deleted stale gate-rejected Working memory");
                }
            }
        }
    }

    if auto_extract_demoted > 0 || gate_rejected_cleaned > 0 {
        info!(auto_extract_demoted, gate_rejected_cleaned, "Working quality cleanup");
    }

    let promote_threshold = req.and_then(|r| r.promote_threshold).unwrap_or(3);
    let promote_min_imp = req.and_then(|r| r.promote_min_importance).unwrap_or(0.6);
    let decay_threshold = req.and_then(|r| r.decay_drop_threshold).unwrap_or(0.01);
    let buffer_ttl = req.and_then(|r| r.buffer_ttl_secs).unwrap_or(86400)
        .saturating_mul(1000);
    let working_age = req.and_then(|r| r.working_age_promote_secs).unwrap_or(7 * 86400)
        .saturating_mul(1000);

    let mut promoted = 0_usize;
    let mut decayed = 0_usize;
    let mut promoted_ids: HashSet<String> = HashSet::new();
    let mut dropped_ids = Vec::new();
    let mut promotion_candidates = Vec::new();

    // Demote session notes that shouldn't be in Core.
    // Session logs are episodic — they belong in Working at most.
    let mut demoted = 0_usize;
    for mem in db.list_by_layer_meta(Layer::Core, 10000, 0).unwrap_or_else(|e| { warn!(error = %e, "list_by_layer_meta(Core) failed"); vec![] }) {
        if (is_session(&mem) || mem.tags.iter().any(|t| t == "ephemeral"))
            && db.demote(&mem.id, Layer::Working).is_ok() {
                demoted += 1;
            }
    }

    // Core overlap detection: find semantically similar Core memories.
    // Incremental scan — only compare NEW Core memories (since last scan)
    // against all existing Core memories, using FTS as a pre-filter.
    {
        let last_scan_ts: i64 = db.get_meta("last_core_overlap_ts")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(0);
        let one_hour_ago = now - 3_600_000;

        let core_mems: Vec<Memory> = db.list_by_layer_meta(Layer::Core, 10000, 0)
            .unwrap_or_default();

        // New memories: created or modified after last scan AND older than 1 hour
        // (the 1-hour grace period avoids scanning memories still being processed)
        let new_mems: Vec<&Memory> = core_mems.iter()
            .filter(|m| {
                let ts = if m.modified_at > 0 { m.modified_at.max(m.created_at) } else { m.created_at };
                ts > last_scan_ts && m.created_at < one_hour_ago
            })
            .collect();

        if new_mems.is_empty() {
            info!("core overlap scan: no new memories since last scan, skipping");
        } else {
            // Build a set of all stable Core memory IDs (older than 1 hour)
            let all_core_ids: Vec<String> = core_mems.iter()
                .filter(|m| m.created_at < one_hour_ago)
                .map(|m| m.id.clone())
                .collect();
            let all_core_set: HashSet<&str> = all_core_ids.iter().map(|s| s.as_str()).collect();

            let mut candidates: Vec<(String, String, f64)> = Vec::new();

            for new_mem in &new_mems {
                // FTS pre-filter: extract keywords from the new memory's content
                // and find Core memories that share tokens
                let fts_hits: HashSet<String> = db.search_fts(&new_mem.content, 100)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(id, _rank)| id)
                    .filter(|id| *id != new_mem.id && all_core_set.contains(id.as_str()))
                    .collect();

                if fts_hits.is_empty() {
                    continue;
                }

                // Get embeddings only for this new memory + its FTS-matched candidates
                let mut ids_to_fetch: Vec<String> = vec![new_mem.id.clone()];
                ids_to_fetch.extend(fts_hits.iter().cloned());
                let embeddings = db.get_embeddings_by_ids(&ids_to_fetch);

                // Find the new memory's embedding
                let new_emb = match embeddings.iter().find(|(id, _)| id == &new_mem.id) {
                    Some((_, emb)) => emb,
                    None => continue,
                };

                // Compare against each FTS candidate
                for (cand_id, cand_emb) in &embeddings {
                    if cand_id == &new_mem.id {
                        continue;
                    }
                    let sim = crate::ai::cosine_similarity(new_emb, cand_emb);
                    if sim > crate::thresholds::CORE_OVERLAP_SIM {
                        let (id_a, id_b) = if new_mem.id < *cand_id {
                            (&new_mem.id, cand_id)
                        } else {
                            (cand_id, &new_mem.id)
                        };
                        let key = format!("overlap:{}:{}", id_a, id_b);
                        if db.get_meta(&key).is_some() {
                            continue;
                        }
                        db.set_meta(&key, &format!("{:.3}", sim)).ok();
                        candidates.push((id_a.clone(), id_b.clone(), sim));
                    }
                }
            }

            if !candidates.is_empty() {
                for (a, b, sim) in &candidates {
                    debug!(id_a = %a, id_b = %b, sim = format!("{:.3}", sim), "core overlap candidate");
                }
            }
            info!(pairs = candidates.len(), new_core = new_mems.len(), "core overlap scan complete (incremental)");
        }

        // Update the scan timestamp so next cycle only processes newer memories
        db.set_meta("last_core_overlap_ts", &now.to_string()).ok();
    }

    // Working -> Core: collect candidates for LLM gate review.
    // Reinforcement score: repetition weighs more because restating > incidental recall.
    // Session notes and ephemeral tags are always blocked.
    // gate-rejected memories get escalating retry cooldowns:
    //   1st rejection (gate-rejected): retry after 24h
    //   2nd rejection (gate-rejected-2): retry after 72h
    //   3rd rejection (gate-rejected-final): never retry, accept verdict
    let gate_retry_ms: i64 = 24 * 3600 * 1000;
    let gate_retry_2_ms: i64 = 72 * 3600 * 1000;
    for mem in db.list_by_layer_meta(Layer::Working, 10000, 0).unwrap_or_else(|e| { warn!(error = %e, "list_by_layer_meta(Working) failed"); vec![] }) {
        if is_session(&mem) || mem.tags.iter().any(|t| t == "ephemeral" || t == "auto-distilled" || t == "distilled") {
            continue;
        }
        // Final rejection — never retry
        if mem.tags.iter().any(|t| t == "gate-rejected-final") {
            continue;
        }
        // Second rejection — longer cooldown
        let is_rejected_2 = mem.tags.iter().any(|t| t == "gate-rejected-2");
        let is_rejected_1 = mem.tags.iter().any(|t| t == "gate-rejected");
        if is_rejected_2 {
            if llm_level == "auto" {
                // Auto mode: third strike — auto-reject as final, skip LLM
                let mut cleaned: Vec<String> = mem.tags.iter()
                    .filter(|t| !t.starts_with("gate-rejected"))
                    .cloned().collect();
                cleaned.push("gate-rejected-final".into());
                let _ = db.update_fields(&mem.id, None, None, None, Some(&cleaned));
                info!(id = %mem.id, "auto-rejected for Core (heuristic: gate-rejected-2, third strike)");
                continue;
            }
            if now - mem.last_accessed < gate_retry_2_ms {
                continue;
            }
            // 72h cooldown expired — remove tag for final attempt
            let cleaned: Vec<String> = mem.tags.iter()
                .filter(|t| t.as_str() != "gate-rejected-2")
                .cloned().collect();
            let _ = db.update_fields(&mem.id, None, None, None, Some(&cleaned));
            info!(id = %mem.id, "gate-rejected-2 cooldown expired, final retry");
        } else if is_rejected_1 {
            if now - mem.last_accessed < gate_retry_ms {
                continue;
            }
            // 24h cooldown expired — remove tag and let it re-evaluate
            let cleaned: Vec<String> = mem.tags.iter()
                .filter(|t| t.as_str() != "gate-rejected")
                .cloned().collect();
            let _ = db.update_fields(&mem.id, None, None, None, Some(&cleaned));
            info!(id = %mem.id, "gate-rejected cooldown expired, retrying promotion");
        }
        let score = reinforcement_score(&mem);
        let by_score = score >= promote_threshold as f64
            && mem.importance >= promote_min_imp;
        let by_age = (now - mem.created_at) > working_age
            && score > 0.0
            && mem.importance >= promote_min_imp;

        if by_score || by_age {
            promotion_candidates.push((mem.id.clone(), mem.content.clone(), mem.access_count, mem.repetition_count, mem.importance, mem.tags.clone()));
        }
    }

    // Buffer -> Working: weighted score OR lesson/procedural with cooldown.
    // Lessons and procedurals are inherently worth keeping — if they survived
    // initial dedup and 2+ hours in buffer, promote them.
    let buffer_threshold = promote_threshold.max(5) as f64;
    let lesson_cooldown_ms: i64 = 2 * 3600 * 1000; // 2 hours
    for mem in db.list_by_layer_meta(Layer::Buffer, 10000, 0).unwrap_or_else(|e| { warn!(error = %e, "list_by_layer_meta(Buffer) failed"); vec![] }) {
        // Distilled sessions already have their content in the project-status
        // summary — promoting them individually would be redundant.
        if mem.tags.iter().any(|t| t == "distilled") { continue; }

        let score = reinforcement_score(&mem);
        let is_lesson = mem.tags.iter().any(|t| t == "lesson");
        let is_procedural = mem.kind == "procedural";
        let age = now - mem.created_at;

        let by_score = score >= buffer_threshold;
        let by_kind = (is_lesson || is_procedural) && age > lesson_cooldown_ms;

        if (by_score || by_kind) && db.promote(&mem.id, Layer::Working).is_ok() {
            promoted_ids.insert(mem.id.clone());
            promoted += 1;
        }
    }

    // Buffer TTL — old L1 entries that weren't accessed enough get dropped.
    // Only rescue to Working if accessed at least half the promote threshold,
    // otherwise it wasn't important enough to keep.
    // Procedural memories and lessons are exempt — they persist indefinitely.
    // Lessons encode past mistakes; forgetting them defeats their purpose.
    for mem in db.list_by_layer_meta(Layer::Buffer, 10000, 0).unwrap_or_else(|e| { warn!(error = %e, "list_by_layer_meta(Buffer) failed"); vec![] }) {
        let is_lesson = mem.tags.iter().any(|t| t == "lesson");
        if promoted_ids.contains(&mem.id) || mem.kind == "procedural" || is_lesson {
            continue;
        }
        let age = now - mem.created_at;
        if age > buffer_ttl {
            let rescue_score = reinforcement_score(&mem);
            // Rescue if: enough access/repetition, OR importance is high enough
            // that the content itself is valuable even without being recalled.
            // This saves design decisions and architecture notes from silent death.
            let worth_keeping = rescue_score >= buffer_threshold / 2.0
                || mem.importance >= crate::thresholds::BUFFER_RESCUE_IMPORTANCE;
            if worth_keeping {
                if db.promote(&mem.id, Layer::Working).is_ok() {
                    promoted_ids.insert(mem.id.clone());
                    promoted += 1;
                }
            } else {
                // Safety net: memories never touched by any process
                // (modified_at == created_at and no _triaged tag) get 48h
                // instead of 24h, preventing data loss from triage failures.
                let was_triaged = mem.tags.iter().any(|t| t == "_triaged")
                    || mem.modified_at != mem.created_at;
                let hard_cap = buffer_ttl * 2; // 48h
                if !was_triaged && age <= hard_cap {
                    debug!(id = %mem.id, "buffer TTL extended: never triaged");
                } else if db.delete(&mem.id).unwrap_or(false) {
                    dropped_ids.push(mem.id.clone());
                    decayed += 1;
                }
            }
        }
    }

    // Buffer capacity cap — protects against proxy extraction flooding.
    // If the consolidation LLM is down or rate-limited, buffer grows unbounded.
    // Hard cap with FIFO eviction (oldest first, exempt lessons/procedural).
    let buffer_cap: usize = std::env::var("ENGRAM_BUFFER_CAP")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let buffer_mems: Vec<Memory> = db.list_by_layer_meta(Layer::Buffer, 10000, 0)
        .unwrap_or_else(|e| { warn!(error = %e, "list buffer for cap check failed"); vec![] });
    if buffer_mems.len() > buffer_cap {
        let mut evictable: Vec<&Memory> = buffer_mems.iter()
            .filter(|m| {
                m.kind != "procedural"
                    && !m.tags.iter().any(|t| t == "lesson")
                    && !promoted_ids.contains(&m.id)
            })
            .collect();
        let overflow = buffer_mems.len().saturating_sub(buffer_cap);
        if overflow > 0 && !evictable.is_empty() {
            // Oldest first (FIFO)
            evictable.sort_by_key(|m| m.created_at);
            let to_drop = overflow.min(evictable.len());
            for mem in &evictable[..to_drop] {
                if db.delete(&mem.id).unwrap_or(false) {
                    dropped_ids.push(mem.id.clone());
                    decayed += 1;
                }
            }
            info!(buffer_size = buffer_mems.len(), cap = buffer_cap,
                dropped = to_drop, "buffer over cap — evicted oldest");
        }
    }

    // Working gate-rejected TTL: memories that received final rejection
    // (gate-rejected-final) or were rejected and idle expire after 7 days.
    // gate-rejected / gate-rejected-2 still have retry chances, so only
    // expire them if not accessed recently.
    let gate_rejected_ttl = 7 * 86_400 * 1000; // 7 days in ms
    for mem in db.list_by_layer_meta(Layer::Working, 10000, 0).unwrap_or_else(|e| { warn!(error = %e, "list working for gate-rejected TTL failed"); vec![] }) {
        let is_any_rejected = mem.tags.iter().any(|t| t.starts_with("gate-rejected"));
        if !is_any_rejected { continue; }
        if promoted_ids.contains(&mem.id) || mem.kind == "procedural" { continue; }
        if mem.tags.iter().any(|t| t == "lesson") { continue; }
        let since_access = now - mem.last_accessed;
        if since_access > gate_rejected_ttl
            && db.delete(&mem.id).unwrap_or(false) {
                dropped_ids.push(mem.id.clone());
                decayed += 1;
                info!(id = %mem.id, content = %truncate_chars(&mem.content, 50),
                    days_since_access = since_access / 86_400_000,
                    "gate-rejected Working memory expired");
        }
    }

    // Drop decayed Buffer/Working entries — but skip anything we just promoted,
    // procedural memories, or lessons (they don't decay).
    for mem in db.get_decayed(decay_threshold).unwrap_or_else(|e| { warn!(error = %e, "get_decayed failed"); vec![] }) {
        let is_lesson = mem.tags.iter().any(|t| t == "lesson");
        if promoted_ids.contains(&mem.id) || mem.kind == "procedural" || is_lesson {
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
        decayed: decayed + gate_rejected_cleaned,
        demoted: demoted + auto_extract_demoted,
        merged: 0,
        importance_decayed,
        gate_rejected: 0,
        promoted_ids: promoted_ids.into_iter().collect(),
        dropped_ids,
        merged_ids: vec![],
        reconciled: 0,
        reconciled_ids: vec![],
        buffer_deduped: 0,
        triaged: 0,
        facts_extracted: 0,
        promotion_candidates,
        distilled: 0,
    }
}

/// Gate result: approved (with optional kind) or rejected.
#[derive(Debug, Deserialize)]
pub struct GateResult {
    pub decision: String,
    #[serde(default)]
    pub kind: Option<String>,
}

async fn llm_promotion_gate(cfg: &AiConfig, content: &str, access_count: i64, rep_count: i64) -> Result<(bool, Option<String>, Option<ai::Usage>, String, u64), EngramError> {
    let truncated = truncate_chars(content, 500);

    let mut context_parts = Vec::new();
    if access_count >= 30 {
        context_parts.push(format!("recalled {} times (high practical utility)", access_count));
    }
    if rep_count >= 2 {
        context_parts.push(format!("restated {} times (the user/agent keeps emphasizing this)", rep_count));
    }

    let user_msg = if context_parts.is_empty() {
        truncated.to_string()
    } else {
        format!("{}\n\n[Context: {}]", truncated, context_parts.join("; "))
    };

    let schema = prompts::gate_schema();

    let tcr: ai::ToolCallResult<GateResult> = ai::llm_tool_call(
        cfg, "gate", prompts::GATE_SYSTEM, &user_msg,
        "gate_decision", "Decide whether to promote this memory to Core",
        schema,
    ).await?;

    let approved = tcr.value.decision == "approve";
    let kind = if approved { tcr.value.kind.or(Some("semantic".into())) } else { None };
    Ok((approved, kind, tcr.usage, tcr.model, tcr.duration_ms))
}

pub fn layer_label(l: Layer) -> &'static str {
    match l { Layer::Core => "core", Layer::Working => "working", _ => "buffer" }
}

/// Merge tags from multiple sources, deduplicating and capping at `cap`.
pub fn merge_tags(base: &[String], others: &[&[String]], cap: usize) -> Vec<String> {
    let mut merged: Vec<String> = base.to_vec();
    for tags in others {
        for t in *tags {
            if !merged.contains(t) {
                merged.push(t.clone());
            }
        }
    }
    merged.truncate(cap);
    merged
}

pub fn format_ts(ms: i64) -> String {
    // Simple relative time format — no chrono dependency needed
    let age_secs = (crate::db::now_ms() - ms) / 1000;
    if age_secs < 3600 {
        format!("{}m ago", age_secs / 60)
    } else if age_secs < 86400 {
        format!("{}h ago", age_secs / 3600)
    } else {
        format!("{}d ago", age_secs / 86400)
    }
}

