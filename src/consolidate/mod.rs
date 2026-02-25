use std::collections::HashSet;
use crate::ai::{self, AiConfig};
use crate::db::{Layer, Memory, MemoryDB};
use crate::error::EngramError;
use crate::util::truncate_chars;
use crate::SharedDB;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

mod audit;
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
use triage::triage_buffer;

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
    pub promotion_candidates: Vec<(String, String, i64, i64)>, // (id, content, access_count, repetition_count)
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
                for (id, content, access_count, rep_count) in &candidates {
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
                                    // Escalate: no tag → gate-rejected → gate-rejected-2 → gate-rejected-final
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
                let cands: Vec<String> = candidates.iter().map(|(id, _, _, _)| id.clone()).collect();
                let promoted_ids: Vec<String> = tokio::task::spawn_blocking(move || {
                    cands.into_iter().filter(|id| db2.promote(id, Layer::Core).is_ok()).collect::<Vec<_>>()
                }).await.unwrap_or_default();
                result.promoted += promoted_ids.len();
                result.promoted_ids.extend(promoted_ids);
            }
        }
    }

    // Reconcile: detect same-topic memories where newer one updates/supersedes older.
    // Runs every consolidation cycle (not just when merge is requested).
    if let Some(ref cfg) = ai {
        let (count, ids) = reconcile_updates(&db, cfg).await;
        result.reconciled = count;
        result.reconciled_ids = ids;
    }

    // Session distillation: turn accumulated session notes into project context.
    // Session notes can't promote to Core (by design), but the project knowledge
    // they contain is valuable. When 3+ session notes pile up, LLM distills them
    // into a single project-status memory that CAN promote normally.
    if let Some(ref cfg) = ai {
        result.distilled = distill_sessions(&db, cfg).await;
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

    // Buffer triage: LLM evaluates buffer memories that have usage signal.
    // Replaces mechanical access_count threshold with intelligent promotion.
    // Memories must be >1h old and have at least one access to be considered.
    if let Some(ref cfg) = ai {
        let min_age = 600 * 1000; // 10 minutes — gives time for dedup to run first
        let (count, ids) = triage_buffer(&db, cfg, min_age, 20, &result.promoted_ids).await;
        result.triaged = count;
        for id in ids {
            result.promoted_ids.push(id);
        }
        result.promoted += count;
    }

    // Extract fact triples from Working/Core memories that don't have any yet.
    // Facts auto-extraction disabled — the extracted triples were low quality
    // and duplicated information already in memories. Keep the API for manual use.
    // if let Some(ref cfg) = ai {
    //     result.facts_extracted = extract_facts_batch(&db, cfg, 5).await;
    // }

    if do_merge {
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
    if let Some(ref cfg) = ai {
        update_core_summary(&db, cfg).await;
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

pub fn consolidate_sync(db: &MemoryDB, req: Option<&ConsolidateRequest>) -> ConsolidateResponse {
    cleanup_stale_tags(db);
    fix_stuck_buffer(db);

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
    // O(n²) pairwise scan — fine for <50 Core memories.
    {
        let core_mems: Vec<Memory> = db.list_by_layer_meta(Layer::Core, 10000, 0)
            .unwrap_or_default();
        let one_hour_ago = now - 3_600_000;
        let core_ids: Vec<String> = core_mems.iter()
            .filter(|m| m.created_at < one_hour_ago)
            .map(|m| m.id.clone())
            .collect();
        let embeddings = db.get_embeddings_by_ids(&core_ids);

        let mut candidates: Vec<(String, String, f64)> = Vec::new();
        for i in 0..embeddings.len() {
            for j in (i + 1)..embeddings.len() {
                let sim = crate::ai::cosine_similarity(&embeddings[i].1, &embeddings[j].1);
                if sim > 0.70 {
                    let (id_a, id_b) = if embeddings[i].0 < embeddings[j].0 {
                        (&embeddings[i].0, &embeddings[j].0)
                    } else {
                        (&embeddings[j].0, &embeddings[i].0)
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
        info!(pairs = candidates.len(), "core overlap scan complete");
    }

    // Working -> Core: collect candidates for LLM gate review.
    // Reinforcement score = access_count + repetition_count * 2.5
    // Repetition weighs 2.5x because restating > incidental recall hit.
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
        let score = mem.access_count as f64 + mem.repetition_count as f64 * 2.5;
        let by_score = score >= promote_threshold as f64
            && mem.importance >= promote_min_imp;
        let by_age = (now - mem.created_at) > working_age
            && score > 0.0
            && mem.importance >= promote_min_imp;

        if by_score || by_age {
            promotion_candidates.push((mem.id.clone(), mem.content.clone(), mem.access_count, mem.repetition_count));
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

        let score = mem.access_count as f64 + mem.repetition_count as f64 * 2.5;
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
            let rescue_score = mem.access_count as f64 + mem.repetition_count as f64 * 2.5;
            // Rescue if: enough access/repetition, OR importance is high enough
            // that the content itself is valuable even without being recalled.
            // This saves design decisions and architecture notes from silent death.
            let worth_keeping = rescue_score >= buffer_threshold / 2.0
                || mem.importance >= 0.7;
            if worth_keeping {
                if db.promote(&mem.id, Layer::Working).is_ok() {
                    promoted_ids.insert(mem.id.clone());
                    promoted += 1;
                }
            } else if db.delete(&mem.id).unwrap_or(false) {
                dropped_ids.push(mem.id.clone());
                decayed += 1;
            }
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

    // Working layer capacity cap — evict least-useful memories when over limit.
    // Time-based decay doesn't work at vibecoding speed (3 days of decay threshold
    // when the entire project was built in 3 days). Instead, cap Working at a fixed
    // size and evict by utility: lowest access_count first, then oldest.
    // Lessons and procedural memories are exempt from eviction.
    let working_cap: usize = std::env::var("ENGRAM_WORKING_CAP")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(30);
    let mut working_mems: Vec<Memory> = db.list_by_layer_meta(Layer::Working, 10000, 0)
        .unwrap_or_else(|e| { warn!(error = %e, "list working for cap check failed"); vec![] });

    // Partition: exempt (lesson/procedural) vs evictable
    let (exempt, mut evictable): (Vec<_>, Vec<_>) = working_mems.drain(..).partition(|m| {
        m.kind == "procedural"
            || m.tags.iter().any(|t| t == "lesson")
            || promoted_ids.contains(&m.id)
    });

    if exempt.len() + evictable.len() > working_cap && evictable.len() > 0 {
        let target = working_cap.saturating_sub(exempt.len());
        if evictable.len() > target {
            // Sort: lowest access_count first, then oldest first
            evictable.sort_by(|a, b| {
                a.access_count.cmp(&b.access_count)
                    .then(a.last_accessed.cmp(&b.last_accessed))
            });
            let to_demote = evictable.len() - target;
            for mem in &evictable[..to_demote] {
                if db.demote(&mem.id, Layer::Buffer).is_ok() {
                    info!(id = %mem.id, content = %truncate_chars(&mem.content, 50),
                        access_count = mem.access_count,
                        "Working over cap — demoted to Buffer");
                    decayed += 1;
                }
            }
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
        decayed,
        demoted,
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

const GATE_SYSTEM: &str = "\
Core is PERMANENT memory that survives total context loss. The litmus test: \
if the agent wakes up with zero context, would this memory alone be useful? \
If it only makes sense alongside the code/docs/conversation it came from, REJECT.\n\
\n\
APPROVE:\n\
- \"never force-push to main\" (lesson — prevents repeating a mistake)\n\
- \"user prefers Chinese, hates verbose explanations\" (identity/preference — shapes behavior)\n\
- \"all public output must hide AI identity\" (constraint — hard rule that never changes)\n\
- \"we chose SQLite over Postgres for zero-dep deployment\" (decision rationale — the WHY)\n\
\n\
REJECT:\n\
- \"Recall quality: 1) bilingual expansion 2) FTS gating 3) gate evidence\" (changelog — lists WHAT was done)\n\
- \"Fixed bug: triage didn't filter by namespace\" (operational — belongs in git history)\n\
- \"Session: refactored auth, deployed v0.7, 143 tests pass\" (session log — ephemeral status)\n\
- \"Improvement plan: add compression, fix lifecycle, batch ops\" (plan — not a lesson)\n\
- \"cosine threshold 0.78, buffer TTL 24h, promote at 5 accesses\" (config values — will go stale)\n\
- \"HNSW index replaces brute-force search\" (implementation detail — in the code already)\n\
\n\
The pattern: APPROVE captures WHY and NEVER and WHO. REJECT captures WHAT and WHEN and HOW MUCH.\n\
Numbered lists of changes (1) did X 2) did Y 3) did Z) are almost always changelogs → REJECT.\n\
If it reads like a commit message or progress report, REJECT.";

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

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["approve", "reject"],
                "description": "Whether to promote to Core"
            },
            "kind": {
                "type": "string",
                "enum": ["semantic", "procedural", "episodic"],
                "description": "Memory kind (only when approving)"
            }
        },
        "required": ["decision"]
    });

    let tcr: ai::ToolCallResult<GateResult> = ai::llm_tool_call(
        cfg, "gate", GATE_SYSTEM, &user_msg,
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

