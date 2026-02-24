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

pub use audit::{audit_memories, AuditOp, AuditResult};
pub use sandbox::{sandbox_audit, SandboxResult, Grade, OpGrade};

use distill::distill_sessions;
use merge::{merge_similar, reconcile_updates};
// Re-imported for test access via `use super::*`
#[cfg(test)]
use merge::find_clusters;
use summary::update_core_summary;
use triage::{dedup_buffer, triage_buffer};

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
    pub(crate) promotion_candidates: Vec<(String, String, i64)>, // (id, content, access_count)
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
                for (id, content, access_count) in &candidates {
                    match llm_promotion_gate(cfg, content, *access_count).await {
                        Ok((true, kind)) => {
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
                        Ok((false, _)) => {
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
                let cands: Vec<String> = candidates.iter().map(|(id, _, _)| id.clone()).collect();
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
    for mem in db.list_by_layer_meta(Layer::Core, 10000, 0).unwrap_or_else(|e| { warn!(error = %e, "list_by_layer_meta(Core) failed"); vec![] }) {
        if (is_session(&mem) || mem.tags.iter().any(|t| t == "ephemeral"))
            && db.demote(&mem.id, Layer::Working).is_ok() {
                demoted += 1;
            }
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
            promotion_candidates.push((mem.id.clone(), mem.content.clone(), mem.access_count));
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
            promoted_ids.push(mem.id.clone());
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
                    promoted_ids.push(mem.id.clone());
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
        promoted_ids,
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
    - High-utility operational knowledge: if the context note says this memory has been recalled many times (15+), \
    it has proven its practical value regardless of category. Favor APPROVE for heavily-used memories \
    — the agent's actual behavior is a stronger signal than content classification.\n\
    \n\
    REJECT (keep in Working) if it is:\n\
    - Operational: bug fixes, code changes, version bumps, deployment logs (UNLESS heavily recalled)\n\
    - Ephemeral: session summaries, daily progress, temporary status\n\
    - Temporal snapshot: contains specific numbers, version strings, counts, percentages, or describes 'current state' \
    — these go stale and don't belong in permanent memory\n\
    - Technical detail: specific code patterns, API signatures, config values, tech stack specs\n\
    - System documentation: descriptions of how a system works, feature lists, implementation summaries, \
    TODO/roadmap lists — these belong in docs/README, not in Core memory. \
    Core is for WHY decisions were made, not WHAT was built.\n\
    - Duplicate: restates something that's obviously already known\n\
    - Research notes: survey results, market analysis, feature lists\n\
    \n\
    Reply with ONLY one word: APPROVE or REJECT\n\
    \n\
    If APPROVE, also specify the kind on the same line:\n\
    - APPROVE semantic  (facts, knowledge, preferences, lessons, identity)\n\
    - APPROVE procedural  (workflows, rules, how-to processes, behavioral directives)\n\
    - APPROVE episodic  (specific dated events, interactions with context)\n\
    \n\
    If unsure about kind, default to semantic.";

/// Gate result: approved (with optional kind) or rejected.
#[derive(Debug, Deserialize)]
struct GateResult {
    decision: String,
    #[serde(default)]
    kind: Option<String>,
}

async fn llm_promotion_gate(cfg: &AiConfig, content: &str, access_count: i64) -> Result<(bool, Option<String>), EngramError> {
    let truncated = truncate_chars(content, 500);

    // When a memory has significant usage history, tell the gate so it can
    // weigh empirical evidence, not just content analysis.
    let user_msg = if access_count >= 10 {
        format!(
            "{}\n\n[Context: This memory has been recalled {} times by the agent, \
             indicating high practical utility despite any prior rejections.]",
            truncated, access_count
        )
    } else {
        truncated.to_string()
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

    let result: GateResult = ai::llm_tool_call(
        cfg, "gate", GATE_SYSTEM, &user_msg,
        "gate_decision", "Decide whether to promote this memory to Core",
        schema,
    ).await?;

    let approved = result.decision == "approve";
    let kind = if approved { result.kind.or(Some("semantic".into())) } else { None };
    Ok((approved, kind))
}

pub(super) fn layer_label(l: Layer) -> &'static str {
    match l { Layer::Core => "core", Layer::Working => "working", _ => "buffer" }
}

/// Merge tags from multiple sources, deduplicating and capping at `cap`.
pub(super) fn merge_tags(base: &[String], others: &[&[String]], cap: usize) -> Vec<String> {
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

pub(super) fn format_ts(ms: i64) -> String {
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

#[cfg(test)]
#[path = "../consolidate_tests.rs"]
mod tests;
