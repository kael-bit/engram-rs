use crate::ai::{self, AiConfig, cosine_similarity};
use crate::db::{Layer, Memory, MemoryDB};
use crate::prompts;
use crate::SharedDB;
use crate::util::truncate_chars;
use serde::Deserialize;
use tracing::{debug, info, warn};

use super::merge_tags;

#[derive(Debug, Deserialize)]
struct TriageDecision {
    id: String,
    action: String,
    #[serde(default)]
    kind: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TriageResult {
    decisions: Vec<TriageDecision>,
}

/// Evaluate buffer memories using LLM and promote worthy ones to Working.
/// Only considers memories older than `min_age_ms` with at least one access.
/// Returns (promoted_count, promoted_ids).
pub(super) async fn triage_buffer(
    db: &SharedDB, cfg: &AiConfig,
    min_age_ms: i64, max_batch: usize,
    already_promoted: &[String],
) -> (usize, Vec<String>) {
    let now = crate::db::now_ms();

    let db2 = db.clone();
    let promoted_set: Vec<String> = already_promoted.to_vec();
    let all_buffers: Vec<Memory> = tokio::task::spawn_blocking(move || {
        db2.list_by_layer_meta(Layer::Buffer, 10000, 0)
            .unwrap_or_default()
            .into_iter()
            .filter(|m| {
                let dominated = m.tags.iter().any(|t| t == "distilled" || t == "ephemeral");
                let old_enough = (now - m.created_at) > min_age_ms;
                !dominated && old_enough && !promoted_set.contains(&m.id)
            })
            .collect()
    }).await.unwrap_or_default();

    // Process each namespace separately
    let mut ns_groups: std::collections::HashMap<&str, Vec<&Memory>> = std::collections::HashMap::new();
    for m in &all_buffers {
        ns_groups.entry(&m.namespace).or_default().push(m);
    }

    let mut total_promoted = 0;
    let mut all_ids = Vec::new();

    for (ns, mems) in &ns_groups {
        let candidates: Vec<_> = mems.iter().take(max_batch).collect();
        if candidates.is_empty() { continue; }

        let mut user_msg = String::with_capacity(candidates.len() * 200);
        for m in &candidates {
            let preview = truncate_chars(&m.content, 300);
            let tags = m.tags.join(", ");
            user_msg.push_str(&format!(
                "[{}] (ac={}, rep={}, tags=[{}]) {}\n\n",
                crate::util::short_id(&m.id), m.access_count, m.repetition_count, tags, preview
            ));
        }

        let schema = prompts::triage_schema();

        let result: TriageResult = match ai::llm_tool_call(
            cfg, "gate", prompts::TRIAGE_SYSTEM, &user_msg,
            "triage_decisions", "Decide which memories to promote or keep",
            schema,
        ).await {
            Ok(r) => {
                if let Some(ref u) = r.usage {
                    let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
                    let _ = db.log_llm_call("triage", &r.model, u.prompt_tokens, u.completion_tokens, cached, r.duration_ms);
                }
                r.value
            }
            Err(e) => {
                tracing::warn!(ns = %ns, error = %e, "buffer triage LLM call failed");
                continue;
            }
        };

        for d in &result.decisions {
            if d.action != "promote" { continue; }
            if let Some(mem) = candidates.iter().find(|m| m.id.starts_with(&d.id)) {
                let db3 = db.clone();
                let mid = mem.id.clone();
                let kind = d.kind.clone();
                let mk = mem.kind.clone();
                let promoted = tokio::task::spawn_blocking(move || {
                    if db3.promote(&mid, Layer::Working).is_ok() {
                        if let Some(ref k) = kind {
                            if k == "procedural" && mk == "semantic" {
                                let _ = db3.update_kind(&mid, k);
                            }
                        }
                        true
                    } else { false }
                }).await.unwrap_or(false);
                if promoted {
                    debug!(id = %mem.id, ns = %ns, kind = ?d.kind, "triage: promoted buffer → working");
                    all_ids.push(mem.id.clone());
                    total_promoted += 1;
                }
            }
        }

        // Mark KEEP decisions with _triaged tag so buffer TTL safety net
        // knows triage has evaluated them (prevents 48h extension).
        for d in &result.decisions {
            if d.action != "keep" { continue; }
            if let Some(mem) = candidates.iter().find(|m| m.id.starts_with(&d.id)) {
                if !mem.tags.iter().any(|t| t == "_triaged") {
                    let db4 = db.clone();
                    let mid = mem.id.clone();
                    let mut tags: Vec<String> = mem.tags.to_vec();
                    tags.push("_triaged".into());
                    let _ = tokio::task::spawn_blocking(move || {
                        let _ = db4.update_fields(&mid, None, None, None, Some(&tags));
                    }).await;
                }
            }
        }

        info!(ns = %ns, candidates = candidates.len(), promoted = total_promoted, "buffer triage batch complete");
    }

    info!(total = total_promoted, "buffer triage complete");
    (total_promoted, all_ids)
}

/// Merge near-duplicate buffer memories based on cosine similarity.
/// No LLM calls — keeps the newest entry, sums access counts, merges tags.
pub fn dedup_buffer(db: &MemoryDB) -> usize {
    let all = db.get_all_with_embeddings().unwrap_or_else(|e| { warn!(error = %e, "get_all_with_embeddings failed"); vec![] });
    let buffers: Vec<_> = all.iter()
        .filter(|(m, e)| !e.is_empty() && m.layer == Layer::Buffer)
        .collect();

    if buffers.len() < 2 {
        return 0;
    }

    let mut removed: Vec<String> = Vec::new();
    let threshold = crate::thresholds::TRIAGE_DEDUP_SIM;

    for i in 0..buffers.len() {
        if removed.contains(&buffers[i].0.id) { continue; }
        for j in (i + 1)..buffers.len() {
            if removed.contains(&buffers[j].0.id) { continue; }
            if buffers[i].0.namespace != buffers[j].0.namespace { continue; }

            let sim = cosine_similarity(&buffers[i].1, &buffers[j].1);
            if sim < threshold { continue; }

            // Keep the newer one
            let (discard, keep) = if buffers[i].0.created_at >= buffers[j].0.created_at {
                (&buffers[j].0, &buffers[i].0)
            } else {
                (&buffers[i].0, &buffers[j].0)
            };

            // Transfer access count and unique tags
            let total_access = keep.access_count + discard.access_count;
            let tags = merge_tags(&keep.tags, &[&discard.tags], 20);

            let imp = keep.importance.max(discard.importance);
            let _ = db.update_fields(&keep.id, None, None, Some(imp), Some(&tags));
            let _ = db.set_access_count(&keep.id, total_access);

            if db.delete(&discard.id).unwrap_or(false) {
                debug!(
                    keep = %keep.id, discard = %discard.id,
                    sim = format!("{:.3}", sim),
                    "buffer dedup: removed duplicate"
                );
                removed.push(discard.id.clone());
            }
        }
    }

    if !removed.is_empty() {
        info!(count = removed.len(), "buffer dedup complete");
    }
    removed.len()
}

