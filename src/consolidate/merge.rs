use crate::ai::{self, AiConfig, cosine_similarity};
use crate::db::{Layer, Memory};
use crate::error::EngramError;
use crate::SharedDB;
use crate::util::truncate_chars;
use serde::Deserialize;
use tracing::{debug, info, warn};

use super::{format_ts, layer_label, merge_tags};

const MERGE_SYSTEM: &str = "Merge these related memory entries into a single concise note. Rules:\n\
    - Preserve ALL specific names, tools, libraries, versions, and technical terms.\n\
    - If one entry updates or supersedes the other, keep the latest state.\n\
    - Remove only truly redundant/repeated sentences.\n\
    - Names, numbers, versions, dates, tool names > vague summaries. Never drop specific terms.\n\
    - Keep it under 400 characters if possible.\n\
    - Same language as originals. Output only the merged text, nothing else.";

const RECONCILE_PROMPT: &str = "You are comparing two memory entries about potentially the same topic.\n\
    The NEWER entry was created after the OLDER one.\n\n\
    Decide:\n\
    - update: The newer entry is an updated version of the same information. \
    The older one is now stale/outdated and should be removed.\n\
    - absorb: The newer entry contains all useful info from the older one plus more. \
    The older one is redundant.\n\
    - keep_both: They cover genuinely different aspects or the older one has \
    unique details not in the newer one.";

/// Detect same-topic memories where a newer one supersedes an older one.
/// Runs every consolidation cycle. Uses a moderate similarity threshold
/// (0.55-0.78) to find topically related but not near-duplicate pairs.
/// Near-duplicates (>0.78) are handled by merge_similar instead.
pub(super) async fn reconcile_updates(db: &SharedDB, cfg: &AiConfig) -> (usize, Vec<String>) {
    let db2 = db.clone();
    let all = tokio::task::spawn_blocking(move || db2.get_all_with_embeddings())
        .await
        .unwrap_or_else(|e| {
            warn!(error = %e, "reconcile: get_all_with_embeddings task failed");
            Ok(vec![])
        })
        .unwrap_or_else(|e| {
            warn!(error = %e, "reconcile: get_all_with_embeddings failed");
            vec![]
        });

    let wc: Vec<&(Memory, Vec<f32>)> = all
        .iter()
        .filter(|(m, e)| !e.is_empty() && (m.layer == Layer::Working || m.layer == Layer::Core))
        .collect();

    if wc.len() < 2 {
        return (0, vec![]);
    }

    // Skip if nothing changed since last reconcile
    let fingerprint = {
        let count = wc.len();
        let max_mod = wc.iter().map(|(_m, _)| _m.modified_at).max().unwrap_or(0);
        format!("{}:{}", count, max_mod)
    };
    {
        let db_fp = db.clone();
        let fp = fingerprint.clone();
        let last = tokio::task::spawn_blocking(move || db_fp.get_meta("reconcile_fingerprint"))
            .await.unwrap_or_default();
        if last.as_deref() == Some(fp.as_str()) {
            debug!("reconcile: no changes since last run, skipping");
            return (0, vec![]);
        }
    }

    let mut reconciled = 0;
    let mut removed_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Pass 1: Within Working/Core — newer supersedes older on same topic.
    for i in 0..wc.len() {
        if removed_ids.contains(&wc[i].0.id) { continue; }
        for j in (i + 1)..wc.len() {
            if removed_ids.contains(&wc[j].0.id) { continue; }
            if let Some(id) = try_reconcile_pair(
                db, cfg, &wc[i].0, &wc[i].1, &wc[j].0, &wc[j].1,
                None, &removed_ids,
            ).await {
                removed_ids.insert(id);
                reconciled += 1;
            }
        }
    }

    // Pass 2: Buffer -> Working/Core — buffer item updates a higher-layer memory.
    let buffers: Vec<&(Memory, Vec<f32>)> = all
        .iter()
        .filter(|(m, e)| !e.is_empty() && m.layer == Layer::Buffer && !removed_ids.contains(&m.id))
        .collect();

    for buf in &buffers {
        if removed_ids.contains(&buf.0.id) { continue; }
        for target in &wc {
            if removed_ids.contains(&target.0.id) { continue; }
            // Buffer must be newer than the target
            if buf.0.created_at <= target.0.created_at { continue; }
            if let Some(id) = try_reconcile_pair(
                db, cfg, &target.0, &target.1, &buf.0, &buf.1,
                Some(target.0.layer), &removed_ids,
            ).await {
                removed_ids.insert(id);
                reconciled += 1;
                break; // one buffer replaces at most one WC item
            }
        }
    }

    if reconciled > 0 {
        info!(reconciled, "reconciliation complete");
    }

    // Save fingerprint so we skip next time if nothing changed
    if !fingerprint.is_empty() {
        let db_save = db.clone();
        let _ = tokio::task::spawn_blocking(move || {
            db_save.set_meta("reconcile_fingerprint", &fingerprint)
        }).await;
    }

    (reconciled, removed_ids.into_iter().collect())
}

/// Try to reconcile a pair of memories. Returns the removed (older) memory's ID on success.
///
/// When `promote_newer_to` is Some, the newer memory gets promoted to that layer
/// (used when a buffer item supersedes a Working/Core item).
#[allow(clippy::too_many_arguments)]
async fn try_reconcile_pair(
    db: &SharedDB, cfg: &AiConfig,
    a: &Memory, a_emb: &[f32], b: &Memory, b_emb: &[f32],
    promote_newer_to: Option<Layer>,
    already_removed: &std::collections::HashSet<String>,
) -> Option<String> {
    if already_removed.contains(&a.id) || already_removed.contains(&b.id) {
        return None;
    }

    let sim = cosine_similarity(a_emb, b_emb);
    if !(0.55..=0.78).contains(&sim) { return None; }

    let (older, newer) = if a.created_at <= b.created_at { (a, b) } else { (b, a) };

    let one_hour_ms: i64 = 3600 * 1000;
    if (newer.created_at - older.created_at) < one_hour_ms { return None; }
    if older.namespace != newer.namespace { return None; }

    let user_msg = format!(
        "OLDER (created {}, layer={}):\n{}\n\nNEWER (created {}, layer={}):\n{}",
        format_ts(older.created_at),
        layer_label(older.layer),
        truncate_chars(&older.content, 400),
        format_ts(newer.created_at),
        layer_label(newer.layer),
        truncate_chars(&newer.content, 400),
    );

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["update", "absorb", "keep_both"],
                "description": "How to reconcile the two memories"
            }
        },
        "required": ["decision"]
    });

    #[derive(Deserialize)]
    struct ReconcileDecision { decision: String }

    let result: ReconcileDecision = match ai::llm_tool_call(
        cfg, "merge", RECONCILE_PROMPT, &user_msg,
        "reconcile_decision", "Decide how to reconcile two memories",
        schema,
    ).await {
        Ok(r) => {
            if let Some(ref u) = r.usage {
                let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
                let _ = db.log_llm_call("reconcile", &r.model, u.prompt_tokens, u.completion_tokens, cached, r.duration_ms);
            }
            r.value
        }
        Err(e) => { warn!(error = %e, "reconcile LLM call failed"); return None; }
    };

    let decision = result.decision;
    if decision != "update" && decision != "absorb" {
        debug!(older = %older.id, newer = %newer.id, "reconcile: keeping both");
        return None;
    }

    let new_tags = merge_tags(&newer.tags, &[&older.tags], 20);
    let imp = newer.importance.max(older.importance);
    let access = newer.access_count + older.access_count;

    let newer_id = newer.id.clone();
    let older_id = older.id.clone();
    let db2 = db.clone();
    let result = tokio::task::spawn_blocking(move || {
        if let Some(layer) = promote_newer_to {
            db2.promote(&newer_id, layer)?;
        }
        db2.update_fields(&newer_id, None, None, Some(imp), Some(&new_tags))?;
        db2.set_access_count(&newer_id, access)?;
        db2.delete(&older_id)?;
        Ok::<_, EngramError>(())
    }).await;

    match result {
        Ok(Ok(())) => {
            info!(
                newer = %newer.id, older = %older.id,
                sim = format!("{:.3}", sim), decision = %decision,
                "reconciled: newer supersedes older"
            );
            Some(older.id.clone())
        }
        Ok(Err(e)) => { warn!(error = %e, "reconcile update failed"); None }
        Err(e) => { warn!(error = %e, "reconcile task panicked"); None }
    }
}

pub(super) async fn merge_similar(db: &SharedDB, cfg: &AiConfig) -> (usize, Vec<String>) {
    let db2 = db.clone();
    let all = tokio::task::spawn_blocking(move || db2.get_all_with_embeddings())
        .await
        .unwrap_or_else(|e| {
            warn!(error = %e, "get_all_with_embeddings task failed");
            Ok(vec![])
        })
        .unwrap_or_else(|e| {
            warn!(error = %e, "get_all_with_embeddings failed");
            vec![]
        });

    if all.len() < 2 {
        return (0, vec![]);
    }

    let mut merged_total = 0;
    let mut merged_ids = Vec::new();

    for layer in [Layer::Buffer, Layer::Working, Layer::Core] {
        let layer_mems: Vec<&(Memory, Vec<f32>)> =
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
            let ns_mems: Vec<&(Memory, Vec<f32>)> =
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
                Ok(r) => {
                    if let Some(ref u) = r.usage {
                        let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
                        let _ = db.log_llm_call("merge", &r.model, u.prompt_tokens, u.completion_tokens, cached, r.duration_ms);
                    }
                    r.content.trim().to_string()
                }
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
                truncate_chars(&merged_content, 600)
            } else {
                merged_content
            };

            // Skip if merged output isn't shorter than total input — LLM failed to condense
            let total_input_len: usize = cluster.iter()
                .map(|&i| ns_mems[i].0.content.chars().count())
                .sum();
            if total_input_len == 0 || merged_content.chars().count() >= total_input_len {
                let preview: String = cluster.iter()
                    .map(|&i| truncate_chars(&ns_mems[i].0.content, 40))
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
            let tag_slices: Vec<&[String]> = cluster.iter()
                .map(|&idx| ns_mems[idx].0.tags.as_slice())
                .collect();
            let all_tags = merge_tags(&[], &tag_slices, 20);
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
                    Ok(er) if !er.embeddings.is_empty() => {
                        if let Some(ref u) = er.usage {
                            let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
                            let _ = db.log_llm_call("merge_embed", &cfg.embed_model, u.prompt_tokens, u.completion_tokens, cached, 0);
                        }
                        if let Some(emb) = er.embeddings.into_iter().next() {
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
                absorbed.push(truncate_chars(&loser.content, 60));
                let id = ns_mems[idx].0.id.clone();
                let db2 = db.clone();
                let _ = tokio::task::spawn_blocking(move || db2.delete(&id)).await;
            }

            let winner_preview: String = truncate_chars(&ns_mems[best_idx].0.content, 60);
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

pub(super) fn find_clusters(mems: &[&(Memory, Vec<f32>)], threshold: f64) -> Vec<Vec<usize>> {
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
