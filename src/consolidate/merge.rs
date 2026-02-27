use crate::ai::{self, AiConfig, cosine_similarity};
use crate::db::{Layer, Memory};
use crate::error::EngramError;
use crate::prompts;
use crate::SharedDB;
use crate::util::truncate_chars;
use serde::Deserialize;
use tracing::{debug, info, warn};

use super::{format_ts, layer_label, merge_tags};

use crate::thresholds;

/// Cosine threshold for near-duplicate detection and merging.
const MERGE_SIMILARITY: f64 = thresholds::MERGE_SIM;

/// Cosine window for reconcile: related-but-not-duplicate pairs.
const RECONCILE_MIN_SIM: f64 = thresholds::RECONCILE_MIN_SIM;
const RECONCILE_MAX_SIM: f64 = thresholds::RECONCILE_MAX_SIM;

/// Detect same-topic memories where a newer one supersedes an older one.
pub fn reconcile_pair_key(id_a: &str, id_b: &str) -> String {
    if id_a < id_b { format!("{}:{}", id_a, id_b) } else { format!("{}:{}", id_b, id_a) }
}

/// Runs every consolidation cycle. Uses a moderate similarity threshold
/// (RECONCILE_MIN_SIM..MERGE_SIMILARITY) to find topically related but not near-duplicate pairs.
/// Near-duplicates (>MERGE_SIMILARITY) are handled by merge_similar instead.
pub(super) async fn reconcile_updates(db: &SharedDB, cfg: &AiConfig) -> (usize, Vec<String>) {
    let now = crate::db::now_ms();

    // Load last_reconcile_ts and all memories with embeddings
    let db2 = db.clone();
    let (last_reconcile_ts, all): (i64, Vec<(Memory, Vec<f32>)>) = tokio::task::spawn_blocking(move || {
        let ts: i64 = db2.get_meta("last_reconcile_ts")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let mems = db2.get_all_with_embeddings_from_index();
        (ts, mems)
    })
    .await
    .unwrap_or_else(|e| {
        warn!(error = %e, "reconcile: load task failed");
        (0, vec![])
    });

    let wc: Vec<&(Memory, Vec<f32>)> = all
        .iter()
        .filter(|(m, e)| !e.is_empty() && (m.layer == Layer::Working || m.layer == Layer::Core))
        .collect();

    if wc.len() < 2 {
        return (0, vec![]);
    }

    // Determine which Working/Core memories are "new" (modified or created after last run).
    // On first run (last_reconcile_ts == 0), all memories are considered new.
    let new_wc_ids: std::collections::HashSet<&str> = wc.iter()
        .filter(|(m, _)| {
            let ts = m.modified_at.max(m.created_at);
            ts > last_reconcile_ts
        })
        .map(|(m, _)| m.id.as_str())
        .collect();

    if last_reconcile_ts > 0 && new_wc_ids.is_empty() {
        debug!("reconcile: no Working/Core changes since last run, skipping");
        return (0, vec![]);
    }

    let mut reconciled = 0;
    let mut removed_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Load cached keep_both decisions to avoid re-asking LLM
    let keep_both_cache: std::collections::HashSet<String> = {
        let db_kb = db.clone();
        tokio::task::spawn_blocking(move || {
            db_kb.get_meta_prefix("reconcile_kb:")
                .into_iter()
                .filter_map(|key| key.strip_prefix("reconcile_kb:").map(|s| s.to_string()))
                .collect()
        }).await.unwrap_or_default()
    };

    // Pass 1: new Working/Core vs ALL Working/Core (not all×all).
    // On first run new_wc_ids contains all, so this becomes the full scan.
    for i in 0..wc.len() {
        if removed_ids.contains(&wc[i].0.id) { continue; }
        // Only start a comparison chain when i is a "new" memory
        if !new_wc_ids.contains(wc[i].0.id.as_str()) { continue; }
        for j in (i + 1)..wc.len() {
            if removed_ids.contains(&wc[j].0.id) { continue; }
            if let Some(id) = try_reconcile_pair(
                db, cfg, &wc[i].0, &wc[i].1, &wc[j].0, &wc[j].1,
                None, &removed_ids, &keep_both_cache,
            ).await {
                removed_ids.insert(id);
                reconciled += 1;
            }
        }
    }

    // Pass 2: Buffers newer than last_reconcile_ts vs Working/Core.
    let buffers: Vec<&(Memory, Vec<f32>)> = all
        .iter()
        .filter(|(m, e)| {
            !e.is_empty()
                && m.layer == crate::db::Layer::Buffer
                && !removed_ids.contains(&m.id)
                && (last_reconcile_ts == 0 || m.created_at > last_reconcile_ts || m.modified_at > last_reconcile_ts)
        })
        .collect();

    for buf in &buffers {
        if removed_ids.contains(&buf.0.id) { continue; }
        for target in &wc {
            if removed_ids.contains(&target.0.id) { continue; }
            // Buffer must be newer than the target
            if buf.0.created_at <= target.0.created_at { continue; }
            if let Some(id) = try_reconcile_pair(
                db, cfg, &target.0, &target.1, &buf.0, &buf.1,
                Some(target.0.layer), &removed_ids, &keep_both_cache,
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

    // Save last_reconcile_ts = now
    let db_save = db.clone();
    let _ = tokio::task::spawn_blocking(move || {
        db_save.set_meta("last_reconcile_ts", &now.to_string())
    }).await;

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
    keep_both_cache: &std::collections::HashSet<String>,
) -> Option<String> {
    if already_removed.contains(&a.id) || already_removed.contains(&b.id) {
        return None;
    }

    let sim = cosine_similarity(a_emb, b_emb);
    if !(RECONCILE_MIN_SIM..=RECONCILE_MAX_SIM).contains(&sim) { return None; }

    let (older, newer) = if a.created_at <= b.created_at { (a, b) } else { (b, a) };

    let one_hour_ms: i64 = 3600 * 1000;
    if (newer.created_at - older.created_at) < one_hour_ms { return None; }

    // Never let a lower-layer memory absorb a higher-layer one.
    // e.g. Working(L2) must not supersede Core(L3).
    if newer.layer < older.layer {
        return None;
    }

    if older.namespace != newer.namespace {
        // Allow project→default merging: if one is "default", the merge result
        // stays in "default". Never pull default memories into a project namespace.
        let has_default = older.namespace == "default" || newer.namespace == "default";
        if !has_default {
            return None; // two different project namespaces, skip
        }
    }

    // Skip pairs we already judged as keep_both
    let pair_key = reconcile_pair_key(&older.id, &newer.id);
    if keep_both_cache.contains(&pair_key) {
        debug!(older = %older.id, newer = %newer.id, "reconcile: skip cached keep_both");
        return None;
    }

    let user_msg = format!(
        "OLDER (created {}, layer={}):\n{}\n\nNEWER (created {}, layer={}):\n{}",
        format_ts(older.created_at),
        layer_label(older.layer),
        truncate_chars(&older.content, 400),
        format_ts(newer.created_at),
        layer_label(newer.layer),
        truncate_chars(&newer.content, 400),
    );

    let schema = prompts::reconcile_schema();

    #[derive(Deserialize)]
    struct ReconcileDecision {
        decision: String,
        merged_content: Option<String>,
    }

    let result: ReconcileDecision = match ai::llm_tool_call(
        cfg, "merge", prompts::RECONCILE_PROMPT, &user_msg,
        "reconcile_decision", "Decide how to reconcile two memories and optionally merge them",
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
        // Persist keep_both so we never re-ask about this pair
        let pair_key = reconcile_pair_key(&older.id, &newer.id);
        let db_meta = db.clone();
        let _ = tokio::task::spawn_blocking(move || {
            db_meta.set_meta(&format!("reconcile_kb:{}", pair_key), "1")
        }).await;
        return None;
    }

    let new_tags = merge_tags(&newer.tags, &[&older.tags], 20);
    let imp = newer.importance.max(older.importance);
    let access = newer.access_count + older.access_count;

    // Use LLM-merged content; fall back to newer's content if missing
    let merged_content = result.merged_content
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| newer.content.clone());

    // Namespace merge direction: default is sticky. If the older (deleted) memory
    // is in "default" but newer (surviving) is in a project namespace, we must
    // move the survivor to "default" — never pull default into project scope.
    let needs_ns_fix = older.namespace == "default" && newer.namespace != "default";

    let newer_id = newer.id.clone();
    let older_id = older.id.clone();
    let db2 = db.clone();
    let result = tokio::task::spawn_blocking(move || {
        if let Some(layer) = promote_newer_to {
            db2.promote(&newer_id, layer)?;
        }
        db2.update_fields(&newer_id, Some(&merged_content), None, Some(imp), Some(&new_tags))?;
        db2.set_access_count(&newer_id, access)?;
        if needs_ns_fix {
            db2.set_namespace(&newer_id, "default")?;
        }
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
    let now = crate::db::now_ms();

    let db2 = db.clone();
    let (last_merge_ts, all): (i64, Vec<(Memory, Vec<f32>)>) = tokio::task::spawn_blocking(move || {
        let ts: i64 = db2.get_meta("last_merge_ts")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let mems = db2.get_all_with_embeddings_from_index();
        (ts, mems)
    })
    .await
    .unwrap_or_else(|e| {
        warn!(error = %e, "merge_similar: load task failed");
        (0, vec![])
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

        // Group by namespace. "default" memories join every project group
        // so cross-namespace dedup can happen (result stays in "default").
        let mut by_ns: std::collections::HashMap<&str, Vec<usize>> =
            std::collections::HashMap::new();
        let mut default_indices: Vec<usize> = Vec::new();
        for (i, (m, _)) in layer_mems.iter().enumerate() {
            if m.namespace == "default" {
                default_indices.push(i);
            }
            by_ns.entry(&m.namespace).or_default().push(i);
        }
        // Inject default memories into every project group for cross-ns dedup
        for (ns, indices) in by_ns.iter_mut() {
            if *ns != "default" && !default_indices.is_empty() {
                for &di in &default_indices {
                    if !indices.contains(&di) {
                        indices.push(di);
                    }
                }
            }
        }

        for ns_indices in by_ns.values() {
            if ns_indices.len() < 2 {
                continue;
            }
            let ns_mems: Vec<&(Memory, Vec<f32>)> =
                ns_indices.iter().map(|&i| layer_mems[i]).collect();

        // Incremental merge: only compare memories newer than last_merge_ts against all.
        // On first run (last_merge_ts == 0), all memories are treated as new (full scan).
        // text-embedding-3-small produces lower cosine scores for short CJK text,
        // but 0.68 was too aggressive — it merged related-but-distinct memories
        // (e.g. two v0.6.0 progress notes), destroying specific terms like "r2d2".
        // MERGE_SIMILARITY limits merging to near-duplicates with high content overlap.
        let new_indices: Vec<usize> = (0..ns_mems.len())
            .filter(|&i| {
                let m = &ns_mems[i].0;
                last_merge_ts == 0 || m.created_at > last_merge_ts || m.modified_at > last_merge_ts
            })
            .collect();

        if new_indices.is_empty() {
            continue;
        }

        // find_clusters_incremental: new × all (new mems seed clusters, old mems join them)
        let clusters = find_clusters_incremental(&ns_mems, &new_indices, MERGE_SIMILARITY);

        for cluster in clusters {
            if cluster.len() < 2 {
                continue;
            }

            let mut input = String::new();
            for (i, &idx) in cluster.iter().enumerate() {
                use std::fmt::Write;
                let _ = writeln!(input, "{}. {}", i + 1, ns_mems[idx].0.content);
            }

            let merged_content = match ai::llm_chat_as(cfg, "merge", prompts::MERGE_SYSTEM, &input).await {
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

            // If any cluster member is in "default" namespace, the merged result
            // must stay in "default" (project→default is allowed, not reverse).
            let cluster_has_default = cluster.iter()
                .any(|&i| ns_mems[i].0.namespace == "default");
            let winner_needs_ns_fix = cluster_has_default
                && ns_mems[best_idx].0.namespace != "default";

            {
                let db2 = db.clone();
                let id = best_id.clone();
                let content = merged_content.clone();
                let tags = all_tags;
                let imp = max_importance;
                let needs_ns = winner_needs_ns_fix;
                let result = tokio::task::spawn_blocking(move || {
                    db2.update_fields(&id, Some(&content), None, Some(imp), Some(&tags))?;
                    db2.set_access_count(&id, total_access)?;
                    if needs_ns {
                        db2.set_namespace(&id, "default")?;
                    }
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

    // Save last_merge_ts = now
    let db_save = db.clone();
    let _ = tokio::task::spawn_blocking(move || {
        db_save.set_meta("last_merge_ts", &now.to_string())
    }).await;

    (merged_total, merged_ids)
}

pub fn find_clusters(mems: &[&(Memory, Vec<f32>)], threshold: f64) -> Vec<Vec<usize>> {
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

/// Incremental cluster finder: only seeds from `new_indices`, but allows old memories
/// to join clusters seeded by new ones. Returns only clusters that contain at least one
/// new memory and at least one other memory.
///
/// This reduces the comparison from O(n²) to O(new × n) when most memories are unchanged.
fn find_clusters_incremental(
    mems: &[&(Memory, Vec<f32>)],
    new_indices: &[usize],
    threshold: f64,
) -> Vec<Vec<usize>> {
    let n = mems.len();
    let new_set: std::collections::HashSet<usize> = new_indices.iter().copied().collect();
    let mut used = vec![false; n];
    let mut clusters = Vec::new();

    // Only seed clusters from new memories
    for &i in new_indices {
        if used[i] {
            continue;
        }
        used[i] = true;
        let mut cluster = vec![i];

        // Compare against ALL memories (not just new ones)
        for j in 0..n {
            if j == i || used[j] {
                continue;
            }
            if cosine_similarity(&mems[i].1, &mems[j].1) > threshold {
                cluster.push(j);
                used[j] = true;
            }
        }

        // Only emit clusters that actually have 2+ members or include an old mem
        // (a new mem alone doesn't need merging)
        if cluster.len() >= 2 {
            clusters.push(cluster);
        }
    }

    // For completeness: if any non-new memory was clustered with a new one above,
    // it's already captured. We don't need to seed from old memories.
    // But: if an old memory was used as the seed in a previous full scan,
    // mark it as "used" so we don't double-count. (Already handled by used[] flag.)
    let _ = new_set; // used for documentation clarity

    clusters
}
