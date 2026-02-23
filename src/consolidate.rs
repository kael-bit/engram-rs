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
    pub merged: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub promoted_ids: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub dropped_ids: Vec<String>,
    /// IDs of memories that absorbed others during merge (winners).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub merged_ids: Vec<String>,
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
    .unwrap_or_default();

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
    let buffer_ttl = req.and_then(|r| r.buffer_ttl_secs).unwrap_or(3600) * 1000;
    let working_age = req.and_then(|r| r.working_age_promote_secs).unwrap_or(7 * 86400) * 1000;

    let now = crate::db::now_ms();
    let mut promoted = 0_usize;
    let mut decayed = 0_usize;
    let mut promoted_ids = Vec::new();
    let mut dropped_ids = Vec::new();

    // Working → Core: access-based or age-based (single pass)
    for mem in db.list_by_layer(Layer::Working, 10000, 0) {
        let dominated_by_access = mem.access_count >= promote_threshold
            && mem.importance >= promote_min_imp;
        let aged_in = (now - mem.created_at) > working_age && mem.access_count > 0;

        if (dominated_by_access || aged_in)
            && db.promote(&mem.id, Layer::Core).is_ok()
        {
            promoted_ids.push(mem.id.clone());
            promoted += 1;
        }
    }

    // Buffer → Working when accessed enough (reinforcement-based)
    let buffer_promote_count = promote_threshold.max(2);
    for mem in db.list_by_layer(Layer::Buffer, 10000, 0) {
        if mem.access_count >= buffer_promote_count
            && db.promote(&mem.id, Layer::Working).is_ok() {
                promoted_ids.push(mem.id.clone());
                promoted += 1;
            }
    }

    // Drop decayed Buffer/Working entries
    for mem in db.get_decayed(decay_threshold) {
        if db.delete(&mem.id).unwrap_or(false) {
            dropped_ids.push(mem.id.clone());
            decayed += 1;
        }
    }

    // Buffer TTL — old L1 entries that weren't accessed enough get dropped.
    // Accessed ones were already promoted above; stragglers with any access
    // get one more chance via Working, the rest expire.
    for mem in db.list_by_layer(Layer::Buffer, 10000, 0) {
        if promoted_ids.contains(&mem.id) {
            continue;
        }
        let age = now - mem.created_at;
        if age > buffer_ttl {
            if mem.access_count > 0 {
                // Accessed at least once but not enough to auto-promote;
                // give it a second life in Working layer
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

    if promoted > 0 || decayed > 0 {
        info!(promoted, decayed, "consolidation complete");
    } else {
        debug!("consolidation: nothing to do");
    }

    ConsolidateResponse {
        promoted,
        decayed,
        merged: 0,
        promoted_ids,
        dropped_ids,
        merged_ids: vec![],
    }
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
        .unwrap_or_default();

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

            let merged_content = match ai::llm_chat(cfg, MERGE_SYSTEM, &input).await {
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
            if merged_content.chars().count() >= total_input_len && total_input_len > 0 {
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
            if cfg.has_embed() {
                match ai::get_embeddings(cfg, &[merged_content]).await {
                    Ok(embs) if !embs.is_empty() => {
                        if let Some(emb) = embs.into_iter().next() {
                            let db2 = db.clone();
                            let id = best_id.clone();
                            let _ = tokio::task::spawn_blocking(move || {
                                db2.set_embedding(&id, &emb)
                            })
                            .await;
                        }
                    }
                    Err(e) => warn!(error = %e, "embedding for merged memory failed"),
                    _ => {}
                }
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
                decay_rate: 1.0,
                source: "test".into(),
                tags: vec![],
                namespace: "default".into(),
                embedding: None,
            },
            emb,
        )
    }

    #[test]
    fn cluster_similar_vectors() {
        let mems = vec![
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
        let mems = vec![
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
        let mems = vec![
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
            decay_rate: layer.default_decay(),
            source: "test".into(),
            tags: vec![],
            namespace: "default".into(),
            embedding: None,
        }
    }

    #[test]
    fn promote_high_access_working() {
        let db = test_db();
        let now = crate::db::now_ms();
        // working memory with enough accesses and importance → should promote
        let good = mem_with_ts("promote-me", Layer::Working, 0.8, 5, now - 1000, now);
        // working memory with low access → should stay
        let meh = mem_with_ts("leave-me", Layer::Working, 0.8, 1, now - 1000, now);
        db.import(&[good, meh]).unwrap();

        let result = consolidate_sync(&db, None);
        assert_eq!(result.promoted, 1);
        assert!(result.promoted_ids.contains(&"promote-me".to_string()));

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
        // old working memory with decent importance → should promote by age
        let old = mem_with_ts("old-but-worthy", Layer::Working, 0.6, 1, eight_days_ago, now);
        // fresh working memory → should stay
        let fresh = mem_with_ts("too-young", Layer::Working, 0.6, 1, now - 1000, now);
        db.import(&[old, fresh]).unwrap();

        let result = consolidate_sync(&db, None);
        assert!(result.promoted_ids.contains(&"old-but-worthy".to_string()));
        assert!(!result.promoted_ids.contains(&"too-young".to_string()));
    }

    #[test]
    fn drop_expired_low_importance_buffer() {
        let db = test_db();
        let now = crate::db::now_ms();
        let two_hours_ago = now - 7200_000;
        // old buffer, never accessed → should be dropped
        let expendable = mem_with_ts("bye", Layer::Buffer, 0.2, 0, two_hours_ago, two_hours_ago);
        // old buffer, accessed once → should be rescued to working
        let valuable = mem_with_ts("save-me", Layer::Buffer, 0.5, 1, two_hours_ago, two_hours_ago);
        db.import(&[expendable, valuable]).unwrap();

        let _result = consolidate_sync(&db, None);
        assert!(db.get("bye").unwrap().is_none(), "never-accessed buffer should be gone");
        let saved = db.get("save-me").unwrap();
        assert!(saved.is_some(), "accessed buffer should survive");
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
        // Buffer memory with enough accesses should promote to Working
        let accessed = mem_with_ts("recalled", Layer::Buffer, 0.1, 3, now - 1000, now);
        db.import(&[accessed]).unwrap();

        let r = consolidate_sync(&db, None);
        assert_eq!(r.promoted, 1);
        let got = db.get("recalled").unwrap().unwrap();
        assert_eq!(got.layer, Layer::Working);
    }

    #[test]
    fn buffer_ttl_accessed_once_promotes() {
        let db = test_db();
        let two_hours_ago = crate::db::now_ms() - 7200 * 1000;
        // Old buffer with 1 access but not enough for auto-promote;
        // TTL logic should still rescue it to Working
        let accessed = mem_with_ts("rescued", Layer::Buffer, 0.1, 1, two_hours_ago, two_hours_ago);
        db.import(&[accessed]).unwrap();

        let r = consolidate_sync(&db, None);
        assert!(r.promoted >= 1);
        let got = db.get("rescued").unwrap().unwrap();
        assert_eq!(got.layer, Layer::Working);
    }

    #[test]
    fn buffer_ttl_never_accessed_drops() {
        let db = test_db();
        let two_hours_ago = crate::db::now_ms() - 7200 * 1000;
        // Old buffer with 0 accesses — should be dropped
        let unused = mem_with_ts("forgotten", Layer::Buffer, 0.1, 0, two_hours_ago, two_hours_ago);
        db.import(&[unused]).unwrap();

        let r = consolidate_sync(&db, None);
        assert!(r.decayed >= 1);
        assert!(db.get("forgotten").unwrap().is_none());
    }
}
