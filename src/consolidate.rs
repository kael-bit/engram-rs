use crate::ai::{self, AiConfig, cosine_similarity};
use crate::db::{Layer, Memory, MemoryDB};
use crate::{lock_db, SharedDB};
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
}

pub async fn consolidate(
    db: SharedDB,
    req: Option<ConsolidateRequest>,
    ai: Option<AiConfig>,
) -> ConsolidateResponse {
    let do_merge = req.as_ref().and_then(|r| r.merge).unwrap_or(false);

    let db2 = db.clone();
    let mut result = tokio::task::spawn_blocking(move || {
        consolidate_sync(&lock_db(&db2), req.as_ref())
    })
    .await
    .unwrap_or_default();

    if do_merge {
        if let Some(cfg) = ai {
            result.merged = merge_similar(&db, &cfg).await;
        }
    }

    result
}

fn consolidate_sync(db: &MemoryDB, req: Option<&ConsolidateRequest>) -> ConsolidateResponse {
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

    // Promote high-value Working → Core (access-based)
    for mem in db.list_by_layer(Layer::Working) {
        if mem.access_count >= promote_threshold && mem.importance >= promote_min_imp
            && db.promote(&mem.id, Layer::Core).is_ok() {
                promoted_ids.push(mem.id.clone());
                promoted += 1;
            }
    }

    // Age-based Working → Core promotion
    for mem in db.list_by_layer(Layer::Working) {
        if promoted_ids.contains(&mem.id) {
            continue;
        }
        let age = now - mem.created_at;
        if age > working_age && mem.importance >= 0.5
            && db.promote(&mem.id, Layer::Core).is_ok() {
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

    // Buffer TTL — old L1 entries promote or drop
    for mem in db.list_by_layer(Layer::Buffer) {
        let age = now - mem.created_at;
        if age > buffer_ttl {
            if mem.importance > 0.3 {
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
    }
}

const MERGE_SYSTEM: &str = "Merge these memories into one concise entry. \
    Keep all important information, remove redundancy. \
    Use the same language as the originals. Output only the merged text.";

async fn merge_similar(db: &SharedDB, cfg: &AiConfig) -> usize {
    let db2 = db.clone();
    let all = tokio::task::spawn_blocking(move || lock_db(&db2).get_all_with_embeddings())
        .await
        .unwrap_or_default();

    if all.len() < 2 {
        return 0;
    }

    let mut merged_total = 0;

    for layer in [Layer::Buffer, Layer::Working, Layer::Core] {
        let layer_mems: Vec<&(Memory, Vec<f64>)> =
            all.iter().filter(|(m, _)| m.layer == layer).collect();

        if layer_mems.len() < 2 {
            continue;
        }

        // text-embedding-3-small produces lower cosine scores for short CJK text,
        // so 0.75 was too aggressive. 0.68 catches real semantic duplicates without
        // merging unrelated memories (validated on actual CJK memory pairs).
        let clusters = find_clusters(&layer_mems, 0.68);

        for cluster in clusters {
            if cluster.len() < 2 {
                continue;
            }

            let mut input = String::new();
            for (i, &idx) in cluster.iter().enumerate() {
                use std::fmt::Write;
                let _ = writeln!(input, "{}. {}", i + 1, layer_mems[idx].0.content);
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

            // keep the entry with highest importance
            let best_idx = *cluster
                .iter()
                .max_by(|&&a, &&b| {
                    layer_mems[a]
                        .0
                        .importance
                        .partial_cmp(&layer_mems[b].0.importance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();

            // merge all tags
            let mut all_tags: Vec<String> = Vec::new();
            for &idx in &cluster {
                for tag in &layer_mems[idx].0.tags {
                    if !all_tags.contains(tag) {
                        all_tags.push(tag.clone());
                    }
                }
            }

            // update the winner
            let best_id = layer_mems[best_idx].0.id.clone();
            {
                let db2 = db.clone();
                let id = best_id.clone();
                let content = merged_content.clone();
                let tags = all_tags;
                let ok = tokio::task::spawn_blocking(move || {
                    lock_db(&db2).update_fields(&id, Some(&content), None, None, Some(&tags))
                })
                .await
                .ok()
                .and_then(|r| r.ok());

                if ok.is_none() {
                    warn!(id = %best_id, "failed to update merged memory");
                    continue;
                }
            }

            // regenerate embedding for merged content
            if cfg.has_embed() {
                match ai::get_embeddings(cfg, &[merged_content]).await {
                    Ok(embs) if !embs.is_empty() => {
                        let emb = embs.into_iter().next().unwrap();
                        let db2 = db.clone();
                        let id = best_id.clone();
                        let _ = tokio::task::spawn_blocking(move || {
                            lock_db(&db2).set_embedding(&id, &emb)
                        })
                        .await;
                    }
                    Err(e) => warn!(error = %e, "embedding for merged memory failed"),
                    _ => {}
                }
            }

            // delete the rest
            for &idx in &cluster {
                if idx == best_idx {
                    continue;
                }
                let id = layer_mems[idx].0.id.clone();
                let db2 = db.clone();
                let _ = tokio::task::spawn_blocking(move || lock_db(&db2).delete(&id)).await;
            }

            merged_total += 1;
        }
    }

    if merged_total > 0 {
        info!(merged = merged_total, "memory merge complete");
    }

    merged_total
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
}
