//! Vector index and semantic search (HNSW-accelerated).

use rusqlite::params;
use std::collections::{HashMap, HashSet};

use hnsw_rs::prelude::*;

use super::*;
use crate::thresholds;

/// Sort scored results by similarity descending, then truncate to limit.
fn top_k(scored: &mut Vec<(String, f64)>, limit: usize) {
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit);
}

/// In-memory vector entry: embedding + namespace for filtering without DB lookup.
#[derive(Clone)]
pub struct VecEntry {
    pub emb: Vec<f32>,
    pub namespace: String,
}

fn hnsw_ef_search() -> usize {
    static VAL: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("ENGRAM_HNSW_EF_SEARCH")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(64)
    });
    *VAL
}

/// Combined vector index: HashMap for ID lookups + HNSW for ANN search.
pub struct VecIndex {
    entries: HashMap<String, VecEntry>,
    hnsw: Hnsw<'static, f32, DistCosine>,
    id_to_idx: HashMap<String, usize>,
    idx_to_id: Vec<String>,
}

impl VecIndex {
    pub(crate) fn new() -> Self {
        Self::with_capacity(thresholds::HNSW_INITIAL_CAPACITY)
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let cap = capacity.max(1000);
        Self {
            entries: HashMap::new(),
            hnsw: Hnsw::<f32, DistCosine>::new(
                thresholds::HNSW_MAX_NB_CONN,
                cap,
                thresholds::HNSW_MAX_LAYER,
                thresholds::HNSW_EF_CONSTRUCTION,
                DistCosine,
            ),
            id_to_idx: HashMap::new(),
            idx_to_id: Vec::new(),
        }
    }

    fn alloc_idx(&mut self, id: &str) -> usize {
        if let Some(&idx) = self.id_to_idx.get(id) {
            return idx;
        }
        let idx = self.idx_to_id.len();
        self.idx_to_id.push(id.to_string());
        self.id_to_idx.insert(id.to_string(), idx);
        idx
    }

    pub(crate) fn insert(&mut self, id: String, entry: VecEntry) {
        let idx = self.alloc_idx(&id);
        self.hnsw.insert((&entry.emb, idx));
        self.entries.insert(id, entry);
    }

    pub(crate) fn remove(&mut self, id: &str) {
        self.entries.remove(id);
        // HNSW doesn't support true deletion — the point remains in the graph
        // but won't be returned since we verify against `entries` post-search.
        // When ghost ratio exceeds 20%, rebuild to reclaim wasted traversal.
        let ratio = self.ghost_ratio();
        if ratio > 0.2 {
            let ghosts = self.idx_to_id.len() - self.entries.len();
            let total = self.idx_to_id.len();
            tracing::info!(ghosts, total, "rebuilding HNSW index");
            self.rebuild_hnsw();
        }
    }

    pub fn get(&self, id: &str) -> Option<&VecEntry> {
        self.entries.get(id)
    }

    pub fn contains_key(&self, id: &str) -> bool {
        self.entries.contains_key(id)
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&String, &VecEntry)> {
        self.entries.iter()
    }

    /// Search HNSW, return (id, similarity) pairs. Filters out removed entries.
    fn search_hnsw(&self, query: &[f32], k: usize) -> Vec<(String, f64)> {
        if self.entries.is_empty() {
            return vec![];
        }
        let neighbours = self.hnsw.search(query, k, hnsw_ef_search());
        neighbours
            .into_iter()
            .filter_map(|n| {
                let id = self.idx_to_id.get(n.d_id)?;
                // skip if deleted from entries
                if !self.entries.contains_key(id) {
                    return None;
                }
                // DistCosine returns 1 - cos_sim, so similarity = 1 - distance
                let sim = 1.0 - n.distance as f64;
                if sim > 0.0 {
                    Some((id.clone(), sim))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Rebuild the HNSW graph from scratch using current entries.
    fn rebuild_hnsw(&mut self) {
        let cap = self.entries.len().max(1000);
        self.hnsw = Hnsw::<f32, DistCosine>::new(
            thresholds::HNSW_MAX_NB_CONN,
            cap,
            thresholds::HNSW_MAX_LAYER,
            thresholds::HNSW_EF_CONSTRUCTION,
            DistCosine,
        );
        self.id_to_idx.clear();
        self.idx_to_id.clear();

        for (id, entry) in &self.entries {
            let idx = self.idx_to_id.len();
            self.idx_to_id.push(id.clone());
            self.id_to_idx.insert(id.clone(), idx);
            self.hnsw.insert((&entry.emb, idx));
        }
    }

    /// Ghost ratio: fraction of HNSW graph nodes that have been removed from entries.
    fn ghost_ratio(&self) -> f64 {
        let total = self.idx_to_id.len();
        if total == 0 { return 0.0; }
        (total - self.entries.len()) as f64 / total as f64
    }
}

impl MemoryDB {
    /// Load all embeddings from DB into the in-memory vector index.
    pub(super) fn load_vec_index(&self) {
        let Ok(conn) = self.conn() else { return };

        // Count rows first to size the HNSW graph appropriately
        let count: usize = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0) as usize;

        let Ok(mut stmt) = conn.prepare(
            "SELECT id, embedding, namespace FROM memories WHERE embedding IS NOT NULL",
        ) else {
            return;
        };

        let pairs: Vec<(String, VecEntry)> = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                let namespace: String = row.get(2)?;
                Ok((
                    id,
                    VecEntry {
                        emb: crate::ai::bytes_to_embedding(&blob),
                        namespace,
                    },
                ))
            })
            .map(|iter| {
                iter.filter_map(|r| r.map_err(|e| tracing::warn!("row parse: {e}")).ok())
                    .collect()
            })
            .unwrap_or_default();

        if let Ok(mut idx) = self.vec_index.write() {
            let capacity = count.checked_mul(2).unwrap_or(count).max(1000);
            *idx = VecIndex::with_capacity(capacity);
            let count = pairs.len();
            for (id, entry) in pairs {
                idx.insert(id, entry);
            }
            tracing::debug!(count, "loaded vector index");
        }
    }

    /// Add or update an embedding in the vector index.
    pub(super) fn vec_index_put(&self, id: &str, emb: Vec<f32>) {
        let namespace = self
            .get(id)
            .ok()
            .flatten()
            .map(|m| m.namespace)
            .unwrap_or_else(|| "default".into());
        if let Ok(mut idx) = self.vec_index.write() {
            idx.insert(id.to_string(), VecEntry { emb, namespace });
        }
    }

    /// Remove an embedding from the vector index.
    pub fn vec_index_remove(&self, id: &str) {
        if let Ok(mut idx) = self.vec_index.write() {
            idx.remove(id);
        }
    }

    /// Get embeddings for a set of IDs from the in-memory vec index.
    pub fn get_embeddings_by_ids(&self, ids: &[String]) -> Vec<(String, Vec<f32>)> {
        let idx = match self.vec_index.read() {
            Ok(idx) => idx,
            Err(_) => return vec![],
        };
        ids.iter()
            .filter_map(|id| idx.get(id).map(|e| (id.clone(), e.emb.clone())))
            .collect()
    }

    pub fn set_embedding(&self, id: &str, embedding: &[f32]) -> Result<(), EngramError> {
        let bytes = crate::ai::embedding_to_bytes(embedding);
        self.conn()?.execute(
            "UPDATE memories SET embedding = ?1 WHERE id = ?2",
            params![bytes, id],
        )?;
        self.vec_index_put(id, embedding.to_vec());
        Ok(())
    }

    pub fn get_all_with_embeddings(&self) -> Result<Vec<(Memory, Vec<f32>)>, EngramError> {
        let conn = self.conn()?;
        let mut stmt =
            conn.prepare("SELECT * FROM memories WHERE embedding IS NOT NULL")?;

        Ok(stmt
            .query_map([], |row| {
                let mem = row_to_memory_with_embedding(row)?;
                Ok(mem)
            })
            .map(|iter| {
                iter.filter_map(|r| r.map_err(|e| tracing::warn!("row parse: {e}")).ok())
                    .filter_map(|m| {
                        let emb = m.embedding.clone()?;
                        Some((m, emb))
                    })
                    .collect()
            })
            .unwrap_or_default())
    }

    /// Get all memories with embeddings, reading metadata from SQLite but
    /// embeddings from the in-memory vec_index (avoids reading large blobs
    /// from the DB, reducing memory spikes during consolidation).
    pub fn get_all_with_embeddings_from_index(&self) -> Vec<(Memory, Vec<f32>)> {
        let id_emb_pairs: Vec<(String, Vec<f32>)> = {
            let idx = match self.vec_index.read() {
                Ok(idx) => idx,
                Err(_) => return vec![],
            };
            if idx.len() == 0 {
                return vec![];
            }
            idx.iter().map(|(id, entry)| (id.clone(), entry.emb.clone())).collect()
        };

        let conn = match self.conn() {
            Ok(c) => c,
            Err(_) => return vec![],
        };

        // Build a lookup map from ID → embedding
        let id_to_emb: std::collections::HashMap<String, Vec<f32>> =
            id_emb_pairs.into_iter().collect();
        let ids: Vec<&str> = id_to_emb.keys().map(|s| s.as_str()).collect();

        // Fetch metadata (no embedding blob) in batches to avoid SQLite variable limits
        let mut result = Vec::with_capacity(ids.len());

        for chunk in ids.chunks(thresholds::HNSW_BATCH_SIZE) {
            let placeholders = chunk.iter().enumerate()
                .map(|(i, _)| format!("?{}", i + 1))
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!(
                "SELECT {} FROM memories WHERE id IN ({})",
                super::memory::META_COLS,
                placeholders
            );
            let mut stmt = match conn.prepare(&sql) {
                Ok(s) => s,
                Err(e) => { tracing::warn!("prepare batch meta: {e}"); continue; }
            };
            let params: Vec<rusqlite::types::Value> = chunk.iter()
                .map(|s| rusqlite::types::Value::Text(s.to_string()))
                .collect();
            let rows = stmt.query_map(rusqlite::params_from_iter(params.iter()), row_to_memory_meta);
            if let Ok(rows) = rows {
                for row in rows {
                    match row {
                        Ok(mem) => {
                            if let Some(emb) = id_to_emb.get(&mem.id) {
                                result.push((mem, emb.clone()));
                            }
                        }
                        Err(e) => tracing::warn!("row parse: {e}"),
                    }
                }
            }
        }

        result
    }

    /// Semantic search using HNSW index, with optional namespace filtering.
    pub fn search_semantic(&self, query_emb: &[f32], limit: usize) -> Vec<(String, f64)> {
        self.search_semantic_ns(query_emb, limit, None)
    }

    pub fn search_semantic_ns(
        &self,
        query_emb: &[f32],
        limit: usize,
        ns: Option<&str>,
    ) -> Vec<(String, f64)> {
        if let Ok(idx) = self.vec_index.read() {
            if idx.len() > 0 {
                if ns.is_none() {
                    // No namespace filter — direct HNSW search
                    let mut results = idx.search_hnsw(query_emb, limit);
                    results.sort_by(|a, b| {
                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    results.truncate(limit);
                    return results;
                }

                // With namespace filter: over-fetch from HNSW, then filter
                let fetch = limit * thresholds::NS_OVERFETCH;
                let candidates = idx.search_hnsw(query_emb, fetch);
                let mut scored: Vec<(String, f64)> = candidates
                    .into_iter()
                    .filter(|(id, _)| {
                        idx.get(id)
                            .map(|e| e.namespace == ns.unwrap())
                            .unwrap_or(false)
                    })
                    .collect();

                // If HNSW didn't return enough after filtering, fall back to brute-force
                // for this namespace (small namespaces in a big index)
                if scored.len() < limit {
                    scored = idx
                        .iter()
                        .filter(|(_, entry)| entry.namespace == ns.unwrap())
                        .map(|(id, entry)| {
                            let sim = crate::ai::cosine_similarity(query_emb, &entry.emb);
                            (id.clone(), sim)
                        })
                        .filter(|(_, sim)| *sim > 0.0)
                        .collect();
                }
                top_k(&mut scored, limit);
                return scored;
            }
        }

        // Fallback to DB scan (empty index)
        let all = self.get_all_with_embeddings().unwrap_or_default();
        let mut scored: Vec<(String, f64)> = all
            .into_iter()
            .filter(|(mem, _)| ns.is_none_or(|n| mem.namespace == n))
            .map(|(mem, emb)| {
                let sim = crate::ai::cosine_similarity(query_emb, &emb);
                (mem.id, sim)
            })
            .filter(|(_, sim)| *sim > 0.0)
            .collect();
        top_k(&mut scored, limit);
        scored
    }

    /// Semantic search restricted to a set of candidate IDs.
    /// Still uses brute-force over the given IDs (already O(|ids|), not O(n)).
    pub fn search_semantic_by_ids(
        &self,
        query_emb: &[f32],
        ids: &HashSet<String>,
        limit: usize,
    ) -> Vec<(String, f64)> {
        if let Ok(idx) = self.vec_index.read() {
            let mut scored: Vec<(String, f64)> = ids
                .iter()
                .filter_map(|id| {
                    idx.get(id).map(|entry| {
                        let sim = crate::ai::cosine_similarity(query_emb, &entry.emb);
                        (id.clone(), sim)
                    })
                })
                .filter(|(_, sim)| *sim > 0.0)
                .collect();
            top_k(&mut scored, limit);
            return scored;
        }
        vec![]
    }

    pub fn list_missing_embeddings(&self, limit: usize) -> Vec<(String, String)> {
        let conn = match self.conn() {
            Ok(c) => c,
            Err(_) => return vec![],
        };
        let mut stmt = match conn
            .prepare("SELECT id, content FROM memories WHERE embedding IS NULL LIMIT ?1")
        {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        stmt.query_map(params![limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map(|rows| {
            rows.filter_map(|r| r.map_err(|e| tracing::warn!("row parse: {e}")).ok())
                .collect()
        })
        .unwrap_or_default()
    }
}

