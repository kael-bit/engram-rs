//! Vector index and semantic search.

use rusqlite::params;
use std::collections::HashSet;

use super::*;

/// Sort scored results by similarity descending, then truncate to limit.
fn top_k(scored: &mut Vec<(String, f64)>, limit: usize) {
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit);
}

/// In-memory vector entry: embedding + namespace for filtering without DB lookup.
#[derive(Clone)]
pub(crate) struct VecEntry {
    emb: Vec<f64>,
    namespace: String,
}

impl MemoryDB {
    /// Load all embeddings from DB into the in-memory vector index.
    pub(super) fn load_vec_index(&self) {
        let Ok(conn) = self.conn() else { return };
        let Ok(mut stmt) = conn.prepare(
            "SELECT id, embedding, namespace FROM memories WHERE embedding IS NOT NULL"
        ) else { return };

        let pairs: Vec<(String, VecEntry)> = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            let namespace: String = row.get(2)?;
            Ok((id, VecEntry { emb: crate::ai::bytes_to_embedding(&blob), namespace }))
        })
        .map(|iter| iter.filter_map(std::result::Result::ok).collect())
        .unwrap_or_default();

        if let Ok(mut idx) = self.vec_index.write() {
            idx.clear();
            let count = pairs.len();
            for (id, entry) in pairs {
                idx.insert(id, entry);
            }
            tracing::debug!(count, "loaded vector index");
        }
    }

    /// Add or update an embedding in the vector index.
    pub(super) fn vec_index_put(&self, id: &str, emb: Vec<f64>) {
        // Look up namespace from DB; fall back to "default".
        let namespace = self.get(id).ok().flatten()
            .map(|m| m.namespace)
            .unwrap_or_else(|| "default".into());
        if let Ok(mut idx) = self.vec_index.write() {
            idx.insert(id.to_string(), VecEntry { emb, namespace });
        }
    }

    /// Remove an embedding from the vector index.
    pub(super) fn vec_index_remove(&self, id: &str) {
        if let Ok(mut idx) = self.vec_index.write() {
            idx.remove(id);
        }
    }

    pub fn set_embedding(&self, id: &str, embedding: &[f64]) -> Result<(), EngramError> {
        let bytes = crate::ai::embedding_to_bytes(embedding);
        self.conn()?.execute(
            "UPDATE memories SET embedding = ?1 WHERE id = ?2",
            params![bytes, id],
        )?;
        self.vec_index_put(id, embedding.to_vec());
        Ok(())
    }

    pub fn get_all_with_embeddings(&self) -> Vec<(Memory, Vec<f64>)> {
        let Ok(conn) = self.conn() else { return vec![]; };
        let Ok(mut stmt) = conn
            .prepare("SELECT * FROM memories WHERE embedding IS NOT NULL")
        else {
            return vec![];
        };

        stmt.query_map([], |row| {
            let mem = row_to_memory_with_embedding(row)?;
            Ok(mem)
        })
        .map(|iter| {
            iter.filter_map(std::result::Result::ok)
                .filter_map(|m| {
                    let emb = m.embedding.clone()?;
                    Some((m, emb))
                })
                .collect()
        })
        .unwrap_or_default()
    }

    /// Semantic search: find memories closest to a query embedding.
    ///
    /// Uses brute-force cosine similarity. Suitable for collections up to ~10k memories.
    /// For larger datasets, consider an external vector index.
    /// Brute-force cosine similarity search. O(n) over all embeddings.
    /// Fine for <10k memories; for larger scales, consider IVF or HNSW indexing.
    pub fn search_semantic(&self, query_emb: &[f64], limit: usize) -> Vec<(String, f64)> {
        self.search_semantic_ns(query_emb, limit, None)
    }

    pub fn search_semantic_ns(
        &self, query_emb: &[f64], limit: usize, ns: Option<&str>,
    ) -> Vec<(String, f64)> {
        // Try in-memory index first (much faster — no SQL + no blob deser)
        if let Ok(idx) = self.vec_index.read() {
            if !idx.is_empty() {
                let mut scored: Vec<(String, f64)> = idx
                    .iter()
                    .filter(|(_, entry)| ns.is_none_or(|n| entry.namespace == n))
                    .map(|(id, entry)| {
                        let sim = crate::ai::cosine_similarity(query_emb, &entry.emb);
                        (id.clone(), sim)
                    })
                    .filter(|(_, sim)| *sim > 0.0)
                    .collect();

                top_k(&mut scored, limit);
                return scored;
            }
        }

        // Fallback to DB scan
        let all = self.get_all_with_embeddings();
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
    /// Only computes cosine similarity for the given IDs, skipping everything else.
    /// Used when FTS/facts already produced enough candidates to avoid a full scan.
    pub fn search_semantic_by_ids(
        &self, query_emb: &[f64], ids: &HashSet<String>, limit: usize,
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
        let mut stmt = match conn.prepare(
            "SELECT id, content FROM memories WHERE embedding IS NULL LIMIT ?1"
        ) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        stmt.query_map(params![limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map(|rows| rows.filter_map(std::result::Result::ok).collect())
        .unwrap_or_default()
    }

}

#[cfg(test)]
mod tests {
    
    use crate::db::{MemoryDB, MemoryInput};

    fn test_db() -> MemoryDB {
        MemoryDB::open(":memory:").unwrap()
    }

    fn dummy_emb(seed: f64) -> Vec<f64> {
        // 1536-dim dummy embedding with seed value
        let mut v = vec![0.0; 1536];
        v[0] = seed;
        v[1] = 1.0 - seed;
        v
    }

    #[test]
    fn vec_index_stores_namespace() {
        let db = test_db();
        let m = db.insert(MemoryInput {
            content: "test vec ns".into(),
            namespace: Some("custom-ns".into()),
            ..Default::default()
        }).unwrap();

        let emb = dummy_emb(0.8);
        db.set_embedding(&m.id, &emb).unwrap();

        // Index should store the namespace
        let idx = db.vec_index.read().unwrap();
        let entry = idx.get(&m.id).expect("must be in index");
        assert_eq!(entry.namespace, "custom-ns");
    }

    #[test]
    fn search_semantic_ns_filters_by_namespace() {
        let db = test_db();

        let m1 = db.insert(MemoryInput {
            content: "alpha".into(),
            namespace: Some("ns-a".into()),
            ..Default::default()
        }).unwrap();
        let m2 = db.insert(MemoryInput {
            content: "beta".into(),
            namespace: Some("ns-b".into()),
            ..Default::default()
        }).unwrap();

        // Same-ish embeddings so both would match
        db.set_embedding(&m1.id, &dummy_emb(0.9)).unwrap();
        db.set_embedding(&m2.id, &dummy_emb(0.85)).unwrap();

        let query = dummy_emb(0.9);

        // No namespace filter → both
        let all = db.search_semantic_ns(&query, 10, None);
        assert_eq!(all.len(), 2);

        // ns-a only
        let nsa = db.search_semantic_ns(&query, 10, Some("ns-a"));
        assert_eq!(nsa.len(), 1);
        assert_eq!(nsa[0].0, m1.id);

        // ns-b only
        let nsb = db.search_semantic_ns(&query, 10, Some("ns-b"));
        assert_eq!(nsb.len(), 1);
        assert_eq!(nsb[0].0, m2.id);
    }

    #[test]
    fn search_semantic_by_ids_restricts() {
        let db = test_db();
        let m1 = db.insert(MemoryInput {
            content: "first".into(),
            ..Default::default()
        }).unwrap();
        let m2 = db.insert(MemoryInput {
            content: "second".into(),
            ..Default::default()
        }).unwrap();

        db.set_embedding(&m1.id, &dummy_emb(0.7)).unwrap();
        db.set_embedding(&m2.id, &dummy_emb(0.75)).unwrap();

        let query = dummy_emb(0.7);
        let mut ids = std::collections::HashSet::new();
        ids.insert(m1.id.clone());

        let results = db.search_semantic_by_ids(&query, &ids, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, m1.id);
    }

    #[test]
    fn missing_embeddings_lists_correctly() {
        let db = test_db();
        db.insert(MemoryInput {
            content: "has embedding".into(),
            ..Default::default()
        }).unwrap();
        let _m2 = db.insert(MemoryInput {
            content: "no embedding".into(),
            ..Default::default()
        }).unwrap();

        // m2 has no embedding (only auto-embed when AI is configured)
        let missing = db.list_missing_embeddings(10);
        // Both should be missing since there's no AI in test
        assert!(missing.len() >= 1);
        assert!(missing.iter().any(|(_, c)| c.contains("no embedding")));
    }

    #[test]
    fn vec_index_remove_works() {
        let db = test_db();
        let m = db.insert(MemoryInput {
            content: "remove me".into(),
            ..Default::default()
        }).unwrap();
        db.set_embedding(&m.id, &dummy_emb(0.5)).unwrap();

        assert!(db.vec_index.read().unwrap().contains_key(&m.id));
        db.vec_index_remove(&m.id);
        assert!(!db.vec_index.read().unwrap().contains_key(&m.id));
    }
}
