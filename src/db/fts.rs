//! FTS5 full-text search and vacuum operations.

use rusqlite::params;

use super::*;

impl MemoryDB {
    /// Insert a row into FTS with CJK segmentation.
    pub(super) fn fts_insert(&self, id: &str, content: &str, tags_json: &str) -> Result<(), EngramError> {
        let processed = append_segmented(content);
        self.conn()?.execute(
            "INSERT INTO memories_fts(id, content, tags) VALUES (?1, ?2, ?3)",
            params![id, processed, tags_json],
        )?;
        Ok(())
    }

    pub(super) fn fts_delete(&self, id: &str) -> Result<(), EngramError> {
        self.conn()?
            .execute("DELETE FROM memories_fts WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Rebuild FTS index from scratch (idempotent, runs on startup).
    pub(super) fn rebuild_fts(&self) -> Result<(), EngramError> {
        let fts_count: i64 = self
            .conn()?
            .query_row("SELECT COUNT(*) FROM memories_fts", [], |r| r.get(0))?;
        let mem_count: i64 = self
            .conn()?
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))?;

        if mem_count == 0 {
            return Ok(());
        }

        // Check if we need to rebuild: count mismatch or missing CJK segments
        let needs_rebuild = fts_count != mem_count || {
            // Sample a CJK-containing memory from the FTS table
            let has_cjk_content: bool = self
                .conn()?
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM memories WHERE content GLOB '*[一-龥]*' LIMIT 1)",
                    [],
                    |r| r.get(0),
                )
                .unwrap_or(false);

            if has_cjk_content {
                // Check if FTS has jieba segments appended (indicated by longer content).
                // Sample multiple CJK memories to avoid single-entry false positives.
                let samples: Vec<(i64, i64)> = self.conn()?.prepare(
                    "SELECT LENGTH(m.content), LENGTH(f.content) \
                     FROM memories m JOIN memories_fts f ON m.id = f.id \
                     WHERE m.content GLOB '*[一-龥]*' LIMIT 5"
                )?.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
                    .filter_map(|r| r.ok())
                    .collect();

                if samples.is_empty() {
                    true // can't check, rebuild to be safe
                } else {
                    // If ALL samples have fts_len <= mem_len, segments are missing
                    samples.iter().all(|(mem_len, fts_len)| *fts_len <= *mem_len)
                }
            } else {
                false // no CJK content, no rebuild needed
            }
        };

        if !needs_rebuild {
            return Ok(());
        }

        // Full rebuild
        self.conn()?.execute("DELETE FROM memories_fts", [])?;
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare("SELECT id, content, tags FROM memories")?;
        let rows: Vec<(String, String, String)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?
            .filter_map(|r| r.map_err(|e| tracing::warn!("row parse: {e}")).ok())
            .collect();

        for (id, content, tags) in &rows {
            self.fts_insert(id, content, tags)?;
        }
        tracing::info!(count = rows.len(), "rebuilt FTS index with CJK segmentation");
        Ok(())
    }

    /// Full-text search using FTS5. Returns `(id, bm25_score)` pairs.
    pub fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<(String, f64)>, EngramError> {
        self.search_fts_ns(query, limit, None)
    }

    /// FTS search with optional namespace filter (JOIN against memories table).
    pub fn search_fts_ns(&self, query: &str, limit: usize, ns: Option<&str>) -> Result<Vec<(String, f64)>, EngramError> {
        // Sanitize: keep alphanumeric + CJK, insert spaces at CJK/latin boundaries
        let mut sanitized = String::with_capacity(query.len() * 2);
        let mut prev_cjk = false;
        for c in query.chars() {
            if c.is_alphanumeric() || is_cjk(c) {
                let cur_cjk = is_cjk(c);
                if !sanitized.is_empty() && cur_cjk != prev_cjk {
                    sanitized.push(' ');
                }
                sanitized.push(c);
                prev_cjk = cur_cjk;
            } else {
                sanitized.push(' ');
                prev_cjk = false;
            }
        }
        let sanitized = sanitized.trim();
        if sanitized.is_empty() {
            return Ok(vec![]);
        }

        let processed = append_segmented(sanitized);
        let raw_terms: Vec<String> = processed.split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
        let terms = filter_fts_terms(&raw_terms, self);
        if terms.is_empty() {
            return Ok(vec![]);
        }
        let fts_query: String = terms.join(" OR ");

        let conn = self.conn()?;

        if let Some(ns) = ns {
            let mut stmt = conn.prepare(
                "SELECT f.id, f.rank FROM memories_fts f \
                 JOIN memories m ON m.id = f.id \
                 WHERE f.memories_fts MATCH ?1 AND m.namespace = ?3 \
                 ORDER BY f.rank LIMIT ?2",
            )?;
            Ok(stmt.query_map(params![fts_query, limit as i64, ns], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
            })
            .map(|iter| iter.filter_map(|r| r.map_err(|e| tracing::warn!("row parse: {e}")).ok()).map(|(id, rank)| (id, -rank)).collect())
            .unwrap_or_default())
        } else {
            let mut stmt = conn.prepare(
                "SELECT id, rank FROM memories_fts \
                 WHERE memories_fts MATCH ?1 ORDER BY rank LIMIT ?2",
            )?;
            Ok(stmt.query_map(params![fts_query, limit as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
            })
            .map(|iter| iter.filter_map(|r| r.map_err(|e| tracing::warn!("row parse: {e}")).ok()).map(|(id, rank)| (id, -rank)).collect())
            .unwrap_or_default())
        }
    }

    /// Count how many FTS documents contain a given term.
    /// Used by recall for IDF-based term weighting.
    pub fn term_doc_frequency(&self, term: &str) -> usize {
        let conn = match self.conn() {
            Ok(c) => c,
            Err(_) => return 0,
        };
        // FTS5 MATCH with a single term returns all docs containing it
        let count: i64 = conn.query_row(
            "SELECT count(*) FROM memories_fts WHERE memories_fts MATCH ?1",
            params![term],
            |row| row.get(0),
        ).unwrap_or(0);
        count.max(0) as usize
    }

    /// Auto-repair FTS index: remove orphans and rebuild missing entries.
    /// Returns (orphans_removed, missing_rebuilt).
    pub fn repair_fts(&self) -> Result<(usize, usize), EngramError> {
        let conn = self.conn()?;

        let orphans = conn.execute(
            "DELETE FROM memories_fts WHERE id NOT IN (SELECT id FROM memories)", []
        )?;

        let mut stmt = conn.prepare(
            "SELECT id, content, tags FROM memories WHERE id NOT IN (SELECT id FROM memories_fts)"
        )?;
        let missing: Vec<(String, String, String)> = stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?.filter_map(|r| r.map_err(|e| tracing::warn!("row parse: {e}")).ok()).collect();

        let rebuilt = missing.len();
        for (id, content, tags_json) in &missing {
            let processed = append_segmented(content);
            conn.execute(
                "INSERT INTO memories_fts(id, content, tags) VALUES (?1, ?2, ?3)",
                params![id, processed, tags_json],
            )?;
        }

        Ok((orphans, rebuilt))
    }

    /// Drop and rebuild the entire FTS index. Public API for force repair.
    pub fn force_rebuild_fts(&self) -> Result<(usize, usize), EngramError> {
        let conn = self.conn()?;

        let deleted = conn.execute("DELETE FROM memories_fts", [])?;

        let mut stmt = conn.prepare("SELECT id, content, tags FROM memories")?;
        let rows: Vec<(String, String, String)> = stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?.filter_map(|r| r.map_err(|e| tracing::warn!("row parse: {e}")).ok()).collect();

        let rebuilt = rows.len();
        for (id, content, tags_json) in &rows {
            let processed = append_segmented(content);
            conn.execute(
                "INSERT INTO memories_fts(id, content, tags) VALUES (?1, ?2, ?3)",
                params![id, processed, tags_json],
            )?;
        }

        tracing::info!(deleted, rebuilt, "full FTS rebuild");
        Ok((0, rebuilt))
    }

    /// Run incremental vacuum, returning bytes freed.
    pub fn vacuum_incremental(&self, pages: u32) -> Result<i64, EngramError> {
        let conn = self.conn()?;
        let before: i64 = conn.query_row(
            "SELECT page_count * page_size FROM pragma_page_count, pragma_page_size",
            [], |r| r.get(0),
        )?;
        conn.execute_batch(&format!("PRAGMA incremental_vacuum({pages})"))?;
        let after: i64 = conn.query_row(
            "SELECT page_count * page_size FROM pragma_page_count, pragma_page_size",
            [], |r| r.get(0),
        )?;
        Ok(before - after)
    }

    /// Full vacuum — reclaims all free pages. Blocks writes.
    pub fn vacuum_full(&self) -> Result<i64, EngramError> {
        let conn = self.conn()?;
        let before: i64 = conn.query_row(
            "SELECT page_count * page_size FROM pragma_page_count, pragma_page_size",
            [], |r| r.get(0),
        )?;
        conn.execute_batch("VACUUM")?;
        let after: i64 = conn.query_row(
            "SELECT page_count * page_size FROM pragma_page_count, pragma_page_size",
            [], |r| r.get(0),
        )?;
        Ok(before - after)
    }

}

/// Dynamic noise detection threshold: terms appearing in more than this
/// fraction of documents are considered noise (no discriminative value).
const NOISE_DF_RATIO: f64 = 0.8;

/// Extract meaningful query terms with dynamic noise filtering.
/// Terms appearing in >80% of corpus are auto-filtered — works for any
/// language without a hardcoded stopword list. The data decides what's noise.
/// Returns (terms, total_docs) so callers can reuse the doc count.
pub fn extract_query_terms(query: &str, db: &super::MemoryDB) -> (Vec<String>, f64) {
    let total_docs = db.count_filtered(None, None, None, None).unwrap_or(1).max(1) as f64;
    let words = super::jieba().cut_for_search(query, false);
    let mut seen = std::collections::HashSet::new();
    let terms = words.iter()
        .map(|w| w.trim().to_lowercase())
        .filter(|w| !w.is_empty())
        .filter(|w| seen.insert(w.clone()))
        .filter(|w| {
            if total_docs < 5.0 {
                return true; // Too few docs for meaningful df stats
            }
            let df = db.term_doc_frequency(w);
            (df as f64 / total_docs) <= NOISE_DF_RATIO
        })
        .collect();
    (terms, total_docs)
}

/// Filter FTS query terms: remove corpus-noise terms.
/// When corpus is too small (<5 docs), skip df filtering to avoid false negatives.
pub fn filter_fts_terms(terms: &[String], db: &super::MemoryDB) -> Vec<String> {
    let total_docs = db.count_filtered(None, None, None, None).unwrap_or(0) as f64;
    if total_docs < 5.0 {
        // Too few documents for meaningful df statistics
        return terms.iter().filter(|w| !w.is_empty()).cloned().collect();
    }
    terms.iter()
        .filter(|w| !w.is_empty())
        .filter(|w| {
            let df = db.term_doc_frequency(w);
            (df as f64 / total_docs) <= NOISE_DF_RATIO
        })
        .cloned()
        .collect()
}


