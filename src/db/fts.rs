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
                // If there's CJK content, check if FTS has longer content (bigrams appended)
                let sample: Option<(i64, i64)> = self
                    .conn()?
                    .query_row(
                        "SELECT LENGTH(m.content), LENGTH(f.content) \
                         FROM memories m JOIN memories_fts f ON m.id = f.id \
                         WHERE m.content GLOB '*[一-龥]*' LIMIT 1",
                        [],
                        |r| Ok((r.get(0)?, r.get(1)?)),
                    )
                    .ok();

                match sample {
                    Some((mem_len, fts_len)) => fts_len <= mem_len, // no bigrams appended
                    None => true, // can't check, rebuild to be safe
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
            .filter_map(|r| r.ok())
            .collect();

        for (id, content, tags) in &rows {
            self.fts_insert(id, content, tags)?;
        }
        tracing::info!(count = rows.len(), "rebuilt FTS index with CJK segmentation");
        Ok(())
    }

    /// Full-text search using FTS5. Returns `(id, bm25_score)` pairs.
    pub fn search_fts(&self, query: &str, limit: usize) -> Vec<(String, f64)> {
        let sanitized: String = query
            .chars()
            .map(|c| if c.is_alphanumeric() || is_cjk(c) { c } else { ' ' })
            .collect();
        let sanitized = sanitized.trim();
        if sanitized.is_empty() {
            return vec![];
        }

        // Pre-process query: segment CJK so "天气" matches bigram-indexed content
        let processed = append_segmented(sanitized);
        let fts_query: String = processed.split_whitespace().collect::<Vec<_>>().join(" OR ");

        let Ok(conn) = self.conn() else { return vec![]; };
        let Ok(mut stmt) = conn.prepare(
            "SELECT id, rank FROM memories_fts \
             WHERE memories_fts MATCH ?1 ORDER BY rank LIMIT ?2",
        ) else {
            return vec![];
        };

        stmt.query_map(params![fts_query, limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })
        .map(|iter| iter.filter_map(|r| r.ok()).map(|(id, rank)| (id, -rank)).collect())
        .unwrap_or_default()
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
        })?.filter_map(|r| r.ok()).collect();

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
        })?.filter_map(|r| r.ok()).collect();

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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> MemoryDB {
        MemoryDB::open(":memory:").expect("in-memory db")
    }

    #[test]
    fn fts_search_finds_content() {
        let db = test_db();
        db.insert(MemoryInput {
            content: "the quick brown fox jumps".into(),
            layer: None,
            importance: None,
            ..Default::default()
        })
        .unwrap();

        let results = db.search_fts("quick fox", 10);
        assert!(!results.is_empty());
    }

    #[test]
    fn fts_cjk_bigram_search() {
        let db = test_db();
        db.insert(MemoryInput {
            content: "今天天气很好适合出门散步".into(),
            layer: None,
            importance: None,
            ..Default::default()
        })
        .unwrap();

        // Two-char CJK words should match via bigrams
        assert!(!db.search_fts("天气", 10).is_empty(), "天气 should match");
        assert!(!db.search_fts("散步", 10).is_empty(), "散步 should match");
        assert!(!db.search_fts("出门散步", 10).is_empty(), "出门散步 should match");
    }
}
