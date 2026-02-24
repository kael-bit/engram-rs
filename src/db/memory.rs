//! Memory CRUD operations.

use rusqlite::params;
use uuid::Uuid;

/// Column list excluding the embedding blob. Used in "meta" queries to avoid
/// deserializing large vectors when only scalar fields are needed.
const META_COLS: &str = "id, content, layer, importance, created_at, last_accessed, \
    access_count, repetition_count, decay_rate, source, tags, namespace, kind";

use super::*;

impl MemoryDB {
    pub fn insert(&self, input: MemoryInput) -> Result<Memory, EngramError> {
        validate_input(&input)?;

        // Near-duplicate detection: if an existing memory has very similar content,
        // update it instead of creating a duplicate. Uses token overlap as a proxy.
        // Crucially, this counts as a repetition — touch the memory to reinforce it.
        // Spaced repetition: repeated exposure strengthens memory, not just retrieval.
        let do_dedup = !input.skip_dedup.unwrap_or(false);
        if do_dedup {
        let ns = input.namespace.as_deref().unwrap_or("default");
        if let Some(existing) = self.find_near_duplicate(&input.content, ns) {
            tracing::debug!(existing_id = %existing.id, "near-duplicate found, reinforcing");
            // Repetition = reinforcement. This is the core insight:
            // if someone keeps writing similar content, they clearly care about it.
            let _ = self.reinforce(&existing.id);
            let tags = input.tags.unwrap_or_default();
            // Merge tags from both
            let mut merged_tags: Vec<String> = existing.tags.clone();
            for t in &tags {
                if !merged_tags.contains(t) {
                    merged_tags.push(t.clone());
                }
            }
            // reinforce() already bumped importance by 0.05; use post-reinforce value
            let reinforced_imp = (existing.importance + 0.05).min(1.0);
            let imp = input
                .importance
                .map(|new_imp| new_imp.max(reinforced_imp))
                .unwrap_or(reinforced_imp);
            // Keep the higher layer
            let layer = input
                .layer
                .map(|new_l| new_l.max(existing.layer as u8))
                .unwrap_or(existing.layer as u8);

            return self
                .update_fields(
                    &existing.id,
                    Some(&input.content),
                    Some(layer),
                    Some(imp),
                    Some(&merged_tags),
                )
                .and_then(|opt| {
                    // handle supersedes even in dedup path
                    if let Some(ref old_ids) = input.supersedes {
                        for old_id in old_ids {
                            if old_id != &existing.id {
                                if let Err(e) = self.delete(old_id) {
                                    tracing::warn!(old_id, error = %e, "supersede delete failed");
                                }
                            }
                        }
                    }
                    opt.ok_or(EngramError::Internal("update after dedup failed".into()))
                });
        }
        } // do_dedup

        let now = now_ms();
        let importance = input.importance.unwrap_or(0.5).clamp(0.0, 1.0);

        // All memories start in Buffer. Layer promotion is earned through
        // access frequency, not declared at insert time. Caller can still
        // set an explicit layer for admin/migration use, but importance
        // alone doesn't bypass the promotion path.
        let layer_val = input.layer.unwrap_or(1);
        let layer: Layer = layer_val.try_into()?;
        let id = Uuid::new_v4().to_string();
        let source = input.source.unwrap_or_else(|| "api".into());
        let mut tags = input.tags.unwrap_or_default();

        let namespace = input.namespace.unwrap_or_else(|| "default".into());
        let kind = input.kind.unwrap_or_else(|| "semantic".into());

        let decay = if kind == "procedural" { 0.01 } else { layer.default_decay() };

        let risk_score = crate::safety::assess_injection_risk(&input.content);
        if risk_score >= 0.7 && !tags.contains(&"suspicious".to_string()) {
            tags.push("suspicious".into());
        }
        let tags_json = serde_json::to_string(&tags).unwrap_or_else(|_| "[]".into());

        self.conn()?.execute(
            "INSERT INTO memories \
             (id, content, layer, importance, created_at, last_accessed, \
              access_count, decay_rate, source, tags, namespace, risk_score, kind) \
             VALUES (?1,?2,?3,?4,?5,?6,0,?7,?8,?9,?10,?11,?12)",
            params![
                id,
                input.content,
                layer_val,
                importance,
                now,
                now,
                decay,
                source,
                tags_json,
                namespace,
                risk_score,
                kind,
            ],
        )?;

        self.fts_insert(&id, &input.content, &tags_json)?;

        // supersede: delete old memories that this one replaces
        if let Some(ref old_ids) = input.supersedes {
            for old_id in old_ids {
                if let Err(e) = self.delete(old_id) {
                    tracing::warn!(old_id, error = %e, "failed to delete superseded memory");
                }
            }
        }

        Ok(Memory {
            id,
            content: input.content,
            layer,
            importance,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            repetition_count: 0,
            decay_rate: decay,
            source,
            tags,
            namespace,
            embedding: None,
            risk_score,
            kind,
        })
    }

    /// Batch insert within a single transaction. Skips dedup for speed.
    /// Returns the successfully inserted memories.
    pub fn insert_batch(&self, inputs: Vec<MemoryInput>) -> Result<Vec<Memory>, EngramError> {
        let conn = self.conn()?;
        conn.execute_batch("BEGIN")?;
        let mut results = Vec::with_capacity(inputs.len());
        let result = (|| -> Result<(), EngramError> {
            for input in inputs {
                if let Err(e) = validate_input(&input) {
                    tracing::warn!(error = %e, "batch: skipping invalid input");
                    continue;
                }
                let _importance = input.importance.unwrap_or(0.5).clamp(0.0, 1.0);
                let now = now_ms();
                let layer_val = input.layer.unwrap_or(1);
                let layer: Layer = match layer_val.try_into() {
                    Ok(l) => l,
                    Err(_) => continue,
                };
                let id = Uuid::new_v4().to_string();
                let importance = input.importance.unwrap_or(0.5).clamp(0.0, 1.0);
                let source = input.source.unwrap_or_else(|| "api".into());
                let mut tags = input.tags.unwrap_or_default();
                let namespace = input.namespace.unwrap_or_else(|| "default".into());
                let kind = input.kind.unwrap_or_else(|| "semantic".into());

                let decay = if kind == "procedural" { 0.01 } else { layer.default_decay() };

                let risk_score = crate::safety::assess_injection_risk(&input.content);
                if risk_score >= 0.7 && !tags.contains(&"suspicious".to_string()) {
                    tags.push("suspicious".into());
                }
                let tags_json = serde_json::to_string(&tags).unwrap_or_else(|_| "[]".into());

                conn.execute(
                    "INSERT INTO memories \
                     (id, content, layer, importance, created_at, last_accessed, \
                      access_count, decay_rate, source, tags, namespace, risk_score, kind) \
                     VALUES (?1,?2,?3,?4,?5,?6,0,?7,?8,?9,?10,?11,?12)",
                    params![
                        id, input.content, layer_val, importance, now, now,
                        decay, source, tags_json, namespace, risk_score, kind
                    ],
                )?;
                let processed = append_segmented(&input.content);
                conn.execute(
                    "INSERT INTO memories_fts(id, content, tags) VALUES (?1, ?2, ?3)",
                    params![id, processed, tags_json],
                )?;

                results.push(Memory {
                    id,
                    content: input.content,
                    layer,
                    importance,
                    created_at: now,
                    last_accessed: now,
                    access_count: 0,
                    repetition_count: 0,
                    decay_rate: decay,
                    source,
                    tags,
                    namespace,
                    embedding: None,
                    risk_score,
                    kind,
                });
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                conn.execute_batch("COMMIT")?;
                Ok(results)
            }
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK");
                Err(e)
            }
        }
    }

    pub fn get(&self, id: &str) -> Result<Option<Memory>, EngramError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare("SELECT * FROM memories WHERE id = ?1")?;
        let mut rows = stmt.query(params![id])?;
        match rows.next()? {
            Some(row) => Ok(Some(row_to_memory(row)?)),
            None => Ok(None),
        }
    }

    /// Resolve a short ID prefix to a full UUID.
    /// If the input is already a full UUID (36+ chars), returns it as-is.
    /// Returns NotFound if no match, Validation error if ambiguous.
    pub fn resolve_prefix(&self, prefix: &str) -> Result<String, EngramError> {
        if prefix.len() >= 36 {
            return Ok(prefix.to_string());
        }
        let conn = self.conn()?;
        let pattern = format!("{}%", prefix);
        let mut stmt = conn.prepare("SELECT id FROM memories WHERE id LIKE ?1 LIMIT 2")?;
        let ids: Vec<String> = stmt.query_map(params![pattern], |row| row.get(0))?
            .filter_map(|r| r.map_err(|e| tracing::warn!("row parse: {e}")).ok())
            .collect();
        match ids.len() {
            0 => Err(EngramError::NotFound),
            1 => Ok(ids.into_iter().next().unwrap()),
            _ => Err(EngramError::Validation(format!("prefix '{}' matches multiple memories", prefix))),
        }
    }

    pub fn delete(&self, id: &str) -> Result<bool, EngramError> {
        let conn = self.conn()?;
        // Copy to trash before deleting
        let moved = conn.execute(
            "INSERT OR REPLACE INTO trash (id, content, layer, importance, created_at, deleted_at, tags, namespace, source, kind)
             SELECT id, content, layer, importance, created_at, ?2, tags, namespace, source, kind
             FROM memories WHERE id = ?1",
            params![id, now_ms()],
        )?;
        self.delete_facts_by_memory(id)?;
        let n = conn.execute("DELETE FROM memories WHERE id = ?1", params![id])?;
        if n > 0 {
            self.fts_delete(id)?;
            self.vec_index_remove(id);
        }
        Ok(n > 0 || moved > 0)
    }

    /// Delete all memories in a namespace. Returns how many were removed.
    pub fn delete_namespace(&self, ns: &str) -> Result<usize, EngramError> {
        let conn = self.conn()?;

        // Collect IDs to remove from vec index
        let mut stmt = conn.prepare("SELECT id FROM memories WHERE namespace = ?1")?;
        let ids: Vec<String> = stmt.query_map(params![ns], |row| row.get(0))?
            .filter_map(|r| r.map_err(|e| tracing::warn!("row parse: {e}")).ok())
            .collect();
        drop(stmt);

        // Delete facts associated with memories in this namespace
        conn.execute(
            "DELETE FROM facts WHERE memory_id IN (SELECT id FROM memories WHERE namespace = ?1)",
            params![ns],
        )?;

        // batch-delete FTS entries for this namespace
        conn.execute(
            "DELETE FROM memories_fts WHERE id IN (SELECT id FROM memories WHERE namespace = ?1)",
            params![ns],
        )?;
        let n = conn.execute("DELETE FROM memories WHERE namespace = ?1", params![ns])?;

        for id in &ids {
            self.vec_index_remove(id);
        }

        Ok(n)
    }

    // -- Trash (soft-delete recovery) --

    pub fn trash_list(&self, limit: usize, offset: usize) -> Result<Vec<TrashEntry>, EngramError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, content, layer, importance, created_at, deleted_at, tags, namespace, source, kind \
             FROM trash ORDER BY deleted_at DESC LIMIT ?1 OFFSET ?2"
        )?;
        let rows = stmt.query_map(params![limit as i64, offset as i64], |row| {
            let tags_json: String = row.get(6)?;
            let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
            Ok(TrashEntry {
                id: row.get(0)?,
                content: row.get(1)?,
                layer: row.get(2)?,
                importance: row.get(3)?,
                created_at: row.get(4)?,
                deleted_at: row.get(5)?,
                tags,
                namespace: row.get(7)?,
                source: row.get(8)?,
                kind: row.get(9)?,
            })
        })?.filter_map(|r| r.map_err(|e| tracing::warn!("row parse: {e}")).ok()).collect();
        Ok(rows)
    }

    pub fn trash_restore(&self, id: &str) -> Result<bool, EngramError> {
        let conn = self.conn()?;
        // Read from trash
        let mut stmt = conn.prepare(
            "SELECT id, content, layer, importance, created_at, tags, namespace, source, kind FROM trash WHERE id = ?1"
        )?;
        type TrashRow = (String, String, i64, f64, i64, String, String, String, String);
        let entry: Option<TrashRow> =
            stmt.query_row(params![id], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?,
                    row.get(4)?, row.get(5)?, row.get(6)?, row.get(7)?, row.get(8)?))
            }).ok();
        drop(stmt);

        if let Some((rid, content, layer, importance, created_at, tags_json, ns, source, kind)) = entry {
            let now = now_ms();
            conn.execute(
                "INSERT OR REPLACE INTO memories (id, content, layer, importance, created_at, last_accessed, \
                 access_count, repetition_count, decay_rate, source, tags, namespace, kind) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0, 0, 1.0, ?7, ?8, ?9, ?10)",
                params![rid, content, layer, importance, created_at, now, source, tags_json, ns, kind],
            )?;
            // Re-index FTS
            self.fts_insert(&rid, &content, &tags_json)?;
            // Remove from trash
            conn.execute("DELETE FROM trash WHERE id = ?1", params![id])?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn trash_purge(&self) -> Result<usize, EngramError> {
        let n = self.conn()?.execute("DELETE FROM trash", [])?;
        Ok(n)
    }

    pub fn trash_count(&self) -> Result<usize, EngramError> {
        let n: i64 = self.conn()?.query_row("SELECT COUNT(*) FROM trash", [], |r| r.get(0))?;
        Ok(n as usize)
    }

    pub fn touch(&self, id: &str) -> Result<(), EngramError> {
        // Recall-based reinforcement. Only bump importance for
        // Buffer/Working — Core memories are already at their ceiling.
        self.conn()?.execute(
            "UPDATE memories SET last_accessed = ?1, access_count = access_count + 1, \
             importance = CASE WHEN layer < 3 THEN MIN(1.0, importance + 0.02) ELSE importance END \
             WHERE id = ?2",
            params![now_ms(), id],
        )?;
        Ok(())
    }

    /// Repetition-based reinforcement — stronger than recall touch.
    /// Called when near-duplicate content is written again, indicating
    /// the author considers this information worth restating.
    pub fn reinforce(&self, id: &str) -> Result<(), EngramError> {
        self.conn()?.execute(
            "UPDATE memories SET last_accessed = ?1, \
             repetition_count = repetition_count + 1, \
             importance = MIN(1.0, importance + 0.05) WHERE id = ?2",
            params![now_ms(), id],
        )?;
        Ok(())
    }

    /// Decay importance for memories not accessed recently.
    /// Returns the number of memories affected.
    pub fn decay_importance(&self, idle_hours: f64, decay_amount: f64, floor: f64) -> Result<usize, EngramError> {
        let cutoff = now_ms() - (idle_hours * 3_600_000.0) as i64;
        // Only decay Buffer and Working — Core memories earned their spot
        // and shouldn't lose importance from inactivity.
        let n = self.conn()?.execute(
            "UPDATE memories SET importance = MAX(?1, importance - ?2) \
             WHERE last_accessed < ?3 AND importance > ?1 AND layer < 3",
            params![floor, decay_amount, cutoff],
        )?;
        Ok(n)
    }

    /// Set access_count directly (used when merging memories to preserve history).
    pub fn set_access_count(&self, id: &str, count: i64) -> Result<(), EngramError> {
        self.conn()?.execute(
            "UPDATE memories SET access_count = ?1 WHERE id = ?2",
            params![count, id],
        )?;
        Ok(())
    }

    /// List all memories in a given layer, ordered by importance descending.
    /// List memories by layer, excluding embedding blobs.
    /// Supports optional namespace filter in SQL.
    pub fn list_by_layer_meta(&self, layer: Layer, limit: usize, offset: usize) -> Vec<Memory> {
        self.list_by_layer_meta_ns(layer, limit, offset, None)
    }

    pub fn list_by_layer_meta_ns(
        &self, layer: Layer, limit: usize, offset: usize, ns: Option<&str>,
    ) -> Vec<Memory> {
        let Ok(conn) = self.conn() else { return vec![]; };
        let mut sql = format!(
            "SELECT {META_COLS} FROM memories WHERE layer = ?1"
        );
        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![
            Box::new(layer as u8),
            Box::new(limit as i64),
            Box::new(offset as i64),
        ];
        if let Some(ns) = ns {
            params.push(Box::new(ns.to_string()));
            sql += &format!(" AND namespace = ?{}", params.len());
        }
        sql += " ORDER BY importance DESC LIMIT ?2 OFFSET ?3";
        let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(std::convert::AsRef::as_ref).collect();
        let Ok(mut stmt) = conn.prepare(&sql) else { return vec![]; };
        stmt.query_map(param_refs.as_slice(), row_to_memory_meta)
            .map(|iter| iter.filter_map(|r| r.map_err(|e| tracing::warn!("row parse: {e}")).ok()).collect())
            .unwrap_or_default()
    }

    /// Find memories below a decay score threshold (Buffer/Working only).
    /// Uses meta-only query — embeddings not needed for decay checks.
    pub(crate) fn get_decayed(&self, threshold: f64) -> Vec<Memory> {
        let now = now_ms();
        let Ok(conn) = self.conn() else { return vec![]; };
        let sql = format!("SELECT {META_COLS} FROM memories WHERE layer < 3");
        let Ok(mut stmt) = conn.prepare(&sql) else {
            return vec![];
        };
        stmt.query_map([], row_to_memory_meta)
            .map(|iter| {
                iter.filter_map(|r| r.map_err(|e| tracing::warn!("row parse: {e}")).ok())
                    .filter(|m| {
                        let hours = (now - m.last_accessed) as f64 / 3_600_000.0;
                        let score = m.importance * (-m.decay_rate * hours / 168.0).exp();
                        score < threshold
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// List all memories with pagination.
    pub fn list_all(&self, limit: usize, offset: usize) -> Result<Vec<Memory>, EngramError> {
        self.list_filtered(limit, offset, None, None, None)
    }

    pub fn list_all_ns(
        &self,
        limit: usize,
        offset: usize,
        ns: Option<&str>,
    ) -> Result<Vec<Memory>, EngramError> {
        self.list_filtered(limit, offset, ns, None, None)
    }

    /// List memories with optional namespace, layer, and tag filters — all pushed to SQL.
    pub fn list_filtered(
        &self,
        limit: usize,
        offset: usize,
        ns: Option<&str>,
        layer: Option<u8>,
        tag: Option<&str>,
    ) -> Result<Vec<Memory>, EngramError> {
        let conn = self.conn()?;

        let mut params_vec: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let mut clauses = Vec::new();

        if let Some(n) = ns {
            params_vec.push(Box::new(n.to_string()));
            clauses.push(format!("namespace = ?{}", params_vec.len()));
        }
        if let Some(l) = layer {
            params_vec.push(Box::new(l as i64));
            clauses.push(format!("layer = ?{}", params_vec.len()));
        }
        if let Some(t) = tag {
            let pattern = format!("%\"{}\"%" , t.replace('"', ""));
            params_vec.push(Box::new(pattern));
            clauses.push(format!("tags LIKE ?{}", params_vec.len()));
        }

        params_vec.push(Box::new(limit as i64));
        let limit_idx = params_vec.len();
        params_vec.push(Box::new(offset as i64));
        let offset_idx = params_vec.len();

        let mut sql = String::from("SELECT * FROM memories");
        if !clauses.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&clauses.join(" AND "));
        }
        sql.push_str(&format!(" ORDER BY created_at DESC LIMIT ?{limit_idx} OFFSET ?{offset_idx}"));

        let mut stmt = conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params_vec.iter().map(std::convert::AsRef::as_ref).collect();
        let rows: Vec<Memory> = stmt
            .query_map(param_refs.as_slice(), row_to_memory)?
            .filter_map(|r| r.map_err(|e| tracing::warn!("row parse: {e}")).ok())
            .collect();
        Ok(rows)
    }

    /// List memories created since a given timestamp, ordered by creation time descending.
    pub fn list_since(&self, since_ms: i64, limit: usize) -> Result<Vec<Memory>, EngramError> {
        self.list_since_filtered(since_ms, limit, None, None, None, None)
    }

    /// Filtered variant — pushes all filters to SQL instead of Rust `retain()`.
    pub fn list_since_filtered(
        &self,
        since_ms: i64,
        limit: usize,
        ns: Option<&str>,
        layer: Option<u8>,
        min_importance: Option<f64>,
        source: Option<&str>,
    ) -> Result<Vec<Memory>, EngramError> {
        let conn = self.conn()?;

        // Collect parameter values first, track SQL fragments
        let mut params_vec: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(since_ms)];
        let mut clauses = Vec::new();

        if let Some(n) = ns {
            params_vec.push(Box::new(n.to_string()));
            clauses.push(format!("namespace = ?{}", params_vec.len()));
        }
        if let Some(l) = layer {
            params_vec.push(Box::new(l as i64));
            clauses.push(format!("layer = ?{}", params_vec.len()));
        }
        if let Some(mi) = min_importance {
            params_vec.push(Box::new(mi));
            clauses.push(format!("importance >= ?{}", params_vec.len()));
        }
        if let Some(s) = source {
            params_vec.push(Box::new(s.to_string()));
            clauses.push(format!("source = ?{}", params_vec.len()));
        }
        params_vec.push(Box::new(limit as i64));
        let limit_idx = params_vec.len();

        let mut sql = "SELECT * FROM memories WHERE created_at >= ?1".to_string();
        for c in &clauses {
            sql.push_str(" AND ");
            sql.push_str(c);
        }
        sql.push_str(&format!(" ORDER BY created_at DESC LIMIT ?{limit_idx}"));

        let mut stmt = conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params_vec.iter().map(std::convert::AsRef::as_ref).collect();
        let rows = stmt
            .query_map(param_refs.as_slice(), row_to_memory)?
            .filter_map(|r| r.map_err(|e| tracing::warn!("row parse: {e}")).ok())
            .collect();
        Ok(rows)
    }

    /// List memories that have a specific tag (exact match).
    pub fn list_by_tag(&self, tag: &str, ns: Option<&str>) -> Result<Vec<Memory>, EngramError> {
        let pattern = format!("%\"{}\"%", tag.replace('"', ""));
        let conn = self.conn()?;
        if let Some(ns) = ns {
            let mut stmt = conn.prepare(
                "SELECT * FROM memories WHERE tags LIKE ?1 AND namespace = ?2 \
                 ORDER BY created_at DESC",
            )?;
            let rows = stmt.query_map(params![pattern, ns], row_to_memory)?
                .filter_map(|r| r.map_err(|e| tracing::warn!("row parse: {e}")).ok())
                .collect();
            Ok(rows)
        } else {
            let mut stmt = conn.prepare(
                "SELECT * FROM memories WHERE tags LIKE ?1 \
                 ORDER BY created_at DESC",
            )?;
            let rows = stmt.query_map(params![pattern], row_to_memory)?
                .filter_map(|r| r.map_err(|e| tracing::warn!("row parse: {e}")).ok())
                .collect();
            Ok(rows)
        }
    }

    /// Promote a memory to a higher layer.
    pub fn promote(&self, id: &str, target: Layer) -> Result<Option<Memory>, EngramError> {
        let Some(m) = self.get(id)? else {
            return Ok(None);
        };
        if m.layer as u8 >= target as u8 {
            return Ok(Some(m));
        }

        self.conn()?.execute(
            "UPDATE memories SET layer = ?1, decay_rate = ?2, last_accessed = ?3 WHERE id = ?4",
            params![target as u8, target.default_decay(), now_ms(), id],
        )?;
        self.get(id)
    }

    pub fn demote(&self, id: &str, target: Layer) -> Result<Option<Memory>, EngramError> {
        let Some(m) = self.get(id)? else {
            return Ok(None);
        };
        if (m.layer as u8) <= (target as u8) {
            return Ok(Some(m));
        }
        self.conn()?.execute(
            "UPDATE memories SET layer = ?1, decay_rate = ?2 WHERE id = ?3",
            params![target as u8, target.default_decay(), id],
        )?;
        self.get(id)
    }

    pub fn stats(&self) -> Stats {
        self.stats_impl(None)
    }

    pub fn stats_ns(&self, ns: &str) -> Stats {
        self.stats_impl(Some(ns))
    }

    fn stats_impl(&self, ns: Option<&str>) -> Stats {
        let mut s = Stats {
            total: 0, buffer: 0, working: 0, core: 0,
            by_kind: KindStats::default(),
        };
        let Ok(conn) = self.conn() else { return s; };
        let ns_clause = if ns.is_some() { " WHERE namespace = ?1" } else { "" };

        // Layer counts
        {
            let sql = format!("SELECT layer, COUNT(*) FROM memories{ns_clause} GROUP BY layer");
            let mut apply = |rows: rusqlite::Rows| {
                for pair in rows.mapped(|r| Ok((r.get::<_,u8>(0)?, r.get::<_,i64>(1)? as usize))).flatten() {
                    s.total += pair.1;
                    match pair.0 { 1 => s.buffer = pair.1, 2 => s.working = pair.1, 3 => s.core = pair.1, _ => {} }
                }
            };
            if let Ok(mut stmt) = conn.prepare(&sql) {
                if let Some(n) = ns {
                    if let Ok(rows) = stmt.query(params![n]) { apply(rows); }
                } else if let Ok(rows) = stmt.query([]) { apply(rows); }
            }
        }

        // Kind counts
        {
            let sql = format!("SELECT kind, COUNT(*) FROM memories{ns_clause} GROUP BY kind");
            let mut apply = |rows: rusqlite::Rows| {
                for pair in rows.mapped(|r| Ok((r.get::<_,String>(0)?, r.get::<_,i64>(1)? as usize))).flatten() {
                    match pair.0.as_str() {
                        "semantic" => s.by_kind.semantic = pair.1,
                        "episodic" => s.by_kind.episodic = pair.1,
                        "procedural" => s.by_kind.procedural = pair.1,
                        _ => {}
                    }
                }
            };
            if let Ok(mut stmt) = conn.prepare(&sql) {
                if let Some(n) = ns {
                    if let Ok(rows) = stmt.query(params![n]) { apply(rows); }
                } else if let Ok(rows) = stmt.query([]) { apply(rows); }
            }
        }

        s
    }

    pub fn list_namespaces(&self) -> Vec<String> {
        let Ok(conn) = self.conn() else { return vec![] };
        let mut stmt = match conn.prepare("SELECT DISTINCT namespace FROM memories ORDER BY namespace") {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        stmt.query_map([], |r| r.get(0))
            .map(|rows| rows.flatten().collect())
            .unwrap_or_default()
    }

    /// Check DB integrity: FTS sync, orphans, missing embeddings.
    pub fn integrity(&self) -> IntegrityReport {
        let conn = match self.conn() {
            Ok(c) => c,
            Err(_) => return IntegrityReport::default(),
        };
        let total: i64 = conn.query_row("SELECT count(*) FROM memories", [], |r| r.get(0)).unwrap_or(0);
        let fts_count: i64 = conn.query_row("SELECT count(DISTINCT id) FROM memories_fts", [], |r| r.get(0)).unwrap_or(0);
        let orphan_fts: i64 = conn.query_row(
            "SELECT count(*) FROM memories_fts WHERE id NOT IN (SELECT id FROM memories)", [], |r| r.get(0)
        ).unwrap_or(0);
        let missing_fts: i64 = conn.query_row(
            "SELECT count(*) FROM memories WHERE id NOT IN (SELECT id FROM memories_fts)", [], |r| r.get(0)
        ).unwrap_or(0);
        let no_embedding: i64 = conn.query_row(
            "SELECT count(*) FROM memories WHERE embedding IS NULL", [], |r| r.get(0)
        ).unwrap_or(0);
        IntegrityReport {
            total: total as usize,
            fts_indexed: fts_count as usize,
            orphan_fts: orphan_fts as usize,
            missing_fts: missing_fts as usize,
            missing_embedding: no_embedding as usize,
            ok: orphan_fts == 0 && missing_fts == 0,
        }
    }

    pub fn update_fields(
        &self,
        id: &str,
        content: Option<&str>,
        layer: Option<u8>,
        importance: Option<f64>,
        tags: Option<&[String]>,
    ) -> Result<Option<Memory>, EngramError> {
        if self.get(id)?.is_none() {
            return Ok(None);
        }

        // Validate everything before touching the DB
        if let Some(c) = content {
            if c.trim().is_empty() {
                return Err(EngramError::EmptyContent);
            }
            if c.chars().count() > MAX_CONTENT_LEN {
                return Err(EngramError::ContentTooLong);
            }
        }
        if let Some(l) = layer {
            if !(1..=3).contains(&l) {
                return Err(EngramError::InvalidLayer(l));
            }
        }
        if let Some(t) = tags {
            if t.len() > MAX_TAGS {
                return Err(EngramError::Validation(format!("too many tags (max {MAX_TAGS})")));
            }
            if let Some(tag) = t.iter().find(|tag| tag.chars().count() > MAX_TAG_LEN) {
                return Err(EngramError::Validation(format!("tag '{}' too long", tag)));
            }
        }

        let mut set_clauses: Vec<String> = Vec::new();
        let mut values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(c) = content {
            set_clauses.push("content=?".into());
            values.push(Box::new(c.to_string()));
        }
        if let Some(l) = layer {
            set_clauses.push("layer=?".into());
            values.push(Box::new(l));
        }
        if let Some(i) = importance {
            set_clauses.push("importance=?".into());
            values.push(Box::new(i.clamp(0.0, 1.0)));
        }
        if let Some(t) = tags {
            set_clauses.push("tags=?".into());
            let j = serde_json::to_string(t).unwrap_or_else(|_| "[]".into());
            values.push(Box::new(j));
        }

        if !set_clauses.is_empty() {
            values.push(Box::new(id.to_string()));
            let sql = format!(
                "UPDATE memories SET {} WHERE id=?",
                set_clauses.join(", ")
            );
            let params: Vec<&dyn rusqlite::types::ToSql> = values.iter().map(std::convert::AsRef::as_ref).collect();
            self.conn()?.execute(&sql, params.as_slice())?;
        }

        // Rebuild FTS entry if content or tags changed
        if content.is_some() || tags.is_some() {
            if let Ok(Some(mem)) = self.get(id) {
                self.fts_delete(id)?;
                let tags_json = serde_json::to_string(&mem.tags).unwrap_or_else(|_| "[]".into());
                self.fts_insert(id, &mem.content, &tags_json)?;
            }
        }

        self.get(id)
    }

    /// Update just the kind field for a memory.
    pub fn update_kind(&self, id: &str, kind: &str) -> Result<(), EngramError> {
        self.conn()?.execute(
            "UPDATE memories SET kind = ?1 WHERE id = ?2",
            params![kind, id],
        )?;
        Ok(())
    }

    /// Export all memories as a JSON-serializable vec (for backup/migration).
    /// Embeddings are excluded to keep exports portable.
    pub fn export_all(&self) -> Result<Vec<Memory>, EngramError> {
        self.export_with_embeddings(false)
    }

    pub fn export_with_embeddings(&self, include: bool) -> Result<Vec<Memory>, EngramError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare("SELECT * FROM memories ORDER BY created_at ASC")?;
        let mapper = if include { row_to_memory_with_embedding } else { row_to_memory };
        let rows = stmt
            .query_map([], mapper)?
            .filter_map(|r| r.map_err(|e| tracing::warn!("row parse: {e}")).ok())
            .collect();
        Ok(rows)
    }

    /// Import memories from an export. Skips entries whose id already exists.
    /// Returns count of newly imported memories.
    pub fn import(&self, memories: &[Memory]) -> Result<usize, EngramError> {
        let conn = self.conn()?;
        conn.execute_batch("BEGIN")?;
        let mut imported = 0;
        let result = (|| -> Result<(), EngramError> {
            for m in memories {
                // If same id exists, skip (idempotent re-import).
                // Use INSERT OR IGNORE pattern.
                let exists: bool = conn.query_row(
                    "SELECT COUNT(*) FROM memories WHERE id = ?1",
                    params![m.id],
                    |r| r.get::<_, i64>(0),
                ).unwrap_or(0) > 0;

                let actual_id = if exists {
                    // When importing into a different namespace and id collides,
                    // mint a fresh id so the memory still gets imported.
                    let same_ns: bool = conn.query_row(
                        "SELECT COUNT(*) FROM memories WHERE id = ?1 AND namespace = ?2",
                        params![m.id, m.namespace],
                        |r| r.get::<_, i64>(0),
                    ).unwrap_or(0) > 0;
                    if same_ns {
                        continue; // true duplicate — same id, same namespace
                    }
                    uuid::Uuid::new_v4().to_string()
                } else {
                    m.id.clone()
                };
                let tags_json = serde_json::to_string(&m.tags).unwrap_or_else(|_| "[]".into());
                conn.execute(
                    "INSERT INTO memories \
                     (id, content, layer, importance, created_at, last_accessed, \
                      access_count, repetition_count, decay_rate, source, tags, namespace, embedding, risk_score, kind) \
                     VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15)",
                    params![
                        actual_id,
                        m.content,
                        m.layer as u8,
                        m.importance,
                        m.created_at,
                        m.last_accessed,
                        m.access_count,
                        m.repetition_count,
                        m.decay_rate,
                        m.source,
                        tags_json,
                        m.namespace,
                        m.embedding.as_ref().map(|e| crate::ai::embedding_to_bytes(e)),
                        m.risk_score,
                        m.kind,
                    ],
                )?;
                let processed = append_segmented(&m.content);
                conn.execute(
                    "INSERT INTO memories_fts(id, content, tags) VALUES (?1, ?2, ?3)",
                    params![actual_id, processed, tags_json],
                )?;
                imported += 1;
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                conn.execute_batch("COMMIT")?;
                Ok(imported)
            }
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK");
                Err(e)
            }
        }
    }

    /// Check if content is a near-duplicate of an existing memory.
    /// Uses token-level Jaccard similarity (threshold ~0.8).
    /// Check if content is near-duplicate of an existing memory (default namespace).
    pub fn is_near_duplicate(&self, content: &str) -> bool {
        self.find_near_duplicate_threshold(content, "", 0.8).is_some()
    }

    /// Check with a custom Jaccard threshold (default is 0.8).
    pub fn is_near_duplicate_with(&self, content: &str, threshold: f64) -> bool {
        self.find_near_duplicate_threshold(content, "", threshold).is_some()
    }

    pub(crate) fn find_near_duplicate(&self, content: &str, ns: &str) -> Option<Memory> {
        self.find_near_duplicate_threshold(content, ns, 0.8)
    }

    fn find_near_duplicate_threshold(&self, content: &str, ns: &str, threshold: f64) -> Option<Memory> {
        // Use only the first 200 chars for the FTS query — enough for similarity matching
        let query_text = if content.len() > 200 {
            &content[..content.char_indices().take(200).last().map(|(i, c)| i + c.len_utf8()).unwrap_or(200)]
        } else {
            content
        };
        let candidates = self.search_fts_ns(query_text, 5, Some(ns));
        if candidates.is_empty() {
            return None;
        }

        let new_tokens = tokenize_for_dedup(content);
        if new_tokens.len() < 3 {
            return None;
        }

        for (id, _) in &candidates {
            if let Ok(Some(mem)) = self.get(id) {
                if mem.namespace != ns {
                    continue;
                }
                let old_tokens = tokenize_for_dedup(&mem.content);
                let intersection = new_tokens.intersection(&old_tokens).count();
                let union = new_tokens.union(&old_tokens).count();
                if union > 0 {
                    let jaccard = intersection as f64 / union as f64;
                    if jaccard > threshold {
                        return Some(mem);
                    }
                }
            }
        }
        None
    }

}


#[cfg(test)]
#[path = "memory_tests.rs"]
mod memory_tests;
