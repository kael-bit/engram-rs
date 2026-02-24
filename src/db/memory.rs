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
            .filter_map(|r| r.ok())
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
            .filter_map(|r| r.ok())
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
        })?.filter_map(|r| r.ok()).collect();
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
    pub fn list_by_layer(&self, layer: Layer, limit: usize, offset: usize) -> Vec<Memory> {
        let Ok(conn) = self.conn() else { return vec![]; };
        let Ok(mut stmt) = conn.prepare(
            "SELECT * FROM memories WHERE layer = ?1 ORDER BY importance DESC LIMIT ?2 OFFSET ?3",
        ) else {
            return vec![];
        };

        stmt.query_map(params![layer as u8, limit as i64, offset as i64], row_to_memory)
            .map(|iter| iter.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
    }

    /// Like `list_by_layer` but excludes the embedding blob.
    /// Use this when you only need metadata — saves significant memory when
    /// the DB has thousands of entries with 1536-dim embeddings.
    pub fn list_by_layer_meta(&self, layer: Layer, limit: usize, offset: usize) -> Vec<Memory> {
        self.list_by_layer_meta_ns(layer, limit, offset, None)
    }

    /// Like `list_by_layer_meta` but with optional namespace filter in SQL.
    pub fn list_by_layer_meta_ns(
        &self, layer: Layer, limit: usize, offset: usize, ns: Option<&str>,
    ) -> Vec<Memory> {
        let Ok(conn) = self.conn() else { return vec![]; };
        if let Some(ns) = ns {
            let sql = format!(
                "SELECT {META_COLS} FROM memories WHERE layer = ?1 AND namespace = ?4 \
                 ORDER BY importance DESC LIMIT ?2 OFFSET ?3"
            );
            let Ok(mut stmt) = conn.prepare(&sql) else { return vec![]; };
            stmt.query_map(
                params![layer as u8, limit as i64, offset as i64, ns],
                row_to_memory_meta,
            )
            .map(|iter| iter.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
        } else {
            let sql = format!(
                "SELECT {META_COLS} FROM memories WHERE layer = ?1 \
                 ORDER BY importance DESC LIMIT ?2 OFFSET ?3"
            );
            let Ok(mut stmt) = conn.prepare(&sql) else { return vec![]; };
            stmt.query_map(
                params![layer as u8, limit as i64, offset as i64],
                row_to_memory_meta,
            )
            .map(|iter| iter.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
        }
    }

    /// Like `get_decayed` but excludes the embedding blob.
    pub(crate) fn get_decayed_meta(&self, threshold: f64) -> Vec<Memory> {
        let now = now_ms();
        let Ok(conn) = self.conn() else { return vec![]; };
        let sql = format!("SELECT {META_COLS} FROM memories WHERE layer < 3");
        let Ok(mut stmt) = conn.prepare(&sql) else {
            return vec![];
        };
        stmt.query_map([], row_to_memory_meta)
            .map(|iter| {
                iter.filter_map(|r| r.ok())
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
            params_vec.iter().map(|p| p.as_ref()).collect();
        let rows: Vec<Memory> = stmt
            .query_map(param_refs.as_slice(), row_to_memory)?
            .filter_map(|r| r.ok())
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
            params_vec.iter().map(|p| p.as_ref()).collect();
        let rows = stmt
            .query_map(param_refs.as_slice(), row_to_memory)?
            .filter_map(|r| r.ok())
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
                .filter_map(|r| r.ok())
                .collect();
            Ok(rows)
        } else {
            let mut stmt = conn.prepare(
                "SELECT * FROM memories WHERE tags LIKE ?1 \
                 ORDER BY created_at DESC",
            )?;
            let rows = stmt.query_map(params![pattern], row_to_memory)?
                .filter_map(|r| r.ok())
                .collect();
            Ok(rows)
        }
    }

    /// Find memories whose decay score has fallen below a threshold.
    pub fn get_decayed(&self, threshold: f64) -> Vec<Memory> {
        let now = now_ms();
        let Ok(conn) = self.conn() else { return vec![]; };
        let Ok(mut stmt) = conn.prepare("SELECT * FROM memories WHERE layer < 3") else {
            return vec![];
        };
        stmt.query_map([], row_to_memory)
            .map(|iter| {
                iter.filter_map(|r| r.ok())
                    .filter(|m| {
                        let hours = (now - m.last_accessed) as f64 / 3_600_000.0;
                        let score = m.importance * (-m.decay_rate * hours / 168.0).exp();
                        score < threshold
                    })
                    .collect()
            })
            .unwrap_or_default()
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
        let mut s = Stats {
            total: 0,
            buffer: 0,
            working: 0,
            core: 0,
            by_kind: KindStats::default(),
        };
        let Ok(conn) = self.conn() else { return s; };
        let Ok(mut stmt) = conn
            .prepare("SELECT layer, COUNT(*) FROM memories GROUP BY layer")
        else {
            return s;
        };
        let Ok(rows) = stmt.query_map([], |row| {
            Ok((row.get::<_, u8>(0)?, row.get::<_, i64>(1)? as usize))
        }) else {
            return s;
        };

        for r in rows.flatten() {
            s.total += r.1;
            match r.0 {
                1 => s.buffer = r.1,
                2 => s.working = r.1,
                3 => s.core = r.1,
                _ => {}
            }
        }

        if let Ok(mut stmt) = conn.prepare("SELECT kind, COUNT(*) FROM memories GROUP BY kind") {
            if let Ok(rows) = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as usize))
            }) {
                for r in rows.flatten() {
                    match r.0.as_str() {
                        "semantic" => s.by_kind.semantic = r.1,
                        "episodic" => s.by_kind.episodic = r.1,
                        "procedural" => s.by_kind.procedural = r.1,
                        _ => {}
                    }
                }
            }
        }

        s
    }

    pub fn stats_ns(&self, ns: &str) -> Stats {
        let mut s = Stats { total: 0, buffer: 0, working: 0, core: 0, by_kind: KindStats::default() };
        let Ok(conn) = self.conn() else { return s; };
        let Ok(mut stmt) = conn.prepare(
            "SELECT layer, COUNT(*) FROM memories WHERE namespace = ?1 GROUP BY layer",
        ) else {
            return s;
        };
        let Ok(rows) = stmt.query_map(params![ns], |row| {
            Ok((row.get::<_, u8>(0)?, row.get::<_, i64>(1)? as usize))
        }) else {
            return s;
        };
        for r in rows.flatten() {
            s.total += r.1;
            match r.0 {
                1 => s.buffer = r.1,
                2 => s.working = r.1,
                3 => s.core = r.1,
                _ => {}
            }
        }

        if let Ok(mut stmt) = conn.prepare(
            "SELECT kind, COUNT(*) FROM memories WHERE namespace = ?1 GROUP BY kind",
        ) {
            if let Ok(rows) = stmt.query_map(params![ns], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as usize))
            }) {
                for r in rows.flatten() {
                    match r.0.as_str() {
                        "semantic" => s.by_kind.semantic = r.1,
                        "episodic" => s.by_kind.episodic = r.1,
                        "procedural" => s.by_kind.procedural = r.1,
                        _ => {}
                    }
                }
            }
        }

        s
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
            let params: Vec<&dyn rusqlite::types::ToSql> = values.iter().map(|v| v.as_ref()).collect();
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
            .filter_map(|r| r.ok())
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
        self.find_near_duplicate(content, "").is_some()
    }

    /// Check with a custom Jaccard threshold (default is 0.8).
    pub fn is_near_duplicate_with(&self, content: &str, threshold: f64) -> bool {
        self.find_near_duplicate_threshold(content, "", threshold).is_some()
    }

    /// Compare two strings directly using Jaccard similarity (no DB lookup).
    pub fn is_near_duplicate_pair(&self, a: &str, b: &str, threshold: f64) -> bool {
        let tokens_a = tokenize_for_dedup(a);
        let tokens_b = tokenize_for_dedup(b);
        if tokens_a.len() < 3 || tokens_b.len() < 3 {
            return false;
        }
        let intersection = tokens_a.intersection(&tokens_b).count();
        let union = tokens_a.union(&tokens_b).count();
        union > 0 && (intersection as f64 / union as f64) > threshold
    }

    fn find_near_duplicate(&self, content: &str, ns: &str) -> Option<Memory> {
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
mod tests {
    use super::*;

    fn test_db() -> MemoryDB {
        MemoryDB::open(":memory:").expect("in-memory db")
    }

    #[test]
    fn basic_crud() {
        let db = test_db();
        let mem = db
            .insert(MemoryInput {
                content: "test memory".into(),
                layer: Some(2),
                importance: Some(0.8),
                source: None,
                tags: Some(vec!["test".into()]),
            ..Default::default()
            })
            .unwrap();

        assert_eq!(mem.layer, Layer::Working);
        assert!((mem.importance - 0.8).abs() < f64::EPSILON);
        assert_eq!(mem.tags, vec!["test"]);

        let got = db.get(&mem.id).unwrap().unwrap();
        assert_eq!(got.content, "test memory");
    }

    #[test]
    fn delete_missing() {
        let db = test_db();
        assert!(!db.delete("nonexistent").unwrap());
    }

    #[test]
    fn touch() {
        let db = test_db();
        let mem = db
            .insert(MemoryInput {
                content: "touchable".into(),
                layer: None,
                importance: Some(0.3),
                ..Default::default()
            })
            .unwrap();

        db.touch(&mem.id).unwrap();
        let got = db.get(&mem.id).unwrap().unwrap();
        assert_eq!(got.access_count, 1);
        assert!((got.importance - 0.32).abs() < 0.001, "imp={}", got.importance);

        // multiple touches accumulate
        db.touch(&mem.id).unwrap();
        db.touch(&mem.id).unwrap();
        let got = db.get(&mem.id).unwrap().unwrap();
        assert_eq!(got.access_count, 3);
        assert!((got.importance - 0.36).abs() < 0.001, "imp={}", got.importance);
    }

    #[test]
    fn touch_importance_caps_at_one() {
        let db = test_db();
        let mem = db
            .insert(MemoryInput {
                content: "important thing".into(),
                importance: Some(0.95),
                ..Default::default()
            })
            .unwrap();
        // Two touches would push past 1.0 without the cap
        db.touch(&mem.id).unwrap();
        db.touch(&mem.id).unwrap();
        let got = db.get(&mem.id).unwrap().unwrap();
        assert!(got.importance <= 1.0, "imp should cap at 1.0, got {}", got.importance);
    }

    #[test]
    fn reject_empty() {
        let db = test_db();
        let result = db.insert(MemoryInput {
            content: "   ".into(),
            layer: None,
            importance: None,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn reject_bad_layer() {
        let db = test_db();
        let result = db.insert(MemoryInput {
            content: "test".into(),
            layer: Some(5),
            importance: None,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn clamp_importance() {
        let db = test_db();
        let mem = db
            .insert(MemoryInput {
                content: "clamped".into(),
                layer: None,
                importance: Some(1.5),
                ..Default::default()
            })
            .unwrap();
        assert!((mem.importance - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn promote_moves_layer_up() {
        let db = test_db();
        let mem = db
            .insert(MemoryInput {
                content: "promotable".into(),
                layer: Some(1),
                importance: Some(0.9),
                ..Default::default()
            })
            .unwrap();

        let promoted = db.promote(&mem.id, Layer::Core).unwrap().unwrap();
        assert_eq!(promoted.layer, Layer::Core);
    }

    #[test]
    fn stats() {
        let db = test_db();
        let entries = [
            ("buffer entry alpha", 1),
            ("buffer entry beta zeta", 1),
            ("working entry gamma delta", 2),
            ("core entry epsilon theta", 3),
        ];
        for (content, layer) in entries {
            db.insert(MemoryInput {
                content: content.into(),
                layer: Some(layer),
                importance: None,
                ..Default::default()
            })
            .unwrap();
        }

        let s = db.stats();
        assert_eq!(s.total, 4);
        assert_eq!(s.buffer, 2);
        assert_eq!(s.working, 1);
        assert_eq!(s.core, 1);
    }

    #[test]
    fn partial_update() {
        let db = test_db();
        let mem = db
            .insert(MemoryInput {
                content: "original".into(),
                layer: Some(1),
                importance: Some(0.5),
                ..Default::default()
            })
            .unwrap();

        let updated = db
            .update_fields(&mem.id, Some("updated"), None, Some(0.9), None)
            .unwrap()
            .unwrap();
        assert_eq!(updated.content, "updated");
        assert!((updated.importance - 0.9).abs() < f64::EPSILON);
        assert_eq!(updated.layer, Layer::Buffer); // unchanged
    }

    #[test]
    fn list_all_with_pagination() {
        let db = test_db();
        for i in 0..5 {
            db.insert(MemoryInput {
                content: format!("paginated {i}"),
                layer: None,
                importance: None,
                ..Default::default()
            })
            .unwrap();
        }

        let page1 = db.list_all(3, 0).unwrap();
        assert_eq!(page1.len(), 3);

        let page2 = db.list_all(3, 3).unwrap();
        assert_eq!(page2.len(), 2);
    }

    #[test]
    fn list_filtered_by_ns_layer_tag() {
        let db = test_db();
        db.insert(MemoryInput {
            content: "alpha in ns-a".into(),
            namespace: Some("ns-a".into()),
            tags: Some(vec!["hot".into()]),
            ..Default::default()
        }).unwrap();
        db.insert(MemoryInput {
            content: "beta in ns-b".into(),
            namespace: Some("ns-b".into()),
            tags: Some(vec!["cold".into()]),
            ..Default::default()
        }).unwrap();
        db.insert(MemoryInput {
            content: "gamma in ns-a layer 2".into(),
            namespace: Some("ns-a".into()),
            layer: Some(2),
            tags: Some(vec!["hot".into()]),
            ..Default::default()
        }).unwrap();

        // namespace filter
        let nsa = db.list_filtered(10, 0, Some("ns-a"), None, None).unwrap();
        assert_eq!(nsa.len(), 2);

        // namespace + layer
        let nsa_l2 = db.list_filtered(10, 0, Some("ns-a"), Some(2), None).unwrap();
        assert_eq!(nsa_l2.len(), 1);
        assert!(nsa_l2[0].content.contains("gamma"));

        // tag filter
        let hot = db.list_filtered(10, 0, None, None, Some("hot")).unwrap();
        assert_eq!(hot.len(), 2);

        // all combined
        let combo = db.list_filtered(10, 0, Some("ns-a"), Some(2), Some("hot")).unwrap();
        assert_eq!(combo.len(), 1);

        // no match
        let empty = db.list_filtered(10, 0, Some("ns-a"), None, Some("cold")).unwrap();
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn list_since_filtered_params() {
        let db = test_db();
        let now = super::now_ms();
        db.insert(MemoryInput {
            content: "recent in ns-x".into(),
            namespace: Some("ns-x".into()),
            source: Some("api".into()),
            ..Default::default()
        }).unwrap();
        db.insert(MemoryInput {
            content: "recent in default".into(),
            source: Some("session".into()),
            ..Default::default()
        }).unwrap();

        let since = now - 5000;
        // namespace filter
        let nsx = db.list_since_filtered(since, 10, Some("ns-x"), None, None, None).unwrap();
        assert_eq!(nsx.len(), 1);
        assert!(nsx[0].content.contains("ns-x"));

        // source filter
        let sess = db.list_since_filtered(since, 10, None, None, None, Some("session")).unwrap();
        assert_eq!(sess.len(), 1);
        assert!(sess[0].content.contains("default"));

        // no filter returns all
        let all = db.list_since_filtered(since, 10, None, None, None, None).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn update_kind_changes_field() {
        let db = test_db();
        let mem = db.insert(MemoryInput {
            content: "kind test".into(),
            ..Default::default()
        }).unwrap();
        assert_eq!(mem.kind, "semantic");

        db.update_kind(&mem.id, "procedural").unwrap();
        let got = db.get(&mem.id).unwrap().unwrap();
        assert_eq!(got.kind, "procedural");

        db.update_kind(&mem.id, "episodic").unwrap();
        let got = db.get(&mem.id).unwrap().unwrap();
        assert_eq!(got.kind, "episodic");
    }

    #[test]
    fn dedup_merges_similar() {
        let db = test_db();
        let original = db
            .insert(MemoryInput {
                content: "engram project uses Rust SQLite FTS5 for memory storage and retrieval system".into(),
                layer: Some(1),
                importance: Some(0.5),
                source: None,
                tags: Some(vec!["habit".into()]),
            ..Default::default()
            })
            .unwrap();

        // Insert near-duplicate with one word changed
        let deduped = db
            .insert(MemoryInput {
                content: "engram project uses Rust SQLite FTS5 for memory storage and search system".into(),
                layer: Some(2),
                importance: Some(0.7),
                source: None,
                tags: Some(vec!["preference".into()]),
            ..Default::default()
            })
            .unwrap();

        // Should reuse the same id (updated, not new)
        assert_eq!(deduped.id, original.id);
        // Should keep higher importance
        assert!(deduped.importance >= 0.7);
        // Should merge tags
        assert!(deduped.tags.contains(&"habit".into()));
        assert!(deduped.tags.contains(&"preference".into()));
        // Should be promoted to higher layer
        assert_eq!(deduped.layer, Layer::Working);
        // Total count should still be 1
        assert_eq!(db.stats().total, 1);
    }

    #[test]
    fn cjk_dedup_catches_similar_chinese() {
        let db = test_db();
        let original = db
            .insert(MemoryInput {
                content: "今天下午学习了如何使用向量数据库进行语义搜索和检索任务".into(),
                layer: Some(1),
                importance: Some(0.5),
                source: None,
                tags: Some(vec!["学习".into()]),
            ..Default::default()
            })
            .unwrap();

        // Same meaning, last word changed
        let result = db
            .insert(MemoryInput {
                content: "今天下午学习了如何使用向量数据库进行语义搜索和检索工作".into(),
                layer: Some(2),
                importance: Some(0.7),
                ..Default::default()
            })
            .unwrap();

        assert_eq!(result.id, original.id, "should dedup CJK near-duplicate");
        assert_eq!(db.stats().total, 1);
    }

    #[test]
    fn tokenize_for_dedup_handles_mixed_text() {
        let tokens = tokenize_for_dedup("hello 世界你好 world");
        assert!(tokens.contains("hello"));
        assert!(tokens.contains("world"));
        // jieba segments Chinese properly
        assert!(tokens.contains("世界"), "should contain 世界: {:?}", tokens);
        assert!(tokens.contains("你好"), "should contain 你好: {:?}", tokens);
    }

    #[test]
    fn update_fields_all_at_once() {
        let db = test_db();
        let mem = db
            .insert(MemoryInput {
                content: "before update".into(),
                layer: Some(1),
                importance: Some(0.3),
                source: None,
                tags: Some(vec!["old".into()]),
            ..Default::default()
            })
            .unwrap();

        let new_tags = vec!["new".into(), "shiny".into()];
        let updated = db
            .update_fields(&mem.id, Some("after update"), Some(3), Some(0.95), Some(&new_tags))
            .unwrap()
            .unwrap();

        assert_eq!(updated.content, "after update");
        assert_eq!(updated.layer, Layer::Core);
        assert!((updated.importance - 0.95).abs() < f64::EPSILON);
        assert_eq!(updated.tags, vec!["new", "shiny"]);
    }

    #[test]
    fn row_to_memory_skips_embedding_by_default() {
        let db = test_db();
        let mem = db
            .insert(MemoryInput {
                content: "embedding test".into(),
                layer: None,
                importance: None,
                ..Default::default()
            })
            .unwrap();

        db.set_embedding(&mem.id, &[1.0, 2.0, 3.0]).unwrap();

        // Normal get() should not deserialize the embedding
        let got = db.get(&mem.id).unwrap().unwrap();
        assert!(got.embedding.is_none());

        // But get_all_with_embeddings should have it
        let with_emb = db.get_all_with_embeddings();
        assert_eq!(with_emb.len(), 1);
        assert_eq!(with_emb[0].1, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn supersede_deletes_old() {
        let db = test_db();
        let old = db
            .insert(MemoryInput::new("engram v0.2.1 deployed"))
            .unwrap();
        let old2 = db
            .insert(MemoryInput::new("engram uses port 3917"))
            .unwrap();

        // new memory supersedes the first one
        let new = db
            .insert(
                MemoryInput::new("engram v0.4.0 deployed with /resume endpoint")
                    .supersedes(vec![old.id.clone()]),
            )
            .unwrap();

        assert!(db.get(&old.id).unwrap().is_none(), "old should be deleted");
        assert!(db.get(&old2.id).unwrap().is_some(), "unrelated should stay");
        assert!(db.get(&new.id).unwrap().is_some(), "new should exist");
    }

    #[test]
    fn supersede_multiple() {
        let db = test_db();
        let a = db.insert(MemoryInput::new("fact version 1")).unwrap();
        let b = db.insert(MemoryInput::new("fact version 2")).unwrap();

        let c = db
            .insert(
                MemoryInput::new("fact version 3 (final)")
                    .supersedes(vec![a.id.clone(), b.id.clone()]),
            )
            .unwrap();

        assert!(db.get(&a.id).unwrap().is_none());
        assert!(db.get(&b.id).unwrap().is_none());
        assert_eq!(db.get(&c.id).unwrap().unwrap().content, "fact version 3 (final)");
    }

    #[test]
    fn skip_dedup_allows_similar() {
        let db = test_db();
        let a = db.insert(MemoryInput::new("the sky is blue today")).unwrap();
        // Without skip_dedup, this would merge into `a`
        let b = db
            .insert(MemoryInput::new("the sky is blue today").skip_dedup())
            .unwrap();
        assert_ne!(a.id, b.id, "should create separate memories");
        assert!(db.get(&a.id).unwrap().is_some());
        assert!(db.get(&b.id).unwrap().is_some());
    }

    #[test]
    fn namespace_isolation() {
        let db = test_db();
        let a = db
            .insert(MemoryInput::new("agent-a's secret").namespace("agent-a"))
            .unwrap();
        let b = db
            .insert(MemoryInput::new("agent-b's data").namespace("agent-b"))
            .unwrap();
        let c = db
            .insert(MemoryInput::new("default ns memory"))
            .unwrap();

        // list_all_ns filters by namespace
        let a_mems = db.list_all_ns(50, 0, Some("agent-a")).unwrap();
        assert_eq!(a_mems.len(), 1);
        assert_eq!(a_mems[0].id, a.id);

        let b_mems = db.list_all_ns(50, 0, Some("agent-b")).unwrap();
        assert_eq!(b_mems.len(), 1);
        assert_eq!(b_mems[0].id, b.id);

        // default namespace
        let def_mems = db.list_all_ns(50, 0, Some("default")).unwrap();
        assert_eq!(def_mems.len(), 1);
        assert_eq!(def_mems[0].id, c.id);

        // no namespace filter returns all
        let all = db.list_all_ns(50, 0, None).unwrap();
        assert_eq!(all.len(), 3);

        // stats_ns
        let s = db.stats_ns("agent-a");
        assert_eq!(s.total, 1);
    }

    #[test]
    fn dedup_respects_namespace() {
        let db = test_db();
        let content = "this is identical content for dedup testing purposes";
        let a = db
            .insert(MemoryInput::new(content).namespace("ns-a"))
            .unwrap();
        // Same content in a different namespace should NOT be deduped
        let b = db
            .insert(MemoryInput::new(content).namespace("ns-b"))
            .unwrap();
        assert_ne!(a.id, b.id, "different namespaces should not dedup");

        // Same content in the same namespace SHOULD be deduped (updates existing)
        let a2 = db
            .insert(MemoryInput::new(content).namespace("ns-a"))
            .unwrap();
        assert_eq!(a.id, a2.id, "same namespace should dedup");

        assert_eq!(db.list_all_ns(50, 0, Some("ns-a")).unwrap().len(), 1);
        assert_eq!(db.list_all_ns(50, 0, Some("ns-b")).unwrap().len(), 1);
    }

    #[test]
    fn input_builder_chain() {
        let input = MemoryInput::new("test content")
            .layer(3)
            .importance(0.9)
            .source("unit-test")
            .tags(vec!["a".into(), "b".into()])
            .supersedes(vec!["old-id".into()])
            .skip_dedup()
            .namespace("test-ns");

        assert_eq!(input.content, "test content");
        assert_eq!(input.layer, Some(3));
        assert_eq!(input.importance, Some(0.9));
        assert_eq!(input.source.as_deref(), Some("unit-test"));
        assert_eq!(input.tags.as_ref().unwrap().len(), 2);
        assert_eq!(input.supersedes.as_ref().unwrap(), &["old-id"]);
        assert_eq!(input.skip_dedup, Some(true));
        assert_eq!(input.namespace.as_deref(), Some("test-ns"));
        assert_eq!(input.sync_embed, None);
    }

    #[test]
    fn delete_namespace_removes_all() {
        let db = test_db();
        db.insert(MemoryInput::new("ns-a mem 1").namespace("wipe-me")).unwrap();
        db.insert(MemoryInput::new("ns-a mem 2").namespace("wipe-me")).unwrap();
        db.insert(MemoryInput::new("keep this").namespace("safe")).unwrap();

        let deleted = db.delete_namespace("wipe-me").unwrap();
        assert_eq!(deleted, 2);

        // namespace "safe" untouched
        assert_eq!(db.list_all_ns(10, 0, Some("safe")).unwrap().len(), 1);
        // "wipe-me" is gone
        assert_eq!(db.list_all_ns(10, 0, Some("wipe-me")).unwrap().len(), 0);
    }

    #[test]
    fn delete_batch_by_ids() {
        let db = test_db();
        let m1 = db.insert(MemoryInput::new("batch del 1")).unwrap();
        let m2 = db.insert(MemoryInput::new("batch del 2")).unwrap();
        let m3 = db.insert(MemoryInput::new("batch keep")).unwrap();

        assert!(db.delete(&m1.id).unwrap());
        assert!(db.delete(&m2.id).unwrap());

        assert!(db.get(&m1.id).unwrap().is_none());
        assert!(db.get(&m2.id).unwrap().is_none());
        assert!(db.get(&m3.id).unwrap().is_some());
    }

    #[test]
    fn update_nonexistent_returns_none() {
        let db = test_db();
        let result = db.update_fields("no-such-id", Some("new content"), None, None, None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn export_import_roundtrip() {
        let db = test_db();
        db.insert(MemoryInput::new("roundtrip A").importance(0.8).source("test")).unwrap();
        db.insert(MemoryInput::new("roundtrip B").layer(3).tags(vec!["x".into()])).unwrap();

        let exported = db.export_all().unwrap();
        assert_eq!(exported.len(), 2);

        // import into fresh db
        let db2 = MemoryDB::open(":memory:").unwrap();
        let imported = db2.import(&exported).unwrap();
        assert_eq!(imported, 2);

        let re_exported = db2.export_all().unwrap();
        assert_eq!(re_exported.len(), 2);
        assert_eq!(re_exported[0].content, "roundtrip A");
        assert_eq!(re_exported[1].content, "roundtrip B");
    }

    #[test]
    fn integrity_ok_on_clean_db() {
        let db = test_db();
        db.insert(MemoryInput::new("integrity check")).unwrap();
        let report = db.integrity();
        assert!(report.ok);
        assert_eq!(report.total, 1);
        assert_eq!(report.fts_indexed, 1);
        assert_eq!(report.orphan_fts, 0);
        assert_eq!(report.missing_fts, 0);
    }

    #[test]
    fn repair_fixes_orphan_fts() {
        let db = test_db();
        let mem = db.insert(MemoryInput::new("will break fts")).unwrap();
        // Manually delete from memories but leave FTS orphan
        let conn = db.conn().unwrap();
        conn.execute("DELETE FROM memories WHERE id = ?1", params![mem.id]).unwrap();

        let report = db.integrity();
        assert_eq!(report.orphan_fts, 1);
        assert!(!report.ok);

        let (orphans, rebuilt) = db.repair_fts().unwrap();
        assert_eq!(orphans, 1);
        assert_eq!(rebuilt, 0);

        let report = db.integrity();
        assert!(report.ok);
    }

    #[test]
    fn repair_rebuilds_missing_fts() {
        let db = test_db();
        let mem = db.insert(MemoryInput::new("missing fts entry")).unwrap();
        // Delete FTS but keep memory
        let conn = db.conn().unwrap();
        conn.execute("DELETE FROM memories_fts WHERE id = ?1", params![mem.id]).unwrap();

        let report = db.integrity();
        assert_eq!(report.missing_fts, 1);
        assert!(!report.ok);

        let (orphans, rebuilt) = db.repair_fts().unwrap();
        assert_eq!(orphans, 0);
        assert_eq!(rebuilt, 1);

        let report = db.integrity();
        assert!(report.ok);
        // FTS search should work again
        let results = db.search_fts("missing fts", 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn list_by_tag_returns_matching() {
        let db = test_db();
        let m = db
            .insert(MemoryInput::new("trigger memory").tags(vec!["trigger:git".into()]))
            .unwrap();
        db.insert(MemoryInput::new("untagged memory")).unwrap();

        let results = db.list_by_tag("trigger:git", None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, m.id);
    }

    #[test]
    fn list_by_tag_empty_when_no_match() {
        let db = test_db();
        db.insert(MemoryInput::new("some memory").tags(vec!["other:tag".into()])).unwrap();

        let results = db.list_by_tag("trigger:git", None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn list_by_tag_respects_namespace() {
        let db = test_db();
        let tag = vec!["trigger:deploy".into()];
        let a = db
            .insert(MemoryInput::new("ns-a memory").tags(tag.clone()).namespace("ns-a"))
            .unwrap();
        db.insert(MemoryInput::new("ns-b memory").tags(tag).namespace("ns-b"))
            .unwrap();

        let results = db.list_by_tag("trigger:deploy", Some("ns-a")).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, a.id);
    }

    #[test]
    fn list_by_tag_no_prefix_substring_match() {
        let db = test_db();
        db.insert(MemoryInput::new("push memory").tags(vec!["trigger:git-push".into()])).unwrap();
        db.insert(MemoryInput::new("commit memory").tags(vec!["trigger:git-commit".into()])).unwrap();

        // Searching for "trigger:git" must not match "trigger:git-push" or "trigger:git-commit"
        let results = db.list_by_tag("trigger:git", None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn dedup_reinforces_existing_memory() {
        let db = test_db();
        let original = db.insert(MemoryInput::new(
            "alice prefers dark mode for all user interface applications and code editors"
        )).unwrap();
        assert_eq!(original.access_count, 0);
        assert_eq!(original.repetition_count, 0);
        let orig_imp = original.importance;

        // Write near-duplicate — should reinforce via repetition, not create new
        let updated = db.insert(MemoryInput::new(
            "alice prefers dark mode for all user interface applications and text editors"
        )).unwrap();
        assert_eq!(updated.id, original.id, "should update existing, not create new");
        assert_eq!(updated.access_count, 0, "recall counter should stay at 0");
        assert_eq!(updated.repetition_count, 1, "repetition counter should increment");
        assert!(updated.importance > orig_imp, "dedup should bump importance via reinforce");

        // Third repetition — repetition_count keeps climbing
        let again = db.insert(MemoryInput::new(
            "alice prefers dark mode for all user interface applications and code editors"
        )).unwrap();
        assert_eq!(again.id, original.id, "third repetition should still match");
        assert_eq!(again.repetition_count, 2, "third rep should increment again");
        assert_eq!(again.access_count, 0, "recall counter still untouched");
    }

    #[test]
    fn all_inserts_start_in_buffer() {
        let db = test_db();

        // Low importance → Buffer
        let buf = db.insert(MemoryInput::new("some random fact").importance(0.4)).unwrap();
        assert_eq!(buf.layer, Layer::Buffer);

        // Medium importance → still Buffer (promotion is earned, not declared)
        let work = db.insert(MemoryInput::new("important decision about architecture").importance(0.7)).unwrap();
        assert_eq!(work.layer, Layer::Buffer);

        // High importance → still Buffer
        let core = db.insert(MemoryInput::new("user explicitly said remember this").importance(0.9)).unwrap();
        assert_eq!(core.layer, Layer::Buffer);
        // but importance is preserved for scoring
        assert!(core.importance >= 0.9);

        // Default (no importance) → Buffer
        let def = db.insert(MemoryInput::new("default importance test")).unwrap();
        assert_eq!(def.layer, Layer::Buffer);

        // Explicit layer override still works (for admin/migration)
        let explicit = db.insert(MemoryInput { content: "admin override".into(), layer: Some(3), ..Default::default() }).unwrap();
        assert_eq!(explicit.layer, Layer::Core);
    }

    #[test]
    fn resolve_prefix_works() {
        let db = test_db();
        let m = db.insert(MemoryInput::new("prefix test")).unwrap();
        let prefix = &m.id[..8];

        // Exact match
        assert_eq!(db.resolve_prefix(&m.id).unwrap(), m.id);

        // Prefix match
        assert_eq!(db.resolve_prefix(prefix).unwrap(), m.id);

        // No match
        assert!(db.resolve_prefix("zzzzz").is_err());
    }

    #[test]
    fn batch_insert_all_start_in_buffer() {
        let db = test_db();
        let inputs = vec![
            MemoryInput { importance: Some(0.95), ..MemoryInput::new("explicit remember") },
            MemoryInput { importance: Some(0.75), ..MemoryInput::new("significant fact") },
            MemoryInput { importance: Some(0.3), ..MemoryInput::new("transient note") },
            MemoryInput { layer: Some(2), importance: Some(0.3), ..MemoryInput::new("explicit layer override") },
        ];
        let results = db.insert_batch(inputs).unwrap();
        assert_eq!(results.len(), 4);
        assert_eq!(results[0].layer, Layer::Buffer);     // importance doesn't skip layers
        assert_eq!(results[1].layer, Layer::Buffer);
        assert_eq!(results[2].layer, Layer::Buffer);
        assert_eq!(results[3].layer, Layer::Working);    // explicit layer=2 still works
    }

    #[test]
    fn test_procedural_low_decay() {
        let db = test_db();
        let mem = db.insert(
            MemoryInput::new("how to deploy: run cargo build --release")
                .kind("procedural")
        ).unwrap();
        assert_eq!(mem.kind, "procedural");
        assert!((mem.decay_rate - 0.01).abs() < f64::EPSILON);

        let got = db.get(&mem.id).unwrap().unwrap();
        assert!((got.decay_rate - 0.01).abs() < f64::EPSILON);
        assert_eq!(got.kind, "procedural");
    }

    #[test]
    fn test_kind_default_semantic() {
        let db = test_db();
        let mem = db.insert(MemoryInput::new("the sky is blue")).unwrap();
        assert_eq!(mem.kind, "semantic");
        assert!((mem.decay_rate - Layer::Buffer.default_decay()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_kind_in_output() {
        let db = test_db();
        db.insert(MemoryInput::new("episodic event happened").kind("episodic")).unwrap();
        db.insert(MemoryInput::new("procedural steps to follow").kind("procedural")).unwrap();
        db.insert(MemoryInput::new("semantic fact")).unwrap();

        let all = db.list_all(100, 0).unwrap();
        assert_eq!(all.len(), 3);
        let kinds: Vec<&str> = all.iter().map(|m| m.kind.as_str()).collect();
        assert!(kinds.contains(&"episodic"));
        assert!(kinds.contains(&"procedural"));
        assert!(kinds.contains(&"semantic"));
    }

    #[test]
    fn soft_delete_and_restore() {
        let db = test_db();
        let mem = db.insert(MemoryInput::new("important fact").importance(0.8)).unwrap();
        let id = mem.id.clone();

        // Delete → goes to trash
        assert!(db.delete(&id).unwrap());
        assert!(db.get(&id).unwrap().is_none());
        assert_eq!(db.trash_count().unwrap(), 1);

        let trash = db.trash_list(10, 0).unwrap();
        assert_eq!(trash.len(), 1);
        assert_eq!(trash[0].content, "important fact");
        assert!(trash[0].importance >= 0.8);

        // Restore → back in memories
        assert!(db.trash_restore(&id).unwrap());
        assert!(db.get(&id).unwrap().is_some());
        assert_eq!(db.trash_count().unwrap(), 0);

        // Purge
        db.delete(&id).unwrap();
        assert_eq!(db.trash_count().unwrap(), 1);
        let purged = db.trash_purge().unwrap();
        assert_eq!(purged, 1);
        assert_eq!(db.trash_count().unwrap(), 0);
    }
}
