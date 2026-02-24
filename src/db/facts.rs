//! Fact triple storage and querying.

use rusqlite::params;
use uuid::Uuid;

use super::*;

impl MemoryDB {
    /// Insert a single fact triple linked to a memory.
    /// If `memory_id` is not set on the input, a placeholder memory is created automatically.
    /// Automatically supersedes conflicting facts with the same subject+predicate but different object.
    pub fn insert_fact(&self, input: FactInput, namespace: &str) -> Result<(Fact, Vec<Fact>), EngramError> {
        validate_fact_input(&input)?;
        let memory_id = match input.memory_id {
            Some(ref mid) if !mid.is_empty() => mid.clone(),
            _ => {
                let content = format!("{} {} {}", input.subject, input.predicate, input.object);
                let mem = self.insert(
                    MemoryInput::new(content)
                        .source("fact")
                        .namespace(namespace)
                        .skip_dedup(),
                )?;
                mem.id
            }
        };
        let id = Uuid::new_v4().to_string();
        let now = now_ms();
        let valid_from = input.valid_from.unwrap_or(now);
        self.conn()?.execute(
            "INSERT INTO facts (id, subject, predicate, object, memory_id, namespace, created_at, valid_from) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![id, input.subject, input.predicate, input.object, memory_id, namespace, now, valid_from],
        )?;

        // Auto-resolve: supersede older facts with same subject+predicate but different object
        let existing = self.get_conflicts(&input.subject, &input.predicate, namespace)?;
        let mut superseded = Vec::new();
        let conn = self.conn()?;
        for old in existing {
            if old.id != id && old.object != input.object && old.valid_until.is_none() {
                conn.execute(
                    "UPDATE facts SET valid_until = ?1, superseded_by = ?2 WHERE id = ?3",
                    params![now, id, old.id],
                )?;
                superseded.push(Fact { valid_until: Some(now), superseded_by: Some(id.clone()), ..old });
            }
        }

        Ok((Fact {
            id,
            subject: input.subject,
            predicate: input.predicate,
            object: input.object,
            memory_id,
            namespace: namespace.to_string(),
            created_at: now,
            valid_from: Some(valid_from),
            valid_until: None,
            superseded_by: None,
        }, superseded))
    }

    /// Batch insert multiple facts. Returns inserted facts and all superseded facts.
    pub fn insert_facts(&self, facts: Vec<FactInput>, namespace: &str) -> Result<(Vec<Fact>, Vec<Fact>), EngramError> {
        let mut inserted = Vec::with_capacity(facts.len());
        let mut all_superseded = Vec::new();
        for input in facts {
            let (fact, superseded) = self.insert_fact(input, namespace)?;
            inserted.push(fact);
            all_superseded.extend(superseded);
        }
        Ok((inserted, all_superseded))
    }

    /// Query facts where subject or object matches the entity (case-insensitive).
    /// The query string is split into tokens, and any token matching a subject or object
    /// will return the fact — so "alice 的时区" matches subject="alice".
    /// By default only returns active facts (valid_until IS NULL).
    pub fn query_facts(&self, entity: &str, namespace: &str, include_superseded: bool) -> Result<Vec<Fact>, EngramError> {
        let tokens: Vec<&str> = entity.split(|c: char| c.is_whitespace() || c == '的' || c == '，' || c == ',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if tokens.is_empty() {
            return Ok(vec![]);
        }

        let mut all_facts: Vec<Fact> = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();
        let conn = self.conn()?;

        let active_filter = if include_superseded { "" } else { " AND valid_until IS NULL" };

        for token in &tokens {
            let pattern = format!("%{token}%");
            let sql = format!(
                "SELECT id, subject, predicate, object, memory_id, namespace, created_at, \
                        valid_from, valid_until, superseded_by \
                 FROM facts \
                 WHERE namespace = ?1 \
                   AND (LOWER(subject) LIKE LOWER(?2) OR LOWER(object) LIKE LOWER(?2)){active_filter}"
            );
            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map(params![namespace, pattern], |row| {
                Ok(Fact {
                    id: row.get(0)?,
                    subject: row.get(1)?,
                    predicate: row.get(2)?,
                    object: row.get(3)?,
                    memory_id: row.get(4)?,
                    namespace: row.get(5)?,
                    created_at: row.get(6)?,
                    valid_from: row.get(7)?,
                    valid_until: row.get(8)?,
                    superseded_by: row.get(9)?,
                })
            })?;
            for fact in rows.flatten() {
                if seen_ids.insert(fact.id.clone()) {
                    all_facts.push(fact);
                }
            }
        }
        Ok(all_facts)
    }

    /// Find active facts with the same subject+predicate (case-insensitive) for conflict detection.
    pub fn get_conflicts(&self, subject: &str, predicate: &str, namespace: &str) -> Result<Vec<Fact>, EngramError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, subject, predicate, object, memory_id, namespace, created_at, \
                    valid_from, valid_until, superseded_by \
             FROM facts \
             WHERE LOWER(subject) = LOWER(?1) AND LOWER(predicate) = LOWER(?2) AND namespace = ?3"
        )?;
        let rows = stmt.query_map(params![subject, predicate, namespace], |row| {
            Ok(Fact {
                id: row.get(0)?,
                subject: row.get(1)?,
                predicate: row.get(2)?,
                object: row.get(3)?,
                memory_id: row.get(4)?,
                namespace: row.get(5)?,
                created_at: row.get(6)?,
                valid_from: row.get(7)?,
                valid_until: row.get(8)?,
                superseded_by: row.get(9)?,
            })
        })?;
        Ok(rows.flatten().collect())
    }

    /// Delete a single fact by id. Returns true if it existed.
    pub fn delete_fact(&self, id: &str) -> Result<bool, EngramError> {
        let n = self.conn()?.execute("DELETE FROM facts WHERE id = ?1", params![id])?;
        Ok(n > 0)
    }

    /// Delete all facts linked to a given memory_id. Returns number of facts removed.
    pub fn delete_facts_by_memory(&self, memory_id: &str) -> Result<usize, EngramError> {
        let n = self.conn()?.execute("DELETE FROM facts WHERE memory_id = ?1", params![memory_id])?;
        Ok(n)
    }

    /// Get the full history of a subject+predicate pair, including superseded facts.
    /// Results are ordered by created_at ascending (oldest first).
    pub fn get_fact_history(&self, subject: &str, predicate: &str, namespace: &str) -> Result<Vec<Fact>, EngramError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, subject, predicate, object, memory_id, namespace, created_at, \
                    valid_from, valid_until, superseded_by \
             FROM facts \
             WHERE LOWER(subject) = LOWER(?1) AND LOWER(predicate) = LOWER(?2) AND namespace = ?3 \
             ORDER BY created_at ASC"
        )?;
        let rows = stmt.query_map(params![subject, predicate, namespace], |row| {
            Ok(Fact {
                id: row.get(0)?,
                subject: row.get(1)?,
                predicate: row.get(2)?,
                object: row.get(3)?,
                memory_id: row.get(4)?,
                namespace: row.get(5)?,
                created_at: row.get(6)?,
                valid_from: row.get(7)?,
                valid_until: row.get(8)?,
                superseded_by: row.get(9)?,
            })
        })?;
        Ok(rows.flatten().collect())
    }

    /// List all active facts in a namespace with pagination.
    pub fn list_facts(&self, namespace: &str, limit: usize, offset: usize) -> Result<Vec<Fact>, EngramError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, subject, predicate, object, memory_id, namespace, created_at, \
                    valid_from, valid_until, superseded_by \
             FROM facts WHERE namespace = ?1 AND valid_until IS NULL \
             ORDER BY created_at DESC LIMIT ?2 OFFSET ?3"
        )?;
        let rows = stmt.query_map(params![namespace, limit as i64, offset as i64], |row| {
            Ok(Fact {
                id: row.get(0)?,
                subject: row.get(1)?,
                predicate: row.get(2)?,
                object: row.get(3)?,
                memory_id: row.get(4)?,
                namespace: row.get(5)?,
                created_at: row.get(6)?,
                valid_from: row.get(7)?,
                valid_until: row.get(8)?,
                superseded_by: row.get(9)?,
            })
        })?;
        Ok(rows.flatten().collect())
    }

    /// Walk the fact graph starting from `entity`, up to `hops` levels deep.
    ///
    /// At each level, finds all active facts where the current entity is the subject,
    /// then follows each object as the next entity. Tracks visited entities to avoid
    /// cycles. Returns all discovered chains as `FactChain`s.
    pub fn query_multihop(&self, entity: &str, hops: usize, ns: &str) -> Result<Vec<FactChain>, EngramError> {
        let hops = hops.clamp(1, 3);
        let conn = self.conn()?;

        let mut stmt = conn.prepare(
            "SELECT subject, predicate, object FROM facts \
             WHERE LOWER(subject) = LOWER(?1) AND namespace = ?2 AND valid_until IS NULL"
        )?;

        // BFS-style traversal: (current_entity, path_so_far)
        let mut queue: Vec<(String, Vec<FactTriple>)> = vec![(entity.to_string(), vec![])];
        let mut chains: Vec<FactChain> = Vec::new();
        let mut visited = std::collections::HashSet::new();
        visited.insert(entity.to_lowercase());

        for _depth in 0..hops {
            let mut next_queue = Vec::new();

            for (current, path) in &queue {
                let rows = stmt.query_map(rusqlite::params![current, ns], |row| {
                    Ok(FactTriple {
                        subject: row.get(0)?,
                        predicate: row.get(1)?,
                        object: row.get(2)?,
                    })
                })?;

                for triple in rows.flatten() {
                    let mut new_path = path.clone();
                    let obj_lower = triple.object.to_lowercase();
                    new_path.push(triple);
                    chains.push(FactChain { path: new_path.clone() });

                    // Only follow this object if we haven't visited it yet
                    if visited.insert(obj_lower) {
                        let next_entity = new_path.last().unwrap().object.clone();
                        next_queue.push((next_entity, new_path));
                    }
                }
            }

            if next_queue.is_empty() {
                break;
            }
            queue = next_queue;
        }

        Ok(chains)
    }

    /// Return Working/Core memories that have no linked facts yet.
    pub fn memories_without_facts(&self, namespace: &str, limit: usize) -> Result<Vec<Memory>, EngramError> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT m.id, m.content, m.layer, m.importance, m.created_at, m.last_accessed, \
                    m.access_count, m.decay_rate, m.source, m.tags, m.namespace, \
                    m.repetition_count, m.kind \
             FROM memories m \
             WHERE m.namespace = ?1 AND m.layer >= 2 \
               AND NOT EXISTS (SELECT 1 FROM facts f WHERE f.memory_id = m.id) \
             ORDER BY m.layer DESC, m.importance DESC \
             LIMIT ?2"
        )?;
        let rows = stmt.query_map(params![namespace, limit as i64], |row| {
            Ok(Memory {
                id: row.get(0)?,
                content: row.get(1)?,
                layer: match row.get::<_, i32>(2)? {
                    1 => Layer::Buffer,
                    3 => Layer::Core,
                    _ => Layer::Working,
                },
                importance: row.get(3)?,
                created_at: row.get(4)?,
                last_accessed: row.get(5)?,
                access_count: row.get(6)?,
                decay_rate: row.get(7)?,
                source: row.get::<_, Option<String>>(8)?.unwrap_or_default(),
                tags: {
                    let raw: String = row.get::<_, Option<String>>(9)?.unwrap_or_default();
                    if raw.is_empty() { vec![] } else { raw.split(',').map(std::string::ToString::to_string).collect() }
                },
                namespace: row.get::<_, Option<String>>(10)?.unwrap_or_else(|| "default".into()),
                repetition_count: row.get::<_, Option<i64>>(11)?.unwrap_or(0),
                kind: row.get::<_, Option<String>>(12)?.unwrap_or_else(|| "semantic".into()),
                embedding: None,
                risk_score: 0.0,
            })
        })?;
        Ok(rows.flatten().collect())
    }

}


#[cfg(test)]
#[path = "facts_tests.rs"]
mod tests;
