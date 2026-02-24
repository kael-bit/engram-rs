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
mod tests {
    use super::*;

    fn test_db() -> MemoryDB {
        MemoryDB::open(":memory:").expect("in-memory db")
    }

    #[test]
    fn test_fact_crud() {
        let db = test_db();
        let mem = db.insert(MemoryInput::new("alice lives in Portland")).unwrap();

        let (fact, _) = db.insert_fact(FactInput {
            subject: "alice".into(),
            predicate: "location".into(),
            object: "Portland".into(),
            memory_id: Some(mem.id.clone()),
            valid_from: None,
        }, "default").unwrap();

        assert_eq!(fact.subject, "alice");
        assert_eq!(fact.predicate, "location");
        assert_eq!(fact.object, "Portland");
        assert_eq!(fact.memory_id, mem.id);
        assert_eq!(fact.namespace, "default");

        // query by subject
        let results = db.query_facts("alice", "default", false).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, fact.id);

        // query by object
        let results = db.query_facts("Portland", "default", false).unwrap();
        assert_eq!(results.len(), 1);

        // list all
        let all = db.list_facts("default", 50, 0).unwrap();
        assert_eq!(all.len(), 1);

        // delete
        assert!(db.delete_fact(&fact.id).unwrap());
        assert!(!db.delete_fact(&fact.id).unwrap()); // already gone
        assert!(db.list_facts("default", 50, 0).unwrap().is_empty());
    }

    #[test]
    fn test_fact_conflict_detection() {
        let db = test_db();
        let m1 = db.insert(MemoryInput::new("alice lives in Portland")).unwrap();
        let m2 = db.insert(MemoryInput::new("alice lives in Tokyo")).unwrap();

        let (portland, _) = db.insert_fact(FactInput {
            subject: "alice".into(),
            predicate: "location".into(),
            object: "Portland".into(),
            memory_id: Some(m1.id.clone()),
            valid_from: None,
        }, "default").unwrap();

        let (tokyo, superseded) = db.insert_fact(FactInput {
            subject: "alice".into(),
            predicate: "location".into(),
            object: "Tokyo".into(),
            memory_id: Some(m2.id.clone()),
            valid_from: None,
        }, "default").unwrap();

        // Portland should have been auto-superseded
        assert_eq!(superseded.len(), 1);
        assert_eq!(superseded[0].id, portland.id);
        assert!(superseded[0].valid_until.is_some());
        assert_eq!(superseded[0].superseded_by.as_deref(), Some(tokyo.id.as_str()));

        // get_conflicts returns all (including superseded), but only Tokyo is active
        let all = db.get_conflicts("alice", "location", "default").unwrap();
        assert_eq!(all.len(), 2);
        let active: Vec<_> = all.iter().filter(|f| f.valid_until.is_none()).collect();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].object, "Tokyo");
    }

    #[test]
    fn test_fact_cascade_delete() {
        let db = test_db();
        let mem = db.insert(MemoryInput::new("alice uses dark mode")).unwrap();

        db.insert_fact(FactInput {
            subject: "alice".into(),
            predicate: "theme".into(),
            object: "dark".into(),
            memory_id: Some(mem.id.clone()),
            valid_from: None,
        }, "default").unwrap();

        db.insert_fact(FactInput {
            subject: "alice".into(),
            predicate: "editor".into(),
            object: "neovim".into(),
            memory_id: Some(mem.id.clone()),
            valid_from: None,
        }, "default").unwrap();

        assert_eq!(db.list_facts("default", 50, 0).unwrap().len(), 2);

        // Deleting the memory should cascade-delete associated facts
        db.delete(&mem.id).unwrap();
        assert!(db.list_facts("default", 50, 0).unwrap().is_empty());
    }

    #[test]
    fn test_fact_query_case_insensitive() {
        let db = test_db();
        let mem = db.insert(MemoryInput::new("Alice's timezone")).unwrap();

        db.insert_fact(FactInput {
            subject: "Alice".into(),
            predicate: "timezone".into(),
            object: "America/New_York".into(),
            memory_id: Some(mem.id.clone()),
            valid_from: None,
        }, "default").unwrap();

        // Should match regardless of case
        let r1 = db.query_facts("alice", "default", false).unwrap();
        assert_eq!(r1.len(), 1, "'alice' should match 'Alice'");

        let r2 = db.query_facts("ALICE", "default", false).unwrap();
        assert_eq!(r2.len(), 1, "'ALICE' should match 'Alice'");

        let r3 = db.query_facts("america/new_york", "default", false).unwrap();
        assert_eq!(r3.len(), 1, "'america/new_york' should match 'America/New_York'");

        // Conflict detection is also case-insensitive
        let conflicts = db.get_conflicts("alice", "Timezone", "default").unwrap();
        assert_eq!(conflicts.len(), 1);
    }

    #[test]
    fn test_fact_temporal_supersede() {
        let db = test_db();

        let (old, _) = db.insert_fact(FactInput {
            subject: "alice".into(),
            predicate: "city".into(),
            object: "Portland".into(),
            memory_id: None,
            valid_from: None,
        }, "default").unwrap();

        let (new, superseded) = db.insert_fact(FactInput {
            subject: "alice".into(),
            predicate: "city".into(),
            object: "Tokyo".into(),
            memory_id: None,
            valid_from: None,
        }, "default").unwrap();

        assert_eq!(superseded.len(), 1);
        assert_eq!(superseded[0].id, old.id);
        assert!(superseded[0].valid_until.is_some());
        assert_eq!(superseded[0].superseded_by.as_deref(), Some(new.id.as_str()));

        // The new fact should be active
        assert!(new.valid_until.is_none());
        assert!(new.superseded_by.is_none());
    }

    #[test]
    fn test_fact_query_excludes_superseded() {
        let db = test_db();

        db.insert_fact(FactInput {
            subject: "alice".into(),
            predicate: "lang".into(),
            object: "Python".into(),
            memory_id: None,
            valid_from: None,
        }, "default").unwrap();

        db.insert_fact(FactInput {
            subject: "alice".into(),
            predicate: "lang".into(),
            object: "Rust".into(),
            memory_id: None,
            valid_from: None,
        }, "default").unwrap();

        let active = db.query_facts("alice", "default", false).unwrap();
        let lang_facts: Vec<_> = active.iter().filter(|f| f.predicate == "lang").collect();
        assert_eq!(lang_facts.len(), 1);
        assert_eq!(lang_facts[0].object, "Rust");
    }

    #[test]
    fn test_fact_query_includes_superseded() {
        let db = test_db();

        db.insert_fact(FactInput {
            subject: "alice".into(),
            predicate: "lang".into(),
            object: "Python".into(),
            memory_id: None,
            valid_from: None,
        }, "default").unwrap();

        db.insert_fact(FactInput {
            subject: "alice".into(),
            predicate: "lang".into(),
            object: "Rust".into(),
            memory_id: None,
            valid_from: None,
        }, "default").unwrap();

        let all = db.query_facts("alice", "default", true).unwrap();
        let lang_facts: Vec<_> = all.iter().filter(|f| f.predicate == "lang").collect();
        assert_eq!(lang_facts.len(), 2);
        let objects: Vec<&str> = lang_facts.iter().map(|f| f.object.as_str()).collect();
        assert!(objects.contains(&"Python"));
        assert!(objects.contains(&"Rust"));
    }

    #[test]
    fn test_fact_history_ordering() {
        let db = test_db();

        db.insert_fact(FactInput {
            subject: "alice".into(),
            predicate: "mood".into(),
            object: "happy".into(),
            memory_id: None,
            valid_from: None,
        }, "default").unwrap();

        std::thread::sleep(std::time::Duration::from_millis(10));

        db.insert_fact(FactInput {
            subject: "alice".into(),
            predicate: "mood".into(),
            object: "tired".into(),
            memory_id: None,
            valid_from: None,
        }, "default").unwrap();

        std::thread::sleep(std::time::Duration::from_millis(10));

        db.insert_fact(FactInput {
            subject: "alice".into(),
            predicate: "mood".into(),
            object: "excited".into(),
            memory_id: None,
            valid_from: None,
        }, "default").unwrap();

        let history = db.get_fact_history("alice", "mood", "default").unwrap();
        assert_eq!(history.len(), 3);
        // Oldest first
        assert_eq!(history[0].object, "happy");
        assert_eq!(history[1].object, "tired");
        assert_eq!(history[2].object, "excited");
        // First two should be superseded
        assert!(history[0].valid_until.is_some());
        assert!(history[1].valid_until.is_some());
        assert!(history[2].valid_until.is_none());
        // Chronological order
        assert!(history[0].created_at <= history[1].created_at);
        assert!(history[1].created_at <= history[2].created_at);
    }

    // --- multi-hop tests ---

    #[test]
    fn test_multihop_single_hop() {
        let db = test_db();
        db.insert_fact(FactInput {
            subject: "alice".into(),
            predicate: "works_at".into(),
            object: "Acme".into(),
            memory_id: None, valid_from: None,
        }, "default").unwrap();

        let chains = db.query_multihop("alice", 1, "default").unwrap();
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].path.len(), 1);
        assert_eq!(chains[0].path[0].subject, "alice");
        assert_eq!(chains[0].path[0].object, "Acme");
    }

    #[test]
    fn test_multihop_two_hops() {
        let db = test_db();
        // alice → works_at → Acme
        db.insert_fact(FactInput {
            subject: "alice".into(),
            predicate: "works_at".into(),
            object: "Acme".into(),
            memory_id: None, valid_from: None,
        }, "default").unwrap();
        // Acme → located_in → Tokyo
        db.insert_fact(FactInput {
            subject: "Acme".into(),
            predicate: "located_in".into(),
            object: "Tokyo".into(),
            memory_id: None, valid_from: None,
        }, "default").unwrap();

        let chains = db.query_multihop("alice", 2, "default").unwrap();
        // Should have: [alice→Acme] and [alice→Acme, Acme→Tokyo]
        assert_eq!(chains.len(), 2);

        let single: Vec<_> = chains.iter().filter(|c| c.path.len() == 1).collect();
        let double: Vec<_> = chains.iter().filter(|c| c.path.len() == 2).collect();
        assert_eq!(single.len(), 1);
        assert_eq!(double.len(), 1);

        assert_eq!(double[0].path[0].object, "Acme");
        assert_eq!(double[0].path[1].subject, "Acme");
        assert_eq!(double[0].path[1].object, "Tokyo");
    }

    #[test]
    fn test_multihop_cycle_detection() {
        let db = test_db();
        // A → likes → B
        db.insert_fact(FactInput {
            subject: "A".into(),
            predicate: "likes".into(),
            object: "B".into(),
            memory_id: None, valid_from: None,
        }, "default").unwrap();
        // B → likes → A (cycle!)
        db.insert_fact(FactInput {
            subject: "B".into(),
            predicate: "likes".into(),
            object: "A".into(),
            memory_id: None, valid_from: None,
        }, "default").unwrap();

        // Should not infinite loop; cycle is broken by visited set
        let chains = db.query_multihop("A", 3, "default").unwrap();
        // hop 1: [A→B], hop 2: [A→B, B→A] — but A is already visited so no hop 3
        assert_eq!(chains.len(), 2);
        // No chain should be longer than 2 since the cycle is detected
        assert!(chains.iter().all(|c| c.path.len() <= 2));
    }

    #[test]
    fn test_multihop_namespace_isolation() {
        let db = test_db();
        db.insert_fact(FactInput {
            subject: "alice".into(),
            predicate: "role".into(),
            object: "engineer".into(),
            memory_id: None, valid_from: None,
        }, "ns-a").unwrap();

        db.insert_fact(FactInput {
            subject: "alice".into(),
            predicate: "role".into(),
            object: "designer".into(),
            memory_id: None, valid_from: None,
        }, "ns-b").unwrap();

        let chains_a = db.query_multihop("alice", 2, "ns-a").unwrap();
        assert_eq!(chains_a.len(), 1);
        assert_eq!(chains_a[0].path[0].object, "engineer");

        let chains_b = db.query_multihop("alice", 2, "ns-b").unwrap();
        assert_eq!(chains_b.len(), 1);
        assert_eq!(chains_b[0].path[0].object, "designer");
    }

    #[test]
    fn test_multihop_max_hops_respected() {
        let db = test_db();
        // Build a chain: A → B → C → D → E
        for (s, o) in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")] {
            db.insert_fact(FactInput {
                subject: s.into(),
                predicate: "next".into(),
                object: o.into(),
                memory_id: None, valid_from: None,
            }, "default").unwrap();
        }

        // With hops=2, longest chain should be 2 steps (A→B→C)
        let chains_2 = db.query_multihop("A", 2, "default").unwrap();
        assert!(chains_2.iter().all(|c| c.path.len() <= 2),
            "no chain should exceed 2 hops");
        let max_depth = chains_2.iter().map(|c| c.path.len()).max().unwrap_or(0);
        assert_eq!(max_depth, 2);

        // With hops=3 (max allowed), longest should be 3 steps (A→B→C→D)
        let chains_3 = db.query_multihop("A", 3, "default").unwrap();
        assert!(chains_3.iter().all(|c| c.path.len() <= 3),
            "no chain should exceed 3 hops");
        let max_depth_3 = chains_3.iter().map(|c| c.path.len()).max().unwrap_or(0);
        assert_eq!(max_depth_3, 3);

        // Requesting hops=5 should be clamped to 3
        let chains_5 = db.query_multihop("A", 5, "default").unwrap();
        assert!(chains_5.iter().all(|c| c.path.len() <= 3),
            "hops should be clamped to 3");
    }

    #[test]
    fn test_multihop_skips_superseded() {
        let db = test_db();
        // alice → city → Portland (will be superseded)
        db.insert_fact(FactInput {
            subject: "alice".into(),
            predicate: "city".into(),
            object: "Portland".into(),
            memory_id: None, valid_from: None,
        }, "default").unwrap();
        // alice → city → Tokyo (supersedes Portland)
        db.insert_fact(FactInput {
            subject: "alice".into(),
            predicate: "city".into(),
            object: "Tokyo".into(),
            memory_id: None, valid_from: None,
        }, "default").unwrap();

        let chains = db.query_multihop("alice", 1, "default").unwrap();
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].path[0].object, "Tokyo",
            "should only follow active (non-superseded) facts");
    }

    #[test]
    fn test_multihop_empty_result() {
        let db = test_db();
        let chains = db.query_multihop("nonexistent", 2, "default").unwrap();
        assert!(chains.is_empty());
    }
}
