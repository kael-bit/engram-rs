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
            .map(|s| s.trim())
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
        let m2 = db.insert(MemoryInput::new("alice lives in Beijing")).unwrap();

        let (shanghai, _) = db.insert_fact(FactInput {
            subject: "alice".into(),
            predicate: "location".into(),
            object: "Portland".into(),
            memory_id: Some(m1.id.clone()),
            valid_from: None,
        }, "default").unwrap();

        let (beijing, superseded) = db.insert_fact(FactInput {
            subject: "alice".into(),
            predicate: "location".into(),
            object: "Tokyo".into(),
            memory_id: Some(m2.id.clone()),
            valid_from: None,
        }, "default").unwrap();

        // Portland should have been auto-superseded
        assert_eq!(superseded.len(), 1);
        assert_eq!(superseded[0].id, shanghai.id);
        assert!(superseded[0].valid_until.is_some());
        assert_eq!(superseded[0].superseded_by.as_deref(), Some(beijing.id.as_str()));

        // get_conflicts returns all (including superseded), but only Beijing is active
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
            object: "Asia/Portland".into(),
            memory_id: Some(mem.id.clone()),
            valid_from: None,
        }, "default").unwrap();

        // Should match regardless of case
        let r1 = db.query_facts("alice", "default", false).unwrap();
        assert_eq!(r1.len(), 1, "'alice' should match 'Alice'");

        let r2 = db.query_facts("ALICE", "default", false).unwrap();
        assert_eq!(r2.len(), 1, "'ALICE' should match 'Alice'");

        let r3 = db.query_facts("asia/shanghai", "default", false).unwrap();
        assert_eq!(r3.len(), 1, "'asia/shanghai' should match 'Asia/Portland'");

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
}
