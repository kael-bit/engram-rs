use super::*;

fn test_db() -> MemoryDB {
    MemoryDB::open(":memory:").expect("in-memory db")
}

#[test]
fn test_fact_crud() {
    let db = test_db();
    let mem = db.insert(MemoryInput::new("alice lives in Westville")).unwrap();

    let (fact, _) = db.insert_fact(FactInput {
        subject: "alice".into(),
        predicate: "location".into(),
        object: "Westville".into(),
        memory_id: Some(mem.id.clone()),
        valid_from: None,
    }, "default").unwrap();

    assert_eq!(fact.subject, "alice");
    assert_eq!(fact.predicate, "location");
    assert_eq!(fact.object, "Westville");
    assert_eq!(fact.memory_id, mem.id);
    assert_eq!(fact.namespace, "default");

    // query by subject
    let results = db.query_facts("alice", "default", false).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, fact.id);

    // query by object
    let results = db.query_facts("Westville", "default", false).unwrap();
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
    let m1 = db.insert(MemoryInput::new("alice lives in Westville")).unwrap();
    let m2 = db.insert(MemoryInput::new("alice lives in Eastdale")).unwrap();

    let (portland, _) = db.insert_fact(FactInput {
        subject: "alice".into(),
        predicate: "location".into(),
        object: "Westville".into(),
        memory_id: Some(m1.id.clone()),
        valid_from: None,
    }, "default").unwrap();

    let (tokyo, superseded) = db.insert_fact(FactInput {
        subject: "alice".into(),
        predicate: "location".into(),
        object: "Eastdale".into(),
        memory_id: Some(m2.id.clone()),
        valid_from: None,
    }, "default").unwrap();

    // Westville should have been auto-superseded
    assert_eq!(superseded.len(), 1);
    assert_eq!(superseded[0].id, portland.id);
    assert!(superseded[0].valid_until.is_some());
    assert_eq!(superseded[0].superseded_by.as_deref(), Some(tokyo.id.as_str()));

    // get_conflicts returns all (including superseded), but only Eastdale is active
    let all = db.get_conflicts("alice", "location", "default").unwrap();
    assert_eq!(all.len(), 2);
    let active: Vec<_> = all.iter().filter(|f| f.valid_until.is_none()).collect();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].object, "Eastdale");
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
        object: "Westville".into(),
        memory_id: None,
        valid_from: None,
    }, "default").unwrap();

    let (new, superseded) = db.insert_fact(FactInput {
        subject: "alice".into(),
        predicate: "city".into(),
        object: "Eastdale".into(),
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
    // Acme → located_in → Eastdale
    db.insert_fact(FactInput {
        subject: "Acme".into(),
        predicate: "located_in".into(),
        object: "Eastdale".into(),
        memory_id: None, valid_from: None,
    }, "default").unwrap();

    let chains = db.query_multihop("alice", 2, "default").unwrap();
    // Should have: [alice→Acme] and [alice→Acme, Acme→Eastdale]
    assert_eq!(chains.len(), 2);

    let single: Vec<_> = chains.iter().filter(|c| c.path.len() == 1).collect();
    let double: Vec<_> = chains.iter().filter(|c| c.path.len() == 2).collect();
    assert_eq!(single.len(), 1);
    assert_eq!(double.len(), 1);

    assert_eq!(double[0].path[0].object, "Acme");
    assert_eq!(double[0].path[1].subject, "Acme");
    assert_eq!(double[0].path[1].object, "Eastdale");
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
    // alice → city → Westville (will be superseded)
    db.insert_fact(FactInput {
        subject: "alice".into(),
        predicate: "city".into(),
        object: "Westville".into(),
        memory_id: None, valid_from: None,
    }, "default").unwrap();
    // alice → city → Eastdale (supersedes Westville)
    db.insert_fact(FactInput {
        subject: "alice".into(),
        predicate: "city".into(),
        object: "Eastdale".into(),
        memory_id: None, valid_from: None,
    }, "default").unwrap();

    let chains = db.query_multihop("alice", 1, "default").unwrap();
    assert_eq!(chains.len(), 1);
    assert_eq!(chains[0].path[0].object, "Eastdale",
        "should only follow active (non-superseded) facts");
}

#[test]
fn test_multihop_empty_result() {
    let db = test_db();
    let chains = db.query_multihop("nonexistent", 2, "default").unwrap();
    assert!(chains.is_empty());
}
