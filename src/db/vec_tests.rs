
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
