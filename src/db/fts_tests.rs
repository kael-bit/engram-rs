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

    let results = db.search_fts("quick fox", 10).unwrap();
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
    assert!(!db.search_fts("天气", 10).unwrap().is_empty(), "天气 should match");
    assert!(!db.search_fts("散步", 10).unwrap().is_empty(), "散步 should match");
    assert!(!db.search_fts("出门散步", 10).unwrap().is_empty(), "出门散步 should match");
}

#[test]
fn stopwords_filtered_from_fts_query() {
    assert!(is_stopword("是"));
    assert!(is_stopword("的"));
    assert!(is_stopword("the"));
    assert!(!is_stopword("alice"));
    assert!(!is_stopword("engram"));
    assert!(!is_stopword("部署"));
}

#[test]
fn fts_stopword_only_query_returns_empty() {
    let db = test_db();
    db.insert(MemoryInput {
        content: "some test content for stop words".into(),
        ..Default::default()
    }).unwrap();

    // Pure stop word query should return empty
    let results = db.search_fts("是的了", 10).unwrap();
    assert!(results.is_empty(), "stop-word-only query should return empty");
}

#[test]
fn fts_mixed_latin_cjk_boundary_split() {
    let db = test_db();
    db.insert(MemoryInput {
        content: "alice is my colleague".into(),
        ..Default::default()
    }).unwrap();

    // "alice是谁" should split into "alice" + "是" + "谁", stop words filtered,
    // leaving "alice" — which should match the indexed content.
    let results = db.search_fts("alice是谁", 10).unwrap();
    assert!(!results.is_empty(), "alice是谁 should find content containing alice");
}
