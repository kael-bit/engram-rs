use engram::db::*;
use engram::db::fts::extract_query_terms;

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

    assert!(!db.search_fts("天气", 10).unwrap().is_empty(), "天气 should match");
    assert!(!db.search_fts("散步", 10).unwrap().is_empty(), "散步 should match");
    assert!(!db.search_fts("出门散步", 10).unwrap().is_empty(), "出门散步 should match");
}

#[test]
fn dynamic_noise_filtering() {
    let db = test_db();
    // 10 docs all contain "common", only 1 contains "rare"
    for i in 0..10 {
        let content = if i == 0 {
            "common rare special".to_string()
        } else {
            format!("common word number {i}")
        };
        db.insert(MemoryInput {
            content,
            ..Default::default()
        }).unwrap();
    }

    // "common" at 100% df → filtered as noise; "rare" at 10% df → kept
    let (terms, _) = extract_query_terms("common rare", &db);
    assert!(terms.contains(&"rare".to_string()), "rare term should be kept");
    assert!(!terms.contains(&"common".to_string()), "common term should be filtered");
}

#[test]
fn noise_filter_keeps_rare_short_terms() {
    let db = test_db();
    // "zq" appears in 1/10 docs → rare, should be kept despite being short
    for i in 0..10 {
        let content = if i == 0 {
            "zq appears only here".to_string()
        } else {
            format!("filler content for document {i}")
        };
        db.insert(MemoryInput {
            content,
            ..Default::default()
        }).unwrap();
    }

    let (terms, _) = extract_query_terms("zq test", &db);
    assert!(terms.contains(&"zq".to_string()), "rare short term should be kept");
}

#[test]
fn noise_only_query_returns_empty() {
    let db = test_db();
    for i in 0..10 {
        db.insert(MemoryInput {
            content: format!("common repeated word here doc {i}"),
            ..Default::default()
        }).unwrap();
    }

    // "common" at 100% df → filtered → empty query → empty results
    let results = db.search_fts("common", 10).unwrap();
    assert!(results.is_empty(), "noise-only query should return empty");
}

#[test]
fn fts_mixed_latin_cjk_boundary_split() {
    let db = test_db();
    db.insert(MemoryInput {
        content: "alice is my colleague".into(),
        ..Default::default()
    }).unwrap();

    let results = db.search_fts("alice是谁", 10).unwrap();
    assert!(!results.is_empty(), "alice是谁 should find content containing alice");
}

#[test]
fn small_corpus_skips_df_filtering() {
    let db = test_db();
    // Only 2 docs — too few for meaningful df stats, should skip filtering
    db.insert(MemoryInput {
        content: "alice likes coffee".into(),
        ..Default::default()
    }).unwrap();
    db.insert(MemoryInput {
        content: "alice prefers tea".into(),
        ..Default::default()
    }).unwrap();

    // "alice" is in 2/2 docs (100%) but corpus <5 → no filtering
    let results = db.search_fts("alice", 10).unwrap();
    assert!(!results.is_empty(), "small corpus should not filter terms");
}
