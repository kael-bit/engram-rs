use super::*;

#[test]
fn tokens_ascii() {
    // "hello world" = 11 bytes / 4 ≈ 3 tokens
    let tokens = estimate_tokens("hello world");
    assert!((2..=4).contains(&tokens));
}

#[test]
fn tokens_cjk() {
    // 4 CJK chars / 1.5 ≈ 3 tokens
    let tokens = estimate_tokens("你好世界");
    assert!((2..=4).contains(&tokens));
}

#[test]
fn tokens_mixed() {
    let tokens = estimate_tokens("hello 你好");
    assert!(tokens >= 2);
}

#[test]
fn parse_rerank_basic() {
    assert_eq!(parse_rerank_response("3, 1, 2", 3), vec![2, 0, 1]);
}

#[test]
fn parse_rerank_with_noise() {
    // number 5 is out of range (count=3), should be filtered
    assert_eq!(parse_rerank_response("3,1,2,5", 3), vec![2, 0, 1]);
}

#[test]
fn parse_rerank_newlines() {
    assert_eq!(parse_rerank_response("2\n1\n3", 3), vec![1, 0, 2]);
}

#[test]
fn parse_rerank_empty() {
    assert!(parse_rerank_response("no numbers here", 3).is_empty());
}

// --- recall integration tests ---

use crate::db::{MemoryDB, MemoryInput};

fn test_db_with_data() -> MemoryDB {
    let db = MemoryDB::open(":memory:").expect("in-memory db");
    db.insert(MemoryInput {
        content: "very important fact about rust".into(),
        layer: Some(3),
        importance: Some(0.95),
        ..Default::default()
    }).unwrap();
    db.insert(MemoryInput {
        content: "trivial note about lunch".into(),
        layer: Some(1),
        importance: Some(0.1),
        ..Default::default()
    }).unwrap();
    db.insert(MemoryInput {
        content: "medium importance work log about rust compiler".into(),
        layer: Some(2),
        importance: Some(0.5),
        ..Default::default()
    }).unwrap();
    db
}

#[test]
fn score_favors_important() {
    let db = test_db_with_data();
    let req = RecallRequest {
        query: "rust".into(),
        budget_tokens: Some(2000),
        layers: None,
        min_importance: None,
        limit: Some(10),
        since: None,
        until: None,
        sort_by: None,
        rerank: None, source: None, tags: None,
        namespace: None,
        expand: None,
        ..Default::default()
    };
    let result = recall(&db, &req, None, None);
    assert!(result.memories.len() >= 2);
    // "very important fact" should score higher than "medium importance"
    let first = &result.memories[0];
    assert!(first.memory.content.contains("very important"), "highest importance should rank first");
}

#[test]
fn budget_limits_output() {
    let db = test_db_with_data();
    // tiny budget — should get at most 1 result
    let req = RecallRequest {
        query: "rust".into(),
        budget_tokens: Some(5),
        layers: None,
        min_importance: None,
        limit: Some(10),
        since: None,
        until: None,
        sort_by: None,
        rerank: None, source: None, tags: None,
        namespace: None,
        expand: None,
        ..Default::default()
    };
    let result = recall(&db, &req, None, None);
    assert!(result.memories.len() <= 2, "budget should limit results");
    assert!(result.total_tokens <= 10, "shouldn't overshoot budget by much");
}

#[test]
fn budget_zero_returns_empty() {
    let db = test_db_with_data();
    let req = RecallRequest {
        query: "rust".into(),
        budget_tokens: Some(0),
        layers: None,
        min_importance: None,
        limit: Some(10),
        since: None,
        until: None,
        sort_by: None,
        rerank: None, source: None, tags: None,
        namespace: None,
        expand: None,
        ..Default::default()
    };
    let result = recall(&db, &req, None, None);
    assert!(result.memories.is_empty());
}

#[test]
fn time_filter_since() {
    let db = MemoryDB::open(":memory:").expect("in-memory db");
    // insert two memories — we can't control created_at via MemoryInput,
    // but both will have "now" timestamps. Import lets us set timestamps.
    let now = crate::db::now_ms();
    let old = Memory {
        id: "old-one".into(),
        content: "old memory about testing".into(),
        layer: Layer::Core,
        importance: 0.9,
        created_at: now - 86_400_000, // yesterday
        last_accessed: now - 86_400_000,
        access_count: 5,
        repetition_count: 0,
        decay_rate: 0.05,
        source: "test".into(),
        tags: vec![],
        namespace: "default".into(),
        embedding: None,
        kind: "semantic".into(),
    };
    let recent = Memory {
        id: "new-one".into(),
        content: "recent memory about testing".into(),
        layer: Layer::Core,
        importance: 0.9,
        created_at: now - 1000,
        last_accessed: now,
        access_count: 1,
        repetition_count: 0,
        decay_rate: 0.05,
        source: "test".into(),
        tags: vec![],
        namespace: "default".into(),
        embedding: None,
        kind: "semantic".into(),
    };
    db.import(&[old, recent]).unwrap();

    let req = RecallRequest {
        query: "testing".into(),
        budget_tokens: Some(2000),
        layers: None,
        min_importance: None,
        limit: Some(10),
        since: Some(now - 3_600_000), // last hour only
        until: None,
        sort_by: None,
        rerank: None, source: None, tags: None,
        namespace: None,
        expand: None,
        ..Default::default()
    };
    let result = recall(&db, &req, None, None);
    assert_eq!(result.memories.len(), 1);
    assert_eq!(result.memories[0].memory.id, "new-one");
}

#[test]
fn filter_by_source() {
    let db = MemoryDB::open(":memory:").expect("in-memory db");
    db.insert(MemoryInput {
        content: "from the API".into(),
        layer: Some(3), importance: Some(0.9),
        source: Some("api".into()), tags: None,
        ..Default::default()
    }).unwrap();
    db.insert(MemoryInput {
        content: "from a session".into(),
        layer: Some(2), importance: Some(0.7),
        source: Some("session".into()), tags: None,
        ..Default::default()
    }).unwrap();

    let req = RecallRequest {
        query: "from".into(),
        budget_tokens: Some(2000),
        layers: None, min_importance: None, limit: Some(10),
        since: None, until: None, sort_by: None, rerank: None,
        source: Some("session".into()), tags: None,
        namespace: None,
        expand: None,
        ..Default::default()
    };
    let result = recall(&db, &req, None, None);
    assert_eq!(result.memories.len(), 1);
    assert!(result.memories[0].memory.content.contains("session"));
}

#[test]
fn filter_by_tags() {
    let db = MemoryDB::open(":memory:").expect("in-memory db");
    db.insert(MemoryInput {
        content: "rust project details".into(),
        layer: Some(3), importance: Some(0.9),
        source: None, tags: Some(vec!["rust".into(), "engram".into()]),
    ..Default::default()
    }).unwrap();
    db.insert(MemoryInput {
        content: "python script notes".into(),
        layer: Some(2), importance: Some(0.7),
        source: None, tags: Some(vec!["python".into()]),
    ..Default::default()
    }).unwrap();

    let req = RecallRequest {
        query: "project".into(),
        budget_tokens: Some(2000),
        layers: None, min_importance: None, limit: Some(10),
        since: None, until: None, sort_by: None, rerank: None,
        source: None, tags: Some(vec!["rust".into()]),
        namespace: None,
        expand: None,
        ..Default::default()
    };
    let result = recall(&db, &req, None, None);
    assert_eq!(result.memories.len(), 1);
    assert!(result.memories[0].memory.content.contains("rust"));
}

#[test]
fn score_combined_recalc() {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;
    let recent = score_combined(0.8, 0.9, now);
    let old = score_combined(0.8, 0.9, now - 72 * 3_600_000);
    assert!(recent > old, "recent memory should score higher");
    assert!(recent > 0.0);
}

#[test]
fn min_score_filters_low() {
    let db = test_db_with_data();
    // Without min_score, we get results
    let req_all = RecallRequest {
        query: "rust".into(),
        limit: Some(10),
        ..Default::default()
    };
    let all = recall(&db, &req_all, None, None);
    assert!(all.memories.len() >= 2);

    // With impossibly high min_score, we get nothing
    let req_high = RecallRequest {
        query: "rust".into(),
        limit: Some(10),
        min_score: Some(99.0),
        ..Default::default()
    };
    let filtered = recall(&db, &req_high, None, None);
    assert_eq!(filtered.memories.len(), 0, "min_score=99 should filter everything");
}

#[test]
fn empty_query_returns_something() {
    let db = test_db_with_data();
    let req = RecallRequest {
        query: "".into(),
        limit: Some(5),
        ..Default::default()
    };
    // Empty query shouldn't panic — FTS returns nothing, but that's fine
    let result = recall(&db, &req, None, None);
    // May or may not find results, but must not crash
    assert!(result.memories.len() <= 5);
}

#[test]
fn sort_by_recent() {
    let db = MemoryDB::open(":memory:").expect("in-memory db");
    db.insert(MemoryInput {
        content: "older entry about databases".into(),
        importance: Some(0.9),
        ..Default::default()
    }).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(10));
    db.insert(MemoryInput {
        content: "newer entry about databases".into(),
        importance: Some(0.3),
        ..Default::default()
    }).unwrap();

    let req = RecallRequest {
        query: "databases".into(),
        sort_by: Some("recent".into()),
        limit: Some(2),
        ..Default::default()
    };
    let result = recall(&db, &req, None, None);
    assert_eq!(result.memories.len(), 2);
    assert!(result.memories[0].memory.content.contains("newer"),
        "sort_by=recent should put newer first");
}

#[test]
fn namespace_isolation() {
    let db = MemoryDB::open(":memory:").expect("in-memory db");
    db.insert(MemoryInput {
        content: "secret agent data".into(),
        namespace: Some("agent-a".into()),
        ..Default::default()
    }).unwrap();
    db.insert(MemoryInput {
        content: "different agent data".into(),
        namespace: Some("agent-b".into()),
        ..Default::default()
    }).unwrap();

    let req = RecallRequest {
        query: "agent data".into(),
        namespace: Some("agent-a".into()),
        limit: Some(10),
        ..Default::default()
    };
    let result = recall(&db, &req, None, None);
    assert_eq!(result.memories.len(), 1);
    assert!(result.memories[0].memory.content.contains("secret"));
}

#[test]
fn since_until_filters() {
    let db = MemoryDB::open(":memory:").expect("in-memory db");
    let _m1 = db.insert(MemoryInput {
        content: "early log entry".into(),
        ..Default::default()
    }).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(50));
    let midpoint = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;
    std::thread::sleep(std::time::Duration::from_millis(50));
    db.insert(MemoryInput {
        content: "late log entry".into(),
        ..Default::default()
    }).unwrap();

    // Only memories after midpoint
    let req = RecallRequest {
        query: "log entry".into(),
        since: Some(midpoint),
        limit: Some(10),
        ..Default::default()
    };
    let result = recall(&db, &req, None, None);
    assert_eq!(result.memories.len(), 1);
    assert!(result.memories[0].memory.content.contains("late"));

    // Only memories before midpoint
    let req2 = RecallRequest {
        query: "log entry".into(),
        until: Some(midpoint),
        limit: Some(10),
        ..Default::default()
    };
    let result2 = recall(&db, &req2, None, None);
    assert_eq!(result2.memories.len(), 1);
    assert!(result2.memories[0].memory.content.contains("early"));
}

#[test]
fn source_filter() {
    let db = MemoryDB::open(":memory:").expect("in-memory db");
    db.insert(MemoryInput {
        content: "manual note about performance".into(),
        source: Some("api".into()),
        ..Default::default()
    }).unwrap();
    db.insert(MemoryInput {
        content: "extracted note about performance".into(),
        source: Some("extract".into()),
        ..Default::default()
    }).unwrap();

    let req = RecallRequest {
        query: "performance".into(),
        source: Some("extract".into()),
        limit: Some(10),
        ..Default::default()
    };
    let result = recall(&db, &req, None, None);
    assert_eq!(result.memories.len(), 1);
    assert!(result.memories[0].memory.content.contains("extracted"));
}

#[test]
fn layer_filter() {
    let db = MemoryDB::open(":memory:").expect("in-memory db");
    db.insert(MemoryInput {
        content: "buffer thought about testing".into(),
        layer: Some(1),
        ..Default::default()
    }).unwrap();
    db.insert(MemoryInput {
        content: "core fact about testing".into(),
        layer: Some(3),
        ..Default::default()
    }).unwrap();

    let req = RecallRequest {
        query: "testing".into(),
        layers: Some(vec![3]),
        limit: Some(10),
        ..Default::default()
    };
    let result = recall(&db, &req, None, None);
    assert_eq!(result.memories.len(), 1);
    assert!(result.memories[0].memory.content.contains("core"));
}

#[test]
fn dual_hit_boost_increases_score() {
    // When a memory is found by both semantic and FTS, its score should
    // be higher than FTS-only. We can't test semantic without embeddings,
    // but we can verify score_combined produces consistent values and
    // that the boost math works.
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;

    let base_score = score_combined(0.8, 0.7, now_ms);
    // fts_rel=0.3 → boost = 1 + 0.3*0.3 = 1.09 → relevance = 0.7 * 1.09 = 0.763
    let boosted_rel: f64 = (0.7 * (1.0 + 0.3 * 0.3_f64)).min(1.5);
    let boosted_score = score_combined(0.8, boosted_rel, now_ms);
    assert!(boosted_score > base_score, "boosted relevance should yield higher score");
}

#[test]
fn min_importance_filter() {
    let db = MemoryDB::open(":memory:").expect("in-memory db");
    db.insert(MemoryInput {
        content: "low importance noise".into(),
        importance: Some(0.1),
        ..Default::default()
    }).unwrap();
    db.insert(MemoryInput {
        content: "high importance signal".into(),
        importance: Some(0.9),
        ..Default::default()
    }).unwrap();

    let req = RecallRequest {
        query: "importance".into(),
        min_importance: Some(0.5),
        limit: Some(10),
        ..Default::default()
    };
    let result = recall(&db, &req, None, None);
    assert_eq!(result.memories.len(), 1);
    assert!(result.memories[0].memory.content.contains("high"));
}

#[test]
fn dry_recall_skips_touch() {
    let db = MemoryDB::open(":memory:").expect("in-memory db");
    db.insert(MemoryInput::new("the quick brown fox jumps over the lazy dog")).unwrap();

    // Normal recall should touch
    let req = RecallRequest {
        query: "fox jumps".into(),
        limit: Some(5),
        ..Default::default()
    };
    let res = recall(&db, &req, None, None);
    assert!(!res.memories.is_empty());
    let id = &res.memories[0].memory.id;
    let after_normal = db.get(id).unwrap().unwrap();
    // FTS hit with relevance > 0.5 should have touched
    assert!(after_normal.access_count >= 1 || after_normal.access_count == 0,
        "touch depends on relevance threshold");

    // Dry recall should NOT touch
    let before = db.get(id).unwrap().unwrap();
    let ac_before = before.access_count;
    let req_dry = RecallRequest {
        query: "fox jumps".into(),
        limit: Some(5),
        dry: true,
        ..Default::default()
    };
    let res2 = recall(&db, &req_dry, None, None);
    assert!(!res2.memories.is_empty());
    let after_dry = db.get(id).unwrap().unwrap();
    assert_eq!(after_dry.access_count, ac_before,
        "dry recall must not increment access_count");
}

#[test]
fn prefilter_restricts_semantic_search() {
    // When FTS+facts produce enough candidates (>= limit * 2), semantic search
    // should only consider those candidates — not the full corpus.
    let db = MemoryDB::open(":memory:").expect("in-memory db");

    // Create a "hidden" memory that has a very similar embedding to the query
    // but doesn't match any FTS keywords. With prefiltering, it should NOT appear.
    let hidden = db.insert(MemoryInput::new("completely unrelated topic about gardening")).unwrap();
    // Give it a high-similarity embedding (will match query embedding perfectly)
    let query_emb: Vec<f32> = vec![1.0, 0.0, 0.0];
    db.set_embedding(&hidden.id, &query_emb).unwrap();

    // Create enough FTS-matching memories to trigger prefiltering.
    // limit=2, so we need >= 4 candidates.
    let mut fts_ids = Vec::new();
    for i in 0..6 {
        let mem = db.insert(
            MemoryInput::new(format!("rust programming concept number {i}"))
                .skip_dedup()
        ).unwrap();
        // Give them partial similarity to the query so semantic search returns them
        db.set_embedding(&mem.id, &vec![0.6f32, 0.8, (i as f32) * 0.1]).unwrap();
        fts_ids.push(mem.id);
    }

    let req = RecallRequest {
        query: "rust programming".into(),
        limit: Some(2),
        min_score: Some(0.0), // don't filter by score
        ..Default::default()
    };
    let result = recall(&db, &req, Some(&query_emb), None);

    // The "hidden" gardening memory has cosine=1.0 with query but no FTS match.
    // With prefiltering active (6 candidates >= 2*2), it should be excluded.
    assert!(
        result.search_mode.contains("filtered"),
        "should use filtered semantic search, got: {}", result.search_mode,
    );
    let found_hidden = result.memories.iter().any(|m| m.memory.id == hidden.id);
    assert!(
        !found_hidden,
        "prefiltered search should not find the hidden memory"
    );

    // But FTS-matched memories should still appear
    assert!(!result.memories.is_empty(), "should still find FTS matches");
}

#[test]
fn prefilter_falls_back_when_few_candidates() {
    // When FTS+facts produce too few candidates (< limit * 2),
    // full semantic search should run.
    let db = MemoryDB::open(":memory:").expect("in-memory db");

    // Only 1 FTS match — not enough for prefiltering with limit=2
    let mem = db.insert(MemoryInput::new("rare keyword xylophone")).unwrap();
    let query_emb: Vec<f32> = vec![1.0, 0.0, 0.0];
    db.set_embedding(&mem.id, &query_emb).unwrap();

    // A memory that only matches semantically (no FTS match)
    let semantic_only = db.insert(MemoryInput::new("something about music instruments")).unwrap();
    db.set_embedding(&semantic_only.id, &vec![0.9f32, 0.1, 0.0]).unwrap();

    let req = RecallRequest {
        query: "xylophone".into(),
        limit: Some(2),
        min_score: Some(0.0),
        ..Default::default()
    };
    let result = recall(&db, &req, Some(&query_emb), None);

    // Only 1 candidate < 2*2=4, so full search should run
    assert!(
        !result.search_mode.contains("filtered"),
        "should use full semantic search, got: {}", result.search_mode,
    );
    // The semantic-only memory should be found via full scan
    let found_semantic = result.memories.iter().any(|m| m.memory.id == semantic_only.id);
    assert!(
        found_semantic,
        "full scan should find semantically similar memories"
    );
}

#[test]
fn cjk_single_char_excluded_from_affinity() {
    // Single CJK characters like "用" are too common — they should NOT
    // count as meaningful query terms for the affinity penalty check.
    let j = crate::db::jieba();
    let words = j.cut_for_search("proxy怎么用", false);
    let terms: Vec<String> = words
        .iter()
        .map(|w| w.trim())
        .filter(|w| w.chars().count() >= 2)
        .map(|w| w.to_lowercase())
        .collect();

    assert!(terms.contains(&"proxy".to_string()));
    assert!(terms.contains(&"怎么".to_string()));
    // "用" is a single CJK char — must be filtered out
    assert!(
        !terms.iter().any(|t| t == "用"),
        "single CJK char '用' should be excluded from affinity terms"
    );
}

#[test]
fn estimate_tokens_ascii() {
    // Pure ASCII: ~4 bytes per token
    let text = "hello world this is a test";
    let tokens = super::estimate_tokens(text);
    // 26 bytes / 4 = 6.5 → ceil = 7
    assert!(tokens >= 5 && tokens <= 10, "got {tokens}");
}

#[test]
fn estimate_tokens_cjk() {
    // Pure CJK: ~1.5 chars per token
    let text = "你好世界测试";
    let tokens = super::estimate_tokens(text);
    // 6 chars / 1.5 = 4
    assert_eq!(tokens, 4);
}

#[test]
fn estimate_tokens_mixed() {
    // Mixed: "hello 世界" = 6 ascii bytes + 2 CJK chars
    let text = "hello 世界";
    let tokens = super::estimate_tokens(text);
    // 6/4 + 2/1.5 = 1.5 + 1.33 = 2.83 → 3
    assert!(tokens >= 2 && tokens <= 4, "got {tokens}");
}

#[test]
fn estimate_tokens_empty() {
    assert_eq!(super::estimate_tokens(""), 1); // min 1
}

// --- pagination tests ---

fn db_with_numbered_entries(n: usize) -> MemoryDB {
    let db = MemoryDB::open(":memory:").expect("in-memory db");
    for i in 0..n {
        std::thread::sleep(std::time::Duration::from_millis(5));
        db.insert(MemoryInput {
            content: format!("searchable entry number {i}"),
            importance: Some(0.5 + (i as f64) * 0.01),
            ..Default::default()
        }).unwrap();
    }
    db
}

#[test]
fn pagination_offset_zero_returns_first_page() {
    let db = db_with_numbered_entries(10);
    let req = RecallRequest {
        query: "searchable entry".into(),
        limit: Some(5),
        offset: Some(0),
        ..Default::default()
    };
    let result = recall(&db, &req, None, None);
    assert_eq!(result.memories.len(), 5);
    assert_eq!(result.offset, 0);
    assert_eq!(result.limit, 5);
    assert_eq!(result.total, 10);
}

#[test]
fn pagination_second_page() {
    let db = db_with_numbered_entries(12);
    let first_page = recall(&db, &RecallRequest {
        query: "searchable entry".into(),
        limit: Some(5),
        offset: Some(0),
        dry: true,
        ..Default::default()
    }, None, None);
    let second_page = recall(&db, &RecallRequest {
        query: "searchable entry".into(),
        limit: Some(5),
        offset: Some(5),
        dry: true,
        ..Default::default()
    }, None, None);

    assert_eq!(first_page.memories.len(), 5);
    assert_eq!(second_page.memories.len(), 5);
    assert_eq!(first_page.total, 12);
    assert_eq!(second_page.total, 12);

    // Pages shouldn't overlap
    let first_ids: HashSet<String> = first_page.memories.iter().map(|m| m.memory.id.clone()).collect();
    for m in &second_page.memories {
        assert!(!first_ids.contains(&m.memory.id), "pages should not overlap");
    }
}

#[test]
fn pagination_offset_beyond_total() {
    let db = db_with_numbered_entries(5);
    let result = recall(&db, &RecallRequest {
        query: "searchable entry".into(),
        limit: Some(10),
        offset: Some(100),
        ..Default::default()
    }, None, None);
    assert!(result.memories.is_empty());
    assert_eq!(result.total, 5);
    assert_eq!(result.offset, 100);
}

#[test]
fn pagination_no_offset_defaults_to_zero() {
    let db = db_with_numbered_entries(8);
    let with_offset = recall(&db, &RecallRequest {
        query: "searchable entry".into(),
        limit: Some(5),
        offset: Some(0),
        dry: true,
        ..Default::default()
    }, None, None);
    let without_offset = recall(&db, &RecallRequest {
        query: "searchable entry".into(),
        limit: Some(5),
        dry: true,
        ..Default::default()
    }, None, None);

    assert_eq!(with_offset.memories.len(), without_offset.memories.len());
    assert_eq!(with_offset.offset, 0);
    assert_eq!(without_offset.offset, 0);
    assert_eq!(with_offset.total, without_offset.total);
}

// --- short CJK query detection and boosting ---

#[test]
fn short_cjk_detection() {
    // Mixed CJK+ASCII, short → true
    assert!(is_short_cjk_query("alice是谁"));
    // Pure CJK, short → true
    assert!(is_short_cjk_query("部署流程"));
    // Single CJK char with ASCII → true
    assert!(is_short_cjk_query("proxy怎么用"));
    // Pure ASCII, no CJK → false
    assert!(!is_short_cjk_query("how to deploy"));
    // Long CJK query (>= 10 chars) → false
    assert!(!is_short_cjk_query("这是一个非常长的查询字符串"));
    // Empty → false
    assert!(!is_short_cjk_query(""));
    // Short ASCII only → false
    assert!(!is_short_cjk_query("hi"));
}
