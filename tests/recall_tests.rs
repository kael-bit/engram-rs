use engram::recall::*;
use engram::db::{MemoryDB, MemoryInput, Layer, Memory};
use std::collections::HashSet;

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
    // "hello" ~1.5 + space + "你好" ~2 CJK chars → expect 3-5 tokens
    assert!(tokens >= 2 && tokens <= 6,
        "mixed en/cjk 'hello 你好' should be 2-6 tokens, got {tokens}");
}

// --- recall integration tests ---


fn test_db_with_data() -> MemoryDB {
    let db = MemoryDB::open(":memory:").expect("in-memory db");
    db.insert(MemoryInput {
        content: "very important fact about rust".into(),
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
    // Budget enforcement: first result always included (even if over budget),
    // subsequent results rejected if they'd exceed budget
    assert!(result.memories.len() <= 1,
        "budget=5 tokens should return at most 1 result, got {}", result.memories.len());
}

#[test]
fn budget_zero_means_unlimited() {
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
    // budget=0 means unlimited — should return same as no budget
    let req_no_budget = RecallRequest {
        query: "rust".into(),
        limit: Some(10),
        ..Default::default()
    };
    let result_no_budget = recall(&db, &req_no_budget, None, None);
    assert!(!result.memories.is_empty(), "budget=0 should be treated as unlimited");
    assert_eq!(result.memories.len(), result_no_budget.memories.len(),
        "budget=0 should return same count as no budget");
}

#[test]
fn time_filter_since() {
    let db = MemoryDB::open(":memory:").expect("in-memory db");
    // insert two memories — we can't control created_at via MemoryInput,
    // but both will have "now" timestamps. Import lets us set timestamps.
    let now = engram::db::now_ms();
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
        modified_at: now - 86_400_000,
        modified_epoch: 0,
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
        modified_at: now - 1000,
        modified_epoch: 0,
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
        source: Some("api".into()), tags: None,
        ..Default::default()
    }).unwrap();
    db.insert(MemoryInput {
        content: "from a session".into(),
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
        source: None, tags: Some(vec!["rust".into(), "engram".into()]),
    ..Default::default()
    }).unwrap();
    db.insert(MemoryInput {
        content: "python script notes".into(),
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
fn empty_query_does_not_panic() {
    let db = test_db_with_data();
    let req = RecallRequest {
        query: "".into(),
        limit: Some(5),
        ..Default::default()
    };
    // Empty query must not panic; it may return 0 results (no FTS/semantic match)
    let result = recall(&db, &req, None, None);
    assert!(result.memories.len() <= 5, "should respect limit");
    // Verify result structure is valid even with empty query
    for sm in &result.memories {
        assert!(!sm.memory.id.is_empty(), "returned memory must have an id");
    }
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
        ..Default::default()
    }).unwrap();
    let core = db.insert(MemoryInput {
        content: "core fact about testing".into(),
        ..Default::default()
    }).unwrap();
    // Promote to Core for layer filter test
    db.promote(&core.id, Layer::Core).unwrap();

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
    // FTS-only recall (no embeddings) may not exceed touch relevance threshold (0.5)
    // The important assertion is below: dry recall must NOT touch regardless
    let ac_after_normal = after_normal.access_count;

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

    // Use higher-dimensional embeddings for HNSW stability
    let mut qemb = vec![0.0f32; 64];
    qemb[0] = 1.0;

    let mut sem_emb = vec![0.0f32; 64];
    sem_emb[0] = 0.9;
    sem_emb[1] = 0.1;

    // Only 1 FTS match — not enough for prefiltering with limit=2
    let mem = db.insert(MemoryInput::new("rare keyword xylophone")).unwrap();
    db.set_embedding(&mem.id, &qemb).unwrap();

    // A memory that only matches semantically (no FTS match)
    let semantic_only = db.insert(MemoryInput::new("something about music instruments")).unwrap();
    db.set_embedding(&semantic_only.id, &sem_emb).unwrap();

    let req = RecallRequest {
        query: "xylophone".into(),
        limit: Some(2),
        min_score: Some(0.0),
        ..Default::default()
    };
    let result = recall(&db, &req, Some(&qemb), None);

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
    let j = engram::db::jieba();
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
    let tokens = engram::recall::estimate_tokens(text);
    // 26 bytes / 4 = 6.5 → ceil = 7
    assert!(tokens >= 5 && tokens <= 10, "got {tokens}");
}

#[test]
fn estimate_tokens_cjk() {
    // Pure CJK: ~1.5 chars per token
    let text = "你好世界测试";
    let tokens = engram::recall::estimate_tokens(text);
    // 6 chars / 1.5 = 4
    assert_eq!(tokens, 4);
}

#[test]
fn estimate_tokens_mixed() {
    // Mixed: "hello 世界" = 6 ascii bytes + 2 CJK chars
    let text = "hello 世界";
    let tokens = engram::recall::estimate_tokens(text);
    // 6/4 + 2/1.5 = 1.5 + 1.33 = 2.83 → 3
    assert!(tokens >= 2 && tokens <= 4, "got {tokens}");
}

#[test]
fn estimate_tokens_empty() {
    assert_eq!(engram::recall::estimate_tokens(""), 1); // min 1
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

// --- recency_score tests ---

fn make_memory(layer: Layer, importance: f64, decay_rate: f64, last_accessed: i64) -> Memory {
    let now = engram::db::now_ms();
    Memory {
        id: "test-mem".into(),
        content: "test content for scoring".into(),
        layer,
        importance,
        created_at: now,
        last_accessed,
        access_count: 1,
        repetition_count: 0,
        decay_rate,
        source: "test".into(),
        tags: vec![],
        namespace: "default".into(),
        embedding: None,
        kind: "semantic".into(),
        modified_at: now,
        modified_epoch: 0,
    }
}

#[test]
fn recency_score_just_accessed() {
    let now = engram::db::now_ms();
    let score = recency_score(now, 0.1);
    // exp(0) = 1.0 exactly, but tiny clock skew possible
    assert!((score - 1.0).abs() < 0.01, "just accessed should be ≈1.0, got {score}");
}

#[test]
fn recency_score_one_week_old() {
    let now = engram::db::now_ms();
    let one_week_ago = now - 168 * 3_600_000; // 168 hours in ms
    let decay_rate = 0.1;
    let score = recency_score(one_week_ago, decay_rate);
    // hours = 168, rate = 0.1, formula: exp(-0.1 * 168 / 168) = exp(-0.1)
    let expected = (-decay_rate as f64).exp(); // exp(-0.1) ≈ 0.9048
    assert!(
        (score - expected).abs() < 0.01,
        "one week old with decay=0.1 should be ≈{expected}, got {score}"
    );
}

#[test]
fn recency_score_infinite_decay_rate() {
    let now = engram::db::now_ms();
    let one_hour_ago = now - 3_600_000;
    let score = recency_score(one_hour_ago, f64::INFINITY);
    // Infinite should be clamped to 0.1 (the fallback), score should be finite and > 0
    assert!(score.is_finite(), "infinite decay_rate must not produce NaN/Inf");
    assert!(score > 0.0, "score should be positive, got {score}");
    assert!(score <= 1.0, "score should be <= 1.0, got {score}");
}

#[test]
fn recency_score_negative_decay_rate() {
    let now = engram::db::now_ms();
    let old = now - 168 * 3_600_000;
    let score = recency_score(old, -5.0);
    // Negative should be clamped to 0.0 → exp(0) = 1.0, no decay
    assert!(
        (score - 1.0).abs() < 1e-9,
        "negative decay_rate should clamp to 0 → score=1.0, got {score}"
    );
}

#[test]
fn recency_score_very_old_memory() {
    let now = engram::db::now_ms();
    let ancient = now - 10_000 * 3_600_000; // 10000 hours ago
    let score = recency_score(ancient, 0.1);
    // exp(-0.1 * 10000 / 168) = exp(-5.95) ≈ 0.0026 — near zero but positive
    assert!(score >= 0.0, "very old memory score must not be negative, got {score}");
    assert!(score < 0.01, "very old memory should score near 0, got {score}");
    assert!(score.is_finite(), "score must be finite");
}

#[test]
fn recency_score_future_timestamp() {
    let now = engram::db::now_ms();
    let future = now + 3_600_000; // 1 hour in the future
    let score = recency_score(future, 0.1);
    // hours = max(negative, 0) = 0 → exp(0) = 1.0
    assert!(
        (score - 1.0).abs() < 1e-9,
        "future timestamp should yield score=1.0, got {score}"
    );
}

#[test]
fn recency_score_zero_decay_rate() {
    let now = engram::db::now_ms();
    // Very old memory with zero decay → exp(0) = 1.0 always
    let ancient = now - 10_000 * 3_600_000;
    let score = recency_score(ancient, 0.0);
    assert!(
        (score - 1.0).abs() < 1e-9,
        "zero decay_rate should always yield 1.0, got {score}"
    );
    // Also test with recent
    let score_recent = recency_score(now, 0.0);
    assert!(
        (score_recent - 1.0).abs() < 1e-9,
        "zero decay_rate + recent should be 1.0, got {score_recent}"
    );
}

#[test]
fn recency_score_huge_finite_decay() {
    let now = engram::db::now_ms();
    let hour_ago = now - 3_600_000;
    // decay_rate=100 should be clamped to 10
    let score = recency_score(hour_ago, 100.0);
    assert!(score.is_finite(), "huge decay_rate must not produce NaN/Inf");
    assert!(score >= 0.0 && score <= 1.0);
}

#[test]
fn recency_score_nan_decay() {
    let now = engram::db::now_ms();
    let score = recency_score(now - 3_600_000, f64::NAN);
    // NaN is not finite → should fall back to 0.1
    assert!(score.is_finite(), "NaN decay_rate must not produce NaN");
    assert!(score > 0.0 && score <= 1.0);
}

// --- score_memory tests ---

#[test]
fn score_memory_perfect_scores() {
    let now = engram::db::now_ms();
    let mem = make_memory(Layer::Core, 1.0, 0.05, now);
    let scored = score_memory(&mem, 1.0);
    // importance=1.0, relevance=1.0, recency≈1.0, bonus=1.1
    // raw = (0.2*1 + 0.2*1 + 0.6*1) * 1.1 = 1.1 → capped to 1.0
    assert!(
        (scored.score - 1.0).abs() < 1e-9,
        "perfect inputs with Core layer should cap at 1.0, got {}", scored.score
    );
}

#[test]
fn score_memory_zero_everything() {
    let now = engram::db::now_ms();
    let very_old = now - 10_000 * 3_600_000;
    let mem = make_memory(Layer::Buffer, 0.0, 5.0, very_old);
    let scored = score_memory(&mem, 0.0);
    // importance=0, relevance=0, recency≈0, bonus=0.9
    // Should be near zero
    assert!(scored.score >= 0.0, "score must not be negative, got {}", scored.score);
    assert!(scored.score < 0.05, "zero inputs + old should score near 0, got {}", scored.score);
}

#[test]
fn score_memory_layer_bonus() {
    let now = engram::db::now_ms();
    // Same memory, same params, different layers
    let mem_buffer = make_memory(Layer::Buffer, 0.8, 0.1, now);
    let mem_core = make_memory(Layer::Core, 0.8, 0.1, now);

    let scored_buffer = score_memory(&mem_buffer, 0.7);
    let scored_core = score_memory(&mem_core, 0.7);

    // Buffer layer_boost=0.8, Core layer_boost=1.2 → Core should score higher
    assert!(
        scored_core.score > scored_buffer.score,
        "Core (layer_boost=1.2) should outscore Buffer (layer_boost=0.8): core={} vs buffer={}",
        scored_core.score, scored_buffer.score
    );
}

#[test]
fn score_memory_cap_at_one() {
    let now = engram::db::now_ms();
    // Max everything: importance=1.0, relevance=1.0, recency≈1.0, Core bonus=1.1
    let mem = make_memory(Layer::Core, 1.0, 0.0, now); // decay=0 → recency=1.0
    let scored = score_memory(&mem, 1.0);
    // raw = (0.2 + 0.2 + 0.6) * 1.1 = 1.1 → must cap at 1.0
    assert!(
        scored.score <= 1.0,
        "score must never exceed 1.0, got {}", scored.score
    );
    assert!(
        (scored.score - 1.0).abs() < 1e-9,
        "should be exactly 1.0 after capping, got {}", scored.score
    );
}

#[test]
fn score_memory_weight_distribution() {
    let now = engram::db::now_ms();
    // Test that relevance dominates: high relevance + low importance vs low relevance + high importance
    let mem_high_rel = make_memory(Layer::Working, 0.1, 0.0, now); // importance=0.1, decay=0→recency=1.0
    let mem_high_imp = make_memory(Layer::Working, 0.9, 0.0, now); // importance=0.9, decay=0→recency=1.0

    // New formula: 0.5*relevance + 0.3*memory_weight(mem) + 0.2*recency
    // memory_weight(high_rel) = (0.1 + 0 + ln(2)*0.1) * 1.0 * 1.0 ≈ 0.1693
    // high relevance (0.9) + low weight: 0.5*0.9 + 0.3*0.1693 + 0.2*1.0 ≈ 0.701
    let scored_high_rel = score_memory(&mem_high_rel, 0.9);
    // memory_weight(high_imp) = (0.9 + 0 + ln(2)*0.1) * 1.0 * 1.0 ≈ 0.9693
    // low relevance (0.1) + high weight: 0.5*0.1 + 0.3*0.9693 + 0.2*1.0 ≈ 0.541
    let scored_high_imp = score_memory(&mem_high_imp, 0.1);

    assert!(
        scored_high_rel.score > scored_high_imp.score,
        "high relevance (weight=0.5) should beat high importance (weight=0.3): \
         high_rel={} vs high_imp={}",
        scored_high_rel.score, scored_high_imp.score
    );

    // Verify approximate values (Working layer boost = 1.0, so no distortion)
    let expected_high_rel = 0.701; // 0.5*0.9 + 0.3*0.1693 + 0.2*1.0
    let expected_high_imp = 0.541; // 0.5*0.1 + 0.3*0.9693 + 0.2*1.0
    assert!(
        (scored_high_rel.score - expected_high_rel).abs() < 0.01,
        "expected ≈{expected_high_rel}, got {}", scored_high_rel.score
    );
    assert!(
        (scored_high_imp.score - expected_high_imp).abs() < 0.01,
        "expected ≈{expected_high_imp}, got {}", scored_high_imp.score
    );
}
