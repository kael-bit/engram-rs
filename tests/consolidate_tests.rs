use engram::consolidate::*;
use engram::db::{MemoryDB, MemoryInput, Layer, Memory, now_ms};
use engram::ai::cosine_similarity;

fn make_mem(id: &str, layer: Layer, importance: f64, emb: Vec<f32>) -> (Memory, Vec<f32>) {
    (
        Memory {
            id: id.into(),
            content: format!("memory {id}"),
            layer,
            importance,
            created_at: 0,
            last_accessed: 0,
            access_count: 0,
            repetition_count: 0,
            decay_rate: 1.0,
            source: "test".into(),
            tags: vec![],
            namespace: "default".into(),
            embedding: None,
            kind: "semantic".into(),
            modified_at: 0,
        },
        emb,
    )
}

#[test]
fn cluster_similar_vectors() {
    let mems = [
        make_mem("a", Layer::Working, 0.5, vec![1.0f32, 0.0, 0.0]),
        make_mem("b", Layer::Working, 0.5, vec![0.999f32, 0.01, 0.0]),
        make_mem("c", Layer::Working, 0.5, vec![0.0f32, 1.0, 0.0]),
    ];
    let refs: Vec<&(Memory, Vec<f32>)> = mems.iter().collect();
    let clusters = find_clusters(&refs, 0.85);

    assert_eq!(clusters.len(), 2);
    let big = clusters.iter().find(|c| c.len() == 2).unwrap();
    assert!(big.contains(&0) && big.contains(&1));
}

#[test]
fn cluster_all_different() {
    let mems = [
        make_mem("a", Layer::Working, 0.5, vec![1.0f32, 0.0, 0.0]),
        make_mem("b", Layer::Working, 0.5, vec![0.0f32, 1.0, 0.0]),
        make_mem("c", Layer::Working, 0.5, vec![0.0f32, 0.0, 1.0]),
    ];
    let refs: Vec<&(Memory, Vec<f32>)> = mems.iter().collect();
    let clusters = find_clusters(&refs, 0.85);

    assert_eq!(clusters.len(), 3);
    assert!(clusters.iter().all(|c| c.len() == 1));
}

#[test]
fn cluster_all_identical() {
    let mems = [
        make_mem("a", Layer::Working, 0.5, vec![1.0f32, 0.0]),
        make_mem("b", Layer::Working, 0.5, vec![1.0f32, 0.0]),
        make_mem("c", Layer::Working, 0.5, vec![1.0f32, 0.0]),
    ];
    let refs: Vec<&(Memory, Vec<f32>)> = mems.iter().collect();
    let clusters = find_clusters(&refs, 0.85);

    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].len(), 3);
}

#[test]
fn cluster_empty_input() {
    let mems: Vec<&(Memory, Vec<f32>)> = vec![];
    let clusters = find_clusters(&mems, 0.85);
    assert!(clusters.is_empty());
}

// --- consolidate_sync tests ---
// These use import() to set up memories with specific timestamps.

fn test_db() -> MemoryDB {
    MemoryDB::open(":memory:").expect("in-memory db")
}

fn mem_with_ts(
    id: &str,
    layer: Layer,
    importance: f64,
    access_count: i64,
    created_ms: i64,
    accessed_ms: i64,
) -> Memory {
    Memory {
        id: id.into(),
        content: format!("test memory {id}"),
        layer,
        importance,
        created_at: created_ms,
        last_accessed: accessed_ms,
        access_count,
        repetition_count: 0,
        decay_rate: layer.default_decay(),
        source: "test".into(),
        tags: vec![],
        namespace: "default".into(),
        embedding: None,
        kind: "semantic".into(),
        modified_at: created_ms,
    }
}

#[test]
fn promote_high_access_working() {
    let db = test_db();
    let now = engram::db::now_ms();
    // working memory with enough accesses and importance → should be a candidate
    let good = mem_with_ts("promote-me", Layer::Working, 0.8, 5, now - 1000, now);
    // working memory with low access → should not be a candidate
    let meh = mem_with_ts("leave-me", Layer::Working, 0.8, 1, now - 1000, now);
    db.import(&[good, meh]).unwrap();

    let result = consolidate_sync(&db, None, "full");
    // Without LLM, candidates are collected but not promoted
    assert_eq!(result.promotion_candidates.len(), 1);
    assert_eq!(result.promotion_candidates[0].0, "promote-me");

    // Simulate no-AI fallback: promote candidates directly
    for (id, _, _, _, _, _) in &result.promotion_candidates {
        db.promote(id, Layer::Core).unwrap();
    }
    let promoted = db.get("promote-me").unwrap().unwrap();
    assert_eq!(promoted.layer, Layer::Core);
    let stayed = db.get("leave-me").unwrap().unwrap();
    assert_eq!(stayed.layer, Layer::Working);
}

#[test]
fn age_promote_old_working() {
    let db = test_db();
    let now = engram::db::now_ms();
    let eight_days_ago = now - 8 * 86_400_000;
    // old working memory with decent importance → should be a candidate by age
    let old = mem_with_ts("old-but-worthy", Layer::Working, 0.6, 1, eight_days_ago, now);
    // fresh working memory → should not be a candidate
    let fresh = mem_with_ts("too-young", Layer::Working, 0.6, 1, now - 1000, now);
    db.import(&[old, fresh]).unwrap();

    let result = consolidate_sync(&db, None, "full");
    let candidate_ids: Vec<&str> = result.promotion_candidates.iter().map(|(id, _, _, _, _, _)| id.as_str()).collect();
    assert!(candidate_ids.contains(&"old-but-worthy"));
    assert!(!candidate_ids.contains(&"too-young"));
}

#[test]
fn drop_expired_low_importance_buffer() {
    let db = test_db();
    let now = engram::db::now_ms();
    let three_days_ago = now - 3 * 86_400_000;
    // old buffer, never accessed → should be dropped after TTL + hard_cap
    let expendable = mem_with_ts("bye", Layer::Buffer, 0.2, 0, three_days_ago, three_days_ago);
    // old buffer, accessed enough (≥ rescue threshold of 2) → rescued to working
    let valuable = mem_with_ts("save-me", Layer::Buffer, 0.5, 3, three_days_ago, three_days_ago);
    // old buffer, accessed once → below rescue threshold, dropped
    let barely = mem_with_ts("not-enough", Layer::Buffer, 0.5, 1, three_days_ago, three_days_ago);
    db.import(&[expendable, valuable, barely]).unwrap();

    let _result = consolidate_sync(&db, None, "full");
    assert!(db.get("bye").unwrap().is_none(), "never-accessed buffer should be gone");
    assert!(db.get("not-enough").unwrap().is_none(), "barely-accessed buffer should be gone after TTL");
    let saved = db.get("save-me").unwrap();
    assert!(saved.is_some(), "well-accessed buffer should survive");
}

#[test]
fn nothing_to_do() {
    let db = test_db();
    let now = engram::db::now_ms();
    // fresh core memory — nothing should happen
    let stable = mem_with_ts("stable", Layer::Core, 0.9, 10, now - 1000, now);
    db.import(&[stable]).unwrap();

    let r = consolidate_sync(&db, None, "full");
    assert_eq!(r.promoted, 0);
    assert_eq!(r.decayed, 0);
}

#[test]
fn buffer_promoted_by_access() {
    let db = test_db();
    let now = engram::db::now_ms();
    // Buffer memory with enough accesses (≥5) should promote to Working
    let accessed = mem_with_ts("recalled", Layer::Buffer, 0.1, 6, now - 1000, now);
    // Buffer with only 3 accesses — not enough, stays in Buffer
    let not_enough = mem_with_ts("still-young", Layer::Buffer, 0.1, 3, now - 1000, now);
    db.import(&[accessed, not_enough]).unwrap();

    let r = consolidate_sync(&db, None, "full");
    assert_eq!(r.promoted, 1);
    let got = db.get("recalled").unwrap().unwrap();
    assert_eq!(got.layer, Layer::Working);
    let stayed = db.get("still-young").unwrap().unwrap();
    assert_eq!(stayed.layer, Layer::Buffer);
}

#[test]
fn buffer_ttl_accessed_enough_promotes() {
    let db = test_db();
    let three_days_ago = engram::db::now_ms() - 3 * 86_400_000;
    // Old buffer with 3 accesses (≥ rescue threshold of 2) — should rescue to Working
    let accessed = mem_with_ts("rescued", Layer::Buffer, 0.1, 3, three_days_ago, three_days_ago);
    // Old buffer with 1 access — below rescue threshold, should be dropped
    let barely = mem_with_ts("barely", Layer::Buffer, 0.1, 1, three_days_ago, three_days_ago);
    db.import(&[accessed, barely]).unwrap();

    let r = consolidate_sync(&db, None, "full");
    assert!(r.promoted >= 1);
    let got = db.get("rescued").unwrap().unwrap();
    assert_eq!(got.layer, Layer::Working);
    assert!(db.get("barely").unwrap().is_none(), "barely-accessed buffer should be dropped after TTL");
}

#[test]
fn buffer_ttl_never_accessed_drops() {
    let db = test_db();
    let three_days_ago = engram::db::now_ms() - 3 * 86_400_000;
    // Old buffer with 0 accesses — should be dropped after TTL + hard_cap (48h)
    let unused = mem_with_ts("forgotten", Layer::Buffer, 0.1, 0, three_days_ago, three_days_ago);
    db.import(&[unused]).unwrap();

    let r = consolidate_sync(&db, None, "full");
    assert!(r.decayed >= 1);
    assert!(db.get("forgotten").unwrap().is_none());
}

#[test]
fn operational_tag_blocks_working_to_core_promotion() {
    let db = test_db();
    let now = engram::db::now_ms();

    // Session-tagged memory — high importance and access count,
    // but must NOT become a candidate (blocked before LLM gate).
    let mut session_mem = mem_with_ts("session-mem", Layer::Working, 0.9, 5, now - 1000, now);
    session_mem.source = "session".into();

    // Normal working memory with identical importance/access — SHOULD be a candidate.
    let normal = mem_with_ts("normal-mem", Layer::Working, 0.9, 5, now - 1000, now);

    // Ephemeral-tagged memory — also blocked.
    let mut ephemeral = mem_with_ts("ephemeral-mem", Layer::Working, 0.9, 5, now - 1000, now);
    ephemeral.tags = vec!["ephemeral".into()];

    db.import(&[session_mem, normal, ephemeral]).unwrap();

    let result = consolidate_sync(&db, None, "full");
    let candidate_ids: Vec<&str> = result.promotion_candidates.iter().map(|(id, _, _, _, _, _)| id.as_str()).collect();

    assert!(candidate_ids.contains(&"normal-mem"), "normal memory should be a candidate");
    assert!(!candidate_ids.contains(&"session-mem"), "session memory must not be a candidate");
    assert!(!candidate_ids.contains(&"ephemeral-mem"), "ephemeral memory must not be a candidate");
}

#[test]
fn lesson_tag_survives_buffer_ttl() {
    let db = test_db();
    let two_days_ago = engram::db::now_ms() - 2 * 86_400_000;

    // Lesson tagged memory — old, zero accesses, but should NOT be dropped
    let mut lesson = mem_with_ts("never-force-push", Layer::Buffer, 0.5, 0, two_days_ago, two_days_ago);
    lesson.tags = vec!["lesson".into(), "trigger:git-push".into()];

    // Regular buffer — same age, same accesses, should be dropped
    let regular = mem_with_ts("some-note", Layer::Buffer, 0.5, 0, two_days_ago, two_days_ago);

    db.import(&[lesson, regular]).unwrap();

    let r = consolidate_sync(&db, None, "full");
    assert!(db.get("never-force-push").unwrap().is_some(), "lesson must survive TTL");
    assert!(db.get("some-note").unwrap().is_none(), "regular buffer should be dropped");
    assert!(r.decayed >= 1);
}

#[test]
fn gate_rejected_skips_promotion() {
    let db = test_db();
    let old = engram::db::now_ms() - 7 * 86_400_000;

    // High-scoring Working memory WITH gate-rejected tag — should NOT be a candidate
    let mut rejected = mem_with_ts("gate-rej", Layer::Working, 0.8, 20, old, engram::db::now_ms());
    rejected.tags = vec!["gate-rejected".into()];
    rejected.repetition_count = 5;

    // High-scoring Working memory WITHOUT gate-rejected — should be a candidate
    let mut eligible = mem_with_ts("eligible", Layer::Working, 0.8, 20, old, engram::db::now_ms());
    eligible.repetition_count = 5;

    db.import(&[rejected, eligible]).unwrap();

    let r = consolidate_sync(&db, None, "full");

    // gate-rejected should NOT appear in promotion candidates
    let candidate_ids: Vec<&str> = r.promotion_candidates.iter().map(|(id, _, _, _, _, _)| id.as_str()).collect();
    assert!(!candidate_ids.contains(&"gate-rej"), "gate-rejected must not be a promotion candidate");
    assert!(candidate_ids.contains(&"eligible"), "eligible should be a promotion candidate");
}

#[test]
fn gate_rejected_retries_after_cooldown() {
    let db = test_db();
    let old = engram::db::now_ms() - 3 * 86_400_000; // 3 days ago
    let stale_access = engram::db::now_ms() - 2 * 86_400_000; // last accessed 2 days ago (>24h)

    // gate-rejected but last_accessed > 7 days ago → cooldown expired
    let mut retry = mem_with_ts("retry-me", Layer::Working, 0.8, 20, old, stale_access);
    retry.tags = vec!["gate-rejected".into()];
    retry.repetition_count = 5;

    db.import(&[retry]).unwrap();

    let r = consolidate_sync(&db, None, "full");

    // Should now be a promotion candidate (tag cleared, eligible again)
    let candidate_ids: Vec<&str> = r.promotion_candidates.iter().map(|(id, _, _, _, _, _)| id.as_str()).collect();
    assert!(candidate_ids.contains(&"retry-me"), "gate-rejected with expired cooldown should retry");

    // Tag should be removed
    let mem = db.get("retry-me").unwrap().unwrap();
    assert!(!mem.tags.contains(&"gate-rejected".to_string()), "gate-rejected tag should be cleared");
}

#[test]
fn high_importance_buffer_rescued_at_ttl() {
    let db = test_db();
    let expired = engram::db::now_ms() - 49 * 3600 * 1000; // 49h ago, past hard_cap (48h)

    // sc=0 but high importance — should be rescued to Working
    let important = mem_with_ts("design-decision", Layer::Buffer, 0.8, 0, expired, expired);

    // sc=0, low importance — should be dropped (past hard_cap even if never triaged)
    let junk = mem_with_ts("random-chat", Layer::Buffer, 0.3, 0, expired, expired);

    db.import(&[important, junk]).unwrap();

    let r = consolidate_sync(&db, None, "full");

    assert!(db.get("design-decision").unwrap().is_some(), "high importance buffer should survive");
    let rescued = db.get("design-decision").unwrap().unwrap();
    assert_eq!(rescued.layer, Layer::Working, "should be promoted to Working");

    assert!(db.get("random-chat").unwrap().is_none(), "low importance buffer should be dropped");
    assert!(r.decayed >= 1, "at least one should be decayed");
}

#[test]
fn distilled_tag_blocks_buffer_promotion() {
    let db = test_db();
    let now = engram::db::now_ms();

    // Buffer memory with enough accesses but tagged distilled — blocked
    let mut distilled = mem_with_ts("distilled-note", Layer::Buffer, 0.5, 8, now - 1000, now);
    distilled.tags = vec!["session".into(), "distilled".into()];
    distilled.source = "session".into();

    // Identical stats without distilled tag — should promote
    let mut normal = mem_with_ts("normal-note", Layer::Buffer, 0.5, 8, now - 1000, now);
    normal.tags = vec![];

    db.import(&[distilled, normal]).unwrap();

    let r = consolidate_sync(&db, None, "full");
    assert_eq!(r.promoted, 1, "only the non-distilled should promote");
    let d = db.get("distilled-note").unwrap().unwrap();
    assert_eq!(d.layer, Layer::Buffer, "distilled must stay in Buffer");
    let n = db.get("normal-note").unwrap().unwrap();
    assert_eq!(n.layer, Layer::Working, "normal should promote to Working");
}

#[test]
fn auto_distilled_blocks_core_promotion() {
    let db = test_db();
    let now = engram::db::now_ms();

    // Working memory with auto-distilled tag — high score, should NOT be a candidate
    let mut ad = mem_with_ts("project-status", Layer::Working, 0.7, 20, now - 1000, now);
    ad.tags = vec!["project-status".into(), "auto-distilled".into()];
    ad.repetition_count = 5;

    // Working memory without auto-distilled — should be a candidate
    let mut normal = mem_with_ts("real-lesson", Layer::Working, 0.7, 20, now - 1000, now);
    normal.repetition_count = 5;

    db.import(&[ad, normal]).unwrap();

    let r = consolidate_sync(&db, None, "full");
    let ids: Vec<&str> = r.promotion_candidates.iter().map(|(id, _, _, _, _, _)| id.as_str()).collect();
    assert!(!ids.contains(&"project-status"), "auto-distilled must not be a Core candidate");
    assert!(ids.contains(&"real-lesson"), "normal Working should be a candidate");
}

#[test]
fn gate_result_deserializes() {
    // Test that GateResult deserializes from function call JSON correctly
    let json = r#"{"decision":"approve","kind":"semantic"}"#;
    let r: GateResult = serde_json::from_str(json).unwrap();
    assert_eq!(r.decision, "approve");
    assert_eq!(r.kind.as_deref(), Some("semantic"));

    let json = r#"{"decision":"approve","kind":"procedural"}"#;
    let r: GateResult = serde_json::from_str(json).unwrap();
    assert_eq!(r.decision, "approve");
    assert_eq!(r.kind.as_deref(), Some("procedural"));

    let json = r#"{"decision":"approve","kind":"episodic"}"#;
    let r: GateResult = serde_json::from_str(json).unwrap();
    assert_eq!(r.decision, "approve");
    assert_eq!(r.kind.as_deref(), Some("episodic"));

    let json = r#"{"decision":"approve"}"#;
    let r: GateResult = serde_json::from_str(json).unwrap();
    assert_eq!(r.decision, "approve");
    assert!(r.kind.is_none());

    let json = r#"{"decision":"reject"}"#;
    let r: GateResult = serde_json::from_str(json).unwrap();
    assert_eq!(r.decision, "reject");
    assert!(r.kind.is_none());
}

#[test]
fn gate_rejected_within_cooldown_skips_promotion() {
    let db = test_db();
    let now = engram::db::now_ms();

    // gate-rejected memory accessed recently (within 24h cooldown)
    let mut gr = mem_with_ts("gr-fresh", Layer::Working, 0.8, 20, now - 3600_000, now - 1000);
    gr.tags = vec!["gate-rejected".into()];
    gr.repetition_count = 5;

    db.import(&[gr]).unwrap();
    let r = consolidate_sync(&db, None, "full");
    let ids: Vec<&str> = r.promotion_candidates.iter().map(|(id, _, _, _, _, _)| id.as_str()).collect();
    assert!(!ids.contains(&"gr-fresh"), "gate-rejected within 24h must not be a candidate");

    // Verify the tag is still there (not cleared prematurely)
    let m = db.get("gr-fresh").unwrap().unwrap();
    assert!(m.tags.iter().any(|t| t == "gate-rejected"), "tag should persist within cooldown");
}

#[test]
fn gate_rejected_after_cooldown_retries() {
    let db = test_db();
    let now = engram::db::now_ms();

    // gate-rejected memory last accessed >24h ago
    let old_access = now - 25 * 3600_000; // 25 hours ago
    let mut gr = mem_with_ts("gr-old", Layer::Working, 0.8, 20, now - 48 * 3600_000, old_access);
    gr.tags = vec!["gate-rejected".into()];
    gr.repetition_count = 5;

    db.import(&[gr]).unwrap();
    let r = consolidate_sync(&db, None, "full");

    // After cooldown, the tag should be removed and it becomes a candidate
    let m = db.get("gr-old").unwrap().unwrap();
    assert!(!m.tags.iter().any(|t| t == "gate-rejected"),
        "tag should be cleared after 24h cooldown");
    let ids: Vec<&str> = r.promotion_candidates.iter().map(|(id, _, _, _, _, _)| id.as_str()).collect();
    assert!(ids.contains(&"gr-old"), "should re-enter promotion pipeline after cooldown");
}

#[test]
fn session_notes_blocked_from_core_promotion() {
    let db = test_db();
    let now = engram::db::now_ms();

    let mut session = mem_with_ts("session-note", Layer::Working, 0.8, 30, now - 1000, now);
    session.source = "session".into();
    session.tags = vec!["session".into()];
    session.repetition_count = 10;

    db.import(&[session]).unwrap();
    let r = consolidate_sync(&db, None, "full");
    let ids: Vec<&str> = r.promotion_candidates.iter().map(|(id, _, _, _, _, _)| id.as_str()).collect();
    assert!(!ids.contains(&"session-note"), "session notes must never reach Core promotion");
}

// --- reconcile_pair_key tests ---

#[test]
fn reconcile_pair_key_is_order_independent() {
    let k1 = reconcile_pair_key("abc-123", "xyz-789");
    let k2 = reconcile_pair_key("xyz-789", "abc-123");
    assert_eq!(k1, k2, "key must be the same regardless of argument order");
}

#[test]
fn reconcile_pair_key_lexicographic() {
    let key = reconcile_pair_key("beta", "alpha");
    assert_eq!(key, "alpha:beta", "smaller id should come first");

    let key2 = reconcile_pair_key("alpha", "beta");
    assert_eq!(key2, "alpha:beta");
}

#[test]
fn reconcile_pair_key_same_id() {
    let key = reconcile_pair_key("same", "same");
    assert_eq!(key, "same:same");
}

#[test]
fn reconcile_pair_key_uuid_style() {
    let a = "550e8400-e29b-41d4-a716-446655440000";
    let b = "6ba7b810-9dad-11d1-80b4-00c04fd430c8";
    let k1 = reconcile_pair_key(a, b);
    let k2 = reconcile_pair_key(b, a);
    assert_eq!(k1, k2);
    assert!(k1.starts_with("550e"), "lexicographically smaller UUID should come first");
}

#[test]
fn buffer_cap_evicts_oldest() {
    std::env::set_var("ENGRAM_BUFFER_CAP", "5");
    let db = std::sync::Arc::new(MemoryDB::open(":memory:").unwrap());

    for i in 0..8 {
        let input = MemoryInput::new(format!("buffer item {i}"))
            .layer(1); // 1 = Buffer
        db.insert(input).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    let before = db.list_by_layer_meta(Layer::Buffer, 100, 0).unwrap();
    assert_eq!(before.len(), 8);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all().build().unwrap();
    rt.block_on(async {
        engram::consolidate::consolidate(db.clone(), None, None).await;
    });

    let after = db.list_by_layer_meta(Layer::Buffer, 100, 0).unwrap();
    assert!(after.len() <= 5, "buffer should be capped at 5, got {}", after.len());

    let contents: Vec<String> = after.iter().map(|m| m.content.clone()).collect();
    assert!(contents.contains(&"buffer item 7".to_string()),
        "newest item should survive: {contents:?}");

    std::env::remove_var("ENGRAM_BUFFER_CAP");
}
