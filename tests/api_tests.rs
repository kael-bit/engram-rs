use engram::api::router;
use engram::AppState;
use engram::EmbedCache;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use tower::ServiceExt;

fn test_state(api_key: Option<&str>) -> AppState {
    let mdb = engram::db::MemoryDB::open(":memory:").unwrap();
    AppState {
        db: std::sync::Arc::new(mdb),
        ai: None,
        api_key: api_key.map(|s| s.to_string()),
        embed_cache: engram::EmbedCache::new(16),
        embed_queue: None,
        proxy: None,
        started_at: std::time::Instant::now(),
        last_proxy_turn: std::sync::Arc::new(std::sync::atomic::AtomicI64::new(0)),
    }
}

async fn body_json(resp: axum::response::Response) -> serde_json::Value {
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

fn json_req(method: &str, uri: &str, body: serde_json::Value) -> axum::http::Request<Body> {
    axum::http::Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

#[allow(dead_code)]
fn authed_json_req(
    method: &str,
    uri: &str,
    body: serde_json::Value,
    token: &str,
) -> axum::http::Request<Body> {
    axum::http::Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .header("authorization", format!("Bearer {token}"))
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

fn get_req(uri: &str, token: Option<&str>) -> axum::http::Request<Body> {
    let mut b = axum::http::Request::builder().method("GET").uri(uri);
    if let Some(t) = token {
        b = b.header("authorization", format!("Bearer {t}"));
    }
    b.body(Body::empty()).unwrap()
}

// --- Auth ---

#[tokio::test]
async fn auth_rejects_no_token() {
    let app = router(test_state(Some("secret123")));
    let resp = app.oneshot(get_req("/recent?hours=1", None)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn auth_rejects_wrong_token() {
    let app = router(test_state(Some("secret123")));
    let resp = app
        .oneshot(get_req("/recent?hours=1", Some("wrongtoken")))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn auth_passes_correct_token() {
    let app = router(test_state(Some("secret123")));
    let resp = app
        .oneshot(get_req("/recent?hours=1", Some("secret123")))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn stats_no_auth_needed() {
    let app = router(test_state(Some("secret123")));
    let resp = app.oneshot(get_req("/stats", None)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = body_json(resp).await;
    assert_eq!(j["total"], 0);
}

// --- Create ---

#[tokio::test]
async fn create_memory_returns_201() {
    let state = test_state(None);
    let app = router(state);
    let resp = app
        .oneshot(json_req(
            "POST",
            "/memories",
            serde_json::json!({"content": "hello world"}),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let j = body_json(resp).await;
    assert_eq!(j["content"], "hello world");
    assert!(j["id"].is_string());
}

#[tokio::test]
async fn create_empty_content_returns_400() {
    let app = router(test_state(None));
    let resp = app
        .oneshot(json_req(
            "POST",
            "/memories",
            serde_json::json!({"content": ""}),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// --- Get / Delete ---

#[tokio::test]
async fn get_missing_returns_404() {
    let app = router(test_state(None));
    let resp = app
        .oneshot(get_req("/memories/nonexistent-id", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn delete_missing_returns_404() {
    let app = router(test_state(None));
    let req = axum::http::Request::builder()
        .method("DELETE")
        .uri("/memories/nonexistent-id")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// --- Recall ---

#[tokio::test]
async fn recall_empty_query_returns_400() {
    let app = router(test_state(None));
    let resp = app
        .oneshot(json_req(
            "POST",
            "/recall",
            serde_json::json!({"query": ""}),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn recall_valid_query_returns_200() {
    let app = router(test_state(None));
    let resp = app
        .oneshot(json_req(
            "POST",
            "/recall",
            serde_json::json!({"query": "test"}),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = body_json(resp).await;
    assert!(j["memories"].is_array());
}

// --- Consolidate ---

#[tokio::test]
async fn consolidate_empty_body() {
    let app = router(test_state(None));
    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/consolidate")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = body_json(resp).await;
    // consolidate returns a result object with counts
    assert!(j.is_object(), "consolidate should return JSON object");
}

// --- Batch delete ---

#[tokio::test]
async fn batch_delete_returns_count() {
    let app = router(test_state(None));
    let resp = app
        .oneshot(json_req(
            "DELETE",
            "/memories",
            serde_json::json!({"ids": ["nope1", "nope2"]}),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = body_json(resp).await;
    assert_eq!(j["deleted"], 0);
}

// --- Namespace ---

#[tokio::test]
async fn namespace_via_header() {
    let state = test_state(None);
    let app = router(state.clone());

    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/memories")
        .header("content-type", "application/json")
        .header("x-namespace", "test-ns")
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({"content": "namespaced"})).unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let j = body_json(resp).await;
    assert_eq!(j["namespace"], "test-ns");
}

#[tokio::test]
async fn batch_create_inserts_all() {
    let app = router(test_state(None));
    let body = serde_json::json!([
        {"content": "batch item 1"},
        {"content": "batch item 2"},
        {"content": "batch item 3"},
    ]);
    let resp = app
        .oneshot(json_req("POST", "/memories/batch", body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = body_json(resp).await;
    assert_eq!(j["inserted"], 3);
    assert_eq!(j["requested"], 3);
}

#[tokio::test]
async fn delete_by_namespace() {
    let app = router(test_state(None));
    // store 2 in ns-a, 1 in ns-b
    for ns in ["ns-a", "ns-a", "ns-b"] {
        let body = serde_json::json!({"content": format!("mem in {ns}"), "namespace": ns, "skip_dedup": true});
        let resp = app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
    }
    // delete ns-a
    let resp = app.clone()
        .oneshot(json_req("DELETE", "/memories", serde_json::json!({"namespace": "ns-a"})))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = body_json(resp).await;
    assert_eq!(j["deleted"], 2);
    // ns-b still has its memory
    let resp = app.oneshot(Request::builder().uri("/memories?ns=ns-b").body(Body::empty()).unwrap()).await.unwrap();
    let j = body_json(resp).await;
    assert_eq!(j["count"], 1);
}

#[tokio::test]
async fn sync_embed_field_accepted() {
    // Just verifies the API accepts sync_embed without error
    // (actual embedding generation requires AI config)
    let app = router(test_state(None));
    let body = serde_json::json!({
        "content": "sync embed test",
        "sync_embed": true
    });
    let resp = app.oneshot(json_req("POST", "/memories", body)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let j = body_json(resp).await;
    assert!(!j["id"].as_str().unwrap().is_empty());
}

#[tokio::test]
async fn list_memories_returns_all() {
    let app = router(test_state(None));
    for i in 0..3 {
        let body = serde_json::json!({"content": format!("list test {i}"), "skip_dedup": true});
        app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();
    }
    let resp = app.oneshot(Request::builder().uri("/memories").body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = body_json(resp).await;
    assert!(j["count"].as_i64().unwrap() >= 3);
    assert!(j["memories"].as_array().unwrap().len() >= 3);
}

#[tokio::test]
async fn update_memory_changes_content() {
    let app = router(test_state(None));
    let body = serde_json::json!({"content": "before update"});
    let resp = app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();
    let created = body_json(resp).await;
    let id = created["id"].as_str().unwrap();

    let patch = serde_json::json!({"content": "after update"});
    let resp = app.clone().oneshot(json_req("PATCH", &format!("/memories/{id}"), patch)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let updated = body_json(resp).await;
    assert_eq!(updated["content"], "after update");
}

#[tokio::test]
async fn patch_memory_kind() {
    let app = router(test_state(None));
    let body = serde_json::json!({"content": "kind patch test"});
    let resp = app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();
    let created = body_json(resp).await;
    let id = created["id"].as_str().unwrap();
    // "semantic" is default and skipped in serialization
    assert!(created.get("kind").is_none() || created["kind"] == "semantic");

    let patch = serde_json::json!({"kind": "procedural"});
    let resp = app.clone().oneshot(json_req("PATCH", &format!("/memories/{id}"), patch)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let updated = body_json(resp).await;
    assert_eq!(updated["kind"], "procedural");
}

#[tokio::test]
async fn export_import_roundtrip() {
    let app = router(test_state(None));
    // create
    let body = serde_json::json!({"content": "roundtrip test", "namespace": "rt-test"});
    app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();

    // export
    let resp = app.clone().oneshot(
        Request::builder().uri("/export").body(Body::empty()).unwrap()
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let exported = body_json(resp).await;
    assert!(exported["count"].as_i64().unwrap() >= 1);

    // import into fresh app
    let app2 = router(test_state(None));
    let resp = app2.oneshot(json_req("POST", "/import", exported)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let imported = body_json(resp).await;
    assert!(imported["imported"].as_i64().unwrap() >= 1);
}

#[tokio::test]
async fn list_recent_returns_memories() {
    let app = router(test_state(None));
    let body = serde_json::json!({"content": "recent test entry"});
    app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();

    let resp = app.oneshot(
        Request::builder().uri("/recent?hours=1").body(Body::empty()).unwrap()
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = body_json(resp).await;
    assert!(!j["memories"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn quick_search_finds_match() {
    let app = router(test_state(None));
    let body = serde_json::json!({"content": "quicksearch xylophone unique"});
    app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();

    let resp = app.oneshot(
        Request::builder().uri("/search?q=xylophone").body(Body::empty()).unwrap()
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = body_json(resp).await;
    let mems = j["memories"].as_array().unwrap();
    assert!(!mems.is_empty(), "should find the memory with unique term");
}

#[tokio::test]
async fn resume_returns_structured_sections() {
    let app = router(test_state(None));

    // Seed: a high-importance core memory (identity)
    let body = serde_json::json!({
        "content": "I am the test agent, created for integration testing",
        "layer": 3, "importance": 0.9
    });
    app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();

    // Seed: a session memory with next-action tag
    let body = serde_json::json!({
        "content": "next step: write more tests",
        "layer": 2, "importance": 0.7,
        "source": "session", "tags": ["next-action"]
    });
    app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();

    // Seed: a regular session memory
    let body = serde_json::json!({
        "content": "did some refactoring today",
        "layer": 2, "importance": 0.6,
        "source": "session"
    });
    app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();

    let resp = app.oneshot(
        Request::builder().uri("/resume?hours=1&format=json").body(Body::empty()).unwrap()
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = body_json(resp).await;

    // Structure checks
    assert!(j["core"].is_array());
    assert!(j["working"].is_array());
    assert!(j["buffer"].is_array());
    assert!(j["recent"].is_array());
    assert!(j["sessions"].is_array());
    assert!(j["hours"].as_f64().unwrap() > 0.0);

    // Core should include the high-importance core memory
    let core = j["core"].as_array().unwrap();
    assert!(!core.is_empty(), "should have core memories");
    assert!(core[0]["content"].as_str().unwrap().contains("test agent"));

    // Sessions should have the session memory
    let sessions = j["sessions"].as_array().unwrap();
    assert!(!sessions.is_empty(), "should have session memories");

    // Session memories must NOT leak into buffer or working sections
    let buffer = j["buffer"].as_array().unwrap();
    for b in buffer {
        let src = b["source"].as_str().unwrap_or("");
        assert_ne!(src, "session", "session memory leaked into buffer: {}", b["content"]);
    }
    let working = j["working"].as_array().unwrap();
    for w in working {
        let src = w["source"].as_str().unwrap_or("");
        assert_ne!(src, "session", "session memory leaked into working: {}", w["content"]);
    }
}

#[tokio::test]
async fn resume_session_notes_in_buffer_go_to_sessions() {
    let app = router(test_state(None));

    // Session note stored at buffer level (default layer=1) — real-world scenario
    let body = serde_json::json!({
        "content": "Session: did auth refactor, tests pass",
        "source": "session",
        "importance": 0.6
    });
    app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();

    // Session note with next-action tag at buffer level
    let body = serde_json::json!({
        "content": "Next: write integration tests for new auth flow",
        "source": "session", "tags": ["next-action"]
    });
    app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();

    // Non-session buffer memory
    let body = serde_json::json!({"content": "user prefers dark mode"});
    app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();

    let resp = app.oneshot(
        Request::builder().uri("/resume?hours=1&format=json").body(Body::empty()).unwrap()
    ).await.unwrap();
    let j = body_json(resp).await;

    // Buffer-level session notes should appear in sessions, not buffer
    let sessions = j["sessions"].as_array().unwrap();
    assert!(sessions.iter().any(|s|
        s["content"].as_str().unwrap_or("").contains("auth refactor")),
        "session note missing from sessions section");

    // next-action tagged memories should appear in sessions (not as separate section)
    assert!(sessions.iter().any(|s|
        s["content"].as_str().unwrap_or("").contains("integration tests")),
        "next-action tagged note missing from sessions");

    let buffer = j["buffer"].as_array().unwrap();
    for b in buffer {
        let src = b["source"].as_str().unwrap_or("");
        assert_ne!(src, "session", "session note leaked into buffer: {}", b["content"]);
    }
    // Non-session buffer should still be there
    assert!(buffer.iter().any(|b|
        b["content"].as_str().unwrap_or("").contains("dark mode")),
        "non-session buffer memory missing");
}

#[tokio::test]
async fn resume_respects_namespace() {
    let app = router(test_state(None));

    // Seed in ns-a
    let body = serde_json::json!({
        "content": "ns-a identity", "layer": 3, "importance": 0.9,
        "namespace": "ns-a"
    });
    app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();

    // Seed in ns-b
    let body = serde_json::json!({
        "content": "ns-b identity", "layer": 3, "importance": 0.9,
        "namespace": "ns-b"
    });
    app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();

    // Resume with ns=ns-a
    let resp = app.oneshot(
        Request::builder().uri("/resume?hours=1&ns=ns-a&format=json").body(Body::empty()).unwrap()
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = body_json(resp).await;

    let core = j["core"].as_array().unwrap();
    assert!(core.iter().all(|m| {
        m["namespace"].as_str().unwrap_or("default") == "ns-a"
            || m["namespace"].as_str().is_none()
    }), "should only return ns-a memories");
}

#[tokio::test]
async fn export_excludes_embeddings_by_default() {
    let app = router(test_state(None));
    let body = serde_json::json!({"content": "export test"});
    app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();

    let resp = app.oneshot(
        Request::builder().uri("/export").body(Body::empty()).unwrap()
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = body_json(resp).await;
    assert!(j["count"].as_u64().unwrap() >= 1);
    // no embedding field in default export
    let first = &j["memories"][0];
    assert!(first.get("embedding").is_none(), "embedding should be omitted by default");
}

#[tokio::test]
async fn repair_returns_counts() {
    let app = router(test_state(None));
    let body = serde_json::json!({"content": "repair test"});
    app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();

    let resp = app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/repair")
            .body(Body::empty())
            .unwrap()
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = body_json(resp).await;
    assert_eq!(j["orphans_removed"], 0);
    assert_eq!(j["fts_rebuilt"], 0);
}

#[tokio::test]
async fn vacuum_returns_freed() {
    let app = router(test_state(None));
    let resp = app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/vacuum")
            .body(Body::empty())
            .unwrap()
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = body_json(resp).await;
    assert!(j["freed_bytes"].is_number());
    assert_eq!(j["mode"], "incremental");
}

#[tokio::test]
async fn triggers_returns_matching_memories() {
    let app = router(test_state(None));

    // Create a memory with trigger tag
    let body = serde_json::json!({
        "content": "never commit internal docs to public repos",
        "tags": ["lesson", "trigger:git-push"]
    });
    app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();

    // Create a non-trigger memory
    let body2 = serde_json::json!({"content": "unrelated note"});
    app.clone().oneshot(json_req("POST", "/memories", body2)).await.unwrap();

    // Query triggers for git-push
    let resp = app.clone().oneshot(
        Request::builder()
            .uri("/triggers/git-push")
            .header("Authorization", "Bearer test-key")
            .body(Body::empty()).unwrap()
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = body_json(resp).await;
    assert_eq!(j["action"], "git-push");
    assert_eq!(j["count"], 1);
    assert!(j["memories"][0]["content"].as_str().unwrap().contains("internal docs"));
}

#[tokio::test]
async fn triggers_empty_when_no_match() {
    let app = router(test_state(None));
    let resp = app.oneshot(
        Request::builder()
            .uri("/triggers/nonexistent-action")
            .header("Authorization", "Bearer test-key")
            .body(Body::empty()).unwrap()
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = body_json(resp).await;
    assert_eq!(j["count"], 0);
}

#[tokio::test]
async fn resume_compact_includes_kind() {
    let app = router(test_state(None));

    // Procedural memory
    let body = serde_json::json!({
        "content": "Deploy: test, build, stop, copy, start",
        "layer": 3,
        "importance": 0.9,
        "kind": "procedural"
    });
    app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();

    let resp = app.oneshot(
        Request::builder().uri("/resume?hours=1&compact=true&format=json").body(Body::empty()).unwrap()
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = body_json(resp).await;

    let core = j["core"].as_array().unwrap();
    assert!(!core.is_empty(), "should have core memories");
    let proc_mem = core.iter().find(|m| m["content"].as_str().unwrap_or("").contains("Deploy"));
    assert!(proc_mem.is_some(), "should find the procedural memory");
    assert_eq!(proc_mem.unwrap()["kind"].as_str(), Some("procedural"),
        "compact output should include kind field for non-semantic memories");
}

#[tokio::test]
async fn list_memories_total_respects_tag_filter() {
    let app = router(test_state(None));

    // Create memories: 2 with tag "alpha", 3 without
    for i in 0..2 {
        let body = serde_json::json!({"content": format!("tagged {i}"), "tags": ["alpha"], "skip_dedup": true});
        app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();
    }
    for i in 0..3 {
        let body = serde_json::json!({"content": format!("plain {i}"), "tags": ["beta"], "skip_dedup": true});
        app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();
    }

    // Unfiltered: total should be 5
    let resp = app.clone().oneshot(
        Request::builder().uri("/memories").body(Body::empty()).unwrap()
    ).await.unwrap();
    let j = body_json(resp).await;
    assert_eq!(j["total"].as_i64().unwrap(), 5);

    // Filtered by tag=alpha: total should be 2
    let resp = app.clone().oneshot(
        Request::builder().uri("/memories?tag=alpha").body(Body::empty()).unwrap()
    ).await.unwrap();
    let j = body_json(resp).await;
    assert_eq!(j["count"].as_i64().unwrap(), 2);
    assert_eq!(j["total"].as_i64().unwrap(), 2, "total must match filtered count");
}

#[tokio::test]
async fn list_memories_total_respects_kind_filter() {
    let app = router(test_state(None));

    let body = serde_json::json!({"content": "a procedure", "kind": "procedural", "skip_dedup": true});
    app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();
    for i in 0..2 {
        let body = serde_json::json!({"content": format!("semantic {i}"), "skip_dedup": true});
        app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();
    }

    let resp = app.clone().oneshot(
        Request::builder().uri("/memories?kind=procedural").body(Body::empty()).unwrap()
    ).await.unwrap();
    let j = body_json(resp).await;
    assert_eq!(j["count"].as_i64().unwrap(), 1);
    assert_eq!(j["total"].as_i64().unwrap(), 1, "total must reflect kind filter");
}

#[tokio::test]
async fn list_memories_total_respects_layer_filter() {
    let app = router(test_state(None));

    let body = serde_json::json!({"content": "core mem", "layer": 3, "skip_dedup": true});
    app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();
    for i in 0..2 {
        let body = serde_json::json!({"content": format!("buf {i}"), "skip_dedup": true});
        app.clone().oneshot(json_req("POST", "/memories", body)).await.unwrap();
    }

    let resp = app.clone().oneshot(
        Request::builder().uri("/memories?layer=3").body(Body::empty()).unwrap()
    ).await.unwrap();
    let j = body_json(resp).await;
    assert_eq!(j["count"].as_i64().unwrap(), 1);
    assert_eq!(j["total"].as_i64().unwrap(), 1, "total must reflect layer filter");
}
