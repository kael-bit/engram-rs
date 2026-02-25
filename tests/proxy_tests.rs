use engram::proxy::parse::{extract_assistant_msg, extract_message_content};
use engram::proxy::window::extract_user_via_watermark;

#[test]
fn extract_content_plain_string() {
    let msg = serde_json::json!({"role": "user", "content": "hello world"});
    assert_eq!(extract_message_content(&msg).unwrap(), "hello world");
}

#[test]
fn extract_content_array_blocks() {
    let msg = serde_json::json!({
        "role": "user",
        "content": [
            {"type": "text", "text": "first part"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
            {"type": "text", "text": "second part"}
        ]
    });
    assert_eq!(extract_message_content(&msg).unwrap(), "first part second part");
}

#[test]
fn extract_content_no_content_field() {
    let msg = serde_json::json!({"role": "user"});
    assert!(extract_message_content(&msg).is_none());
}

#[test]
fn extract_content_null_content() {
    let msg = serde_json::json!({"role": "assistant", "content": null});
    assert!(extract_message_content(&msg).is_none());
}

#[test]
fn extract_content_empty_blocks() {
    let msg = serde_json::json!({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:..."}}
        ]
    });
    assert!(extract_message_content(&msg).is_none());
}

#[test]
fn extract_assistant_openai_buffered() {
    let body = serde_json::json!({
        "choices": [{"message": {"content": "hi there"}}]
    });
    let raw = serde_json::to_vec(&body).unwrap();
    assert_eq!(extract_assistant_msg(&raw), "hi there");
}

#[test]
fn extract_assistant_anthropic_buffered() {
    let body = serde_json::json!({
        "content": [
            {"type": "text", "text": "response from claude"}
        ]
    });
    let raw = serde_json::to_vec(&body).unwrap();
    assert_eq!(extract_assistant_msg(&raw), "response from claude");
}

#[test]
fn extract_assistant_sse_openai() {
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hello \"}}]}\ndata: {\"choices\":[{\"delta\":{\"content\":\"world\"}}]}\ndata: [DONE]\n";
    assert_eq!(extract_assistant_msg(sse.as_bytes()), "hello world");
}

#[test]
fn extract_assistant_empty() {
    assert!(extract_assistant_msg(b"{}").is_empty());
}

// --- extract_user_via_watermark tests ---

fn test_state() -> engram::AppState {
    let mdb = engram::db::MemoryDB::open(":memory:").unwrap();
    engram::AppState {
        db: std::sync::Arc::new(mdb),
        ai: None,
        api_key: None,
        embed_cache: engram::EmbedCache::new(4),
        proxy: None,
        started_at: std::time::Instant::now(),
        last_proxy_turn: std::sync::Arc::new(std::sync::atomic::AtomicI64::new(0)),
    }
}

fn make_req(messages: &[(&str, &str)]) -> Vec<u8> {
    let msgs: Vec<serde_json::Value> = messages
        .iter()
        .map(|(role, content)| serde_json::json!({"role": role, "content": content}))
        .collect();
    serde_json::to_vec(&serde_json::json!({"messages": msgs})).unwrap()
}

#[test]
fn watermark_first_request_returns_none() {
    let state = test_state();
    let req = make_req(&[("user", "hello"), ("assistant", "hi"), ("user", "how are you")]);
    // First call sets the watermark baseline — nothing extracted yet
    let result = extract_user_via_watermark(&state, &req, "sess-a");
    assert!(result.is_none());
    // 3 filtered messages: user + assistant + user (system excluded if any)
    assert_eq!(state.db.get_watermark("sess-a").unwrap(), 3);
}

#[test]
fn watermark_second_request_extracts_new_user_msgs() {
    let state = test_state();
    let key = "sess-b";

    let req1 = make_req(&[("user", "first"), ("assistant", "reply1")]);
    extract_user_via_watermark(&state, &req1, key);

    // New turn appended
    let req2 = make_req(&[
        ("user", "first"),
        ("assistant", "reply1"),
        ("user", "second question"),
        ("assistant", "reply2"),
    ]);
    let result = extract_user_via_watermark(&state, &req2, key);
    assert_eq!(result.unwrap(), "second question");
}

#[test]
fn watermark_context_truncation_returns_none() {
    let state = test_state();
    let key = "sess-c";

    // Establish watermark at 4 messages
    let req1 = make_req(&[
        ("user", "a"), ("assistant", "b"),
        ("user", "c"), ("assistant", "d"),
    ]);
    extract_user_via_watermark(&state, &req1, key);
    assert_eq!(state.db.get_watermark(key).unwrap(), 4);

    // Context truncated: provider dropped old messages, only 2 remain
    let req2 = make_req(&[("user", "c"), ("assistant", "d")]);
    let result = extract_user_via_watermark(&state, &req2, key);
    assert!(result.is_none());
    // Watermark resets to current count
    assert_eq!(state.db.get_watermark(key).unwrap(), 2);
}

#[test]
fn watermark_empty_messages_returns_none() {
    let state = test_state();
    let req = serde_json::to_vec(&serde_json::json!({"messages": []})).unwrap();
    let result = extract_user_via_watermark(&state, &req, "sess-d");
    assert!(result.is_none());
}

#[test]
fn watermark_mixed_roles_only_user_content() {
    let state = test_state();
    let key = "sess-e";

    let req1 = make_req(&[("user", "baseline")]);
    extract_user_via_watermark(&state, &req1, key);

    // New messages: assistant + user interleaved
    let req2 = make_req(&[
        ("user", "baseline"),
        ("assistant", "ignored reply"),
        ("user", "this matters"),
        ("assistant", "also ignored"),
    ]);
    let result = extract_user_via_watermark(&state, &req2, key).unwrap();
    assert!(result.contains("this matters"));
    assert!(!result.contains("ignored"));
}

#[test]
fn watermark_multiple_new_user_messages() {
    let state = test_state();
    let key = "sess-f";

    let req1 = make_req(&[("user", "init"), ("assistant", "ack")]);
    extract_user_via_watermark(&state, &req1, key);

    // Three new turns, two have user messages
    let req2 = make_req(&[
        ("user", "init"),
        ("assistant", "ack"),
        ("user", "question one"),
        ("assistant", "answer one"),
        ("user", "question two"),
        ("assistant", "answer two"),
    ]);
    let result = extract_user_via_watermark(&state, &req2, key).unwrap();
    assert!(result.contains("question one"));
    assert!(result.contains("question two"));
}

#[test]
fn watermark_system_messages_ignored_in_count() {
    let state = test_state();
    let key = "sess-g";

    // System messages should be filtered out of the watermark count
    let msgs = serde_json::json!({
        "messages": [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    });
    let req1 = serde_json::to_vec(&msgs).unwrap();
    extract_user_via_watermark(&state, &req1, key);
    // Only user+assistant counted (system filtered)
    assert_eq!(state.db.get_watermark(key).unwrap(), 2);

    let msgs2 = serde_json::json!({
        "messages": [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "new question"},
        ]
    });
    let req2 = serde_json::to_vec(&msgs2).unwrap();
    let result = extract_user_via_watermark(&state, &req2, key).unwrap();
    assert_eq!(result, "new question");
}

#[test]
fn watermark_no_new_user_msgs_returns_empty_string() {
    let state = test_state();
    let key = "sess-h";

    let req1 = make_req(&[("user", "start"), ("assistant", "ok")]);
    extract_user_via_watermark(&state, &req1, key);

    // Only an assistant message added (no new user content)
    let req2 = make_req(&[
        ("user", "start"),
        ("assistant", "ok"),
        ("assistant", "followup"),
    ]);
    let result = extract_user_via_watermark(&state, &req2, key);
    // New messages exist but none are user role → returns Some("")
    assert_eq!(result.unwrap(), "");
}

#[test]
fn watermark_invalid_json_returns_none() {
    let state = test_state();
    let result = extract_user_via_watermark(&state, b"not json at all", "sess-i");
    assert!(result.is_none());
}

#[test]
fn watermark_missing_messages_field_returns_none() {
    let state = test_state();
    let req = serde_json::to_vec(&serde_json::json!({"model": "gpt-4"})).unwrap();
    let result = extract_user_via_watermark(&state, &req, "sess-j");
    assert!(result.is_none());
}
