use super::parse::{extract_assistant_msg, extract_message_content};

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
