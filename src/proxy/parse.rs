use crate::util::truncate_chars;

/// Extract text content from a single message object.
/// Handles both plain string and array-of-blocks formats.
pub fn extract_message_content(msg: &serde_json::Value) -> Option<String> {
    let content = msg.get("content")?;

    // Plain string
    if let Some(s) = content.as_str() {
        return Some(s.to_string());
    }

    // Array of content blocks (vision, tool results, etc.)
    if let Some(blocks) = content.as_array() {
        let mut out = String::new();
        for block in blocks {
            if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                    if !out.is_empty() {
                        out.push(' ');
                    }
                    out.push_str(t);
                }
            }
        }
        if !out.is_empty() {
            return Some(out);
        }
    }

    None
}

/// Grab the assistant response text from various API formats.
pub fn extract_assistant_msg(raw: &[u8]) -> String {
    let text = String::from_utf8_lossy(raw);

    // Try SSE: scan all lines for data: prefixes (don't rely on first line).
    let mut assembled = String::new();
    let mut saw_data = false;
    for line in text.lines() {
        let Some(data) = line.strip_prefix("data: ") else { continue };
        saw_data = true;
        if data == "[DONE]" { continue; }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
            // OpenAI streaming
            if let Some(delta) = v
                .pointer("/choices/0/delta/content")
                .and_then(|c| c.as_str())
            {
                assembled.push_str(delta);
            }
            // Anthropic streaming
            if let Some(delta) = v.pointer("/delta/text").and_then(|c| c.as_str()) {
                assembled.push_str(delta);
            }
        }
    }
    if saw_data && !assembled.is_empty() {
        return truncate_chars(&assembled, 6000);
    }

    // Non-streaming JSON response
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
        // OpenAI format
        if let Some(content) = v
            .pointer("/choices/0/message/content")
            .and_then(|c| c.as_str())
        {
            return truncate_chars(content, 6000);
        }
        // Anthropic format
        if let Some(blocks) = v.get("content").and_then(|c| c.as_array()) {
            let mut out = String::new();
            for block in blocks {
                if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                    if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                        out.push_str(t);
                    }
                }
            }
            if !out.is_empty() {
                return truncate_chars(&out, 6000);
            }
        }
    }
    String::new()
}
