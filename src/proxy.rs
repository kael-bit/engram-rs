use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderValue, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use futures::StreamExt;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, error, info, warn};

use crate::{ai, db, AppState};

// Sliding window thresholds: flush when enough context accumulates.
const WINDOW_MAX_TURNS: usize = 8;
const WINDOW_MAX_CHARS: usize = 16000;

static PROXY_REQUESTS: AtomicU64 = AtomicU64::new(0);
static PROXY_EXTRACTED: AtomicU64 = AtomicU64::new(0);

pub fn proxy_stats(db: Option<&db::MemoryDB>) -> (u64, u64, usize) {
    let buffered = db.map(|d| d.proxy_turn_count()).unwrap_or(0);
    (
        PROXY_REQUESTS.load(Ordering::Relaxed),
        PROXY_EXTRACTED.load(Ordering::Relaxed),
        buffered,
    )
}

#[derive(Clone)]
pub struct ProxyConfig {
    pub upstream: String,
    pub default_key: Option<String>,
    pub client: reqwest::Client,
}

pub async fn handle(
    State(state): State<AppState>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let Some(ref proxy) = state.proxy else {
        return (StatusCode::NOT_FOUND, "proxy not configured").into_response();
    };

    let path = uri.path().strip_prefix("/proxy").unwrap_or(uri.path());
    PROXY_REQUESTS.fetch_add(1, Ordering::Relaxed);

    let upstream_url = if let Some(q) = uri.query() {
        format!("{}{}?{}", proxy.upstream, path, q)
    } else {
        format!("{}{}", proxy.upstream, path)
    };

    let req_bytes = match axum::body::to_bytes(body, 10 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            error!("failed to read request body: {e}");
            return (StatusCode::BAD_REQUEST, "failed to read body").into_response();
        }
    };
    let req_capture = req_bytes.to_vec();

    let mut upstream_req = proxy.client.request(
        reqwest::Method::from_bytes(method.as_str().as_bytes()).unwrap_or(reqwest::Method::POST),
        &upstream_url,
    );

    for (name, value) in &headers {
        let skip = matches!(
            name.as_str(),
            "host" | "connection" | "transfer-encoding" | "content-length"
        );
        if !skip {
            if let Ok(v) = value.to_str() {
                upstream_req = upstream_req.header(name.as_str(), v);
            }
        }
    }

    if !headers.contains_key("authorization") {
        if let Some(ref key) = proxy.default_key {
            upstream_req = upstream_req.header("authorization", format!("Bearer {key}"));
        }
    }

    upstream_req = upstream_req.body(req_bytes);

    let upstream_res = match upstream_req.send().await {
        Ok(r) => r,
        Err(e) => {
            error!("upstream request failed: {e}");
            return (StatusCode::BAD_GATEWAY, format!("upstream error: {e}")).into_response();
        }
    };

    let status = StatusCode::from_u16(upstream_res.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    let mut res_headers = HeaderMap::new();
    for (name, value) in upstream_res.headers() {
        if let Ok(v) = HeaderValue::from_bytes(value.as_bytes()) {
            if let Ok(n) = axum::http::header::HeaderName::from_bytes(name.as_str().as_bytes()) {
                res_headers.insert(n, v);
            }
        }
    }

    let is_stream = upstream_res
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.contains("text/event-stream"));

    // Hash auth token to separate conversation windows per caller
    let session_key = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(|s| {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            s.hash(&mut h);
            format!("{:016x}", h.finish())
        })
        .unwrap_or_else(|| "default".into());

    if is_stream {
        stream_response(state, status, res_headers, upstream_res, req_capture, session_key).await
    } else {
        buffered_response(state, status, res_headers, upstream_res, req_capture, session_key).await
    }
}

async fn buffered_response(
    state: AppState,
    status: StatusCode,
    headers: HeaderMap,
    upstream_res: reqwest::Response,
    req_capture: Vec<u8>,
    session_key: String,
) -> Response {
    let res_bytes = match upstream_res.bytes().await {
        Ok(b) => b,
        Err(e) => {
            error!("failed to read upstream response: {e}");
            return (StatusCode::BAD_GATEWAY, "failed to read response").into_response();
        }
    };
    let res_capture = res_bytes.to_vec();

    if status.is_success() {
        tokio::spawn(buffer_exchange(state, req_capture, res_capture, session_key));
    }

    let mut response = (status, res_bytes.to_vec()).into_response();
    *response.headers_mut() = headers;
    response
}

async fn stream_response(
    state: AppState,
    status: StatusCode,
    headers: HeaderMap,
    upstream_res: reqwest::Response,
    req_capture: Vec<u8>,
    session_key: String,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Vec<u8>, std::io::Error>>(32);
    let state_clone = state.clone();
    let extract_ok = status.is_success();

    tokio::spawn(async move {
        let mut buf = Vec::new();
        let mut stream = upstream_res.bytes_stream();

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    buf.extend_from_slice(&bytes);
                    // Client disconnect is fine — we still have the full response in buf.
                    // Don't break; keep draining the upstream so we capture everything.
                    let _ = tx.send(Ok(bytes.to_vec())).await;
                }
                Err(e) => {
                    warn!("stream chunk error: {e}");
                    let _ = tx.send(Err(std::io::Error::other(e.to_string()))).await;
                    break;
                }
            }
        }
        drop(tx);

        if extract_ok && !buf.is_empty() {
            buffer_exchange(state_clone, req_capture, buf, session_key).await;
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let body = Body::from_stream(stream);

    let mut response = Response::builder()
        .status(status)
        .body(body)
        .unwrap_or_else(|_| Response::new(Body::empty()));
    *response.headers_mut() = headers;
    response
}

/// Parse a single exchange from raw request/response bytes, add to the sliding window.
/// When window is full, flush and extract memories from the accumulated context.
async fn buffer_exchange(state: AppState, req_raw: Vec<u8>, res_raw: Vec<u8>, session_key: String) {
    let user_msg = extract_last_user_msg(&req_raw);
    let assistant_msg = extract_assistant_msg(&res_raw);

    // Skip non-chat requests (models list, health checks, etc.)
    if user_msg.is_empty() && assistant_msg.is_empty() {
        return;
    }

    // Skip subagent traffic — their conversations are operational, not worth extracting
    if user_msg.contains("[Subagent Context]") || user_msg.contains("[Subagent Task]") {
        debug!("proxy: skipping subagent turn");
        return;
    }

    // Skip very short tool-use-only turns
    if user_msg.len() + assistant_msg.len() < 80 {
        return;
    }

    let turn = format!("User: {}\nAssistant: {}", user_msg, assistant_msg);

    // Persist turn to SQLite so it survives restarts
    if let Err(e) = state.db.save_proxy_turn(&session_key, &turn) {
        warn!("proxy: failed to persist turn: {e}");
    }

    // Check if we've accumulated enough to extract
    if state.db.proxy_session_should_flush(&session_key, WINDOW_MAX_TURNS, WINDOW_MAX_CHARS) {
        if let Ok(ctx) = state.db.drain_proxy_session(&session_key) {
            if !ctx.is_empty() {
                extract_from_context(state, &ctx).await;
            }
        }
    }
}

/// Flush all remaining buffered turns. Called from background timer and shutdown.
pub async fn flush_window(state: &AppState) {
    let pending = match state.db.drain_all_proxy_turns() {
        Ok(p) => p,
        Err(e) => {
            warn!("proxy: failed to drain turns: {e}");
            return;
        }
    };
    for (_key, context) in pending {
        if !context.is_empty() {
            extract_from_context(state.clone(), &context).await;
        }
    }
}

async fn extract_from_context(state: AppState, context: &str) {
    let Some(ref ai_cfg) = state.ai else {
        return;
    };

    let turn_count = context.matches("User: ").count();
    info!(
        turns = turn_count,
        chars = context.len(),
        "proxy: extraction window flushed"
    );

    if context.len() < 100 {
        info!("proxy: context too thin to extract from ({} chars)", context.len());
        return;
    }

    // Strip repetitive system/template content before extracting
    let filtered = strip_boilerplate(context);
    if filtered.len() < 100 {
        debug!("proxy: context too thin after stripping boilerplate");
        return;
    }
    let preview: String = filtered.chars().take(12000).collect();

    // Fetch recent buffer memories so the LLM can skip already-known concepts
    let recent_mems = state.db.list_since(
        db::now_ms() - 6 * 3600 * 1000,
        30,
    ).unwrap_or_default();
    let existing_knowledge = if recent_mems.is_empty() {
        String::new()
    } else {
        let lines: Vec<String> = recent_mems.iter()
            .map(|m| {
                let truncated: String = m.content.chars().take(120).collect();
                format!("- {}", truncated)
            })
            .collect();
        format!("\n\n=== ALREADY IN MEMORY (do NOT extract anything that overlaps with these) ===\n{}\n", lines.join("\n"))
    };

    let system = "You extract long-term memories from a multi-turn LLM conversation.\n\
        Your bar is EXTREMELY high. Most windows produce 0 extractions. That's correct.\n\n\
        ASK YOURSELF: Would a human write this in their personal notebook? If not, skip it.\n\n\
        ONLY extract:\n\
        - A USER's stated preference, rule, or boundary (their words, not the assistant's interpretation)\n\
        - A hard-won lesson from a real mistake (not 'we should do X' — only 'we did X and it broke because Y')\n\
        - A workflow rule the USER explicitly established (not the assistant proposing one)\n\n\
        NEVER extract (automatic reject):\n\
        - What was built, fixed, changed, deployed, or refactored\n\
        - Bug descriptions, root causes, or fixes\n\
        - Code review findings or refactoring suggestions\n\
        - System health, metrics, or status observations\n\
        - The assistant's analysis, proposals, or explanations\n\
        - Anything the assistant concluded on its own (only USER-stated facts matter)\n\
        - Descriptions of how something works or was designed\n\
        - Anything that overlaps with ALREADY IN MEMORY below\n\
        - Meta-observations about the extraction process itself\n\n\
        LANGUAGE: Write in the same language the USER uses. If the user speaks Chinese, output Chinese.\n\n\
        DEDUP: If your extraction is essentially the same point as something in ALREADY IN MEMORY, skip it entirely.";

    let user = format!(
        "Extract 0-1 memories from this conversation. Zero is the expected default — only extract if something truly important was said by the USER. Call with empty items if nothing qualifies.{existing_knowledge}\n\n\
         === CONVERSATION ({} turns) ===\n{preview}",
        context.matches("User: ").count()
    );

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "One sentence, concrete, under 150 chars"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "1-4 relevant tags"
                        },
                        "kind": {
                            "type": "string",
                            "enum": ["semantic", "episodic", "procedural"],
                            "description": "semantic=facts/decisions/lessons/preferences (most memories are this). episodic=specific dated events. procedural=reusable step-by-step workflows ONLY (e.g. 'deploy: test→build→stop→copy→start'). Code changes, prompt edits, bug fixes are NOT procedural — they are semantic."
                        }
                    },
                    "required": ["content", "tags", "kind"]
                },
                "maxItems": 1
            }
        },
        "required": ["items"]
    });

    #[derive(serde::Deserialize)]
    struct ExtractResult {
        #[serde(default)]
        items: Vec<ExtractionEntry>,
    }

    let entries: Vec<ExtractionEntry> = match ai::llm_tool_call::<ExtractResult>(
        ai_cfg, "proxy", system, &user,
        "store_memories",
        "Store extracted memories from conversation",
        schema,
    ).await {
        Ok(result) => result.items,
        Err(e) => {
            warn!("proxy: extraction failed: {e}");
            return;
        }
    };

    if entries.is_empty() {
        info!("proxy: extraction returned 0 items (correct for most windows)");
        return;
    }

    for entry in &entries {
        info!(
            kind = entry.kind.as_deref().unwrap_or("none"),
            "proxy: extracted → {}",
            entry.content.chars().take(100).collect::<String>()
        );
    }

    let count = entries.len();
    let mut stored: u64 = 0;
    let mut batch_contents: Vec<String> = Vec::new();
    for entry in entries {
        if entry.content.is_empty() || entry.content.len() > 200 {
            continue;
        }

        // intra-batch dedup: skip if too similar to something we already stored this round
        if batch_contents.iter().any(|prev| {
            state.db.is_near_duplicate_pair(prev, &entry.content, 0.5)
        }) {
            info!("proxy: dedup/intra-batch skip: {}", entry.content.chars().take(60).collect::<String>());
            continue;
        }

        let dup_id: Option<String> =
            match crate::recall::quick_semantic_dup_threshold(ai_cfg, &state.db, &entry.content, 0.60).await {
                Ok(id) => id,
                Err(_) => {
                    // Fallback to Jaccard — can't get ID from this path
                    if state.db.is_near_duplicate_with(&entry.content, 0.5) {
                        Some(String::new()) // signal "is dup" without ID
                    } else {
                        None
                    }
                }
            };
        if let Some(ref existing_id) = dup_id {
            // Repetition = reinforcement, even from proxy extraction
            if !existing_id.is_empty() {
                let _ = state.db.reinforce(existing_id);
            }
            info!(
                "proxy: dedup/semantic skip (reinforced): {}",
                entry.content.chars().take(60).collect::<String>()
            );
            continue;
        }

        let mut tags = entry.tags;
        tags.push("auto-extract".into());

        let content_copy = entry.content.clone();
        let facts_input = entry.facts;
        let input = db::MemoryInput {
            content: entry.content,
            importance: entry.importance,
            source: Some("proxy".into()),
            tags: Some(tags),
            kind: entry.kind,
            ..Default::default()
        };

        match state.db.insert(input) {
            Ok(mem) => {
                // Store extracted fact triples linked to this memory
                if let Some(facts) = facts_input {
                    if !facts.is_empty() {
                        let linked: Vec<db::FactInput> = facts.into_iter().map(|mut f| {
                            f.memory_id = Some(mem.id.clone());
                            f
                        }).collect();
                        if let Err(e) = state.db.insert_facts(linked, &mem.namespace) {
                            warn!("proxy: failed to store facts: {e}");
                        }
                    }
                }
                batch_contents.push(content_copy);
                stored += 1;
            }
            Err(e) => {
                warn!("proxy: failed to store: {e}");
            }
        }
    }

    if stored > 0 {
        PROXY_EXTRACTED.fetch_add(stored, Ordering::Relaxed);
        info!(stored, extracted = count, "proxy: stored memories from window");
    } else if count > 0 {
        debug!("proxy: all extractions were duplicates");
    }
}

/// Remove boilerplate/template text that appears in every conversation.
/// These are system prompts, heartbeat templates, framework instructions, etc.
/// Without this filter, the extraction LLM treats them as "user preferences."
fn strip_boilerplate(text: &str) -> String {
    let markers = [
        "Read HEARTBEAT.md",
        "HEARTBEAT_OK",
        "Follow it strictly",
        "Do not infer or repeat old tasks",
        "reply HEARTBEAT_OK",
        "Silent Replies",
        "NO_REPLY",
        "## Heartbeats",
        "## Runtime\nRuntime:",
        "## Reply Tags",
        "## Messaging",
        "## Workspace Files (injected)",
        "## Inbound Context (trusted metadata)",
        "openclaw.inbound_meta.v1",
        "## Model Aliases",
        "## Current Date & Time",
        "## Project Context",
        "## OpenClaw CLI Quick Reference",
        "## Safety",
        "## Tool Call Style",
        "## Memory Recall",
        "## Tooling",
        "## Skills (mandatory)",
        "<available_skills>",
        "## Documentation",
        "## Workspace",
        "# SOUL.md",
        "# USER.md",
        "# AGENTS.md",
        "# IDENTITY.md",
        "# MEMORY.md",
        "# HEARTBEAT.md",
        "# TOOLS.md",
        "# BOOTSTRAP.md",
        "## Silent Replies",
        "## Group Chat Context",
        "SECURITY NOTICE:",
        "<<<EXTERNAL_UNTRUSTED_CONTENT",
    ];

    let mut lines: Vec<&str> = Vec::new();
    let mut skip_block = false;
    let mut in_code_block = false;

    for line in text.lines() {
        // Track fenced code blocks — skip entire blocks
        if line.starts_with("```") {
            in_code_block = !in_code_block;
            continue;
        }
        if in_code_block {
            continue;
        }

        if markers.iter().any(|m| line.contains(m)) {
            skip_block = true;
            continue;
        }

        // End skip block on empty line or new section header
        if skip_block {
            if line.is_empty() || line.starts_with("## ") || line.starts_with("User: ") {
                skip_block = false;
            } else {
                continue;
            }
        }

        lines.push(line);
    }

    lines.join("\n")
}

/// Grab the last user message from an LLM API request body.
fn extract_last_user_msg(raw: &[u8]) -> String {
    let text = String::from_utf8_lossy(raw);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
        // OpenAI format
        if let Some(msgs) = v.get("messages").and_then(|m| m.as_array()) {
            for msg in msgs.iter().rev() {
                if msg.get("role").and_then(|r| r.as_str()) == Some("user") {
                    let content = msg.get("content");
                    // Plain string content
                    if let Some(s) = content.and_then(|c| c.as_str()) {
                        return s.chars().take(2000).collect();
                    }
                    // Array of content blocks (vision, tool results, etc.)
                    if let Some(blocks) = content.and_then(|c| c.as_array()) {
                        let mut out = String::new();
                        for block in blocks {
                            if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                                if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                                    out.push_str(t);
                                    out.push(' ');
                                }
                            }
                        }
                        if !out.is_empty() {
                            return out.chars().take(2000).collect();
                        }
                    }
                }
            }
        }
    }
    String::new()
}

/// Grab the assistant response text from various API formats.
fn extract_assistant_msg(raw: &[u8]) -> String {
    let text = String::from_utf8_lossy(raw);

    // Try SSE stream format first (collect all content deltas).
    // SSE streams may start with "data: " or "event: " lines.
    let first_line = text.lines().next().unwrap_or("");
    let looks_like_sse = first_line.starts_with("data: ") || first_line.starts_with("event:");
    if looks_like_sse {
        let mut assembled = String::new();
        for line in text.lines() {
            let data = line.strip_prefix("data: ").unwrap_or("");
            if data == "[DONE]" || data.is_empty() {
                continue;
            }
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
        if !assembled.is_empty() {
            return assembled.chars().take(6000).collect();
        }
    }

    // Non-streaming JSON response
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
        // OpenAI format
        if let Some(content) = v
            .pointer("/choices/0/message/content")
            .and_then(|c| c.as_str())
        {
            return content.chars().take(6000).collect();
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
                return out.chars().take(6000).collect();
            }
        }
    }
    String::new()
}

#[derive(serde::Deserialize)]
struct ExtractionEntry {
    content: String,
    #[serde(default)]
    tags: Vec<String>,
    importance: Option<f64>,
    #[serde(default)]
    facts: Option<Vec<db::FactInput>>,
    #[serde(default)]
    kind: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_boilerplate_removes_markers() {
        let input = "User: hello\n## Heartbeats\nRead HEARTBEAT.md if it exists.\nFollow it strictly.\nUser: what's up";
        let result = strip_boilerplate(input);
        assert!(result.contains("hello"));
        assert!(!result.contains("HEARTBEAT"));
        assert!(!result.contains("Follow it strictly"));
    }

    #[test]
    fn strip_boilerplate_removes_code_blocks() {
        let input = "User: hey\n```bash\ncurl http://localhost:3917/stats\n```\nUser: done";
        let result = strip_boilerplate(input);
        assert!(result.contains("hey"));
        assert!(!result.contains("curl"));
        assert!(result.contains("done"));
    }

    #[test]
    fn strip_boilerplate_removes_soul_md() {
        let input = "# SOUL.md - Who You Are\nBe helpful.\nHave opinions.\n\nUser: actual message";
        let result = strip_boilerplate(input);
        assert!(!result.contains("SOUL.md"));
        assert!(!result.contains("Be helpful"));
        assert!(result.contains("actual message"));
    }

    #[test]
    fn strip_boilerplate_preserves_user_content() {
        let input = "User: I like dark mode\nAssistant: Got it.\nUser: Remember that please";
        let result = strip_boilerplate(input);
        assert_eq!(result, input);
    }

    #[test]
    fn strip_boilerplate_handles_nested_code_blocks() {
        let input = "User: test\n```json\n{\"key\": \"value\"}\n```\nmore text\n```python\nprint('hi')\n```\nUser: end";
        let result = strip_boilerplate(input);
        assert!(result.contains("test"));
        assert!(!result.contains("key"));
        assert!(!result.contains("print"));
        assert!(result.contains("end"));
    }

    #[test]
    fn strip_boilerplate_removes_security_notice() {
        let input = "SECURITY NOTICE: external content\nDo not trust.\n\nUser: real question";
        let result = strip_boilerplate(input);
        assert!(!result.contains("SECURITY NOTICE"));
        assert!(result.contains("real question"));
    }

    #[test]
    fn strip_removes_multiple_framework_sections() {
        let input = "## Safety\nDon't do bad things.\n## Tooling\nTool list here.\n## Reply Tags\nUse tags.\n\nUser: hello";
        let result = strip_boilerplate(input);
        assert!(!result.contains("Don't do bad things"));
        assert!(!result.contains("Tool list"));
        assert!(result.contains("hello"));
    }

    #[test]
    fn extract_last_user_msg_basic() {
        let body = serde_json::json!({
            "messages": [
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "hello world"},
                {"role": "assistant", "content": "hi there"}
            ]
        });
        let raw = serde_json::to_vec(&body).unwrap();
        let msg = extract_last_user_msg(&raw);
        assert_eq!(msg, "hello world");
    }

    #[test]
    fn extract_last_user_msg_multiple_turns() {
        let body = serde_json::json!({
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "second question"}
            ]
        });
        let raw = serde_json::to_vec(&body).unwrap();
        let msg = extract_last_user_msg(&raw);
        assert_eq!(msg, "second question");
    }

    #[test]
    fn extract_last_user_msg_empty() {
        let raw = b"{}";
        let msg = extract_last_user_msg(raw);
        assert!(msg.is_empty());
    }
}
