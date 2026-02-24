use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderValue, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use futures::StreamExt;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, error, info, warn};

use crate::{ai, db, util::truncate_chars, AppState};

// Sliding window thresholds: flush when enough context accumulates.
const WINDOW_MAX_TURNS: usize = 8;
const WINDOW_MAX_CHARS: usize = 16000;
// Debounce: don't flush until this many seconds of quiet after last turn.
const FLUSH_QUIET_SECS: i64 = 30;

static PROXY_REQUESTS: AtomicU64 = AtomicU64::new(0);
static PROXY_EXTRACTED: AtomicU64 = AtomicU64::new(0);

pub fn proxy_stats(db: Option<&db::MemoryDB>) -> (u64, u64, usize) {
    let buffered = db.map(super::db::MemoryDB::proxy_turn_count).unwrap_or(0);
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

    // Deterministic session key from auth token suffix
    let session_key = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(|s| {
            let token = s.strip_prefix("Bearer ").unwrap_or(s).trim();
            // Use char boundary to avoid panic on multi-byte UTF-8
            let start = token.char_indices()
                .rev().nth(15)
                .map(|(i, _)| i)
                .unwrap_or(0);
            token[start..].to_string()
        })
        .unwrap_or_else(|| "default".into());

    // X-Engram-Extract header: callers can opt out of memory extraction.
    // Default is true (extract). Set to "false" to skip (e.g. subagent traffic).
    let extract = headers
        .get("x-engram-extract")
        .and_then(|v| v.to_str().ok())
        .map(|v| v != "false" && v != "0")
        .unwrap_or(true);

    if is_stream {
        stream_response(state, status, res_headers, upstream_res, req_capture, session_key, extract).await
    } else {
        buffered_response(state, status, res_headers, upstream_res, req_capture, session_key, extract).await
    }
}

async fn buffered_response(
    state: AppState,
    status: StatusCode,
    headers: HeaderMap,
    upstream_res: reqwest::Response,
    req_capture: Vec<u8>,
    session_key: String,
    extract: bool,
) -> Response {
    let res_bytes = match upstream_res.bytes().await {
        Ok(b) => b,
        Err(e) => {
            error!("failed to read upstream response: {e}");
            return (StatusCode::BAD_GATEWAY, "failed to read response").into_response();
        }
    };
    let res_capture = res_bytes.to_vec();

    if status.is_success() && extract {
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
    extract: bool,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Vec<u8>, std::io::Error>>(32);
    let extract_ok = status.is_success() && extract;

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
            buffer_exchange(state, req_capture, buf, session_key).await;
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
    let assistant_msg = extract_assistant_msg(&res_raw);

    // Parse the messages array and apply watermark-based filtering
    let user_msg = match extract_user_via_watermark(&state, &req_raw, &session_key) {
        Some(msg) => msg,
        None => {
            // First request, context truncation, or no parseable messages.
            // Still no user message to pair — skip this exchange.
            if assistant_msg.is_empty() {
                return;
            }
            String::new()
        }
    };

    if user_msg.is_empty() && assistant_msg.is_empty() {
        return;
    }

    // Skip very short tool-use-only turns
    if user_msg.len() + assistant_msg.len() < 80 {
        return;
    }

    let turn = format!("User: {}\nAssistant: {}", user_msg, assistant_msg);

    if let Err(e) = state.db.save_proxy_turn(&session_key, &turn) {
        warn!("proxy: failed to persist turn: {e}");
    }
    state.last_proxy_turn.store(
        crate::db::now_ms(),
        std::sync::atomic::Ordering::Relaxed,
    );

    if state.db.proxy_session_should_flush(&session_key, WINDOW_MAX_TURNS, WINDOW_MAX_CHARS)
        && state.db.proxy_session_quiet_for(&session_key, FLUSH_QUIET_SECS)
    {
        if let Ok(ctx) = state.db.drain_proxy_session(&session_key) {
            if !ctx.is_empty() {
                extract_from_context(state, &ctx).await;
            }
        }
    }
}

/// Apply watermark logic to extract new user messages from the request.
/// Returns `None` if this is the first request or a context truncation (watermark reset).
/// Returns `Some(text)` with the concatenated new user messages otherwise.
fn extract_user_via_watermark(state: &AppState, req_raw: &[u8], session_key: &str) -> Option<String> {
    let text = String::from_utf8_lossy(req_raw);
    let body: serde_json::Value = serde_json::from_str(&text).ok()?;

    let messages = body.get("messages")?.as_array()?;
    if messages.is_empty() {
        return None;
    }

    // Keep only user and assistant messages
    let filtered: Vec<&serde_json::Value> = messages
        .iter()
        .filter(|m| {
            matches!(
                m.get("role").and_then(|r| r.as_str()),
                Some("user") | Some("assistant")
            )
        })
        .collect();

    let msg_count = filtered.len() as i64;
    let watermark = state.db.get_watermark(session_key).unwrap_or(0);

    if watermark == 0 {
        // First request: mark everything as seen, extract nothing
        let _ = state.db.set_watermark(session_key, msg_count);
        return None;
    }

    if msg_count <= watermark {
        // Context window was truncated — reset and skip this round
        let _ = state.db.set_watermark(session_key, msg_count);
        return None;
    }

    // Extract user content from messages after the watermark
    let new_msgs = &filtered[watermark as usize..];
    let mut parts: Vec<String> = Vec::new();
    for msg in new_msgs {
        if msg.get("role").and_then(|r| r.as_str()) == Some("user") {
            if let Some(content) = extract_message_content(msg) {
                if !content.is_empty() {
                    parts.push(content);
                }
            }
        }
    }

    let _ = state.db.set_watermark(session_key, msg_count);

    if parts.is_empty() {
        return Some(String::new());
    }

    Some(truncate_chars(&parts.join("\n"), 6000))
}

/// Extract text content from a single message object.
/// Handles both plain string and array-of-blocks formats.
fn extract_message_content(msg: &serde_json::Value) -> Option<String> {
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

    let preview = truncate_chars(context, 12000);

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
                let truncated = truncate_chars(&m.content, 120);
                format!("- {}", truncated)
            })
            .collect();
        format!("\n\n=== ALREADY IN MEMORY (do NOT extract anything that overlaps with these) ===\n{}\n", lines.join("\n"))
    };

    let system = "You extract long-term memories from a multi-turn LLM conversation.\n\
        Your bar is high — most windows produce 0 extractions and that's fine.\n\
        But when real decisions happen, capture them (typically 0-1 per window).\n\n\
        ASK YOURSELF: Would a senior engineer jot this down for the team wiki? If yes, extract it.\n\n\
        EXTRACT these (when the USER states or confirms them):\n\
        - Major architectural decisions ('we chose X over Y', 'switching from Docker to native binaries')\n\
        - Infrastructure changes that affect how the project works ('repo moved to new-org', 'CI now produces binary releases')\n\
        - User-stated constraints or rules ('no Docker', 'code must look human-written')\n\
        - Technology/tool choices and their rationale\n\
        - Warnings or gotchas discovered through real experience ('raw.githubusercontent.com is blocked by GFW')\n\
        - Project milestones and significant state changes ('v2 released', 'migrated to new account')\n\n\
        NEVER extract (automatic reject):\n\
        - Routine code changes (refactors, cleanups, renames)\n\
        - Bug descriptions and their fixes\n\
        - System health, metrics, test results\n\
        - The assistant's own proposals or explanations — only extract what the USER stated\n\
        - Step-by-step implementation details\n\
        - Anything that overlaps with ALREADY IN MEMORY below\n\n\
        KEY DISTINCTION:\n\
        'We decided to use Redis for caching' → EXTRACT.\n\
        'Fixed the Redis connection timeout bug' → REJECT.\n\
        'We moved the repo to kael-bit' → EXTRACT.\n\
        'Renamed main.rs to lib.rs' → REJECT.\n\n\
        LANGUAGE: Write in the same language the USER uses. If the user speaks Chinese, output Chinese.\n\n\
        DEDUP: If your extraction overlaps with something in ALREADY IN MEMORY, skip it.";

    let user = format!(
        "Extract 0-1 memories from this conversation window. Zero is expected for routine work. Only extract if there's a genuine decision, constraint, or discovery worth preserving. Call with empty items if nothing qualifies.{existing_knowledge}\n\n\
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
                "maxItems": 3
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
            truncate_chars(&entry.content, 100)
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
            crate::db::jaccard_similar(prev, &entry.content, 0.5)
        }) {
            info!("proxy: dedup/intra-batch skip: {}", truncate_chars(&entry.content, 60));
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
                truncate_chars(&entry.content, 60)
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
            return truncate_chars(&assembled, 6000);
        }
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
#[path = "proxy_tests.rs"]
mod tests;
