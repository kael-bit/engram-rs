use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderValue, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use futures::StreamExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use tracing::{debug, error, info, warn};

use crate::{ai, db, AppState};

static PROXY_REQUESTS: AtomicU64 = AtomicU64::new(0);
static PROXY_EXTRACTED: AtomicU64 = AtomicU64::new(0);

pub fn proxy_stats() -> (u64, u64, usize) {
    let buffered = {
        let guard = WINDOWS.lock().unwrap();
        guard.as_ref().map(|m| m.values().map(|w| w.turns.len()).sum()).unwrap_or(0)
    };
    (
        PROXY_REQUESTS.load(Ordering::Relaxed),
        PROXY_EXTRACTED.load(Ordering::Relaxed),
        buffered,
    )
}

use std::collections::HashMap;

// Sliding window: buffer recent exchanges per conversation, extract when
// enough context accumulates. Keyed by auth token to separate different callers.
const WINDOW_MAX_TURNS: usize = 5;
const WINDOW_MAX_CHARS: usize = 8000;

struct ConversationWindow {
    turns: Vec<String>,
    total_chars: usize,
}

impl ConversationWindow {
    fn new() -> Self {
        Self { turns: Vec::new(), total_chars: 0 }
    }

    fn push(&mut self, turn: String) {
        self.total_chars += turn.len();
        self.turns.push(turn);
    }

    fn should_flush(&self) -> bool {
        self.turns.len() >= WINDOW_MAX_TURNS || self.total_chars >= WINDOW_MAX_CHARS
    }

    fn drain(&mut self) -> String {
        let text = self.turns.join("\n---\n");
        self.turns.clear();
        self.total_chars = 0;
        text
    }

    fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }
}

static WINDOWS: Mutex<Option<HashMap<String, ConversationWindow>>> = Mutex::new(None);

fn with_window<F, R>(key: &str, f: F) -> R
where
    F: FnOnce(&mut ConversationWindow) -> R,
{
    let mut guard = WINDOWS.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    let w = map.entry(key.to_string()).or_insert_with(ConversationWindow::new);
    f(w)
}

fn drain_all_windows() -> Vec<(String, String)> {
    let mut guard = WINDOWS.lock().unwrap();
    let Some(map) = guard.as_mut() else { return vec![] };
    let mut results = Vec::new();
    for (key, w) in map.iter_mut() {
        if !w.is_empty() {
            results.push((key.clone(), w.drain()));
        }
    }
    // remove empty entries
    map.retain(|_, w| !w.is_empty());
    results
}

#[derive(Clone)]
pub struct ProxyConfig {
    pub upstream: String,
    pub default_key: Option<String>,
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

    let client = reqwest::Client::new();
    let mut upstream_req = client.request(
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

    // Use auth token (last 8 chars) as session key to separate conversation windows
    let session_key = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(|s| {
            let s = s.strip_prefix("Bearer ").unwrap_or(s);
            if s.len() > 8 { s[s.len()-8..].to_string() } else { s.to_string() }
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

    tokio::spawn(async move {
        let mut buf = Vec::new();
        let mut stream = upstream_res.bytes_stream();

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    buf.extend_from_slice(&bytes);
                    if tx.send(Ok(bytes.to_vec())).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    warn!("stream chunk error: {e}");
                    let _ = tx.send(Err(std::io::Error::other(e.to_string()))).await;
                    break;
                }
            }
        }
        drop(tx);

        if !buf.is_empty() {
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

    // Skip very short tool-use-only turns
    if user_msg.len() + assistant_msg.len() < 80 {
        return;
    }

    let turn = format!("User: {}\nAssistant: {}", user_msg, assistant_msg);

    let context = with_window(&session_key, |w| {
        w.push(turn);
        if w.should_flush() {
            Some(w.drain())
        } else {
            None
        }
    });

    if let Some(ctx) = context {
        extract_from_context(state, &ctx).await;
    }
}

/// Flush all remaining buffered turns. Called from background timer.
pub async fn flush_window(state: &AppState) {
    let pending = drain_all_windows();
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

    if context.len() < 100 {
        debug!("proxy: context too thin to extract from");
        return;
    }

    let preview: String = context.chars().take(12000).collect();

    let system = "You extract long-term memories from a multi-turn LLM conversation. Be EXTREMELY selective.\n\
        Most conversations have NOTHING worth extracting. Return [] by default.\n\n\
        You're seeing several turns of conversation. Use the full context to understand what's important.\n\n\
        Only extract if you find:\n\
        - A USER's stated preference, rule, or boundary (not the assistant's)\n\
        - A concrete decision that changes how something works going forward\n\
        - A hard-won lesson from a mistake or failure\n\
        - A critical fact that would be painful to rediscover\n\n\
        NEVER extract:\n\
        - Task descriptions, instructions, or assignments\n\
        - Technical implementation details (code, config, API calls)\n\
        - Summaries or recaps of what was done\n\
        - The assistant's suggestions or explanations\n\
        - Anything about UI, styling, or frontend requirements\n\
        - System prompts or context that was injected\n\
        - Things the assistant already knows from context\n\n\
        MAX 3 items per conversation window. Return JSON array of {\"content\": \"...\", \"tags\": [\"...\"]}.\n\
        Content must be under 150 chars — one sentence, concrete, actionable.";

    let user = format!(
        "Extract 0-3 genuinely important long-term memories from this conversation. Default to [].\n\n\
         === CONVERSATION ({} turns) ===\n{preview}",
        context.matches("User: ").count()
    );

    let extraction = match ai::llm_chat_as(ai_cfg, "proxy", system, &user).await {
        Ok(text) => text,
        Err(e) => {
            warn!("proxy: extraction failed: {e}");
            return;
        }
    };

    let cleaned = extraction
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let entries: Vec<ExtractionEntry> = match serde_json::from_str(cleaned) {
        Ok(e) => e,
        Err(e) => {
            debug!("proxy: parse error: {e}");
            return;
        }
    };

    if entries.is_empty() {
        debug!("proxy: nothing worth extracting from window");
        return;
    }

    let count = entries.len();
    let mut stored: u64 = 0;
    for entry in entries {
        if entry.content.is_empty() || entry.content.len() > 300 {
            continue;
        }

        let is_dup =
            match crate::recall::quick_semantic_dup(ai_cfg, &state.db, &entry.content).await {
                Ok(dup) => dup,
                Err(_) => state.db.is_near_duplicate_with(&entry.content, 0.5),
            };
        if is_dup {
            debug!(
                "proxy: skipping duplicate: {}",
                &entry.content[..entry.content.len().min(60)]
            );
            continue;
        }

        let mut tags = entry.tags;
        tags.push("auto-extract".into());

        let input = db::MemoryInput {
            content: entry.content,
            source: Some("proxy".into()),
            tags: Some(tags),
            ..Default::default()
        };

        if let Err(e) = state.db.insert(input) {
            warn!("proxy: failed to store: {e}");
        } else {
            stored += 1;
        }
    }

    if stored > 0 {
        PROXY_EXTRACTED.fetch_add(stored, Ordering::Relaxed);
        info!(stored, extracted = count, "proxy: stored memories from window");
    } else if count > 0 {
        debug!("proxy: all extractions were duplicates");
    }
}

/// Grab the last user message from an LLM API request body.
fn extract_last_user_msg(raw: &[u8]) -> String {
    let text = String::from_utf8_lossy(raw);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
        // OpenAI format
        if let Some(msgs) = v.get("messages").and_then(|m| m.as_array()) {
            for msg in msgs.iter().rev() {
                if msg.get("role").and_then(|r| r.as_str()) == Some("user") {
                    if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                        return content.chars().take(2000).collect();
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

    // Try SSE stream format first (collect all content deltas)
    if text.starts_with("data: ") || text.contains("\ndata: ") {
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
            return assembled.chars().take(2000).collect();
        }
    }

    // Non-streaming JSON response
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
        // OpenAI format
        if let Some(content) = v
            .pointer("/choices/0/message/content")
            .and_then(|c| c.as_str())
        {
            return content.chars().take(2000).collect();
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
                return out.chars().take(2000).collect();
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
}
