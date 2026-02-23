use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderValue, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use futures::StreamExt;
use tracing::{debug, error, info, warn};

use crate::{ai, db, AppState};

#[derive(Clone)]
pub struct ProxyConfig {
    pub upstream: String,
    pub default_key: Option<String>,
}

/// Transparent proxy: forward everything to upstream, capture req/res bodies,
/// async-extract memories. Works with any LLM provider — we don't parse protocols.
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

    // Build upstream URL: /proxy/v1/chat/completions → upstream + /v1/chat/completions
    let path = uri.path().strip_prefix("/proxy").unwrap_or(uri.path());
    let upstream_url = if let Some(q) = uri.query() {
        format!("{}{}?{}", proxy.upstream, path, q)
    } else {
        format!("{}{}", proxy.upstream, path)
    };

    // Collect request body so we can both forward and capture it
    let req_bytes = match axum::body::to_bytes(body, 10 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            error!("failed to read request body: {e}");
            return (StatusCode::BAD_REQUEST, "failed to read body").into_response();
        }
    };
    let req_capture = req_bytes.to_vec();

    // Build upstream request, forward most headers
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

    // If no auth header and we have a default key, inject it
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

    // Copy response headers
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

    if is_stream {
        stream_response(state, status, res_headers, upstream_res, req_capture).await
    } else {
        buffered_response(state, status, res_headers, upstream_res, req_capture).await
    }
}

async fn buffered_response(
    state: AppState,
    status: StatusCode,
    headers: HeaderMap,
    upstream_res: reqwest::Response,
    req_capture: Vec<u8>,
) -> Response {
    let res_bytes = match upstream_res.bytes().await {
        Ok(b) => b,
        Err(e) => {
            error!("failed to read upstream response: {e}");
            return (StatusCode::BAD_GATEWAY, "failed to read response").into_response();
        }
    };
    let res_capture = res_bytes.to_vec();

    // Async extract memories
    if status.is_success() {
        tokio::spawn(extract_memories(state, req_capture, res_capture));
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
                        break; // client disconnected
                    }
                }
                Err(e) => {
                    warn!("stream chunk error: {e}");
                    let _ = tx
                        .send(Err(std::io::Error::other(e.to_string())))
                        .await;
                    break;
                }
            }
        }
        drop(tx);

        // Extract from completed stream
        if !buf.is_empty() {
            extract_memories(state_clone, req_capture, buf).await;
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

async fn extract_memories(state: AppState, req_raw: Vec<u8>, res_raw: Vec<u8>) {
    let Some(ref ai_cfg) = state.ai else {
        debug!("proxy: no AI config, skipping extraction");
        return;
    };

    let req_text = String::from_utf8_lossy(&req_raw);
    let res_text = String::from_utf8_lossy(&res_raw);

    // Skip tiny exchanges — nothing worth extracting
    if req_text.len() + res_text.len() < 200 {
        debug!("proxy: skipping extraction, exchange too short");
        return;
    }

    // Truncate to avoid blowing up the extraction LLM
    let max_chars = 12000;
    let req_preview: String = req_text.chars().take(max_chars).collect();
    let res_preview: String = res_text.chars().take(max_chars).collect();

    let system = "You extract memories from LLM conversations. Be extremely selective.\n\
        Rules:\n\
        - MAX 3 items per extraction. Quality over quantity.\n\
        - Only extract NEW information: concrete decisions, specific preferences, \
          learned lessons, or critical facts.\n\
        - Each item must be specific and actionable, not vague summaries.\n\
        - Skip: anything that sounds like a summary or recap of prior context.\n\
        - Skip: routine technical details, code snippets, debugging steps.\n\
        - Skip: general knowledge that anyone could look up.\n\
        - If nothing genuinely new and important was discussed, return [].\n\
        Return a JSON array of {\"content\": \"...\", \"tags\": [\"...\"]}. Keep content under 200 chars.";

    let user = format!(
        "Extract at most 3 genuinely new, important memories from this exchange.\n\
         If this is just routine conversation with nothing worth remembering, return [].\n\n\
         === REQUEST ===\n{req_preview}\n\n=== RESPONSE ===\n{res_preview}"
    );

    let extraction = match ai::llm_chat_as(ai_cfg, "proxy", system, &user).await {
        Ok(text) => text,
        Err(e) => {
            warn!("proxy: extraction LLM call failed: {e}");
            return;
        }
    };

    // Parse the extraction result
    let cleaned = extraction
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let entries: Vec<ExtractionEntry> = match serde_json::from_str(cleaned) {
        Ok(e) => e,
        Err(e) => {
            debug!("proxy: couldn't parse extraction result: {e}");
            return;
        }
    };

    if entries.is_empty() {
        debug!("proxy: nothing worth extracting");
        return;
    }

    let count = entries.len();
    for entry in entries {
        if entry.content.is_empty() {
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
            warn!("proxy: failed to store extracted memory: {e}");
        }
    }

    info!(count, "proxy: extracted and stored memories");
}

#[derive(serde::Deserialize)]
struct ExtractionEntry {
    content: String,
    #[serde(default)]
    tags: Vec<String>,
}
