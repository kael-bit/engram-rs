mod extract;
pub mod parse;
mod stats;
pub mod window;

pub use stats::{init_proxy_counters, proxy_stats};
pub use window::flush_window;

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderValue, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use futures::StreamExt;
use tracing::{error, warn};

use crate::AppState;

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
    stats::bump_requests();
    stats::persist_proxy_counters(&state.db);

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
            "host" | "connection" | "transfer-encoding" | "content-length" | "authorization"
        );
        if !skip {
            if let Ok(v) = value.to_str() {
                upstream_req = upstream_req.header(name.as_str(), v);
            }
        }
    }

    if let Some(ref key) = proxy.default_key {
        upstream_req = upstream_req.header("authorization", format!("Bearer {key}"));
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
            let start = token.char_indices()
                .rev().nth(15)
                .map(|(i, _)| i)
                .unwrap_or(0);
            token[start..].to_string()
        })
        .unwrap_or_else(|| "default".into());

    // X-Engram-Extract header: callers can opt out of memory extraction.
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
        tokio::spawn(window::buffer_exchange(state, req_capture, res_capture, session_key));
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
            window::buffer_exchange(state, req_capture, buf, session_key).await;
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


#[cfg(test)]
#[path = "proxy_tests.rs"]
mod tests;
