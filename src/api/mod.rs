use axum::extract::{Query, State};
use axum::http::Request;
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use subtle::ConstantTimeEq;
use tower_http::limit::RequestBodyLimitLayer;
use tracing::warn;

use crate::error::EngramError;
use crate::{ai, db, AppState};

mod admin;
mod facts;
mod memory;
mod recall_handlers;

use admin::*;
use facts::*;
use memory::*;
use recall_handlers::*;

/// Run a blocking closure on the spawn_blocking pool and map JoinError.
async fn blocking<T, F>(f: F) -> Result<T, EngramError>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| EngramError::Internal(e.to_string()))
}

/// Extract namespace from X-Namespace header, defaulting to None (= all namespaces).
fn read_rss_kb() -> u64 {
    // Linux: read from /proc
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/self/statm")
            .ok()
            .and_then(|s| s.split_whitespace().nth(1)?.parse::<u64>().ok())
            .map(|pages| pages * 4)
            .unwrap_or(0)
    }
    // macOS: use mach task_info (would need mach crate, not worth the dep)
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

fn get_namespace(headers: &axum::http::HeaderMap) -> Option<String> {
    headers
        .get("x-namespace")
        .and_then(|v| v.to_str().ok())
        .map(std::string::ToString::to_string)
        .filter(|s| !s.is_empty())
}

/// Auth middleware: checks Bearer token if ENGRAM_API_KEY is configured.
async fn require_auth(
    State(state): State<AppState>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, EngramError> {
    let Some(ref expected) = state.api_key else {
        return Ok(next.run(req).await);
    };

    let unauthorized = || EngramError::Unauthorized;

    let header = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(unauthorized)?;

    let token = header.strip_prefix("Bearer ").ok_or_else(unauthorized)?;

    // constant-time comparison to prevent timing attacks
    if token.as_bytes().ct_eq(expected.as_bytes()).into() {
        Ok(next.run(req).await)
    } else {
        Err(unauthorized())
    }
}

pub fn router(state: AppState) -> Router {
    let public = Router::new()
        .route("/", get(index))
        .route("/health", get(health_only))
        .route("/stats", get(stats))
        .route("/ui", get(serve_ui));

    let protected = Router::new()
        .route("/memories", post(create_memory).get(list_memories).delete(batch_delete))
        .route("/memories/batch", post(batch_create))
        .route(
            "/memories/{id}",
            get(get_memory).patch(update_memory).delete(delete_memory),
        )
        .route("/recall", post(do_recall))
        .route("/search", get(quick_search))
        .route("/recent", get(list_recent))
        .route("/resume", get(do_resume))
        .route("/triggers/{action}", get(get_triggers))
        .route("/consolidate", post(do_consolidate))
        .route("/audit", post(do_audit))
        .route("/audit/sandbox", post(do_audit_sandbox))
        .route("/repair", post(do_repair))
        .route("/vacuum", post(do_vacuum))
        .route("/extract", post(do_extract))
        .route("/export", get(do_export))
        .route("/facts", post(create_facts).get(query_facts))
        .route("/facts/all", get(list_all_facts))
        .route("/facts/graph", get(query_graph))
        .route("/facts/conflicts", get(get_fact_conflicts))
        .route("/facts/history", get(get_fact_history))
        .route("/facts/{id}", delete(delete_fact))
        .route("/trash", get(trash_list).delete(trash_purge))
        .route("/trash/{id}/restore", post(trash_restore))
        .route("/llm-usage", get(llm_usage))
        .layer(middleware::from_fn_with_state(state.clone(), require_auth));

    // Import needs a bigger body limit for exports with embeddings
    let import_route = Router::new()
        .route("/import", post(do_import))
        .layer(middleware::from_fn_with_state(state.clone(), require_auth))
        .layer(RequestBodyLimitLayer::new(32 * 1024 * 1024)); // 32MB

    // Proxy route — transparent forwarding, no auth (clients bring their own API keys).
    // 10MB body limit covers large prompts.
    // flush/window are management endpoints — they need auth.
    let proxy_route = Router::new()
        .route("/proxy/{*path}", axum::routing::any(crate::proxy::handle))
        .layer(RequestBodyLimitLayer::new(10 * 1024 * 1024));
    let proxy_admin = Router::new()
        .route("/proxy/flush", axum::routing::post(proxy_flush))
        .route("/proxy/window", axum::routing::get(proxy_window))
        .layer(middleware::from_fn_with_state(state.clone(), require_auth))
        .layer(RequestBodyLimitLayer::new(64 * 1024));

    // 64KB for normal operations, 32MB for import, 10MB for proxy
    public
        .merge(protected)
        .layer(RequestBodyLimitLayer::new(64 * 1024))
        .merge(import_route)
        .merge(proxy_admin)
        .merge(proxy_route)
        .with_state(state)
}

async fn serve_ui() -> impl axum::response::IntoResponse {
    axum::response::Html(include_str!("../../web/index.html"))
}

/// Shared health data (without endpoints) used by both `/` and `/health`.
async fn health_data(state: &AppState) -> serde_json::Value {
    let db = state.db.clone();
    let (s, integrity, db_size_mb) = blocking(move || {
        let s = db.stats();
        let i = db.integrity();
        let bytes = db.db_size_bytes();
        let mb = (bytes as f64 / 1048576.0 * 10.0).round() / 10.0;
        (s, i, mb)
    })
        .await
        .unwrap_or((
            db::Stats { total: 0, buffer: 0, working: 0, core: 0, by_kind: db::KindStats::default() },
            db::IntegrityReport::default(),
            0.0,
        ));

    let uptime_secs = state.started_at.elapsed().as_secs();
    let rss_kb = read_rss_kb();
    let (cache_len, cache_cap, cache_hits, cache_misses) = state.embed_cache.stats();

    let (proxy_reqs, proxy_extracted, proxy_buffered) = crate::proxy::proxy_stats(Some(&state.db));

    serde_json::json!({
        "name": "engram",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_secs": uptime_secs,
        "rss_kb": rss_kb,
        "db_size_mb": db_size_mb,
        "ai_enabled": state.ai.is_some(),
        "embed_cache": { "size": cache_len, "capacity": cache_cap, "hits": cache_hits, "misses": cache_misses },
        "proxy": {
            "enabled": state.proxy.is_some(),
            "requests": proxy_reqs,
            "extracted": proxy_extracted,
            "buffered_turns": proxy_buffered,
        },
        "integrity": integrity,
        "stats": s,
    })
}

/// GET / — full index with health data + endpoint list.
async fn index(State(state): State<AppState>) -> Json<serde_json::Value> {
    let mut data = health_data(&state).await;
    if let Some(obj) = data.as_object_mut() {
        obj.insert("endpoints".to_string(), serde_json::json!({
            "GET /": "index with health data + endpoint list",
            "GET /health": "health only (uptime, rss, cache, integrity — no endpoints)",
            "GET /stats": "memory counts per layer",
            "POST /memories": "create a memory",
            "POST /memories/batch": "batch create memories (body: [{content, ...}, ...])",
            "GET /memories": "list memories (optional ?layer=N&tag=X&limit=N)",
            "GET /memories/:id": "get a memory by id",
            "PATCH /memories/:id": "update a memory",
            "DELETE /memories/:id": "delete a memory",
            "DELETE /memories": "batch delete (body: {ids: [...]} or {namespace: 'x'})",
            "POST /recall": "hybrid search (semantic + keyword)",
            "GET /search?q=term": "quick keyword search",
            "GET /recent?hours=2": "recent memories by time",
            "GET /resume?hours=4&workspace=tags&limit=100": "full memory bootstrap (core + working + buffer + recent + sessions)",
            "GET /triggers/:action": "pre-action recall (e.g. /triggers/git-push)",
            "POST /consolidate": "run maintenance cycle",
            "POST /audit": "LLM-powered memory reorganization (uses ENGRAM_GATE_MODEL)",
            "POST /audit/sandbox": "Dry-run audit with quality grading (no changes applied)",
            "POST /repair": "auto-repair FTS index; ?force=true for full rebuild",
            "POST /vacuum": "reclaim disk space (?full=true for full vacuum)",
            "POST /extract": "LLM-extract memories from text",
            "GET /export": "export all memories (?embed=true to include vectors)",
            "POST /import": "import memories from JSON",
            "POST /facts": "insert fact triples",
            "GET /facts?entity=X": "query facts by entity",
            "GET /facts/all": "list all facts",
            "GET /facts/graph?entity=X&hops=2": "multi-hop graph traversal from entity",
            "GET /facts/conflicts?subject=X&predicate=Y": "check fact conflicts",
            "GET /facts/history?subject=X&predicate=Y": "fact history with superseded entries",
            "DELETE /facts/:id": "delete a fact",
            "GET /trash": "list soft-deleted memories (?limit=100)",
            "POST /trash/:id/restore": "restore a memory from trash",
            "DELETE /trash": "permanently purge all trash",
            "ANY /proxy/*": "transparent LLM proxy (requires ENGRAM_PROXY_UPSTREAM)",
            "POST /proxy/flush": "flush proxy sliding window for current session",
            "GET /proxy/window": "view proxy sliding window for current session",
            "GET /ui": "web dashboard",
        }));
    }
    Json(data)
}

/// GET /health — health data only (no endpoint list).
async fn health_only(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(health_data(&state).await)
}

async fn stats(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Query(q): Query<std::collections::HashMap<String, String>>,
) -> Json<serde_json::Value> {
    let ns = q.get("ns").or_else(|| q.get("namespace")).cloned().or_else(|| get_namespace(&headers));
    let db = state.db.clone();
    let is_global = ns.is_none();
    let result = blocking(move || {
        let s = match &ns {
            Some(n) => db.stats_ns(n),
            None => db.stats(),
        };
        let mut v = serde_json::to_value(&s).unwrap_or_default();
        if is_global {
            let nss = db.list_namespaces().unwrap_or_default();
            v["namespaces"] = serde_json::json!(nss);
        }
        v
    })
    .await
    .unwrap_or_else(|_| serde_json::json!({"total":0,"buffer":0,"working":0,"core":0}));
    Json(result)
}

/// Fire-and-forget: generate embedding for a memory in the background.
fn spawn_embed(db: crate::SharedDB, cfg: ai::AiConfig, id: String, content: String) {
    tokio::spawn(async move {
        let mut attempts = 0;
        loop {
            attempts += 1;
            match ai::get_embeddings(&cfg, std::slice::from_ref(&content)).await {
                Ok(er) if !er.embeddings.is_empty() => {
                    if let Some(ref u) = er.usage {
                        let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
                        let _ = db.log_llm_call("embed", &cfg.embed_model, u.prompt_tokens, u.completion_tokens, cached, 0);
                    }
                    if let Some(emb) = er.embeddings.into_iter().next() {
                        let _ = tokio::task::spawn_blocking(move || {
                            db.set_embedding(&id, &emb)
                        })
                        .await;
                    }
                    return;
                }
                Err(e) if attempts < 3 => {
                    warn!(error = %e, attempt = attempts, "embedding failed, retrying");
                    tokio::time::sleep(std::time::Duration::from_secs(attempts * 2)).await;
                }
                Err(e) => {
                    warn!(error = %e, id = %id, "embedding failed after 3 attempts");
                    return;
                }
                _ => return,
            }
        }
    });
}

/// Batch embed: generate embeddings for multiple memories at once.
fn spawn_embed_batch(db: crate::SharedDB, cfg: ai::AiConfig, items: Vec<(String, String)>) {
    if items.is_empty() {
        return;
    }
    tokio::spawn(async move {
        let texts: Vec<String> = items.iter().map(|(_, c)| c.clone()).collect();
        let mut attempts = 0;
        loop {
            attempts += 1;
            match ai::get_embeddings(&cfg, &texts).await {
                Ok(er) => {
                    if let Some(ref u) = er.usage {
                        let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
                        let _ = db.log_llm_call("embed_batch", &cfg.embed_model, u.prompt_tokens, u.completion_tokens, cached, 0);
                    }
                    for (emb, (id, _)) in er.embeddings.into_iter().zip(items.iter()) {
                        let db = db.clone();
                        let id = id.clone();
                        let _ = tokio::task::spawn_blocking(move || {
                            db.set_embedding(&id, &emb)
                        })
                        .await;
                    }
                    return;
                }
                Err(e) if attempts < 3 => {
                    warn!(error = %e, attempt = attempts, "batch embedding failed, retrying");
                    tokio::time::sleep(std::time::Duration::from_secs(attempts * 2)).await;
                }
                Err(e) => {
                    warn!(error = %e, "batch embedding failed after 3 attempts");
                    return;
                }
            }
        }
    });
}


#[cfg(test)]
#[path = "api_tests.rs"]
mod api_tests;
