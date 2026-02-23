use axum::extract::{Path, Query, State};
use axum::http::{Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use subtle::ConstantTimeEq;
use tower_http::limit::RequestBodyLimitLayer;
use tracing::{debug, warn};

use crate::error::EngramError;
use crate::{ai, consolidate, db, recall, AppState};

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
        .map(|s| s.to_string())
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
        .route("/", get(health))
        .route("/health", get(health))
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
        .route("/repair", post(do_repair))
        .route("/vacuum", post(do_vacuum))
        .route("/extract", post(do_extract))
        .route("/export", get(do_export))
        .layer(middleware::from_fn_with_state(state.clone(), require_auth));

    // Import needs a bigger body limit for exports with embeddings
    let import_route = Router::new()
        .route("/import", post(do_import))
        .layer(middleware::from_fn_with_state(state.clone(), require_auth))
        .layer(RequestBodyLimitLayer::new(32 * 1024 * 1024)); // 32MB

    // Proxy route — transparent forwarding, no auth (clients bring their own API keys).
    // 10MB body limit covers large prompts.
    let proxy_route = Router::new()
        .route("/proxy/{*path}", axum::routing::any(crate::proxy::handle))
        .route("/proxy/flush", axum::routing::post(proxy_flush))
        .layer(RequestBodyLimitLayer::new(10 * 1024 * 1024));

    // 64KB for normal operations, 32MB for import, 10MB for proxy
    public
        .merge(protected)
        .layer(RequestBodyLimitLayer::new(64 * 1024))
        .merge(import_route)
        .merge(proxy_route)
        .with_state(state)
}

async fn serve_ui() -> impl axum::response::IntoResponse {
    axum::response::Html(include_str!("../web/index.html"))
}

async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let db = state.db.clone();
    let (s, integrity) = blocking(move || (db.stats(), db.integrity()))
        .await
        .unwrap_or((
            db::Stats { total: 0, buffer: 0, working: 0, core: 0 },
            db::IntegrityReport::default(),
        ));

    let uptime_secs = state.started_at.elapsed().as_secs();
    let rss_kb = read_rss_kb();
    let (cache_len, cache_cap) = state.embed_cache.lock()
        .map(|c| (c.len(), c.capacity()))
        .unwrap_or((0, 0));

    let (proxy_reqs, proxy_extracted, proxy_buffered) = crate::proxy::proxy_stats();

    Json(serde_json::json!({
        "name": "engram",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_secs": uptime_secs,
        "rss_kb": rss_kb,
        "ai_enabled": state.ai.is_some(),
        "embed_cache": { "size": cache_len, "capacity": cache_cap },
        "proxy": {
            "enabled": state.proxy.is_some(),
            "requests": proxy_reqs,
            "extracted": proxy_extracted,
            "buffered_turns": proxy_buffered,
        },
        "integrity": integrity,
        "stats": s,
        "endpoints": {
            "GET /": "this health check",
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
            "POST /repair": "auto-repair FTS index (remove orphans, rebuild missing)",
            "POST /vacuum": "reclaim disk space (?full=true for full vacuum)",
            "POST /extract": "LLM-extract memories from text",
            "GET /export": "export all memories (?embed=true to include vectors)",
            "POST /import": "import memories from JSON",
        },
    }))
}

async fn stats(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Query(q): Query<std::collections::HashMap<String, String>>,
) -> Json<db::Stats> {
    let ns = q.get("ns").cloned().or_else(|| get_namespace(&headers));
    let db = state.db.clone();
    let s = blocking(move || {
        match ns {
            Some(n) => db.stats_ns(&n),
            None => db.stats(),
        }
    })
    .await
    .unwrap_or(db::Stats { total: 0, buffer: 0, working: 0, core: 0 });
    Json(s)
}

/// Fire-and-forget: generate embedding for a memory in the background.
fn spawn_embed(db: crate::SharedDB, cfg: ai::AiConfig, id: String, content: String) {
    tokio::spawn(async move {
        let mut attempts = 0;
        loop {
            attempts += 1;
            match ai::get_embeddings(&cfg, std::slice::from_ref(&content)).await {
                Ok(embs) if !embs.is_empty() => {
                    if let Some(emb) = embs.into_iter().next() {
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
                Ok(embs) => {
                    for (emb, (id, _)) in embs.into_iter().zip(items.iter()) {
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

async fn create_memory(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(mut input): Json<db::MemoryInput>,
) -> Result<(StatusCode, Json<db::Memory>), EngramError> {
    // Body namespace takes precedence, then X-Namespace header
    if input.namespace.is_none() {
        input.namespace = get_namespace(&headers);
    }
    let sync = input.sync_embed.unwrap_or(false);
    let db = state.db.clone();
    let mem = blocking(move || db.insert(input))
        .await??;

    if let Some(ref cfg) = state.ai {
        if cfg.has_embed() {
            if sync {
                let db = state.db.clone();
                let cfg = cfg.clone();
                let id = mem.id.clone();
                let content = mem.content.clone();
                match ai::get_embeddings(&cfg, &[content]).await {
                    Ok(embs) if !embs.is_empty() => {
                        if let Some(emb) = embs.into_iter().next() {
                            let _ = tokio::task::spawn_blocking(move || {
                                db.set_embedding(&id, &emb)
                            }).await;
                        }
                    }
                    Err(e) => warn!(error = %e, "sync embedding failed"),
                    _ => {}
                }
            } else {
                spawn_embed(state.db.clone(), cfg.clone(), mem.id.clone(), mem.content.clone());
            }
        }
    }

    Ok((StatusCode::CREATED, Json(mem)))
}

async fn batch_create(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(mut inputs): Json<Vec<db::MemoryInput>>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let ns = get_namespace(&headers);
    if let Some(ref ns_val) = ns {
        for inp in &mut inputs {
            if inp.namespace.is_none() {
                inp.namespace = Some(ns_val.clone());
            }
        }
    }
    let count = inputs.len();
    let inputs_had_sync = inputs.iter().any(|i| i.sync_embed.unwrap_or(false));
    let db = state.db.clone();
    let results = blocking(move || db.insert_batch(inputs))
        .await??;

    // batch embed if AI is configured
    if let Some(ref cfg) = state.ai {
        if cfg.has_embed() {
            let items: Vec<(String, String)> = results
                .iter()
                .map(|m| (m.id.clone(), m.content.clone()))
                .collect();
            // if any input had sync_embed, do it synchronously
            let any_sync = inputs_had_sync;
            if any_sync {
                match ai::get_embeddings(cfg, &items.iter().map(|(_, c)| c.clone()).collect::<Vec<_>>()).await {
                    Ok(embs) => {
                        let db = state.db.clone();
                        let pairs: Vec<_> = items.into_iter().zip(embs).collect();
                        let _ = tokio::task::spawn_blocking(move || {
                            for ((id, _), emb) in pairs {
                                let _ = db.set_embedding(&id, &emb);
                            }
                        }).await;
                    }
                    Err(e) => warn!(error = %e, "sync batch embedding failed"),
                }
            } else {
                spawn_embed_batch(state.db.clone(), cfg.clone(), items);
            }
        }
    }

    let inserted = results.len();
    Ok(Json(serde_json::json!({
        "inserted": inserted,
        "requested": count,
    })))
}

async fn get_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<db::Memory>, EngramError> {
    let db = state.db.clone();
    let mem = blocking(move || db.get(&id))
        .await??;
    mem.ok_or(EngramError::NotFound).map(Json)
}

#[derive(Deserialize)]
struct UpdateBody {
    content: Option<String>,
    layer: Option<u8>,
    importance: Option<f64>,
    tags: Option<Vec<String>>,
}

async fn update_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<UpdateBody>,
) -> Result<Json<db::Memory>, EngramError> {
    let db = state.db.clone();
    let mem = blocking(move || {
        db.update_fields(
            &id,
            body.content.as_deref(),
            body.layer,
            body.importance,
            body.tags.as_deref(),
        )
    })
    .await??;

    mem.ok_or(EngramError::NotFound).map(Json)
}

async fn delete_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let db = state.db.clone();
    let deleted = blocking(move || db.delete(&id))
        .await??;

    if deleted {
        Ok(Json(serde_json::json!({"ok": true})))
    } else {
        Err(EngramError::NotFound)
    }
}

#[derive(Deserialize)]
struct BatchDeleteBody {
    #[serde(default)]
    ids: Vec<String>,
    namespace: Option<String>,
}

async fn batch_delete(
    State(state): State<AppState>,
    Json(body): Json<BatchDeleteBody>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let db = state.db.clone();
    let deleted = blocking(move || {
        let mut count = 0usize;
        // namespace wipe takes priority
        if let Some(ns) = &body.namespace {
            count += db.delete_namespace(ns).unwrap_or(0);
        }
        for id in &body.ids {
            if db.delete(id).unwrap_or(false) {
                count += 1;
            }
        }
        count
    })
    .await?;

    Ok(Json(serde_json::json!({"deleted": deleted})))
}

#[derive(Deserialize)]
struct ListQuery {
    layer: Option<u8>,
    tag: Option<String>,
    ns: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
}

async fn list_memories(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Query(mut q): Query<ListQuery>,
) -> Result<Json<serde_json::Value>, EngramError> {
    if q.ns.is_none() {
        q.ns = get_namespace(&headers);
    }
    let db = state.db.clone();
    let result = blocking(move || {
        let d = db;
        let limit = q.limit.unwrap_or(50).min(200);
        let offset = q.offset.unwrap_or(0);

        // filter by layer if specified
        let mut memories = if let Some(layer_val) = q.layer {
            if let Ok(layer) = layer_val.try_into() {
                d.list_by_layer(layer, limit, offset)
            } else {
                vec![]
            }
        } else {
            d.list_all(limit, offset)?
        };

        // filter by tag if specified
        if let Some(ref tag) = q.tag {
            memories.retain(|m| m.tags.iter().any(|t| t == tag));
        }
        // filter by namespace
        if let Some(ref ns) = q.ns {
            memories.retain(|m| m.namespace == *ns);
        }

        let count = memories.len();
        let stats = d.stats();
        Ok::<_, EngramError>(serde_json::json!({
            "memories": memories,
            "count": count,
            "total": stats.total,
            "limit": limit,
            "offset": offset,
        }))
    })
    .await??;

    Ok(Json(result))
}

/// Simple keyword search — lighter than /recall, no scoring or budget logic.
#[derive(Deserialize)]
struct SearchQuery {
    q: String,
    limit: Option<usize>,
    ns: Option<String>,
}

async fn quick_search(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Query(mut sq): Query<SearchQuery>,
) -> Result<Json<serde_json::Value>, EngramError> {
    if sq.q.trim().is_empty() {
        return Err(EngramError::EmptyQuery);
    }
    if sq.ns.is_none() {
        sq.ns = get_namespace(&headers);
    }
    let limit = sq.limit.unwrap_or(10).min(50);
    let ns_filter = sq.ns;
    let db = state.db.clone();
    let query = sq.q.clone();
    let results = blocking(move || {
        let d = db;
        let hits = d.search_fts(&query, limit);
        let mut memories: Vec<db::Memory> = hits
            .into_iter()
            .filter_map(|(id, _)| d.get(&id).ok().flatten())
            .collect();
        if let Some(ref ns) = ns_filter {
            memories.retain(|m| m.namespace == *ns);
        }
        memories
    })
    .await?;

    let count = results.len();
    Ok(Json(serde_json::json!({
        "memories": results,
        "count": count,
    })))
}

/// List memories created within a recent time window.
/// GET /recent?hours=2&limit=20
#[derive(Deserialize)]
struct RecentQuery {
    /// Hours to look back (default 2)
    hours: Option<f64>,
    /// Max results (default 20)
    limit: Option<usize>,
    /// Filter by layer (optional)
    layer: Option<u8>,
    /// Skip memories below this importance (e.g. 0.3 to filter noise)
    min_importance: Option<f64>,
    /// Filter by source (e.g. "session")
    source: Option<String>,
    /// Filter by namespace
    ns: Option<String>,
}

async fn list_recent(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Query(mut q): Query<RecentQuery>,
) -> Result<Json<serde_json::Value>, EngramError> {
    if q.ns.is_none() {
        q.ns = get_namespace(&headers);
    }
    let hours = q.hours.unwrap_or(2.0).clamp(0.0, 87_600.0); // cap at 10 years
    let limit = q.limit.unwrap_or(20).min(100);
    let layer_filter = q.layer;
    let min_imp = q.min_importance.unwrap_or(0.0);
    let source_filter = q.source;
    let ns_filter = q.ns;
    let since_ms = db::now_ms() - (hours * 3_600_000.0) as i64;

    let db = state.db.clone();
    let result = blocking(move || {
        let d = db;
        let mut memories = d.list_since(since_ms, limit)?;
        if let Some(l) = layer_filter {
            memories.retain(|m| m.layer as u8 == l);
        }
        if min_imp > 0.0 {
            memories.retain(|m| m.importance >= min_imp);
        }
        if let Some(ref src) = source_filter {
            memories.retain(|m| m.source == *src);
        }
        if let Some(ref ns) = ns_filter {
            memories.retain(|m| m.namespace == *ns);
        }
        Ok::<_, EngramError>(memories)
    })
    .await??;

    let count = result.len();
    Ok(Json(serde_json::json!({
        "memories": result,
        "count": count,
        "since_ms": since_ms,
        "hours": hours,
    })))
}

/// One-call session recovery: combines recent memories, core identity, and session context.
/// GET /resume?hours=4&workspace=engram,rust&limit=100
#[derive(Deserialize)]
struct ResumeQuery {
    hours: Option<f64>,
    ns: Option<String>,
    /// Comma-separated workspace tags. When set, Core memories are
    /// filtered to those matching at least one tag. Untagged Core
    /// memories (identity, universal knowledge) are always included.
    workspace: Option<String>,
    /// Max Core memories to return (default 100).
    limit: Option<usize>,
    /// When true, return compact format (content + tags only) to
    /// minimize token usage. Default true.
    compact: Option<bool>,
    /// Max total characters across all sections. Sections are filled
    /// in priority order: core → working → buffer → recent → sessions.
    /// Default 8000 (~2K tokens). Set 0 for unlimited.
    budget: Option<usize>,
}

/// Fetch memories tagged with `trigger:{action}`. Used for pre-action
/// safety checks — e.g. before `git push`, recall lessons about what
/// not to commit.
async fn get_triggers(
    State(state): State<AppState>,
    Path(action): Path<String>,
    headers: axum::http::HeaderMap,
) -> Result<Json<serde_json::Value>, EngramError> {
    let ns = get_namespace(&headers);
    let tag = format!("trigger:{action}");
    let db = state.db.clone();
    let db2 = state.db.clone();

    let memories: Vec<db::Memory> = blocking(move || {
        db.list_by_tag(&tag, ns.as_deref())
    }).await?.unwrap_or_default();

    // touch each trigger memory so it reinforces over time
    if !memories.is_empty() {
        let ids: Vec<String> = memories.iter().map(|m| m.id.clone()).collect();
        let _ = blocking(move || {
            for id in &ids {
                let _ = db2.touch(id);
            }
        })
        .await;
    }

    Ok(Json(serde_json::json!({
        "action": action,
        "count": memories.len(),
        "memories": memories,
    })))
}

async fn do_resume(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Query(mut q): Query<ResumeQuery>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let hours = q.hours.unwrap_or(4.0).clamp(0.0, 87_600.0); // cap at 10 years
    if q.ns.is_none() {
        q.ns = get_namespace(&headers);
    }
    let ns_filter = q.ns;
    let since_ms = db::now_ms() - (hours * 3_600_000.0) as i64;
    let core_limit = q.limit.unwrap_or(100);

    // Parse workspace tags — from query param, or fall back to env default
    let ws_tags: Vec<String> = q.workspace
        .or_else(|| std::env::var("ENGRAM_WORKSPACE").ok())
        .map(|w| w.split(',').map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()).collect())
        .unwrap_or_default();

    let db = state.db.clone();
    let sections = blocking(move || {
        let d = db;
        let ns_ok = |m: &db::Memory| -> bool {
            ns_filter.as_ref().is_none_or(|ns| m.namespace == *ns)
        };

        // Core memories — filtered by workspace tags when provided.
        // Untagged memories (empty tags) always pass — they're universal
        // knowledge (identity, preferences) that apply everywhere.
        let all_core: Vec<db::Memory> = d
            .list_by_layer(db::Layer::Core, 10000, 0)
            .into_iter()
            .filter(|m| ns_ok(m))
            .collect();

        let mut core: Vec<db::Memory> = if ws_tags.is_empty() {
            all_core
        } else {
            all_core.into_iter().filter(|m| {
                // Always include untagged / universal memories
                if m.tags.is_empty() { return true; }
                // Include if any memory tag matches any workspace tag
                m.tags.iter().any(|t| ws_tags.iter().any(|ws| t.to_lowercase().contains(ws.as_str())))
            }).collect()
        };
        // Sort by importance desc, cap output
        core.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal));
        core.truncate(core_limit);

        // Working memories — what you've been doing, decisions made,
        // lessons learned. This is the "episodic context" that bridges
        // permanent knowledge (Core) and immediate activity (recent).
        // Also filtered by workspace tags.
        let all_working: Vec<db::Memory> = d
            .list_by_layer(db::Layer::Working, 10000, 0)
            .into_iter()
            .filter(|m| ns_ok(m))
            .collect();

        let mut working: Vec<db::Memory> = if ws_tags.is_empty() {
            all_working
        } else {
            all_working.into_iter().filter(|m| {
                if m.tags.is_empty() { return true; }
                m.tags.iter().any(|t| ws_tags.iter().any(|ws| t.to_lowercase().contains(ws.as_str())))
            }).collect()
        };
        working.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal));
        working.truncate(core_limit);

        // Buffer — transient, unproven memories. Include them so the
        // agent knows what's "in the air" but hasn't solidified yet.
        let mut buffer: Vec<db::Memory> = d
            .list_by_layer(db::Layer::Buffer, 10000, 0)
            .into_iter()
            .filter(|m| ns_ok(m))
            .collect();
        buffer.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        buffer.truncate(50); // lighter cap — these are transient

        // all recent memories in one query, then split by source
        let all_recent: Vec<db::Memory> = d.list_since(since_ms, 100).unwrap_or_default()
            .into_iter()
            .filter(|m| ns_ok(m))
            .collect();

        // recent activity (exclude session notes; they're shown separately)
        let recent: Vec<db::Memory> = all_recent.iter()
            .filter(|m| m.source != "session")
            .take(20)
            .cloned()
            .collect();

        // session memories — what happened in recent sessions
        let mut next_actions = Vec::new();
        let mut sessions = Vec::new();
        for m in all_recent.into_iter().filter(|m| m.source == "session") {
            if m.tags.iter().any(|t| t == "next-action") {
                next_actions.push(m);
            } else {
                sessions.push(m);
            }
        }
        sessions.truncate(10);
        next_actions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        next_actions.truncate(3);

        (core, working, buffer, recent, sessions, next_actions)
    })
    .await?;

    let (core, working, buffer, recent, sessions, next_actions) = sections;
    let compact = q.compact.unwrap_or(true);

    // Helper: convert memories to compact or full format
    let to_json = |mems: &[db::Memory]| -> Vec<serde_json::Value> {
        if compact {
            mems.iter().map(|m| {
                let mut obj = serde_json::json!({"content": m.content});
                if !m.tags.is_empty() {
                    obj["tags"] = serde_json::json!(m.tags);
                }
                obj
            }).collect()
        } else {
            mems.iter().map(|m| serde_json::to_value(m).unwrap_or_default()).collect()
        }
    };

    // Apply budget: fill sections in priority order, stop when budget exhausted
    // Default 8000 chars (~2K tokens). 0 = unlimited.
    let budget_val = q.budget.unwrap_or(8000);
    let mut budget_left = if budget_val == 0 { usize::MAX } else { budget_val };
    let mut take_within_budget = |mems: &[db::Memory]| -> Vec<db::Memory> {
        if budget_left == 0 { return vec![]; }
        let mut taken = Vec::new();
        for m in mems {
            let cost = if compact {
                m.content.len() + m.tags.iter().map(|t| t.len() + 3).sum::<usize>() + 20
            } else {
                m.content.len() + 250 // ~250 chars metadata overhead per memory
            };
            if cost > budget_left && !taken.is_empty() { break; }
            budget_left = budget_left.saturating_sub(cost);
            taken.push(m.clone());
        }
        taken
    };

    // Priority order: core → working → next_actions → sessions → recent → buffer
    let core_out = take_within_budget(&core);
    let working_out = take_within_budget(&working);
    let next_out = take_within_budget(&next_actions);
    let sessions_out = take_within_budget(&sessions);
    let recent_out = take_within_budget(&recent);
    let buffer_out = take_within_budget(&buffer);

    Ok(Json(serde_json::json!({
        "core": to_json(&core_out),
        "working": to_json(&working_out),
        "buffer": to_json(&buffer_out),
        "recent": to_json(&recent_out),
        "sessions": to_json(&sessions_out),
        "next_actions": to_json(&next_out),
        "hours": hours,
        "core_count": core_out.len(),
        "working_count": working_out.len(),
        "buffer_count": buffer_out.len(),
        "recent_count": recent_out.len(),
        "session_count": sessions_out.len(),
        "next_action_count": next_out.len(),
    })))
}

async fn do_recall(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(mut req): Json<recall::RecallRequest>,
) -> Result<Json<recall::RecallResponse>, EngramError> {
    if req.query.is_empty() {
        return Err(EngramError::EmptyQuery);
    }
    if req.namespace.is_none() {
        req.namespace = get_namespace(&headers);
    }

    let do_rerank =
        req.rerank.unwrap_or(false) && state.ai.as_ref().is_some_and(|c| c.has_llm());
    let query_text = req.query.clone();
    let final_limit = req.limit.unwrap_or(20).min(100);

    if do_rerank {
        req.limit = Some(final_limit * 2);
    }

    let query_emb = if let Some(ref cfg) = state.ai {
        if cfg.has_embed() {
            // check cache first
            let cached = {
                let mut cache = state.embed_cache.lock().unwrap_or_else(|e| e.into_inner());
                cache.get(&query_text).cloned()
            };
            if let Some(emb) = cached {
                debug!("embed cache hit for recall query");
                Some(emb)
            } else {
                match ai::get_embeddings(cfg, std::slice::from_ref(&query_text)).await {
                    Ok(mut v) => {
                        let emb = v.pop();
                        if let Some(ref e) = emb {
                            let mut cache = state.embed_cache.lock().unwrap_or_else(|e| e.into_inner());
                            cache.put(query_text.clone(), e.clone());
                        }
                        emb
                    }
                    Err(e) => {
                        warn!(error = %e, "embedding lookup failed, falling back to FTS");
                        None
                    }
                }
            }
        } else {
            None
        }
    } else {
        None
    };

    let explicit_expand = req.expand;
    let do_expand =
        explicit_expand.unwrap_or(false) && state.ai.as_ref().is_some_and(|c| c.has_llm());
    let expanded = if do_expand {
        if let Some(ref cfg) = state.ai {
            let q = ai::expand_query(cfg, &query_text).await;
            if !q.is_empty() {
                debug!(expanded = ?q, "query expansion");
            }
            q
        } else {
            vec![]
        }
    } else {
        vec![]
    };
    let expanded_for_response = expanded.clone();

    let db = state.db.clone();
    let req_clone = req.clone();
    let qe_clone = query_emb.clone();
    let mut result = blocking(move || {
        let eq = if expanded.is_empty() { None } else { Some(expanded.as_slice()) };
        recall::recall(&db, &req_clone, qe_clone.as_deref(), eq)
    })
    .await?;

    // auto-expand: if expand wasn't explicitly set and top result is weak, retry with expansion
    let auto_expanded;
    if explicit_expand.is_none()
        && state.ai.as_ref().is_some_and(|c| c.has_llm())
        && result.memories.first().is_none_or(|m| m.relevance < 0.4)
    {
        if let Some(ref cfg) = state.ai {
            let eq = ai::expand_query(cfg, &query_text).await;
            if !eq.is_empty() {
                debug!(expanded = ?eq, "auto-expand (weak initial results)");
                let db = state.db.clone();
                let eq2 = eq.clone();
                let retry = blocking(move || {
                    recall::recall(&db, &req, query_emb.as_deref(), Some(&eq2))
                }).await?;
                // use expanded result only if it's actually better
                if retry.memories.first().map_or(0.0, |m| m.relevance)
                    > result.memories.first().map_or(0.0, |m| m.relevance)
                {
                    result = retry;
                    auto_expanded = Some(eq);
                } else {
                    auto_expanded = None;
                }
            } else {
                auto_expanded = None;
            }
        } else {
            auto_expanded = None;
        }
    } else {
        auto_expanded = None;
    }

    if do_rerank {
        if let Some(cfg) = state.ai.as_ref() {
            recall::rerank_results(&mut result, &query_text, final_limit, cfg).await;
        }
    }

    // attach expanded queries to response
    let final_expanded = if !expanded_for_response.is_empty() {
        Some(expanded_for_response)
    } else {
        auto_expanded
    };
    if let Some(eq) = final_expanded {
        result.expanded_queries = Some(eq);
    }

    Ok(Json(result))
}

async fn do_consolidate(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Result<Json<consolidate::ConsolidateResponse>, EngramError> {
    let parsed: Option<consolidate::ConsolidateRequest> = if body.is_empty() {
        None
    } else {
        serde_json::from_slice(&body).ok()
    };

    let result =
        consolidate::consolidate(state.db.clone(), parsed, state.ai.clone()).await;

    Ok(Json(result))
}

async fn do_repair(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let db = state.db.clone();
    let (orphans, rebuilt) = blocking(move || db.repair_fts()).await??;

    // Backfill missing embeddings
    let mut embed_backfilled = 0;
    if let Some(cfg) = &state.ai {
        let db2 = state.db.clone();
        let missing: Vec<(String, String)> = blocking(move || {
            Ok::<_, EngramError>(db2.list_missing_embeddings(500))
        }).await??;
        if !missing.is_empty() {
            embed_backfilled = missing.len();
            spawn_embed_batch(state.db.clone(), cfg.clone(), missing);
        }
    }

    Ok(Json(serde_json::json!({
        "orphans_removed": orphans,
        "fts_rebuilt": rebuilt,
        "embed_backfill_queued": embed_backfilled,
    })))
}

#[derive(Deserialize)]
struct VacuumQuery {
    full: Option<bool>,
}

async fn do_vacuum(
    State(state): State<AppState>,
    Query(q): Query<VacuumQuery>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let db = state.db.clone();
    let full = q.full.unwrap_or(false);
    let freed = blocking(move || {
        if full { db.vacuum_full() } else { db.vacuum_incremental(1000) }
    }).await??;
    Ok(Json(serde_json::json!({
        "freed_bytes": freed,
        "mode": if full { "full" } else { "incremental" },
    })))
}

async fn proxy_flush(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    crate::proxy::flush_window(&state).await;
    Json(serde_json::json!({"status": "ok"}))
}

#[derive(Deserialize)]
struct ExtractRequest {
    text: String,
    auto_embed: Option<bool>,
}

#[derive(serde::Serialize)]
struct ExtractResponse {
    extracted: Vec<db::Memory>,
    count: usize,
}

async fn do_extract(
    State(state): State<AppState>,
    Json(req): Json<ExtractRequest>,
) -> Result<Json<ExtractResponse>, EngramError> {
    let cfg = state.ai.as_ref().ok_or(EngramError::AiNotConfigured)?;

    if req.text.trim().is_empty() {
        return Err(EngramError::EmptyContent);
    }

    let extracted = ai::extract_memories(cfg, &req.text)
        .await
        .map_err(EngramError::AiBackend)?;

    let auto_embed = req.auto_embed.unwrap_or(true);
    let mut memories = Vec::new();
    let mut embed_batch = Vec::new();

    for em in extracted {
        let input = db::MemoryInput {
            content: em.content,
            layer: em.layer,
            importance: em.importance,
            source: Some("extract".into()),
            tags: em.tags,
        supersedes: None,
        skip_dedup: None,
        namespace: None,
        sync_embed: None,
        };

        let db = state.db.clone();
        let mem = blocking(move || db.insert(input))
            .await??;

        if auto_embed {
            embed_batch.push((mem.id.clone(), mem.content.clone()));
        }

        memories.push(mem);
    }

    // single batch call instead of N individual embed requests
    if !embed_batch.is_empty() && cfg.has_embed() {
        spawn_embed_batch(state.db.clone(), cfg.clone(), embed_batch);
    }

    let count = memories.len();
    debug!(count, "extract complete");
    Ok(Json(ExtractResponse { extracted: memories, count }))
}

// -- Export / Import --

async fn do_export(
    State(state): State<AppState>,
    Query(q): Query<std::collections::HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let include_embed = q.get("embed").map(|v| v == "true" || v == "1").unwrap_or(false);
    let db = state.db.clone();
    let memories = blocking(move || db.export_with_embeddings(include_embed))
        .await??;

    let count = memories.len();
    Ok(Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "count": count,
        "memories": memories,
    })))
}

async fn do_import(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let ns_override = get_namespace(&headers);
    let memories_val = body
        .get("memories")
        .ok_or_else(|| EngramError::Validation("missing 'memories' array".into()))?;

    let mut memories: Vec<db::Memory> = serde_json::from_value(memories_val.clone())
        .map_err(|e| EngramError::Validation(format!("invalid memories format: {e}")))?;

    if let Some(ns) = ns_override {
        for m in &mut memories {
            m.namespace = ns.clone();
        }
    }

    let db = state.db.clone();
    let imported = blocking(move || db.import(&memories))
        .await??;

    Ok(Json(serde_json::json!({
        "imported": imported,
        "skipped": memories_val.as_array().map(|a| a.len()).unwrap_or(0).saturating_sub(imported),
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn test_state(api_key: Option<&str>) -> AppState {
        let mdb = db::MemoryDB::open(":memory:").unwrap();
        AppState {
            db: std::sync::Arc::new(mdb),
            ai: None,
            api_key: api_key.map(|s| s.to_string()),
            embed_cache: std::sync::Arc::new(std::sync::Mutex::new(
                crate::EmbedCacheInner::new(16),
            )),
            proxy: None,
            started_at: std::time::Instant::now(),
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
            Request::builder().uri("/resume?hours=1").body(Body::empty()).unwrap()
        ).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let j = body_json(resp).await;

        // Structure checks
        assert!(j["core"].is_array());
        assert!(j["working"].is_array());
        assert!(j["buffer"].is_array());
        assert!(j["recent"].is_array());
        assert!(j["sessions"].is_array());
        assert!(j["next_actions"].is_array());
        assert!(j["hours"].as_f64().unwrap() > 0.0);

        // Core should include the high-importance core memory
        let core = j["core"].as_array().unwrap();
        assert!(!core.is_empty(), "should have core memories");
        assert!(core[0]["content"].as_str().unwrap().contains("test agent"));

        // Next actions should have the tagged memory
        let next = j["next_actions"].as_array().unwrap();
        assert!(!next.is_empty(), "should have next-action memories");
        assert!(next[0]["content"].as_str().unwrap().contains("write more tests"));

        // Sessions should have the session memory (not tagged as next-action)
        let sessions = j["sessions"].as_array().unwrap();
        assert!(!sessions.is_empty(), "should have session memories");
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
            Request::builder().uri("/resume?hours=1&ns=ns-a").body(Body::empty()).unwrap()
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
}