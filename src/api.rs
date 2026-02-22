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
use crate::{ai, consolidate, db, lock_db, recall, AppState};

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
        .route("/stats", get(stats));

    let protected = Router::new()
        .route("/memories", post(create_memory).get(list_memories))
        .route(
            "/memories/{id}",
            get(get_memory).patch(update_memory).delete(delete_memory),
        )
        .route("/recall", post(do_recall))
        .route("/search", get(quick_search))
        .route("/recent", get(list_recent))
        .route("/consolidate", post(do_consolidate))
        .route("/extract", post(do_extract))
        .route("/export", get(do_export))
        .route("/import", post(do_import))
        .layer(middleware::from_fn_with_state(state.clone(), require_auth));

    // 64KB ought to be enough for anybody — content cap is 8K chars (~24KB UTF-8),
    // extract text can be longer but 64KB is plenty
    public
        .merge(protected)
        .layer(RequestBodyLimitLayer::new(64 * 1024))
        .with_state(state)
}

async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let db = state.db.clone();
    let s = tokio::task::spawn_blocking(move || lock_db(&db).stats())
        .await
        .unwrap_or(db::Stats { total: 0, buffer: 0, working: 0, core: 0 });

    Json(serde_json::json!({
        "name": "engram",
        "version": env!("CARGO_PKG_VERSION"),
        "ai_enabled": state.ai.is_some(),
        "stats": s,
        "endpoints": {
            "GET /": "this health check",
            "GET /stats": "memory counts per layer",
            "POST /memories": "create a memory",
            "GET /memories": "list memories (optional ?layer=N&tag=X&limit=N)",
            "GET /memories/:id": "get a memory by id",
            "PATCH /memories/:id": "update a memory",
            "DELETE /memories/:id": "delete a memory",
            "POST /recall": "hybrid search (semantic + keyword)",
            "GET /search?q=term": "quick keyword search",
            "GET /recent?hours=2": "recent memories by time",
            "POST /consolidate": "run maintenance cycle",
            "POST /extract": "LLM-extract memories from text",
            "GET /export": "export all memories as JSON",
            "POST /import": "import memories from JSON",
        },
    }))
}

async fn stats(State(state): State<AppState>) -> Json<db::Stats> {
    let db = state.db.clone();
    let s = tokio::task::spawn_blocking(move || lock_db(&db).stats())
        .await
        .unwrap_or(db::Stats { total: 0, buffer: 0, working: 0, core: 0 });
    Json(s)
}

/// Fire-and-forget: generate embedding for a memory in the background.
fn spawn_embed(db: crate::SharedDB, cfg: ai::AiConfig, id: String, content: String) {
    tokio::spawn(async move {
        match ai::get_embeddings(&cfg, &[content]).await {
            Ok(embs) if !embs.is_empty() => {
                let emb = embs.into_iter().next().unwrap();
                let _ = tokio::task::spawn_blocking(move || {
                    lock_db(&db).set_embedding(&id, &emb)
                })
                .await;
            }
            Err(e) => warn!(error = %e, "embedding generation failed"),
            _ => {}
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
        match ai::get_embeddings(&cfg, &texts).await {
            Ok(embs) => {
                for (emb, (id, _)) in embs.into_iter().zip(items.iter()) {
                    let db = db.clone();
                    let id = id.clone();
                    let _ = tokio::task::spawn_blocking(move || {
                        lock_db(&db).set_embedding(&id, &emb)
                    })
                    .await;
                }
            }
            Err(e) => warn!(error = %e, "batch embedding failed"),
        }
    });
}

async fn create_memory(
    State(state): State<AppState>,
    Json(input): Json<db::MemoryInput>,
) -> Result<(StatusCode, Json<db::Memory>), EngramError> {
    let db = state.db.clone();
    let mem = tokio::task::spawn_blocking(move || lock_db(&db).insert(input))
        .await
        .map_err(|e| EngramError::Internal(e.to_string()))??;

    if let Some(ref cfg) = state.ai {
        if cfg.has_embed() {
            spawn_embed(state.db.clone(), cfg.clone(), mem.id.clone(), mem.content.clone());
        }
    }

    Ok((StatusCode::CREATED, Json(mem)))
}

async fn get_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<db::Memory>, EngramError> {
    let db = state.db.clone();
    let mem = tokio::task::spawn_blocking(move || lock_db(&db).get(&id))
        .await
        .map_err(|e| EngramError::Internal(e.to_string()))??;
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
    let mem = tokio::task::spawn_blocking(move || {
        lock_db(&db).update_fields(
            &id,
            body.content.as_deref(),
            body.layer,
            body.importance,
            body.tags.as_deref(),
        )
    })
    .await
    .map_err(|e| EngramError::Internal(e.to_string()))??;

    mem.ok_or(EngramError::NotFound).map(Json)
}

async fn delete_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let db = state.db.clone();
    let deleted = tokio::task::spawn_blocking(move || lock_db(&db).delete(&id))
        .await
        .map_err(|e| EngramError::Internal(e.to_string()))??;

    if deleted {
        Ok(Json(serde_json::json!({"ok": true})))
    } else {
        Err(EngramError::NotFound)
    }
}

#[derive(Deserialize)]
struct ListQuery {
    layer: Option<u8>,
    tag: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
}

async fn list_memories(
    State(state): State<AppState>,
    Query(q): Query<ListQuery>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let db = state.db.clone();
    let result = tokio::task::spawn_blocking(move || {
        let d = lock_db(&db);
        let limit = q.limit.unwrap_or(50).min(200);
        let offset = q.offset.unwrap_or(0);

        // filter by layer if specified
        let mut memories = if let Some(layer_val) = q.layer {
            if let Ok(layer) = layer_val.try_into() {
                d.list_by_layer(layer)
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
    .await
    .map_err(|e| EngramError::Internal(e.to_string()))??;

    Ok(Json(result))
}

/// Simple keyword search — lighter than /recall, no scoring or budget logic.
#[derive(Deserialize)]
struct SearchQuery {
    q: String,
    limit: Option<usize>,
}

async fn quick_search(
    State(state): State<AppState>,
    Query(sq): Query<SearchQuery>,
) -> Result<Json<serde_json::Value>, EngramError> {
    if sq.q.trim().is_empty() {
        return Err(EngramError::EmptyContent);
    }
    let limit = sq.limit.unwrap_or(10).min(50);
    let db = state.db.clone();
    let query = sq.q.clone();
    let results = tokio::task::spawn_blocking(move || {
        let d = lock_db(&db);
        let hits = d.search_fts(&query, limit);
        let memories: Vec<db::Memory> = hits
            .into_iter()
            .filter_map(|(id, _)| d.get(&id).ok().flatten())
            .collect();
        memories
    })
    .await
    .map_err(|e| EngramError::Internal(e.to_string()))?;

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
}

async fn list_recent(
    State(state): State<AppState>,
    Query(q): Query<RecentQuery>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let hours = q.hours.unwrap_or(2.0);
    let limit = q.limit.unwrap_or(20).min(100);
    let layer_filter = q.layer;
    let min_imp = q.min_importance.unwrap_or(0.0);
    let since_ms = db::now_ms() - (hours * 3_600_000.0) as i64;

    let db = state.db.clone();
    let result = tokio::task::spawn_blocking(move || {
        let d = lock_db(&db);
        let mut memories = d.list_since(since_ms, limit)?;
        if let Some(l) = layer_filter {
            memories.retain(|m| m.layer as u8 == l);
        }
        if min_imp > 0.0 {
            memories.retain(|m| m.importance >= min_imp);
        }
        Ok::<_, EngramError>(memories)
    })
    .await
    .map_err(|e| EngramError::Internal(e.to_string()))??;

    let count = result.len();
    Ok(Json(serde_json::json!({
        "memories": result,
        "count": count,
        "since_ms": since_ms,
        "hours": hours,
    })))
}

async fn do_recall(
    State(state): State<AppState>,
    Json(mut req): Json<recall::RecallRequest>,
) -> Result<Json<recall::RecallResponse>, EngramError> {
    if req.query.is_empty() {
        return Err(EngramError::EmptyContent);
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
            match ai::get_embeddings(cfg, std::slice::from_ref(&query_text)).await {
                Ok(mut v) => v.pop(),
                Err(e) => {
                    warn!(error = %e, "embedding lookup failed, falling back to FTS");
                    None
                }
            }
        } else {
            None
        }
    } else {
        None
    };

    let db = state.db.clone();
    let mut result = tokio::task::spawn_blocking(move || {
        recall::recall(&lock_db(&db), &req, query_emb.as_deref())
    })
    .await
    .map_err(|e| EngramError::Internal(e.to_string()))?;

    if do_rerank {
        let cfg = state.ai.as_ref().unwrap();
        recall::rerank_results(&mut result, &query_text, final_limit, cfg).await;
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
        };

        let db = state.db.clone();
        let mem = tokio::task::spawn_blocking(move || lock_db(&db).insert(input))
            .await
            .map_err(|e| EngramError::Internal(e.to_string()))??;

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
) -> Result<Json<serde_json::Value>, EngramError> {
    let db = state.db.clone();
    let memories = tokio::task::spawn_blocking(move || lock_db(&db).export_all())
        .await
        .map_err(|e| EngramError::Internal(e.to_string()))??;

    let count = memories.len();
    Ok(Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "count": count,
        "memories": memories,
    })))
}

async fn do_import(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let memories_val = body
        .get("memories")
        .ok_or_else(|| EngramError::Validation("missing 'memories' array".into()))?;

    let memories: Vec<db::Memory> = serde_json::from_value(memories_val.clone())
        .map_err(|e| EngramError::Validation(format!("invalid memories format: {e}")))?;

    let db = state.db.clone();
    let imported = tokio::task::spawn_blocking(move || lock_db(&db).import(&memories))
        .await
        .map_err(|e| EngramError::Internal(e.to_string()))??;

    Ok(Json(serde_json::json!({
        "imported": imported,
        "skipped": memories_val.as_array().map(|a| a.len()).unwrap_or(0) - imported,
    })))
}
