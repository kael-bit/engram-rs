//! HTTP API handlers.

use axum::extract::{Path, Query, State};
use axum::http::{Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use subtle::ConstantTimeEq;
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
    // Public routes (no auth)
    let public = Router::new()
        .route("/", get(health))
        .route("/stats", get(stats));

    // Protected routes
    let protected = Router::new()
        .route("/memories", post(create_memory).get(list_memories))
        .route(
            "/memories/{id}",
            get(get_memory).patch(update_memory).delete(delete_memory),
        )
        .route("/recall", post(do_recall))
        .route("/consolidate", post(do_consolidate))
        .route("/extract", post(do_extract))
        .layer(middleware::from_fn_with_state(state.clone(), require_auth));

    public.merge(protected).with_state(state)
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
    }))
}

async fn stats(State(state): State<AppState>) -> Json<db::Stats> {
    let db = state.db.clone();
    let s = tokio::task::spawn_blocking(move || lock_db(&db).stats())
        .await
        .unwrap_or(db::Stats { total: 0, buffer: 0, working: 0, core: 0 });
    Json(s)
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
            let cfg = cfg.clone();
            let db = state.db.clone();
            let id = mem.id.clone();
            let content = mem.content.clone();
            tokio::spawn(async move {
                match ai::get_embeddings(&cfg, &[content]).await {
                    Ok(embs) if !embs.is_empty() => {
                        let emb = embs.into_iter().next().unwrap();
                        let _ = tokio::task::spawn_blocking(move || {
                            lock_db(&db).set_embedding(&id, &emb)
                        }).await;
                    }
                    Err(e) => warn!(error = %e, "embedding generation failed"),
                    _ => {}
                }
            });
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
        if let Some(layer_val) = q.layer {
            if let Ok(layer) = layer_val.try_into() {
                let memories = d.list_by_layer(layer);
                let count = memories.len();
                return Ok(serde_json::json!({"memories": memories, "count": count}));
            }
        }
        let limit = q.limit.unwrap_or(50).min(200);
        let offset = q.offset.unwrap_or(0);
        let memories = d.list_all(limit, offset)?;
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

async fn do_recall(
    State(state): State<AppState>,
    Json(req): Json<recall::RecallRequest>,
) -> Result<Json<recall::RecallResponse>, EngramError> {
    if req.query.is_empty() {
        return Err(EngramError::EmptyContent);
    }

    let query_emb = if let Some(ref cfg) = state.ai {
        if cfg.has_embed() {
            match ai::get_embeddings(cfg, std::slice::from_ref(&req.query)).await {
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
    let result = tokio::task::spawn_blocking(move || {
        recall::recall(&lock_db(&db), &req, query_emb.as_deref())
    })
    .await
    .map_err(|e| EngramError::Internal(e.to_string()))?;

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

    let db = state.db.clone();
    let result = tokio::task::spawn_blocking(move || {
        consolidate::consolidate(&lock_db(&db), parsed.as_ref())
    })
    .await
    .map_err(|e| EngramError::Internal(e.to_string()))?;

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

        if auto_embed && cfg.has_embed() {
            let cfg = cfg.clone();
            let db = state.db.clone();
            let id = mem.id.clone();
            let content = mem.content.clone();
            tokio::spawn(async move {
                match ai::get_embeddings(&cfg, &[content]).await {
                    Ok(embs) if !embs.is_empty() => {
                        let emb = embs.into_iter().next().unwrap();
                        let _ = tokio::task::spawn_blocking(move || {
                            lock_db(&db).set_embedding(&id, &emb)
                        }).await;
                    }
                    Err(e) => warn!(error = %e, "embedding generation failed"),
                    _ => {}
                }
            });
        }

        memories.push(mem);
    }

    let count = memories.len();
    debug!(count, "extract complete");
    Ok(Json(ExtractResponse { extracted: memories, count }))
}
