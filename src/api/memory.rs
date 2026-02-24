//! Memory CRUD handlers.

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::Json;
use serde::Deserialize;
use tracing::warn;

use crate::error::EngramError;
use crate::{ai, db, AppState};
use super::{blocking, get_namespace, spawn_embed, spawn_embed_batch};

pub(super) async fn create_memory(
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

pub(super) async fn batch_create(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Result<Json<serde_json::Value>, EngramError> {
    // Accept both [{...}] and {"memories": [{...}]}
    let mut inputs: Vec<db::MemoryInput> = serde_json::from_slice(&body)
        .or_else(|_| {
            #[derive(serde::Deserialize)]
            struct Wrapped { memories: Vec<db::MemoryInput> }
            serde_json::from_slice::<Wrapped>(&body).map(|w| w.memories)
        })
        .map_err(|e| EngramError::Validation(format!("invalid batch body: {e}")))?;
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

pub(super) async fn get_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<db::Memory>, EngramError> {
    let db = state.db.clone();
    let mem = blocking(move || {
        let full_id = db.resolve_prefix(&id)?;
        db.get(&full_id)
    })
        .await??;
    mem.ok_or(EngramError::NotFound).map(Json)
}

#[derive(Deserialize)]
pub(super) struct UpdateBody {
    content: Option<String>,
    layer: Option<u8>,
    importance: Option<f64>,
    tags: Option<Vec<String>>,
    kind: Option<String>,
}

pub(super) async fn update_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<UpdateBody>,
) -> Result<Json<db::Memory>, EngramError> {
    let db = state.db.clone();
    let mem = blocking(move || {
        let full_id = db.resolve_prefix(&id)?;
        if let Some(ref k) = body.kind {
            db.update_kind(&full_id, k)?;
        }
        db.update_fields(
            &full_id,
            body.content.as_deref(),
            body.layer,
            body.importance,
            body.tags.as_deref(),
        )
    })
    .await??;

    mem.ok_or(EngramError::NotFound).map(Json)
}

pub(super) async fn delete_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let db = state.db.clone();
    let deleted = blocking(move || {
        let full_id = db.resolve_prefix(&id)?;
        db.delete(&full_id)
    })
        .await??;

    if deleted {
        Ok(Json(serde_json::json!({"ok": true})))
    } else {
        Err(EngramError::NotFound)
    }
}

#[derive(Deserialize)]
pub(super) struct BatchDeleteBody {
    #[serde(default)]
    ids: Vec<String>,
    namespace: Option<String>,
}

pub(super) async fn batch_delete(
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
pub(super) struct ListQuery {
    layer: Option<u8>,
    tag: Option<String>,
    #[serde(alias = "namespace")]
    ns: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
}

pub(super) async fn list_memories(
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

        let memories = d.list_filtered(
            limit,
            offset,
            q.ns.as_deref(),
            q.layer,
            q.tag.as_deref(),
        )?;

        let count = memories.len();
        let total = match q.ns.as_deref() {
            Some(ns) => d.stats_ns(ns).total,
            None => d.stats().total,
        };
        Ok::<_, EngramError>(serde_json::json!({
            "memories": memories,
            "count": count,
            "total": total,
            "limit": limit,
            "offset": offset,
        }))
    })
    .await??;

    Ok(Json(result))
}
