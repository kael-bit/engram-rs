//! Memory CRUD handlers.

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::Json;

use crate::extract::LenientJson;
use serde::Deserialize;
use tracing::warn;

use crate::error::EngramError;
use crate::{ai, db, AppState};
use super::{blocking, get_namespace, spawn_embed, spawn_embed_batch};

pub(super) async fn create_memory(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    LenientJson(mut input): LenientJson<db::MemoryInput>,
) -> Result<(StatusCode, Json<db::Memory>), EngramError> {
    if input.namespace.is_none() {
        input.namespace = get_namespace(&headers);
    }
    let sync = input.sync_embed.unwrap_or(false);
    let skip_dedup = input.skip_dedup.unwrap_or(false);

    // Semantic dedup: if AI is available and dedup isn't skipped, check
    // for semantically similar existing memories before inserting.
    // Jaccard (in db.insert) catches textual duplicates; this catches
    // "same concept, different wording" — the gap that caused rep=0.
    if !skip_dedup {
        if let Some(ref cfg) = state.ai {
            if cfg.has_embed() {
                tracing::debug!("semantic dedup: checking before insert");
                match crate::recall::quick_semantic_dup_threshold(
                    cfg, &state.db, &input.content, crate::thresholds::INSERT_DEDUP_SIM,
                ).await {
                    Ok(Some(existing_id)) if !existing_id.is_empty() => {
                        let db = state.db.clone();
                        let eid = existing_id.clone();
                        let _ = tokio::task::spawn_blocking(move || db.reinforce(&eid)).await;

                        let db = state.db.clone();
                        let eid2 = existing_id.clone();
                        let existing = tokio::task::spawn_blocking(move || db.get(&eid2))
                            .await
                            .map_err(|e| EngramError::Internal(e.to_string()))?
                            .map_err(|e| EngramError::Internal(e.to_string()))?
                            .ok_or(EngramError::NotFound)?;

                        // If new content differs meaningfully from existing, merge via
                        // LLM to preserve details from both. Pure restatements (very
                        // similar text) just reinforce without an LLM call.
                        let contents_differ = input.content != existing.content;
                        // Jaccard < 0.8 means there's meaningful textual difference
                        let textually_different = !db::jaccard_similar(
                            &existing.content, &input.content, crate::thresholds::INSERT_MERGE_SIM,
                        );

                        if contents_differ && textually_different {
                            let merged_content = if let Some(ref cfg) = state.ai {
                                merge_memory_contents(cfg, &existing.content, &input.content, &state.db).await
                                    .unwrap_or_else(|_| input.content.clone())
                            } else {
                                // No AI: fall back to longer version
                                if input.content.len() > existing.content.len() {
                                    input.content.clone()
                                } else {
                                    existing.content.clone()
                                }
                            };

                            // Only update if merge actually changed content
                            if merged_content != existing.content {
                                let mut merged_tags = existing.tags.clone();
                                for t in input.tags.as_deref().unwrap_or(&[]) {
                                    if !merged_tags.contains(t) {
                                        merged_tags.push(t.clone());
                                    }
                                }
                                let db = state.db.clone();
                                let eid3 = existing_id.clone();
                                let content = merged_content;
                                let mem = tokio::task::spawn_blocking(move || {
                                    db.update_fields(&eid3, Some(&content), None, None, Some(&merged_tags))
                                })
                                    .await
                                    .map_err(|e| EngramError::Internal(e.to_string()))?
                                    .map_err(|e| EngramError::Internal(e.to_string()))?
                                    .ok_or(EngramError::NotFound)?;

                                if let Some(ref cfg) = state.ai {
                                    spawn_embed(state.db.clone(), cfg.clone(), mem.id.clone(), mem.content.clone());
                                }

                                tracing::info!(
                                    existing_id = %mem.id, rep = mem.repetition_count,
                                    "semantic dedup: LLM-merged content + reinforced"
                                );
                                return Ok((StatusCode::OK, Json(mem)));
                            }
                        }

                        tracing::info!(
                            existing_id = %existing.id, rep = existing.repetition_count,
                            "semantic dedup: reinforced (no new info)"
                        );
                        return Ok((StatusCode::OK, Json(existing)));
                    }
                    _ => {}
                }
            }
        }
    }

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
                    Ok(er) if !er.embeddings.is_empty() => {
                        if let Some(ref u) = er.usage {
                            let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
                            let _ = db.log_llm_call("embed_sync", &cfg.embed_model, u.prompt_tokens, u.completion_tokens, cached, 0);
                        }
                        if let Some(emb) = er.embeddings.into_iter().next() {
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
                    Ok(er) => {
                        if let Some(ref u) = er.usage {
                            let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
                            let _ = state.db.log_llm_call("embed_batch", &cfg.embed_model, u.prompt_tokens, u.completion_tokens, cached, 0);
                        }
                        let db = state.db.clone();
                        let pairs: Vec<_> = items.into_iter().zip(er.embeddings).collect();
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
    LenientJson(body): LenientJson<UpdateBody>,
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
    LenientJson(body): LenientJson<BatchDeleteBody>,
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
    kind: Option<String>,
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
            q.kind.as_deref(),
        )?;

        let count = memories.len();
        let total = if q.tag.is_some() || q.layer.is_some() || q.kind.is_some() {
            d.count_filtered(q.ns.as_deref(), q.layer, q.tag.as_deref(), q.kind.as_deref())
                .unwrap_or(count)
        } else {
            match q.ns.as_deref() {
                Some(ns) => d.stats_ns(ns).total,
                None => d.stats().total,
            }
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

const MERGE_PROMPT: &str = "\
Merge two versions of the same memory into one. Preserve ALL specific details \
from BOTH versions — names, numbers, commands, constraints. \
Output ONLY the merged text, nothing else. Keep the same language as the input. \
Be concise; don't add commentary or explanation.";

async fn merge_memory_contents(
    cfg: &ai::AiConfig,
    existing: &str,
    new: &str,
    db: &crate::SharedDB,
) -> Result<String, EngramError> {
    let user_msg = format!("EXISTING:\n{}\n\nNEW:\n{}", existing, new);
    let result = ai::llm_chat_as(cfg, "gate", MERGE_PROMPT, &user_msg).await?;
    if let Some(ref u) = result.usage {
        let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
        let _ = db.log_llm_call("insert_merge", &result.model, u.prompt_tokens, u.completion_tokens, cached, result.duration_ms);
    }
    let trimmed = result.content.trim().to_string();
    if trimmed.is_empty() {
        return Err(EngramError::Internal("LLM returned empty merge".into()));
    }
    Ok(trimmed)
}
