//! Admin and maintenance handlers.

use axum::extract::{Query, State};
use axum::Json;
use serde::Deserialize;
use tracing::debug;

use crate::error::EngramError;
use crate::{ai, consolidate, db, AppState};
use super::{blocking, get_namespace, spawn_embed_batch};

pub(super) async fn do_consolidate(
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

/// LLM-powered memory audit. Reviews Core+Working, reorganizes via gate model.
/// Configure a capable model with ENGRAM_GATE_MODEL (e.g. claude-sonnet-4-5-20250514).
pub(super) async fn do_audit(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let ai = state.ai.as_ref().ok_or(EngramError::AiNotConfigured)?;
    let result = consolidate::audit_memories(ai, &state.db).await
        ?;
    Ok(Json(serde_json::to_value(&result).unwrap_or_default()))
}

/// Audit sandbox: dry-run audit that grades proposed operations without applying them.
pub(super) async fn do_audit_sandbox(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let ai = state.ai.as_ref().ok_or(EngramError::AiNotConfigured)?;
    let result = consolidate::sandbox_audit(ai, &state.db).await?;
    Ok(Json(serde_json::to_value(&result).unwrap_or_default()))
}

#[derive(Deserialize, Default)]
pub(super) struct RepairQuery {
    #[serde(default)]
    force: bool,
}

pub(super) async fn do_repair(
    State(state): State<AppState>,
    Query(q): Query<RepairQuery>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let db = state.db.clone();
    let force = q.force;
    let (orphans, rebuilt) = blocking(move || {
        if force { db.force_rebuild_fts() } else { db.repair_fts() }
    }).await??;

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
pub(super) struct VacuumQuery {
    full: Option<bool>,
}

pub(super) async fn do_vacuum(
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

pub(super) async fn proxy_flush(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    crate::proxy::flush_window(&state).await;
    Json(serde_json::json!({"status": "ok"}))
}

pub(super) async fn proxy_window(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let db = state.db.clone();
    match blocking(move || db.peek_proxy_turns()).await.unwrap_or(Err(crate::error::EngramError::Internal("spawn failed".into()))) {
        Ok(sessions) => {
            let mut result = serde_json::Map::new();
            for (key, turns) in sessions {
                let arr: Vec<serde_json::Value> = turns.into_iter().map(|(content, ts)| {
                    serde_json::json!({
                        "chars": content.len(),
                        "preview": crate::util::truncate_chars(&content, 200),
                        "created_at": ts
                    })
                }).collect();
                result.insert(key, serde_json::Value::Array(arr));
            }
            Json(serde_json::json!({"sessions": result}))
        }
        Err(e) => Json(serde_json::json!({"error": e.to_string()}))
    }
}

#[derive(Deserialize)]
pub(super) struct ExtractRequest {
    text: String,
    auto_embed: Option<bool>,
}

#[derive(serde::Serialize)]
pub(super) struct ExtractResponse {
    extracted: Vec<db::Memory>,
    count: usize,
}

pub(super) async fn do_extract(
    State(state): State<AppState>,
    Json(req): Json<ExtractRequest>,
) -> Result<Json<ExtractResponse>, EngramError> {
    let cfg = state.ai.as_ref().ok_or(EngramError::AiNotConfigured)?;

    if req.text.trim().is_empty() {
        return Err(EngramError::EmptyContent);
    }

    let extracted = ai::extract_memories(cfg, &req.text)
        .await
        ?;

    let auto_embed = req.auto_embed.unwrap_or(true);
    let mut memories = Vec::new();
    let mut embed_batch = Vec::new();

    for em in extracted {
        let facts_input = em.facts.clone();
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
        kind: em.kind,
        };

        let db = state.db.clone();
        let mem = blocking(move || db.insert(input))
            .await??;

        // Store extracted fact triples linked to this memory
        if let Some(facts) = facts_input {
            if !facts.is_empty() {
                let mem_id = mem.id.clone();
                let linked: Vec<db::FactInput> = facts.into_iter().map(|mut f| {
                    f.memory_id = Some(mem_id.clone());
                    f
                }).collect();
                let fdb = state.db.clone();
                let _ = blocking(move || fdb.insert_facts(linked, "default")).await;
            }
        }

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

pub(super) async fn do_export(
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

pub(super) async fn do_import(
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
        "skipped": memories_val.as_array().map(std::vec::Vec::len).unwrap_or(0).saturating_sub(imported),
    })))
}

// -- Trash (soft-delete recovery) --

#[derive(Deserialize, Default)]
pub(super) struct TrashQuery {
    limit: Option<usize>,
    offset: Option<usize>,
}

pub(super) async fn trash_list(
    State(state): State<AppState>,
    Query(q): Query<TrashQuery>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let limit = q.limit.unwrap_or(100);
    let offset = q.offset.unwrap_or(0);
    let db = state.db.clone();
    let total = {
        let db2 = db.clone();
        blocking(move || db2.trash_count()).await??
    };
    let entries = blocking(move || db.trash_list(limit, offset)).await??;
    let count = entries.len();
    Ok(Json(serde_json::json!({ "total": total, "count": count, "items": entries })))
}

pub(super) async fn trash_restore(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let db = state.db.clone();
    let restored = blocking(move || db.trash_restore(&id)).await??;
    Ok(Json(serde_json::json!({ "restored": restored })))
}

pub(super) async fn trash_purge(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let db = state.db.clone();
    let purged = blocking(move || db.trash_purge()).await??;
    Ok(Json(serde_json::json!({ "purged": purged })))
}
