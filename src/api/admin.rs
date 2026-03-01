//! Admin and maintenance handlers.

use axum::extract::{Query, State};
use axum::Json;
use serde::Deserialize;
use tracing::debug;

use crate::error::EngramError;
use crate::extract::LenientJson;
use crate::{ai, consolidate, db, AppState};
use super::{blocking, get_namespace, spawn_embed_batch};

#[derive(Deserialize, Default)]
pub(super) struct ConsolidateQuery {
    #[serde(default)]
    dry_run: bool,
}

pub(super) async fn do_consolidate(
    State(state): State<AppState>,
    Query(q): Query<ConsolidateQuery>,
    body: axum::body::Bytes,
) -> Result<Json<consolidate::ConsolidateResponse>, EngramError> {
    let parsed: Option<consolidate::ConsolidateRequest> = if body.is_empty() {
        None
    } else {
        serde_json::from_slice(&body).ok()
    };

    let result =
        consolidate::consolidate(state.db.clone(), parsed, state.ai.clone(), q.dry_run).await;

    // Trigger topiary rebuild after consolidation
    if let Some(ref tx) = state.topiary_trigger {
        let _ = tx.send(());
    }

    Ok(Json(result))
}

/// Topic-scoped distillation. Finds bloated topics and condenses overlapping memories.
pub(super) async fn do_audit(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let ai = state.ai.as_ref().ok_or(EngramError::AiNotConfigured)?;
    let result = consolidate::distill_topics(ai, &state.db).await?;
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
    LenientJson(req): LenientJson<ExtractRequest>,
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
        embedding: None,
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

    // Embedding guaranteed available by startup check
    if !embed_batch.is_empty() {
        spawn_embed_batch(state.db.clone(), cfg.clone(), embed_batch);
    }

    let count = memories.len();
    if count > 0 {
        state.last_activity.store(db::now_ms(), std::sync::atomic::Ordering::Relaxed);
    }
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
    LenientJson(body): LenientJson<serde_json::Value>,
) -> Result<Json<serde_json::Value>, EngramError> {
    // Dispatch: if body has "text" key → LLM extraction import
    // Otherwise fall through to legacy JSON import (requires "memories" key)
    if body.get("text").is_some() || body.get("prompt").is_some() {
        return do_import_text(state, headers, body).await;
    }

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

/// LLM-based text import: extract memories from raw text using a caller-provided prompt.
async fn do_import_text(
    state: AppState,
    headers: axum::http::HeaderMap,
    body: serde_json::Value,
) -> Result<Json<serde_json::Value>, EngramError> {
    let cfg = state.ai.as_ref().ok_or(EngramError::AiNotConfigured)?;

    // Validate required fields
    let text = body
        .get("text")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| EngramError::Validation("'text' is required and must be non-empty".into()))?;
    let prompt = body
        .get("prompt")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| EngramError::Validation("'prompt' is required and must be non-empty".into()))?;

    let namespace = body
        .get("namespace")
        .and_then(|v| v.as_str())
        .map(String::from)
        .or_else(|| get_namespace(&headers))
        .unwrap_or_else(|| "default".into());
    let model_override = body
        .get("model")
        .and_then(|v| v.as_str())
        .map(String::from);

    // Define the store_memories function schema for LLM tool calling
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "memories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Memory content text"
                        },
                        "kind": {
                            "type": "string",
                            "enum": ["semantic", "episodic", "procedural"],
                            "description": "semantic=facts/decisions/lessons (default). episodic=dated events. procedural=step-by-step workflows."
                        },
                        "tags": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "1-4 topic tags"
                        }
                    },
                    "required": ["content"]
                }
            }
        },
        "required": ["memories"]
    });

    // Call LLM with function calling
    #[derive(serde::Deserialize)]
    struct ImportMemory {
        content: String,
        #[serde(default)]
        kind: Option<String>,
        #[serde(default)]
        tags: Option<Vec<String>>,
    }

    #[derive(serde::Deserialize)]
    struct StoreMemoriesResult {
        #[serde(default)]
        memories: Vec<ImportMemory>,
    }

    let tool_result = ai::llm_tool_call_with_model::<StoreMemoriesResult>(
        cfg,
        "gate",
        model_override.as_deref(),
        prompt,
        text,
        "store_memories",
        "Store extracted memories from the provided text",
        schema,
    )
    .await
    .map_err(|e| {
        // LLM call failure → 500; parse failure already handled by llm_tool_call_with_model
        // which returns AiBackend error (maps to 502 via EngramError::status_code)
        tracing::error!(error = %e, "import LLM call failed");
        e
    })?;

    // Log LLM usage
    if let Some(ref u) = tool_result.usage {
        let cached = u
            .prompt_tokens_details
            .as_ref()
            .map_or(0, |d| d.cached_tokens);
        let _ = state.db.log_llm_call(
            "import",
            &tool_result.model,
            u.prompt_tokens,
            u.completion_tokens,
            cached,
            tool_result.duration_ms,
        );
    }

    let extracted = tool_result.value.memories;

    // Insert each memory through the normal path (dedup + embed)
    let mut imported_memories = Vec::new();
    for em in extracted {
        // Validate kind
        let kind = em.kind.as_deref().unwrap_or("semantic");
        if !db::is_valid_kind(kind) {
            tracing::warn!(kind = %kind, "import: skipping memory with invalid kind");
            continue;
        }

        let input = db::MemoryInput {
            content: em.content,
            layer: None,
            importance: None,
            source: Some("import".into()),
            tags: em.tags,
            supersedes: None,
            skip_dedup: None,
            namespace: Some(namespace.clone()),
            sync_embed: None,
            kind: em.kind,
            embedding: None,
        };

        let db = state.db.clone();
        match blocking(move || db.insert(input)).await {
            Ok(Ok(mem)) => {
                // Queue embedding
                if let Some(ref eq) = state.embed_queue {
                    eq.push(mem.id.clone(), mem.content.clone());
                } else if let Some(ref cfg) = state.ai {
                    super::spawn_embed(
                        state.db.clone(),
                        cfg.clone(),
                        mem.id.clone(),
                        mem.content.clone(),
                    );
                }
                imported_memories.push(serde_json::json!({
                    "id": mem.id,
                    "content": mem.content,
                    "kind": mem.kind,
                    "tags": mem.tags,
                }));
            }
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "import: failed to insert memory");
            }
            Err(e) => {
                tracing::warn!(error = %e, "import: spawn_blocking failed");
            }
        }
    }

    let count = imported_memories.len();
    if count > 0 {
        state
            .last_activity
            .store(db::now_ms(), std::sync::atomic::Ordering::Relaxed);
    }

    debug!(count, "import text complete");
    Ok(Json(serde_json::json!({
        "imported": count,
        "memories": imported_memories,
    })))
}

/// Import pre-extracted facts: parse numbered facts from text, send to LLM for
/// tag/kind/importance annotation (content is NOT modified), then insert through
/// the normal memory pipeline (dedup + embed).
pub(super) async fn do_import_facts(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    LenientJson(body): LenientJson<serde_json::Value>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let cfg = state.ai.as_ref().ok_or(EngramError::AiNotConfigured)?;

    let text = body
        .get("text")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| EngramError::Validation("'text' is required and must be non-empty".into()))?;

    let namespace = body
        .get("namespace")
        .and_then(|v| v.as_str())
        .map(String::from)
        .or_else(|| get_namespace(&headers))
        .unwrap_or_else(|| "default".into());

    // Parse numbered facts: "1. Some fact text" → (1, "Some fact text")
    let mut facts: Vec<(usize, String)> = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        // Match lines like "1. text" or "123. text"
        if let Some(dot_pos) = trimmed.find(". ") {
            if let Ok(id) = trimmed[..dot_pos].parse::<usize>() {
                let content = trimmed[dot_pos + 2..].trim().to_string();
                if !content.is_empty() {
                    facts.push((id, content));
                }
            }
        }
    }

    if facts.is_empty() {
        return Err(EngramError::Validation("no numbered facts found in text".into()));
    }

    // Call LLM for annotation in batches to avoid timeout on large inputs
    #[derive(serde::Deserialize)]
    struct FactAnnotation {
        id: usize,
        kind: String,
        #[serde(default)]
        tags: Vec<String>,
        #[serde(default = "default_importance")]
        importance: f64,
    }
    fn default_importance() -> f64 { 0.5 }

    #[derive(serde::Deserialize)]
    struct AnnotationResult {
        #[serde(default)]
        memories: Vec<FactAnnotation>,
    }

    const BATCH_SIZE: usize = 30;
    let mut all_annotations: Vec<FactAnnotation> = Vec::new();

    for batch in facts.chunks(BATCH_SIZE) {
        let user_msg: String = batch
            .iter()
            .map(|(id, text)| format!("{id}. {text}"))
            .collect::<Vec<_>>()
            .join("\n");

        let tool_result = ai::llm_tool_call_with_model::<AnnotationResult>(
            cfg,
            "gate",
            None, // use default gate model (sonnet)
            crate::prompts::IMPORT_FACTS_SYSTEM,
            &user_msg,
            "store_memories",
            "Annotate each fact with kind, tags, and importance",
            crate::prompts::import_facts_schema(),
        )
        .await
        .map_err(|e| {
            tracing::error!(batch_start = batch.first().map(|f| f.0), error = %e, "import/facts LLM batch failed");
            e
        })?;

        // Log LLM usage per batch
        if let Some(ref u) = tool_result.usage {
            let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
            let _ = state.db.log_llm_call(
                "import_facts",
                &tool_result.model,
                u.prompt_tokens,
                u.completion_tokens,
                cached,
                tool_result.duration_ms,
            );
        }

        all_annotations.extend(tool_result.value.memories);
        debug!(batch_size = batch.len(), annotations = all_annotations.len(), "import/facts batch complete");
    }

    // Build a lookup: id → original text
    let fact_map: std::collections::HashMap<usize, &str> = facts
        .iter()
        .map(|(id, text)| (*id, text.as_str()))
        .collect();

    let total_annotated = all_annotations.len();
    let mut imported = Vec::new();
    let mut skipped = 0usize;

    for ann in all_annotations {
        let Some(original_text) = fact_map.get(&ann.id) else {
            tracing::warn!(id = ann.id, "import/facts: annotation ID not found in parsed facts");
            skipped += 1;
            continue;
        };

        if !db::is_valid_kind(&ann.kind) {
            tracing::warn!(id = ann.id, kind = %ann.kind, "import/facts: invalid kind, defaulting to semantic");
        }
        let kind = if db::is_valid_kind(&ann.kind) { ann.kind.clone() } else { "semantic".into() };

        let importance = ann.importance.clamp(0.0, 1.0);

        let input = db::MemoryInput {
            content: original_text.to_string(),
            layer: Some(2), // working — pre-extracted facts skip buffer
            importance: Some(importance),
            source: Some("import".into()),
            tags: Some(ann.tags),
            supersedes: None,
            skip_dedup: None,
            namespace: Some(namespace.clone()),
            sync_embed: None,
            kind: Some(kind.clone()),
            embedding: None,
        };

        let db = state.db.clone();
        match blocking(move || db.insert(input)).await {
            Ok(Ok(mem)) => {
                if let Some(ref eq) = state.embed_queue {
                    eq.push(mem.id.clone(), mem.content.clone());
                } else if let Some(ref cfg) = state.ai {
                    super::spawn_embed(
                        state.db.clone(),
                        cfg.clone(),
                        mem.id.clone(),
                        mem.content.clone(),
                    );
                }
                imported.push(serde_json::json!({
                    "id": ann.id,
                    "memory_id": mem.id,
                    "kind": kind,
                    "tags": mem.tags,
                    "importance": importance,
                }));
            }
            Ok(Err(e)) => {
                tracing::warn!(id = ann.id, error = %e, "import/facts: insert failed");
                skipped += 1;
            }
            Err(e) => {
                tracing::warn!(id = ann.id, error = %e, "import/facts: spawn_blocking failed");
                skipped += 1;
            }
        }
    }

    let count = imported.len();
    let unannotated = facts.len().saturating_sub(total_annotated);
    if count > 0 {
        state.last_activity.store(db::now_ms(), std::sync::atomic::Ordering::Relaxed);
    }

    debug!(count, skipped, unannotated, "import/facts complete");
    Ok(Json(serde_json::json!({
        "imported": count,
        "total_facts": facts.len(),
        "skipped": skipped,
        "unannotated": unannotated,
        "memories": imported,
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
    headers: axum::http::HeaderMap,
    Query(q): Query<TrashQuery>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let limit = q.limit.unwrap_or(100);
    let offset = q.offset.unwrap_or(0);
    let ns = get_namespace(&headers);
    let db = state.db.clone();
    let total = {
        let db2 = db.clone();
        let ns2 = ns.clone();
        blocking(move || db2.trash_count(ns2.as_deref())).await??
    };
    let ns2 = ns.clone();
    let entries = blocking(move || db.trash_list(limit, offset, ns2.as_deref())).await??;
    let count = entries.len();
    Ok(Json(serde_json::json!({ "total": total, "count": count, "items": entries })))
}

pub(super) async fn trash_restore(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let db = state.db.clone();
    let ns = get_namespace(&headers);
    let id_copy = id.clone();
    let restored = blocking(move || db.trash_restore(&id_copy, ns.as_deref())).await??;

    // Queue embedding for the restored memory so it's searchable immediately
    if restored {
        if let Some(ref cfg) = state.ai {
            let db2 = state.db.clone();
            let id2 = id.clone();
            if let Ok(Some(mem)) = blocking(move || db2.get(&id2)).await? {
                super::spawn_embed(state.db.clone(), cfg.clone(), mem.id, mem.content);
            }
        }
        state.last_activity.store(db::now_ms(), std::sync::atomic::Ordering::Relaxed);
    }

    Ok(Json(serde_json::json!({ "restored": restored })))
}

pub(super) async fn trash_purge(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
) -> Result<Json<serde_json::Value>, EngramError> {
    let db = state.db.clone();
    let ns = get_namespace(&headers);
    let purged = blocking(move || db.trash_purge(ns.as_deref())).await??;
    Ok(Json(serde_json::json!({ "purged": purged })))
}

#[derive(Deserialize)]
pub(super) struct LlmUsageQuery {
    #[serde(default = "default_days")]
    days: u32,
}
fn default_days() -> u32 { 7 }

pub(super) async fn llm_usage(
    State(state): State<AppState>,
    Query(q): Query<LlmUsageQuery>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let db = state.db.clone();
    let days = q.days;
    let (summary, daily) = blocking(move || {
        let s = db.llm_usage_summary()?;
        let d = db.llm_usage_daily(days)?;
        Ok::<_, EngramError>((s, d))
    }).await??;

    Ok(Json(serde_json::json!({
        "summary": summary,
        "daily": daily,
    })))
}

pub(super) async fn clear_llm_usage(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let db = state.db.clone();
    let count = blocking(move || db.clear_llm_usage()).await??;
    Ok(Json(serde_json::json!({ "deleted": count })))
}
