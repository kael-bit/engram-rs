//! Recall and search handlers.

use axum::extract::{Path, Query, State};
use axum::Json;
use serde::Deserialize;
use tracing::{debug, warn};

use crate::error::EngramError;
use crate::extract::LenientJson;
use crate::scoring::MemoryResult;
use crate::{ai, db, recall, thresholds, AppState};
use super::{blocking, get_namespace};

/// Simple keyword search — lighter than /recall, no scoring or budget logic.
#[derive(Deserialize)]
pub(super) struct SearchQuery {
    q: String,
    limit: Option<usize>,
    #[serde(alias = "namespace")]
    ns: Option<String>,
}

pub(super) async fn quick_search(
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
        let hits = d.search_fts_ns(&query, limit, ns_filter.as_deref()).unwrap_or_default();
        let memories: Vec<db::Memory> = hits
            .into_iter()
            .filter_map(|(id, _)| d.get(&id).ok().flatten())
            .collect();
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
pub(super) struct RecentQuery {
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
    #[serde(alias = "namespace")]
    ns: Option<String>,
}

pub(super) async fn list_recent(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Query(mut q): Query<RecentQuery>,
) -> Result<Json<serde_json::Value>, EngramError> {
    if q.ns.is_none() {
        q.ns = get_namespace(&headers);
    }
    let hours = q.hours.unwrap_or(2.0).clamp(0.0, 87_600.0);
    let limit = q.limit.unwrap_or(20).min(100);
    let layer_filter = q.layer;
    let min_imp = q.min_importance.unwrap_or(0.0);
    let source_filter = q.source;
    let ns_filter = q.ns;
    let since_ms = db::now_ms() - (hours * 3_600_000.0) as i64;

    let db = state.db.clone();
    let result = blocking(move || {
        let d = db;
        d.list_since_filtered(
            since_ms,
            limit,
            ns_filter.as_deref(),
            layer_filter,
            if min_imp > 0.0 { Some(min_imp) } else { None },
            source_filter.as_deref(),
        )
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

/// Fetch memories tagged with `trigger:{action}`. Used for pre-action
/// recall — e.g. before `git push`, recall lessons about what not to commit.
pub(super) async fn get_triggers(
    State(state): State<AppState>,
    Path(action): Path<String>,
    headers: axum::http::HeaderMap,
) -> Result<Json<serde_json::Value>, EngramError> {
    let ns = get_namespace(&headers);
    let tag = format!("trigger:{action}");
    let db = state.db.clone();
    let db2 = state.db.clone();

    let memories: Vec<db::Memory> = blocking(move || {
        db.list_filtered(500, 0, ns.as_deref(), None, Some(&tag), None)
    }).await?.unwrap_or_default();

    // Sort triggers by unified weight (most important first)
    let mut memories = memories;
    memories.sort_by(|a, b| {
        crate::scoring::memory_weight(b)
            .partial_cmp(&crate::scoring::memory_weight(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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
        "memories": memories.iter()
            .map(|m| MemoryResult::from_memory(m, crate::scoring::memory_weight(m)))
            .collect::<Vec<_>>(),
    })))
}

pub(super) async fn do_recall(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    LenientJson(mut req): LenientJson<recall::RecallRequest>,
) -> Result<Json<serde_json::Value>, EngramError> {
    if req.query.is_empty() {
        return Err(EngramError::EmptyQuery);
    }
    if req.namespace.is_none() {
        req.namespace = get_namespace(&headers);
    }
    // Default min_score at API layer (recall function defaults to 0.0 for testability)
    if req.min_score.is_none() {
        req.min_score = Some(0.30);
    }

    let query_text = req.query.clone();

    // Embedding guaranteed available by startup check
    let query_emb = if let Some(ref cfg) = state.ai {
        // check cache first
        let cached = state.embed_cache.get(&query_text);
        if let Some(emb) = cached {
            debug!("embed cache hit for recall query");
            Some(emb)
        } else {
            match ai::get_embeddings(cfg, std::slice::from_ref(&query_text)).await {
                Ok(er) => {
                    if let Some(ref u) = er.usage {
                        let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
                        let _ = state.db.log_llm_call("recall_embed", &cfg.embed_model, u.prompt_tokens, u.completion_tokens, cached, 0);
                    }
                    let emb = er.embeddings.into_iter().next();
                    if let Some(ref e) = emb {
                        state.embed_cache.insert(query_text.clone(), e.clone());
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
    };

    let explicit_expand = req.expand;
    let do_expand =
        explicit_expand.unwrap_or(false) && state.ai.as_ref().is_some_and(super::super::ai::AiConfig::has_llm);
    let expanded = if do_expand {
        if let Some(ref cfg) = state.ai {
            let (q, meta) = ai::expand_query(cfg, &query_text).await;
            if let Some(m) = meta {
                if let Some(ref u) = m.usage {
                    let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
                    let _ = state.db.log_llm_call("query_expand", &m.model, u.prompt_tokens, u.completion_tokens, cached, m.duration_ms);
                }
            }
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
        && state.ai.as_ref().is_some_and(super::super::ai::AiConfig::has_llm)
        && result.memories.first().is_none_or(|m| m.relevance < thresholds::AUTO_EXPAND_THRESHOLD)
    {
        if let Some(ref cfg) = state.ai {
            let (eq, meta) = ai::expand_query(cfg, &query_text).await;
            if let Some(m) = meta {
                if let Some(ref u) = m.usage {
                    let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
                    let _ = state.db.log_llm_call("query_expand", &m.model, u.prompt_tokens, u.completion_tokens, cached, m.duration_ms);
                }
            }
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

    // attach expanded queries to response
    let final_expanded = if !expanded_for_response.is_empty() {
        Some(expanded_for_response)
    } else {
        auto_expanded
    };
    if let Some(eq) = final_expanded {
        result.expanded_queries = Some(eq);
    }

    // Map to clean API response
    let memories: Vec<MemoryResult> = result.memories.iter()
        .map(|sm| MemoryResult::from_memory(&sm.memory, sm.score))
        .collect();

    let mut resp = serde_json::json!({
        "memories": memories,
        "query": query_text,
        "total": result.total,
    });

    if let Some(ref eq) = result.expanded_queries {
        resp["expanded_queries"] = serde_json::json!(eq);
    }
    if let Some(ref fc) = result.fact_chains {
        if !fc.is_empty() {
            resp["fact_chains"] = serde_json::json!(fc);
        }
    }

    Ok(Json(resp))
}
