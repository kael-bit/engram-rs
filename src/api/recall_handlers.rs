//! Recall and search handlers.

use axum::extract::{Path, Query, State};
use axum::Json;
use serde::Deserialize;
use tracing::{debug, warn};

use crate::error::EngramError;
use crate::{ai, db, recall, AppState};
use super::{blocking, get_namespace};

/// Simple keyword search — lighter than /recall, no scoring or budget logic.
#[derive(Deserialize)]
pub(super) struct SearchQuery {
    q: String,
    limit: Option<usize>,
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
        let hits = d.search_fts_ns(&query, limit, ns_filter.as_deref());
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

/// One-call session recovery: combines recent memories, core identity, and session context.
/// GET /resume?hours=4&workspace=engram,rust&limit=100
#[derive(Deserialize)]
pub(super) struct ResumeQuery {
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
        db.list_filtered(500, 0, ns.as_deref(), None, Some(&tag))
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

pub(super) async fn do_resume(
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

        // Workspace tag matching: exact match or prefix (e.g. "engram" matches
        // tag "engram" and "engram:proxy", but NOT "engram-rs")
        let ws_match = |m: &db::Memory| -> bool {
            if ws_tags.is_empty() { return true; }
            if m.tags.is_empty() { return true; } // untagged = universal
            m.tags.iter().any(|t| {
                let tl = t.to_lowercase();
                ws_tags.iter().any(|ws| tl == *ws || tl.starts_with(&format!("{ws}:")))
            })
        };

        // Session note: source="session" OR tag="session" (API-created notes use source=api)
        let is_session = |m: &db::Memory| -> bool {
            m.source == "session" || m.tags.iter().any(|t| t == "session")
        };

        // Use list_by_layer_meta — skip embedding blobs, resume doesn't need them.
        // DB already sorts by importance DESC.
        let core: Vec<db::Memory> = d
            .list_by_layer_meta_ns(db::Layer::Core, core_limit * 2, 0, ns_filter.as_deref())
            .into_iter()
            .filter(|m| ws_match(m))
            .take(core_limit)
            .collect();

        // Working: exclude session-source memories (they go in sessions/next_actions)
        let working: Vec<db::Memory> = d
            .list_by_layer_meta_ns(db::Layer::Working, core_limit * 2, 0, ns_filter.as_deref())
            .into_iter()
            .filter(|m| ws_match(m) && !is_session(m))
            .take(core_limit)
            .collect();

        let mut buffer: Vec<db::Memory> = d
            .list_by_layer_meta_ns(db::Layer::Buffer, 100, 0, ns_filter.as_deref())
            .into_iter()
            .filter(|m| !is_session(m))
            .collect();
        buffer.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        buffer.truncate(50);

        // Dedup: recent shouldn't repeat what's already in layer sections
        let seen: std::collections::HashSet<String> = core.iter()
            .chain(working.iter())
            .chain(buffer.iter())
            .map(|m| m.id.clone())
            .collect();

        let all_recent: Vec<db::Memory> = d
            .list_since_filtered(since_ms, 100, ns_filter.as_deref(), None, None, None)
            .unwrap_or_default();

        // Recent activity: dedup against layer sections so we don't repeat
        // Recent, sessions, next-actions: all deduped against layer sections.
        // If a session note is already in Working, don't repeat it.
        let recent: Vec<db::Memory> = all_recent.iter()
            .filter(|m| !is_session(m) && !seen.contains(&m.id))
            .take(20)
            .cloned()
            .collect();

        let mut next_actions = Vec::new();
        let mut sessions = Vec::new();
        for m in all_recent.into_iter().filter(|m| is_session(m)) {
            if seen.contains(&m.id) { continue; }
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
                let content = crate::safety::sanitize_for_output(&m.content);
                let mut obj = serde_json::json!({
                    "content": content,
                    "layer": m.layer as i32,
                    "importance": m.importance,
                    "created_at": m.created_at,
                });
                if !m.tags.is_empty() {
                    obj["tags"] = serde_json::json!(m.tags);
                }
                obj
            }).collect()
        } else {
            mems.iter().map(|m| {
                let mut val = serde_json::to_value(m).unwrap_or_default();
                if let Some(obj) = val.as_object_mut() {
                    obj.insert(
                        "content".into(),
                        serde_json::Value::String(crate::safety::sanitize_for_output(&m.content)),
                    );
                }
                val
            }).collect()
        }
    };

    // Apply budget: fill sections in priority order, stop when budget exhausted
    // Default 8000 chars (~2K tokens). 0 = unlimited.
    let budget_val = q.budget.unwrap_or(8000);
    let mut budget_left = if budget_val == 0 { usize::MAX } else { budget_val };

    let take_budget = |mems: &[db::Memory], budget: &mut usize, compact: bool| -> Vec<db::Memory> {
        if *budget == 0 { return vec![]; }
        let mut taken = Vec::new();
        for m in mems {
            let cost = if compact {
                m.content.len() + m.tags.iter().map(|t| t.len() + 3).sum::<usize>() + 20
            } else {
                m.content.len() + 250
            };
            if cost > *budget && !taken.is_empty() { break; }
            *budget = budget.saturating_sub(cost);
            taken.push(m.clone());
        }
        taken
    };

    // Core gets at most 60% of budget to leave room for working context.
    let core_cap = if budget_val == 0 { usize::MAX } else { budget_val * 60 / 100 };
    let mut core_budget = budget_left.min(core_cap);
    let core_out = take_budget(&core, &mut core_budget, compact);
    budget_left = budget_left.saturating_sub(budget_left.min(core_cap) - core_budget);

    let working_out = take_budget(&working, &mut budget_left, compact);
    let next_out = take_budget(&next_actions, &mut budget_left, compact);
    let sessions_out = take_budget(&sessions, &mut budget_left, compact);
    let recent_out = take_budget(&recent, &mut budget_left, compact);
    let buffer_out = take_budget(&buffer, &mut budget_left, compact);

    // Touch Working/Core memories served by resume — they're actively being used.
    // Skip Buffer to avoid inflating scores for unproven memories.
    {
        let db = state.db.clone();
        let ids: Vec<String> = core_out.iter().chain(working_out.iter())
            .map(|m| m.id.clone()).collect();
        tokio::task::spawn_blocking(move || {
            for id in &ids {
                let _ = db.touch(id);
            }
        });
    }

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

pub(super) async fn do_recall(
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
        req.rerank.unwrap_or(false) && state.ai.as_ref().is_some_and(super::super::ai::AiConfig::has_llm);
    let query_text = req.query.clone();
    let final_limit = req.limit.unwrap_or(20).min(100);

    if do_rerank {
        req.limit = Some(final_limit * 2);
    }

    let query_emb = if let Some(ref cfg) = state.ai {
        if cfg.has_embed() {
            // check cache first
            let cached = {
                let mut cache = state.embed_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
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
                            let mut cache = state.embed_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
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
        explicit_expand.unwrap_or(false) && state.ai.as_ref().is_some_and(super::super::ai::AiConfig::has_llm);
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
        && state.ai.as_ref().is_some_and(super::super::ai::AiConfig::has_llm)
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
