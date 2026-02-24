//! Recall and search handlers.

use axum::extract::{Path, Query, State};
use axum::Json;
use serde::Deserialize;
use tracing::{debug, warn};

use crate::error::EngramError;
use crate::{ai, db, recall, AppState};
use super::{blocking, get_namespace};

/// Parse next-action items from memory content.
///
/// Recognizes markers: `Next:`, `next:`, `下一步:`, `下一步：`, `待做:`, `待办:`, `TODO:`.
/// After the marker, splits by comma, Chinese enumeration comma (、), or newlines.
pub(crate) fn parse_next_actions(content: &str) -> Vec<String> {
    // Markers to look for (case-sensitive except Next/next)
    let markers: &[&str] = &[
        "Next:", "next:", "NEXT:",
        "下一步:", "下一步：",
        "待做:", "待做：", "待办:", "待办：",
        "TODO:", "Todo:",
    ];

    let mut actions = Vec::new();

    for marker in markers {
        if let Some(pos) = content.find(marker) {
            let after = &content[pos + marker.len()..];
            // Take until end of content or next sentence-ending pattern that isn't part of items
            let text = after.trim();
            if text.is_empty() {
                continue;
            }
            // Split by comma, Chinese comma, newline, or semicolon
            for item in text.split([',', '\n', '、', ';', '；']) {
                let trimmed = item.trim().trim_end_matches(['.', '。']);
                let trimmed = trimmed.trim();
                if !trimmed.is_empty() {
                    actions.push(trimmed.to_string());
                }
            }
            // Only use the first matching marker
            break;
        }
    }

    // Deduplicate (preserve order)
    let mut seen = std::collections::HashSet::new();
    actions.retain(|a| seen.insert(a.clone()));

    actions
}

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
            .unwrap_or_default()
            .into_iter()
            .filter(|m| ws_match(m))
            .take(core_limit)
            .collect();

        // Working: exclude session-source memories (they go in sessions/next_actions)
        let working: Vec<db::Memory> = d
            .list_by_layer_meta_ns(db::Layer::Working, core_limit * 2, 0, ns_filter.as_deref())
            .unwrap_or_default()
            .into_iter()
            .filter(|m| ws_match(m) && !is_session(m))
            .take(core_limit)
            .collect();

        let mut buffer: Vec<db::Memory> = d
            .list_by_layer_meta_ns(db::Layer::Buffer, 100, 0, ns_filter.as_deref())
            .unwrap_or_default()
            .into_iter()
            .filter(|m| !is_session(m))
            .collect();
        buffer.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        buffer.truncate(50);

        // Dedup: recent shouldn't repeat what's already in layer sections
        let layer_ids: std::collections::HashSet<String> = core.iter()
            .chain(working.iter())
            .chain(buffer.iter())
            .map(|m| m.id.clone())
            .collect();

        let all_recent: Vec<db::Memory> = d
            .list_since_filtered(since_ms, 100, ns_filter.as_deref(), None, None, None)
            .unwrap_or_default();

        // Recent captures recently-modified Working/Core that didn't make it
        // into the layer sections (e.g. low-importance Working that was just
        // updated). Dedup against layer sections to avoid repeats.
        let recent: Vec<db::Memory> = all_recent.iter()
            .filter(|m| !is_session(m) && !layer_ids.contains(&m.id))
            .take(20)
            .cloned()
            .collect();

        let mut next_actions = Vec::new();
        let mut sessions = Vec::new();
        for m in all_recent.into_iter().filter(|m| is_session(m)) {
            if layer_ids.contains(&m.id) { continue; }
            if m.tags.iter().any(|t| t == "next-action") {
                next_actions.push(m);
            } else {
                sessions.push(m);
            }
        }
        sessions.truncate(10);
        next_actions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        next_actions.truncate(3);

        // Core relevance filtering: use recent context (buffer + sessions) as
        // signal for what matters right now. Core memories unrelated to current
        // context are excluded — they're not deleted, just not in this resume.
        // Always-relevant: lesson, identity, procedural tags or kind.
        let core = if buffer.is_empty() && sessions.is_empty() && recent.is_empty() {
            // Fresh session with no context — return all Core unfiltered
            core
        } else {
            let context_ids: Vec<String> = buffer.iter()
                .chain(sessions.iter())
                .chain(recent.iter())
                .map(|m| m.id.clone())
                .collect();
            let context_embeds = d.get_embeddings_by_ids(&context_ids);

            if context_embeds.is_empty() {
                core // no embeddings available, skip filtering
            } else {
                // Compute centroid of context embeddings
                let dim = context_embeds[0].1.len();
                let mut centroid = vec![0.0f32; dim];
                for (_, emb) in &context_embeds {
                    for (i, v) in emb.iter().enumerate() {
                        if i < dim { centroid[i] += v; }
                    }
                }
                let n = context_embeds.len() as f32;
                for v in &mut centroid {
                    *v /= n;
                }

                let core_ids: Vec<String> = core.iter().map(|m| m.id.clone()).collect();
                let core_embeds = d.get_embeddings_by_ids(&core_ids);
                let core_embed_map: std::collections::HashMap<&str, &Vec<f32>> = core_embeds.iter()
                    .map(|(id, emb)| (id.as_str(), emb))
                    .collect();

                // No exemptions — every Core memory competes for slots via
                // cosine relevance to current context. Identity/lesson get
                // boosts but still must be somewhat relevant. Hard cap at 20
                // ensures resume stays manageable even at 600+ Core.
                let identity_boost = |m: &db::Memory| -> f64 {
                    if m.tags.iter().any(|t| matches!(t.as_str(), "identity" | "constraint" | "bootstrap")) {
                        0.25
                    } else if m.kind == "procedural" || m.tags.iter().any(|t| matches!(t.as_str(), "lesson" | "procedural" | "preference")) {
                        0.10
                    } else {
                        0.0
                    }
                };

                let mut scored: Vec<(db::Memory, f64)> = core.into_iter().map(|m| {
                    let raw = core_embed_map.get(m.id.as_str())
                        .map(|emb| ai::cosine_similarity(&centroid, emb))
                        .unwrap_or(0.4);
                    let boosted = raw + identity_boost(&m);
                    debug!(id = %crate::util::short_id(&m.id), raw = format!("{raw:.3}"),
                        score = format!("{boosted:.3}"),
                        content = %crate::util::truncate_chars(&m.content, 40),
                        "core relevance");
                    (m, boosted)
                }).collect();

                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let max_core = 20;
                let threshold = 0.35;
                let total = scored.len();
                let kept: Vec<db::Memory> = scored.into_iter()
                    .filter(|(_, s)| *s >= threshold)
                    .take(max_core)
                    .map(|(m, _)| m)
                    .collect();
                if kept.len() < total {
                    debug!(total, kept = kept.len(),
                        dropped = total - kept.len(),
                        threshold, max_core, "core relevance filtering");
                }
                kept
            }
        };

        (core, working, buffer, recent, sessions, next_actions)
    })
    .await?;

    let (core, working, buffer, recent, sessions, next_actions) = sections;
    let compact = q.compact.unwrap_or(true);

    // Helper: convert memories to compact or full format
    let to_json = |mems: &[db::Memory]| -> Vec<serde_json::Value> {
        if compact {
            mems.iter().map(|m| {
                let content = &m.content;
                let mut obj = serde_json::json!({
                    "content": content,
                    "layer": m.layer as i32,
                    "importance": m.importance,
                    "created_at": m.created_at,
                });
                if !m.tags.is_empty() {
                    obj["tags"] = serde_json::json!(m.tags);
                }
                if m.kind != "semantic" {
                    obj["kind"] = serde_json::json!(m.kind);
                }
                obj
            }).collect()
        } else {
            mems.iter().map(|m| {
                let mut val = serde_json::to_value(m).unwrap_or_default();
                if let Some(obj) = val.as_object_mut() {
                    obj.insert(
                        "content".into(),
                        serde_json::Value::String(m.content.clone()),
                    );
                }
                val
            }).collect()
        }
    };

    // Apply budget with proportional content truncation.
    // Default 16000 chars (~4K tokens). 0 = unlimited.
    let budget_val = q.budget.unwrap_or(16_000);
    let mut budget_left = if budget_val == 0 { usize::MAX } else { budget_val };

    // Fit memories into budget. When total exceeds budget, keep the most
    // important items at full length and compress the rest to one-line
    // summaries rather than truncating everything proportionally.
    let fit_section = |mems: &[db::Memory], budget: &mut usize, compact: bool| -> Vec<db::Memory> {
        if *budget == 0 || mems.is_empty() { return vec![]; }

        let overhead = if compact { 20usize } else { 250 };
        let total_cost: usize = mems.iter()
            .map(|m| m.content.len() + if compact { m.tags.iter().map(|t| t.len() + 3).sum::<usize>() } else { 0 } + overhead)
            .sum();

        if total_cost <= *budget {
            *budget -= total_cost;
            return mems.to_vec();
        }

        // Budget is tight. Strategy: keep as many items at full length as
        // possible, then compress remaining to one-line (~60 char) summaries.
        let summary_len = 60;
        let mut result = Vec::new();
        let mut spent = 0usize;

        for (i, m) in mems.iter().enumerate() {
            let tag_cost = if compact { m.tags.iter().map(|t| t.len() + 3).sum::<usize>() } else { 0 };
            let full_cost = m.content.len() + tag_cost + overhead;

            // How much would remaining items cost if compressed?
            let remaining_compressed: usize = mems[i+1..].iter()
                .map(|r| summary_len + if compact { r.tags.iter().map(|t| t.len() + 3).sum::<usize>() } else { 0 } + overhead)
                .sum();

            if spent + full_cost + remaining_compressed <= *budget {
                // Fits at full length with room for compressed rest
                spent += full_cost;
                result.push(m.clone());
            } else {
                // Compress this and all remaining items
                let mut item = m.clone();
                let first_line = m.content.lines().next().unwrap_or(&m.content);
                if first_line.len() > summary_len {
                    let boundary = first_line.floor_char_boundary(summary_len.saturating_sub(1));
                    item.content = format!("{}…", &first_line[..boundary]);
                } else if first_line.len() < m.content.len() {
                    item.content = format!("{}…", first_line);
                }
                let cost = item.content.len() + tag_cost + overhead;
                if spent + cost > *budget && !result.is_empty() { break; }
                spent += cost;
                result.push(item);
            }
        }
        *budget = budget.saturating_sub(spent);
        result
    };

    // Reserve 20% for recent context (sessions + recent + buffer) so they
    // never get starved by Core/Working bloat.
    let recent_reserve = if budget_val == 0 { 0 } else { budget_val * 20 / 100 };
    let main_budget = budget_left.saturating_sub(recent_reserve);

    // Core: 45% cap of total budget
    let core_cap = if budget_val == 0 { usize::MAX } else { budget_val * 45 / 100 };
    let mut core_budget = main_budget.min(core_cap);

    // Estimate cost of full Core listing. If it would blow the budget,
    // check for a pre-computed summary from consolidation.
    let core_overhead = if compact { 20usize } else { 250 };
    let core_total_cost: usize = core.iter()
        .map(|m| m.content.len() + core_overhead)
        .sum();

    let (core_out, core_summary_used) = if core_total_cost > core_budget && core.len() >= 10 {
        // Try the cached summary
        let db_sum = state.db.clone();
        let summary = tokio::task::spawn_blocking(move || db_sum.get_meta("core_summary"))
            .await
            .unwrap_or(None);
        if let Some(ref s) = summary {
            if s.len() + core_overhead < core_budget {
                core_budget -= s.len() + core_overhead;
                (vec![], true)
            } else {
                // Summary itself doesn't fit — fall back to hard truncation
                (fit_section(&core, &mut core_budget, compact), false)
            }
        } else {
            (fit_section(&core, &mut core_budget, compact), false)
        }
    } else {
        (fit_section(&core, &mut core_budget, compact), false)
    };
    budget_left = budget_left.saturating_sub(main_budget.min(core_cap) - core_budget);

    // Sessions + next_actions come before Working — continuity matters more
    let mut session_budget = budget_left.saturating_sub(recent_reserve);
    let sessions_out = fit_section(&sessions, &mut session_budget, compact);
    let next_tagged_out = fit_section(&next_actions, &mut session_budget, compact);
    budget_left = budget_left.saturating_sub(budget_left.saturating_sub(recent_reserve) - session_budget);

    let mut working_budget = budget_left.saturating_sub(recent_reserve);
    let working_out = fit_section(&working, &mut working_budget, compact);
    budget_left = budget_left.saturating_sub(budget_left.saturating_sub(recent_reserve) - working_budget);

    // Recent context gets whatever is left (at least the reserved 20%)
    let recent_out = fit_section(&recent, &mut budget_left, compact);
    let buffer_out = fit_section(&buffer, &mut budget_left, compact);

    // Parse next_actions from session memory content + explicitly tagged next-action memories.
    let mut parsed_actions: Vec<String> = Vec::new();
    // From explicitly tagged next-action memories
    for m in &next_tagged_out {
        let parsed = parse_next_actions(&m.content);
        if parsed.is_empty() {
            // If no marker found, use the whole content as an action
            let trimmed = m.content.trim().to_string();
            if !trimmed.is_empty() {
                parsed_actions.push(trimmed);
            }
        } else {
            parsed_actions.extend(parsed);
        }
    }
    // From session memories (which may contain "Next: ..." markers)
    for m in &sessions_out {
        parsed_actions.extend(parse_next_actions(&m.content));
    }
    // Deduplicate (preserve order)
    {
        let mut seen = std::collections::HashSet::new();
        parsed_actions.retain(|a| seen.insert(a.clone()));
    }

    // Only touch Buffer memories — they need access_count to get promoted.
    // Core is stable (touching inflates access_count to 200+ meaninglessly).
    // Working shouldn't be touched: it resets last_accessed which prevents
    // gate-rejected items from ever decaying.
    {
        let db = state.db.clone();
        let ids: Vec<String> = buffer_out.iter()
            .map(|m| m.id.clone()).collect();
        tokio::task::spawn_blocking(move || {
            for id in &ids {
                let _ = db.touch(id);
            }
        });
    }

    // Read the summary text if we decided to use it
    let core_summary_text = if core_summary_used {
        let db_s = state.db.clone();
        tokio::task::spawn_blocking(move || db_s.get_meta("core_summary"))
            .await.unwrap_or(None)
    } else {
        None
    };

    let core_json = if let Some(ref summary) = core_summary_text {
        // Single summary entry instead of individual memories
        serde_json::json!([{"content": summary, "kind": "summary", "tags": ["core-summary"]}])
    } else {
        serde_json::json!(to_json(&core_out))
    };

    let core_count = if core_summary_used { core.len() } else { core_out.len() };

    Ok(Json(serde_json::json!({
        "core": core_json,
        "working": to_json(&working_out),
        "buffer": to_json(&buffer_out),
        "recent": to_json(&recent_out),
        "sessions": to_json(&sessions_out),
        "next_actions": parsed_actions,
        "hours": hours,
        "core_count": core_count,
        "core_total": core.len(),
        "core_summary_used": core_summary_used,
        "working_count": working_out.len(),
        "buffer_count": buffer_out.len(),
        "recent_count": recent_out.len(),
        "session_count": sessions_out.len(),
        "next_action_count": parsed_actions.len(),
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

    let auto_rerank = std::env::var("ENGRAM_AUTO_RERANK").map(|v| v != "false").unwrap_or(false);
    let do_rerank = req.rerank.unwrap_or(auto_rerank)
        && state.ai.as_ref().is_some_and(super::super::ai::AiConfig::has_llm);
    let query_text = req.query.clone();
    let final_limit = req.limit.unwrap_or(20).min(100);

    if do_rerank {
        req.limit = Some(final_limit * 2);
    }

    let query_emb = if let Some(ref cfg) = state.ai {
        if cfg.has_embed() {
            // check cache first
            let cached = state.embed_cache.get(&query_text);
            if let Some(emb) = cached {
                debug!("embed cache hit for recall query");
                Some(emb)
            } else {
                match ai::get_embeddings(cfg, std::slice::from_ref(&query_text)).await {
                    Ok(mut v) => {
                        let emb = v.pop();
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
