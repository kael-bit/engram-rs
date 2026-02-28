//! Resume endpoint — session recovery handler.

use axum::extract::{Query, State};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use std::collections::HashMap;

use crate::error::EngramError;
use crate::{db, AppState};
use super::blocking;
use super::get_namespace;
use super::topiary_api::build_topiary_resume_section;

/// One-call session recovery.
/// GET /resume?recent_epochs=24&workspace=engram,rust&limit=100
#[derive(Deserialize)]
pub(super) struct ResumeQuery {
    recent_epochs: Option<i64>,
    ns: Option<String>,
    workspace: Option<String>,
    limit: Option<usize>,
    compact: Option<bool>,
    budget: Option<usize>,
    format: Option<String>,
}

pub(super) async fn do_resume(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Query(mut q): Query<ResumeQuery>,
) -> Result<Response, EngramError> {
    if q.ns.is_none() {
        q.ns = get_namespace(&headers);
    }
    let ns_filter = q.ns;
    let recent_epochs = q.recent_epochs.unwrap_or(24);

    // Epoch-based Recent: get current epoch, compute cutoff
    let db_for_epoch = state.db.clone();
    let epoch_cutoff = {
        let current_epoch: i64 = db_for_epoch.get_meta("consolidation_epoch")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        (current_epoch - recent_epochs).max(0)
    };

    let core_limit = q.limit.unwrap_or(100);

    // Parse workspace tags
    let ws_tags: Vec<String> = q.workspace
        .or_else(|| std::env::var("ENGRAM_WORKSPACE").ok())
        .map(|w| w.split(',').map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()).collect())
        .unwrap_or_default();

    let db = state.db.clone();
    let sections = blocking(move || {
        let d = db;

        let ws_match = |m: &db::Memory| -> bool {
            if ws_tags.is_empty() { return true; }
            if m.tags.is_empty() { return true; }
            m.tags.iter().any(|t| {
                let tl = t.to_lowercase();
                ws_tags.iter().any(|ws| tl == *ws || tl.starts_with(&format!("{ws}:")))
            })
        };

        // When a project namespace is set, also include "default" namespace
        let include_default = ns_filter.as_deref().is_some_and(|ns| ns != "default");

        // === Core section ===
        let mut core: Vec<db::Memory> = d
            .list_by_layer_meta_ns(db::Layer::Core, core_limit * 2, 0, ns_filter.as_deref())
            .unwrap_or_default();
        if include_default {
            let default_core = d
                .list_by_layer_meta_ns(db::Layer::Core, core_limit, 0, Some("default"))
                .unwrap_or_default();
            let existing: std::collections::HashSet<String> = core.iter().map(|m| m.id.clone()).collect();
            core.extend(default_core.into_iter().filter(|m| !existing.contains(&m.id)));
        }
        let mut core: Vec<db::Memory> = core.into_iter()
            .filter(|m| ws_match(m))
            .take(core_limit)
            .collect();

        // Sort core by unified memory_weight
        core.sort_by(|a, b| {
            crate::scoring::memory_weight(b)
                .partial_cmp(&crate::scoring::memory_weight(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // === Recent section: non-Core memories modified in recent epochs ===
        let all_recent: Vec<db::Memory> = d
            .list_by_epoch(epoch_cutoff, 200, ns_filter.as_deref())
            .unwrap_or_default();

        let core_ids: std::collections::HashSet<String> = core.iter().map(|m| m.id.clone()).collect();

        // Recent: non-Core, sorted by time descending (already from DB)
        let recent: Vec<db::Memory> = all_recent.into_iter()
            .filter(|m| m.layer != db::Layer::Core && !core_ids.contains(&m.id))
            .collect();

        // === Trigger tags ===
        // Collect all tags matching "trigger:*" with their frequency
        let trigger_tags: Vec<(String, i64)> = {
            let conn = d.conn().ok();
            if let Some(c) = conn {
                let mut stmt = c.prepare(
                    "SELECT m.tags, m.access_count FROM memories m WHERE m.tags LIKE '%trigger:%'"
                ).ok();
                if let Some(ref mut s) = stmt {
                    let mut tag_counts: HashMap<String, i64> = HashMap::new();
                    let rows = s.query_map([], |row| {
                        let tags_str: String = row.get(0)?;
                        let access_count: i64 = row.get(1)?;
                        Ok((tags_str, access_count))
                    });
                    if let Ok(rows) = rows {
                        for row in rows.flatten() {
                            let (tags_str, access_count) = row;
                            // tags are stored as JSON array, e.g. ["lesson","trigger:deploy"]
                            let tags: Vec<String> = serde_json::from_str(&tags_str).unwrap_or_default();
                            for tag in &tags {
                                if let Some(name) = tag.strip_prefix("trigger:") {
                                    *tag_counts.entry(name.to_string()).or_insert(0) += access_count.max(1);
                                }
                            }
                        }
                    }
                    let mut sorted: Vec<(String, i64)> = tag_counts.into_iter().collect();
                    sorted.sort_by(|a, b| b.1.cmp(&a.1));
                    sorted
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            }
        };

        (core, recent, trigger_tags)
    })
    .await?;

    let (core, recent, trigger_tags) = sections;
    let compact = q.compact.unwrap_or(true);

    // Helper: convert memories to compact or full format
    let to_json = |mems: &[db::Memory]| -> Vec<serde_json::Value> {
        if compact {
            mems.iter().map(|m| {
                let mut obj = serde_json::json!({
                    "content": &m.content,
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
                    obj.insert("content".into(), serde_json::Value::String(m.content.clone()));
                }
                val
            }).collect()
        }
    };

    // Load topiary data for Topics section
    let (topiary_section, working_count, buffer_count) = {
        let db_t = state.db.clone();
        let tree_json = tokio::task::spawn_blocking(move || db_t.get_meta("topiary_tree"))
            .await
            .ok()
            .flatten();

        let mut working_count = 0usize;
        let mut buffer_count = 0usize;

        let section = if let Some(ref tj) = tree_json {
            if let Ok(tree_data) = serde_json::from_str::<serde_json::Value>(tj) {
                // Count working/buffer from memory stats
                let db_stats = state.db.clone();
                let stats = tokio::task::spawn_blocking(move || db_stats.stats())
                    .await
                    .unwrap_or(db::Stats { total: 0, buffer: 0, working: 0, core: 0, by_kind: db::KindStats::default() });
                working_count = stats.working;
                buffer_count = stats.buffer;

                build_topiary_resume_section(&tree_data)
            } else {
                None
            }
        } else {
            None
        };

        (section, working_count, buffer_count)
    };

    // Budget for text output
    let budget_val = q.budget.unwrap_or(12_000);
    let adaptive_budget = if budget_val == 0 { usize::MAX } else { budget_val };

    let want_json = q.format.as_deref() == Some("json");

    if want_json {
        // JSON response path — keep compact format unchanged
        Ok(Json(serde_json::json!({
            "core": to_json(&core),
            "recent": to_json(&recent),
            "topics": topiary_section,
            "triggers": trigger_tags.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(),
            "recent_epochs": recent_epochs,
            "core_count": core.len(),
            "recent_count": recent.len(),
        })).into_response())
    } else {
        // Text format: 4-section output
        let text = format_resume_text(
            &core,
            &recent,
            topiary_section.as_deref(),
            core.len(),
            working_count,
            buffer_count,
            &trigger_tags,
            adaptive_budget,
        );
        Ok((
            [(axum::http::header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            text,
        ).into_response())
    }
}

fn format_resume_text(
    core: &[db::Memory],
    recent: &[db::Memory],
    topiary_section: Option<&str>,
    core_count: usize,
    working_count: usize,
    buffer_count: usize,
    trigger_tags: &[(String, i64)],
    budget: usize,
) -> String {
    let mut lines = Vec::new();
    let char_cost = |s: &str| -> usize { s.chars().count() };
    let mut budget_left = budget;

    // === Core section ===
    // Full text, no truncation, up to ~2k tokens (~8000 chars)
    let core_budget = 8000.min(budget_left);
    if !core.is_empty() {
        lines.push(format!("=== Core ({}) ===", core.len()));
        let mut spent = 0usize;
        for m in core {
            let cost = char_cost(&m.content) + 2; // newline overhead
            if spent + cost > core_budget && spent > 0 {
                break;
            }
            let kind_tag = if m.kind != "semantic" { format!(" ({})", m.kind) } else { String::new() };
            lines.push(format!("{}{}", m.content, kind_tag));
            spent += cost;
        }
        budget_left = budget_left.saturating_sub(spent);
        lines.push(String::new());
    }

    // === Recent section ===
    // Non-Core memories, time descending, up to ~1k tokens (~4000 chars)
    let recent_budget = 4000.min(budget_left);
    if !recent.is_empty() {
        lines.push("=== Recent ===".to_string());
        let mut spent = 0usize;
        for m in recent {
            let cost = char_cost(&m.content) + 2;
            if spent + cost > recent_budget && spent > 0 {
                break;
            }
            lines.push(m.content.clone());
            spent += cost;
        }
        lines.push(String::new());
    }

    // === Topics section ===
    if topiary_section.is_some() || !trigger_tags.is_empty() {
        lines.push(format!("=== Topics (Core: {}, Working: {}, Buffer: {}) ===", core_count, working_count, buffer_count));
        if let Some(topics) = topiary_section {
            lines.push(topics.trim_end().to_string());
        }
        lines.push(String::new());

        // Triggers sub-section
        if !trigger_tags.is_empty() {
            let trigger_names: Vec<&str> = trigger_tags.iter().map(|(t, _)| t.as_str()).collect();
            lines.push(format!("Triggers: {}", trigger_names.join(", ")));
        }
        lines.push(String::new());
    }

    lines.join("\n")
}

// (format_timestamp removed — Recent section no longer includes timestamps)
