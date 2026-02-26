//! Topiary topic API handlers.

use axum::extract::State;
use axum::Json;
use serde::Deserialize;
use tracing::debug;

use crate::error::EngramError;
use crate::AppState;
use super::blocking;

/// POST /topic — fetch topic details by IDs.
#[derive(Deserialize)]
pub(super) struct TopicRequest {
    ids: Vec<String>,
}

pub(super) async fn topiary_topic_handler(
    State(state): State<AppState>,
    Json(body): Json<TopicRequest>,
) -> Result<Json<serde_json::Value>, EngramError> {
    let db = state.db.clone();

    // Load the full serialized TopicTree and entry IDs
    let (full_tree_json, entry_ids_json) = blocking(move || {
        let tree = db.get_meta("topiary_tree_full");
        let ids = db.get_meta("topiary_entry_ids");
        (tree, ids)
    })
    .await?;

    let full_tree_json = full_tree_json
        .ok_or_else(|| EngramError::Internal("topiary tree not built yet".into()))?;
    let entry_ids_json = entry_ids_json
        .ok_or_else(|| EngramError::Internal("topiary entry IDs not available".into()))?;

    let full_tree: crate::topiary::TopicTree = serde_json::from_str(&full_tree_json)
        .map_err(|e| EngramError::Internal(format!("failed to parse topiary tree: {e}")))?;
    let entry_ids: Vec<String> = serde_json::from_str(&entry_ids_json)
        .map_err(|e| EngramError::Internal(format!("failed to parse entry IDs: {e}")))?;

    let mut result = serde_json::Map::new();

    for requested_id in &body.ids {
        if let Some(leaf) = crate::topiary::find_leaf(&full_tree.roots, requested_id) {
            let name = leaf.name.as_deref().unwrap_or("unnamed");
            // Resolve member indices to entry IDs, then load memories
            let mem_ids: Vec<String> = leaf.members.iter()
                .filter_map(|&idx| entry_ids.get(idx).cloned())
                .collect();

            let db2 = state.db.clone();
            let memories: Vec<serde_json::Value> = blocking(move || {
                mem_ids.iter().filter_map(|id| {
                    db2.get(id).ok().flatten().map(|m| {
                        serde_json::json!({
                            "id": m.id,
                            "content": m.content,
                            "layer": m.layer as i32,
                            "importance": m.importance,
                            "tags": m.tags,
                        })
                    })
                }).collect()
            }).await?;

            debug!(topic_id = %requested_id, name = %name, memories = memories.len(), "topic lookup");

            result.insert(requested_id.clone(), serde_json::json!({
                "name": name,
                "memories": memories,
            }));
        }
    }

    Ok(Json(serde_json::Value::Object(result)))
}

/// Build a concise Topics section from the cached topiary tree JSON.
/// Used by the resume endpoint to include topic summary.
pub(super) fn build_topiary_resume_section(tree_data: &serde_json::Value) -> Option<String> {
    let topics = tree_data.get("topics").and_then(|v| v.as_array())?;

    let mut leaves: Vec<(&str, &str, u64)> = Vec::new();
    collect_topic_leaves(topics, &mut leaves);

    if leaves.is_empty() {
        return None;
    }

    // Sort by member count descending
    leaves.sort_by(|a, b| b.2.cmp(&a.2));

    let mut out = String::new();
    for (id, name, count) in &leaves {
        out.push_str(&format!("{}: \"{}\" [{}]\n", id, name, count));
    }

    Some(out)
}

/// Find a topic by ID recursively in the JSON topic tree.
#[allow(dead_code)]
pub(super) fn find_topic_by_id<'a>(topics: &'a [serde_json::Value], id: &str) -> Option<&'a serde_json::Value> {
    for topic in topics {
        if topic.get("id").and_then(|v| v.as_str()) == Some(id) {
            return Some(topic);
        }
        if let Some(children) = topic.get("children").and_then(|v| v.as_array()) {
            if let Some(found) = find_topic_by_id(children, id) {
                return Some(found);
            }
        }
    }
    None
}

/// Collect all leaf topics recursively.
pub(super) fn collect_topic_leaves<'a>(topics: &'a [serde_json::Value], out: &mut Vec<(&'a str, &'a str, u64)>) {
    for topic in topics {
        if let Some(children) = topic.get("children").and_then(|v| v.as_array()) {
            collect_topic_leaves(children, out);
        } else {
            let id = topic.get("id").and_then(|v| v.as_str()).unwrap_or("?");
            let name = topic.get("name").and_then(|v| v.as_str()).unwrap_or("unnamed");
            let count = topic.get("member_count").and_then(|v| v.as_u64()).unwrap_or(0);
            out.push((id, name, count));
        }
    }
}
