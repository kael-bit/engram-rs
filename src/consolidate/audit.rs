//! Topic-scoped distillation: when a topic has too many members, condense
//! overlapping memories into fewer, richer ones. Replaces the old global-scan
//! audit. One topic per LLM call, 1-2 topics per consolidation cycle.

use crate::ai::{self, AiConfig};
use crate::db::{Memory, MemoryDB};
use crate::error::EngramError;
use crate::prompts;
use crate::thresholds;
use crate::SharedDB;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

// ── Result types ───────────────────────────────────────────────────────────

/// Result of a single topic distillation.
#[derive(Debug, Default, Serialize)]
pub struct DistillResult {
    pub topic_name: String,
    pub input_count: usize,
    pub output_count: usize,
    pub absorbed_ids: Vec<String>,
}

/// Result of a full distillation cycle.
#[derive(Debug, Default, Serialize)]
pub struct AuditResult {
    pub topics_reviewed: usize,
    pub topics_distilled: usize,
    pub total_absorbed: usize,
    pub details: Vec<DistillResult>,
}

// ── LLM response schema ───────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct DistillResponse {
    distilled: Vec<DistilledEntry>,
}

#[derive(Debug, Deserialize)]
struct DistilledEntry {
    content: String,
    #[serde(default)]
    source_ids: Vec<String>,
    #[serde(default)]
    kind: Option<String>,
}

// ── Public API ─────────────────────────────────────────────────────────────

/// Run topic distillation: find bloated topics, condense 1-2 per cycle.
/// Called from consolidation loop or manually via POST /audit.
pub async fn distill_topics(
    cfg: &AiConfig,
    db: &SharedDB,
) -> Result<AuditResult, EngramError> {
    let mut result = AuditResult::default();

    // 1. Load the current topiary tree to find bloated topics
    let db2 = db.clone();
    let bloated = tokio::task::spawn_blocking(move || find_bloated_topics(&db2))
        .await
        .map_err(|e| EngramError::Internal(format!("spawn: {e}")))?;

    if bloated.is_empty() {
        return Ok(result);
    }

    result.topics_reviewed = bloated.len();

    // 2. Distill up to thresholds::DISTILL_MAX_PER_CYCLE topics
    for topic in bloated.into_iter().take(thresholds::DISTILL_MAX_PER_CYCLE) {
        match distill_one_topic(cfg, db, &topic).await {
            Ok(dr) => {
                result.total_absorbed += dr.absorbed_ids.len();
                result.topics_distilled += 1;
                info!(
                    topic = %dr.topic_name,
                    input = dr.input_count,
                    output = dr.output_count,
                    absorbed = dr.absorbed_ids.len(),
                    "topic distilled"
                );
                result.details.push(dr);
            }
            Err(e) => {
                warn!(topic = %topic.name, error = %e, "topic distillation failed");
            }
        }
    }

    Ok(result)
}

// ── Internal ───────────────────────────────────────────────────────────────

/// A bloated topic: name + member memory IDs, sorted by most bloated first.
struct BloatedTopic {
    name: String,
    member_ids: Vec<String>,
}

/// Parse the cached topiary tree and find leaf topics exceeding the threshold.
fn find_bloated_topics(db: &MemoryDB) -> Vec<BloatedTopic> {
    let tree_json = match db.get_meta("topiary_tree") {
        Some(s) => s,
        None => return vec![],
    };
    let entry_ids_json = match db.get_meta("topiary_entry_ids") {
        Some(s) => s,
        None => return vec![],
    };

    let tree: serde_json::Value = match serde_json::from_str(&tree_json) {
        Ok(v) => v,
        Err(_) => return vec![],
    };
    let entry_ids: Vec<String> = match serde_json::from_str(&entry_ids_json) {
        Ok(v) => v,
        Err(_) => return vec![],
    };

    let mut bloated = Vec::new();
    if let Some(topics) = tree.get("topics").and_then(|t| t.as_array()) {
        collect_bloated_leaves(topics, &entry_ids, &mut bloated);
    }

    // Sort by member count descending (most bloated first)
    bloated.sort_by(|a, b| b.member_ids.len().cmp(&a.member_ids.len()));
    bloated
}

/// Recursively collect leaf topics that exceed the distillation threshold.
fn collect_bloated_leaves(
    nodes: &[serde_json::Value],
    entry_ids: &[String],
    out: &mut Vec<BloatedTopic>,
) {
    for node in nodes {
        let children = node.get("children").and_then(|c| c.as_array());
        if let Some(kids) = children {
            if !kids.is_empty() {
                collect_bloated_leaves(kids, entry_ids, out);
                continue;
            }
        }
        // Leaf node
        let members = match node.get("members").and_then(|m| m.as_array()) {
            Some(arr) => arr,
            None => continue,
        };
        if members.len() < thresholds::DISTILL_THRESHOLD {
            continue;
        }
        let name = node
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("unnamed")
            .to_string();
        let ids: Vec<String> = members
            .iter()
            .filter_map(|idx| {
                idx.as_u64()
                    .and_then(|i| entry_ids.get(i as usize))
                    .cloned()
            })
            .collect();
        if ids.len() >= thresholds::DISTILL_THRESHOLD {
            out.push(BloatedTopic {
                name,
                member_ids: ids,
            });
        }
    }
}

/// Distill a single topic: load members, LLM condense, replace.
async fn distill_one_topic(
    cfg: &AiConfig,
    db: &SharedDB,
    topic: &BloatedTopic,
) -> Result<DistillResult, EngramError> {
    // Load full memories for the topic members
    let db2 = db.clone();
    let ids = topic.member_ids.clone();
    let memories: Vec<Memory> = tokio::task::spawn_blocking(move || {
        ids.iter()
            .filter_map(|id| db2.get(id).ok().flatten())
            .collect()
    })
    .await
    .map_err(|e| EngramError::Internal(format!("spawn: {e}")))?;

    if memories.len() < thresholds::DISTILL_THRESHOLD {
        return Ok(DistillResult {
            topic_name: topic.name.clone(),
            input_count: memories.len(),
            ..Default::default()
        });
    }

    // Build the distillation prompt
    let mut user_prompt = format!(
        "Topic: \"{}\"\n\n{} memories to distill:\n\n",
        topic.name,
        memories.len()
    );
    for (i, m) in memories.iter().enumerate() {
        let layer_name = match m.layer {
            crate::db::Layer::Core => "Core",
            crate::db::Layer::Working => "Working",
            crate::db::Layer::Buffer => "Buffer",
        };
        user_prompt.push_str(&format!(
            "[{}] id={} layer={} kind={} imp={:.1}\n{}\n\n",
            i, &m.id[..8], layer_name, m.kind, m.importance, m.content
        ));
    }

    // Call LLM
    let tcr = ai::llm_tool_call::<DistillResponse>(
        cfg,
        "audit",
        prompts::DISTILL_TOPIC_SYSTEM,
        &user_prompt,
        "distill_topic",
        "Condense overlapping memories into fewer, richer entries. Preserve ALL specific details.",
        prompts::distill_topic_schema(),
    )
    .await?;

    if let Some(ref u) = tcr.usage {
        let cached = u
            .prompt_tokens_details
            .as_ref()
            .map_or(0, |d| d.cached_tokens);
        let _ = db.log_llm_call(
            "distill",
            &tcr.model,
            u.prompt_tokens,
            u.completion_tokens,
            cached,
            tcr.duration_ms,
        );
    }

    let response = tcr.value;
    if response.distilled.is_empty() {
        return Ok(DistillResult {
            topic_name: topic.name.clone(),
            input_count: memories.len(),
            ..Default::default()
        });
    }

    // Build a set of source IDs that got absorbed
    let mem_map: std::collections::HashMap<String, Memory> =
        memories.iter().map(|m| (m.id.clone(), m.clone())).collect();
    // Also map short IDs
    let short_map: std::collections::HashMap<String, String> =
        memories.iter().map(|m| (m.id[..8].to_string(), m.id.clone())).collect();

    let db3 = db.clone();
    let distilled = response.distilled;
    let input_count = memories.len();
    let topic_name = topic.name.clone();

    // Apply: insert distilled entries, tag sources as absorbed
    let result = tokio::task::spawn_blocking(move || {
        let mut absorbed = Vec::new();

        for entry in &distilled {
            // Resolve source_ids (could be short or full)
            let sources: Vec<String> = entry
                .source_ids
                .iter()
                .filter_map(|sid| {
                    if mem_map.contains_key(sid.as_str()) {
                        Some(sid.clone())
                    } else {
                        short_map.get(sid.as_str()).cloned()
                    }
                })
                .collect();

            if sources.len() < 2 || entry.content.is_empty() {
                continue;
            }

            // Determine the highest layer among sources
            let max_layer = sources
                .iter()
                .filter_map(|id| mem_map.get(id.as_str()))
                .map(|m| m.layer as u8)
                .max()
                .unwrap_or(2);

            let kind = entry.kind.as_deref().unwrap_or("semantic").to_string();

            // Insert the distilled memory
            let input = crate::db::MemoryInput {
                content: entry.content.clone(),
                tags: Some(vec!["distilled".into()]),
                source: Some("distill".into()),
                importance: Some(0.7),
                kind: Some(kind),
                layer: Some(max_layer),
                supersedes: Some(sources.clone()),
                ..Default::default()
            };

            match db3.insert(input) {
                Ok(_) => {
                    absorbed.extend(sources);
                }
                Err(e) => {
                    warn!(error = %e, "distill insert failed");
                }
            }
        }

        DistillResult {
            topic_name,
            input_count,
            output_count: distilled.len(),
            absorbed_ids: absorbed,
        }
    })
    .await
    .map_err(|e| EngramError::Internal(format!("spawn: {e}")))?;

    Ok(result)
}
