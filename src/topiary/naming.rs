//! LLM-powered topic naming, adapted for engram's AI infrastructure.

use std::collections::HashMap;
use tracing::{debug, info, warn};

use super::{Entry, TopicNode};
use crate::ai::{self, AiConfig};
use crate::db::MemoryDB;
use crate::prompts;
use crate::thresholds;

/// Stats from a naming run.
pub struct NamingStats {
    pub llm_calls: usize,
    pub dirty_count: usize,
    pub named: usize,
}

/// Name all dirty leaf topics in the tree via batched LLM calls.
pub async fn name_tree(
    roots: &mut [TopicNode],
    all_entries: &[Entry],
    cfg: &AiConfig,
    db: &MemoryDB,
) -> NamingStats {
    let batch_size = thresholds::TOPIARY_NAMING_BATCH_SIZE;

    let mut dirty_leaves: Vec<(String, usize, Vec<String>)> = Vec::new();
    for root in roots.iter() {
        collect_dirty_leaves(root, all_entries, &mut dirty_leaves);
    }

    let dirty_count = dirty_leaves.len();
    if dirty_count == 0 {
        debug!("topiary naming: no dirty topics");
        return NamingStats {
            llm_calls: 0,
            dirty_count: 0,
            named: 0,
        };
    }

    let num_batches = dirty_count.div_ceil(batch_size);
    info!(
        dirty = dirty_count,
        batches = num_batches,
        "topiary naming dirty topics"
    );

    let mut existing_names: Vec<String> = Vec::new();
    for root in roots.iter() {
        collect_existing_names(root, &mut existing_names);
    }

    let mut all_names: HashMap<String, String> = HashMap::new();
    let mut llm_calls = 0;

    for (batch_idx, chunk) in dirty_leaves.chunks(batch_size).enumerate() {
        let mut prompt = String::new();

        let context_names: Vec<&str> = existing_names
            .iter()
            .chain(all_names.values())
            .map(|s| s.as_str())
            .collect();
        if !context_names.is_empty() {
            prompt.push_str("Already assigned names (avoid duplicates with these):\n");
            for name in &context_names {
                prompt.push_str(&format!("- {name}\n"));
            }
            prompt.push('\n');
        }

        for (id, member_count, samples) in chunk {
            prompt.push_str(&format!("{} ({} entries):\n", id, member_count));
            for s in samples {
                let truncated: String = s.chars().take(thresholds::TOPIARY_NAMING_SAMPLE_CHARS).collect();
                prompt.push_str(&format!("- \"{truncated}\"\n"));
            }
            prompt.push('\n');
        }

        #[derive(serde::Deserialize)]
        struct SetTopicNamesArgs {
            topics: Vec<TopicNameEntry>,
        }
        #[derive(serde::Deserialize)]
        struct TopicNameEntry {
            id: String,
            name: String,
        }

        match ai::llm_tool_call::<SetTopicNamesArgs>(
            cfg,
            "naming",
            prompts::TOPIC_NAMING_SYSTEM,
            &prompt,
            "set_topic_names",
            "Set names for topic clusters",
            prompts::topic_naming_schema(),
        )
        .await
        {
            Ok(result) => {
                if let Some(ref u) = result.usage {
                    let cached = u
                        .prompt_tokens_details
                        .as_ref()
                        .map_or(0, |d| d.cached_tokens);
                    let _ = db.log_llm_call(
                        "naming",
                        &result.model,
                        u.prompt_tokens,
                        u.completion_tokens,
                        cached,
                        result.duration_ms,
                    );
                }
                let requested_ids: Vec<&str> = chunk.iter().map(|(id, _, _)| id.as_str()).collect();
                let mut map: HashMap<String, String> = HashMap::new();
                for t in result.value.topics {
                    // Try exact match first, then fuzzy match for LLM ID hallucinations
                    if requested_ids.contains(&t.id.as_str()) {
                        map.insert(t.id, t.name);
                    } else {
                        // Try to find the requested ID that the LLM's response refers to
                        // e.g. LLM returns "topic_kb1" or "Kb1" for requested "kb1"
                        let lower = t.id.to_lowercase();
                        if let Some(&rid) = requested_ids.iter().find(|&&r| lower.contains(r)) {
                            if !map.contains_key(rid) {
                                debug!(
                                    returned_id = %t.id,
                                    matched_to = rid,
                                    "topiary naming: fuzzy-matched LLM topic ID"
                                );
                                map.insert(rid.to_string(), t.name);
                            }
                        } else {
                            debug!(
                                returned_id = %t.id,
                                "topiary naming: unrecognized topic ID from LLM"
                            );
                        }
                    }
                }
                let matched = map.keys().filter(|k| requested_ids.contains(&k.as_str())).count();
                if matched < requested_ids.len() {
                    let missing: Vec<&&str> = requested_ids.iter().filter(|id| !map.contains_key(**id)).collect();
                    warn!(
                        requested = requested_ids.len(),
                        returned = map.len(),
                        matched,
                        ?missing,
                        "topiary naming: LLM returned fewer topics than requested"
                    );
                }
                all_names.extend(map);
            }
            Err(e) => {
                warn!(
                    batch = batch_idx + 1,
                    total_batches = num_batches,
                    error = %e,
                    "topiary naming batch failed"
                );
            }
        }
        llm_calls += 1;
    }

    let mut applied = 0;
    for root in roots.iter_mut() {
        applied += apply_names_to_leaves(root, &all_names);
    }
    info!(named = applied, dirty = dirty_count, "topiary naming complete");

    NamingStats {
        llm_calls,
        dirty_count,
        named: applied,
    }
}

/// Collect dirty leaf info: (id, member_count, sample_texts).
fn collect_dirty_leaves(
    node: &TopicNode,
    all_entries: &[Entry],
    out: &mut Vec<(String, usize, Vec<String>)>,
) {
    if node.is_leaf() {
        if node.dirty && !node.members.is_empty() {
            let limit = node.members.len().min(thresholds::TOPIARY_NAMING_MAX_SAMPLES);
            let samples: Vec<String> = node
                .members
                .iter()
                .take(limit)
                .filter_map(|&i| all_entries.get(i).map(|e| e.text.clone()))
                .collect();
            out.push((node.id.clone(), node.members.len(), samples));
        }
    } else {
        for child in &node.children {
            collect_dirty_leaves(child, all_entries, out);
        }
    }
}

/// Collect names of non-dirty leaves (already named) for dedup context.
fn collect_existing_names(node: &TopicNode, out: &mut Vec<String>) {
    if node.is_leaf() {
        if !node.dirty {
            if let Some(ref name) = node.name {
                out.push(name.clone());
            }
        }
    } else {
        for child in &node.children {
            collect_existing_names(child, out);
        }
    }
}

/// Apply names from the LLM response to matching dirty leaves, and clear dirty flag.
fn apply_names_to_leaves(node: &mut TopicNode, names: &HashMap<String, String>) -> usize {
    if node.is_leaf() {
        if node.dirty {
            if let Some(name) = names.get(&node.id) {
                let clean: String = name.chars().take(50).collect();
                node.name = Some(clean);
                node.named_at_size = node.members.len();
                node.dirty = false;
                return 1;
            }
        }
        0
    } else {
        let mut count = 0;
        for child in &mut node.children {
            count += apply_names_to_leaves(child, names);
        }
        count
    }
}
