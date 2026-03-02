//! LLM-powered topic naming, adapted for engram's AI infrastructure.

use std::collections::{HashMap, HashSet};
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
        debug!("topiary naming: no dirty topics, refreshing internal node names");
        // Still refresh internal node names (no LLM needed) — they may be
        // stale after leaf renames in a previous cycle.
        for root in roots.iter_mut() {
            name_internal_nodes(root);
        }
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

    // Post-naming dedup: detect duplicate names across all leaves
    {
        let mut all_leaf_names: Vec<(String, String, usize)> = Vec::new();
        for root in roots.iter() {
            collect_all_leaf_names(root, &mut all_leaf_names);
        }

        // Group by name (case-insensitive)
        let mut by_name: HashMap<String, Vec<(String, usize)>> = HashMap::new();
        for (id, name, size) in &all_leaf_names {
            by_name
                .entry(name.to_lowercase())
                .or_default()
                .push((id.clone(), *size));
        }

        // For each duplicate group, keep the largest, mark others dirty
        let mut dupes_to_dirty: HashSet<String> = HashSet::new();
        for (_name, mut entries) in by_name {
            if entries.len() > 1 {
                // Sort by size descending, keep the largest
                entries.sort_by(|a, b| b.1.cmp(&a.1));
                for (id, _) in entries.iter().skip(1) {
                    dupes_to_dirty.insert(id.clone());
                }
            }
        }

        if !dupes_to_dirty.is_empty() {
            let mut dirtied = 0;
            for root in roots.iter_mut() {
                dirtied += dedup_leaf_names(root, &dupes_to_dirty);
            }
            info!(dirtied, "topiary naming: marked duplicate-named topics as dirty for re-naming");
        }
    }

    // Name internal nodes bottom-up from child names (no LLM needed)
    for root in roots.iter_mut() {
        name_internal_nodes(root);
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

/// Collect all leaf names and their IDs for dedup checking.
fn collect_all_leaf_names(node: &TopicNode, out: &mut Vec<(String, String, usize)>) {
    if node.is_leaf() {
        if let Some(ref name) = node.name {
            out.push((node.id.clone(), name.clone(), node.members.len()));
        }
    } else {
        for child in &node.children {
            collect_all_leaf_names(child, out);
        }
    }
}

/// Mark duplicate-named leaves as dirty (keeping the largest one).
fn dedup_leaf_names(node: &mut TopicNode, dupes_to_dirty: &HashSet<String>) -> usize {
    if node.is_leaf() {
        if dupes_to_dirty.contains(&node.id) {
            debug!(
                topic_id = %node.id,
                name = node.name.as_deref().unwrap_or("?"),
                "topiary naming: marking duplicate-named topic as dirty"
            );
            node.dirty = true;
            return 1;
        }
        0
    } else {
        let mut count = 0;
        for child in &mut node.children {
            count += dedup_leaf_names(child, dupes_to_dirty);
        }
        count
    }
}

/// Name internal (non-leaf) nodes bottom-up by summarizing child names.
/// Uses the most common meaningful words from children's names.
/// Recursively collect leaf topic names with their member counts.
fn collect_leaf_names(node: &TopicNode, out: &mut Vec<(usize, String)>) {
    if node.is_leaf() {
        if let Some(name) = node.name.as_deref().filter(|n| *n != "unnamed") {
            out.push((node.total_members(), name.to_string()));
        }
        return;
    }
    for child in node.children.iter() {
        collect_leaf_names(child, out);
    }
}

pub fn name_internal_nodes(node: &mut TopicNode) {
    name_internal_nodes_inner(node);
}

/// Returns the set of names used by this node and all descendants.
fn name_internal_nodes_inner(node: &mut TopicNode) -> std::collections::HashSet<String> {
    if node.is_leaf() {
        // Leaf names are NOT added to the used set — only internal node
        // combined names need dedup.  Adding leaf names would cause every
        // candidate to be filtered out (since internal nodes pick from
        // descendant leaf names).
        return std::collections::HashSet::new();
    }

    // Recurse into children first (bottom-up), collecting internal-node names
    let mut descendant_used_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    for child in node.children.iter_mut() {
        let child_names = name_internal_nodes_inner(child);
        descendant_used_names.extend(child_names);
    }

    // Only internal node names — leaf names are excluded
    let child_used_names = &descendant_used_names;

    // Collect leaf names from all children (not child internal node names,
    // which would cause parent-child name duplication).
    let mut leaf_names: Vec<(usize, String)> = Vec::new();
    for child in node.children.iter() {
        collect_leaf_names(child, &mut leaf_names);
    }
    // Sort by member count descending, dedup by name, skip names already used by children
    leaf_names.sort_by(|a, b| b.0.cmp(&a.0));
    let mut seen = std::collections::HashSet::new();
    leaf_names.retain(|(_, name)| {
        seen.insert(name.clone()) && !child_used_names.contains(name)
    });

    if leaf_names.is_empty() {
        return descendant_used_names;
    }

    let primary = &leaf_names[0].1;

    if leaf_names.len() <= 1 {
        node.name = Some(primary.clone());
    } else {
        // Pick secondary that won't duplicate the primary
        let secondary = &leaf_names[1].1;
        let extra = if leaf_names.len() > 2 {
            format!(" +{}", leaf_names.len() - 2)
        } else {
            String::new()
        };
        let combined = format!("{}, {}{}", primary, secondary, extra);

        // If the combined name matches any direct child's name, try to
        // differentiate by using alternative leaf names (skip to 3rd, 4th, etc.)
        let combined_lower = combined.to_lowercase();
        let child_collision = node.children.iter().any(|c| {
            c.name.as_ref().is_some_and(|n| n.to_lowercase() == combined_lower)
        });
        let combined = if child_collision && leaf_names.len() > 2 {
            // Try picking different leaf names that aren't in the primary/secondary
            let alt_names: Vec<&String> = leaf_names[2..]
                .iter()
                .map(|(_, n)| n)
                .collect();
            if !alt_names.is_empty() {
                let alt_extra = if alt_names.len() > 1 {
                    format!(" +{}", alt_names.len() - 1)
                } else {
                    String::new()
                };
                format!("{}, {}{}", primary, alt_names[0], alt_extra)
            } else {
                combined
            }
        } else {
            combined
        };

        // Truncate to 60 chars at word boundary
        let truncated = if combined.len() <= 60 {
            combined
        } else {
            let mut end = 60;
            while end > 0 && !combined.is_char_boundary(end) {
                end -= 1;
            }
            if let Some(last_space) = combined[..end].rfind(' ') {
                if last_space > 20 {
                    combined[..last_space].to_string()
                } else {
                    combined.chars().take(60).collect()
                }
            } else {
                combined.chars().take(60).collect()
            }
        };
        // Strip trailing punctuation, prepositions, and conjunctions left by truncation
        let trimmed = truncated
            .trim_end_matches(|c: char| c == ',' || c.is_whitespace())
            .to_string();
        let trimmed = strip_trailing_stopwords(&trimmed);
        node.name = Some(trimmed);
    }

    // Add our own name to the used set before returning
    if let Some(ref name) = node.name {
        descendant_used_names.insert(name.clone());
    }
    descendant_used_names
}

/// Strip trailing stopwords (prepositions, conjunctions, articles) that look
/// awkward when truncation cuts mid-phrase.
fn strip_trailing_stopwords(s: &str) -> String {
    const STOPWORDS: &[&str] = &[
        " and", " or", " for", " the", " of", " in", " on", " to",
        " with", " by", " a", " an", " &",
    ];
    let mut result = s.to_string();
    loop {
        let mut changed = false;
        for sw in STOPWORDS {
            if let Some(prefix) = result.strip_suffix(sw) {
                if !prefix.is_empty() {
                    result = prefix.to_string();
                    changed = true;
                    break;
                }
            }
        }
        if !changed {
            break;
        }
    }
    // Final trim of punctuation/whitespace
    result
        .trim_end_matches(|c: char| c == ',' || c.is_whitespace())
        .to_string()
}
