//! Debounced async background worker for topiary tree rebuilds.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use super::{Entry, TopicNode, TopicTree};
use crate::ai::AiConfig;
use crate::db::{Layer, MemoryDB};
use crate::thresholds;

/// Spawn the topiary background worker.
///
/// Listens on `trigger_rx` for rebuild signals. Debounces: after receiving
/// a signal, waits 5 seconds of quiet before rebuilding.
pub fn spawn_worker(
    db: Arc<MemoryDB>,
    ai: Option<AiConfig>,
    mut trigger_rx: mpsc::UnboundedReceiver<()>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let debounce_ms = thresholds::TOPIARY_DEBOUNCE_MS;

        // On startup, check if a cached tree exists. If not, trigger immediate rebuild.
        {
            let db2 = db.clone();
            let has_tree = tokio::task::spawn_blocking(move || db2.get_meta("topiary_tree"))
                .await
                .ok()
                .flatten()
                .is_some();
            if !has_tree {
                info!("topiary: no cached tree, triggering initial build");
                // Fall through to the rebuild logic
                do_rebuild(&db, ai.as_ref()).await;
            }
        }

        loop {
            // Phase 1: wait for a trigger
            match trigger_rx.recv().await {
                Some(()) => {}
                None => {
                    debug!("topiary worker: channel closed, exiting");
                    break;
                }
            }

            // Phase 2: debounce — drain any signals in next 5 seconds
            let deadline = tokio::time::Instant::now()
                + std::time::Duration::from_millis(debounce_ms);
            loop {
                let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                if remaining.is_zero() {
                    break;
                }
                match tokio::time::timeout(remaining, trigger_rx.recv()).await {
                    Ok(Some(())) => {
                        // Got another signal — reset is implicit (we keep looping)
                        continue;
                    }
                    Ok(None) => {
                        debug!("topiary worker: channel closed during debounce");
                        return;
                    }
                    Err(_) => break, // timeout — debounce window expired
                }
            }

            // Phase 3: rebuild
            do_rebuild(&db, ai.as_ref()).await;
        }
    })
}

async fn do_rebuild(db: &Arc<MemoryDB>, ai: Option<&AiConfig>) {
    let start = std::time::Instant::now();

    // Step 1: Load all Working + Buffer memories with embeddings
    let db2 = db.clone();
    let entries_result = tokio::task::spawn_blocking(move || {
        load_entries(&db2)
    })
    .await;

    let entries = match entries_result {
        Ok(e) => e,
        Err(e) => {
            warn!(error = %e, "topiary rebuild: failed to load entries");
            return;
        }
    };

    if entries.is_empty() {
        debug!("topiary rebuild: no entries with embeddings, skipping");
        return;
    }

    let entry_count = entries.len();

    // Step 1.5: Check if entry set matches cache — skip full rebuild if unchanged
    // Fingerprint includes IDs + content lengths to detect PATCH/merge changes
    let current_ids: HashSet<String> = entries.iter().map(|e| e.id.clone()).collect();
    let current_fingerprint: u64 = entries.iter().map(|e| e.text.len() as u64).sum();
    let db_cache = db.clone();
    let (cached_tree_full, cached_ids_json, cached_tree_json, cached_fingerprint) =
        tokio::task::spawn_blocking(move || {
            (
                db_cache.get_meta("topiary_tree_full"),
                db_cache.get_meta("topiary_entry_ids"),
                db_cache.get_meta("topiary_tree"),
                db_cache.get_meta("topiary_fingerprint"),
            )
        })
        .await
        .unwrap_or((None, None, None, None));

    // If entry set AND content fingerprint match cache, skip rebuild
    if let (Some(ref ids_json), Some(ref tree_json)) = (&cached_ids_json, &cached_tree_json) {
        if let Ok(cached_ids) = serde_json::from_str::<Vec<String>>(ids_json) {
            let cached_set: HashSet<&str> = cached_ids.iter().map(|s| s.as_str()).collect();
            let current_set: HashSet<&str> = current_ids.iter().map(|s| s.as_str()).collect();
            let fp_matches = cached_fingerprint
                .as_ref()
                .and_then(|s| s.parse::<u64>().ok()) == Some(current_fingerprint);
            if cached_set == current_set && fp_matches {
                // Verify the cached tree is valid (has content)
                if tree_json.len() > 2 {
                    info!(
                        entries = entry_count,
                        elapsed_ms = start.elapsed().as_millis() as u64,
                        "topiary: entry set unchanged, using cached tree"
                    );
                    return;
                }
            }
        }
    }

    // Step 1.6: Incremental insert path — if only a few entries were added
    // (and none removed), deserialize the cached tree and insert new entries
    // instead of rebuilding from scratch. This avoids k-means reshuffling
    // that marks stable topics dirty and wastes LLM naming tokens.
    let mut tree = TopicTree::new(
        thresholds::TOPIARY_ASSIGN_THRESHOLD,
        thresholds::TOPIARY_MERGE_THRESHOLD,
    );
    let mut used_incremental = false;

    let try_incremental = cached_tree_full.is_some() && cached_ids_json.is_some();
    if try_incremental {
        let cached_ids_parsed: Option<Vec<String>> = cached_ids_json
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok());

        if let Some(ref cached_ids) = cached_ids_parsed {
            let cached_set: HashSet<&str> = cached_ids.iter().map(|s| s.as_str()).collect();
            let current_set: HashSet<&str> = current_ids.iter().map(|s| s.as_str()).collect();

            // Find added entries (in current but not cached)
            let added_indices: Vec<usize> = entries
                .iter()
                .enumerate()
                .filter(|(_, e)| !cached_set.contains(e.id.as_str()))
                .map(|(i, _)| i)
                .collect();
            // Find removed entries (in cached but not current)
            let removed_count = cached_set
                .iter()
                .filter(|id| !current_set.contains(*id))
                .count();

            // Allow incremental even with removals — consolidation may merge/drop
            // entries, which previously forced a full rebuild and wasted naming tokens.
            // Removed entries are handled by filtering unmappable members below.
            let total_changes = added_indices.len() + removed_count;
            let can_incremental = total_changes > 0
                && total_changes <= thresholds::TOPIARY_INCREMENTAL_MAX;

            if can_incremental {
                if let Some(ref full_json) = cached_tree_full {
                    if let Ok(mut cached_tree) = serde_json::from_str::<TopicTree>(full_json) {
                        // Remap member indices: cached tree uses indices into old entry list,
                        // but current entry list may have different ordering.
                        let old_id_to_idx: HashMap<&str, usize> = cached_ids
                            .iter()
                            .enumerate()
                            .map(|(i, id)| (id.as_str(), i))
                            .collect();
                        let new_id_to_idx: HashMap<&str, usize> = entries
                            .iter()
                            .enumerate()
                            .map(|(i, e)| (e.id.as_str(), i))
                            .collect();

                        // Build old_idx → new_idx mapping
                        let idx_map: HashMap<usize, usize> = old_id_to_idx
                            .iter()
                            .filter_map(|(id, &old_i)| {
                                new_id_to_idx.get(id).map(|&new_i| (old_i, new_i))
                            })
                            .collect();

                        // Remap all member indices in the tree
                        remap_members(&mut cached_tree.roots, &idx_map);

                        // Prune leaves that became empty after remap (deleted memories)
                        let pruned = prune_empty_leaves(&mut cached_tree.roots);
                        if pruned > 0 {
                            info!(pruned, "topiary incremental: pruned empty leaves after remap");
                        }

                        // Clear stale dirty flags from cache — previous cycles may
                        // have skipped LLM naming (small-cluster optimization) and
                        // stored topics as dirty. In the incremental path only the
                        // topic receiving a new entry should be dirty.
                        clear_dirty_flags(&mut cached_tree.roots);

                        let dirty_after_clear = count_dirty_leaves(&cached_tree.roots);
                        if dirty_after_clear > 0 {
                            warn!(
                                dirty_after_clear,
                                "topiary incremental: dirty leaves remain after clear_dirty_flags (BUG)"
                            );
                        }

                        // Insert new entries (insert already assigns to best topic
                        // or creates a new one; skip full consolidate which would
                        // run split/merge/absorb/hierarchy passes and mark many
                        // stable topics dirty unnecessarily).
                        for &new_idx in &added_indices {
                            cached_tree.insert(new_idx, &entries[new_idx].embedding);
                        }

                        let dirty_after_insert = count_dirty_leaves(&cached_tree.roots);
                        debug!(
                            added = added_indices.len(),
                            dirty_after_insert,
                            "topiary incremental: dirty count after insert"
                        );

                        let topic_count: usize =
                            cached_tree.roots.iter().map(|r| r.leaf_count()).sum();
                        info!(
                            entries = entry_count,
                            added = added_indices.len(),
                            topics = topic_count,
                            elapsed_ms = start.elapsed().as_millis() as u64,
                            "topiary incremental insert"
                        );

                        tree = cached_tree;
                        used_incremental = true;
                    }
                }
            }
        }
    }

    if !used_incremental {
        // Step 2: Build topic tree (full rebuild)
        tree = TopicTree::new(
            thresholds::TOPIARY_ASSIGN_THRESHOLD,
            thresholds::TOPIARY_MERGE_THRESHOLD,
        );
        for (i, entry) in entries.iter().enumerate() {
            tree.insert(i, &entry.embedding);
        }

        // Step 3: Consolidate
        tree.consolidate(&entries);

        let topic_count: usize = tree.roots.iter().map(|r| r.leaf_count()).sum();
        info!(
            entries = entry_count,
            topics = topic_count,
            elapsed_ms = start.elapsed().as_millis() as u64,
            "topiary tree built (full rebuild)"
        );

        // Step 3.5: Inherit names from previously cached tree
        let inherited = inherit_names(&mut tree.roots, &entries, cached_tree_full, cached_ids_json);
        if inherited > 0 {
            info!(inherited, "topiary inherited names from cache");
        }
    }

    // Step 4: Name dirty topics (if AI available)
    // Optimization: count dirty leaves first. If all are tiny (≤2 members),
    // use fallback names instead of burning LLM tokens on small clusters
    // that will likely grow and get renamed soon anyway.
    if let Some(cfg) = ai {
        if cfg.has_llm() {
            let dirty_count = count_dirty_leaves(&tree.roots);
            let dirty_total_members = sum_dirty_members(&tree.roots);
            let skip_llm = dirty_count > 0
                && dirty_count <= 3
                && dirty_total_members <= dirty_count * 2;
            if skip_llm {
                debug!(
                    dirty_count,
                    dirty_total_members,
                    "topiary: skipping LLM naming for small dirty clusters, using fallback"
                );
            } else {
                let stats = super::naming::name_tree(&mut tree.roots, &entries, cfg, db).await;
                if stats.named > 0 {
                    info!(
                        named = stats.named,
                        llm_calls = stats.llm_calls,
                        "topiary naming complete"
                    );
                }
            }
        }
    }

    // Step 4.5: Assign fallback names to any still-unnamed topics.
    // Previously we discarded the entire tree — but that wasted
    // already-inherited and LLM-named topics.  A fallback is better
    // than losing the whole rebuild.
    {
        let unnamed = count_unnamed_leaves(&tree.roots);
        if unnamed > 0 {
            let fallback_count = assign_fallback_names(&mut tree.roots, &entries);
            info!(
                unnamed,
                fallback_count,
                "topiary: assigned fallback names to unnamed topics"
            );
        }
    }

    // Step 5: Serialize and store
    let topic_count: usize = tree.roots.iter().map(|r| r.leaf_count()).sum();
    let store_start = std::time::Instant::now();
    let tree_json = tree.to_json(&entries);
    let json_str = match serde_json::to_string(&tree_json) {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, "topiary rebuild: failed to serialize tree");
            return;
        }
    };

    // Also serialize the full tree (with centroids) for potential future incremental updates
    let full_tree = match serde_json::to_string(&tree) {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, "topiary rebuild: failed to serialize full tree");
            return;
        }
    };

    // Store the entry ID mapping so /topic can resolve entries
    let entry_ids: Vec<&str> = entries.iter().map(|e| e.id.as_str()).collect();
    let entry_ids_json = match serde_json::to_string(&entry_ids) {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, "topiary rebuild: failed to serialize entry IDs");
            return;
        }
    };

    let db2 = db.clone();
    let fp_str = current_fingerprint.to_string();
    let store_result = tokio::task::spawn_blocking(move || {
        db2.set_meta("topiary_tree", &json_str)?;
        db2.set_meta("topiary_tree_full", &full_tree)?;
        db2.set_meta("topiary_entry_ids", &entry_ids_json)?;
        db2.set_meta("topiary_fingerprint", &fp_str)?;
        Ok::<_, crate::error::EngramError>(())
    })
    .await;

    match store_result {
        Ok(Ok(())) => {
            info!(
                entries = entry_count,
                topics = topic_count,
                elapsed_ms = store_start.elapsed().as_millis() as u64,
                total_ms = start.elapsed().as_millis() as u64,
                "topiary tree stored"
            );
        }
        Ok(Err(e)) => warn!(error = %e, "topiary rebuild: failed to store tree"),
        Err(e) => warn!(error = %e, "topiary rebuild: store task panicked"),
    }
}

/// Load Working + Buffer memories that have embeddings, convert to Entry.
fn load_entries(db: &MemoryDB) -> Vec<Entry> {
    let mut entries = Vec::new();

    // Load all layers
    let core = db
        .list_by_layer_meta(Layer::Core, 10_000, 0)
        .unwrap_or_default();
    let working = db
        .list_by_layer_meta(Layer::Working, 10_000, 0)
        .unwrap_or_default();
    let buffer = db
        .list_by_layer_meta(Layer::Buffer, 10_000, 0)
        .unwrap_or_default();

    // Collect all IDs
    let all_ids: Vec<String> = core
        .iter()
        .chain(working.iter())
        .chain(buffer.iter())
        .map(|m| m.id.clone())
        .collect();

    // Batch-fetch embeddings
    let embeddings = db.get_embeddings_by_ids(&all_ids);
    let emb_map: std::collections::HashMap<&str, &Vec<f32>> = embeddings
        .iter()
        .map(|(id, emb)| (id.as_str(), emb))
        .collect();

    for mem in core.iter().chain(working.iter()).chain(buffer.iter()) {
        if let Some(emb) = emb_map.get(mem.id.as_str()) {
            entries.push(Entry {
                id: mem.id.clone(),
                text: mem.content.clone(),
                embedding: (*emb).clone(),
                tags: mem.tags.clone(),
            });
        }
    }

    entries
}

// ── Name inheritance helpers ──────────────────────────────────────────────

/// Inherit topic names from the previously cached tree for leaves whose
/// member composition hasn't changed significantly (Jaccard >= threshold).
pub fn inherit_names(
    new_roots: &mut [TopicNode],
    new_entries: &[Entry],
    old_tree_json: Option<String>,
    old_entry_ids_json: Option<String>,
) -> usize {
    let old_tree_str = match old_tree_json {
        Some(s) => s,
        None => return 0,
    };
    let old_ids_str = match old_entry_ids_json {
        Some(s) => s,
        None => return 0,
    };
    let old_tree: TopicTree = match serde_json::from_str(&old_tree_str) {
        Ok(t) => t,
        Err(_) => return 0,
    };
    let old_entry_ids: Vec<String> = match serde_json::from_str(&old_ids_str) {
        Ok(ids) => ids,
        Err(_) => return 0,
    };

    // Collect old named leaves: (name, set of member entry IDs)
    let mut old_named: Vec<(String, HashSet<String>)> = Vec::new();
    for root in &old_tree.roots {
        collect_named_leaves(root, &old_entry_ids, &mut old_named);
    }

    if old_named.is_empty() {
        return 0;
    }

    // Collect all dirty leaves with their best match scores
    let mut candidates: Vec<(String, String, f32)> = Vec::new(); // (node_id, old_name, jaccard)
    for root in new_roots.iter() {
        collect_inherit_candidates(root, new_entries, &old_named, &mut candidates);
    }

    // Sort by score descending — best matches first
    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Greedy assignment: each old name can only be used once
    let mut used_names: HashSet<String> = HashSet::new();
    let mut assignments: HashMap<String, String> = HashMap::new();
    for (node_id, old_name, score) in &candidates {
        if *score < thresholds::TOPIARY_INHERIT_THRESHOLD {
            break; // sorted descending, no more good matches
        }
        if used_names.contains(old_name) {
            continue;
        }
        used_names.insert(old_name.clone());
        assignments.insert(node_id.clone(), old_name.clone());
    }

    // Apply assignments
    let mut inherited = 0;
    for root in new_roots.iter_mut() {
        inherited += apply_inherited_names(root, &assignments);
    }
    inherited
}

fn collect_named_leaves(
    node: &TopicNode,
    entry_ids: &[String],
    out: &mut Vec<(String, HashSet<String>)>,
) {
    if node.is_leaf() {
        if let Some(ref name) = node.name {
            let ids: HashSet<String> = node
                .members
                .iter()
                .filter_map(|&i| entry_ids.get(i).cloned())
                .collect();
            if !ids.is_empty() {
                out.push((name.clone(), ids));
            }
        }
    } else {
        for child in &node.children {
            collect_named_leaves(child, entry_ids, out);
        }
    }
}

/// Collect (node_id, best_old_name, jaccard) for all dirty unnamed leaves.
fn collect_inherit_candidates(
    node: &TopicNode,
    new_entries: &[Entry],
    old_named: &[(String, HashSet<String>)],
    out: &mut Vec<(String, String, f32)>,
) {
    if node.is_leaf() {
        if !node.dirty || node.name.is_some() {
            return;
        }
        let new_ids: HashSet<String> = node
            .members
            .iter()
            .filter_map(|&i| new_entries.get(i).map(|e| e.id.clone()))
            .collect();
        if new_ids.is_empty() {
            return;
        }

        let mut best_score = 0.0f32;
        let mut best_name: Option<&str> = None;
        for (name, old_ids) in old_named {
            let intersection = new_ids.iter().filter(|id| old_ids.contains(*id)).count();
            let union = new_ids.len() + old_ids.len() - intersection;
            let jaccard = if union > 0 {
                intersection as f32 / union as f32
            } else {
                0.0
            };
            if jaccard > best_score {
                best_score = jaccard;
                best_name = Some(name.as_str());
            }
        }

        if let Some(name) = best_name {
            out.push((node.id.clone(), name.to_string(), best_score));
        }
    } else {
        for child in &node.children {
            collect_inherit_candidates(child, new_entries, old_named, out);
        }
    }
}

/// Apply inherited names to matching dirty leaves.
fn apply_inherited_names(
    node: &mut TopicNode,
    assignments: &HashMap<String, String>,
) -> usize {
    if node.is_leaf() {
        if node.dirty {
            if let Some(name) = assignments.get(&node.id) {
                node.name = Some(name.clone());
                node.named_at_size = node.members.len();
                node.dirty = false;
                return 1;
            }
        }
        0
    } else {
        let mut count = 0;
        for child in &mut node.children {
            count += apply_inherited_names(child, assignments);
        }
        count
    }
}

/// Assign fallback names to unnamed leaves. Uses the most common tag
/// among a topic's members, or falls back to "Topic <id>".
pub fn assign_fallback_names(roots: &mut [TopicNode], entries: &[Entry]) -> usize {
    let mut count = 0;
    for root in roots.iter_mut() {
        count += assign_fallback_in(root, entries);
    }
    count
}

fn assign_fallback_in(node: &mut TopicNode, entries: &[Entry]) -> usize {
    if node.is_leaf() {
        if node.name.is_some() {
            return 0;
        }
        // Pick the most frequent tag across this topic's members
        let mut tag_counts: HashMap<&str, usize> = HashMap::new();
        for &idx in &node.members {
            if let Some(entry) = entries.get(idx) {
                for tag in &entry.tags {
                    *tag_counts.entry(tag.as_str()).or_insert(0) += 1;
                }
            }
        }
        let name = tag_counts
            .into_iter()
            .max_by_key(|&(_, c)| c)
            .map(|(t, _)| {
                // Title case: "deploy-flow" → "Deploy Flow"
                t.split(&['-', '_'][..])
                    .map(|w| {
                        let mut c = w.chars();
                        match c.next() {
                            Some(f) => f.to_uppercase().to_string() + c.as_str(),
                            None => String::new(),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_else(|| format!("Topic {}", node.id));
        node.name = Some(name);
        node.dirty = false;
        1
    } else {
        let mut count = 0;
        for child in &mut node.children {
            count += assign_fallback_in(child, entries);
        }
        count
    }
}

pub fn count_unnamed_leaves(roots: &[TopicNode]) -> usize {
    roots.iter().map(count_unnamed_in).sum()
}

fn count_unnamed_in(node: &TopicNode) -> usize {
    if node.is_leaf() {
        if node.name.is_none() {
            1
        } else {
            0
        }
    } else {
        node.children.iter().map(count_unnamed_in).sum()
    }
}

/// Count dirty leaves in the tree.
fn count_dirty_leaves(roots: &[TopicNode]) -> usize {
    fn count(node: &TopicNode) -> usize {
        if node.is_leaf() {
            if node.dirty { 1 } else { 0 }
        } else {
            node.children.iter().map(|c| count(c)).sum()
        }
    }
    roots.iter().map(|r| count(r)).sum()
}

/// Sum of member counts across all dirty leaves.
fn sum_dirty_members(roots: &[TopicNode]) -> usize {
    fn sum(node: &TopicNode) -> usize {
        if node.is_leaf() {
            if node.dirty { node.members.len() } else { 0 }
        } else {
            node.children.iter().map(|c| sum(c)).sum()
        }
    }
    roots.iter().map(|r| sum(r)).sum()
}

/// Remap member indices in all tree nodes from old entry list positions to new ones.
/// Entries that no longer exist (missing from idx_map) are dropped.
fn remap_members(roots: &mut [TopicNode], idx_map: &HashMap<usize, usize>) {
    for node in roots.iter_mut() {
        if node.is_leaf() {
            node.members = node
                .members
                .iter()
                .filter_map(|old_idx| idx_map.get(old_idx).copied())
                .collect();
        } else {
            remap_members(&mut node.children, idx_map);
            // Update parent member list (union of children)
            node.members = node
                .children
                .iter()
                .flat_map(|c| c.members.iter().copied())
                .collect();
        }
    }
}

/// Remove empty leaves (0 members) left behind after remap_members drops
/// entries that were deleted. Also removes internal nodes whose children were
/// all pruned. Returns the number of nodes removed.
fn prune_empty_leaves(roots: &mut Vec<TopicNode>) -> usize {
    let mut pruned = 0;
    // Process children first (bottom-up)
    for node in roots.iter_mut() {
        if !node.is_leaf() {
            pruned += prune_empty_leaves(&mut node.children);
            // Recalculate parent members after pruning children
            node.members = node
                .children
                .iter()
                .flat_map(|c| c.members.iter().copied())
                .collect();
        }
    }
    let before = roots.len();
    roots.retain(|node| {
        // Keep internal nodes that still have children, keep leaves with members
        if node.is_leaf() {
            !node.members.is_empty()
        } else {
            !node.children.is_empty()
        }
    });
    pruned += before - roots.len();
    pruned
}

/// Clear all dirty flags in the tree. Used after deserializing a cached tree
/// for incremental insert — stale dirty flags from previous cycles (e.g. when
/// LLM naming was skipped for small clusters) should not carry over.
fn clear_dirty_flags(roots: &mut [TopicNode]) {
    for node in roots.iter_mut() {
        node.dirty = false;
        if !node.is_leaf() {
            clear_dirty_flags(&mut node.children);
        }
    }
}
