//! Debounced async background worker for topiary tree rebuilds.

use std::collections::HashSet;
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
    let current_ids: HashSet<String> = entries.iter().map(|e| e.id.clone()).collect();
    let db_cache = db.clone();
    let (cached_tree_full, cached_ids_json, cached_tree_json) =
        tokio::task::spawn_blocking(move || {
            (
                db_cache.get_meta("topiary_tree_full"),
                db_cache.get_meta("topiary_entry_ids"),
                db_cache.get_meta("topiary_tree"),
            )
        })
        .await
        .unwrap_or((None, None, None));

    // If entry set is identical to cache, restore cached tree and skip rebuild
    if let (Some(ref ids_json), Some(ref tree_json)) = (&cached_ids_json, &cached_tree_json) {
        if let Ok(cached_ids) = serde_json::from_str::<Vec<String>>(ids_json) {
            let cached_set: HashSet<&str> = cached_ids.iter().map(|s| s.as_str()).collect();
            let current_set: HashSet<&str> = current_ids.iter().map(|s| s.as_str()).collect();
            if cached_set == current_set {
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

    // Step 2: Build topic tree (entries changed — full rebuild needed)
    let mut tree = TopicTree::new(
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
        "topiary tree built"
    );

    // Step 3.5: Inherit names from previously cached tree
    let inherited = inherit_names(&mut tree.roots, &entries, cached_tree_full, cached_ids_json);
    if inherited > 0 {
        info!(inherited, "topiary inherited names from cache");
    }

    // Step 4: Name dirty topics (if AI available)
    if let Some(cfg) = ai {
        if cfg.has_llm() {
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

    // Step 4.5: Don't store tree with unnamed topics — keep cached version
    {
        let unnamed = count_unnamed_leaves(&tree.roots);
        if unnamed > 0 {
            warn!(
                unnamed,
                "topiary: {} unnamed topics, keeping cached tree", unnamed
            );
            return;
        }
    }

    // Step 5: Serialize and store
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
    let store_result = tokio::task::spawn_blocking(move || {
        db2.set_meta("topiary_tree", &json_str)?;
        db2.set_meta("topiary_tree_full", &full_tree)?;
        db2.set_meta("topiary_entry_ids", &entry_ids_json)?;
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
/// member composition hasn't changed significantly (Jaccard >= 0.5).
fn inherit_names(
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

    let mut inherited = 0;
    for root in new_roots.iter_mut() {
        inherited += inherit_in_node(root, new_entries, &old_named);
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

fn inherit_in_node(
    node: &mut TopicNode,
    new_entries: &[Entry],
    old_named: &[(String, HashSet<String>)],
) -> usize {
    if node.is_leaf() {
        if !node.dirty || node.name.is_some() {
            return 0;
        }
        let new_ids: HashSet<String> = node
            .members
            .iter()
            .filter_map(|&i| new_entries.get(i).map(|e| e.id.clone()))
            .collect();
        if new_ids.is_empty() {
            return 0;
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

        if best_score >= 0.5 {
            if let Some(name) = best_name {
                node.name = Some(name.to_string());
                node.named_at_size = node.members.len();
                node.dirty = false;
                return 1;
            }
        }
        0
    } else {
        let mut count = 0;
        for child in &mut node.children {
            count += inherit_in_node(child, new_entries, old_named);
        }
        count
    }
}

fn count_unnamed_leaves(roots: &[TopicNode]) -> usize {
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
