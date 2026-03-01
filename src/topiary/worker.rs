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
                .and_then(|s| s.parse::<u64>().ok())
                .map_or(false, |fp| fp == current_fingerprint);
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
fn assign_fallback_names(roots: &mut [TopicNode], entries: &[Entry]) -> usize {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topiary::Entry;

    fn make_leaf(id: &str, name: Option<&str>, members: Vec<usize>) -> TopicNode {
        let size = members.len();
        TopicNode {
            id: id.to_string(),
            name: name.map(|s| s.to_string()),
            centroid: vec![],
            members,
            children: vec![],
            dirty: name.is_none(),
            avg_sim: 0.0,
            named_at_size: if name.is_some() { size } else { 0 },
        }
    }

    fn make_entry(id: &str) -> Entry {
        Entry {
            id: id.to_string(),
            text: String::new(),
            embedding: vec![],
            tags: vec![],
        }
    }

    fn old_tree_json(roots: Vec<TopicNode>) -> String {
        let tree = TopicTree::new(0.3, 0.55).with_roots(roots);
        serde_json::to_string(&tree).unwrap()
    }

    // ── inherit_names ─────────────────────────────────────────────────

    #[test]
    fn inherit_exact_match() {
        let entries = vec![make_entry("m1"), make_entry("m2"), make_entry("m3")];
        let mut roots = vec![make_leaf("kb1", None, vec![0, 1, 2])];

        let tree_json = old_tree_json(vec![make_leaf("old1", Some("Deploy"), vec![0, 1, 2])]);
        let ids_json = serde_json::to_string(&vec!["m1", "m2", "m3"]).unwrap();

        let n = inherit_names(&mut roots, &entries, Some(tree_json), Some(ids_json));
        assert_eq!(n, 1);
        assert_eq!(roots[0].name.as_deref(), Some("Deploy"));
        assert!(!roots[0].dirty);
    }

    #[test]
    fn inherit_above_threshold() {
        // Old: {m1,m2,m3,m4,m5}, New: {m1,m2,m3,m6} → Jaccard 3/6=0.5 ≥ 0.3
        let entries = vec![
            make_entry("m1"), make_entry("m2"),
            make_entry("m3"), make_entry("m6"),
        ];
        let mut roots = vec![make_leaf("kb1", None, vec![0, 1, 2, 3])];

        let tree_json = old_tree_json(vec![
            make_leaf("old1", Some("Config"), vec![0, 1, 2, 3, 4]),
        ]);
        let ids_json = serde_json::to_string(&vec!["m1", "m2", "m3", "m4", "m5"]).unwrap();

        let n = inherit_names(&mut roots, &entries, Some(tree_json), Some(ids_json));
        assert_eq!(n, 1);
        assert_eq!(roots[0].name.as_deref(), Some("Config"));
    }

    #[test]
    fn inherit_below_threshold_rejected() {
        // Old: {m1..m10}, New: {m1, m11..m19} → Jaccard 1/19 ≈ 0.05 < 0.3
        let entries: Vec<Entry> = (0..10)
            .map(|i| {
                let id = if i == 0 { "m1".into() } else { format!("m{}", 10 + i) };
                make_entry(&id)
            })
            .collect();
        let mut roots = vec![make_leaf("kb1", None, (0..10).collect())];

        let tree_json = old_tree_json(vec![
            make_leaf("old1", Some("Big"), (0..10).collect()),
        ]);
        let ids_json = serde_json::to_string(
            &(1..=10).map(|i| format!("m{i}")).collect::<Vec<_>>(),
        ).unwrap();

        let n = inherit_names(&mut roots, &entries, Some(tree_json), Some(ids_json));
        assert_eq!(n, 0);
        assert!(roots[0].name.is_none());
    }

    #[test]
    fn inherit_greedy_dedup() {
        // Old "Deploy" = {m1,m2,m3,m4}
        // New A = {m1,m2}, B = {m3,m4} — both Jaccard 0.5 to "Deploy"
        // Only one should get the name
        let entries = vec![
            make_entry("m1"), make_entry("m2"),
            make_entry("m3"), make_entry("m4"),
        ];
        let mut roots = vec![
            make_leaf("kb1", None, vec![0, 1]),
            make_leaf("kb2", None, vec![2, 3]),
        ];

        let tree_json = old_tree_json(vec![
            make_leaf("old1", Some("Deploy"), vec![0, 1, 2, 3]),
        ]);
        let ids_json = serde_json::to_string(&vec!["m1", "m2", "m3", "m4"]).unwrap();

        let n = inherit_names(&mut roots, &entries, Some(tree_json), Some(ids_json));
        assert_eq!(n, 1);
        let named = roots.iter().filter(|r| r.name.as_deref() == Some("Deploy")).count();
        assert_eq!(named, 1, "only one topic should get the name");
    }

    #[test]
    fn inherit_multiple_old_topics() {
        // Old: "Deploy"={m1,m2}, "Config"={m3,m4}
        // New: A={m1,m2,m5}, B={m3,m4}
        let entries = vec![
            make_entry("m1"), make_entry("m2"),
            make_entry("m3"), make_entry("m4"),
            make_entry("m5"),
        ];
        let mut roots = vec![
            make_leaf("kb1", None, vec![0, 1, 4]),
            make_leaf("kb2", None, vec![2, 3]),
        ];

        let tree_json = old_tree_json(vec![
            make_leaf("old1", Some("Deploy"), vec![0, 1]),
            make_leaf("old2", Some("Config"), vec![2, 3]),
        ]);
        let ids_json = serde_json::to_string(&vec!["m1", "m2", "m3", "m4"]).unwrap();

        let n = inherit_names(&mut roots, &entries, Some(tree_json), Some(ids_json));
        assert_eq!(n, 2);
        assert_eq!(roots[0].name.as_deref(), Some("Deploy"));
        assert_eq!(roots[1].name.as_deref(), Some("Config"));
    }

    #[test]
    fn inherit_no_cache() {
        let entries = vec![make_entry("m1")];
        let mut roots = vec![make_leaf("kb1", None, vec![0])];
        assert_eq!(inherit_names(&mut roots, &entries, None, None), 0);
    }

    #[test]
    fn inherit_already_named_skipped() {
        let entries = vec![make_entry("m1"), make_entry("m2")];
        let mut roots = vec![make_leaf("kb1", Some("Existing"), vec![0, 1])];

        let tree_json = old_tree_json(vec![
            make_leaf("old1", Some("OldName"), vec![0, 1]),
        ]);
        let ids_json = serde_json::to_string(&vec!["m1", "m2"]).unwrap();

        let n = inherit_names(&mut roots, &entries, Some(tree_json), Some(ids_json));
        assert_eq!(n, 0);
        assert_eq!(roots[0].name.as_deref(), Some("Existing"));
    }

    #[test]
    fn inherit_best_match_wins() {
        // Old: "Alpha"={m1,m2}, "Beta"={m3,m4,m5}
        // New A={m1,m2,m3} best→Alpha(0.67), New B={m3,m4,m5} best→Beta(1.0)
        // B gets Beta first (higher score), A gets Alpha
        let entries = vec![
            make_entry("m1"), make_entry("m2"), make_entry("m3"),
            make_entry("m4"), make_entry("m5"),
        ];
        let mut roots = vec![
            make_leaf("kb1", None, vec![0, 1, 2]),
            make_leaf("kb2", None, vec![2, 3, 4]),
        ];

        let tree_json = old_tree_json(vec![
            make_leaf("old1", Some("Alpha"), vec![0, 1]),
            make_leaf("old2", Some("Beta"), vec![2, 3, 4]),
        ]);
        let ids_json = serde_json::to_string(&vec!["m1", "m2", "m3", "m4", "m5"]).unwrap();

        let n = inherit_names(&mut roots, &entries, Some(tree_json), Some(ids_json));
        assert_eq!(n, 2);
        assert_eq!(roots[0].name.as_deref(), Some("Alpha"));
        assert_eq!(roots[1].name.as_deref(), Some("Beta"));
    }

    // ── count_unnamed ─────────────────────────────────────────────────

    #[test]
    fn count_unnamed_mixed() {
        let roots = vec![
            make_leaf("kb1", Some("Named"), vec![0]),
            make_leaf("kb2", None, vec![1]),
            make_leaf("kb3", Some("Also"), vec![2]),
            make_leaf("kb4", None, vec![3]),
        ];
        assert_eq!(count_unnamed_leaves(&roots), 2);
    }

    // ── fallback names ────────────────────────────────────────────────

    fn make_entry_with_tags(id: &str, tags: &[&str]) -> Entry {
        Entry {
            id: id.to_string(),
            text: String::new(),
            embedding: vec![],
            tags: tags.iter().map(|t| t.to_string()).collect(),
        }
    }

    #[test]
    fn fallback_uses_most_common_tag() {
        let entries = vec![
            make_entry_with_tags("m1", &["deploy-flow", "ops"]),
            make_entry_with_tags("m2", &["deploy-flow"]),
            make_entry_with_tags("m3", &["ops"]),
        ];
        let mut roots = vec![make_leaf("kb1", None, vec![0, 1, 2])];
        let n = assign_fallback_names(&mut roots, &entries);
        assert_eq!(n, 1);
        // "deploy-flow" appears 2 times, "ops" appears 2 times — either is valid
        let name = roots[0].name.as_deref().unwrap();
        assert!(
            name == "Deploy Flow" || name == "Ops",
            "expected tag-based name, got: {name}"
        );
        assert!(!roots[0].dirty);
    }

    #[test]
    fn fallback_no_tags_uses_topic_id() {
        let entries = vec![make_entry("m1"), make_entry("m2")];
        let mut roots = vec![make_leaf("kb1", None, vec![0, 1])];
        let n = assign_fallback_names(&mut roots, &entries);
        assert_eq!(n, 1);
        assert_eq!(roots[0].name.as_deref(), Some("Topic kb1"));
    }

    #[test]
    fn fallback_skips_already_named() {
        let entries = vec![make_entry("m1")];
        let mut roots = vec![make_leaf("kb1", Some("Existing"), vec![0])];
        let n = assign_fallback_names(&mut roots, &entries);
        assert_eq!(n, 0);
        assert_eq!(roots[0].name.as_deref(), Some("Existing"));
    }
}
