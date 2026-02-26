//! Debounced async background worker for topiary tree rebuilds.

use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use super::{Entry, TopicTree};
use crate::ai::AiConfig;
use crate::db::{Layer, MemoryDB};

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
        const DEBOUNCE_MS: u64 = 5_000;

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
                + std::time::Duration::from_millis(DEBOUNCE_MS);
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

    // Step 2: Build topic tree
    let mut tree = TopicTree::new(0.30, 0.55);
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

    // Step 5: Serialize and store
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
                elapsed_ms = start.elapsed().as_millis() as u64,
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
            });
        }
    }

    entries
}
