use crate::db::{Layer, MemoryDB};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

#[derive(Debug, Deserialize)]
pub struct ConsolidateRequest {
    pub promote_threshold: Option<i64>,
    pub promote_min_importance: Option<f64>,
    pub decay_drop_threshold: Option<f64>,
    /// Buffer entries older than this (seconds) get promoted or dropped.
    pub buffer_ttl_secs: Option<i64>,
    /// Working entries older than this (seconds) with decent importance auto-promote to core.
    pub working_age_promote_secs: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct ConsolidateResponse {
    pub promoted: usize,
    pub decayed: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub promoted_ids: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub dropped_ids: Vec<String>,
}

pub fn consolidate(db: &MemoryDB, req: Option<&ConsolidateRequest>) -> ConsolidateResponse {
    let promote_threshold = req.and_then(|r| r.promote_threshold).unwrap_or(3);
    let promote_min_imp = req.and_then(|r| r.promote_min_importance).unwrap_or(0.6);
    let decay_threshold = req.and_then(|r| r.decay_drop_threshold).unwrap_or(0.01);
    let buffer_ttl = req.and_then(|r| r.buffer_ttl_secs).unwrap_or(3600) * 1000;
    // Working memories that survive 7 days get auto-promoted
    let working_age = req.and_then(|r| r.working_age_promote_secs).unwrap_or(7 * 86400) * 1000;

    let now = crate::db::now_ms();
    let mut promoted = 0_usize;
    let mut decayed = 0_usize;
    let mut promoted_ids = Vec::new();
    let mut dropped_ids = Vec::new();

    // Phase 1: Promote high-value Working → Core (access-based)
    for mem in db.list_by_layer(Layer::Working) {
        if mem.access_count >= promote_threshold && mem.importance >= promote_min_imp
            && db.promote(&mem.id, Layer::Core).is_ok() {
                promoted_ids.push(mem.id.clone());
                promoted += 1;
            }
    }

    // Phase 2: Age-based Working → Core promotion
    // If a working memory has survived long enough and has decent importance, promote it.
    for mem in db.list_by_layer(Layer::Working) {
        if promoted_ids.contains(&mem.id) {
            continue;
        }
        let age = now - mem.created_at;
        if age > working_age && mem.importance >= 0.5
            && db.promote(&mem.id, Layer::Core).is_ok() {
                promoted_ids.push(mem.id.clone());
                promoted += 1;
            }
    }

    // Phase 3: Drop decayed Buffer/Working entries
    for mem in db.get_decayed(decay_threshold) {
        if db.delete(&mem.id).unwrap_or(false) {
            dropped_ids.push(mem.id.clone());
            decayed += 1;
        }
    }

    // Phase 4: Buffer TTL — old L1 entries promote or drop
    for mem in db.list_by_layer(Layer::Buffer) {
        let age = now - mem.created_at;
        if age > buffer_ttl {
            if mem.importance > 0.3 {
                if db.promote(&mem.id, Layer::Working).is_ok() {
                    promoted_ids.push(mem.id.clone());
                    promoted += 1;
                }
            } else if db.delete(&mem.id).unwrap_or(false) {
                dropped_ids.push(mem.id.clone());
                decayed += 1;
            }
        }
    }

    if promoted > 0 || decayed > 0 {
        info!(promoted, decayed, "consolidation complete");
    } else {
        debug!("consolidation: nothing to do");
    }

    ConsolidateResponse {
        promoted,
        decayed,
        promoted_ids,
        dropped_ids,
    }
}
