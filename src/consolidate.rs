use crate::db::{Layer, MemoryDB};
use serde::{Deserialize, Serialize};
use tracing::debug;

/// Optional parameters for consolidation.
#[derive(Debug, Deserialize)]
pub struct ConsolidateRequest {
    pub promote_threshold: Option<i64>,
    pub promote_min_importance: Option<f64>,
    pub decay_drop_threshold: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct ConsolidateResponse {
    pub promoted: usize,
    // TODO: implement memory merging (dedup similar content)
    pub decayed: usize,
}

/// Run a consolidation cycle.
pub fn consolidate(db: &MemoryDB, req: Option<&ConsolidateRequest>) -> ConsolidateResponse {
    let promote_threshold = req.and_then(|r| r.promote_threshold).unwrap_or(3);
    let promote_min_imp = req
        .and_then(|r| r.promote_min_importance)
        .unwrap_or(0.6);
    let decay_threshold = req
        .and_then(|r| r.decay_drop_threshold)
        .unwrap_or(0.01);

    let mut promoted = 0_usize;
    let mut decayed = 0_usize;

    // Phase 1: Promote high-value Working → Core
    for mem in db.list_by_layer(Layer::Working) {
        if mem.access_count >= promote_threshold && mem.importance >= promote_min_imp
            && db.promote(&mem.id, Layer::Core).is_ok() {
                promoted += 1;
            }
    }

    // Phase 2: Drop decayed Buffer/Working entries
    for mem in db.get_decayed(decay_threshold) {
        if db.delete(&mem.id).unwrap_or(false) {
            decayed += 1;
        }
    }

    // Phase 3: Buffer cleanup — old L1 entries promote or drop
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_millis() as i64;
    let one_hour = 3_600_000_i64;

    for mem in db.list_by_layer(Layer::Buffer) {
        if mem.created_at < now - one_hour {
            if mem.importance > 0.3 {
                if db.promote(&mem.id, Layer::Working).is_ok() {
                    promoted += 1;
                }
            } else if db.delete(&mem.id).unwrap_or(false) {
                decayed += 1;
            }
        }
    }

    debug!(promoted, decayed, "consolidation complete");

    ConsolidateResponse {
        promoted,
        decayed,
    }
}
