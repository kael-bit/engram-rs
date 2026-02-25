use std::sync::atomic::{AtomicU64, Ordering};

use crate::db;

static PROXY_REQUESTS: AtomicU64 = AtomicU64::new(0);
static PROXY_EXTRACTED: AtomicU64 = AtomicU64::new(0);
static PROXY_PERSIST_COUNTER: AtomicU64 = AtomicU64::new(0);

pub fn init_proxy_counters(db: &db::MemoryDB) {
    if let Some(v) = db.get_meta("proxy_requests_total") {
        if let Ok(n) = v.parse::<u64>() {
            PROXY_REQUESTS.store(n, Ordering::Relaxed);
        }
    }
    if let Some(v) = db.get_meta("proxy_extracted_total") {
        if let Ok(n) = v.parse::<u64>() {
            PROXY_EXTRACTED.store(n, Ordering::Relaxed);
        }
    }
}

pub(crate) fn persist_proxy_counters(db: &db::MemoryDB) {
    let count = PROXY_PERSIST_COUNTER.fetch_add(1, Ordering::Relaxed);
    if count.is_multiple_of(10) {
        let _ = db.set_meta("proxy_requests_total", &PROXY_REQUESTS.load(Ordering::Relaxed).to_string());
        let _ = db.set_meta("proxy_extracted_total", &PROXY_EXTRACTED.load(Ordering::Relaxed).to_string());
    }
}

pub fn proxy_stats(db: Option<&db::MemoryDB>) -> (u64, u64, usize) {
    let buffered = db.map(db::MemoryDB::proxy_turn_count).unwrap_or(0);
    (
        PROXY_REQUESTS.load(Ordering::Relaxed),
        PROXY_EXTRACTED.load(Ordering::Relaxed),
        buffered,
    )
}

pub(crate) fn bump_requests() {
    PROXY_REQUESTS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn bump_extracted(n: u64) {
    PROXY_EXTRACTED.fetch_add(n, Ordering::Relaxed);
}
