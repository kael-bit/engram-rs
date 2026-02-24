#![recursion_limit = "256"]

pub mod ai;
pub mod api;
pub mod consolidate;
pub mod db;
pub mod error;
pub mod proxy;
pub mod recall;
pub mod util;

use std::sync::Arc;

pub type SharedDB = Arc<db::MemoryDB>;

/// Run a blocking DB operation on tokio's blocking thread pool.
///
/// All synchronous MemoryDB calls in async context MUST go through this
/// to avoid starving tokio worker threads.
pub async fn db_call<F, T>(db: &SharedDB, f: F) -> Result<T, error::EngramError>
where
    F: FnOnce(&db::MemoryDB) -> T + Send + 'static,
    T: Send + 'static,
{
    let db = Arc::clone(db);
    tokio::task::spawn_blocking(move || f(&db))
        .await
        .map_err(|e| error::EngramError::Internal(e.to_string()))
}

#[derive(Clone)]
pub struct AppState {
    pub db: SharedDB,
    pub ai: Option<ai::AiConfig>,
    pub api_key: Option<String>,
    pub embed_cache: EmbedCache,
    pub proxy: Option<proxy::ProxyConfig>,
    pub started_at: std::time::Instant,
    pub last_proxy_turn: std::sync::Arc<std::sync::atomic::AtomicI64>,
}

use lru::LruCache;
use std::num::NonZeroUsize;

#[derive(Clone)]
pub struct EmbedCache {
    inner: std::sync::Arc<parking_lot::Mutex<EmbedCacheInner>>,
}

struct EmbedCacheInner {
    cache: LruCache<String, Vec<f32>>,
    hits: u64,
    misses: u64,
}

impl EmbedCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: std::sync::Arc::new(parking_lot::Mutex::new(EmbedCacheInner {
                cache: LruCache::new(
                    NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(128).unwrap()),
                ),
                hits: 0,
                misses: 0,
            })),
        }
    }

    pub fn get(&self, key: &str) -> Option<Vec<f32>> {
        let mut inner = self.inner.lock();
        let val = inner.cache.get(key).cloned();
        if val.is_some() {
            inner.hits += 1;
        } else {
            inner.misses += 1;
        }
        val
    }

    pub fn insert(&self, key: String, value: Vec<f32>) {
        let mut inner = self.inner.lock();
        inner.cache.put(key, value);
    }

    pub fn stats(&self) -> (usize, usize, u64, u64) {
        let inner = self.inner.lock();
        (inner.cache.len(), inner.cache.cap().get(), inner.hits, inner.misses)
    }
}
