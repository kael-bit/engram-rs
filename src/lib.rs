#![recursion_limit = "256"]

pub mod ai;
pub mod api;
pub mod consolidate;
pub mod db;
pub mod error;
pub mod proxy;
pub mod recall;
pub mod safety;
pub mod util;

use std::sync::Arc;

pub type SharedDB = Arc<db::MemoryDB>;

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
    inner: std::sync::Arc<std::sync::Mutex<EmbedCacheInner>>,
}

struct EmbedCacheInner {
    cache: LruCache<String, Vec<f64>>,
    hits: u64,
    misses: u64,
}

impl EmbedCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: std::sync::Arc::new(std::sync::Mutex::new(EmbedCacheInner {
                cache: LruCache::new(
                    NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(128).unwrap()),
                ),
                hits: 0,
                misses: 0,
            })),
        }
    }

    pub fn get(&self, key: &str) -> Option<Vec<f64>> {
        let mut inner = self.inner.lock().unwrap();
        let val = inner.cache.get(key).cloned();
        if val.is_some() {
            inner.hits += 1;
        } else {
            inner.misses += 1;
        }
        val
    }

    pub fn insert(&self, key: String, value: Vec<f64>) {
        let mut inner = self.inner.lock().unwrap();
        inner.cache.put(key, value);
    }

    pub fn stats(&self) -> (usize, usize, u64, u64) {
        let inner = self.inner.lock().unwrap();
        (inner.cache.len(), inner.cache.cap().get(), inner.hits, inner.misses)
    }
}
