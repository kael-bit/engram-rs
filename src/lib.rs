#![recursion_limit = "256"]

pub mod ai;
pub mod api;
pub mod consolidate;
pub mod db;
pub mod error;
pub mod extract;
pub mod prompts;
pub mod proxy;
pub mod recall;
pub mod scoring;
pub mod thresholds;
pub mod topiary;
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
    pub embed_queue: Option<EmbedQueue>,
    pub proxy: Option<proxy::ProxyConfig>,
    pub started_at: std::time::Instant,
    pub last_proxy_turn: std::sync::Arc<std::sync::atomic::AtomicI64>,
    pub last_activity: std::sync::Arc<std::sync::atomic::AtomicI64>,
    pub topiary_trigger: Option<tokio::sync::mpsc::UnboundedSender<()>>,
}

use lru::LruCache;
use std::num::NonZeroUsize;

#[derive(Clone)]
pub struct EmbedCache {
    inner: std::sync::Arc<parking_lot::Mutex<EmbedCacheInner>>,
    db: Option<SharedDB>,
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
            db: None,
        }
    }

    /// Create and warm from persistent DB cache.
    pub fn with_db(capacity: usize, db: &SharedDB) -> Self {
        let entries = db.embed_cache_load_all();
        let loaded = entries.len();
        let cap = NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(128).unwrap());
        let mut cache = LruCache::new(cap);
        for (q, emb) in entries {
            cache.put(q, emb);
        }
        if loaded > 0 {
            tracing::info!(loaded, "embed cache warmed from db");
        }
        Self {
            inner: std::sync::Arc::new(parking_lot::Mutex::new(EmbedCacheInner {
                cache,
                hits: 0,
                misses: 0,
            })),
            db: Some(db.clone()),
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
        if let Some(ref db) = self.db {
            db.embed_cache_put(&key, &value);
        }
        let mut inner = self.inner.lock();
        inner.cache.put(key, value);
    }

    pub fn stats(&self) -> (usize, usize, u64, u64) {
        let inner = self.inner.lock();
        (inner.cache.len(), inner.cache.cap().get(), inner.hits, inner.misses)
    }
}

/// Embed queue: collects (id, content) pairs and batches them for embedding.
/// Background worker drains the queue every 200ms or when 50 items accumulate.
#[derive(Clone)]
pub struct EmbedQueue {
    tx: std::sync::Arc<tokio::sync::mpsc::UnboundedSender<(String, String)>>,
    pending: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl EmbedQueue {
    pub fn new(db: SharedDB, cfg: ai::AiConfig, topiary_tx: Option<tokio::sync::mpsc::UnboundedSender<()>>) -> Self {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let pending = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let pending2 = pending.clone();
        tokio::spawn(Self::worker(rx, db, cfg, pending2, topiary_tx));
        Self {
            tx: std::sync::Arc::new(tx),
            pending,
        }
    }

    pub fn push(&self, id: String, content: String) {
        self.pending.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _ = self.tx.send((id, content));
    }

    pub fn pending_count(&self) -> usize {
        self.pending.load(std::sync::atomic::Ordering::Relaxed)
    }

    async fn worker(
        mut rx: tokio::sync::mpsc::UnboundedReceiver<(String, String)>,
        db: SharedDB,
        cfg: ai::AiConfig,
        pending: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        topiary_tx: Option<tokio::sync::mpsc::UnboundedSender<()>>,
    ) {
        use tracing::{info, warn};

        let mut batch: Vec<(String, String)> = Vec::with_capacity(thresholds::EMBED_BATCH_SIZE);

        loop {
            // Phase 1: block until first item arrives (or channel closes)
            match rx.recv().await {
                Some(item) => batch.push(item),
                None => break,
            }

            // Phase 2: time window starts now — collect until deadline or cap
            let deadline = tokio::time::Instant::now()
                + std::time::Duration::from_millis(thresholds::EMBED_WINDOW_MS);

            while batch.len() < thresholds::EMBED_BATCH_SIZE {
                let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                if remaining.is_zero() {
                    break;
                }
                match tokio::time::timeout(remaining, rx.recv()).await {
                    Ok(Some(item)) => batch.push(item),
                    Ok(None) => break,  // channel closed
                    Err(_) => break,    // timeout — window expired
                }
            }

            // Phase 3: flush
            let items = std::mem::take(&mut batch);
            let count = items.len();
            info!(batch_size = count, "embed queue flushing");

            let texts: Vec<String> = items.iter().map(|(_, c)| c.clone()).collect();
            {
                use backon::{ExponentialBuilder, Retryable};

                let result = (|| ai::get_embeddings(&cfg, &texts))
                    .retry(ExponentialBuilder::default().with_max_times(3))
                    .notify(|err, dur| {
                        warn!(error = %err, retry_after = ?dur, batch_size = count, "embed queue batch failed, retrying");
                    })
                    .await;

                match result {
                    Ok(er) => {
                        if let Some(ref u) = er.usage {
                            let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
                            let _ = db.log_llm_call("embed_queue", &cfg.embed_model, u.prompt_tokens, u.completion_tokens, cached, 0);
                        }
                        for (emb, (id, _)) in er.embeddings.into_iter().zip(items.iter()) {
                            let db = db.clone();
                            let id = id.clone();
                            let _ = tokio::task::spawn_blocking(move || {
                                db.set_embedding(&id, &emb)
                            }).await;
                        }
                        // Trigger topiary rebuild after embeddings are stored
                        if let Some(ref tx) = topiary_tx {
                            let _ = tx.send(());
                        }
                    }
                    Err(e) => {
                        warn!(error = %e, batch_size = count, "embed queue batch failed after retries");
                    }
                }
            }
            pending.fetch_sub(count, std::sync::atomic::Ordering::Relaxed);
        }
    }
}
