//! engram — three-layer memory engine for AI agents.
//! buffer → working → core, with decay + promotion.

mod ai;
mod api;
mod consolidate;
mod db;
mod error;
mod proxy;
mod recall;

use clap::Parser;
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "engram", version, about = "Hierarchical memory engine for AI agents")]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value = "3917", env = "ENGRAM_PORT")]
    port: u16,

    /// SQLite database path
    #[arg(short, long, default_value = "engram.db", env = "ENGRAM_DB")]
    db: String,
}

pub type SharedDB = Arc<db::MemoryDB>;

#[derive(Clone)]
pub struct AppState {
    pub(crate) db: SharedDB,
    pub(crate) ai: Option<ai::AiConfig>,
    pub(crate) api_key: Option<String>,
    pub(crate) embed_cache: EmbedCache,
    pub(crate) proxy: Option<proxy::ProxyConfig>,
    pub(crate) started_at: std::time::Instant,
}

/// Small LRU-ish cache for query embeddings to avoid repeated API calls.
/// Key = query text, Value = embedding vector.
pub type EmbedCache = std::sync::Arc<std::sync::Mutex<EmbedCacheInner>>;

pub struct EmbedCacheInner {
    map: std::collections::HashMap<String, Vec<f64>>,
    order: std::collections::VecDeque<String>,
    cap: usize,
}

impl EmbedCacheInner {
    pub fn new(cap: usize) -> Self {
        Self {
            map: std::collections::HashMap::with_capacity(cap),
            order: std::collections::VecDeque::with_capacity(cap),
            cap,
        }
    }

    pub fn get(&mut self, key: &str) -> Option<&Vec<f64>> {
        if self.map.contains_key(key) {
            // promote to back (most recently used)
            if let Some(pos) = self.order.iter().position(|k| k == key) {
                self.order.remove(pos);
                self.order.push_back(key.to_string());
            }
        }
        self.map.get(key)
    }

    pub fn put(&mut self, key: String, val: Vec<f64>) {
        if self.map.contains_key(&key) {
            // update value in-place and refresh position in order
            self.map.insert(key.clone(), val);
            if let Some(pos) = self.order.iter().position(|k| k == &key) {
                self.order.remove(pos);
                self.order.push_back(key);
            }
            return;
        }
        if self.cap == 0 {
            return;
        }
        if self.order.len() >= self.cap {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
        self.order.push_back(key.clone());
        self.map.insert(key, val);
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let args = Args::parse();
    let mdb = db::MemoryDB::open(&args.db).expect("failed to open database");
    let shared: SharedDB = Arc::new(mdb);

    let ai_cfg = ai::AiConfig::from_env();
    let ai_status = match &ai_cfg {
        Some(cfg) => {
            let mut parts = vec![];
            if cfg.has_llm() {
                parts.push(format!("llm={}", cfg.llm_model));
            }
            if cfg.has_embed() {
                parts.push(format!("embed={}", cfg.embed_model));
            }
            parts.join(", ")
        }
        None => "disabled".into(),
    };

    let api_key = std::env::var("ENGRAM_API_KEY").ok();
    let auth_status = if api_key.is_some() { "enabled" } else { "disabled" };

    let proxy_cfg = std::env::var("ENGRAM_PROXY_UPSTREAM").ok().map(|upstream| {
        let upstream = upstream.trim_end_matches('/').to_string();
        info!(upstream = %upstream, "proxy enabled");
        proxy::ProxyConfig {
            upstream,
            default_key: std::env::var("ENGRAM_PROXY_KEY").ok(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .expect("failed to build proxy client"),
        }
    });

    let embed_cache: EmbedCache =
        std::sync::Arc::new(std::sync::Mutex::new(EmbedCacheInner::new(128)));
    let state = AppState {
        db: shared.clone(), ai: ai_cfg, api_key, embed_cache,
        proxy: proxy_cfg,
        started_at: std::time::Instant::now(),
    };
    let app = api::router(state.clone());

    // background consolidation — runs every ENGRAM_CONSOLIDATE_MINS (default 30)
    let consolidate_mins: u64 = std::env::var("ENGRAM_CONSOLIDATE_MINS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    let auto_merge = std::env::var("ENGRAM_AUTO_MERGE")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if consolidate_mins > 0 {
        let bg_state = state.clone();
        tokio::spawn(async move {
            let interval = std::time::Duration::from_secs(consolidate_mins.saturating_mul(60));
            // wait a bit before first run so startup isn't slowed
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            loop {
                let req = if auto_merge {
                    Some(consolidate::ConsolidateRequest {
                        merge: Some(true),
                        promote_threshold: None,
                        promote_min_importance: None,
                        decay_drop_threshold: None,
                        buffer_ttl_secs: None,
                        working_age_promote_secs: None,
                    })
                } else {
                    None
                };
                let ai_cfg = bg_state.ai.clone();
                let r = consolidate::consolidate(
                    bg_state.db.clone(), req, ai_cfg,
                ).await;
                if r.promoted > 0 || r.decayed > 0 || r.merged > 0 {
                    info!(
                        promoted = r.promoted,
                        decayed = r.decayed,
                        merged = r.merged,
                        "auto-consolidate"
                    );
                }
                tokio::time::sleep(interval).await;
            }
        });
        info!(every_mins = consolidate_mins, auto_merge = auto_merge, "background consolidation enabled");
    }

    // Flush proxy conversation window periodically (every 2 min)
    if state.proxy.is_some() {
        let flush_state = state.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(120)).await;
                proxy::flush_window(&flush_state).await;
            }
        });
    }

    info!(
        version = env!("CARGO_PKG_VERSION"),
        port = args.port,
        db = %args.db,
        ai = %ai_status,
        auth = auth_status,
        "engram starting"
    );

    let addr = format!("0.0.0.0:{}", args.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("failed to bind address");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");
}

async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};

    let mut sigterm = signal(SignalKind::terminate()).expect("failed to register SIGTERM handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = sigterm.recv() => {}
    }
    info!("shutting down");
}
