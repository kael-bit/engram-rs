//! engram — three-layer memory engine for AI agents.
//! buffer → working → core, with decay + promotion.

mod ai;
mod api;
mod consolidate;
mod db;
mod error;
mod recall;

use clap::Parser;
use std::sync::{Arc, Mutex};
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

pub type SharedDB = Arc<Mutex<db::MemoryDB>>;

pub fn lock_db(db: &SharedDB) -> std::sync::MutexGuard<'_, db::MemoryDB> {
    db.lock().unwrap_or_else(|poisoned| {
        tracing::warn!("recovering from poisoned mutex");
        poisoned.into_inner()
    })
}

#[derive(Clone)]
pub struct AppState {
    pub(crate) db: SharedDB,
    pub(crate) ai: Option<ai::AiConfig>,
    pub(crate) api_key: Option<String>,
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
    let shared: SharedDB = Arc::new(Mutex::new(mdb));

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

    let state = AppState { db: shared.clone(), ai: ai_cfg, api_key };
    let app = api::router(state.clone());

    // background consolidation — runs every ENGRAM_CONSOLIDATE_MINS (default 30)
    let consolidate_mins: u64 = std::env::var("ENGRAM_CONSOLIDATE_MINS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    if consolidate_mins > 0 {
        let bg_state = state.clone();
        tokio::spawn(async move {
            let interval = std::time::Duration::from_secs(consolidate_mins * 60);
            // wait a bit before first run so startup isn't slowed
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            loop {
                let db = bg_state.db.clone();
                let result = tokio::task::spawn_blocking(move || {
                    consolidate::consolidate_sync(&lock_db(&db), None)
                })
                .await;
                match result {
                    Ok(r) => {
                        if r.promoted > 0 || r.decayed > 0 {
                            info!(promoted = r.promoted, decayed = r.decayed, "auto-consolidate");
                        }
                    }
                    Err(e) => tracing::warn!(error = %e, "auto-consolidate failed"),
                }
                tokio::time::sleep(interval).await;
            }
        });
        info!(every_mins = consolidate_mins, "background consolidation enabled");
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
    let _ = tokio::signal::ctrl_c().await;
    info!("shutting down");
}
