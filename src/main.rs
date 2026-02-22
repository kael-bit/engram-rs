//! engram: hierarchical memory engine for AI agents.
//!
//! Three-layer model based on Atkinson-Shiffrin memory theory:
//! buffer (sensory) → working (short-term) → core (long-term).

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

    let state = AppState { db: shared, ai: ai_cfg, api_key };
    let app = api::router(state);

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
