//! engram — three-layer memory engine for AI agents.
//! buffer → working → core, with decay + promotion.

use engram::{ai, api, consolidate, db, proxy, topiary, AppState, EmbedCache, EmbedQueue, SharedDB};

use clap::Parser;
use std::sync::Arc;
use tracing::{debug, info, warn};
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

    let embed_cache_cap: usize = std::env::var("ENGRAM_EMBED_CACHE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let embed_cache = EmbedCache::new(embed_cache_cap);

    // Topiary trigger channel
    let (topiary_tx, topiary_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
    let topiary_tx_clone = topiary_tx.clone();

    let embed_queue = ai_cfg.as_ref()
        .filter(|c| c.has_embed())
        .map(|c| EmbedQueue::new(shared.clone(), c.clone(), Some(topiary_tx_clone)));

    // Spawn topiary worker
    let _topiary_handle = topiary::worker::spawn_worker(
        shared.clone(),
        ai_cfg.clone(),
        topiary_rx,
    );

    proxy::init_proxy_counters(&shared);
    let state = AppState {
        db: shared.clone(), ai: ai_cfg, api_key, embed_cache,
        embed_queue,
        proxy: proxy_cfg,
        started_at: std::time::Instant::now(),
        last_proxy_turn: std::sync::Arc::new(std::sync::atomic::AtomicI64::new(0)),
        last_activity: std::sync::Arc::new(std::sync::atomic::AtomicI64::new(db::now_ms())),
        topiary_trigger: Some(topiary_tx),
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
            let mut last_consolidation_ts: i64 = db::now_ms();
            loop {
                // Skip consolidation if no write activity since last run
                let last_act = bg_state.last_activity.load(std::sync::atomic::Ordering::Relaxed);
                if last_act <= last_consolidation_ts {
                    debug!("consolidation skipped: no activity since last run");
                    tokio::time::sleep(interval).await;
                    continue;
                }

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
                    bg_state.db.clone(), req, ai_cfg, false,
                ).await;
                // Trigger topiary rebuild after consolidation
                if let Some(ref tx) = bg_state.topiary_trigger {
                    let _ = tx.send(());
                }
                if r.promoted > 0 || r.decayed > 0 || r.merged > 0
                    || r.gate_rejected > 0 || r.demoted > 0 || r.reconciled > 0
                    || r.distilled > 0
                {
                    info!(
                        promoted = r.promoted,
                        decayed = r.decayed,
                        merged = r.merged,
                        reconciled = r.reconciled,
                        gate_rejected = r.gate_rejected,
                        demoted = r.demoted,
                        distilled = r.distilled,
                        "auto-consolidate"
                    );
                } else {
                    debug!("auto-consolidate: no changes");
                }
                last_consolidation_ts = db::now_ms();
                tokio::time::sleep(interval).await;
            }
        });
        info!(every_mins = consolidate_mins, auto_merge = auto_merge, "background consolidation enabled");
    }

    // Background audit — runs every ENGRAM_AUDIT_HOURS (default 12, 0 = disabled)
    let audit_hours: u64 = std::env::var("ENGRAM_AUDIT_HOURS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    if audit_hours > 0 {
        if let Some(ref ai) = state.ai {
            let audit_db = state.db.clone();
            let audit_ai = ai.clone();
            let audit_last_activity = state.last_activity.clone();
            tokio::spawn(async move {
                let interval_secs = audit_hours.saturating_mul(3600);
                let interval = std::time::Duration::from_secs(interval_secs);

                // Check when audit last ran (survives restarts)
                let last_run: i64 = audit_db.get_meta("last_audit_ms")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                let now_ms = engram::db::now_ms();
                let elapsed_ms = now_ms - last_run;
                let remaining_ms = (interval_secs as i64 * 1000) - elapsed_ms;

                if remaining_ms > 300_000 {
                    // Haven't reached the interval yet — sleep until it's due
                    tokio::time::sleep(std::time::Duration::from_millis(remaining_ms as u64)).await;
                } else {
                    // Overdue or first run — wait 5 min then go
                    tokio::time::sleep(std::time::Duration::from_secs(300)).await;
                }

                let mut last_audit_ts: i64 = engram::db::now_ms();
                loop {
                    // Skip audit if no write activity since last run
                    let last_act = audit_last_activity.load(std::sync::atomic::Ordering::Relaxed);
                    if last_act <= last_audit_ts {
                        debug!("audit skipped: no activity since last run");
                        tokio::time::sleep(interval).await;
                        continue;
                    }

                    match consolidate::sandbox_audit(&audit_ai, &audit_db, true).await {
                        Ok(r) if r.applied > 0 => {
                            info!(
                                reviewed = r.total_reviewed,
                                proposed = r.ops_proposed,
                                applied = r.applied,
                                skipped = r.skipped,
                                score = format!("{:.0}%", r.score * 100.0),
                                "auto-audit"
                            );
                        }
                        Ok(r) if !r.safe_to_apply && r.ops_proposed > 0 => {
                            warn!(
                                proposed = r.ops_proposed,
                                score = format!("{:.0}%", r.score * 100.0),
                                "auto-audit: score below threshold, nothing applied"
                            );
                        }
                        Ok(_) => { /* no changes or no ops, stay quiet */ }
                        Err(e) => {
                            warn!("auto-audit failed: {e}");
                        }
                    }
                    if let Err(e) = audit_db.set_meta("last_audit_ms", &engram::db::now_ms().to_string()) {
                        warn!("failed to update last_audit_ms: {e}");
                    }
                    last_audit_ts = engram::db::now_ms();
                    tokio::time::sleep(interval).await;
                }
            });
            info!(every_hours = audit_hours, model = %ai.model_for("gate"), "background audit enabled");
        }
    }

    // Flush proxy conversation window with debounce.
    // Instead of a fixed clock, wait for 30s of silence after the last turn.
    if state.proxy.is_some() {
        let flush_state = state.clone();
        tokio::spawn(async move {
            let silence_threshold_ms: i64 = 30_000; // 30s of quiet
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                let last = flush_state.last_proxy_turn.load(std::sync::atomic::Ordering::Relaxed);
                if last == 0 {
                    continue; // no turns yet
                }
                let now = engram::db::now_ms();
                let silence = now - last;
                if silence >= silence_threshold_ms {
                    // Reset before flushing so new turns during flush start a fresh debounce
                    flush_state.last_proxy_turn.store(0, std::sync::atomic::Ordering::Relaxed);
                    proxy::flush_window(&flush_state).await;
                }
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

    let state_for_shutdown = state.clone();
    let addr = format!("0.0.0.0:{}", args.port);

    // Try to inherit a socket from systemd socket activation or systemfd.
    // Falls back to binding a new socket if none is passed.
    let listener = {
        let mut lfd = listenfd::ListenFd::from_env();
        if let Ok(Some(std_listener)) = lfd.take_tcp_listener(0) {
            std_listener.set_nonblocking(true).expect("set_nonblocking");
            info!("inherited socket from systemd/systemfd");
            tokio::net::TcpListener::from_std(std_listener).expect("from_std listener")
        } else {
            tokio::net::TcpListener::bind(&addr)
                .await
                .expect("failed to bind address")
        }
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");

    // Flush any buffered proxy conversations before exit
    if state_for_shutdown.proxy.is_some() {
        info!("flushing proxy window before exit");
        proxy::flush_window(&state_for_shutdown).await;
    }
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sigterm =
            signal(SignalKind::terminate()).expect("failed to register SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = sigterm.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to listen for ctrl_c");
    }
    info!("shutting down");
}
