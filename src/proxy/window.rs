use tracing::warn;

use crate::{error::EngramError, util::truncate_chars, AppState};
use super::extract::extract_from_context;
use super::parse::extract_message_content;

const WINDOW_MAX_TURNS: usize = 8;
const WINDOW_MAX_CHARS: usize = 16000;
// Debounce: don't flush until this many seconds of quiet after last turn.
const FLUSH_QUIET_SECS: i64 = 30;

/// Parse a single exchange from raw request/response bytes, add to the sliding window.
/// When window is full, flush and extract memories from the accumulated context.
pub(crate) async fn buffer_exchange(state: AppState, req_raw: Vec<u8>, res_raw: Vec<u8>, session_key: String) {
    let assistant_msg = super::parse::extract_assistant_msg(&res_raw);

    // Watermark extraction involves DB reads/writes — run on blocking thread
    let wm_state = state.clone();
    let wm_raw = req_raw.clone();
    let wm_key = session_key.clone();
    let user_msg = match tokio::task::spawn_blocking(move || {
        extract_user_via_watermark(&wm_state, &wm_raw, &wm_key)
    }).await {
        Ok(Some(msg)) => msg,
        Ok(None) | Err(_) => {
            if assistant_msg.is_empty() {
                return;
            }
            String::new()
        }
    };

    if user_msg.is_empty() && assistant_msg.is_empty() {
        return;
    }

    // Skip very short tool-use-only turns
    if user_msg.len() + assistant_msg.len() < 80 {
        return;
    }

    let turn = format!("User: {}\nAssistant: {}", user_msg, assistant_msg);

    // All DB operations on a blocking thread to avoid starving tokio workers
    let db = state.db.clone();
    let sk = session_key.clone();
    let turn_c = turn.clone();
    let should_flush = tokio::task::spawn_blocking(move || -> Result<Option<String>, EngramError> {
        db.save_proxy_turn(&sk, &turn_c)?;
        if db.proxy_session_should_flush(&sk, WINDOW_MAX_TURNS, WINDOW_MAX_CHARS)
            && db.proxy_session_quiet_for(&sk, FLUSH_QUIET_SECS)
        {
            let ctx = db.drain_proxy_session(&sk)?;
            if !ctx.is_empty() {
                return Ok(Some(ctx));
            }
        }
        Ok(None)
    }).await;

    state.last_proxy_turn.store(
        crate::db::now_ms(),
        std::sync::atomic::Ordering::Relaxed,
    );

    match should_flush {
        Ok(Ok(Some(ctx))) => extract_from_context(state, &ctx).await,
        Ok(Err(e)) => warn!("proxy: DB error in turn save: {e}"),
        Err(e) => warn!("proxy: spawn_blocking failed: {e}"),
        _ => {}
    }
}

/// Flush all remaining buffered turns. Called from background timer and shutdown.
pub async fn flush_window(state: &AppState) {
    let db = state.db.clone();
    let pending = match tokio::task::spawn_blocking(move || db.drain_all_proxy_turns()).await {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            warn!("proxy: failed to drain turns: {e}");
            return;
        }
        Err(e) => {
            warn!("proxy: spawn_blocking failed: {e}");
            return;
        }
    };
    for (_key, context) in pending {
        if !context.is_empty() {
            extract_from_context(state.clone(), &context).await;
        }
    }
}

/// Apply watermark logic to extract new user messages from the request.
pub(super) fn extract_user_via_watermark(state: &AppState, req_raw: &[u8], session_key: &str) -> Option<String> {
    let text = String::from_utf8_lossy(req_raw);
    let body: serde_json::Value = serde_json::from_str(&text).ok()?;

    let messages = body.get("messages")?.as_array()?;
    if messages.is_empty() {
        return None;
    }

    let filtered: Vec<&serde_json::Value> = messages
        .iter()
        .filter(|m| {
            matches!(
                m.get("role").and_then(|r| r.as_str()),
                Some("user") | Some("assistant")
            )
        })
        .collect();

    let msg_count = filtered.len() as i64;
    let watermark = state.db.get_watermark(session_key).unwrap_or(0);

    if watermark == 0 {
        let _ = state.db.set_watermark(session_key, msg_count);
        return None;
    }

    if msg_count <= watermark {
        let _ = state.db.set_watermark(session_key, msg_count);
        return None;
    }

    let new_msgs = &filtered[watermark as usize..];
    let mut parts: Vec<String> = Vec::new();
    for msg in new_msgs {
        if msg.get("role").and_then(|r| r.as_str()) == Some("user") {
            if let Some(content) = extract_message_content(msg) {
                if !content.is_empty() {
                    parts.push(content);
                }
            }
        }
    }

    let _ = state.db.set_watermark(session_key, msg_count);

    if parts.is_empty() {
        return Some(String::new());
    }

    Some(truncate_chars(&parts.join("\n"), 6000))
}
