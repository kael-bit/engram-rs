use crate::ai::{self, AiConfig};
use crate::db::Layer;
use crate::SharedDB;
use crate::util::truncate_chars;
use tracing::{info, warn};

/// Build or refresh a compressed summary of all Core memories.
///
/// The summary is stored in `engram_meta` and keyed by a hash of Core
/// content. Resume uses this when the full Core listing would exceed the
/// budget — one LLM-generated paragraph instead of N truncated lines.
pub(super) async fn update_core_summary(db: &SharedDB, cfg: &AiConfig) {
    let db2 = db.clone();
    let core = match tokio::task::spawn_blocking(move || {
        db2.list_by_layer_meta(Layer::Core, 500, 0)
    }).await {
        Ok(Ok(c)) => c,
        _ => return,
    };

    // Only bother summarizing when there are enough Core memories to cause
    // budget pressure. Below this threshold, full listing always fits.
    if core.len() < 10 { return; }

    // Hash Core IDs + content lengths to detect changes cheaply.
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for m in &core {
        m.id.hash(&mut hasher);
        m.content.len().hash(&mut hasher);
    }
    let hash = format!("{:016x}", hasher.finish());

    // Skip if nothing changed since last summary
    let db3 = db.clone();
    let h = hash.clone();
    let cached_hash = tokio::task::spawn_blocking(move || {
        db3.get_meta("core_summary_hash")
    }).await.unwrap_or(None);

    if cached_hash.as_deref() == Some(&h) { return; }

    // Build input text — full content with tags for context
    let mut input = String::with_capacity(core.len() * 200);
    for m in &core {
        let tags = if m.tags.is_empty() { String::new() } else { format!(" [{}]", m.tags.join(",")) };
        let kind_label = if m.kind != "semantic" { format!(" ({})", m.kind) } else { String::new() };
        input.push_str(&format!("-{kind_label}{tags} {}\n", truncate_chars(&m.content, 300)));
    }

    let system = "Compress the following list of core memories into a dense summary.\n\
        Preserve ALL specific facts, names, values, constraints, and lessons.\n\
        Combine related items into single sentences. Use the same language as the input.\n\
        Target length: under 1500 characters. Be extremely concise — telegraphic style is fine.\n\
        No preamble, no headers, no bullet points — flowing prose paragraphs only.";

    let summary = match ai::llm_chat_as(cfg, "gate", system, &input).await {
        Ok(r) => {
            if let Some(ref u) = r.usage {
                let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
                let _ = db.log_llm_call("core_summary", &r.model, u.prompt_tokens, u.completion_tokens, cached, r.duration_ms);
            }
            r.content.trim().to_string()
        }
        Err(e) => {
            warn!(error = %e, "core summary generation failed");
            return;
        }
    };

    if summary.is_empty() || summary.len() > input.len() {
        warn!(summary_len = summary.len(), input_len = input.len(),
            "core summary invalid (empty or longer than input)");
        return;
    }

    // Hard cap: if LLM ignored the length instruction, truncate at sentence boundary
    let summary = if summary.len() > 2000 {
        let safe = crate::util::truncate_chars(&summary, 2000);
        let boundary = safe.rfind(['\u{3002}', '.', '\n'])
            .unwrap_or(safe.len().saturating_sub(3));
        format!("{}…", &safe[..boundary])
    } else {
        summary
    };

    let db4 = db.clone();
    let s = summary.clone();
    let _ = tokio::task::spawn_blocking(move || {
        let _ = db4.set_meta("core_summary", &s);
        let _ = db4.set_meta("core_summary_hash", &h);
    }).await;

    info!(core_count = core.len(), input_chars = input.len(),
        summary_chars = summary.len(),
        ratio = format!("{:.0}%", summary.len() as f64 / input.len() as f64 * 100.0),
        "updated core summary cache");
}
