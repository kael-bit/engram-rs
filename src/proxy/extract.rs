use tracing::{debug, info, warn};

use crate::{ai, db, error::EngramError, prompts, thresholds, util::truncate_chars, AppState};
use super::stats::{bump_extracted, persist_proxy_counters};

pub(crate) async fn extract_from_context(state: AppState, context: &str) {
    let Some(ref ai_cfg) = state.ai else {
        return;
    };

    let turn_count = context.matches("User: ").count();
    info!(
        turns = turn_count,
        chars = context.len(),
        "proxy: extraction window flushed"
    );

    if context.len() < 100 {
        info!("proxy: context too thin to extract from ({} chars)", context.len());
        return;
    }

    let preview = truncate_chars(context, 12000);

    let db = state.db.clone();
    let db2 = db.clone();
    let recent_mems = tokio::task::spawn_blocking(move || {
        db2.list_since(db::now_ms() - 6 * 3600 * 1000, 30)
    }).await.unwrap_or(Ok(vec![])).unwrap_or_default();
    let existing_knowledge = if recent_mems.is_empty() {
        String::new()
    } else {
        let lines: Vec<String> = recent_mems.iter()
            .map(|m| {
                let truncated = truncate_chars(&m.content, 120);
                format!("- {}", truncated)
            })
            .collect();
        format!("\n\n=== ALREADY IN MEMORY (do NOT extract anything that overlaps with these) ===\n{}\n", lines.join("\n"))
    };

    let system = prompts::PROXY_EXTRACT_SYSTEM;

    let user = format!(
        "Extract memories from this conversation window. Call with empty items if nothing qualifies.{existing_knowledge}\n\n\
         === CONVERSATION ({} turns) ===\n{preview}",
        context.matches("User: ").count()
    );

    let schema = prompts::proxy_extract_schema();

    #[derive(serde::Deserialize)]
    struct ExtractResult {
        #[serde(default)]
        items: Vec<ExtractionEntry>,
    }

    let entries: Vec<ExtractionEntry> = match ai::llm_tool_call::<ExtractResult>(
        ai_cfg, "proxy", system, &user,
        "store_memories",
        "Store extracted memories from conversation",
        schema,
    ).await {
        Ok(tcr) => {
            if let Some(ref u) = tcr.usage {
                let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
                let _ = db.log_llm_call("proxy_extract", &tcr.model, u.prompt_tokens, u.completion_tokens, cached, tcr.duration_ms);
            }
            tcr.value.items
        }
        Err(e) => {
            warn!("proxy: extraction failed: {e}");
            return;
        }
    };

    if entries.is_empty() {
        info!("proxy: extraction returned 0 items (correct for most windows)");
        return;
    }

    for entry in &entries {
        info!(
            kind = entry.kind.as_deref().unwrap_or("none"),
            "proxy: extracted → {}",
            truncate_chars(&entry.content, 100)
        );
    }

    let count = entries.len();
    let mut stored: u64 = 0;
    let mut batch_contents: Vec<String> = Vec::new();
    for entry in entries {
        if entry.content.is_empty() || entry.content.len() > 200 {
            continue;
        }

        if entry.importance.unwrap_or(thresholds::PROXY_MIN_IMPORTANCE) < thresholds::PROXY_MIN_IMPORTANCE {
            debug!("proxy: importance filter skip ({:.2}): {}",
                entry.importance.unwrap_or(0.0), truncate_chars(&entry.content, 60));
            continue;
        }

        // Hard filter: reject code-operation noise that weak models keep extracting
        if is_code_noise(&entry.content) {
            debug!("proxy: code-noise filter skip: {}", truncate_chars(&entry.content, 60));
            continue;
        }

        if batch_contents.iter().any(|prev| {
            crate::db::jaccard_similar(prev, &entry.content, thresholds::PROXY_DEDUP_THRESHOLD)
        }) {
            info!("proxy: dedup/intra-batch skip: {}", truncate_chars(&entry.content, 60));
            continue;
        }

        let dup_id: Option<String> =
            match crate::recall::quick_semantic_dup_threshold(ai_cfg, &state.db, &entry.content, crate::thresholds::PROXY_DEDUP_SIM).await {
                Ok(id) => id,
                Err(_) => {
                    let db = state.db.clone();
                    let c = entry.content.clone();
                    tokio::task::spawn_blocking(move || {
                        if db.is_near_duplicate_with(&c, thresholds::PROXY_DEDUP_THRESHOLD) { Some(String::new()) } else { None }
                    }).await.unwrap_or(None)
                }
            };
        if let Some(ref existing_id) = dup_id {
            if !existing_id.is_empty() {
                let db = state.db.clone();
                let eid = existing_id.clone();
                let _ = tokio::task::spawn_blocking(move || db.reinforce(&eid)).await;
            }
            info!(
                "proxy: dedup/semantic skip (reinforced): {}",
                truncate_chars(&entry.content, 60)
            );
            continue;
        }

        let mut tags = entry.tags;
        tags.push("auto-extract".into());

        let content_copy = entry.content.clone();
        let facts_input = entry.facts;
        let input = db::MemoryInput {
            content: entry.content,
            importance: entry.importance,
            source: Some("proxy".into()),
            tags: Some(tags),
            kind: entry.kind,
            ..Default::default()
        };

        let db = state.db.clone();
        match tokio::task::spawn_blocking(move || db.insert(input)).await {
            Ok(Ok(mem)) => {
                if let Some(facts) = facts_input {
                    if !facts.is_empty() {
                        let linked: Vec<db::FactInput> = facts.into_iter().map(|mut f| {
                            f.memory_id = Some(mem.id.clone());
                            f
                        }).collect();
                        let db2 = state.db.clone();
                        let ns = mem.namespace.clone();
                        if let Err(e) = tokio::task::spawn_blocking(move || db2.insert_facts(linked, &ns)).await.unwrap_or(Err(EngramError::Internal("spawn failed".into()))) {
                            warn!("proxy: failed to store facts: {e}");
                        }
                    }
                }
                batch_contents.push(content_copy);
                stored += 1;
            }
            Ok(Err(e)) => {
                warn!("proxy: failed to store: {e}");
            }
            Err(e) => {
                warn!("proxy: spawn_blocking failed: {e}");
            }
        }
    }

    if stored > 0 {
        bump_extracted(stored);
        persist_proxy_counters(&db);
        info!(stored, extracted = count, "proxy: stored memories from window");
    } else if count > 0 {
        debug!("proxy: all extractions were duplicates");
    }
}

#[derive(serde::Deserialize)]
struct ExtractionEntry {
    content: String,
    #[serde(default)]
    tags: Vec<String>,
    importance: Option<f64>,
    #[serde(default)]
    facts: Option<Vec<db::FactInput>>,
    #[serde(default)]
    kind: Option<String>,
}

/// Reject code-operation noise that LLMs keep extracting despite prompt instructions.
/// Returns true if the content looks like a routine implementation detail.
pub fn is_code_noise(content: &str) -> bool {
    let lower = content.to_lowercase();

    // Code-level implementation details
    let noise_signals = [
        "pub(crate)", "pub fn", "#[cfg(test)]", "#[test]",
        "impl ", "struct ", "enum ", "trait ",
        "cargo test", "cargo build", "cargo check",
        "clippy", "compile", "compilation",
        ".rs:", ".rs`", "mod.rs", "lib.rs", "main.rs",
        "commit ", "merge conflict",
        "test pass", "tests pass", "test fail",
        "refactor", "visibility", "accessibility",
        "doc-test", "inline test",
        "replace", "rename", "moved to", "split into",
        "magic number", "dead code", "unused import",
        "parse_", "extract_", "resolve_",
        "codebase", "source code", "implementation",
        "the mutex", "the rwlock", "the pool",
    ];

    let hit_count = noise_signals.iter()
        .filter(|s| lower.contains(*s))
        .count();

    // 2+ noise signals = almost certainly code noise
    if hit_count >= 2 {
        return true;
    }

    // Single signal + no lesson/decision markers = likely noise
    if hit_count == 1 {
        let has_value_signal = lower.contains("lesson")
            || lower.contains("decision")
            || lower.contains("never ")
            || lower.contains("always ")
            || lower.contains("don't ")
            || lower.contains("不要")
            || lower.contains("必须")
            || lower.contains("禁止");
        return !has_value_signal;
    }

    false
}
