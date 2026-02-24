use crate::ai::{self, AiConfig};
use crate::SharedDB;
use crate::util::truncate_chars;
use tracing::{info, warn};

#[allow(dead_code)]
const FACT_EXTRACT_PROMPT: &str = "Extract factual triples from this memory text. \
    A triple is (subject, predicate, object) representing a concrete, stable relationship.\n\
    Examples: (user, prefers, dark mode), (engram, uses, SQLite), (project, language, Rust)\n\n\
    Rules:\n\
    - Only extract concrete, stable facts — NOT transient states or opinions\n\
    - Subject/object should be short noun phrases (1-3 words)\n\
    - Predicate should be a verb or relationship label\n\
    - Skip if the text is purely procedural/operational with no factual content\n\n\
    Output a JSON array of objects: [{\"subject\": \"...\", \"predicate\": \"...\", \"object\": \"...\"}]\n\
    Return [] if no facts can be extracted. Output ONLY the JSON array.";

#[allow(dead_code)]
pub(super) async fn extract_facts_batch(db: &SharedDB, cfg: &AiConfig, limit: usize) -> usize {
    let db2 = db.clone();
    let mems = match tokio::task::spawn_blocking(move || {
        db2.memories_without_facts("default", limit)
    }).await {
        Ok(Ok(mems)) => mems,
        _ => return 0,
    };

    if mems.is_empty() {
        return 0;
    }

    let mut total = 0;
    for mem in &mems {
        let content = truncate_chars(&mem.content, 500);
        let result = match ai::llm_chat_as(cfg, "extract", FACT_EXTRACT_PROMPT, &content).await {
            Ok(r) => r,
            Err(e) => {
                warn!(error = %e, id = %mem.id, "fact extraction LLM failed");
                continue;
            }
        };

        if let Some(ref u) = result.usage {
            let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
            let db_log = db.clone();
            let model = result.model.clone();
            let dur = result.duration_ms;
            let inp = u.prompt_tokens;
            let out = u.completion_tokens;
            let _ = tokio::task::spawn_blocking(move || {
                db_log.log_llm_call("facts_extract", &model, inp, out, cached, dur)
            }).await;
        }

        let json_str = crate::ai::unwrap_json(&result.content);
        let triples: Vec<serde_json::Value> = match serde_json::from_str(&json_str) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if triples.is_empty() {
            // Insert a sentinel fact so we don't retry this memory every cycle
            let sentinel = crate::db::FactInput {
                subject: "_no_facts".into(),
                predicate: "_scanned".into(),
                object: crate::util::short_id(&mem.id).to_string(),
                memory_id: Some(mem.id.clone()),
                valid_from: None,
            };
            let db2 = db.clone();
            let ns = mem.namespace.clone();
            let _ = tokio::task::spawn_blocking(move || db2.insert_fact(sentinel, &ns)).await;
            continue;
        }

        let mut count = 0;
        for t in &triples {
            let (Some(subj), Some(pred), Some(obj)) = (
                t.get("subject").and_then(|v| v.as_str()),
                t.get("predicate").and_then(|v| v.as_str()),
                t.get("object").and_then(|v| v.as_str()),
            ) else { continue };

            let input = crate::db::FactInput {
                subject: subj.to_string(),
                predicate: pred.to_string(),
                object: obj.to_string(),
                memory_id: Some(mem.id.clone()),
                valid_from: None,
            };
            let db2 = db.clone();
            let ns = mem.namespace.clone();
            match tokio::task::spawn_blocking(move || db2.insert_fact(input, &ns)).await {
                Ok(Ok(_)) => count += 1,
                Ok(Err(e)) => warn!(error = %e, "fact insert failed"),
                Err(e) => warn!(error = %e, "fact insert task panicked"),
            }
        }

        if count > 0 {
            info!(id = %mem.id, count, "extracted facts from memory");
            total += count;
        }
    }

    total
}
