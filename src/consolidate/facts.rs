use crate::ai::{self, AiConfig};
use crate::SharedDB;
use crate::util::truncate_chars;
use serde::Deserialize;
use tracing::{info, warn};

#[allow(dead_code)]
const FACT_EXTRACT_PROMPT: &str = "Extract factual triples from this memory text. \
    A triple is (subject, predicate, object) representing a concrete, stable relationship.\n\
    Examples: (user, prefers, dark mode), (engram, uses, SQLite), (project, language, Rust)\n\n\
    Rules:\n\
    - Only extract concrete, stable facts — NOT transient states or opinions\n\
    - Subject/object should be short noun phrases (1-3 words)\n\
    - Predicate should be a verb or relationship label\n\
    - Skip if the text is purely procedural/operational with no factual content";

#[derive(Deserialize)]
struct FactsResult {
    facts: Vec<FactTriple>,
}

#[derive(Deserialize)]
struct FactTriple {
    subject: String,
    predicate: String,
    object: String,
}

#[allow(dead_code)]
fn fact_extract_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "description": "Extracted triples. Empty array if nothing to extract.",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string", "description": "Short noun phrase (1-3 words)"},
                        "predicate": {"type": "string", "description": "Verb or relationship label"},
                        "object": {"type": "string", "description": "Short noun phrase (1-3 words)"}
                    },
                    "required": ["subject", "predicate", "object"]
                }
            }
        },
        "required": ["facts"]
    })
}

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
        let result: ai::ToolCallResult<FactsResult> = match ai::llm_tool_call(
            cfg, "extract", FACT_EXTRACT_PROMPT, &content,
            "extract_facts", "Extract factual triples from the text",
            fact_extract_schema(),
        ).await {
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

        let triples = result.value.facts;

        if triples.is_empty() {
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
            let input = crate::db::FactInput {
                subject: t.subject.clone(),
                predicate: t.predicate.clone(),
                object: t.object.clone(),
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
