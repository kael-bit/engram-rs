//! Talks to OpenAI-compatible APIs for embeddings and LLM calls.
//! All optional — see AiConfig::from_env().

use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::debug;

const AI_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone)]
pub struct AiConfig {
    pub llm_url: String,
    pub llm_key: String,
    pub llm_model: String,
    pub embed_url: String,
    pub embed_key: String,
    pub embed_model: String,
    pub client: reqwest::Client,
    // Per-component model overrides (fall back to llm_model if None)
    pub merge_model: Option<String>,
    pub extract_model: Option<String>,
    pub rerank_model: Option<String>,
    pub expand_model: Option<String>,
    pub proxy_model: Option<String>,
}

impl AiConfig {
    pub fn model_for(&self, component: &str) -> &str {
        let m = match component {
            "merge" => self.merge_model.as_deref(),
            "extract" => self.extract_model.as_deref(),
            "rerank" => self.rerank_model.as_deref(),
            "expand" => self.expand_model.as_deref(),
            "proxy" => self.proxy_model.as_deref(),
            _ => None,
        };
        m.unwrap_or(&self.llm_model)
    }

    /// Returns `None` if `ENGRAM_LLM_URL` is not set.
    pub fn from_env() -> Option<Self> {
        let llm_url = std::env::var("ENGRAM_LLM_URL").ok()?;
        let llm_key = std::env::var("ENGRAM_LLM_KEY").unwrap_or_default();
        let llm_model =
            std::env::var("ENGRAM_LLM_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());
        let embed_url = std::env::var("ENGRAM_EMBED_URL").unwrap_or_else(|_| {
            // Only rewrite if this looks like a chat completions endpoint
            if llm_url.contains("/chat/completions") {
                llm_url.replace("/chat/completions", "/embeddings")
            } else {
                format!("{}/embeddings", llm_url.trim_end_matches('/'))
            }
        });
        let embed_key =
            std::env::var("ENGRAM_EMBED_KEY").unwrap_or_else(|_| llm_key.clone());
        let embed_model = std::env::var("ENGRAM_EMBED_MODEL")
            .unwrap_or_else(|_| "text-embedding-3-small".into());

        let client = reqwest::Client::builder()
            .timeout(AI_TIMEOUT)
            .build()
            .expect("failed to build HTTP client");

        Some(Self {
            llm_url,
            llm_key,
            llm_model,
            embed_url,
            embed_key,
            embed_model,
            client,
            merge_model: std::env::var("ENGRAM_MERGE_MODEL").ok(),
            extract_model: std::env::var("ENGRAM_EXTRACT_MODEL").ok(),
            rerank_model: std::env::var("ENGRAM_RERANK_MODEL").ok(),
            expand_model: std::env::var("ENGRAM_EXPAND_MODEL").ok(),
            proxy_model: std::env::var("ENGRAM_PROXY_MODEL").ok(),
        })
    }

    pub fn has_llm(&self) -> bool {
        !self.llm_url.is_empty()
    }

    pub fn has_embed(&self) -> bool {
        !self.embed_url.is_empty()
    }
}


#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f64,
}

#[derive(Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

/// Send a chat completion request, return the response text.
#[allow(dead_code)]
pub async fn llm_chat(cfg: &AiConfig, system: &str, user: &str) -> Result<String, String> {
    llm_chat_as(cfg, "", system, user).await
}

/// Like llm_chat but uses a component-specific model if configured.
pub async fn llm_chat_as(cfg: &AiConfig, component: &str, system: &str, user: &str) -> Result<String, String> {
    let model = cfg.model_for(component).to_string();
    let req = ChatRequest {
        model,
        messages: vec![
            ChatMessage { role: "system".into(), content: system.into() },
            ChatMessage { role: "user".into(), content: user.into() },
        ],
        temperature: 0.1,
    };

    let mut builder = cfg.client.post(&cfg.llm_url).json(&req);
    if !cfg.llm_key.is_empty() {
        builder = builder.header("Authorization", format!("Bearer {}", cfg.llm_key));
    }

    let resp = builder
        .send()
        .await
        .map_err(|e| format!("LLM request failed: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("LLM returned {status}: {body}"));
    }

    let chat: ChatResponse = resp
        .json()
        .await
        .map_err(|e| format!("LLM response parse failed: {e}"))?;
    Ok(chat
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .unwrap_or_default())
}

const EXPAND_PROMPT: &str = "Given a search query for a PERSONAL knowledge base (notes, decisions, logs), \
    generate 3-5 alternative search phrases that would help find relevant stored notes. \
    Bridge abstraction levels: abstract→concrete, concrete→abstract. \
    Example: 可观测性 → 日志系统 Loki Grafana 监控告警. \
    Focus on rephrasing the INTENT, not listing random related technologies. \
    If the query asks about a tool/library choice, rephrase as: why/decision/migration/选择/替换. \
    Output one phrase per line, no numbering. Same language as input.";

/// Generate alternative query phrasings for better recall coverage.
pub async fn expand_query(cfg: &AiConfig, query: &str) -> Vec<String> {
    match llm_chat_as(cfg, "expand", EXPAND_PROMPT, query).await {
        Ok(text) => text
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty() && l.len() > 2)
            .take(5)
            .collect(),
        Err(_) => vec![],
    }
}


const EXTRACT_SYSTEM_PROMPT: &str = r#"You are a memory extraction engine. Given a conversation or text, extract important facts, decisions, preferences, and events as discrete memory entries.

For each memory, output a JSON array of objects with these fields:
- "content": the memory text (concise, self-contained, one fact per entry)
- "tags": relevant keyword tags (array of strings)

Rules:
- Extract 0-3 entries per input. Zero is fine if nothing is worth remembering.
- Each entry must be self-contained (understandable without context)
- Prefer concise entries (under 200 chars) over verbose ones
- Write content in the same language as the input

EXTRACT these (worth remembering):
- Identity: who someone is, their preferences, principles, personality
- Decisions: choices made and why, trade-offs considered
- Lessons: mistakes, insights, things that worked or didn't
- Relationships: facts about people, how they relate to each other
- Strategic: goals, plans, architectural choices

SKIP these (not worth remembering):
- System prompts, instructions, templates, or configuration that appears in every conversation
- Heartbeat checks, health status, routine monitoring output
- Operational details: bug fixes, version bumps, deployment steps, code changes
- Transient states: "service is running", "memory at 33%", "tests passing"
- Debug info, log output, error messages
- Summaries or recaps of work done (these are session logs, not memories)
- Anything that looks like it was injected by a framework rather than said by a human

Output ONLY the JSON array, no other text. Return [] if nothing is worth extracting."#;

/// Extract structured memories from raw text using an LLM.
pub async fn extract_memories(
    cfg: &AiConfig,
    text: &str,
) -> Result<Vec<ExtractedMemory>, String> {
    debug!(model = %cfg.model_for("extract"), "extracting memories");

    let raw = llm_chat_as(cfg, "extract", EXTRACT_SYSTEM_PROMPT, text).await?;
    let json_str = unwrap_json(&raw);
    let memories: Vec<ExtractedMemory> = serde_json::from_str(&json_str)
        .map_err(|e| format!("failed to parse extracted memories: {e}\nraw: {raw}"))?;

    debug!(count = memories.len(), "extracted memories from text");
    Ok(memories)
}

/// Extract a JSON array from LLM output that may be wrapped in markdown code blocks.
fn unwrap_json(raw: &str) -> String {
    let trimmed = raw.trim();
    if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            return trimmed[start..=end].to_string();
        }
    }
    trimmed.to_string()
}

/// A memory extracted by the LLM.
#[derive(Debug, Serialize, Deserialize)]
pub struct ExtractedMemory {
    pub content: String,
    pub importance: Option<f64>,
    pub tags: Option<Vec<String>>,
    pub layer: Option<u8>,
}


#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
struct EmbedData {
    embedding: Vec<f64>,
}

/// Generate embeddings for one or more texts.
pub async fn get_embeddings(
    cfg: &AiConfig,
    texts: &[String],
) -> Result<Vec<Vec<f64>>, String> {
    if texts.is_empty() {
        return Ok(vec![]);
    }

    let req = EmbedRequest {
        model: cfg.embed_model.clone(),
        input: texts.to_vec(),
    };

    let mut builder = cfg.client.post(&cfg.embed_url).json(&req);
    if !cfg.embed_key.is_empty() {
        builder = builder.header("Authorization", format!("Bearer {}", cfg.embed_key));
    }

    let resp = builder
        .send()
        .await
        .map_err(|e| format!("embedding request failed: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("embedding API returned {status}: {body}"));
    }

    let embed_resp: EmbedResponse = resp
        .json()
        .await
        .map_err(|e| format!("embedding response parse failed: {e}"))?;

    let embeddings: Vec<Vec<f64>> = embed_resp.data.into_iter().map(|d| d.embedding).collect();
    if embeddings.len() != texts.len() {
        return Err(format!(
            "embedding count mismatch: sent {} texts, got {} embeddings",
            texts.len(),
            embeddings.len()
        ));
    }
    Ok(embeddings)
}


/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Serialize an f64 vector to bytes (little-endian) for SQLite BLOB storage.
pub fn embedding_to_bytes(v: &[f64]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(v.len() * 8);
    for &f in v {
        buf.extend_from_slice(&f.to_le_bytes());
    }
    buf
}

/// Deserialize bytes back to an f64 vector.
pub fn bytes_to_embedding(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8)
        .map(|chunk| {
            let arr: [u8; 8] = chunk.try_into().expect("chunks_exact guarantees 8 bytes");
            f64::from_le_bytes(arr)
        })
        .collect()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_same_vec() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_perpendicular() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn cosine_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn embedding_roundtrip() {
        let original = vec![1.0, -2.5, 3.125, 0.0, f64::MAX];
        let bytes = embedding_to_bytes(&original);
        let decoded = bytes_to_embedding(&bytes);
        assert_eq!(original, decoded);
    }

    #[test]
    fn unwrap_json_from_markdown() {
        let raw = "```json\n[{\"content\": \"test\"}]\n```";
        let result = unwrap_json(raw);
        assert_eq!(result, "[{\"content\": \"test\"}]");
    }

    #[test]
    fn unwrap_json_bare() {
        let raw = "[{\"content\": \"test\"}]";
        let result = unwrap_json(raw);
        assert_eq!(result, raw);
    }

    #[test]
    fn unwrap_json_with_leading_text() {
        let raw = "Here are the results:\n```json\n[1, 2, 3]\n```\nDone!";
        let result = unwrap_json(raw);
        assert_eq!(result, "[1, 2, 3]");
    }

    #[test]
    fn unwrap_json_no_array() {
        // Without brackets, returns trimmed input
        let raw = "  {\"key\": \"value\"}  ";
        let result = unwrap_json(raw);
        assert_eq!(result, "{\"key\": \"value\"}");
    }

    #[test]
    fn expand_output_parsing() {
        // Simulates what expand_query does to LLM output
        let raw = "连接池实现方案\nr2d2 SQLite pool\n\nab\n数据库并发访问\n";
        let parsed: Vec<String> = raw
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty() && l.len() > 2)
            .take(5)
            .collect();
        assert_eq!(parsed, vec![
            "连接池实现方案",
            "r2d2 SQLite pool",
            "数据库并发访问",
        ]);
        // "ab" filtered out (len <= 2)
    }
}
