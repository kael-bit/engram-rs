//! Optional AI backend integration (OpenAI-compatible API).
//! Provides embedding generation and LLM-based memory extraction.
//! engram works without this module — pure FTS is the fallback.

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
}

impl AiConfig {
    /// Returns `None` if `ENGRAM_LLM_URL` is not set.
    pub fn from_env() -> Option<Self> {
        let llm_url = std::env::var("ENGRAM_LLM_URL").ok()?;
        let llm_key = std::env::var("ENGRAM_LLM_KEY").unwrap_or_default();
        let llm_model =
            std::env::var("ENGRAM_LLM_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());
        let embed_url = std::env::var("ENGRAM_EMBED_URL")
            .unwrap_or_else(|_| llm_url.replace("/chat/completions", "/embeddings"));
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
pub async fn llm_chat(cfg: &AiConfig, system: &str, user: &str) -> Result<String, String> {
    let req = ChatRequest {
        model: cfg.llm_model.clone(),
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


const EXTRACT_SYSTEM_PROMPT: &str = r#"You are a memory extraction engine. Given a conversation or text, extract important facts, decisions, preferences, and events as discrete memory entries.

For each memory, output a JSON array of objects with these fields:
- "content": the memory text (concise, self-contained, one fact per entry)
- "importance": 0.0-1.0 (1.0 = critical identity/principle, 0.5 = normal fact, 0.1 = trivial)
- "tags": relevant keyword tags (array of strings)
- "layer": suggested layer (1=buffer/temporary, 2=working/recent, 3=core/permanent)

Rules:
- Extract 1-10 entries per input
- Each entry must be self-contained (understandable without context)
- Prefer concise entries over verbose ones
- Merge redundant information
- Write content in the same language as the input
- Skip trivial metadata (counts, sizes, timestamps) unless they carry meaningful context
- Assign higher importance to: identity, preferences, decisions, relationships, lessons learned
- Assign lower importance to: debug info, transient states, routine actions, version snapshots
- Output ONLY the JSON array, no other text"#;

/// Extract structured memories from raw text using an LLM.
pub async fn extract_memories(
    cfg: &AiConfig,
    text: &str,
) -> Result<Vec<ExtractedMemory>, String> {
    debug!(model = %cfg.llm_model, "extracting memories");

    let raw = llm_chat(cfg, EXTRACT_SYSTEM_PROMPT, text).await?;
    let json_str = extract_json_from_response(&raw);
    let memories: Vec<ExtractedMemory> = serde_json::from_str(&json_str)
        .map_err(|e| format!("failed to parse extracted memories: {e}\nraw: {raw}"))?;

    debug!(count = memories.len(), "extracted memories from text");
    Ok(memories)
}

/// Extract a JSON array from LLM output that may be wrapped in markdown code blocks.
fn extract_json_from_response(raw: &str) -> String {
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

    Ok(embed_resp.data.into_iter().map(|d| d.embedding).collect())
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
    fn cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn cosine_empty_vectors() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn embedding_roundtrip() {
        let original = vec![1.0, -2.5, 3.14159, 0.0, f64::MAX];
        let bytes = embedding_to_bytes(&original);
        let decoded = bytes_to_embedding(&bytes);
        assert_eq!(original, decoded);
    }

    #[test]
    fn extract_json_from_markdown_block() {
        let raw = "```json\n[{\"content\": \"test\"}]\n```";
        let result = extract_json_from_response(raw);
        assert_eq!(result, "[{\"content\": \"test\"}]");
    }

    #[test]
    fn extract_json_bare_array() {
        let raw = "[{\"content\": \"test\"}]";
        let result = extract_json_from_response(raw);
        assert_eq!(result, raw);
    }
}
