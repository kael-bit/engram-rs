//! Talks to OpenAI-compatible APIs for embeddings and LLM calls.
//! All optional — see AiConfig::from_env().

use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::debug;

use crate::error::EngramError;

fn ai_err(msg: impl Into<String>) -> EngramError {
    EngramError::AiBackend(msg.into())
}

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
    pub gate_model: Option<String>,
}

impl AiConfig {
    pub fn model_for(&self, component: &str) -> &str {
        let m = match component {
            "merge" => self.merge_model.as_deref(),
            "extract" => self.extract_model.as_deref(),
            "rerank" => self.rerank_model.as_deref(),
            "expand" => self.expand_model.as_deref(),
            "proxy" => self.proxy_model.as_deref(),
            "gate" => self.gate_model.as_deref(),
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
            gate_model: std::env::var("ENGRAM_GATE_MODEL").ok(),
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
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDef>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ToolDef {
    #[serde(rename = "type")]
    tool_type: String,
    function: FunctionDef,
}

#[derive(Serialize)]
struct FunctionDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: u32,
}

pub struct LlmResult {
    pub content: String,
    pub usage: Option<Usage>,
    pub model: String,
    pub duration_ms: u64,
}

pub struct ToolCallResult<T> {
    pub value: T,
    pub usage: Option<Usage>,
    pub model: String,
    pub duration_ms: u64,
}

pub struct EmbedResult {
    pub embeddings: Vec<Vec<f32>>,
    pub usage: Option<Usage>,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Deserialize)]
struct ToolCall {
    function: ToolCallFunction,
}

#[derive(Deserialize)]
struct ToolCallFunction {
    arguments: String,
}

/// Send a chat completion request, return the response text.
#[allow(dead_code)]
pub async fn llm_chat(cfg: &AiConfig, system: &str, user: &str) -> Result<String, EngramError> {
    Ok(llm_chat_as(cfg, "", system, user).await?.content)
}

/// Like llm_chat but uses a component-specific model if configured.
/// Returns LlmResult with usage stats and model info.
pub async fn llm_chat_as(cfg: &AiConfig, component: &str, system: &str, user: &str) -> Result<LlmResult, EngramError> {
    let model = cfg.model_for(component).to_string();
    let req = ChatRequest {
        model: model.clone(),
        messages: vec![
            ChatMessage { role: "system".into(), content: system.into() },
            ChatMessage { role: "user".into(), content: user.into() },
        ],
        temperature: 0.1,
        tools: None,
        tool_choice: None,
    };

    let mut builder = cfg.client.post(&cfg.llm_url).json(&req);
    if !cfg.llm_key.is_empty() {
        builder = builder.header("Authorization", format!("Bearer {}", cfg.llm_key));
    }

    let start = std::time::Instant::now();
    let resp = builder
        .send()
        .await
        .map_err(|e| ai_err(format!("LLM request failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(ai_err(format!("LLM returned {status}: {body}")));
    }

    let chat: ChatResponse = resp
        .json()
        .await
        .map_err(|e| ai_err(format!("LLM response parse failed: {e}")))?;
    let duration_ms = start.elapsed().as_millis() as u64;
    let content = chat
        .choices
        .first()
        .and_then(|c| c.message.content.clone())
        .unwrap_or_default();
    Ok(LlmResult { content, usage: chat.usage, model, duration_ms })
}

/// Call LLM with a function/tool definition, get back structured JSON.
/// Forces the model to call the named function, returns the parsed arguments with usage.
pub async fn llm_tool_call<T: serde::de::DeserializeOwned>(
    cfg: &AiConfig,
    component: &str,
    system: &str,
    user: &str,
    fn_name: &str,
    fn_desc: &str,
    parameters: serde_json::Value,
) -> Result<ToolCallResult<T>, EngramError> {
    let model = cfg.model_for(component).to_string();
    let req = ChatRequest {
        model: model.clone(),
        messages: vec![
            ChatMessage { role: "system".into(), content: system.into() },
            ChatMessage { role: "user".into(), content: user.into() },
        ],
        temperature: 0.1,
        tools: Some(vec![ToolDef {
            tool_type: "function".into(),
            function: FunctionDef {
                name: fn_name.into(),
                description: fn_desc.into(),
                parameters,
            },
        }]),
        tool_choice: Some(serde_json::json!({"type": "function", "function": {"name": fn_name}})),
    };

    let mut builder = cfg.client.post(&cfg.llm_url).json(&req);
    if !cfg.llm_key.is_empty() {
        builder = builder.header("Authorization", format!("Bearer {}", cfg.llm_key));
    }

    let start = std::time::Instant::now();
    let resp = builder
        .send()
        .await
        .map_err(|e| ai_err(format!("LLM tool call failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(ai_err(format!("LLM returned {status}: {body}")));
    }

    let chat: ChatResponse = resp
        .json()
        .await
        .map_err(|e| ai_err(format!("LLM tool response parse failed: {e}")))?;
    let duration_ms = start.elapsed().as_millis() as u64;

    let args = chat.choices.first()
        .and_then(|c| c.message.tool_calls.as_ref())
        .and_then(|tc| tc.first())
        .map(|tc| tc.function.arguments.clone())
        .ok_or_else(|| ai_err("no tool call in response"))?;

    let value: T = serde_json::from_str(&args)
        .map_err(|e| ai_err(format!("tool call arguments parse failed: {e}: {args}")))?;

    Ok(ToolCallResult { value, usage: chat.usage, model, duration_ms })
}

const EXPAND_PROMPT: &str = "Given a search query for a PERSONAL knowledge base (notes, decisions, logs), \
    generate 4-6 alternative search phrases that would help find relevant stored notes. \
    Bridge abstraction levels: abstract→concrete, concrete→abstract. \
    \
    CRITICAL: The knowledge base contains BOTH Chinese and English notes. \
    You MUST include expansions in BOTH languages regardless of query language. \
    \
    Examples: \
    who am I → my identity, my identity and role, who am I, my name and positioning, identity bootstrap. \
    security lessons → security lesson, security mistakes and lessons, security discipline. \
    部署 → deploy procedure, 部署流程和步骤, deployment workflow, systemd 部署. \
    task delegation workflow → task specifications, task delegation workflow, subagent best practices. \
    GitHub配置 → GitHub SSH setup, GitHub 仓库和账号, repo migration. \
    \
    Focus on rephrasing the INTENT, not listing random related technologies. \
    If the query asks about a tool/library choice, rephrase as: why/decision/migration/选择/替换. \
    NEVER output explanations, commentary, or bullet points with dashes. \
    Even for very short queries (1-2 words), always produce at least 4 phrases.";

/// Generate alternative query phrasings for better recall coverage.
/// Returns (expansions, ToolCallResult metadata) for optional usage logging.
pub async fn expand_query(cfg: &AiConfig, query: &str) -> (Vec<String>, Option<ToolCallResult<()>>) {
    #[derive(Deserialize)]
    struct ExpandResult { queries: Vec<String> }

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Alternative search phrases (3-5)"
            }
        },
        "required": ["queries"]
    });

    match llm_tool_call::<ExpandResult>(
        cfg, "expand", EXPAND_PROMPT, query,
        "expanded_queries", "Generate alternative search phrases for the query",
        schema,
    ).await {
        Ok(tcr) => {
            let queries: Vec<String> = tcr.value.queries.into_iter()
                .filter(|q| !q.is_empty() && q.len() > 2)
                .take(5)
                .collect();
            let meta = ToolCallResult { value: (), usage: tcr.usage, model: tcr.model, duration_ms: tcr.duration_ms };
            (queries, Some(meta))
        }
        Err(_) => (vec![], None),
    }
}


const EXTRACT_SYSTEM_PROMPT: &str = r#"You are a memory extraction engine. Given a conversation or text, extract important facts, decisions, preferences, and events as discrete memory entries.

For each memory, output a JSON array of objects with these fields:
- "content": the memory text (concise, self-contained, one fact per entry)
- "tags": relevant keyword tags (array of strings)
- "importance": float 0.0-1.0, how important this is to remember long-term

Importance scale:
- 0.9-1.0: User EXPLICITLY asked to remember this ("记住", "remember this", "don't forget"). Core knowledge.
- 0.7-0.8: Significant decisions, strong preferences, lessons learned, identity-defining facts. Worth keeping.
- 0.4-0.6: Useful context, minor preferences, background info. May fade if not reinforced.

Rules:
- Extract 0-3 entries per input. Zero is fine if nothing is worth remembering.
- Each entry must be self-contained (understandable without context)
- Prefer concise entries (under 200 chars) over verbose ones
- Write content in the same language as the input. NEVER translate — if the conversation is in Chinese, output Chinese. If mixed, use the dominant language.
- importance MUST reflect user intent — if they say "记住" or "remember", it's 0.9+

EXTRACT these (worth remembering):
- Identity: who someone is, their preferences, principles, personality
- Decisions: choices made and why, trade-offs considered
- Lessons: mistakes, insights, things that worked or didn't
- Self-reflections: realizations about own behavior patterns, blind spots, habits to change (these are HIGH value — 0.8+)
- Relationships: facts about people, how they relate to each other
- Strategic: goals, plans, architectural choices

SKIP these (not worth remembering):
- Operational details: bug fixes, version bumps, deployment steps, code changes
- Implementation notes: "update X to do Y", "add Z to W", "fix A in B" — these are code tasks, not memories
- Transient states: "service is running", "memory at 33%", "tests passing"
- Debug info, log output, error messages
- Summaries or recaps of work done (these are session logs, not memories) — BUT self-critical reflections about patterns/habits ARE worth extracting
- Instructions from one agent to another (e.g. "also add to proxy", "fix now — add touch")

HARD REJECT — NEVER extract these as memories (they are scaffolding, not knowledge):
- System prompts and injected instructions (content from SOUL.md, AGENTS.md, HEARTBEAT.md, TOOLS.md, USER.md, IDENTITY.md, MEMORY.md, or similar)
- Operational directives: "every heartbeat do X", "run this command on wake", "before shutdown do Y"
- Configuration templates and boilerplate: API keys, curl examples, service names, file paths used as reference
- Tool usage patterns and API call templates: "use this endpoint", "call this script"
- Meta-instructions about how to behave, respond, or format output
- Heartbeat checks, health status pings, routine monitoring output
- Anything that reads like a rule/playbook for an agent rather than a human-stated fact or preference
- Framework-injected context that appears in every conversation"#;

/// Extract structured memories from raw text using an LLM.
pub async fn extract_memories(
    cfg: &AiConfig,
    text: &str,
) -> Result<Vec<ExtractedMemory>, EngramError> {
    debug!(model = %cfg.model_for("extract"), "extracting memories");

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "memories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Concise memory text, under 200 chars"},
                        "importance": {"type": "number", "description": "0.0-1.0 importance score"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "kind": {
                            "type": "string",
                            "enum": ["semantic", "episodic", "procedural"],
                            "description": "semantic=facts, episodic=events, procedural=how-to (never decay)"
                        }
                    },
                    "required": ["content"]
                }
            }
        },
        "required": ["memories"]
    });

    #[derive(serde::Deserialize)]
    struct ExtractResult {
        #[serde(default)]
        memories: Vec<ExtractedMemory>,
    }

    let result = llm_tool_call::<ExtractResult>(
        cfg, "extract", EXTRACT_SYSTEM_PROMPT, text,
        "store_memories",
        "Store extracted memories from text",
        schema,
    ).await?;

    debug!(count = result.value.memories.len(), "extracted memories from text");
    Ok(result.value.memories)
}

/// Extract a JSON array from LLM output that may be wrapped in markdown code blocks.
pub fn unwrap_json(raw: &str) -> String {
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
    pub facts: Option<Vec<crate::db::FactInput>>,
    pub kind: Option<String>,
}


#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct EmbedData {
    embedding: Vec<f32>,
}

/// Generate embeddings for one or more texts.
pub async fn get_embeddings(
    cfg: &AiConfig,
    texts: &[String],
) -> Result<EmbedResult, EngramError> {
    if texts.is_empty() {
        return Ok(EmbedResult { embeddings: vec![], usage: None });
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
        .map_err(|e| ai_err(format!("embedding request failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(ai_err(format!("embedding API returned {status}: {body}")));
    }

    let embed_resp: EmbedResponse = resp
        .json()
        .await
        .map_err(|e| ai_err(format!("embedding response parse failed: {e}")))?;

    let embeddings: Vec<Vec<f32>> = embed_resp.data.into_iter().map(|d| d.embedding).collect();
    if embeddings.len() != texts.len() {
        return Err(ai_err(format!(
            "embedding count mismatch: sent {} texts, got {} embeddings",
            texts.len(),
            embeddings.len()
        )));
    }
    Ok(EmbedResult { embeddings, usage: embed_resp.usage })
}


/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        let (ai, bi) = (a[i] as f64, b[i] as f64);
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Serialize an f32 vector to bytes (little-endian) for SQLite BLOB storage.
pub fn embedding_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(v.len() * 4);
    for &f in v {
        buf.extend_from_slice(&f.to_le_bytes());
    }
    buf
}

/// Deserialize bytes back to an f32 vector.
/// Handles legacy f64 blobs (8 bytes per dim) by down-converting automatically.
pub fn bytes_to_embedding(b: &[u8]) -> Vec<f32> {
    // Legacy f64 blobs are exactly 12288 bytes (1536 × 8). New f32 blobs are 6144 (1536 × 4).
    if b.len() == 1536 * 8 {
        return b.chunks_exact(8)
            .map(|chunk| {
                let arr: [u8; 8] = chunk.try_into().expect("8 bytes");
                f64::from_le_bytes(arr) as f32
            })
            .collect();
    }
    b.chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = chunk.try_into().expect("4 bytes");
            f32::from_le_bytes(arr)
        })
        .collect()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_same_vec() {
        let v: Vec<f32> = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_perpendicular() {
        let a: Vec<f32> = vec![1.0, 0.0];
        let b: Vec<f32> = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn cosine_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn embedding_roundtrip() {
        let original: Vec<f32> = vec![1.0, -2.5, 3.125, 0.0, f32::MAX];
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
        let parse = |raw: &str| -> Vec<String> {
            raw.lines()
                .map(|l| l.trim().trim_start_matches("- ").trim().to_string())
                .filter(|l| {
                    !l.is_empty() && l.len() > 2
                        && !l.contains("过于简洁")
                        && !l.contains("缺乏上下文")
                        && !l.contains("无法生成")
                        && !l.starts_with("这个查询")
                        && !l.starts_with("该查询")
                        && !l.starts_with("注意")
                        && !l.starts_with("Note:")
                })
                .take(5)
                .collect()
        };

        // Normal case
        let parsed = parse("连接池实现方案\nr2d2 SQLite pool\n\nab\n数据库并发访问\n");
        assert_eq!(parsed, vec![
            "连接池实现方案",
            "r2d2 SQLite pool",
            "数据库并发访问",
        ]);

        // Meta-commentary filtered out
        let parsed = parse("这个查询过于简洁，无法生成有意义的替代搜索短语。\n\"alice是谁\" 缺乏上下文，可能指：\n- 某个具体的人名\n- 项目/产品代号\n- 团队成员昵称");
        assert_eq!(parsed, vec![
            "某个具体的人名",
            "项目/产品代号",
            "团队成员昵称",
        ]);

        // Dash-prefixed lines cleaned
        let parsed = parse("- alice的身份信息\n- alice是什么角色\n- 关于alice的描述");
        assert_eq!(parsed, vec![
            "alice的身份信息",
            "alice是什么角色",
            "关于alice的描述",
        ]);
    }
}
