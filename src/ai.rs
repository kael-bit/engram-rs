//! Talks to OpenAI-compatible or Anthropic-native APIs for LLM calls,
//! and OpenAI-compatible APIs for embeddings.
//! All optional — see AiConfig::from_env().

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tracing::debug;

use crate::error::EngramError;

fn ai_err(msg: impl Into<String>) -> EngramError {
    EngramError::AiBackend(msg.into())
}

const AI_TIMEOUT: Duration = Duration::from_secs(30);

/// Which LLM API wire format to use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LlmProvider {
    OpenAI,
    Anthropic,
}

/// Per-component LLM connection overrides. Falls back to global AiConfig defaults.
#[derive(Clone, Debug, Default)]
struct ComponentConfig {
    url: Option<String>,
    key: Option<String>,
    model: Option<String>,
    provider: Option<LlmProvider>,
}

/// Fully resolved LLM connection config for a specific component call.
pub struct ResolvedConfig {
    pub url: String,
    pub key: String,
    pub model: String,
    pub provider: LlmProvider,
}

/// All known LLM component names.
const COMPONENTS: &[&str] = &[
    "gate", "audit", "merge", "extract", "expand", "proxy", "triage", "summary", "naming",
];

/// Read per-component overrides from environment variables.
/// Pattern: `ENGRAM_{COMPONENT}_URL`, `_KEY`, `_MODEL`, `_PROVIDER`.
fn component_config_from_env(component: &str) -> ComponentConfig {
    let prefix = format!("ENGRAM_{}", component.to_uppercase());
    ComponentConfig {
        url: std::env::var(format!("{prefix}_URL")).ok(),
        key: std::env::var(format!("{prefix}_KEY")).ok(),
        model: std::env::var(format!("{prefix}_MODEL")).ok(),
        provider: std::env::var(format!("{prefix}_PROVIDER")).ok().and_then(|v| {
            match v.to_lowercase().as_str() {
                "anthropic" | "claude" => Some(LlmProvider::Anthropic),
                "openai" => Some(LlmProvider::OpenAI),
                _ => None,
            }
        }),
    }
}

#[derive(Clone)]
pub struct AiConfig {
    // Global defaults
    pub provider: LlmProvider,
    pub llm_url: String,
    pub llm_key: String,
    pub llm_model: String,

    // Embedding (always OpenAI format, independent)
    pub embed_url: String,
    pub embed_key: String,
    pub embed_model: String,

    pub client: reqwest::Client,

    // Per-component overrides (accessed via for_component / model_for)
    components: HashMap<String, ComponentConfig>,
}

impl AiConfig {
    /// Get fully resolved LLM config for a component (component-specific > global fallback).
    pub fn for_component(&self, component: &str) -> ResolvedConfig {
        let comp = self.components.get(component);
        ResolvedConfig {
            url: comp.and_then(|c| c.url.clone()).unwrap_or_else(|| self.llm_url.clone()),
            key: comp.and_then(|c| c.key.clone()).unwrap_or_else(|| self.llm_key.clone()),
            model: comp.and_then(|c| c.model.clone()).unwrap_or_else(|| self.llm_model.clone()),
            provider: comp.and_then(|c| c.provider.clone()).unwrap_or_else(|| self.provider.clone()),
        }
    }

    /// Convenience: get model name for a component (backward compat).
    pub fn model_for(&self, component: &str) -> &str {
        self.components
            .get(component)
            .and_then(|c| c.model.as_deref())
            .unwrap_or(&self.llm_model)
    }

    /// Returns `None` if `ENGRAM_LLM_URL` is not set.
    pub fn from_env() -> Option<Self> {
        let llm_url = std::env::var("ENGRAM_LLM_URL").unwrap_or_default();
        let embed_url_explicit = std::env::var("ENGRAM_EMBED_URL").ok();

        // Need at least one of LLM or Embed URL to create config
        if llm_url.is_empty() && embed_url_explicit.is_none() {
            return None;
        }

        let llm_key = std::env::var("ENGRAM_LLM_KEY").unwrap_or_default();
        let llm_model =
            std::env::var("ENGRAM_LLM_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());

        // Determine provider from env, default to openai
        let provider = match std::env::var("ENGRAM_LLM_PROVIDER")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "anthropic" | "claude" => LlmProvider::Anthropic,
            _ => LlmProvider::OpenAI,
        };

        let embed_url = std::env::var("ENGRAM_EMBED_URL").unwrap_or_else(|_| {
            if llm_url.is_empty() {
                String::new()
            } else if llm_url.contains("/chat/completions") {
                llm_url.replace("/chat/completions", "/embeddings")
            } else {
                format!("{}/embeddings", llm_url.trim_end_matches('/'))
            }
        });
        let embed_key = std::env::var("ENGRAM_EMBED_KEY").unwrap_or_else(|_| llm_key.clone());
        let embed_model = std::env::var("ENGRAM_EMBED_MODEL")
            .unwrap_or_else(|_| "text-embedding-3-small".into());

        let client = reqwest::Client::builder()
            .timeout(AI_TIMEOUT)
            .build()
            .expect("failed to build HTTP client");

        // Build per-component configs from env vars
        let components: HashMap<String, ComponentConfig> = COMPONENTS
            .iter()
            .map(|&name| (name.to_string(), component_config_from_env(name)))
            .filter(|(_, c)| c.url.is_some() || c.key.is_some() || c.model.is_some() || c.provider.is_some())
            .collect();

        Some(Self {
            provider,
            llm_url,
            llm_key,
            llm_model,
            embed_url,
            embed_key,
            embed_model,
            client,
            components,
        })
    }

    pub fn has_llm(&self) -> bool {
        !self.llm_url.is_empty()
    }

    pub fn has_embed(&self) -> bool {
        !self.embed_url.is_empty()
    }
}

// ---------------------------------------------------------------------------
// OpenAI wire types
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Anthropic wire types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    temperature: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[derive(Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    input: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: Option<u32>,
}

impl AnthropicUsage {
    fn to_usage(&self) -> Usage {
        let cached = self.cache_read_input_tokens.unwrap_or(0);
        Usage {
            prompt_tokens: self.input_tokens,
            completion_tokens: self.output_tokens,
            total_tokens: self.input_tokens + self.output_tokens,
            prompt_tokens_details: if cached > 0 {
                Some(PromptTokensDetails { cached_tokens: cached })
            } else {
                None
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Helper: add auth headers based on provider
// ---------------------------------------------------------------------------

fn add_auth(resolved: &ResolvedConfig, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    let mut b = builder;
    if !resolved.key.is_empty() {
        match resolved.provider {
            LlmProvider::Anthropic => {
                b = b
                    .header("x-api-key", &resolved.key)
                    .header("anthropic-version", "2023-06-01");
            }
            LlmProvider::OpenAI => {
                b = b.header("Authorization", format!("Bearer {}", resolved.key));
            }
        }
    }
    b
}

// ---------------------------------------------------------------------------
// LLM chat
// ---------------------------------------------------------------------------

/// Like llm_chat but uses a component-specific model if configured.
/// Returns LlmResult with usage stats and model info.
pub async fn llm_chat_as(cfg: &AiConfig, component: &str, system: &str, user: &str) -> Result<LlmResult, EngramError> {
    let resolved = cfg.for_component(component);

    match resolved.provider {
        LlmProvider::Anthropic => llm_chat_anthropic(&cfg.client, &resolved, system, user).await,
        LlmProvider::OpenAI => llm_chat_openai(&cfg.client, &resolved, system, user).await,
    }
}

async fn llm_chat_openai(client: &reqwest::Client, resolved: &ResolvedConfig, system: &str, user: &str) -> Result<LlmResult, EngramError> {
    let req = ChatRequest {
        model: resolved.model.clone(),
        messages: vec![
            ChatMessage { role: "system".into(), content: system.into() },
            ChatMessage { role: "user".into(), content: user.into() },
        ],
        temperature: 0.1,
        tools: None,
        tool_choice: None,
    };

    let builder = add_auth(resolved, client.post(&resolved.url).json(&req));

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
    Ok(LlmResult { content, usage: chat.usage, model: resolved.model.clone(), duration_ms })
}

async fn llm_chat_anthropic(client: &reqwest::Client, resolved: &ResolvedConfig, system: &str, user: &str) -> Result<LlmResult, EngramError> {
    let req = AnthropicRequest {
        model: resolved.model.clone(),
        max_tokens: 4096,
        system: if system.is_empty() { None } else { Some(system.to_string()) },
        messages: vec![AnthropicMessage { role: "user".into(), content: user.into() }],
        temperature: 0.1,
        tools: None,
        tool_choice: None,
    };

    let builder = add_auth(resolved, client.post(&resolved.url).json(&req));

    let start = std::time::Instant::now();
    let resp = builder
        .send()
        .await
        .map_err(|e| ai_err(format!("Anthropic request failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(ai_err(format!("Anthropic returned {status}: {body}")));
    }

    let ar: AnthropicResponse = resp
        .json()
        .await
        .map_err(|e| ai_err(format!("Anthropic response parse failed: {e}")))?;
    let duration_ms = start.elapsed().as_millis() as u64;

    let content = ar.content.iter()
        .filter(|b| b.block_type == "text")
        .filter_map(|b| b.text.as_deref())
        .collect::<Vec<_>>()
        .join("");

    let usage = ar.usage.as_ref().map(|u| u.to_usage());
    Ok(LlmResult { content, usage, model: resolved.model.clone(), duration_ms })
}

// ---------------------------------------------------------------------------
// LLM tool call
// ---------------------------------------------------------------------------

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
    let resolved = cfg.for_component(component);

    match resolved.provider {
        LlmProvider::Anthropic => llm_tool_call_anthropic(&cfg.client, &resolved, system, user, fn_name, fn_desc, parameters).await,
        LlmProvider::OpenAI => llm_tool_call_openai(&cfg.client, &resolved, system, user, fn_name, fn_desc, parameters).await,
    }
}

async fn llm_tool_call_openai<T: serde::de::DeserializeOwned>(
    client: &reqwest::Client,
    resolved: &ResolvedConfig,
    system: &str,
    user: &str,
    fn_name: &str,
    fn_desc: &str,
    parameters: serde_json::Value,
) -> Result<ToolCallResult<T>, EngramError> {
    let req = ChatRequest {
        model: resolved.model.clone(),
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

    let builder = add_auth(resolved, client.post(&resolved.url).json(&req));

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

    Ok(ToolCallResult { value, usage: chat.usage, model: resolved.model.clone(), duration_ms })
}

async fn llm_tool_call_anthropic<T: serde::de::DeserializeOwned>(
    client: &reqwest::Client,
    resolved: &ResolvedConfig,
    system: &str,
    user: &str,
    fn_name: &str,
    fn_desc: &str,
    parameters: serde_json::Value,
) -> Result<ToolCallResult<T>, EngramError> {
    let req = AnthropicRequest {
        model: resolved.model.clone(),
        max_tokens: 4096,
        system: if system.is_empty() { None } else { Some(system.to_string()) },
        messages: vec![AnthropicMessage { role: "user".into(), content: user.into() }],
        temperature: 0.1,
        tools: Some(vec![AnthropicTool {
            name: fn_name.into(),
            description: fn_desc.into(),
            input_schema: parameters,
        }]),
        tool_choice: Some(serde_json::json!({"type": "tool", "name": fn_name})),
    };

    let builder = add_auth(resolved, client.post(&resolved.url).json(&req));

    let start = std::time::Instant::now();
    let resp = builder
        .send()
        .await
        .map_err(|e| ai_err(format!("Anthropic tool call failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(ai_err(format!("Anthropic returned {status}: {body}")));
    }

    let ar: AnthropicResponse = resp
        .json()
        .await
        .map_err(|e| ai_err(format!("Anthropic tool response parse failed: {e}")))?;
    let duration_ms = start.elapsed().as_millis() as u64;

    // Find the tool_use content block
    let input = ar.content.iter()
        .find(|b| b.block_type == "tool_use")
        .and_then(|b| b.input.as_ref())
        .ok_or_else(|| ai_err("no tool_use block in Anthropic response"))?;

    let args = serde_json::to_string(input)
        .map_err(|e| ai_err(format!("failed to serialize tool input: {e}")))?;

    let value: T = serde_json::from_str(&args)
        .map_err(|e| ai_err(format!("tool call arguments parse failed: {e}: {args}")))?;

    let usage = ar.usage.as_ref().map(|u| u.to_usage());
    Ok(ToolCallResult { value, usage, model: resolved.model.clone(), duration_ms })
}

// ---------------------------------------------------------------------------
// Query expansion
// ---------------------------------------------------------------------------

use crate::prompts;

/// Generate alternative query phrasings for better recall coverage.
/// Returns (expansions, ToolCallResult metadata) for optional usage logging.
pub async fn expand_query(cfg: &AiConfig, query: &str) -> (Vec<String>, Option<ToolCallResult<()>>) {
    #[derive(Deserialize)]
    struct ExpandResult { queries: Vec<String> }

    let schema = prompts::expand_query_schema();

    match llm_tool_call::<ExpandResult>(
        cfg, "expand", prompts::EXPAND_PROMPT, query,
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

// ---------------------------------------------------------------------------
// Memory extraction
// ---------------------------------------------------------------------------

/// Map importance label from LLM to numeric value.
fn importance_from_label(label: &str) -> f64 {
    match label.to_lowercase().as_str() {
        "critical" => 0.9,
        "high" => 0.7,
        "medium" => 0.5,
        "low" => 0.3,
        _ => 0.5,
    }
}

/// Extract structured memories from raw text using an LLM.
pub async fn extract_memories(
    cfg: &AiConfig,
    text: &str,
) -> Result<Vec<ExtractedMemory>, EngramError> {
    debug!(model = %cfg.model_for("extract"), "extracting memories");

    let schema = prompts::extract_memories_schema();

    #[derive(serde::Deserialize)]
    struct RawMemory {
        content: String,
        #[serde(default)]
        importance: Option<String>,
        #[serde(default)]
        tags: Option<Vec<String>>,
        #[serde(default)]
        layer: Option<u8>,
        #[serde(default)]
        facts: Option<Vec<crate::db::FactInput>>,
        #[serde(default)]
        kind: Option<String>,
    }

    #[derive(serde::Deserialize)]
    struct ExtractResult {
        #[serde(default)]
        memories: Vec<RawMemory>,
    }

    let result = llm_tool_call::<ExtractResult>(
        cfg, "extract", prompts::EXTRACT_SYSTEM_PROMPT, text,
        "store_memories",
        "Store extracted memories from text",
        schema,
    ).await?;

    let memories: Vec<ExtractedMemory> = result.value.memories.into_iter().map(|raw| {
        let imp = raw.importance.as_deref().map_or(0.5, importance_from_label);
        ExtractedMemory {
            content: raw.content,
            importance: Some(imp),
            tags: raw.tags,
            layer: raw.layer,
            facts: raw.facts,
            kind: raw.kind,
        }
    }).collect();

    debug!(count = memories.len(), "extracted memories from text");
    Ok(memories)
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

// ---------------------------------------------------------------------------
// Embeddings (always OpenAI-compatible)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Vector utilities — delegate to crate::util (single source of truth)
// ---------------------------------------------------------------------------

pub use crate::util::cosine_similarity;

/// Serialize an f32 vector to bytes (little-endian) for SQLite BLOB storage.
pub fn embedding_to_bytes(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
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
    // Zero-copy reinterpret: allocate f32 vec and copy bytes into it via bytemuck
    let n = b.len() / 4;
    let mut v = vec![0f32; n];
    bytemuck::cast_slice_mut::<f32, u8>(&mut v).copy_from_slice(&b[..n * 4]);
    v
}
