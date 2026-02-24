//! SQLite-backed memory storage with FTS5 full-text search.

mod memory;
mod fts;
mod facts;
mod proxy;
mod vec;

use std::sync::{OnceLock, RwLock};

pub(crate) fn jieba() -> &'static jieba_rs::Jieba {
    static INSTANCE: OnceLock<jieba_rs::Jieba> = OnceLock::new();
    INSTANCE.get_or_init(jieba_rs::Jieba::new)
}

use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use serde::{Deserialize, Serialize};

/// Set busy_timeout on every connection handed out by the pool.
/// Prevents SQLITE_BUSY under concurrent write pressure (consolidation + API).
#[derive(Debug)]
struct BusyTimeoutCustomizer;
impl r2d2::CustomizeConnection<rusqlite::Connection, rusqlite::Error> for BusyTimeoutCustomizer {
    fn on_acquire(&self, conn: &mut rusqlite::Connection) -> Result<(), rusqlite::Error> {
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        Ok(())
    }
}

type PooledConn = r2d2::PooledConnection<SqliteConnectionManager>;

use crate::error::EngramError;


const MAX_CONTENT_LEN: usize = 8192;
const MAX_SOURCE_LEN: usize = 64;
const MAX_TAGS: usize = 20;
const MAX_TAG_LEN: usize = 32;


/// Buffer → Working → Core, each with different decay and recall bonus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub enum Layer {
    Buffer = 1,
    Working = 2,
    Core = 3,
}

impl TryFrom<u8> for Layer {
    type Error = EngramError;

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            1 => Ok(Layer::Buffer),
            2 => Ok(Layer::Working),
            3 => Ok(Layer::Core),
            _ => Err(EngramError::InvalidLayer(v)),
        }
    }
}

impl From<Layer> for u8 {
    fn from(l: Layer) -> u8 {
        l as u8
    }
}

impl Layer {
    pub fn default_decay(self) -> f64 {
        match self {
            Layer::Buffer => 5.0,
            Layer::Working => 1.0,
            Layer::Core => 0.05,
        }
    }

    pub fn score_bonus(self) -> f64 {
        match self {
            Layer::Buffer => 0.9,
            Layer::Working => 1.0,
            Layer::Core => 1.1,
        }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub content: String,
    pub layer: Layer,
    pub importance: f64,
    pub created_at: i64,
    pub last_accessed: i64,
    pub access_count: i64,
    pub repetition_count: i64,
    pub decay_rate: f64,
    pub source: String,
    pub tags: Vec<String>,
    #[serde(default = "default_ns", skip_serializing_if = "is_default_ns")]
    pub namespace: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    #[serde(default = "default_kind", skip_serializing_if = "is_default_kind")]
    pub kind: String,
    /// Timestamp of last real modification (content, layer, tags, kind change).
    /// Not updated by touch/access — only by actual edits.
    #[serde(default)]
    pub modified_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrashEntry {
    pub id: String,
    pub content: String,
    pub layer: i64,
    pub importance: f64,
    pub created_at: i64,
    pub deleted_at: i64,
    pub tags: Vec<String>,
    #[serde(default = "default_ns", skip_serializing_if = "is_default_ns")]
    pub namespace: String,
    pub source: String,
    #[serde(default = "default_kind", skip_serializing_if = "is_default_kind")]
    pub kind: String,
}

fn is_default_kind(k: &str) -> bool {
    k == "semantic"
}

fn default_kind() -> String {
    "semantic".into()
}

fn is_default_ns(ns: &str) -> bool {
    ns == "default"
}

fn default_ns() -> String {
    "default".into()
}

#[derive(Debug, Default, Deserialize)]
pub struct MemoryInput {
    #[serde(default)]
    pub content: String,
    pub layer: Option<u8>,
    pub importance: Option<f64>,
    pub source: Option<String>,
    pub tags: Option<Vec<String>>,
    /// If set, the new memory replaces these old ones (by id). They get deleted.
    #[serde(default)]
    pub supersedes: Option<Vec<String>>,
    /// Skip near-duplicate detection. Useful when storing intentionally similar memories.
    #[serde(default)]
    pub skip_dedup: Option<bool>,
    /// Namespace for multi-agent isolation.
    #[serde(default)]
    pub namespace: Option<String>,
    /// Wait for embedding to be generated before returning (default: false/async).
    #[serde(default)]
    pub sync_embed: Option<bool>,
    /// Memory kind: "semantic" (default), "episodic", or "procedural".
    #[serde(default)]
    pub kind: Option<String>,
}

impl MemoryInput {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            ..Default::default()
        }
    }

    pub fn layer(mut self, l: u8) -> Self {
        self.layer = Some(l);
        self
    }

    pub fn importance(mut self, i: f64) -> Self {
        self.importance = Some(i);
        self
    }

    pub fn source(mut self, s: impl Into<String>) -> Self {
        self.source = Some(s.into());
        self
    }

    pub fn tags(mut self, t: Vec<String>) -> Self {
        self.tags = Some(t);
        self
    }

    pub fn supersedes(mut self, ids: Vec<String>) -> Self {
        self.supersedes = Some(ids);
        self
    }

    pub fn skip_dedup(mut self) -> Self {
        self.skip_dedup = Some(true);
        self
    }

    pub fn namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = Some(ns.into());
        self
    }

    pub fn kind(mut self, k: impl Into<String>) -> Self {
        self.kind = Some(k.into());
        self
    }
}

/// A memory with its computed relevance score.
#[derive(Debug, Clone, Serialize)]
pub struct ScoredMemory {
    #[serde(flatten)]
    pub memory: Memory,
    pub score: f64,
    pub relevance: f64,
    pub recency: f64,
}

const MAX_FACT_FIELD_LEN: usize = 512;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub memory_id: String,
    pub namespace: String,
    pub created_at: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactInput {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub memory_id: Option<String>,
    #[serde(default)]
    pub valid_from: Option<i64>,
}

/// A single step in a fact chain: subject → predicate → object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// An ordered chain of fact triples representing a multi-hop path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactChain {
    pub path: Vec<FactTriple>,
}

fn validate_fact_input(input: &FactInput) -> Result<(), EngramError> {
    if input.subject.trim().is_empty() || input.predicate.trim().is_empty() || input.object.trim().is_empty() {
        return Err(EngramError::Validation("fact subject/predicate/object must not be empty".into()));
    }
    if input.subject.chars().count() > MAX_FACT_FIELD_LEN {
        return Err(EngramError::Validation("fact subject too long".into()));
    }
    if input.predicate.chars().count() > MAX_FACT_FIELD_LEN {
        return Err(EngramError::Validation("fact predicate too long".into()));
    }
    if input.object.chars().count() > MAX_FACT_FIELD_LEN {
        return Err(EngramError::Validation("fact object too long".into()));
    }
    Ok(())
}

#[derive(Debug, Serialize)]
pub struct Stats {
    pub total: usize,
    pub buffer: usize,
    pub working: usize,
    pub core: usize,
    pub by_kind: KindStats,
}

#[derive(Debug, Default, Serialize)]
pub struct KindStats {
    pub semantic: usize,
    pub episodic: usize,
    pub procedural: usize,
}

#[derive(Debug, Default, Serialize)]
pub struct IntegrityReport {
    pub total: usize,
    pub fts_indexed: usize,
    pub orphan_fts: usize,
    pub missing_fts: usize,
    pub missing_embedding: usize,
    pub ok: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct DailyLlmUsage {
    pub date: String,
    pub component: String,
    pub model: String,
    pub calls: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_tokens: i64,
    pub avg_duration_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComponentUsage {
    pub calls: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_tokens: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LlmUsageSummary {
    pub total_calls: i64,
    pub total_input_tokens: i64,
    pub total_output_tokens: i64,
    pub total_cached_tokens: i64,
    pub today_calls: i64,
    pub today_input_tokens: i64,
    pub today_output_tokens: i64,
    pub by_component: std::collections::HashMap<String, ComponentUsage>,
    pub by_model: std::collections::HashMap<String, ComponentUsage>,
}


fn validate_input(input: &MemoryInput) -> Result<(), EngramError> {
    let content = input.content.trim();
    if content.is_empty() {
        return Err(EngramError::EmptyContent);
    }
    if content.chars().count() > MAX_CONTENT_LEN {
        return Err(EngramError::ContentTooLong);
    }
    if let Some(l) = input.layer {
        if !(1..=3).contains(&l) {
            return Err(EngramError::InvalidLayer(l));
        }
    }
    if let Some(ref src) = input.source {
        if src.len() > MAX_SOURCE_LEN {
            return Err(EngramError::Validation("source too long".into()));
        }
    }
    if let Some(ref tags) = input.tags {
        if tags.len() > MAX_TAGS {
            return Err(EngramError::Validation(format!("too many tags (max {MAX_TAGS})")));
        }
        if let Some(t) = tags.iter().find(|t| t.chars().count() > MAX_TAG_LEN) {
            return Err(EngramError::Validation(format!("tag '{}' too long (max {MAX_TAG_LEN})", t)));
        }
    }
    Ok(())
}


pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_millis() as i64
}

// FTS5 unicode61 tokenizer handles Latin scripts fine but can't segment CJK.
// We use jieba for proper Chinese word segmentation and fall back to bigrams
// for Japanese/Korean which jieba doesn't cover.
pub fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Basic
        | '\u{3400}'..='\u{4DBF}' // CJK Extension A
        | '\u{F900}'..='\u{FAFF}' // CJK Compatibility
        | '\u{3040}'..='\u{30FF}' // Hiragana + Katakana
        | '\u{AC00}'..='\u{D7AF}' // Hangul
    )
}

fn is_cjk_ideograph(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}'
        | '\u{3400}'..='\u{4DBF}'
        | '\u{F900}'..='\u{FAFF}'
    )
}

/// Segment CJK text properly and append tokens for FTS5 indexing.
/// Chinese goes through jieba; Japanese/Korean falls back to bigrams.
fn append_segmented(text: &str) -> String {
    let has_cjk = text.chars().any(is_cjk);
    if !has_cjk {
        return text.to_string();
    }

    // Split CJK/latin boundaries so "alice是谁" → "alice 是谁"
    let mut spaced = String::with_capacity(text.len() * 2);
    let mut prev_cjk: Option<bool> = None;
    for c in text.chars() {
        if c.is_alphanumeric() || is_cjk(c) {
            let cur = is_cjk(c);
            if let Some(prev) = prev_cjk {
                if cur != prev {
                    spaced.push(' ');
                }
            }
            spaced.push(c);
            prev_cjk = Some(cur);
        } else {
            spaced.push(c);
            prev_cjk = None;
        }
    }

    // jieba handles Chinese (CJK ideographs)
    let has_chinese = spaced.chars().any(is_cjk_ideograph);
    let mut extra_tokens = Vec::new();

    if has_chinese {
        let words = jieba().cut_for_search(&spaced, false);
        for w in words {
            let trimmed = w.trim();
            if trimmed.len() > 1 && trimmed.chars().any(is_cjk) {
                extra_tokens.push(trimmed.to_string());
            }
        }
    }

    // Bigrams for kana/hangul (jieba doesn't segment these)
    let chars: Vec<char> = spaced.chars().collect();
    for i in 0..chars.len().saturating_sub(1) {
        let a = chars[i];
        let b = chars[i + 1];
        let non_ideo = |c: char| is_cjk(c) && !is_cjk_ideograph(c);
        if non_ideo(a) && non_ideo(b) {
            let mut s = String::with_capacity(8);
            s.push(a);
            s.push(b);
            extra_tokens.push(s);
        }
    }

    if extra_tokens.is_empty() {
        spaced
    } else {
        format!("{} {}", spaced, extra_tokens.join(" "))
    }
}


const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    layer INTEGER NOT NULL DEFAULT 1,
    importance REAL NOT NULL DEFAULT 0.5,
    created_at INTEGER NOT NULL,
    last_accessed INTEGER NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    repetition_count INTEGER NOT NULL DEFAULT 0,
    decay_rate REAL NOT NULL DEFAULT 1.0,
    source TEXT NOT NULL DEFAULT 'manual',
    tags TEXT NOT NULL DEFAULT '[]',
    namespace TEXT NOT NULL DEFAULT 'default',
    embedding BLOB,
    kind TEXT NOT NULL DEFAULT 'semantic',
    modified_at INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_layer ON memories(layer);
CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance);
CREATE INDEX IF NOT EXISTS idx_last_accessed ON memories(last_accessed);

CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    namespace TEXT NOT NULL DEFAULT 'default',
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject, namespace);
CREATE INDEX IF NOT EXISTS idx_facts_object ON facts(object, namespace);
CREATE INDEX IF NOT EXISTS idx_facts_memory ON facts(memory_id);
CREATE INDEX IF NOT EXISTS idx_facts_predicate ON facts(predicate, namespace);

CREATE TABLE IF NOT EXISTS proxy_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_proxy_turns_session ON proxy_turns(session_key);

CREATE TABLE IF NOT EXISTS proxy_sessions (
    session_key TEXT PRIMARY KEY,
    watermark INTEGER NOT NULL DEFAULT 0,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS trash (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    layer INTEGER NOT NULL,
    importance REAL NOT NULL,
    created_at INTEGER NOT NULL,
    deleted_at INTEGER NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    namespace TEXT NOT NULL DEFAULT 'default',
    source TEXT NOT NULL DEFAULT 'api',
    kind TEXT NOT NULL DEFAULT 'semantic'
);
CREATE INDEX IF NOT EXISTS idx_trash_deleted ON trash(deleted_at);

CREATE TABLE IF NOT EXISTS engram_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    component TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_llm_usage_ts ON llm_usage(ts);
CREATE INDEX IF NOT EXISTS idx_llm_usage_component ON llm_usage(component);
"#;

// Use content= external content FTS — we manage inserts/deletes ourselves
// so we can pre-process CJK text with bigrams before indexing.
const FTS_SCHEMA: &str =
    "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(\
     id UNINDEXED, content, tags, tokenize='unicode61')";

// Drop old triggers if they exist (migrating from trigger-based to manual FTS)
const DROP_TRIGGERS: [&str; 3] = [
    "DROP TRIGGER IF EXISTS mem_ai",
    "DROP TRIGGER IF EXISTS mem_ad",
    "DROP TRIGGER IF EXISTS mem_au",
];


/// SQLite-backed memory store.
pub struct MemoryDB {
    pool: Pool<SqliteConnectionManager>,
    /// In-memory HNSW-backed vector index for fast semantic search.
    vec_index: RwLock<vec::VecIndex>,
}


impl MemoryDB {
    fn conn(&self) -> Result<PooledConn, EngramError> {
        self.pool.get().map_err(|e| EngramError::Internal(format!("pool: {e}")))
    }

    /// Database file size in bytes (via SQLite pragma).
    pub fn db_size_bytes(&self) -> i64 {
        self.conn()
            .and_then(|c| c.query_row(
                "SELECT page_count * page_size FROM pragma_page_count, pragma_page_size",
                [], |r| r.get(0),
            ).map_err(|e| EngramError::Internal(e.to_string())))
            .unwrap_or(0)
    }

    pub fn get_meta(&self, key: &str) -> Option<String> {
        self.conn().ok().and_then(|c| {
            c.query_row("SELECT value FROM engram_meta WHERE key = ?1", [key], |r| r.get(0)).ok()
        })
    }

    pub fn set_meta(&self, key: &str, value: &str) -> Result<(), EngramError> {
        let c = self.conn()?;
        c.execute(
            "INSERT OR REPLACE INTO engram_meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![key, value],
        )?;
        Ok(())
    }

    pub fn log_llm_call(
        &self, component: &str, model: &str,
        input_tokens: u32, output_tokens: u32, cached_tokens: u32,
        duration_ms: u64,
    ) -> Result<(), EngramError> {
        let c = self.conn()?;
        c.execute(
            "INSERT INTO llm_usage (ts, component, model, input_tokens, output_tokens, cached_tokens, duration_ms) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![now_ms() as i64, component, model, input_tokens, output_tokens, cached_tokens, duration_ms as i64],
        )?;
        Ok(())
    }

    pub fn llm_usage_daily(&self, days: u32) -> Result<Vec<DailyLlmUsage>, EngramError> {
        let c = self.conn()?;
        let cutoff = now_ms() as i64 - (days as i64 * 86_400_000);
        let mut stmt = c.prepare(
            "SELECT date(ts/1000, 'unixepoch') as d, component, model, \
             COUNT(*) as calls, SUM(input_tokens), SUM(output_tokens), SUM(cached_tokens), \
             AVG(duration_ms) \
             FROM llm_usage WHERE ts >= ?1 \
             GROUP BY d, component, model ORDER BY d DESC, calls DESC"
        )?;
        let rows = stmt.query_map([cutoff], |r| {
            Ok(DailyLlmUsage {
                date: r.get(0)?,
                component: r.get(1)?,
                model: r.get(2)?,
                calls: r.get(3)?,
                input_tokens: r.get(4)?,
                output_tokens: r.get(5)?,
                cached_tokens: r.get(6)?,
                avg_duration_ms: r.get::<_, f64>(7)? as u64,
            })
        })?.collect::<Result<Vec<_>, _>>()
        .map_err(|e| EngramError::Internal(e.to_string()))?;
        Ok(rows)
    }

    pub fn llm_usage_summary(&self) -> Result<LlmUsageSummary, EngramError> {
        let c = self.conn()?;
        let today_start = {
            let now = now_ms() as i64;
            // Start of today in UTC
            now - (now % 86_400_000)
        };

        let total: (i64, i64, i64, i64) = c.query_row(
            "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), COALESCE(SUM(cached_tokens),0) FROM llm_usage",
            [], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
        ).unwrap_or((0, 0, 0, 0));

        let today: (i64, i64, i64, i64) = c.query_row(
            "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), COALESCE(SUM(cached_tokens),0) FROM llm_usage WHERE ts >= ?1",
            [today_start], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
        ).unwrap_or((0, 0, 0, 0));

        let mut by_component = std::collections::HashMap::new();
        {
            let mut stmt = c.prepare(
                "SELECT component, COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), COALESCE(SUM(cached_tokens),0) \
                 FROM llm_usage GROUP BY component"
            )?;
            let rows = stmt.query_map([], |r| {
                Ok((r.get::<_, String>(0)?, ComponentUsage {
                    calls: r.get(1)?,
                    input_tokens: r.get(2)?,
                    output_tokens: r.get(3)?,
                    cached_tokens: r.get(4)?,
                }))
            })?.collect::<Result<Vec<_>, _>>()
            .map_err(|e| EngramError::Internal(e.to_string()))?;
            for (k, v) in rows { by_component.insert(k, v); }
        }

        let mut by_model = std::collections::HashMap::new();
        {
            let mut stmt = c.prepare(
                "SELECT model, COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), COALESCE(SUM(cached_tokens),0) \
                 FROM llm_usage GROUP BY model"
            )?;
            let rows = stmt.query_map([], |r| {
                Ok((r.get::<_, String>(0)?, ComponentUsage {
                    calls: r.get(1)?,
                    input_tokens: r.get(2)?,
                    output_tokens: r.get(3)?,
                    cached_tokens: r.get(4)?,
                }))
            })?.collect::<Result<Vec<_>, _>>()
            .map_err(|e| EngramError::Internal(e.to_string()))?;
            for (k, v) in rows { by_model.insert(k, v); }
        }

        Ok(LlmUsageSummary {
            total_calls: total.0,
            total_input_tokens: total.1,
            total_output_tokens: total.2,
            total_cached_tokens: total.3,
            today_calls: today.0,
            today_input_tokens: today.1,
            today_output_tokens: today.2,
            by_component,
            by_model,
        })
    }

    /// Open (or create) a database at the given path.
    /// Pool size defaults to 8 (1 writer + 7 readers in WAL mode).
    pub fn open(path: &str) -> Result<Self, EngramError> {
        let pool_size = if path == ":memory:" { 2 } else { 8 };
        let manager = if path == ":memory:" {
            // Shared cache so all pool connections see the same in-memory DB.
            // Each test gets a unique name to avoid cross-test pollution.
            let name = uuid::Uuid::new_v4().to_string();
            SqliteConnectionManager::file(format!("file:{name}?mode=memory&cache=shared"))
        } else {
            SqliteConnectionManager::file(path)
        };
        let pool = Pool::builder()
            .max_size(pool_size)
            .connection_customizer(Box::new(BusyTimeoutCustomizer))
            .build(manager)
            .map_err(|e| EngramError::Internal(format!("pool: {e}")))?;

        // initialize schema on a fresh connection
        let conn = pool.get().map_err(|e| EngramError::Internal(e.to_string()))?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA auto_vacuum=INCREMENTAL;")?;
        conn.execute_batch(SCHEMA)?;
        conn.execute(FTS_SCHEMA, [])?;
        for t in &DROP_TRIGGERS {
            conn.execute(t, [])?;
        }
        if conn.prepare("SELECT embedding FROM memories LIMIT 0").is_err() {
            conn.execute("ALTER TABLE memories ADD COLUMN embedding BLOB", [])?;
        }
        if conn.prepare("SELECT namespace FROM memories LIMIT 0").is_err() {
            conn.execute(
                "ALTER TABLE memories ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'",
                [],
            )?;
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_namespace ON memories(namespace)",
                [],
            )?;
        }
        if conn.prepare("SELECT repetition_count FROM memories LIMIT 0").is_err() {
            conn.execute(
                "ALTER TABLE memories ADD COLUMN repetition_count INTEGER NOT NULL DEFAULT 0",
                [],
            )?;
        }
        if conn.prepare("SELECT valid_from FROM facts LIMIT 0").is_err() {
            conn.execute("ALTER TABLE facts ADD COLUMN valid_from INTEGER", [])?;
            conn.execute("ALTER TABLE facts ADD COLUMN valid_until INTEGER", [])?;
            conn.execute("ALTER TABLE facts ADD COLUMN superseded_by TEXT", [])?;
        }
        if conn.prepare("SELECT kind FROM memories LIMIT 0").is_err() {
            conn.execute("ALTER TABLE memories ADD COLUMN kind TEXT NOT NULL DEFAULT 'semantic'", [])?;
        }
        if conn.prepare("SELECT modified_at FROM memories LIMIT 0").is_err() {
            conn.execute("ALTER TABLE memories ADD COLUMN modified_at INTEGER NOT NULL DEFAULT 0", [])?;
            // Backfill: set modified_at = created_at for existing rows
            conn.execute("UPDATE memories SET modified_at = created_at WHERE modified_at = 0", [])?;
        }
        drop(conn);
        let db = Self { pool, vec_index: RwLock::new(vec::VecIndex::new()) };
        db.rebuild_fts()?;
        db.load_vec_index();
        Ok(db)
    }

}

fn tokenize_for_dedup(text: &str) -> std::collections::HashSet<String> {
    let has_chinese = text.chars().any(is_cjk_ideograph);
    if has_chinese {
        jieba().cut_for_search(text, false)
            .into_iter()
            .map(|w| w.trim().to_lowercase())
            .filter(|w| w.len() > 1)
            .collect()
    } else {
        // Latin + kana/hangul bigrams
        let mut tokens: std::collections::HashSet<String> = text
            .split_whitespace()
            .map(str::to_lowercase)
            .collect();
        let chars: Vec<char> = text.chars().collect();
        for i in 0..chars.len().saturating_sub(1) {
            if is_cjk(chars[i]) && is_cjk(chars[i + 1]) {
                let mut bigram = String::with_capacity(8);
                bigram.push(chars[i]);
                bigram.push(chars[i + 1]);
                tokens.insert(bigram);
            }
        }
        tokens
    }
}

/// Jaccard similarity between two text snippets (no DB involved).
/// Returns true if similarity exceeds threshold.
pub(crate) fn jaccard_similar(a: &str, b: &str, threshold: f64) -> bool {
    let ta = tokenize_for_dedup(a);
    let tb = tokenize_for_dedup(b);
    if ta.len() < 3 || tb.len() < 3 { return false; }
    let inter = ta.intersection(&tb).count();
    let union = ta.union(&tb).count();
    union > 0 && (inter as f64 / union as f64) > threshold
}

fn row_to_memory(row: &rusqlite::Row) -> rusqlite::Result<Memory> {
    row_to_memory_impl(row, false)
}

fn row_to_memory_with_embedding(row: &rusqlite::Row) -> rusqlite::Result<Memory> {
    row_to_memory_impl(row, true)
}

/// Row mapper for queries that select explicit columns without `embedding`.
/// Column order: id, content, layer, importance, created_at, last_accessed,
/// access_count, decay_rate, source, tags, namespace, kind
fn row_to_memory_meta(row: &rusqlite::Row) -> rusqlite::Result<Memory> {
    let layer_val: u8 = row.get("layer")?;
    let tags_str: String = row.get("tags")?;
    Ok(Memory {
        id: row.get("id")?,
        content: row.get("content")?,
        layer: layer_val.try_into().unwrap_or(Layer::Buffer),
        importance: row.get("importance")?,
        created_at: row.get("created_at")?,
        last_accessed: row.get("last_accessed")?,
        access_count: row.get("access_count")?,
        repetition_count: row.get::<_, i64>("repetition_count").unwrap_or(0),
        decay_rate: row.get("decay_rate")?,
        source: row.get("source")?,
        tags: serde_json::from_str(&tags_str).unwrap_or_default(),
        namespace: row.get::<_, String>("namespace").unwrap_or_else(|_| "default".into()),
        embedding: None,
        kind: row.get::<_, String>("kind").unwrap_or_else(|_| "semantic".into()),
        modified_at: row.get::<_, i64>("modified_at").unwrap_or(0),
    })
}

fn row_to_memory_impl(row: &rusqlite::Row, include_embedding: bool) -> rusqlite::Result<Memory> {
    let layer_val: u8 = row.get("layer")?;
    let tags_str: String = row.get("tags")?;
    let embedding = if include_embedding {
        let blob: Option<Vec<u8>> = row.get("embedding").ok();
        blob.map(|b| crate::ai::bytes_to_embedding(&b))
    } else {
        None
    };
    Ok(Memory {
        id: row.get("id")?,
        content: row.get("content")?,
        layer: layer_val.try_into().unwrap_or(Layer::Buffer),
        importance: row.get("importance")?,
        created_at: row.get("created_at")?,
        last_accessed: row.get("last_accessed")?,
        access_count: row.get("access_count")?,
        repetition_count: row.get::<_, i64>("repetition_count").unwrap_or(0),
        decay_rate: row.get("decay_rate")?,
        source: row.get("source")?,
        tags: serde_json::from_str(&tags_str).unwrap_or_default(),
        namespace: row.get::<_, String>("namespace").unwrap_or_else(|_| "default".into()),
        embedding,
        kind: row.get::<_, String>("kind").unwrap_or_else(|_| "semantic".into()),
        modified_at: row.get::<_, i64>("modified_at").unwrap_or(0),
    })
}


#[cfg(test)]
mod meta_tests {
    use super::*;

    #[test]
    fn meta_get_set() {
        let db = MemoryDB::open(":memory:").unwrap();
        assert_eq!(db.get_meta("nonexistent"), None);
        db.set_meta("last_audit_ms", "1234567890").unwrap();
        assert_eq!(db.get_meta("last_audit_ms"), Some("1234567890".to_string()));
        db.set_meta("last_audit_ms", "9999999999").unwrap();
        assert_eq!(db.get_meta("last_audit_ms"), Some("9999999999".to_string()));
    }
}
