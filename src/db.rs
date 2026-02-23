//! SQLite-backed memory storage with FTS5 full-text search.

use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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
            Layer::Buffer => 0.8,
            Layer::Working => 1.0,
            Layer::Core => 1.3,
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
    pub decay_rate: f64,
    pub source: String,
    pub tags: Vec<String>,
    #[serde(skip_serializing_if = "is_default_ns")]
    pub namespace: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f64>>,
}

fn is_default_ns(ns: &str) -> bool {
    ns == "default"
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

#[derive(Debug, Serialize)]
pub struct Stats {
    pub total: usize,
    pub buffer: usize,
    pub working: usize,
    pub core: usize,
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

// FTS5 unicode61 tokenizer splits on word boundaries which works for Latin
// scripts but butchers CJK since there are no spaces. We bigram CJK chars
// to get usable index terms.
pub fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Basic
        | '\u{3400}'..='\u{4DBF}' // CJK Extension A
        | '\u{F900}'..='\u{FAFF}' // CJK Compatibility
        | '\u{3040}'..='\u{30FF}' // Hiragana + Katakana
        | '\u{AC00}'..='\u{D7AF}' // Hangul
    )
}

/// Append CJK bigrams to text so FTS5 unicode61 can actually index Chinese/Japanese/Korean.
/// Original text is preserved intact; bigrams are appended after a space.
fn append_cjk_bigrams(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut bigrams = Vec::new();
    for i in 0..chars.len().saturating_sub(1) {
        if is_cjk(chars[i]) && is_cjk(chars[i + 1]) {
            let mut s = String::with_capacity(8);
            s.push(chars[i]);
            s.push(chars[i + 1]);
            bigrams.push(s);
        }
    }
    if bigrams.is_empty() {
        text.to_string()
    } else {
        format!("{} {}", text, bigrams.join(" "))
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
    decay_rate REAL NOT NULL DEFAULT 1.0,
    source TEXT NOT NULL DEFAULT 'manual',
    tags TEXT NOT NULL DEFAULT '[]',
    namespace TEXT NOT NULL DEFAULT 'default',
    embedding BLOB
);

CREATE INDEX IF NOT EXISTS idx_layer ON memories(layer);
CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance);
CREATE INDEX IF NOT EXISTS idx_last_accessed ON memories(last_accessed);
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
}

impl MemoryDB {
    fn conn(&self) -> PooledConn {
        self.pool.get().expect("db pool exhausted")
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
            .build(manager)
            .map_err(|e| EngramError::Internal(format!("pool: {e}")))?;

        // initialize schema on a fresh connection
        let conn = pool.get().map_err(|e| EngramError::Internal(e.to_string()))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
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
        drop(conn);
        let db = Self { pool };
        db.rebuild_fts()?;
        Ok(db)
    }

    /// Insert a row into FTS with CJK bigram preprocessing.
    fn fts_insert(&self, id: &str, content: &str, tags_json: &str) -> Result<(), EngramError> {
        let processed = append_cjk_bigrams(content);
        self.conn().execute(
            "INSERT INTO memories_fts(id, content, tags) VALUES (?1, ?2, ?3)",
            params![id, processed, tags_json],
        )?;
        Ok(())
    }

    fn fts_delete(&self, id: &str) -> Result<(), EngramError> {
        self.conn()
            .execute("DELETE FROM memories_fts WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Rebuild FTS index from scratch (idempotent, runs on startup).
    fn rebuild_fts(&self) -> Result<(), EngramError> {
        let fts_count: i64 = self
            .conn()
            .query_row("SELECT COUNT(*) FROM memories_fts", [], |r| r.get(0))?;
        let mem_count: i64 = self
            .conn()
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))?;

        if mem_count == 0 {
            return Ok(());
        }

        // Check if we need to rebuild: count mismatch or missing CJK bigrams
        let needs_rebuild = fts_count != mem_count || {
            // Sample a CJK-containing memory from the FTS table
            let has_cjk_content: bool = self
                .conn()
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM memories WHERE content GLOB '*[一-龥]*' LIMIT 1)",
                    [],
                    |r| r.get(0),
                )
                .unwrap_or(false);

            if has_cjk_content {
                // If there's CJK content, check if FTS has longer content (bigrams appended)
                let sample: Option<(i64, i64)> = self
                    .conn()
                    .query_row(
                        "SELECT LENGTH(m.content), LENGTH(f.content) \
                         FROM memories m JOIN memories_fts f ON m.id = f.id \
                         WHERE m.content GLOB '*[一-龥]*' LIMIT 1",
                        [],
                        |r| Ok((r.get(0)?, r.get(1)?)),
                    )
                    .ok();

                match sample {
                    Some((mem_len, fts_len)) => fts_len <= mem_len, // no bigrams appended
                    None => true, // can't check, rebuild to be safe
                }
            } else {
                false // no CJK content, no rebuild needed
            }
        };

        if !needs_rebuild {
            return Ok(());
        }

        // Full rebuild
        self.conn().execute("DELETE FROM memories_fts", [])?;
        let conn = self.conn();
        let mut stmt = conn
            .prepare("SELECT id, content, tags FROM memories")?;
        let rows: Vec<(String, String, String)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();

        for (id, content, tags) in &rows {
            self.fts_insert(id, content, tags)?;
        }
        tracing::info!(count = rows.len(), "rebuilt FTS index with CJK bigrams");
        Ok(())
    }

    pub fn insert(&self, input: MemoryInput) -> Result<Memory, EngramError> {
        validate_input(&input)?;

        // Near-duplicate detection: if an existing memory has very similar content,
        // update it instead of creating a duplicate. Uses token overlap as a proxy.
        let do_dedup = !input.skip_dedup.unwrap_or(false);
        if do_dedup {
        let ns = input.namespace.as_deref().unwrap_or("default");
        if let Some(existing) = self.find_near_duplicate(&input.content, ns) {
            tracing::debug!(existing_id = %existing.id, "near-duplicate found, updating instead");
            let tags = input.tags.unwrap_or_default();
            // Merge tags from both
            let mut merged_tags: Vec<String> = existing.tags.clone();
            for t in &tags {
                if !merged_tags.contains(t) {
                    merged_tags.push(t.clone());
                }
            }
            // Keep the higher importance
            let imp = input
                .importance
                .map(|new_imp| new_imp.max(existing.importance))
                .unwrap_or(existing.importance);
            // Keep the higher layer
            let layer = input
                .layer
                .map(|new_l| new_l.max(existing.layer as u8))
                .unwrap_or(existing.layer as u8);

            return self
                .update_fields(
                    &existing.id,
                    Some(&input.content),
                    Some(layer),
                    Some(imp),
                    Some(&merged_tags),
                )
                .and_then(|opt| {
                    // handle supersedes even in dedup path
                    if let Some(ref old_ids) = input.supersedes {
                        for old_id in old_ids {
                            if old_id != &existing.id {
                                if let Err(e) = self.delete(old_id) {
                                    tracing::warn!(old_id, error = %e, "supersede delete failed");
                                }
                            }
                        }
                    }
                    opt.ok_or(EngramError::Internal("update after dedup failed".into()))
                });
        }
        } // do_dedup

        let now = now_ms();
        let layer_val = input.layer.unwrap_or(1);
        let layer: Layer = layer_val.try_into()?;
        let id = Uuid::new_v4().to_string();
        let importance = input.importance.unwrap_or(0.5).clamp(0.0, 1.0);
        let source = input.source.unwrap_or_else(|| "api".into());
        let tags = input.tags.unwrap_or_default();
        let tags_json = serde_json::to_string(&tags).unwrap_or_else(|_| "[]".into());

        let namespace = input.namespace.unwrap_or_else(|| "default".into());

        self.conn().execute(
            "INSERT INTO memories \
             (id, content, layer, importance, created_at, last_accessed, \
              access_count, decay_rate, source, tags, namespace) \
             VALUES (?1,?2,?3,?4,?5,?6,0,?7,?8,?9,?10)",
            params![
                id,
                input.content,
                layer_val,
                importance,
                now,
                now,
                layer.default_decay(),
                source,
                tags_json,
                namespace
            ],
        )?;

        self.fts_insert(&id, &input.content, &tags_json)?;

        // supersede: delete old memories that this one replaces
        if let Some(ref old_ids) = input.supersedes {
            for old_id in old_ids {
                if let Err(e) = self.delete(old_id) {
                    tracing::warn!(old_id, error = %e, "failed to delete superseded memory");
                }
            }
        }

        Ok(Memory {
            id,
            content: input.content,
            layer,
            importance,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            decay_rate: layer.default_decay(),
            source,
            tags,
            namespace,
            embedding: None,
        })
    }

    /// Batch insert within a single transaction. Skips dedup for speed.
    /// Returns the successfully inserted memories.
    pub fn insert_batch(&self, inputs: Vec<MemoryInput>) -> Result<Vec<Memory>, EngramError> {
        let conn = self.conn();
        conn.execute_batch("BEGIN")?;
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            if let Err(e) = validate_input(&input) {
                tracing::warn!(error = %e, "batch: skipping invalid input");
                continue;
            }
            let now = now_ms();
            let layer_val = input.layer.unwrap_or(1);
            let layer: Layer = match layer_val.try_into() {
                Ok(l) => l,
                Err(_) => continue,
            };
            let id = Uuid::new_v4().to_string();
            let importance = input.importance.unwrap_or(0.5).clamp(0.0, 1.0);
            let source = input.source.unwrap_or_else(|| "api".into());
            let tags = input.tags.unwrap_or_default();
            let tags_json = serde_json::to_string(&tags).unwrap_or_else(|_| "[]".into());
            let namespace = input.namespace.unwrap_or_else(|| "default".into());

            conn.execute(
                "INSERT INTO memories \
                 (id, content, layer, importance, created_at, last_accessed, \
                  access_count, decay_rate, source, tags, namespace) \
                 VALUES (?1,?2,?3,?4,?5,?6,0,?7,?8,?9,?10)",
                params![
                    id, input.content, layer_val, importance, now, now,
                    layer.default_decay(), source, tags_json, namespace
                ],
            )?;
            let processed = append_cjk_bigrams(&input.content);
            conn.execute(
                "INSERT INTO memories_fts(id, content, tags) VALUES (?1, ?2, ?3)",
                params![id, processed, tags_json],
            )?;

            results.push(Memory {
                id,
                content: input.content,
                layer,
                importance,
                created_at: now,
                last_accessed: now,
                access_count: 0,
                decay_rate: layer.default_decay(),
                source,
                tags,
                namespace,
                embedding: None,
            });
        }
        conn.execute_batch("COMMIT")?;
        Ok(results)
    }

    pub fn get(&self, id: &str) -> Result<Option<Memory>, EngramError> {
        let conn = self.conn();
        let mut stmt = conn.prepare("SELECT * FROM memories WHERE id = ?1")?;
        let mut rows = stmt.query(params![id])?;
        match rows.next()? {
            Some(row) => Ok(Some(row_to_memory(row)?)),
            None => Ok(None),
        }
    }

    pub fn delete(&self, id: &str) -> Result<bool, EngramError> {
        self.fts_delete(id)?;
        let n = self.conn().execute("DELETE FROM memories WHERE id = ?1", params![id])?;
        Ok(n > 0)
    }

    /// Delete all memories in a namespace. Returns how many were removed.
    pub fn delete_namespace(&self, ns: &str) -> Result<usize, EngramError> {
        // collect IDs first so we can clean FTS
        let conn = self.conn();
        let mut stmt = conn.prepare("SELECT id FROM memories WHERE namespace = ?1")?;
        let ids: Vec<String> = stmt
            .query_map(params![ns], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        drop(stmt);
        drop(conn);

        for id in &ids {
            self.fts_delete(id)?;
        }
        let n = self.conn().execute("DELETE FROM memories WHERE namespace = ?1", params![ns])?;
        Ok(n)
    }

    pub fn touch(&self, id: &str) -> Result<(), EngramError> {
        self.conn().execute(
            "UPDATE memories SET last_accessed = ?1, access_count = access_count + 1 WHERE id = ?2",
            params![now_ms(), id],
        )?;
        Ok(())
    }

    /// Full-text search using FTS5. Returns `(id, bm25_score)` pairs.
    pub fn search_fts(&self, query: &str, limit: usize) -> Vec<(String, f64)> {
        let sanitized: String = query
            .chars()
            .map(|c| if c.is_alphanumeric() || is_cjk(c) { c } else { ' ' })
            .collect();
        let sanitized = sanitized.trim();
        if sanitized.is_empty() {
            return vec![];
        }

        // Pre-process query: add CJK bigrams so "天气" matches bigram-indexed content
        let processed = append_cjk_bigrams(sanitized);
        let fts_query: String = processed.split_whitespace().collect::<Vec<_>>().join(" OR ");

        let conn = self.conn();
        let Ok(mut stmt) = conn.prepare(
            "SELECT id, rank FROM memories_fts \
             WHERE memories_fts MATCH ?1 ORDER BY rank LIMIT ?2",
        ) else {
            return vec![];
        };

        stmt.query_map(params![fts_query, limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })
        .map(|iter| iter.filter_map(|r| r.ok()).map(|(id, rank)| (id, -rank)).collect())
        .unwrap_or_default()
    }

    /// List all memories in a given layer, ordered by importance descending.
    pub fn list_by_layer(&self, layer: Layer, limit: usize, offset: usize) -> Vec<Memory> {
        let conn = self.conn();
        let Ok(mut stmt) = conn.prepare(
            "SELECT * FROM memories WHERE layer = ?1 ORDER BY importance DESC LIMIT ?2 OFFSET ?3",
        ) else {
            return vec![];
        };

        stmt.query_map(params![layer as u8, limit as i64, offset as i64], row_to_memory)
            .map(|iter| iter.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
    }

    /// List all memories with pagination.
    pub fn list_all(&self, limit: usize, offset: usize) -> Result<Vec<Memory>, EngramError> {
        self.list_all_ns(limit, offset, None)
    }

    pub fn list_all_ns(
        &self,
        limit: usize,
        offset: usize,
        ns: Option<&str>,
    ) -> Result<Vec<Memory>, EngramError> {
        match ns {
            Some(ns) => {
                let conn = self.conn();
                let mut stmt = conn.prepare(
                    "SELECT * FROM memories WHERE namespace = ?3 \
                     ORDER BY last_accessed DESC LIMIT ?1 OFFSET ?2",
                )?;
                let rows: Vec<Memory> = stmt
                    .query_map(params![limit as i64, offset as i64, ns], row_to_memory)?
                    .filter_map(|r| r.ok())
                    .collect();
                Ok(rows)
            }
            None => {
                let conn = self.conn();
                let mut stmt = conn.prepare(
                    "SELECT * FROM memories ORDER BY last_accessed DESC LIMIT ?1 OFFSET ?2",
                )?;
                let rows: Vec<Memory> = stmt
                    .query_map(params![limit as i64, offset as i64], row_to_memory)?
                    .filter_map(|r| r.ok())
                    .collect();
                Ok(rows)
            }
        }
    }

    /// List memories created since a given timestamp, ordered by creation time descending.
    pub fn list_since(&self, since_ms: i64, limit: usize) -> Result<Vec<Memory>, EngramError> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT * FROM memories WHERE created_at >= ?1 ORDER BY created_at DESC LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![since_ms, limit as i64], row_to_memory)?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Find memories whose decay score has fallen below a threshold.
    pub fn get_decayed(&self, threshold: f64) -> Vec<Memory> {
        let now = now_ms();
        let conn = self.conn();
        let Ok(mut stmt) = conn.prepare("SELECT * FROM memories WHERE layer < 3") else {
            return vec![];
        };
        stmt.query_map([], row_to_memory)
            .map(|iter| {
                iter.filter_map(|r| r.ok())
                    .filter(|m| {
                        let hours = (now - m.last_accessed) as f64 / 3_600_000.0;
                        let score = m.importance * (-m.decay_rate * hours / 168.0).exp();
                        score < threshold
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Promote a memory to a higher layer.
    pub fn promote(&self, id: &str, target: Layer) -> Result<Option<Memory>, EngramError> {
        let Some(m) = self.get(id)? else {
            return Ok(None);
        };
        if m.layer as u8 >= target as u8 {
            return Ok(Some(m));
        }

        self.conn().execute(
            "UPDATE memories SET layer = ?1, decay_rate = ?2, last_accessed = ?3 WHERE id = ?4",
            params![target as u8, target.default_decay(), now_ms(), id],
        )?;
        self.get(id)
    }

    pub fn stats(&self) -> Stats {
        let mut s = Stats {
            total: 0,
            buffer: 0,
            working: 0,
            core: 0,
        };
        let conn = self.conn();
        let Ok(mut stmt) = conn
            .prepare("SELECT layer, COUNT(*) FROM memories GROUP BY layer")
        else {
            return s;
        };
        let Ok(rows) = stmt.query_map([], |row| {
            Ok((row.get::<_, u8>(0)?, row.get::<_, i64>(1)? as usize))
        }) else {
            return s;
        };

        for r in rows.flatten() {
            s.total += r.1;
            match r.0 {
                1 => s.buffer = r.1,
                2 => s.working = r.1,
                3 => s.core = r.1,
                _ => {}
            }
        }
        s
    }

    pub fn stats_ns(&self, ns: &str) -> Stats {
        let mut s = Stats { total: 0, buffer: 0, working: 0, core: 0 };
        let conn = self.conn();
        let Ok(mut stmt) = conn.prepare(
            "SELECT layer, COUNT(*) FROM memories WHERE namespace = ?1 GROUP BY layer",
        ) else {
            return s;
        };
        let Ok(rows) = stmt.query_map(params![ns], |row| {
            Ok((row.get::<_, u8>(0)?, row.get::<_, i64>(1)? as usize))
        }) else {
            return s;
        };
        for r in rows.flatten() {
            s.total += r.1;
            match r.0 {
                1 => s.buffer = r.1,
                2 => s.working = r.1,
                3 => s.core = r.1,
                _ => {}
            }
        }
        s
    }    pub fn update_fields(
        &self,
        id: &str,
        content: Option<&str>,
        layer: Option<u8>,
        importance: Option<f64>,
        tags: Option<&[String]>,
    ) -> Result<Option<Memory>, EngramError> {
        if self.get(id)?.is_none() {
            return Ok(None);
        }

        // Validate everything before touching the DB
        if let Some(c) = content {
            if c.trim().is_empty() {
                return Err(EngramError::EmptyContent);
            }
            if c.chars().count() > MAX_CONTENT_LEN {
                return Err(EngramError::ContentTooLong);
            }
        }
        if let Some(l) = layer {
            if !(1..=3).contains(&l) {
                return Err(EngramError::InvalidLayer(l));
            }
        }
        if let Some(t) = tags {
            if t.len() > MAX_TAGS {
                return Err(EngramError::Validation(format!("too many tags (max {MAX_TAGS})")));
            }
            if let Some(tag) = t.iter().find(|tag| tag.chars().count() > MAX_TAG_LEN) {
                return Err(EngramError::Validation(format!("tag '{}' too long", tag)));
            }
        }

        let mut set_clauses: Vec<String> = Vec::new();
        let mut values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(c) = content {
            set_clauses.push("content=?".into());
            values.push(Box::new(c.to_string()));
        }
        if let Some(l) = layer {
            set_clauses.push("layer=?".into());
            values.push(Box::new(l));
        }
        if let Some(i) = importance {
            set_clauses.push("importance=?".into());
            values.push(Box::new(i.clamp(0.0, 1.0)));
        }
        if let Some(t) = tags {
            set_clauses.push("tags=?".into());
            let j = serde_json::to_string(t).unwrap_or_else(|_| "[]".into());
            values.push(Box::new(j));
        }

        if !set_clauses.is_empty() {
            values.push(Box::new(id.to_string()));
            let sql = format!(
                "UPDATE memories SET {} WHERE id=?",
                set_clauses.join(", ")
            );
            let params: Vec<&dyn rusqlite::types::ToSql> = values.iter().map(|v| v.as_ref()).collect();
            self.conn().execute(&sql, params.as_slice())?;
        }

        // Rebuild FTS entry if content or tags changed
        if content.is_some() || tags.is_some() {
            if let Ok(Some(mem)) = self.get(id) {
                self.fts_delete(id)?;
                let tags_json = serde_json::to_string(&mem.tags).unwrap_or_else(|_| "[]".into());
                self.fts_insert(id, &mem.content, &tags_json)?;
            }
        }

        self.get(id)
    }

    pub fn set_embedding(&self, id: &str, embedding: &[f64]) -> Result<(), EngramError> {
        let bytes = crate::ai::embedding_to_bytes(embedding);
        self.conn().execute(
            "UPDATE memories SET embedding = ?1 WHERE id = ?2",
            params![bytes, id],
        )?;
        Ok(())
    }

    pub fn get_all_with_embeddings(&self) -> Vec<(Memory, Vec<f64>)> {
        let conn = self.conn();
        let Ok(mut stmt) = conn
            .prepare("SELECT * FROM memories WHERE embedding IS NOT NULL")
        else {
            return vec![];
        };

        stmt.query_map([], |row| {
            let mem = row_to_memory_with_embedding(row)?;
            Ok(mem)
        })
        .map(|iter| {
            iter.filter_map(|r| r.ok())
                .filter_map(|m| {
                    let emb = m.embedding.clone()?;
                    Some((m, emb))
                })
                .collect()
        })
        .unwrap_or_default()
    }

    /// Semantic search: find memories closest to a query embedding.
    ///
    /// Uses brute-force cosine similarity. Suitable for collections up to ~10k memories.
    /// For larger datasets, consider an external vector index.
    pub fn search_semantic(&self, query_emb: &[f64], limit: usize) -> Vec<(String, f64)> {
        let all = self.get_all_with_embeddings();
        let mut scored: Vec<(String, f64)> = all
            .into_iter()
            .map(|(mem, emb)| {
                let sim = crate::ai::cosine_similarity(query_emb, &emb);
                (mem.id, sim)
            })
            .filter(|(_, sim)| *sim > 0.0)
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        scored
    }

    /// Check if content is a near-duplicate of an existing memory.
    /// Uses token-level Jaccard similarity (threshold ~0.8).
    fn find_near_duplicate(&self, content: &str, ns: &str) -> Option<Memory> {
        // Use only the first 200 chars for the FTS query — enough for similarity matching
        let query_text = if content.len() > 200 {
            &content[..content.char_indices().take(200).last().map(|(i, c)| i + c.len_utf8()).unwrap_or(200)]
        } else {
            content
        };
        let candidates = self.search_fts(query_text, 5);
        if candidates.is_empty() {
            return None;
        }

        let new_tokens = tokenize_for_dedup(content);
        if new_tokens.len() < 3 {
            return None;
        }

        for (id, _) in &candidates {
            if let Ok(Some(mem)) = self.get(id) {
                if mem.namespace != ns {
                    continue;
                }
                let old_tokens = tokenize_for_dedup(&mem.content);
                let intersection = new_tokens.intersection(&old_tokens).count();
                let union = new_tokens.union(&old_tokens).count();
                if union > 0 {
                    let jaccard = intersection as f64 / union as f64;
                    if jaccard > 0.8 {
                        return Some(mem);
                    }
                }
            }
        }
        None
    }

    /// Export all memories as a JSON-serializable vec (for backup/migration).
    /// Embeddings are excluded to keep exports portable.
    pub fn export_all(&self) -> Result<Vec<Memory>, EngramError> {
        self.export_with_embeddings(false)
    }

    pub fn export_with_embeddings(&self, include: bool) -> Result<Vec<Memory>, EngramError> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare("SELECT * FROM memories ORDER BY created_at ASC")?;
        let mapper = if include { row_to_memory_with_embedding } else { row_to_memory };
        let rows = stmt
            .query_map([], mapper)?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Import memories from an export. Skips entries whose id already exists.
    /// Returns count of newly imported memories.
    pub fn import(&self, memories: &[Memory]) -> Result<usize, EngramError> {
        let conn = self.conn();
        conn.execute_batch("BEGIN")?;
        let mut imported = 0;
        for m in memories {
            // check for existing id using the same connection
            let exists: bool = conn.query_row(
                "SELECT COUNT(*) FROM memories WHERE id = ?1",
                params![m.id],
                |r| r.get::<_, i64>(0),
            ).unwrap_or(0) > 0;
            if exists {
                continue;
            }
            let tags_json = serde_json::to_string(&m.tags).unwrap_or_else(|_| "[]".into());
            conn.execute(
                "INSERT INTO memories \
                 (id, content, layer, importance, created_at, last_accessed, \
                  access_count, decay_rate, source, tags) \
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
                params![
                    m.id,
                    m.content,
                    m.layer as u8,
                    m.importance,
                    m.created_at,
                    m.last_accessed,
                    m.access_count,
                    m.decay_rate,
                    m.source,
                    tags_json,
                ],
            )?;
            let processed = append_cjk_bigrams(&m.content);
            conn.execute(
                "INSERT INTO memories_fts(id, content, tags) VALUES (?1, ?2, ?3)",
                params![m.id, processed, tags_json],
            )?;
            imported += 1;
        }
        conn.execute_batch("COMMIT")?;
        Ok(imported)
    }
}

/// Tokenize text for dedup comparison. Whitespace-splits for Latin text,
/// and also generates CJK bigrams so Chinese/Japanese/Korean content gets
/// meaningful tokens instead of nothing.
fn tokenize_for_dedup(text: &str) -> std::collections::HashSet<String> {
    let mut tokens: std::collections::HashSet<String> = text
        .split_whitespace()
        .filter(|w| !w.chars().all(is_cjk)) // pure CJK chunks are handled by bigrams below
        .map(|w| w.to_lowercase())
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

fn row_to_memory(row: &rusqlite::Row) -> rusqlite::Result<Memory> {
    row_to_memory_impl(row, false)
}

fn row_to_memory_with_embedding(row: &rusqlite::Row) -> rusqlite::Result<Memory> {
    row_to_memory_impl(row, true)
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
        decay_rate: row.get("decay_rate")?,
        source: row.get("source")?,
        tags: serde_json::from_str(&tags_str).unwrap_or_default(),
        namespace: row.get::<_, String>("namespace").unwrap_or_else(|_| "default".into()),
        embedding,
    })
}


#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> MemoryDB {
        MemoryDB::open(":memory:").expect("in-memory db")
    }

    #[test]
    fn basic_crud() {
        let db = test_db();
        let mem = db
            .insert(MemoryInput {
                content: "test memory".into(),
                layer: Some(2),
                importance: Some(0.8),
                source: None,
                tags: Some(vec!["test".into()]),
            supersedes: None,
            skip_dedup: None,
            namespace: None,
            sync_embed: None,
            })
            .unwrap();

        assert_eq!(mem.layer, Layer::Working);
        assert!((mem.importance - 0.8).abs() < f64::EPSILON);
        assert_eq!(mem.tags, vec!["test"]);

        let got = db.get(&mem.id).unwrap().unwrap();
        assert_eq!(got.content, "test memory");
    }

    #[test]
    fn delete_missing() {
        let db = test_db();
        assert!(!db.delete("nonexistent").unwrap());
    }

    #[test]
    fn touch() {
        let db = test_db();
        let mem = db
            .insert(MemoryInput {
                content: "touchable".into(),
                layer: None,
                importance: None,
                source: None,
                tags: None,
            supersedes: None,
            skip_dedup: None,
            namespace: None,
            sync_embed: None,
            })
            .unwrap();

        db.touch(&mem.id).unwrap();
        let got = db.get(&mem.id).unwrap().unwrap();
        assert_eq!(got.access_count, 1);
    }

    #[test]
    fn reject_empty() {
        let db = test_db();
        let result = db.insert(MemoryInput {
            content: "   ".into(),
            layer: None,
            importance: None,
            source: None,
            tags: None,
        supersedes: None,
        skip_dedup: None,
        namespace: None,
        sync_embed: None,
        });
        assert!(result.is_err());
    }

    #[test]
    fn reject_bad_layer() {
        let db = test_db();
        let result = db.insert(MemoryInput {
            content: "test".into(),
            layer: Some(5),
            importance: None,
            source: None,
            tags: None,
        supersedes: None,
        skip_dedup: None,
        namespace: None,
        sync_embed: None,
        });
        assert!(result.is_err());
    }

    #[test]
    fn clamp_importance() {
        let db = test_db();
        let mem = db
            .insert(MemoryInput {
                content: "clamped".into(),
                layer: None,
                importance: Some(1.5),
                source: None,
                tags: None,
            supersedes: None,
            skip_dedup: None,
            namespace: None,
            sync_embed: None,
            })
            .unwrap();
        assert!((mem.importance - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn promote_moves_layer_up() {
        let db = test_db();
        let mem = db
            .insert(MemoryInput {
                content: "promotable".into(),
                layer: Some(1),
                importance: Some(0.9),
                source: None,
                tags: None,
            supersedes: None,
            skip_dedup: None,
            namespace: None,
            sync_embed: None,
            })
            .unwrap();

        let promoted = db.promote(&mem.id, Layer::Core).unwrap().unwrap();
        assert_eq!(promoted.layer, Layer::Core);
    }

    #[test]
    fn fts_search_finds_content() {
        let db = test_db();
        db.insert(MemoryInput {
            content: "the quick brown fox jumps".into(),
            layer: None,
            importance: None,
            source: None,
            tags: None,
        supersedes: None,
        skip_dedup: None,
        namespace: None,
        sync_embed: None,
        })
        .unwrap();

        let results = db.search_fts("quick fox", 10);
        assert!(!results.is_empty());
    }

    #[test]
    fn fts_cjk_bigram_search() {
        let db = test_db();
        db.insert(MemoryInput {
            content: "今天天气很好适合出门散步".into(),
            layer: None,
            importance: None,
            source: None,
            tags: None,
        supersedes: None,
        skip_dedup: None,
        namespace: None,
        sync_embed: None,
        })
        .unwrap();

        // Two-char CJK words should match via bigrams
        assert!(!db.search_fts("天气", 10).is_empty(), "天气 should match");
        assert!(!db.search_fts("很好", 10).is_empty(), "很好 should match");
        assert!(!db.search_fts("天气", 10).is_empty(), "天气 should match");
    }

    #[test]
    fn stats() {
        let db = test_db();
        let entries = [
            ("buffer entry alpha", 1),
            ("buffer entry beta zeta", 1),
            ("working entry gamma delta", 2),
            ("core entry epsilon theta", 3),
        ];
        for (content, layer) in entries {
            db.insert(MemoryInput {
                content: content.into(),
                layer: Some(layer),
                importance: None,
                source: None,
                tags: None,
            supersedes: None,
            skip_dedup: None,
            namespace: None,
            sync_embed: None,
            })
            .unwrap();
        }

        let s = db.stats();
        assert_eq!(s.total, 4);
        assert_eq!(s.buffer, 2);
        assert_eq!(s.working, 1);
        assert_eq!(s.core, 1);
    }

    #[test]
    fn partial_update() {
        let db = test_db();
        let mem = db
            .insert(MemoryInput {
                content: "original".into(),
                layer: Some(1),
                importance: Some(0.5),
                source: None,
                tags: None,
            supersedes: None,
            skip_dedup: None,
            namespace: None,
            sync_embed: None,
            })
            .unwrap();

        let updated = db
            .update_fields(&mem.id, Some("updated"), None, Some(0.9), None)
            .unwrap()
            .unwrap();
        assert_eq!(updated.content, "updated");
        assert!((updated.importance - 0.9).abs() < f64::EPSILON);
        assert_eq!(updated.layer, Layer::Buffer); // unchanged
    }

    #[test]
    fn list_all_with_pagination() {
        let db = test_db();
        for i in 0..5 {
            db.insert(MemoryInput {
                content: format!("paginated {i}"),
                layer: None,
                importance: None,
                source: None,
                tags: None,
            supersedes: None,
            skip_dedup: None,
            namespace: None,
            sync_embed: None,
            })
            .unwrap();
        }

        let page1 = db.list_all(3, 0).unwrap();
        assert_eq!(page1.len(), 3);

        let page2 = db.list_all(3, 3).unwrap();
        assert_eq!(page2.len(), 2);
    }

    #[test]
    fn dedup_merges_similar() {
        let db = test_db();
        let original = db
            .insert(MemoryInput {
                content: "engram project uses Rust SQLite FTS5 for memory storage and retrieval system".into(),
                layer: Some(1),
                importance: Some(0.5),
                source: None,
                tags: Some(vec!["habit".into()]),
            supersedes: None,
            skip_dedup: None,
            namespace: None,
            sync_embed: None,
            })
            .unwrap();

        // Insert near-duplicate with one word changed
        let deduped = db
            .insert(MemoryInput {
                content: "engram project uses Rust SQLite FTS5 for memory storage and search system".into(),
                layer: Some(2),
                importance: Some(0.7),
                source: None,
                tags: Some(vec!["preference".into()]),
            supersedes: None,
            skip_dedup: None,
            namespace: None,
            sync_embed: None,
            })
            .unwrap();

        // Should reuse the same id (updated, not new)
        assert_eq!(deduped.id, original.id);
        // Should keep higher importance
        assert!(deduped.importance >= 0.7);
        // Should merge tags
        assert!(deduped.tags.contains(&"habit".into()));
        assert!(deduped.tags.contains(&"preference".into()));
        // Should be promoted to higher layer
        assert_eq!(deduped.layer, Layer::Working);
        // Total count should still be 1
        assert_eq!(db.stats().total, 1);
    }

    #[test]
    fn cjk_dedup_catches_similar_chinese() {
        let db = test_db();
        let original = db
            .insert(MemoryInput {
                content: "今天下午学习了如何使用向量数据库进行语义搜索和检索任务".into(),
                layer: Some(1),
                importance: Some(0.5),
                source: None,
                tags: Some(vec!["学习".into()]),
            supersedes: None,
            skip_dedup: None,
            namespace: None,
            sync_embed: None,
            })
            .unwrap();

        // Same meaning, last word changed
        let result = db
            .insert(MemoryInput {
                content: "今天下午学习了如何使用向量数据库进行语义搜索和检索工作".into(),
                layer: Some(2),
                importance: Some(0.7),
                source: None,
                tags: None,
            supersedes: None,
            skip_dedup: None,
            namespace: None,
            sync_embed: None,
            })
            .unwrap();

        assert_eq!(result.id, original.id, "should dedup CJK near-duplicate");
        assert_eq!(db.stats().total, 1);
    }

    #[test]
    fn tokenize_for_dedup_handles_mixed_text() {
        let tokens = tokenize_for_dedup("hello 世界你好 world");
        assert!(tokens.contains("hello"));
        assert!(tokens.contains("world"));
        // CJK bigrams
        assert!(tokens.contains("世界"));
        assert!(tokens.contains("界你"));
        assert!(tokens.contains("你好"));
    }

    #[test]
    fn update_fields_all_at_once() {
        let db = test_db();
        let mem = db
            .insert(MemoryInput {
                content: "before update".into(),
                layer: Some(1),
                importance: Some(0.3),
                source: None,
                tags: Some(vec!["old".into()]),
            supersedes: None,
            skip_dedup: None,
            namespace: None,
            sync_embed: None,
            })
            .unwrap();

        let new_tags = vec!["new".into(), "shiny".into()];
        let updated = db
            .update_fields(&mem.id, Some("after update"), Some(3), Some(0.95), Some(&new_tags))
            .unwrap()
            .unwrap();

        assert_eq!(updated.content, "after update");
        assert_eq!(updated.layer, Layer::Core);
        assert!((updated.importance - 0.95).abs() < f64::EPSILON);
        assert_eq!(updated.tags, vec!["new", "shiny"]);
    }

    #[test]
    fn row_to_memory_skips_embedding_by_default() {
        let db = test_db();
        let mem = db
            .insert(MemoryInput {
                content: "embedding test".into(),
                layer: None,
                importance: None,
                source: None,
                tags: None,
            supersedes: None,
            skip_dedup: None,
            namespace: None,
            sync_embed: None,
            })
            .unwrap();

        db.set_embedding(&mem.id, &[1.0, 2.0, 3.0]).unwrap();

        // Normal get() should not deserialize the embedding
        let got = db.get(&mem.id).unwrap().unwrap();
        assert!(got.embedding.is_none());

        // But get_all_with_embeddings should have it
        let with_emb = db.get_all_with_embeddings();
        assert_eq!(with_emb.len(), 1);
        assert_eq!(with_emb[0].1, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn supersede_deletes_old() {
        let db = test_db();
        let old = db
            .insert(MemoryInput::new("engram v0.2.1 deployed"))
            .unwrap();
        let old2 = db
            .insert(MemoryInput::new("engram uses port 3917"))
            .unwrap();

        // new memory supersedes the first one
        let new = db
            .insert(
                MemoryInput::new("engram v0.4.0 deployed with /resume endpoint")
                    .supersedes(vec![old.id.clone()]),
            )
            .unwrap();

        assert!(db.get(&old.id).unwrap().is_none(), "old should be deleted");
        assert!(db.get(&old2.id).unwrap().is_some(), "unrelated should stay");
        assert!(db.get(&new.id).unwrap().is_some(), "new should exist");
    }

    #[test]
    fn supersede_multiple() {
        let db = test_db();
        let a = db.insert(MemoryInput::new("fact version 1")).unwrap();
        let b = db.insert(MemoryInput::new("fact version 2")).unwrap();

        let c = db
            .insert(
                MemoryInput::new("fact version 3 (final)")
                    .supersedes(vec![a.id.clone(), b.id.clone()]),
            )
            .unwrap();

        assert!(db.get(&a.id).unwrap().is_none());
        assert!(db.get(&b.id).unwrap().is_none());
        assert_eq!(db.get(&c.id).unwrap().unwrap().content, "fact version 3 (final)");
    }

    #[test]
    fn skip_dedup_allows_similar() {
        let db = test_db();
        let a = db.insert(MemoryInput::new("the sky is blue today")).unwrap();
        // Without skip_dedup, this would merge into `a`
        let b = db
            .insert(MemoryInput::new("the sky is blue today").skip_dedup())
            .unwrap();
        assert_ne!(a.id, b.id, "should create separate memories");
        assert!(db.get(&a.id).unwrap().is_some());
        assert!(db.get(&b.id).unwrap().is_some());
    }

    #[test]
    fn namespace_isolation() {
        let db = test_db();
        let a = db
            .insert(MemoryInput::new("agent-a's secret").namespace("agent-a"))
            .unwrap();
        let b = db
            .insert(MemoryInput::new("agent-b's data").namespace("agent-b"))
            .unwrap();
        let c = db
            .insert(MemoryInput::new("default ns memory"))
            .unwrap();

        // list_all_ns filters by namespace
        let a_mems = db.list_all_ns(50, 0, Some("agent-a")).unwrap();
        assert_eq!(a_mems.len(), 1);
        assert_eq!(a_mems[0].id, a.id);

        let b_mems = db.list_all_ns(50, 0, Some("agent-b")).unwrap();
        assert_eq!(b_mems.len(), 1);
        assert_eq!(b_mems[0].id, b.id);

        // default namespace
        let def_mems = db.list_all_ns(50, 0, Some("default")).unwrap();
        assert_eq!(def_mems.len(), 1);
        assert_eq!(def_mems[0].id, c.id);

        // no namespace filter returns all
        let all = db.list_all_ns(50, 0, None).unwrap();
        assert_eq!(all.len(), 3);

        // stats_ns
        let s = db.stats_ns("agent-a");
        assert_eq!(s.total, 1);
    }

    #[test]
    fn dedup_respects_namespace() {
        let db = test_db();
        let content = "this is identical content for dedup testing purposes";
        let a = db
            .insert(MemoryInput::new(content).namespace("ns-a"))
            .unwrap();
        // Same content in a different namespace should NOT be deduped
        let b = db
            .insert(MemoryInput::new(content).namespace("ns-b"))
            .unwrap();
        assert_ne!(a.id, b.id, "different namespaces should not dedup");

        // Same content in the same namespace SHOULD be deduped (updates existing)
        let a2 = db
            .insert(MemoryInput::new(content).namespace("ns-a"))
            .unwrap();
        assert_eq!(a.id, a2.id, "same namespace should dedup");

        assert_eq!(db.list_all_ns(50, 0, Some("ns-a")).unwrap().len(), 1);
        assert_eq!(db.list_all_ns(50, 0, Some("ns-b")).unwrap().len(), 1);
    }
}
