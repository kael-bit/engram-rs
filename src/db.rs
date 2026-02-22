//! SQLite-backed memory storage with FTS5 full-text search.

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::EngramError;


const MAX_CONTENT_LEN: usize = 8192;
const MAX_SOURCE_LEN: usize = 64;
const MAX_TAGS: usize = 20;
const MAX_TAG_LEN: usize = 32;


/// Three cognitive memory layers with increasing permanence.
///
/// Based on the Atkinson-Shiffrin model: sensory (buffer) →
/// short-term (working) → long-term (core). Each layer has a different
/// default decay rate and scoring bonus during recall.
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
    #[serde(skip)]
    pub embedding: Option<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryInput {
    pub content: String,
    pub layer: Option<u8>,
    pub importance: Option<f64>,
    pub source: Option<String>,
    pub tags: Option<Vec<String>>,
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
    embedding BLOB
);

CREATE INDEX IF NOT EXISTS idx_layer ON memories(layer);
CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance);
CREATE INDEX IF NOT EXISTS idx_last_accessed ON memories(last_accessed);
"#;

const FTS_SCHEMA: &str =
    "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(\
     id UNINDEXED, content, tags, tokenize='unicode61')";

const TRIGGERS: [&str; 3] = [
    "CREATE TRIGGER IF NOT EXISTS mem_ai AFTER INSERT ON memories \
     BEGIN INSERT INTO memories_fts(id, content, tags) \
     VALUES (new.id, new.content, new.tags); END",
    "CREATE TRIGGER IF NOT EXISTS mem_ad AFTER DELETE ON memories \
     BEGIN DELETE FROM memories_fts WHERE id = old.id; END",
    "CREATE TRIGGER IF NOT EXISTS mem_au AFTER UPDATE OF content, tags ON memories \
     BEGIN DELETE FROM memories_fts WHERE id = old.id; \
     INSERT INTO memories_fts(id, content, tags) \
     VALUES (new.id, new.content, new.tags); END",
];


/// SQLite-backed memory store.
pub struct MemoryDB {
    conn: Connection,
}

impl MemoryDB {
    /// Open (or create) a database at the given path.
    pub fn open(path: &str) -> Result<Self, EngramError> {
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        conn.execute_batch(SCHEMA)?;
        conn.execute(FTS_SCHEMA, [])?;
        for t in &TRIGGERS {
            conn.execute(t, [])?;
        }
        // Migration: add embedding column if missing
        if conn.prepare("SELECT embedding FROM memories LIMIT 0").is_err() {
            conn.execute("ALTER TABLE memories ADD COLUMN embedding BLOB", [])?;
        }
        Ok(Self { conn })
    }

    pub fn insert(&self, input: MemoryInput) -> Result<Memory, EngramError> {
        validate_input(&input)?;

        let now = now_ms();
        let layer_val = input.layer.unwrap_or(1);
        let layer: Layer = layer_val.try_into()?;
        let id = Uuid::new_v4().to_string();
        let importance = input.importance.unwrap_or(0.5).clamp(0.0, 1.0);
        let source = input.source.unwrap_or_else(|| "api".into());
        let tags = input.tags.unwrap_or_default();
        let tags_json = serde_json::to_string(&tags).unwrap_or_else(|_| "[]".into());

        self.conn.execute(
            "INSERT INTO memories \
             (id, content, layer, importance, created_at, last_accessed, \
              access_count, decay_rate, source, tags) \
             VALUES (?1,?2,?3,?4,?5,?6,0,?7,?8,?9)",
            params![
                id,
                input.content,
                layer_val,
                importance,
                now,
                now,
                layer.default_decay(),
                source,
                tags_json
            ],
        )?;

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
            embedding: None,
        })
    }

    pub fn get(&self, id: &str) -> Result<Option<Memory>, EngramError> {
        let mut stmt = self.conn.prepare("SELECT * FROM memories WHERE id = ?1")?;
        let mut rows = stmt.query(params![id])?;
        match rows.next()? {
            Some(row) => Ok(Some(row_to_memory(row)?)),
            None => Ok(None),
        }
    }

    pub fn delete(&self, id: &str) -> Result<bool, EngramError> {
        let n = self.conn.execute("DELETE FROM memories WHERE id = ?1", params![id])?;
        Ok(n > 0)
    }

    pub fn touch(&self, id: &str) -> Result<(), EngramError> {
        self.conn.execute(
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

        // FTS5 unicode61 splits CJK into single chars; use OR for partial matching.
        let fts_query: String = sanitized.split_whitespace().collect::<Vec<_>>().join(" OR ");

        let Ok(mut stmt) = self.conn.prepare(
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
    pub fn list_by_layer(&self, layer: Layer) -> Vec<Memory> {
        let Ok(mut stmt) = self.conn.prepare(
            "SELECT * FROM memories WHERE layer = ?1 ORDER BY importance DESC",
        ) else {
            return vec![];
        };

        stmt.query_map(params![layer as u8], row_to_memory)
            .map(|iter| iter.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
    }

    /// List all memories with pagination.
    pub fn list_all(&self, limit: usize, offset: usize) -> Result<Vec<Memory>, EngramError> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM memories ORDER BY last_accessed DESC LIMIT ?1 OFFSET ?2",
        )?;
        let rows = stmt
            .query_map(params![limit as i64, offset as i64], |row| {
                row_to_memory(row)
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Find memories whose decay score has fallen below a threshold.
    pub fn get_decayed(&self, threshold: f64) -> Vec<Memory> {
        let now = now_ms();
        let Ok(mut stmt) = self.conn.prepare("SELECT * FROM memories WHERE layer < 3") else {
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

        self.conn.execute(
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
        let Ok(mut stmt) = self
            .conn
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

    pub fn update_fields(
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

        if let Some(c) = content {
            if c.trim().is_empty() {
                return Err(EngramError::EmptyContent);
            }
            if c.chars().count() > MAX_CONTENT_LEN {
                return Err(EngramError::ContentTooLong);
            }
            self.conn
                .execute("UPDATE memories SET content=?1 WHERE id=?2", params![c, id])?;
        }
        if let Some(l) = layer {
            if !(1..=3).contains(&l) {
                return Err(EngramError::InvalidLayer(l));
            }
            self.conn
                .execute("UPDATE memories SET layer=?1 WHERE id=?2", params![l, id])?;
        }
        if let Some(i) = importance {
            let clamped = i.clamp(0.0, 1.0);
            self.conn.execute(
                "UPDATE memories SET importance=?1 WHERE id=?2",
                params![clamped, id],
            )?;
        }
        if let Some(t) = tags {
            if t.len() > MAX_TAGS {
                return Err(EngramError::Validation(format!("too many tags (max {MAX_TAGS})")));
            }
            if let Some(tag) = t.iter().find(|tag| tag.chars().count() > MAX_TAG_LEN) {
                return Err(EngramError::Validation(format!("tag '{}' too long", tag)));
            }
            let j = serde_json::to_string(t).unwrap_or_else(|_| "[]".into());
            self.conn
                .execute("UPDATE memories SET tags=?1 WHERE id=?2", params![j, id])?;
        }
        self.get(id)
    }

    pub fn set_embedding(&self, id: &str, embedding: &[f64]) -> Result<(), EngramError> {
        let bytes = crate::ai::embedding_to_bytes(embedding);
        self.conn.execute(
            "UPDATE memories SET embedding = ?1 WHERE id = ?2",
            params![bytes, id],
        )?;
        Ok(())
    }

    pub fn get_all_with_embeddings(&self) -> Vec<(Memory, Vec<f64>)> {
        let Ok(mut stmt) = self
            .conn
            .prepare("SELECT * FROM memories WHERE embedding IS NOT NULL")
        else {
            return vec![];
        };

        stmt.query_map([], |row| {
            let mem = row_to_memory(row)?;
            let blob: Vec<u8> = row.get("embedding")?;
            let emb = crate::ai::bytes_to_embedding(&blob);
            Ok((mem, emb))
        })
        .map(|iter| iter.filter_map(|r| r.ok()).collect())
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
}

fn row_to_memory(row: &rusqlite::Row) -> rusqlite::Result<Memory> {
    let layer_val: u8 = row.get("layer")?;
    let tags_str: String = row.get("tags")?;
    let emb_blob: Option<Vec<u8>> = row.get("embedding").ok();
    let embedding = emb_blob.map(|b| crate::ai::bytes_to_embedding(&b));
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
        })
        .unwrap();

        let results = db.search_fts("quick fox", 10);
        assert!(!results.is_empty());
    }

    #[test]
    fn stats() {
        let db = test_db();
        for layer in [1, 1, 2, 3] {
            db.insert(MemoryInput {
                content: format!("mem layer {layer}"),
                layer: Some(layer),
                importance: None,
                source: None,
                tags: None,
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
            })
            .unwrap();
        }

        let page1 = db.list_all(3, 0).unwrap();
        assert_eq!(page1.len(), 3);

        let page2 = db.list_all(3, 3).unwrap();
        assert_eq!(page2.len(), 2);
    }
}
