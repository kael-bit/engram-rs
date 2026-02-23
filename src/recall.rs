//! Hybrid recall: semantic + keyword retrieval with budget awareness.

use crate::ai::{self, AiConfig};
use crate::db::{is_cjk, Layer, Memory, MemoryDB, ScoredMemory};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use tracing::warn;

// scoring weights — should add up to 1.0
// relevance is king: a perfectly relevant low-importance memory
// beats a vaguely related high-importance one
const WEIGHT_RELEVANCE: f64 = 0.6;
const WEIGHT_IMPORTANCE: f64 = 0.2;
const WEIGHT_RECENCY: f64 = 0.2;

/// Recall request parameters.
#[derive(Debug, Default, Clone, Deserialize)]
pub struct RecallRequest {
    #[serde(default)]
    pub query: String,
    pub budget_tokens: Option<usize>,
    pub layers: Option<Vec<u8>>,
    pub min_importance: Option<f64>,
    pub limit: Option<usize>,
    /// Only include memories created at or after this timestamp (unix ms).
    pub since: Option<i64>,
    /// Only include memories created at or before this timestamp (unix ms).
    pub until: Option<i64>,
    /// Sort order: "score" (default), "recent" (by created_at desc), "accessed" (by last_accessed desc).
    pub sort_by: Option<String>,
    /// Whether to use LLM to re-rank results.
    pub rerank: Option<bool>,
    /// Filter by source (e.g. "session", "extract", "api").
    pub source: Option<String>,
    /// Filter by tags — memory must have ALL specified tags.
    pub tags: Option<Vec<String>>,
    /// Filter by namespace.
    pub namespace: Option<String>,
    /// Expand query with LLM-generated synonyms for better coverage.
    pub expand: Option<bool>,
    /// Drop results below this score threshold (0.0-1.0).
    pub min_score: Option<f64>,
}

/// Recall response with scored memories and metadata.
#[derive(Debug, serde::Serialize)]
pub struct RecallResponse {
    pub memories: Vec<ScoredMemory>,
    pub total_tokens: usize,
    pub layer_breakdown: HashMap<u8, usize>,
    pub search_mode: String,
    /// Applied time filter, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_filter: Option<TimeFilter>,
    /// LLM-expanded query variants, if expand was requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expanded_queries: Option<Vec<String>>,
}

#[derive(Debug, serde::Serialize)]
pub struct TimeFilter {
    pub since: Option<i64>,
    pub until: Option<i64>,
    pub sort_by: String,
}

/// CJK chars ≈ 1.5 tokens each, Latin text ≈ 4 bytes per token.
pub fn estimate_tokens(text: &str) -> usize {
    let mut cjk_count = 0_usize;
    let mut other_bytes = 0_usize;
    for c in text.chars() {
        if is_cjk(c) {
            cjk_count += 1;
        } else {
            other_bytes += c.len_utf8();
        }
    }
    let t = cjk_count as f64 / 1.5 + other_bytes as f64 / 4.0;
    t.ceil().max(1.0) as usize
}

/// Calculate recency score using exponential decay.
fn recency_score(last_accessed: i64, decay_rate: f64) -> f64 {
    let now = crate::db::now_ms();
    let hours = ((now - last_accessed) as f64 / 3_600_000.0).max(0.0);
    (-decay_rate * hours / 168.0).exp()
}

#[cfg(test)]
fn score_combined(importance: f64, relevance: f64, last_accessed: i64) -> f64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;
    let age_hours = ((now - last_accessed) as f64 / 3_600_000.0).max(0.0);
    // simplified recency for rescore — uses default decay
    let recency = (-0.1 * age_hours).exp();
    WEIGHT_IMPORTANCE * importance + WEIGHT_RECENCY * recency + WEIGHT_RELEVANCE * relevance
}

fn score_memory(mem: &Memory, relevance: f64) -> ScoredMemory {
    let recency = recency_score(mem.last_accessed, mem.decay_rate);
    let bonus = mem.layer.score_bonus();
    let score =
        (WEIGHT_IMPORTANCE * mem.importance + WEIGHT_RECENCY * recency + WEIGHT_RELEVANCE * relevance) * bonus;

    ScoredMemory {
        memory: mem.clone(),
        score,
        relevance,
        recency,
    }
}

/// Run a hybrid recall query.
///
/// Combines semantic search (if `query_emb` is provided) with FTS5 keyword search,
/// always includes core memories, then selects the best results within the token budget.
pub fn recall(
    db: &MemoryDB,
    req: &RecallRequest,
    query_emb: Option<&[f64]>,
    extra_queries: Option<&[String]>,
) -> RecallResponse {
    let budget = req.budget_tokens.unwrap_or(2000);
    let limit = req.limit.unwrap_or(20).min(100);
    let min_imp = req.min_importance.unwrap_or(0.0);
    let sort_by = req.sort_by.as_deref().unwrap_or("score");
    let layer_filter: Option<Vec<Layer>> = req.layers.as_ref().map(|ls| {
        ls.iter()
            .filter_map(|&l| l.try_into().ok())
            .collect()
    });

    // Time filter closure
    let time_ok = |mem: &Memory| -> bool {
        if let Some(since) = req.since {
            if mem.created_at < since {
                return false;
            }
        }
        if let Some(until) = req.until {
            if mem.created_at > until {
                return false;
            }
        }
        true
    };

    let passes_filters = |mem: &Memory| -> bool {
        if mem.importance < min_imp {
            return false;
        }
        if !time_ok(mem) {
            return false;
        }
        if let Some(ref lf) = layer_filter {
            if !lf.contains(&mem.layer) {
                return false;
            }
        }
        if let Some(ref src) = req.source {
            if mem.source != *src {
                return false;
            }
        }
        if let Some(ref required_tags) = req.tags {
            if !required_tags.iter().all(|t| mem.tags.contains(t)) {
                return false;
            }
        }
        if let Some(ref ns) = req.namespace {
            if mem.namespace != *ns {
                return false;
            }
        }
        true
    };

    let mut scored: Vec<ScoredMemory> = Vec::new();
    let mut seen = HashSet::new();
    let mut search_mode = "fts".to_string();

    // Semantic search (if embedding available)
    if let Some(qemb) = query_emb {
        let semantic_results = db.search_semantic_ns(qemb, limit * 3, req.namespace.as_deref());
        if !semantic_results.is_empty() {
            search_mode = "semantic+fts".to_string();
            let max_sim = semantic_results
                .iter()
                .map(|r| r.1)
                .fold(0.001_f64, f64::max);
            for (id, sim) in &semantic_results {
                if let Ok(Some(mem)) = db.get(id) {
                    if !passes_filters(&mem) {
                        continue;
                    }
                    let relevance = sim / max_sim;
                    seen.insert(id.clone());
                    scored.push(score_memory(&mem, relevance));
                }
            }
        }
    }

    // FTS keyword search (always runs)
    let fts = db.search_fts(&req.query, limit * 3);
    let max_bm25 = fts.iter().map(|r| r.1).fold(0.001_f64, f64::max);

    // O(1) lookup: memory id → index in scored
    let scored_idx: HashMap<String, usize> = scored
        .iter()
        .enumerate()
        .map(|(i, s)| (s.memory.id.clone(), i))
        .collect();

    for (id, bm25) in &fts {
        let fts_rel = bm25 / max_bm25;
        if seen.contains(id) {
            // Boost: found by both semantic AND keyword — strong relevance signal.
            // Use multiplicative boost so it still helps even when relevance is already high.
            if let Some(&idx) = scored_idx.get(id) {
                let sm = &mut scored[idx];
                let boost = 1.0 + fts_rel * 0.3;  // 1.0 to 1.3x multiplier
                sm.relevance = (sm.relevance * boost).min(1.5);
                // rescore using the full scoring formula (decay_rate + layer bonus)
                let rescored = score_memory(&sm.memory, sm.relevance);
                sm.score = rescored.score;
                sm.recency = rescored.recency;
            }
            continue;
        }
        if let Ok(Some(mem)) = db.get(id) {
            if !passes_filters(&mem) {
                continue;
            }
            seen.insert(id.clone());
            scored.push(score_memory(&mem, fts_rel));
        }
    }

    // Expanded FTS queries (from LLM query expansion)
    if let Some(queries) = extra_queries {
        for eq in queries {
            let extra_fts = db.search_fts(eq, limit);
            let emax = extra_fts.iter().map(|r| r.1).fold(0.001_f64, f64::max);
            for (id, bm25) in &extra_fts {
                if seen.contains(id) {
                    continue;
                }
                if let Ok(Some(mem)) = db.get(id) {
                    if !passes_filters(&mem) {
                        continue;
                    }
                    // slight penalty so direct matches rank higher
                    let relevance = (bm25 / emax) * 0.85;
                    seen.insert(id.clone());
                    scored.push(score_memory(&mem, relevance));
                }
            }
        }
    }

    // Pad with unseen core memories when we don't have enough hits.
    // Low relevance=0.1 means they won't outrank real matches.
    if scored.len() < limit {
        for mem in db.list_by_layer(Layer::Core, (limit - scored.len()) * 3, 0) {
            if seen.contains(&mem.id) {
                continue;
            }
            if !passes_filters(&mem) {
                continue;
            }
            scored.push(score_memory(&mem, 0.1));
        }
    }

    // Sort based on requested order
    match sort_by {
        "recent" => scored.sort_by(|a, b| {
            b.memory.created_at.cmp(&a.memory.created_at)
        }),
        "accessed" => scored.sort_by(|a, b| {
            b.memory.last_accessed.cmp(&a.memory.last_accessed)
        }),
        _ => scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
    }

    // Budget-aware selection
    let mut selected = Vec::new();
    let mut total_tokens = 0;
    let mut breakdown = HashMap::new();
    let min_score = req.min_score.unwrap_or(0.0);

    for sm in scored {
        if sm.score < min_score {
            continue;
        }
        let tokens = estimate_tokens(&sm.memory.content);
        if total_tokens + tokens > budget {
            // always return at least one result, unless budget is explicitly 0
            if budget == 0 {
                break;
            }
            if !selected.is_empty() {
                continue;
            }
        }
        if selected.len() >= limit {
            break;
        }

        let layer_key = sm.memory.layer as u8;
        *breakdown.entry(layer_key).or_insert(0) += 1;
        total_tokens += tokens;

        // Only bump access stats for genuinely relevant results
        if sm.relevance > 0.5 {
            let _ = db.touch(&sm.memory.id);
        }
        selected.push(sm);
    }

    let time_filter = if req.since.is_some() || req.until.is_some() || sort_by != "score" {
        Some(TimeFilter {
            since: req.since,
            until: req.until,
            sort_by: sort_by.to_string(),
        })
    } else {
        None
    };

    RecallResponse {
        memories: selected,
        total_tokens,
        layer_breakdown: breakdown,
        search_mode,
        time_filter,
        expanded_queries: None,
    }
}

const RERANK_SYSTEM: &str = "Given a query and numbered memories, return the numbers \
    sorted by relevance (most relevant first). Output only comma-separated numbers.";

/// Re-rank recall results using an LLM. Falls back to original order on failure.
pub async fn rerank_results(
    response: &mut RecallResponse,
    query: &str,
    limit: usize,
    cfg: &AiConfig,
) {
    if response.memories.len() < 2 {
        return;
    }

    let mut numbered = String::new();
    for (i, sm) in response.memories.iter().enumerate() {
        use std::fmt::Write;
        let _ = writeln!(numbered, "{}. {}", i + 1, sm.memory.content);
    }

    let user = format!("Query: {query}\n\nMemories:\n{numbered}");

    match ai::llm_chat_as(cfg, "rerank", RERANK_SYSTEM, &user).await {
        Ok(raw) => {
            let order = parse_rerank_response(&raw, response.memories.len());
            if order.is_empty() {
                return;
            }

            let old = std::mem::take(&mut response.memories);
            let mut seen = HashSet::new();
            for idx in &order {
                if seen.insert(*idx) && *idx < old.len() {
                    response.memories.push(old[*idx].clone());
                }
            }
            // append any the LLM didn't mention
            for (i, sm) in old.into_iter().enumerate() {
                if !seen.contains(&i) {
                    response.memories.push(sm);
                }
            }
            response.memories.truncate(limit);
            response.search_mode = format!("{} +rerank", response.search_mode);
        }
        Err(e) => {
            warn!(error = %e, "LLM rerank failed, keeping original order");
        }
    }
}

fn parse_rerank_response(raw: &str, count: usize) -> Vec<usize> {
    raw.split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<usize>().ok())
        .filter(|&n| n >= 1 && n <= count)
        .map(|n| n - 1)
        .collect()
}

/// Check if content is semantically duplicate of an existing memory.
/// Uses embedding cosine similarity (threshold 0.78).
pub async fn quick_semantic_dup(
    ai_cfg: &AiConfig,
    db: &MemoryDB,
    content: &str,
) -> Result<bool, String> {
    let embeddings = ai::get_embeddings(ai_cfg, &[content.to_string()]).await?;
    let emb = embeddings.first().ok_or("no embedding returned")?;
    let candidates = db.search_semantic(emb, 3);
    for (_, score) in &candidates {
        if *score > 0.78 {
            return Ok(true);
        }
    }
    Ok(false)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_ascii() {
        // "hello world" = 11 bytes / 4 ≈ 3 tokens
        let tokens = estimate_tokens("hello world");
        assert!((2..=4).contains(&tokens));
    }

    #[test]
    fn tokens_cjk() {
        // 4 CJK chars / 1.5 ≈ 3 tokens
        let tokens = estimate_tokens("你好世界");
        assert!((2..=4).contains(&tokens));
    }

    #[test]
    fn tokens_mixed() {
        let tokens = estimate_tokens("hello 你好");
        assert!(tokens >= 2);
    }

    #[test]
    fn parse_rerank_basic() {
        assert_eq!(parse_rerank_response("3, 1, 2", 3), vec![2, 0, 1]);
    }

    #[test]
    fn parse_rerank_with_noise() {
        // number 5 is out of range (count=3), should be filtered
        assert_eq!(parse_rerank_response("3,1,2,5", 3), vec![2, 0, 1]);
    }

    #[test]
    fn parse_rerank_newlines() {
        assert_eq!(parse_rerank_response("2\n1\n3", 3), vec![1, 0, 2]);
    }

    #[test]
    fn parse_rerank_empty() {
        assert!(parse_rerank_response("no numbers here", 3).is_empty());
    }

    // --- recall integration tests ---

    use crate::db::{MemoryDB, MemoryInput};

    fn test_db_with_data() -> MemoryDB {
        let db = MemoryDB::open(":memory:").expect("in-memory db");
        db.insert(MemoryInput {
            content: "very important fact about rust".into(),
            layer: Some(3),
            importance: Some(0.95),
            ..Default::default()
        }).unwrap();
        db.insert(MemoryInput {
            content: "trivial note about lunch".into(),
            layer: Some(1),
            importance: Some(0.1),
            ..Default::default()
        }).unwrap();
        db.insert(MemoryInput {
            content: "medium importance work log about rust compiler".into(),
            layer: Some(2),
            importance: Some(0.5),
            ..Default::default()
        }).unwrap();
        db
    }

    #[test]
    fn score_favors_important() {
        let db = test_db_with_data();
        let req = RecallRequest {
            query: "rust".into(),
            budget_tokens: Some(2000),
            layers: None,
            min_importance: None,
            limit: Some(10),
            since: None,
            until: None,
            sort_by: None,
            rerank: None, source: None, tags: None,
            namespace: None,
            expand: None,
            ..Default::default()
        };
        let result = recall(&db, &req, None, None);
        assert!(result.memories.len() >= 2);
        // "very important fact" should score higher than "medium importance"
        let first = &result.memories[0];
        assert!(first.memory.content.contains("very important"), "highest importance should rank first");
    }

    #[test]
    fn budget_limits_output() {
        let db = test_db_with_data();
        // tiny budget — should get at most 1 result
        let req = RecallRequest {
            query: "rust".into(),
            budget_tokens: Some(5),
            layers: None,
            min_importance: None,
            limit: Some(10),
            since: None,
            until: None,
            sort_by: None,
            rerank: None, source: None, tags: None,
            namespace: None,
            expand: None,
            ..Default::default()
        };
        let result = recall(&db, &req, None, None);
        assert!(result.memories.len() <= 2, "budget should limit results");
        assert!(result.total_tokens <= 10, "shouldn't overshoot budget by much");
    }

    #[test]
    fn budget_zero_returns_empty() {
        let db = test_db_with_data();
        let req = RecallRequest {
            query: "rust".into(),
            budget_tokens: Some(0),
            layers: None,
            min_importance: None,
            limit: Some(10),
            since: None,
            until: None,
            sort_by: None,
            rerank: None, source: None, tags: None,
            namespace: None,
            expand: None,
            ..Default::default()
        };
        let result = recall(&db, &req, None, None);
        assert!(result.memories.is_empty());
    }

    #[test]
    fn time_filter_since() {
        let db = MemoryDB::open(":memory:").expect("in-memory db");
        // insert two memories — we can't control created_at via MemoryInput,
        // but both will have "now" timestamps. Import lets us set timestamps.
        let now = crate::db::now_ms();
        let old = Memory {
            id: "old-one".into(),
            content: "old memory about testing".into(),
            layer: Layer::Core,
            importance: 0.9,
            created_at: now - 86_400_000, // yesterday
            last_accessed: now - 86_400_000,
            access_count: 5,
            decay_rate: 0.05,
            source: "test".into(),
            tags: vec![],
            namespace: "default".into(),
            embedding: None,
        };
        let recent = Memory {
            id: "new-one".into(),
            content: "recent memory about testing".into(),
            layer: Layer::Core,
            importance: 0.9,
            created_at: now - 1000,
            last_accessed: now,
            access_count: 1,
            decay_rate: 0.05,
            source: "test".into(),
            tags: vec![],
            namespace: "default".into(),
            embedding: None,
        };
        db.import(&[old, recent]).unwrap();

        let req = RecallRequest {
            query: "testing".into(),
            budget_tokens: Some(2000),
            layers: None,
            min_importance: None,
            limit: Some(10),
            since: Some(now - 3_600_000), // last hour only
            until: None,
            sort_by: None,
            rerank: None, source: None, tags: None,
            namespace: None,
            expand: None,
            ..Default::default()
        };
        let result = recall(&db, &req, None, None);
        assert_eq!(result.memories.len(), 1);
        assert_eq!(result.memories[0].memory.id, "new-one");
    }

    #[test]
    fn filter_by_source() {
        let db = MemoryDB::open(":memory:").expect("in-memory db");
        db.insert(MemoryInput {
            content: "from the API".into(),
            layer: Some(3), importance: Some(0.9),
            source: Some("api".into()), tags: None,
            ..Default::default()
        }).unwrap();
        db.insert(MemoryInput {
            content: "from a session".into(),
            layer: Some(2), importance: Some(0.7),
            source: Some("session".into()), tags: None,
            ..Default::default()
        }).unwrap();

        let req = RecallRequest {
            query: "from".into(),
            budget_tokens: Some(2000),
            layers: None, min_importance: None, limit: Some(10),
            since: None, until: None, sort_by: None, rerank: None,
            source: Some("session".into()), tags: None,
            namespace: None,
            expand: None,
            ..Default::default()
        };
        let result = recall(&db, &req, None, None);
        assert_eq!(result.memories.len(), 1);
        assert!(result.memories[0].memory.content.contains("session"));
    }

    #[test]
    fn filter_by_tags() {
        let db = MemoryDB::open(":memory:").expect("in-memory db");
        db.insert(MemoryInput {
            content: "rust project details".into(),
            layer: Some(3), importance: Some(0.9),
            source: None, tags: Some(vec!["rust".into(), "engram".into()]),
        ..Default::default()
        }).unwrap();
        db.insert(MemoryInput {
            content: "python script notes".into(),
            layer: Some(2), importance: Some(0.7),
            source: None, tags: Some(vec!["python".into()]),
        ..Default::default()
        }).unwrap();

        let req = RecallRequest {
            query: "project".into(),
            budget_tokens: Some(2000),
            layers: None, min_importance: None, limit: Some(10),
            since: None, until: None, sort_by: None, rerank: None,
            source: None, tags: Some(vec!["rust".into()]),
            namespace: None,
            expand: None,
            ..Default::default()
        };
        let result = recall(&db, &req, None, None);
        assert_eq!(result.memories.len(), 1);
        assert!(result.memories[0].memory.content.contains("rust"));
    }

    #[test]
    fn score_combined_recalc() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        let recent = score_combined(0.8, 0.9, now);
        let old = score_combined(0.8, 0.9, now - 72 * 3_600_000);
        assert!(recent > old, "recent memory should score higher");
        assert!(recent > 0.0);
    }

    #[test]
    fn min_score_filters_low() {
        let db = test_db_with_data();
        // Without min_score, we get results
        let req_all = RecallRequest {
            query: "rust".into(),
            limit: Some(10),
            ..Default::default()
        };
        let all = recall(&db, &req_all, None, None);
        assert!(all.memories.len() >= 2);

        // With impossibly high min_score, we get nothing
        let req_high = RecallRequest {
            query: "rust".into(),
            limit: Some(10),
            min_score: Some(99.0),
            ..Default::default()
        };
        let filtered = recall(&db, &req_high, None, None);
        assert_eq!(filtered.memories.len(), 0, "min_score=99 should filter everything");
    }

    #[test]
    fn empty_query_returns_something() {
        let db = test_db_with_data();
        let req = RecallRequest {
            query: "".into(),
            limit: Some(5),
            ..Default::default()
        };
        // Empty query shouldn't panic — FTS returns nothing, but that's fine
        let result = recall(&db, &req, None, None);
        // May or may not find results, but must not crash
        assert!(result.memories.len() <= 5);
    }

    #[test]
    fn sort_by_recent() {
        let db = MemoryDB::open(":memory:").expect("in-memory db");
        db.insert(MemoryInput {
            content: "older entry about databases".into(),
            importance: Some(0.9),
            ..Default::default()
        }).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        db.insert(MemoryInput {
            content: "newer entry about databases".into(),
            importance: Some(0.3),
            ..Default::default()
        }).unwrap();

        let req = RecallRequest {
            query: "databases".into(),
            sort_by: Some("recent".into()),
            limit: Some(2),
            ..Default::default()
        };
        let result = recall(&db, &req, None, None);
        assert_eq!(result.memories.len(), 2);
        assert!(result.memories[0].memory.content.contains("newer"),
            "sort_by=recent should put newer first");
    }

    #[test]
    fn namespace_isolation() {
        let db = MemoryDB::open(":memory:").expect("in-memory db");
        db.insert(MemoryInput {
            content: "secret agent data".into(),
            namespace: Some("agent-a".into()),
            ..Default::default()
        }).unwrap();
        db.insert(MemoryInput {
            content: "different agent data".into(),
            namespace: Some("agent-b".into()),
            ..Default::default()
        }).unwrap();

        let req = RecallRequest {
            query: "agent data".into(),
            namespace: Some("agent-a".into()),
            limit: Some(10),
            ..Default::default()
        };
        let result = recall(&db, &req, None, None);
        assert_eq!(result.memories.len(), 1);
        assert!(result.memories[0].memory.content.contains("secret"));
    }

    #[test]
    fn since_until_filters() {
        let db = MemoryDB::open(":memory:").expect("in-memory db");
        let _m1 = db.insert(MemoryInput {
            content: "early log entry".into(),
            ..Default::default()
        }).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        let midpoint = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        std::thread::sleep(std::time::Duration::from_millis(50));
        db.insert(MemoryInput {
            content: "late log entry".into(),
            ..Default::default()
        }).unwrap();

        // Only memories after midpoint
        let req = RecallRequest {
            query: "log entry".into(),
            since: Some(midpoint),
            limit: Some(10),
            ..Default::default()
        };
        let result = recall(&db, &req, None, None);
        assert_eq!(result.memories.len(), 1);
        assert!(result.memories[0].memory.content.contains("late"));

        // Only memories before midpoint
        let req2 = RecallRequest {
            query: "log entry".into(),
            until: Some(midpoint),
            limit: Some(10),
            ..Default::default()
        };
        let result2 = recall(&db, &req2, None, None);
        assert_eq!(result2.memories.len(), 1);
        assert!(result2.memories[0].memory.content.contains("early"));
    }

    #[test]
    fn source_filter() {
        let db = MemoryDB::open(":memory:").expect("in-memory db");
        db.insert(MemoryInput {
            content: "manual note about performance".into(),
            source: Some("api".into()),
            ..Default::default()
        }).unwrap();
        db.insert(MemoryInput {
            content: "extracted note about performance".into(),
            source: Some("extract".into()),
            ..Default::default()
        }).unwrap();

        let req = RecallRequest {
            query: "performance".into(),
            source: Some("extract".into()),
            limit: Some(10),
            ..Default::default()
        };
        let result = recall(&db, &req, None, None);
        assert_eq!(result.memories.len(), 1);
        assert!(result.memories[0].memory.content.contains("extracted"));
    }

    #[test]
    fn layer_filter() {
        let db = MemoryDB::open(":memory:").expect("in-memory db");
        db.insert(MemoryInput {
            content: "buffer thought about testing".into(),
            layer: Some(1),
            ..Default::default()
        }).unwrap();
        db.insert(MemoryInput {
            content: "core fact about testing".into(),
            layer: Some(3),
            ..Default::default()
        }).unwrap();

        let req = RecallRequest {
            query: "testing".into(),
            layers: Some(vec![3]),
            limit: Some(10),
            ..Default::default()
        };
        let result = recall(&db, &req, None, None);
        assert_eq!(result.memories.len(), 1);
        assert!(result.memories[0].memory.content.contains("core"));
    }

    #[test]
    fn dual_hit_boost_increases_score() {
        // When a memory is found by both semantic and FTS, its score should
        // be higher than FTS-only. We can't test semantic without embeddings,
        // but we can verify score_combined produces consistent values and
        // that the boost math works.
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let base_score = score_combined(0.8, 0.7, now_ms);
        // fts_rel=0.3 → boost = 1 + 0.3*0.3 = 1.09 → relevance = 0.7 * 1.09 = 0.763
        let boosted_rel: f64 = (0.7 * (1.0 + 0.3 * 0.3_f64)).min(1.5);
        let boosted_score = score_combined(0.8, boosted_rel, now_ms);
        assert!(boosted_score > base_score, "boosted relevance should yield higher score");
    }

    #[test]
    fn min_importance_filter() {
        let db = MemoryDB::open(":memory:").expect("in-memory db");
        db.insert(MemoryInput {
            content: "low importance noise".into(),
            importance: Some(0.1),
            ..Default::default()
        }).unwrap();
        db.insert(MemoryInput {
            content: "high importance signal".into(),
            importance: Some(0.9),
            ..Default::default()
        }).unwrap();

        let req = RecallRequest {
            query: "importance".into(),
            min_importance: Some(0.5),
            limit: Some(10),
            ..Default::default()
        };
        let result = recall(&db, &req, None, None);
        assert_eq!(result.memories.len(), 1);
        assert!(result.memories[0].memory.content.contains("high"));
    }
}
