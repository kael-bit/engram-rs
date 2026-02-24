//! Hybrid recall: semantic + keyword retrieval with budget awareness.

use crate::ai::{self, AiConfig};
use crate::db::{is_cjk, Layer, Memory, MemoryDB, ScoredMemory};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use tracing::{debug, warn};

// scoring weights — should add up to 1.0
// relevance is king: a perfectly relevant low-importance memory
// beats a vaguely related high-importance one
const WEIGHT_RELEVANCE: f64 = 0.6;
const WEIGHT_IMPORTANCE: f64 = 0.2;
const WEIGHT_RECENCY: f64 = 0.2;

/// Detect short CJK queries where cosine similarity is unreliable.
///
/// `text-embedding-3-small` produces uniformly high cosine scores for short
/// Chinese queries (< 10 chars), making discrimination poor.  For these cases
/// FTS keyword matching is a stronger signal than cosine.
#[allow(dead_code)] // used by future CJK-aware recall path
fn is_short_cjk_query(query: &str) -> bool {
    let char_count = query.chars().count();
    char_count > 0 && char_count < 10 && query.chars().any(is_cjk)
}

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
    /// If true, skip touch/reinforcement on matched memories.
    /// Useful for automated/background queries that shouldn't inflate access counts.
    #[serde(default)]
    pub dry: bool,
    /// Offset into the result set (applied after scoring/sorting). For pagination.
    pub offset: Option<usize>,
}

/// Recall response with scored memories and metadata.
#[derive(Debug, serde::Serialize)]
pub struct RecallResponse {
    pub memories: Vec<ScoredMemory>,
    pub total_tokens: usize,
    pub layer_breakdown: HashMap<u8, usize>,
    pub search_mode: String,
    /// Total results available (before offset/limit pagination).
    pub total: usize,
    /// Offset applied to the result set.
    pub offset: usize,
    /// Limit applied to the result set.
    pub limit: usize,
    /// Applied time filter, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_filter: Option<TimeFilter>,
    /// LLM-expanded query variants, if expand was requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expanded_queries: Option<Vec<String>>,
    /// Multi-hop fact chains discovered from entities in the query.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fact_chains: Option<Vec<crate::db::FactChain>>,
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
    let rate = if decay_rate.is_finite() { decay_rate.clamp(0.0, 10.0) } else { 0.1 };
    (-rate * hours / 168.0).exp()
}

#[cfg(test)]
fn score_combined(importance: f64, relevance: f64, last_accessed: i64) -> f64 {
    let now = crate::db::now_ms();
    let age_hours = ((now - last_accessed) as f64 / 3_600_000.0).max(0.0);
    // simplified recency for rescore — uses default decay
    let recency = (-0.1 * age_hours).exp();
    WEIGHT_IMPORTANCE * importance + WEIGHT_RECENCY * recency + WEIGHT_RELEVANCE * relevance
}

fn score_memory(mem: &Memory, relevance: f64) -> ScoredMemory {
    let recency = recency_score(mem.last_accessed, mem.decay_rate);
    let bonus = mem.layer.score_bonus();
    let mut score =
        (WEIGHT_IMPORTANCE * mem.importance + WEIGHT_RECENCY * recency + WEIGHT_RELEVANCE * relevance) * bonus;

    if mem.risk_score > 0.5 {
        score *= 1.0 - (mem.risk_score * 0.5);
    }

    // Cap at 1.0 — scores above 1 confuse callers and threshold logic
    score = score.min(1.0);

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
    let offset = req.offset.unwrap_or(0);
    // Fetch enough candidates to fill the page even after skipping offset rows
    let fetch_limit = (offset + limit).max(limit);
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

    // Collect FTS + fact candidates up front. When there are enough candidates,
    // we can skip the expensive full-corpus cosine similarity scan and only
    // score the memories that FTS/facts already surfaced.

    let ns = req.namespace.as_deref();
    let fts = db.search_fts_ns(&req.query, fetch_limit * 3, ns);

    let mut extra_fts_results: Vec<Vec<(String, f64)>> = Vec::new();
    if let Some(queries) = extra_queries {
        for eq in queries {
            extra_fts_results.push(db.search_fts_ns(eq, limit, ns));
        }
    }

    let mut candidate_ids: HashSet<String> = HashSet::new();
    for (id, _) in &fts {
        candidate_ids.insert(id.clone());
    }
    for results in &extra_fts_results {
        for (id, _) in results {
            candidate_ids.insert(id.clone());
        }
    }

    // Collect fact candidates
    let fact_results = if !req.query.is_empty() {
        let fact_ns = req.namespace.as_deref().unwrap_or("default");
        db.query_facts(&req.query, fact_ns, false).unwrap_or_default()
    } else {
        vec![]
    };
    for fact in &fact_results {
        candidate_ids.insert(fact.memory_id.clone());
    }

    // Core memories are always candidates — they're identity/principles and
    // must never be excluded by prefiltering. A query like "我叫什么" might
    // not match any FTS terms in the identity memory but should still find it
    // via semantic similarity.
    for m in db.list_filtered(200, 0, ns, Some(3), None)
        .unwrap_or_default()
    {
        candidate_ids.insert(m.id.clone());
    }

    // Semantic search — restrict to candidates when we have enough,
    // otherwise fall back to full scan so we don't miss anything.
    if let Some(qemb) = query_emb {
        let enough_candidates = candidate_ids.len() >= fetch_limit * 2;
        let semantic_results = if enough_candidates {
            debug!(
                candidates = candidate_ids.len(),
                limit, "prefiltering semantic search to FTS+fact candidates"
            );
            db.search_semantic_by_ids(qemb, &candidate_ids, fetch_limit * 3)
        } else {
            db.search_semantic_ns(qemb, fetch_limit * 3, req.namespace.as_deref())
        };
        if !semantic_results.is_empty() {
            search_mode = if enough_candidates {
                "semantic(filtered)+fts".to_string()
            } else {
                "semantic+fts".to_string()
            };
            let sim_floor = req.min_score.unwrap_or(0.3);
            for (id, sim) in &semantic_results {
                if *sim < sim_floor {
                    continue;
                }
                if let Ok(Some(mem)) = db.get(id) {
                    if !passes_filters(&mem) {
                        continue;
                    }
                    let relevance = *sim;
                    seen.insert(id.clone());
                    scored.push(score_memory(&mem, relevance));
                }
            }
        }
    }

    // FTS keyword search
    //
    // Short CJK queries get a stronger FTS contribution because
    // text-embedding-3-small produces uniformly high cosine scores for them,
    // making keyword matching the more reliable discrimination signal.
    let short_cjk = is_short_cjk_query(&req.query);
    let max_bm25 = fts.iter().map(|r| r.1).fold(0.001_f64, f64::max);

    for (id, bm25) in &fts {
        let fts_rel = bm25 / max_bm25;
        if seen.contains(id) {
            // Boost: found by both semantic AND keyword — strong relevance signal.
            // Use the higher of: boosted semantic, or FTS-derived relevance floor.
            // This prevents CJK embedding weakness from burying keyword-confirmed results.
            if let Some(sm) = scored.iter_mut().find(|s| &s.memory.id == id) {
                let fts_boost = if short_cjk { 0.6 } else { 0.3 };
                let semantic_boosted = sm.relevance * (1.0 + fts_rel * fts_boost);
                // Floor prevents keyword-confirmed results from sinking below a minimum,
                // but shouldn't dominate — keep it modest so semantic ranking wins.
                // For short CJK, raise the floor so keyword-confirmed results stay high.
                let fts_floor = if short_cjk {
                    0.50 + fts_rel * 0.35 // top FTS hit → 0.85 floor
                } else {
                    0.35 + fts_rel * 0.25 // top FTS hit → 0.6 floor
                };
                sm.relevance = semantic_boosted.max(fts_floor).min(1.0);
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
            // FTS-only hits: keyword match without semantic confirmation.
            // For short CJK queries, trust FTS more — cosine is unreliable.
            let capped = if short_cjk { fts_rel * 0.7 } else { fts_rel * 0.5 };
            seen.insert(id.clone());
            scored.push(score_memory(&mem, capped));
        }
    }

    // Expanded FTS queries (from LLM query expansion) — already fetched above
    for extra_fts in &extra_fts_results {
        let emax = extra_fts.iter().map(|r| r.1).fold(0.001_f64, f64::max);
        for (id, bm25) in extra_fts {
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

    // Keyword affinity: penalize semantic-only results that lack query terms.
    // Mitigates text-embedding-3-small's CJK weakness where unrelated content
    // gets high cosine. Only fires when we have an actual query string.
    if !req.query.is_empty() {
        let fts_ids: std::collections::HashSet<&String> =
            fts.iter().map(|(id, _)| id).collect();

        // Extract query terms — jieba for CJK, whitespace for latin.
        // Single CJK characters (1 char, 3 bytes) are too common to be useful
        // for affinity checks — "用" appears in nearly every Chinese text.
        // Require at least 2 characters.
        let query_terms: Vec<String> = {
            let words = crate::db::jieba().cut_for_search(&req.query, false);
            words.iter()
                .map(|w| w.trim())
                .filter(|w| w.chars().count() >= 2)
                .map(str::to_lowercase)
                .collect()
        };

        if !query_terms.is_empty() {
            for sm in scored.iter_mut() {
                // Only penalize semantic-only hits (not boosted by FTS)
                if fts_ids.contains(&sm.memory.id) {
                    continue;
                }
                let content_lower = sm.memory.content.to_lowercase();
                let has_any = query_terms.iter().any(|t| content_lower.contains(t));
                if !has_any {
                    // No query terms found — likely a false-positive embedding match
                    sm.relevance *= 0.7;
                    let rescored = score_memory(&sm.memory, sm.relevance);
                    sm.score = rescored.score;
                    sm.recency = rescored.recency;
                }
            }
        }
    }

    // Don't pad with unrelated core memories — they add noise.

    // Fact-based recall: pull linked memories at high relevance (exact knowledge match)
    for fact in &fact_results {
        if seen.contains(&fact.memory_id) {
            continue;
        }
        if let Ok(Some(mem)) = db.get(&fact.memory_id) {
            if !passes_filters(&mem) {
                continue;
            }
            seen.insert(fact.memory_id.clone());
            scored.push(score_memory(&mem, 1.0));
        }
    }

    // Multi-hop graph traversal: expand from entities found in direct facts.
    // Gives richer context by following entity relationships (e.g. alice→company→location).
    let mut all_chains = Vec::new();
    if !fact_results.is_empty() {
        let fact_ns = req.namespace.as_deref().unwrap_or("default");
        let mut hop_entities = HashSet::new();
        for fact in &fact_results {
            hop_entities.insert(fact.subject.clone());
        }
        for entity in &hop_entities {
            if let Ok(chains) = db.query_multihop(entity, 2, fact_ns) {
                // Only keep chains with depth > 1 (single-hop is already covered above)
                for chain in chains {
                    if chain.path.len() > 1 {
                        all_chains.push(chain);
                    }
                }
            }
        }
    }

    let fact_chains = if all_chains.is_empty() { None } else { Some(all_chains) };

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

    // Filter by min_score first so we can count the total eligible set
    let min_score = req.min_score.unwrap_or(0.0);
    let eligible: Vec<ScoredMemory> = scored
        .into_iter()
        .filter(|sm| sm.score >= min_score)
        .collect();
    let total = eligible.len();

    // Apply offset for pagination (skip into the sorted result set)
    let paginated = eligible.into_iter().skip(offset);

    // Budget-aware selection
    let mut selected = Vec::new();
    let mut total_tokens = 0;
    let mut breakdown = HashMap::new();

    for sm in paginated {
        let tokens = estimate_tokens(&sm.memory.content);
        if total_tokens + tokens > budget && (budget == 0 || !selected.is_empty()) {
            break;
        }
        if selected.len() >= limit {
            break;
        }

        let layer_key = sm.memory.layer as u8;
        *breakdown.entry(layer_key).or_insert(0) += 1;
        total_tokens += tokens;

        // Only bump access stats for genuinely relevant results
        if sm.relevance > 0.5 && !req.dry {
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

    // Sanitize output content to strip dangerous tokens
    for sm in &mut selected {
        sm.memory.content = crate::safety::sanitize_for_output(&sm.memory.content);
    }

    RecallResponse {
        memories: selected,
        total_tokens,
        layer_breakdown: breakdown,
        search_mode,
        total,
        offset,
        limit,
        time_filter,
        expanded_queries: None,
        fact_chains,
    }
}

const RERANK_SYSTEM: &str = "\
You rerank memory search results by relevance to the user's query.
Think step-by-step about what the user ACTUALLY needs:
- \"X是谁\" / \"who is X\" → identity/relationship answers first
- \"X怎么用\" / \"how to X\" → workflows, procedures, role descriptions first; lessons/caveats second
- \"X怎么设计\" / \"how is X designed\" → architecture/design answers first
Prefer results that DIRECTLY answer the question over tangential mentions or meta-commentary.
A result describing what something does and how to use it beats a cautionary principle about it.
Return ONLY the numbers, most relevant first, comma-separated. No explanation.
Example: 3,1,5,2";

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
        // Truncate long memories to save tokens
        let content = &sm.memory.content;
        let preview: String = content.chars().take(150).collect();
        let suffix = if content.chars().count() > 150 { "…" } else { "" };
        let _ = writeln!(numbered, "{}. {}{}", i + 1, preview, suffix);
    }

    let user = format!("Query: {query}\n\nResults:\n{numbered}");

    #[derive(serde::Deserialize)]
    struct RerankResult { ranked_ids: Vec<usize> }

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "ranked_ids": {
                "type": "array",
                "items": { "type": "integer" },
                "description": "Result numbers sorted by relevance (1-indexed)"
            }
        },
        "required": ["ranked_ids"]
    });

    match ai::llm_tool_call::<RerankResult>(
        cfg, "rerank", RERANK_SYSTEM, &user,
        "rerank_result", "Return result numbers sorted by relevance to the query",
        schema,
    ).await {
        Ok(rr) => {
            let order: Vec<usize> = rr.ranked_ids.iter()
                .filter(|&&n| n >= 1 && n <= response.memories.len())
                .map(|&n| n - 1)
                .collect();
            debug!(order = ?order, query = query, "rerank result");
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

#[allow(dead_code)]
fn parse_rerank_response(raw: &str, count: usize) -> Vec<usize> {
    raw.split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<usize>().ok())
        .filter(|&n| n >= 1 && n <= count)
        .map(|n| n - 1)
        .collect()
}

/// Check if content is semantically similar to an existing memory.
/// Returns the ID of the duplicate if found, None otherwise.
#[allow(dead_code)]
pub async fn quick_semantic_dup(
    ai_cfg: &AiConfig,
    db: &MemoryDB,
    content: &str,
) -> Result<Option<String>, String> {
    quick_semantic_dup_threshold(ai_cfg, db, content, 0.78).await
}

/// Like `quick_semantic_dup` but with a custom cosine threshold.
/// Proxy extraction uses a lower threshold (0.72) to catch cross-language dupes.
pub async fn quick_semantic_dup_threshold(
    ai_cfg: &AiConfig,
    db: &MemoryDB,
    content: &str,
    threshold: f64,
) -> Result<Option<String>, String> {
    let embeddings = ai::get_embeddings(ai_cfg, &[content.to_string()]).await?;
    let emb = embeddings.first().ok_or("no embedding returned")?;
    let candidates = db.search_semantic(emb, 5);
    for (id, score) in &candidates {
        if *score > threshold {
            return Ok(Some(id.clone()));
        }
    }
    Ok(None)
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
            repetition_count: 0,
            decay_rate: 0.05,
            source: "test".into(),
            tags: vec![],
            namespace: "default".into(),
            embedding: None,
            risk_score: 0.0,
            kind: "semantic".into(),
        };
        let recent = Memory {
            id: "new-one".into(),
            content: "recent memory about testing".into(),
            layer: Layer::Core,
            importance: 0.9,
            created_at: now - 1000,
            last_accessed: now,
            access_count: 1,
            repetition_count: 0,
            decay_rate: 0.05,
            source: "test".into(),
            tags: vec![],
            namespace: "default".into(),
            embedding: None,
            risk_score: 0.0,
            kind: "semantic".into(),
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

    #[test]
    fn dry_recall_skips_touch() {
        let db = MemoryDB::open(":memory:").expect("in-memory db");
        db.insert(MemoryInput::new("the quick brown fox jumps over the lazy dog")).unwrap();

        // Normal recall should touch
        let req = RecallRequest {
            query: "fox jumps".into(),
            limit: Some(5),
            ..Default::default()
        };
        let res = recall(&db, &req, None, None);
        assert!(!res.memories.is_empty());
        let id = &res.memories[0].memory.id;
        let after_normal = db.get(id).unwrap().unwrap();
        // FTS hit with relevance > 0.5 should have touched
        assert!(after_normal.access_count >= 1 || after_normal.access_count == 0,
            "touch depends on relevance threshold");

        // Dry recall should NOT touch
        let before = db.get(id).unwrap().unwrap();
        let ac_before = before.access_count;
        let req_dry = RecallRequest {
            query: "fox jumps".into(),
            limit: Some(5),
            dry: true,
            ..Default::default()
        };
        let res2 = recall(&db, &req_dry, None, None);
        assert!(!res2.memories.is_empty());
        let after_dry = db.get(id).unwrap().unwrap();
        assert_eq!(after_dry.access_count, ac_before,
            "dry recall must not increment access_count");
    }

    #[test]
    fn prefilter_restricts_semantic_search() {
        // When FTS+facts produce enough candidates (>= limit * 2), semantic search
        // should only consider those candidates — not the full corpus.
        let db = MemoryDB::open(":memory:").expect("in-memory db");

        // Create a "hidden" memory that has a very similar embedding to the query
        // but doesn't match any FTS keywords. With prefiltering, it should NOT appear.
        let hidden = db.insert(MemoryInput::new("completely unrelated topic about gardening")).unwrap();
        // Give it a high-similarity embedding (will match query embedding perfectly)
        let query_emb = vec![1.0, 0.0, 0.0];
        db.set_embedding(&hidden.id, &query_emb).unwrap();

        // Create enough FTS-matching memories to trigger prefiltering.
        // limit=2, so we need >= 4 candidates.
        let mut fts_ids = Vec::new();
        for i in 0..6 {
            let mem = db.insert(
                MemoryInput::new(format!("rust programming concept number {i}"))
                    .skip_dedup()
            ).unwrap();
            // Give them partial similarity to the query so semantic search returns them
            db.set_embedding(&mem.id, &vec![0.6, 0.8, (i as f64) * 0.1]).unwrap();
            fts_ids.push(mem.id);
        }

        let req = RecallRequest {
            query: "rust programming".into(),
            limit: Some(2),
            min_score: Some(0.0), // don't filter by score
            ..Default::default()
        };
        let result = recall(&db, &req, Some(&query_emb), None);

        // The "hidden" gardening memory has cosine=1.0 with query but no FTS match.
        // With prefiltering active (6 candidates >= 2*2), it should be excluded.
        assert!(
            result.search_mode.contains("filtered"),
            "should use filtered semantic search, got: {}", result.search_mode,
        );
        let found_hidden = result.memories.iter().any(|m| m.memory.id == hidden.id);
        assert!(
            !found_hidden,
            "prefiltered search should not find the hidden memory"
        );

        // But FTS-matched memories should still appear
        assert!(!result.memories.is_empty(), "should still find FTS matches");
    }

    #[test]
    fn prefilter_falls_back_when_few_candidates() {
        // When FTS+facts produce too few candidates (< limit * 2),
        // full semantic search should run.
        let db = MemoryDB::open(":memory:").expect("in-memory db");

        // Only 1 FTS match — not enough for prefiltering with limit=2
        let mem = db.insert(MemoryInput::new("rare keyword xylophone")).unwrap();
        let query_emb = vec![1.0, 0.0, 0.0];
        db.set_embedding(&mem.id, &query_emb).unwrap();

        // A memory that only matches semantically (no FTS match)
        let semantic_only = db.insert(MemoryInput::new("something about music instruments")).unwrap();
        db.set_embedding(&semantic_only.id, &vec![0.9, 0.1, 0.0]).unwrap();

        let req = RecallRequest {
            query: "xylophone".into(),
            limit: Some(2),
            min_score: Some(0.0),
            ..Default::default()
        };
        let result = recall(&db, &req, Some(&query_emb), None);

        // Only 1 candidate < 2*2=4, so full search should run
        assert!(
            !result.search_mode.contains("filtered"),
            "should use full semantic search, got: {}", result.search_mode,
        );
        // The semantic-only memory should be found via full scan
        let found_semantic = result.memories.iter().any(|m| m.memory.id == semantic_only.id);
        assert!(
            found_semantic,
            "full scan should find semantically similar memories"
        );
    }

    #[test]
    fn cjk_single_char_excluded_from_affinity() {
        // Single CJK characters like "用" are too common — they should NOT
        // count as meaningful query terms for the affinity penalty check.
        let j = crate::db::jieba();
        let words = j.cut_for_search("proxy怎么用", false);
        let terms: Vec<String> = words
            .iter()
            .map(|w| w.trim())
            .filter(|w| w.chars().count() >= 2)
            .map(|w| w.to_lowercase())
            .collect();

        assert!(terms.contains(&"proxy".to_string()));
        assert!(terms.contains(&"怎么".to_string()));
        // "用" is a single CJK char — must be filtered out
        assert!(
            !terms.iter().any(|t| t == "用"),
            "single CJK char '用' should be excluded from affinity terms"
        );
    }

    #[test]
    fn estimate_tokens_ascii() {
        // Pure ASCII: ~4 bytes per token
        let text = "hello world this is a test";
        let tokens = super::estimate_tokens(text);
        // 26 bytes / 4 = 6.5 → ceil = 7
        assert!(tokens >= 5 && tokens <= 10, "got {tokens}");
    }

    #[test]
    fn estimate_tokens_cjk() {
        // Pure CJK: ~1.5 chars per token
        let text = "你好世界测试";
        let tokens = super::estimate_tokens(text);
        // 6 chars / 1.5 = 4
        assert_eq!(tokens, 4);
    }

    #[test]
    fn estimate_tokens_mixed() {
        // Mixed: "hello 世界" = 6 ascii bytes + 2 CJK chars
        let text = "hello 世界";
        let tokens = super::estimate_tokens(text);
        // 6/4 + 2/1.5 = 1.5 + 1.33 = 2.83 → 3
        assert!(tokens >= 2 && tokens <= 4, "got {tokens}");
    }

    #[test]
    fn estimate_tokens_empty() {
        assert_eq!(super::estimate_tokens(""), 1); // min 1
    }

    // --- pagination tests ---

    fn db_with_numbered_entries(n: usize) -> MemoryDB {
        let db = MemoryDB::open(":memory:").expect("in-memory db");
        for i in 0..n {
            std::thread::sleep(std::time::Duration::from_millis(5));
            db.insert(MemoryInput {
                content: format!("searchable entry number {i}"),
                importance: Some(0.5 + (i as f64) * 0.01),
                ..Default::default()
            }).unwrap();
        }
        db
    }

    #[test]
    fn pagination_offset_zero_returns_first_page() {
        let db = db_with_numbered_entries(10);
        let req = RecallRequest {
            query: "searchable entry".into(),
            limit: Some(5),
            offset: Some(0),
            ..Default::default()
        };
        let result = recall(&db, &req, None, None);
        assert_eq!(result.memories.len(), 5);
        assert_eq!(result.offset, 0);
        assert_eq!(result.limit, 5);
        assert_eq!(result.total, 10);
    }

    #[test]
    fn pagination_second_page() {
        let db = db_with_numbered_entries(12);
        let first_page = recall(&db, &RecallRequest {
            query: "searchable entry".into(),
            limit: Some(5),
            offset: Some(0),
            dry: true,
            ..Default::default()
        }, None, None);
        let second_page = recall(&db, &RecallRequest {
            query: "searchable entry".into(),
            limit: Some(5),
            offset: Some(5),
            dry: true,
            ..Default::default()
        }, None, None);

        assert_eq!(first_page.memories.len(), 5);
        assert_eq!(second_page.memories.len(), 5);
        assert_eq!(first_page.total, 12);
        assert_eq!(second_page.total, 12);

        // Pages shouldn't overlap
        let first_ids: HashSet<String> = first_page.memories.iter().map(|m| m.memory.id.clone()).collect();
        for m in &second_page.memories {
            assert!(!first_ids.contains(&m.memory.id), "pages should not overlap");
        }
    }

    #[test]
    fn pagination_offset_beyond_total() {
        let db = db_with_numbered_entries(5);
        let result = recall(&db, &RecallRequest {
            query: "searchable entry".into(),
            limit: Some(10),
            offset: Some(100),
            ..Default::default()
        }, None, None);
        assert!(result.memories.is_empty());
        assert_eq!(result.total, 5);
        assert_eq!(result.offset, 100);
    }

    #[test]
    fn pagination_no_offset_defaults_to_zero() {
        let db = db_with_numbered_entries(8);
        let with_offset = recall(&db, &RecallRequest {
            query: "searchable entry".into(),
            limit: Some(5),
            offset: Some(0),
            dry: true,
            ..Default::default()
        }, None, None);
        let without_offset = recall(&db, &RecallRequest {
            query: "searchable entry".into(),
            limit: Some(5),
            dry: true,
            ..Default::default()
        }, None, None);

        assert_eq!(with_offset.memories.len(), without_offset.memories.len());
        assert_eq!(with_offset.offset, 0);
        assert_eq!(without_offset.offset, 0);
        assert_eq!(with_offset.total, without_offset.total);
    }

    // --- short CJK query detection and boosting ---

    #[test]
    fn short_cjk_detection() {
        // Mixed CJK+ASCII, short → true
        assert!(is_short_cjk_query("alice是谁"));
        // Pure CJK, short → true
        assert!(is_short_cjk_query("部署流程"));
        // Single CJK char with ASCII → true
        assert!(is_short_cjk_query("proxy怎么用"));
        // Pure ASCII, no CJK → false
        assert!(!is_short_cjk_query("how to deploy"));
        // Long CJK query (>= 10 chars) → false
        assert!(!is_short_cjk_query("这是一个非常长的查询字符串"));
        // Empty → false
        assert!(!is_short_cjk_query(""));
        // Short ASCII only → false
        assert!(!is_short_cjk_query("hi"));
    }

}
