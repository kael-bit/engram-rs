//! Hybrid recall: semantic + keyword retrieval with budget awareness.

use crate::ai::{self, AiConfig};
use crate::db::{is_cjk, Layer, Memory, MemoryDB, ScoredMemory};
use crate::error::EngramError;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use tracing::{debug, warn};

// scoring weights — should add up to 1.0
// relevance is king: a perfectly relevant low-importance memory
// beats a vaguely related high-importance one
const WEIGHT_RELEVANCE: f64 = 0.6;
const WEIGHT_IMPORTANCE: f64 = 0.2;
const WEIGHT_RECENCY: f64 = 0.2;

/// Default minimum cosine similarity to include a result.
const DEFAULT_SIM_FLOOR: f64 = 0.3;

/// Detect short CJK queries where cosine similarity is unreliable.
///
/// `text-embedding-3-small` produces uniformly high cosine scores for short
/// Chinese queries (< 10 chars), making discrimination poor.  For these cases
/// FTS keyword matching is a stronger signal than cosine.
pub fn is_short_cjk_query(query: &str) -> bool {
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
pub fn recency_score(last_accessed: i64, decay_rate: f64) -> f64 {
    let now = crate::db::now_ms();
    let hours = ((now - last_accessed) as f64 / 3_600_000.0).max(0.0);
    let rate = if decay_rate.is_finite() { decay_rate.clamp(0.0, 10.0) } else { 0.1 };
    (-rate * hours / 168.0).exp()
}

pub fn score_combined(importance: f64, relevance: f64, last_accessed: i64) -> f64 {
    let now = crate::db::now_ms();
    let age_hours = ((now - last_accessed) as f64 / 3_600_000.0).max(0.0);
    // simplified recency for rescore — uses default decay
    let recency = (-0.1 * age_hours).exp();
    WEIGHT_IMPORTANCE * importance + WEIGHT_RECENCY * recency + WEIGHT_RELEVANCE * relevance
}

pub fn score_memory(mem: &Memory, relevance: f64) -> ScoredMemory {
    let recency = recency_score(mem.last_accessed, mem.decay_rate);
    let bonus = mem.layer.score_bonus();
    let mut score =
        (WEIGHT_IMPORTANCE * mem.importance + WEIGHT_RECENCY * recency + WEIGHT_RELEVANCE * relevance) * bonus;

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
    query_emb: Option<&[f32]>,
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
            // Allow "default" namespace memories through when filtering by a project namespace
            if mem.namespace != *ns && mem.namespace != "default" {
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
    let fts = db.search_fts_ns(&req.query, fetch_limit * 3, ns).unwrap_or_default();

    let mut extra_fts_results: Vec<Vec<(String, f64)>> = Vec::new();
    if let Some(queries) = extra_queries {
        for eq in queries {
            extra_fts_results.push(db.search_fts_ns(eq, limit, ns).unwrap_or_default());
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
    for m in db.list_filtered(200, 0, ns, Some(3), None, None)
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
            let sim_floor = req.min_score.unwrap_or(DEFAULT_SIM_FLOOR);
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
                // Gate FTS boost by semantic confidence: if semantic score is low,
                // the keyword match is likely coincidental (e.g. "配置" appearing
                // in an unrelated memory). Scale boost by how confident semantic is.
                let sem_gate = (sm.relevance / 0.45).min(1.0);
                let semantic_boosted = sm.relevance * (1.0 + fts_rel * fts_boost * sem_gate);
                // Floor for keyword-confirmed results, also gated by semantic score
                let fts_floor = if short_cjk {
                    0.50 + fts_rel * 0.35
                } else {
                    0.35 + fts_rel * 0.25
                };
                let gated_floor = fts_floor * sem_gate;
                sm.relevance = semantic_boosted.max(gated_floor).min(1.0);
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
            // Without semantic backing, cap low — keyword alone is unreliable.
            let capped = if short_cjk { fts_rel * 0.4 } else { fts_rel * 0.25 };
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
    let budget = if budget == 0 { usize::MAX } else { budget };
    let mut selected = Vec::new();
    let mut total_tokens = 0;
    let mut breakdown = HashMap::new();

    for sm in paginated {
        let tokens = estimate_tokens(&sm.memory.content);
        if total_tokens + tokens > budget && !selected.is_empty() {
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
    db: &crate::SharedDB,
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
        Ok(tcr) => {
            if let Some(ref u) = tcr.usage {
                let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
                let _ = db.log_llm_call("rerank", &tcr.model, u.prompt_tokens, u.completion_tokens, cached, tcr.duration_ms);
            }
            let order: Vec<usize> = tcr.value.ranked_ids.iter()
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
            // Re-assign scores so they reflect the reranked order.
            // Highest-ranked gets the max score from the set, then
            // linearly interpolate down to preserve magnitude info.
            if response.memories.len() > 1 {
                let max_s = response.memories.iter().map(|m| m.score)
                    .fold(f64::NEG_INFINITY, f64::max);
                let min_s = response.memories.iter().map(|m| m.score)
                    .fold(f64::INFINITY, f64::min);
                let n = response.memories.len() as f64;
                for (i, sm) in response.memories.iter_mut().enumerate() {
                    sm.score = max_s - (max_s - min_s) * (i as f64 / (n - 1.0));
                }
            }
            response.search_mode = format!("{} +rerank", response.search_mode);
        }
        Err(e) => {
            warn!(error = %e, "LLM rerank failed, keeping original order");
        }
    }
}

/// Check if content is semantically similar to an existing memory.
/// Returns the ID of the duplicate if found, None otherwise.
#[allow(dead_code)]
pub async fn quick_semantic_dup(
    ai_cfg: &AiConfig,
    db: &MemoryDB,
    content: &str,
) -> Result<Option<String>, EngramError> {
    quick_semantic_dup_threshold(ai_cfg, db, content, crate::thresholds::RECALL_DEDUP_SIM).await
}

/// Like `quick_semantic_dup` but with a custom cosine threshold.
/// Proxy extraction uses a lower threshold (0.72) to catch cross-language dupes.
pub async fn quick_semantic_dup_threshold(
    ai_cfg: &AiConfig,
    db: &MemoryDB,
    content: &str,
    threshold: f64,
) -> Result<Option<String>, EngramError> {
    let (dup, _emb) = quick_semantic_dup_with_embedding(ai_cfg, db, content, threshold).await?;
    Ok(dup)
}

/// Like `quick_semantic_dup_threshold` but also returns the computed embedding
/// so callers can reuse it (e.g. pass to `MemoryInput` for DB-level dedup).
pub async fn quick_semantic_dup_with_embedding(
    ai_cfg: &AiConfig,
    db: &MemoryDB,
    content: &str,
    threshold: f64,
) -> Result<(Option<String>, Vec<f32>), EngramError> {
    let er = ai::get_embeddings(ai_cfg, &[content.to_string()]).await?;
    if let Some(ref u) = er.usage {
        let cached = u.prompt_tokens_details.as_ref().map_or(0, |d| d.cached_tokens);
        let _ = db.log_llm_call("dedup_embed", &ai_cfg.embed_model, u.prompt_tokens, u.completion_tokens, cached, 0);
    }
    let emb = er.embeddings.into_iter().next()
        .ok_or_else(|| EngramError::AiBackend("no embedding returned".into()))?;
    let candidates = db.search_semantic(&emb, 10);
    if let Some((top_id, top_score)) = candidates.first() {
        tracing::debug!(
            top_id = &top_id[..8.min(top_id.len())],
            top_score = %format!("{:.4}", top_score),
            threshold = %format!("{:.2}", threshold),
            total_candidates = candidates.len(),
            "semantic dedup check"
        );
    }
    for (id, score) in &candidates {
        if *score > threshold {
            return Ok((Some(id.clone()), emb));
        }
    }
    Ok((None, emb))
}


