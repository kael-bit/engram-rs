//! Hybrid recall: semantic + keyword retrieval with budget awareness.

use crate::ai::{self, AiConfig};
use crate::db::{is_cjk, Layer, Memory, MemoryDB, ScoredMemory};
use crate::error::EngramError;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use tracing::debug;

// scoring weights — should add up to 1.0
use crate::thresholds::{RECALL_WEIGHT_RELEVANCE, RECALL_WEIGHT_WEIGHT, RECALL_WEIGHT_RECENCY, RECALL_SIM_FLOOR};

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
    // rerank removed — FTS+semantic scoring is sufficient
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

/// Simplified scoring fallback — used when only importance (not full Memory) is available.
/// For full scoring, prefer `crate::scoring::memory_weight(&mem)`.
pub fn score_combined(importance: f64, relevance: f64, last_accessed: i64) -> f64 {
    let now = crate::db::now_ms();
    let age_hours = ((now - last_accessed) as f64 / 3_600_000.0).max(0.0);
    // simplified recency for rescore — uses default decay
    let recency = (-0.1 * age_hours).exp();
    RECALL_WEIGHT_WEIGHT * importance + RECALL_WEIGHT_RECENCY * recency + RECALL_WEIGHT_RELEVANCE * relevance
}

pub fn score_memory(mem: &Memory, relevance: f64) -> ScoredMemory {
    let recency = recency_score(mem.last_accessed, mem.decay_rate);
    let weight = crate::scoring::memory_weight(mem);
    // Multiplicative: relevance is the gate, weight/recency are modifiers only.
    // relevance=0 → score=0 regardless of weight/recency (no free passes for Core).
    let mut score = relevance * (1.0 + 0.4 * weight + 0.2 * recency);

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
        // Tags: soft boost instead of hard filter.
        // Memories matching hint tags get relevance boost; others are NOT excluded.
        // if let Some(ref required_tags) = req.tags {
        //     if !required_tags.iter().all(|t| mem.tags.contains(t)) {
        //         return false;
        //     }
        // }
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
            // sim_floor is a hard minimum for raw cosine — kept very low so
            // short CJK queries (which produce uniformly low cosine) aren't
            // dropped prematurely. min_score filters final scored results later.
            let sim_floor = RECALL_SIM_FLOOR;
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
                .filter(|w| !w.is_empty())
                .map(str::to_lowercase)
                .collect()
        };

        if !query_terms.is_empty() {
            let no_fts_penalty: f64 = std::env::var("ENGRAM_NO_FTS_PENALTY")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(0.85);
            for sm in scored.iter_mut() {
                // Only penalize semantic-only hits (not boosted by FTS)
                if fts_ids.contains(&sm.memory.id) {
                    continue;
                }
                let content_lower = sm.memory.content.to_lowercase();
                let has_any = query_terms.iter().any(|t| content_lower.contains(t));
                if !has_any {
                    // No query terms found — likely a false-positive embedding match
                    sm.relevance *= no_fts_penalty;
                    let rescored = score_memory(&sm.memory, sm.relevance);
                    sm.score = rescored.score;
                    sm.recency = rescored.recency;
                }
            }
        }
    }

    // IDF-weighted term boost: rare query terms that appear in a memory's
    // content/tags get a relevance boost proportional to their corpus rarity.
    // This helps discriminating queries like "discord someone shared proposal"
    // where "discord" is rare (high IDF) and "proposal" is common (low IDF).
    if !req.query.is_empty() && !scored.is_empty() {
        let (terms, total_docs) = crate::db::fts::extract_query_terms(&req.query, db);
        if !terms.is_empty() {
            // Compute IDF for each term: ln(N / df)
            let term_idfs: Vec<(String, f64)> = terms.iter().filter_map(|t| {
                let df = db.term_doc_frequency(t);
                if df == 0 {
                    // Term not in corpus at all — no discriminative value, skip
                    return None;
                }
                let idf = (total_docs / df as f64).ln();
                Some((t.clone(), idf))
            }).collect();
            let total_idf: f64 = term_idfs.iter().map(|(_, idf)| idf).sum();

            if total_idf > 0.0 {
                let idf_alpha: f64 = std::env::var("ENGRAM_IDF_BOOST_ALPHA")
                    .ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);

                for sm in scored.iter_mut() {
                    let text = format!("{} {}", sm.memory.content, sm.memory.tags.join(" "))
                        .to_lowercase();
                    let hit_idf: f64 = term_idfs.iter()
                        .filter(|(term, _)| text.contains(term.as_str()))
                        .map(|(_, idf)| idf)
                        .sum();
                    let affinity = hit_idf / total_idf;
                    if affinity > 0.0 {
                        sm.relevance *= 1.0 + idf_alpha * affinity;
                        let rescored = score_memory(&sm.memory, sm.relevance);
                        sm.score = rescored.score;
                        sm.recency = rescored.recency;
                    } else {
                        // No rare query terms found — likely a false-positive
                        // embedding match. Penalize to push down noise.
                        let miss_penalty: f64 = std::env::var("ENGRAM_IDF_MISS_PENALTY")
                            .ok().and_then(|v| v.parse().ok()).unwrap_or(0.85);
                        sm.relevance *= miss_penalty;
                        let rescored = score_memory(&sm.memory, sm.relevance);
                        sm.score = rescored.score;
                        sm.recency = rescored.recency;
                    }
                }
            } else if !terms.is_empty() {
                // All query terms have df=0: none exist in the corpus at all.
                // Every result is a false positive from embedding noise.
                // Apply a heavy penalty so scores reflect low confidence.
                let orphan_penalty: f64 = std::env::var("ENGRAM_ORPHAN_QUERY_PENALTY")
                    .ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);
                for sm in scored.iter_mut() {
                    sm.relevance *= orphan_penalty;
                    let rescored = score_memory(&sm.memory, sm.relevance);
                    sm.score = rescored.score;
                    sm.recency = rescored.recency;
                }
            }
        }
    }

    // Explicit tag boost: when agent passes tags=["discord"], memories with
    // matching tags get a relevance boost (not a hard filter).
    if let Some(ref hint_tags) = req.tags {
        if !hint_tags.is_empty() {
            let tag_boost: f64 = std::env::var("ENGRAM_TAG_BOOST")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(0.2);
            for sm in scored.iter_mut() {
                let overlap = sm.memory.tags.iter()
                    .filter(|t| hint_tags.iter().any(|ht| ht.eq_ignore_ascii_case(t)))
                    .count();
                if overlap > 0 {
                    sm.relevance *= 1.0 + tag_boost * overlap as f64;
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

// rerank removed — FTS+semantic scoring is sufficient without LLM re-ranking

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


