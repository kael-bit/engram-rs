//! Hybrid recall: semantic + keyword retrieval with budget awareness.

use crate::ai::{self, AiConfig};
use crate::db::{is_cjk, Layer, Memory, MemoryDB, ScoredMemory};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use tracing::warn;

// scoring weights — should add up to 1.0
const WEIGHT_IMPORTANCE: f64 = 0.4;
const WEIGHT_RECENCY: f64 = 0.3;
const WEIGHT_RELEVANCE: f64 = 0.3;

/// Recall request parameters.
#[derive(Debug, Deserialize)]
pub struct RecallRequest {
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
    let hours = (now - last_accessed) as f64 / 3_600_000.0;
    (-decay_rate * hours / 168.0).exp()
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

    let mut scored: Vec<ScoredMemory> = Vec::new();
    let mut seen = HashSet::new();
    let mut search_mode = "fts".to_string();

    // Semantic search (if embedding available)
    if let Some(qemb) = query_emb {
        let semantic_results = db.search_semantic(qemb, limit * 3);
        if !semantic_results.is_empty() {
            search_mode = "semantic+fts".to_string();
            let max_sim = semantic_results
                .iter()
                .map(|r| r.1)
                .fold(0.001_f64, f64::max);
            for (id, sim) in &semantic_results {
                if let Ok(Some(mem)) = db.get(id) {
                    if mem.importance < min_imp {
                        continue;
                    }
                    if !time_ok(&mem) {
                        continue;
                    }
                    if let Some(ref lf) = layer_filter {
                        if !lf.contains(&mem.layer) {
                            continue;
                        }
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

    for (id, bm25) in &fts {
        if seen.contains(id) {
            continue;
        }
        if let Ok(Some(mem)) = db.get(id) {
            if mem.importance < min_imp {
                continue;
            }
            if !time_ok(&mem) {
                continue;
            }
            if let Some(ref lf) = layer_filter {
                if !lf.contains(&mem.layer) {
                    continue;
                }
            }
            let relevance = bm25 / max_bm25;
            seen.insert(id.clone());
            scored.push(score_memory(&mem, relevance));
        }
    }

    // Pad with unseen core memories when we don't have enough hits.
    // Low relevance=0.1 means they won't outrank real matches.
    if scored.len() < limit {
        for mem in db.list_by_layer(Layer::Core) {
            if seen.contains(&mem.id) {
                continue;
            }
            if mem.importance < min_imp {
                continue;
            }
            if !time_ok(&mem) {
                continue;
            }
            if let Some(ref lf) = layer_filter {
                if !lf.contains(&Layer::Core) {
                    continue;
                }
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

    for sm in scored {
        let tokens = estimate_tokens(&sm.memory.content);
        if total_tokens + tokens > budget {
            // always return at least one result, unless budget is explicitly 0
            if !selected.is_empty() || budget == 0 {
                break;
            }
        }
        if selected.len() >= limit {
            break;
        }

        let layer_key = sm.memory.layer as u8;
        *breakdown.entry(layer_key).or_insert(0) += 1;
        total_tokens += tokens;

        // Only bump access stats for genuinely relevant results
        if sm.relevance > 0.2 {
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

    match ai::llm_chat(cfg, RERANK_SYSTEM, &user).await {
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_ascii() {
        // "hello world" = 11 bytes / 4 ≈ 3 tokens
        let tokens = estimate_tokens("hello world");
        assert!(tokens >= 2 && tokens <= 4);
    }

    #[test]
    fn tokens_cjk() {
        // 4 CJK chars / 1.5 ≈ 3 tokens
        let tokens = estimate_tokens("你好世界");
        assert!(tokens >= 2 && tokens <= 4);
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
}
