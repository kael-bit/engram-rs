//! Hybrid recall: semantic + keyword retrieval with budget awareness.

use crate::db::{is_cjk, Layer, Memory, MemoryDB, ScoredMemory};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

const W_IMPORTANCE: f64 = 0.4;
const W_RECENCY: f64 = 0.3;
const W_RELEVANCE: f64 = 0.3;

/// Recall request parameters.
#[derive(Debug, Deserialize)]
pub struct RecallRequest {
    pub query: String,
    pub budget_tokens: Option<usize>,
    pub layers: Option<Vec<u8>>,
    pub min_importance: Option<f64>,
    pub limit: Option<usize>,
}

/// Recall response with scored memories and metadata.
#[derive(Debug, serde::Serialize)]
pub struct RecallResponse {
    pub memories: Vec<ScoredMemory>,
    pub total_tokens: usize,
    pub layer_breakdown: HashMap<u8, usize>,
    pub search_mode: String,
}

/// Estimate token count for mixed CJK/ASCII text.
///
/// CJK characters average ~1.5 tokens each; Latin text averages ~4 bytes per token.
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

/// Compute the composite score for a memory given its relevance.
fn score_memory(mem: &Memory, relevance: f64) -> ScoredMemory {
    let recency = recency_score(mem.last_accessed, mem.decay_rate);
    let bonus = mem.layer.score_bonus();
    let score =
        (W_IMPORTANCE * mem.importance + W_RECENCY * recency + W_RELEVANCE * relevance) * bonus;

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
    let layer_filter: Option<Vec<Layer>> = req.layers.as_ref().map(|ls| {
        ls.iter()
            .filter_map(|&l| l.try_into().ok())
            .collect()
    });

    let mut scored: Vec<ScoredMemory> = Vec::new();
    let mut seen = HashSet::new();
    let mut search_mode = "fts".to_string();

    // Phase 1: Semantic search (if embedding available)
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

    // Phase 2: FTS keyword search (always runs)
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

    // Phase 3: Always include core memories
    for mem in db.list_by_layer(Layer::Core) {
        if seen.contains(&mem.id) {
            continue;
        }
        if mem.importance < min_imp {
            continue;
        }
        if let Some(ref lf) = layer_filter {
            if !lf.contains(&Layer::Core) {
                continue;
            }
        }
        scored.push(score_memory(&mem, 0.0));
    }

    // Sort by score descending
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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

        let _ = db.touch(&sm.memory.id);
        selected.push(sm);
    }

    RecallResponse {
        memories: selected,
        total_tokens,
        layer_breakdown: breakdown,
        search_mode,
    }
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
}
