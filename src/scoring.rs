use serde::Serialize;

use crate::db::{Layer, Memory};
use crate::thresholds;

/// Clean API response for a single memory — used at the HTTP boundary only.
/// Internal types (`ScoredMemory`, `RecallResponse`) remain unchanged.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryResult {
    /// Short ID (first 8 characters)
    pub id: String,
    pub content: String,
    /// Unified score (recall: relevance-weighted; triggers: memory_weight)
    pub score: f64,
    /// Layer name
    pub layer: String,
    /// Only present if non-empty
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Memory kind (semantic, episodic, procedural)
    pub kind: String,
}

impl MemoryResult {
    pub fn from_memory(mem: &Memory, score: f64) -> Self {
        let layer_name = match mem.layer {
            Layer::Core => "core",
            Layer::Working => "working",
            Layer::Buffer => "buffer",
        };
        Self {
            id: crate::util::short_id(&mem.id).to_string(),
            content: mem.content.clone(),
            score,
            layer: layer_name.to_string(),
            tags: mem.tags.clone(),
            kind: mem.kind.clone(),
        }
    }
}

/// Unified memory weight — used across all ranking contexts.
/// Combines decayable importance with permanent reinforcement signals.
///
/// # Formula (v2 — additive biases)
///
/// ```text
/// base = importance + rep_bonus + access_bonus
/// weight = base + kind_bias + layer_bias
/// ```
///
/// - `importance` is in `[0, 1]` (clamped on write).
/// - `rep_bonus` = `SCALE × ln(1 + rep_count)`, capped at `REP_BONUS_CAP`.
/// - `access_bonus` = `SCALE × ln(1 + access_count)`, capped at `ACCESS_BONUS_CAP`.
/// - `kind_bias` and `layer_bias` are small additive offsets (e.g. +0.15, -0.1).
///
/// The return value is **not** clamped to `[0, 1]` and can exceed `1.0`.
///
/// **Design rationale (v2):**
/// - Additive biases prevent the 2.4× multiplicative gap between procedural×core
///   and episodic×buffer that existed in v1.
/// - Logarithmic rep/access bonuses with higher caps preserve long-tail
///   discrimination (v1 saturated at 5 reps / 19 accesses).
pub fn memory_weight(mem: &Memory) -> f64 {
    let rep_bonus = ((1.0 + mem.repetition_count as f64).ln() * thresholds::REP_BONUS_SCALE)
        .min(thresholds::REP_BONUS_CAP);
    let access_bonus = ((1.0 + mem.access_count as f64).ln() * thresholds::ACCESS_BONUS_SCALE)
        .min(thresholds::ACCESS_BONUS_CAP);

    let kind_bias = match mem.kind.as_str() {
        "procedural" => thresholds::KIND_BIAS_PROCEDURAL,
        "episodic" => thresholds::KIND_BIAS_EPISODIC,
        _ => 0.0, // semantic and unknown
    };
    let layer_bias = match mem.layer {
        Layer::Core => thresholds::LAYER_BIAS_CORE,
        Layer::Working => 0.0,
        Layer::Buffer => thresholds::LAYER_BIAS_BUFFER,
    };

    let base = mem.importance + rep_bonus + access_bonus;
    base + kind_bias + layer_bias
}
