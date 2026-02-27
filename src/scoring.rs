use serde::Serialize;

use crate::db::{Layer, Memory};

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
    /// Only present if not "semantic" (the default)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
}

impl MemoryResult {
    pub fn from_memory(mem: &Memory, score: f64) -> Self {
        let layer_name = match mem.layer {
            Layer::Core => "core",
            Layer::Working => "working",
            Layer::Buffer => "buffer",
        };
        Self {
            id: mem.id[..mem.id.len().min(8)].to_string(),
            content: mem.content.clone(),
            score,
            layer: layer_name.to_string(),
            tags: mem.tags.clone(),
            kind: if mem.kind != "semantic" { Some(mem.kind.clone()) } else { None },
        }
    }
}

/// Unified memory weight — used across all ranking contexts.
/// Combines decayable importance with permanent reinforcement signals.
pub fn memory_weight(mem: &Memory) -> f64 {
    let rep_bonus = (mem.repetition_count as f64 * 0.1).min(0.5);
    let access_bonus = ((1.0 + mem.access_count as f64).ln() * 0.1).min(0.3);

    let kind_boost = if mem.kind == "procedural" { 1.3 } else { 1.0 };
    let layer_boost = match mem.layer {
        Layer::Core => 1.2,
        Layer::Working => 1.0,
        Layer::Buffer => 0.8,
    };

    (mem.importance + rep_bonus + access_bonus) * kind_boost * layer_boost
}
