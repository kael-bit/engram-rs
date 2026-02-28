use engram::db::{Layer, Memory};
use engram::scoring::{memory_weight, MemoryResult};

fn make_memory(kind: &str, layer: Layer, importance: f64, reps: i64, access: i64) -> Memory {
    Memory {
        id: "test-1234-5678".to_string(),
        content: "test content".to_string(),
        layer,
        importance,
        created_at: 1000,
        last_accessed: 1000,
        access_count: access,
        repetition_count: reps,
        decay_rate: 1.0,
        source: "api".to_string(),
        tags: vec![],
        modified_at: 1000,
        modified_epoch: 0,
        namespace: "default".to_string(),
        kind: kind.to_string(),
        embedding: None,
    }
}

#[test]
fn weight_baseline_semantic_working() {
    let m = make_memory("semantic", Layer::Working, 0.5, 0, 0);
    let w = memory_weight(&m);
    // semantic Working: kind_boost=1.0, layer_boost=1.0
    // weight = (0.5 + 0 + 0) * 1.0 * 1.0 = 0.5
    assert!((w - 0.5).abs() < 1e-6, "got {w}");
}

#[test]
fn weight_procedural_boost() {
    let m = make_memory("procedural", Layer::Working, 0.5, 0, 0);
    let w = memory_weight(&m);
    // procedural boost = 1.3
    assert!((w - 0.65).abs() < 1e-6, "got {w}");
}

#[test]
fn weight_episodic_penalty() {
    let m = make_memory("episodic", Layer::Working, 0.5, 0, 0);
    let w = memory_weight(&m);
    // episodic boost = 0.8
    assert!((w - 0.4).abs() < 1e-6, "got {w}");
}

#[test]
fn weight_core_boost() {
    let m = make_memory("semantic", Layer::Core, 0.5, 0, 0);
    let w = memory_weight(&m);
    // core boost = 1.2
    assert!((w - 0.6).abs() < 1e-6, "got {w}");
}

#[test]
fn weight_buffer_penalty() {
    let m = make_memory("semantic", Layer::Buffer, 0.5, 0, 0);
    let w = memory_weight(&m);
    // buffer boost = 0.8
    assert!((w - 0.4).abs() < 1e-6, "got {w}");
}

#[test]
fn weight_repetition_bonus() {
    let m = make_memory("semantic", Layer::Working, 0.5, 3, 0);
    let w = memory_weight(&m);
    // rep_bonus = min(3 * 0.1, 0.5) = 0.3
    // weight = (0.5 + 0.3) * 1.0 * 1.0 = 0.8
    assert!((w - 0.8).abs() < 1e-6, "got {w}");
}

#[test]
fn weight_repetition_capped() {
    let m = make_memory("semantic", Layer::Working, 0.5, 100, 0);
    let w = memory_weight(&m);
    // rep_bonus = min(100 * 0.1, 0.5) = 0.5 (capped)
    // weight = (0.5 + 0.5) * 1.0 * 1.0 = 1.0
    assert!((w - 1.0).abs() < 1e-6, "got {w}");
}

#[test]
fn weight_access_bonus() {
    let m = make_memory("semantic", Layer::Working, 0.5, 0, 10);
    let w = memory_weight(&m);
    // access_bonus = min(ln(11) * 0.1, 0.3) = min(0.2398, 0.3) ≈ 0.2398
    let expected = 0.5 + (11.0f64.ln() * 0.1).min(0.3);
    assert!((w - expected).abs() < 1e-4, "got {w}, expected {expected}");
}

#[test]
fn weight_combined_boosts() {
    // procedural + Core + reps
    let m = make_memory("procedural", Layer::Core, 0.8, 2, 5);
    let w = memory_weight(&m);
    let rep_bonus = (2.0 * 0.1f64).min(0.5);
    let access_bonus = (6.0f64.ln() * 0.1).min(0.3);
    let expected = (0.8 + rep_bonus + access_bonus) * 1.3 * 1.2;
    assert!((w - expected).abs() < 1e-4, "got {w}, expected {expected}");
}

#[test]
fn weight_zero_importance() {
    let m = make_memory("semantic", Layer::Working, 0.0, 0, 0);
    let w = memory_weight(&m);
    assert!((w - 0.0).abs() < 1e-6, "got {w}");
}

// ── MemoryResult ───────────────────────────────────────────────────────────

#[test]
fn memory_result_from_memory() {
    let m = make_memory("procedural", Layer::Core, 0.9, 0, 0);
    let r = MemoryResult::from_memory(&m, 0.75);
    assert_eq!(r.id, "test-123"); // short_id truncates to 8 chars
    assert_eq!(r.layer, "core");
    assert_eq!(r.kind, Some("procedural".to_string()));
    assert!((r.score - 0.75).abs() < 1e-6);
}

#[test]
fn memory_result_semantic_kind_omitted() {
    let m = make_memory("semantic", Layer::Working, 0.5, 0, 0);
    let r = MemoryResult::from_memory(&m, 0.5);
    assert_eq!(r.kind, None); // semantic is default, should be omitted
}

#[test]
fn memory_result_buffer_layer() {
    let m = make_memory("episodic", Layer::Buffer, 0.3, 0, 0);
    let r = MemoryResult::from_memory(&m, 0.3);
    assert_eq!(r.layer, "buffer");
    assert_eq!(r.kind, Some("episodic".to_string()));
}
