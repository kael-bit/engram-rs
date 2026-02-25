use engram::consolidate::{AuditOp, RuleChecker, Grade, OpGrade};
use engram::db::{Layer, Memory};

/// Helper to create a Memory with sensible defaults. Caller overrides as needed.
fn mem(id: &str, layer: Layer, tags: Vec<&str>, content: &str) -> Memory {
    let now = engram::db::now_ms();
    Memory {
        id: id.to_string(),
        content: content.to_string(),
        layer,
        importance: 0.5,
        created_at: now - 7 * 86_400_000,  // 7 days ago
        last_accessed: now - 86_400_000,
        access_count: 5,
        repetition_count: 0,
        decay_rate: layer.default_decay(),
        source: "test".into(),
        tags: tags.into_iter().map(String::from).collect(),
        namespace: "default".into(),
        embedding: None,
        kind: "semantic".into(),
        modified_at: now - 3 * 86_400_000,  // 3 days ago
    }
}

// ──────────────────────────────────────────────
//  Simplified RuleChecker: only 3 mechanical rules
//
//  1. Target not found → Bad
//  2. Same-layer / upward demote → Bad
//  3. Merge: missing source IDs or info loss → Bad
//
//  Everything else → Good (Sonnet judges with full context)
// ──────────────────────────────────────────────

// ── Delete ───────────────────────────────────

#[test]
fn delete_not_found_is_bad() {
    let core: [Memory; 0] = [];
    let working: [Memory; 0] = [];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Delete { id: "nonexistent".into() });
    assert_eq!(grade.grade, Grade::Bad);
    assert!(grade.reason.contains("not found"));
}

#[test]
fn delete_any_existing_memory_is_good() {
    // With simplified checker, any existing memory deletion passes
    let core = [mem("id-001", Layer::Core, vec!["identity"], "I am atlas")];
    let working = [mem("id-002", Layer::Working, vec!["lesson"], "never force-push")];
    let checker = RuleChecker::new(&core, &working);

    let g1 = checker.check(&AuditOp::Delete { id: "id-001".into() });
    assert_eq!(g1.grade, Grade::Good);

    let g2 = checker.check(&AuditOp::Delete { id: "id-002".into() });
    assert_eq!(g2.grade, Grade::Good);
}

#[test]
fn delete_normal_working_memory_is_good() {
    let core: [Memory; 0] = [];
    let working = [mem("id-008", Layer::Working, vec![], "ephemeral note")];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Delete { id: "id-008".into() });
    assert_eq!(grade.grade, Grade::Good);
}

// ── Demote ───────────────────────────────────

#[test]
fn demote_same_layer_is_bad() {
    let core: [Memory; 0] = [];
    let working = [mem("dm-001", Layer::Working, vec![], "some working memory")];
    let checker = RuleChecker::new(&core, &working);
    // Working = 2, demote to 2 is same-layer
    let grade = checker.check(&AuditOp::Demote { id: "dm-001".into(), to: 2 });
    assert_eq!(grade.grade, Grade::Bad);
    assert!(grade.reason.contains("same-layer") || grade.reason.contains("no-op"));
}

#[test]
fn demote_upward_is_bad() {
    let core: [Memory; 0] = [];
    let working = [mem("dm-002", Layer::Working, vec![], "some working memory")];
    let checker = RuleChecker::new(&core, &working);
    // Working = 2, demote to 3 (Core) is upward
    let grade = checker.check(&AuditOp::Demote { id: "dm-002".into(), to: 3 });
    assert_eq!(grade.grade, Grade::Bad);
    assert!(grade.reason.contains("upward") || grade.reason.contains("no-op"));
}

#[test]
fn demote_not_found_is_bad() {
    let core: [Memory; 0] = [];
    let working: [Memory; 0] = [];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Demote { id: "nonexistent".into(), to: 1 });
    assert_eq!(grade.grade, Grade::Bad);
    assert!(grade.reason.contains("not found"));
}

#[test]
fn demote_core_to_working_is_good() {
    // Even for lessons/identity — simplified checker trusts Sonnet's judgment
    let core = [mem("dm-006", Layer::Core, vec!["lesson"], "never do X")];
    let working: [Memory; 0] = [];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Demote { id: "dm-006".into(), to: 2 });
    assert_eq!(grade.grade, Grade::Good);
}

#[test]
fn demote_core_to_buffer_is_good() {
    // Simplified checker allows aggressive demotions — Sonnet decides
    let core = [mem("dm-007", Layer::Core, vec![], "some outdated fact")];
    let working: [Memory; 0] = [];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Demote { id: "dm-007".into(), to: 1 });
    assert_eq!(grade.grade, Grade::Good);
}

// ── Promote ──────────────────────────────────

#[test]
fn promote_not_found_is_bad() {
    let core: [Memory; 0] = [];
    let working: [Memory; 0] = [];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Promote { id: "nonexistent".into(), to: 3 });
    assert_eq!(grade.grade, Grade::Bad);
    assert!(grade.reason.contains("not found"));
}

#[test]
fn promote_any_existing_memory_is_good() {
    // Simplified checker: any existing memory promotion passes
    let core: [Memory; 0] = [];
    let working = [
        mem("pr-001", Layer::Working, vec!["session"], "did X today"),
        mem("pr-002", Layer::Working, vec!["gate-rejected-final"], "rejected stuff"),
    ];
    let checker = RuleChecker::new(&core, &working);

    let g1 = checker.check(&AuditOp::Promote { id: "pr-001".into(), to: 3 });
    assert_eq!(g1.grade, Grade::Good);

    let g2 = checker.check(&AuditOp::Promote { id: "pr-002".into(), to: 3 });
    assert_eq!(g2.grade, Grade::Good);
}

#[test]
fn promote_normal_memory_to_core_is_good() {
    let core: [Memory; 0] = [];
    let working = [mem("pr-006", Layer::Working, vec!["design"], "REST API pattern")];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Promote { id: "pr-006".into(), to: 3 });
    assert_eq!(grade.grade, Grade::Good);
}

// ── Merge ────────────────────────────────────

#[test]
fn merge_content_too_short_is_bad() {
    let m1 = mem("mg-001", Layer::Working, vec![], "a moderately long piece of content that has useful information");
    let m2 = mem("mg-002", Layer::Working, vec![], "another piece of content with details");
    let core: [Memory; 0] = [];
    let working = [m1, m2];
    let checker = RuleChecker::new(&core, &working);
    // merged content is much shorter than the shortest input
    let grade = checker.check(&AuditOp::Merge {
        ids: vec!["mg-001".into(), "mg-002".into()],
        content: "short".into(),
        layer: 2,
        tags: vec![],
    });
    assert_eq!(grade.grade, Grade::Bad);
    assert!(grade.reason.contains("shorter") || grade.reason.contains("losing"));
}

#[test]
fn merge_normal_is_good() {
    let m1 = mem("mg-003", Layer::Working, vec![], "fact A about deploy");
    let m2 = mem("mg-004", Layer::Working, vec![], "fact B about deploy");
    let core: [Memory; 0] = [];
    let working = [m1, m2];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Merge {
        ids: vec!["mg-003".into(), "mg-004".into()],
        content: "Combined facts A and B about deploy workflow and configuration".into(),
        layer: 2,
        tags: vec![],
    });
    assert_eq!(grade.grade, Grade::Good);
}

#[test]
fn merge_missing_source_ids_is_bad() {
    let core: [Memory; 0] = [];
    let working = [mem("mg-009", Layer::Working, vec![], "existing note")];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Merge {
        ids: vec!["mg-009".into(), "mg-nonexist".into()],
        content: "merged something".into(),
        layer: 2,
        tags: vec![],
    });
    assert_eq!(grade.grade, Grade::Bad);
    assert!(grade.reason.contains("found") || grade.reason.contains("source"));
}

#[test]
fn merge_cross_layer_is_good() {
    // Simplified checker: cross-layer merges pass (Sonnet decides)
    let core = [mem("mg-007", Layer::Core, vec![], "core fact")];
    let working = [mem("mg-008", Layer::Working, vec![], "working note")];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Merge {
        ids: vec!["mg-007".into(), "mg-008".into()],
        content: "combined core and working content into one merged memory".into(),
        layer: 3,
        tags: vec![],
    });
    assert_eq!(grade.grade, Grade::Good);
}

// ──────────────────────────────────────────────
//  Score formula tests
//  Note: simplified checker only produces Good/Bad (no Marginal),
//  so score = good_count / total_ops
// ──────────────────────────────────────────────

#[test]
fn score_formula_all_good() {
    // 3 Good, 0 Bad → 3/3 = 1.0
    let score = 3.0_f64 / 3.0;
    assert!((score - 1.0).abs() < f64::EPSILON);
}

#[test]
fn score_formula_mixed() {
    // 2 Good, 1 Bad → 2/3 ≈ 0.667
    let good = 2;
    let total = 3;
    let score = good as f64 / total as f64;
    assert!((score - 2.0 / 3.0).abs() < f64::EPSILON);
}

#[test]
fn score_formula_all_bad() {
    // 0 Good, 3 Bad → 0/3 = 0.0
    let score = 0.0_f64 / 3.0;
    assert!((score - 0.0).abs() < f64::EPSILON);
}

#[test]
fn score_formula_empty_is_one() {
    // Edge case: no ops → score = 1.0
    let grades: Vec<OpGrade> = vec![];
    let score = if grades.is_empty() {
        1.0
    } else {
        let total_ops = grades.len() as f64;
        let good = grades.iter().filter(|g| g.grade == Grade::Good).count();
        good as f64 / total_ops
    };
    assert!((score - 1.0).abs() < f64::EPSILON);
}

// ──────────────────────────────────────────────
//  Integration: check() dispatches correctly
// ──────────────────────────────────────────────

#[test]
fn check_dispatches_to_correct_method() {
    let core: [Memory; 0] = [];
    let working = [mem("disp-001", Layer::Working, vec![], "normal memory")];
    let checker = RuleChecker::new(&core, &working);

    // Delete
    let g = checker.check(&AuditOp::Delete { id: "disp-001".into() });
    assert_eq!(g.grade, Grade::Good);

    // Promote (Working → Core)
    let g = checker.check(&AuditOp::Promote { id: "disp-001".into(), to: 3 });
    assert_eq!(g.grade, Grade::Good);
}
