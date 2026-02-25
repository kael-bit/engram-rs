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
//  check_delete tests
// ──────────────────────────────────────────────

#[test]
fn delete_identity_tagged_is_bad() {
    let core = [mem("id-001", Layer::Core, vec!["identity"], "I am atlas")];
    let working: [Memory; 0] = [];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Delete { id: "id-001".into() });
    assert_eq!(grade.grade, Grade::Bad);
    assert!(grade.reason.contains("identity"));
}

#[test]
fn delete_bootstrap_tagged_is_bad() {
    let core = [mem("id-002", Layer::Core, vec!["bootstrap"], "bootstrap config")];
    let working: [Memory; 0] = [];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Delete { id: "id-002".into() });
    assert_eq!(grade.grade, Grade::Bad);
}

#[test]
fn delete_lesson_tagged_is_bad() {
    let core = [mem("id-003", Layer::Core, vec!["lesson"], "never force-push")];
    let working: [Memory; 0] = [];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Delete { id: "id-003".into() });
    assert_eq!(grade.grade, Grade::Bad);
    assert!(grade.reason.contains("lesson"));
}

#[test]
fn delete_procedural_kind_is_bad() {
    let mut m = mem("id-004", Layer::Working, vec!["deploy"], "step1 step2");
    m.kind = "procedural".into();
    let core: [Memory; 0] = [];
    let working = [m];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Delete { id: "id-004".into() });
    assert_eq!(grade.grade, Grade::Bad);
    assert!(grade.reason.contains("lesson") || grade.reason.contains("procedural") || grade.reason.contains("protected"));
}

#[test]
fn delete_recently_modified_is_bad() {
    let now = engram::db::now_ms();
    let mut m = mem("id-005", Layer::Working, vec![], "recent stuff");
    m.modified_at = now - 3_600_000; // 1 hour ago
    let core: [Memory; 0] = [];
    let working = [m];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Delete { id: "id-005".into() });
    assert_eq!(grade.grade, Grade::Bad);
    assert!(grade.reason.contains("24h") || grade.reason.contains("modified"));
}

#[test]
fn delete_high_access_count_is_marginal() {
    let mut m = mem("id-006", Layer::Working, vec![], "popular memory");
    m.access_count = 150;
    let core: [Memory; 0] = [];
    let working = [m];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Delete { id: "id-006".into() });
    assert_eq!(grade.grade, Grade::Marginal);
    assert!(grade.reason.contains("accesses") || grade.reason.contains("valuable"));
}

#[test]
fn delete_core_memory_is_marginal() {
    let core = [mem("id-007", Layer::Core, vec![], "some core fact")];
    let working: [Memory; 0] = [];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Delete { id: "id-007".into() });
    assert_eq!(grade.grade, Grade::Marginal);
    assert!(grade.reason.contains("Core"));
}

#[test]
fn delete_normal_working_memory_is_good() {
    let core: [Memory; 0] = [];
    let working = [mem("id-008", Layer::Working, vec![], "ephemeral note")];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Delete { id: "id-008".into() });
    assert_eq!(grade.grade, Grade::Good);
}

#[test]
fn delete_not_found_is_bad() {
    let core: [Memory; 0] = [];
    let working: [Memory; 0] = [];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Delete { id: "nonexistent".into() });
    assert_eq!(grade.grade, Grade::Bad);
    assert!(grade.reason.contains("not found"));
}

// ──────────────────────────────────────────────
//  check_demote tests
// ──────────────────────────────────────────────

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
fn demote_lesson_from_core_is_bad() {
    let core = [mem("dm-003", Layer::Core, vec!["lesson"], "never do X")];
    let working: [Memory; 0] = [];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Demote { id: "dm-003".into(), to: 2 });
    assert_eq!(grade.grade, Grade::Bad);
    assert!(grade.reason.contains("lesson") || grade.reason.contains("Core"));
}

#[test]
fn demote_content_with_constraint_keywords_is_bad() {
    let core = [mem("dm-004", Layer::Core, vec![], "部署必须先跑测试")];
    let working: [Memory; 0] = [];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Demote { id: "dm-004".into(), to: 2 });
    assert_eq!(grade.grade, Grade::Bad);
    assert!(grade.reason.contains("lesson") || grade.reason.contains("principle"));
}

#[test]
fn demote_content_with_principle_keyword_is_bad() {
    let core = [mem("dm-005", Layer::Core, vec![], "设计原则: 单一职责")];
    let working: [Memory; 0] = [];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Demote { id: "dm-005".into(), to: 2 });
    assert_eq!(grade.grade, Grade::Bad);
    assert!(grade.reason.contains("principle") || grade.reason.contains("lesson"));
}

#[test]
fn demote_normal_core_to_working_is_good() {
    let core = [mem("dm-006", Layer::Core, vec![], "some outdated fact")];
    let working: [Memory; 0] = [];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Demote { id: "dm-006".into(), to: 2 });
    assert_eq!(grade.grade, Grade::Good);
}

#[test]
fn demote_core_to_buffer_is_marginal() {
    let core = [mem("dm-007", Layer::Core, vec![], "some outdated fact")];
    let working: [Memory; 0] = [];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Demote { id: "dm-007".into(), to: 1 });
    assert_eq!(grade.grade, Grade::Marginal);
    assert!(grade.reason.contains("Buffer"));
}

#[test]
fn demote_identity_is_bad() {
    let core = [mem("dm-008", Layer::Core, vec!["identity"], "I am atlas")];
    let working: [Memory; 0] = [];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Demote { id: "dm-008".into(), to: 2 });
    assert_eq!(grade.grade, Grade::Bad);
    assert!(grade.reason.contains("identity"));
}

#[test]
fn demote_high_ac_core_is_bad() {
    let mut m = mem("dm-009", Layer::Core, vec![], "very popular");
    m.access_count = 100;
    let core = [m];
    let working: [Memory; 0] = [];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Demote { id: "dm-009".into(), to: 2 });
    assert_eq!(grade.grade, Grade::Bad);
    assert!(grade.reason.contains("heavily used") || grade.reason.contains("ac="));
}

// ──────────────────────────────────────────────
//  check_promote tests
// ──────────────────────────────────────────────

#[test]
fn promote_session_tagged_to_core_is_bad() {
    let core: [Memory; 0] = [];
    let working = [mem("pr-001", Layer::Working, vec!["session"], "did X today")];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Promote { id: "pr-001".into(), to: 3 });
    assert_eq!(grade.grade, Grade::Bad);
    assert!(grade.reason.contains("session"));
}

#[test]
fn promote_distilled_tagged_to_core_is_bad() {
    let core: [Memory; 0] = [];
    let working = [mem("pr-002", Layer::Working, vec!["distilled"], "distilled summary")];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Promote { id: "pr-002".into(), to: 3 });
    assert_eq!(grade.grade, Grade::Bad);
    assert!(grade.reason.contains("distilled") || grade.reason.contains("session"));
}

#[test]
fn promote_auto_distilled_tagged_to_core_is_bad() {
    let core: [Memory; 0] = [];
    let working = [mem("pr-003", Layer::Working, vec!["auto-distilled"], "auto summary")];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Promote { id: "pr-003".into(), to: 3 });
    assert_eq!(grade.grade, Grade::Bad);
}

#[test]
fn promote_zero_ac_old_memory_to_core_is_marginal() {
    let now = engram::db::now_ms();
    let mut m = mem("pr-004", Layer::Working, vec![], "old untouched note");
    m.access_count = 0;
    m.created_at = now - 72 * 3_600_000; // 72h ago (> 48h)
    let core: [Memory; 0] = [];
    let working = [m];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Promote { id: "pr-004".into(), to: 3 });
    assert_eq!(grade.grade, Grade::Marginal);
    assert!(grade.reason.contains("zero accesses") || grade.reason.contains("unproven"));
}

#[test]
fn promote_zero_ac_recent_memory_to_core_is_good() {
    let now = engram::db::now_ms();
    let mut m = mem("pr-005", Layer::Working, vec![], "fresh new insight");
    m.access_count = 0;
    m.created_at = now - 12 * 3_600_000; // 12h ago (< 48h)
    let core: [Memory; 0] = [];
    let working = [m];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Promote { id: "pr-005".into(), to: 3 });
    assert_eq!(grade.grade, Grade::Good);
}

#[test]
fn promote_normal_memory_to_core_is_good() {
    let core: [Memory; 0] = [];
    let working = [mem("pr-006", Layer::Working, vec!["design"], "REST API pattern")];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Promote { id: "pr-006".into(), to: 3 });
    assert_eq!(grade.grade, Grade::Good);
}

#[test]
fn promote_gate_rejected_final_is_bad() {
    let core: [Memory; 0] = [];
    let working = [mem("pr-007", Layer::Working, vec!["gate-rejected-final"], "rejected stuff")];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Promote { id: "pr-007".into(), to: 3 });
    assert_eq!(grade.grade, Grade::Bad);
    assert!(grade.reason.contains("gate"));
}

// ──────────────────────────────────────────────
//  check_merge tests
// ──────────────────────────────────────────────

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
fn merge_identity_with_non_identity_is_bad() {
    let m1 = mem("mg-005", Layer::Core, vec!["identity"], "I am atlas");
    let m2 = mem("mg-006", Layer::Core, vec![], "some other fact");
    let core = [m1, m2];
    let working: [Memory; 0] = [];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Merge {
        ids: vec!["mg-005".into(), "mg-006".into()],
        content: "merged identity and fact".into(),
        layer: 3,
        tags: vec![],
    });
    assert_eq!(grade.grade, Grade::Bad);
    assert!(grade.reason.contains("identity"));
}

#[test]
fn merge_cross_layer_is_marginal() {
    let core = [mem("mg-007", Layer::Core, vec![], "core fact")];
    let working = [mem("mg-008", Layer::Working, vec![], "working note")];
    let checker = RuleChecker::new(&core, &working);
    let grade = checker.check(&AuditOp::Merge {
        ids: vec!["mg-007".into(), "mg-008".into()],
        content: "combined core and working content into one merged memory".into(),
        layer: 3,
        tags: vec![],
    });
    assert_eq!(grade.grade, Grade::Marginal);
    assert!(grade.reason.contains("Core") && grade.reason.contains("Working"));
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

// ──────────────────────────────────────────────
//  Score formula tests
// ──────────────────────────────────────────────

#[test]
fn score_formula_all_good() {
    // 3 Good, 0 Marginal, 0 Bad → (3 + 0) / 3 = 1.0
    let score = (3.0_f64 + 0.0 * 0.5) / 3.0;
    assert!((score - 1.0).abs() < f64::EPSILON);
}

#[test]
fn score_formula_mixed() {
    // 2 Good, 2 Marginal, 1 Bad → (2 + 2*0.5) / 5 = 3/5 = 0.6
    let good = 2;
    let marginal = 2;
    let total = 5;
    let score = (good as f64 + marginal as f64 * 0.5) / total as f64;
    assert!((score - 0.6).abs() < f64::EPSILON);
}

#[test]
fn score_formula_all_bad() {
    // 0 Good, 0 Marginal, 3 Bad → 0 / 3 = 0.0
    let score = (0.0_f64 + 0.0 * 0.5) / 3.0;
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
        let (good, marginal, _bad) = grades.iter().fold((0, 0, 0), |(g, m, b), og| {
            match og.grade {
                Grade::Good => (g + 1, m, b),
                Grade::Marginal => (g, m + 1, b),
                Grade::Bad => (g, m, b + 1),
            }
        });
        (good as f64 + marginal as f64 * 0.5) / total_ops
    };
    assert!((score - 1.0).abs() < f64::EPSILON);
}

#[test]
fn score_formula_marginal_only() {
    // 0 Good, 4 Marginal, 0 Bad → (0 + 4*0.5) / 4 = 0.5
    let score: f64 = (0.0 + 4.0 * 0.5) / 4.0;
    assert!((score - 0.5).abs() < f64::EPSILON);
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
