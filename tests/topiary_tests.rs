use engram::topiary::worker::{assign_fallback_names, count_unnamed_leaves, inherit_names};
use engram::topiary::{Entry, TopicNode, TopicTree};

fn make_leaf(id: &str, name: Option<&str>, members: Vec<usize>) -> TopicNode {
    let size = members.len();
    TopicNode {
        id: id.to_string(),
        name: name.map(|s| s.to_string()),
        centroid: vec![],
        members,
        children: vec![],
        dirty: name.is_none(),
        avg_sim: 0.0,
        named_at_size: if name.is_some() { size } else { 0 },
    }
}

fn make_entry(id: &str) -> Entry {
    Entry {
        id: id.to_string(),
        text: String::new(),
        embedding: vec![],
        tags: vec![],
    }
}

fn make_entry_with_tags(id: &str, tags: &[&str]) -> Entry {
    Entry {
        id: id.to_string(),
        text: String::new(),
        embedding: vec![],
        tags: tags.iter().map(|t| t.to_string()).collect(),
    }
}

fn old_tree_json(roots: Vec<TopicNode>) -> String {
    let tree = TopicTree::new(0.3, 0.55).with_roots(roots);
    serde_json::to_string(&tree).unwrap()
}

// ── inherit_names ─────────────────────────────────────────────────────

#[test]
fn inherit_exact_match() {
    let entries = vec![make_entry("m1"), make_entry("m2"), make_entry("m3")];
    let mut roots = vec![make_leaf("kb1", None, vec![0, 1, 2])];

    let tree_json = old_tree_json(vec![make_leaf("old1", Some("Deploy"), vec![0, 1, 2])]);
    let ids_json = serde_json::to_string(&vec!["m1", "m2", "m3"]).unwrap();

    let n = inherit_names(&mut roots, &entries, Some(tree_json), Some(ids_json));
    assert_eq!(n, 1);
    assert_eq!(roots[0].name.as_deref(), Some("Deploy"));
    assert!(!roots[0].dirty);
}

#[test]
fn inherit_above_threshold() {
    // Old: {m1,m2,m3,m4,m5}, New: {m1,m2,m3,m6} → Jaccard 3/6=0.5 ≥ 0.3
    let entries = vec![
        make_entry("m1"),
        make_entry("m2"),
        make_entry("m3"),
        make_entry("m6"),
    ];
    let mut roots = vec![make_leaf("kb1", None, vec![0, 1, 2, 3])];

    let tree_json = old_tree_json(vec![make_leaf(
        "old1",
        Some("Config"),
        vec![0, 1, 2, 3, 4],
    )]);
    let ids_json = serde_json::to_string(&vec!["m1", "m2", "m3", "m4", "m5"]).unwrap();

    let n = inherit_names(&mut roots, &entries, Some(tree_json), Some(ids_json));
    assert_eq!(n, 1);
    assert_eq!(roots[0].name.as_deref(), Some("Config"));
}

#[test]
fn inherit_below_threshold_rejected() {
    // Old: {m1..m10}, New: {m1, m11..m19} → Jaccard 1/19 ≈ 0.05 < 0.3
    let entries: Vec<Entry> = (0..10)
        .map(|i| {
            let id = if i == 0 {
                "m1".into()
            } else {
                format!("m{}", 10 + i)
            };
            make_entry(&id)
        })
        .collect();
    let mut roots = vec![make_leaf("kb1", None, (0..10).collect())];

    let tree_json = old_tree_json(vec![make_leaf("old1", Some("Big"), (0..10).collect())]);
    let ids_json =
        serde_json::to_string(&(1..=10).map(|i| format!("m{i}")).collect::<Vec<_>>()).unwrap();

    let n = inherit_names(&mut roots, &entries, Some(tree_json), Some(ids_json));
    assert_eq!(n, 0);
    assert!(roots[0].name.is_none());
}

#[test]
fn inherit_greedy_dedup() {
    // Old "Deploy" = {m1,m2,m3,m4}
    // New A = {m1,m2}, B = {m3,m4} — both Jaccard 0.5 to "Deploy"
    // Only one should get the name
    let entries = vec![
        make_entry("m1"),
        make_entry("m2"),
        make_entry("m3"),
        make_entry("m4"),
    ];
    let mut roots = vec![
        make_leaf("kb1", None, vec![0, 1]),
        make_leaf("kb2", None, vec![2, 3]),
    ];

    let tree_json = old_tree_json(vec![make_leaf(
        "old1",
        Some("Deploy"),
        vec![0, 1, 2, 3],
    )]);
    let ids_json = serde_json::to_string(&vec!["m1", "m2", "m3", "m4"]).unwrap();

    let n = inherit_names(&mut roots, &entries, Some(tree_json), Some(ids_json));
    assert_eq!(n, 1);
    let named = roots
        .iter()
        .filter(|r| r.name.as_deref() == Some("Deploy"))
        .count();
    assert_eq!(named, 1, "only one topic should get the name");
}

#[test]
fn inherit_multiple_old_topics() {
    // Old: "Deploy"={m1,m2}, "Config"={m3,m4}
    // New: A={m1,m2,m5}, B={m3,m4}
    let entries = vec![
        make_entry("m1"),
        make_entry("m2"),
        make_entry("m3"),
        make_entry("m4"),
        make_entry("m5"),
    ];
    let mut roots = vec![
        make_leaf("kb1", None, vec![0, 1, 4]),
        make_leaf("kb2", None, vec![2, 3]),
    ];

    let tree_json = old_tree_json(vec![
        make_leaf("old1", Some("Deploy"), vec![0, 1]),
        make_leaf("old2", Some("Config"), vec![2, 3]),
    ]);
    let ids_json = serde_json::to_string(&vec!["m1", "m2", "m3", "m4"]).unwrap();

    let n = inherit_names(&mut roots, &entries, Some(tree_json), Some(ids_json));
    assert_eq!(n, 2);
    assert_eq!(roots[0].name.as_deref(), Some("Deploy"));
    assert_eq!(roots[1].name.as_deref(), Some("Config"));
}

#[test]
fn inherit_no_cache() {
    let entries = vec![make_entry("m1")];
    let mut roots = vec![make_leaf("kb1", None, vec![0])];
    assert_eq!(inherit_names(&mut roots, &entries, None, None), 0);
}

#[test]
fn inherit_already_named_skipped() {
    let entries = vec![make_entry("m1"), make_entry("m2")];
    let mut roots = vec![make_leaf("kb1", Some("Existing"), vec![0, 1])];

    let tree_json = old_tree_json(vec![make_leaf("old1", Some("OldName"), vec![0, 1])]);
    let ids_json = serde_json::to_string(&vec!["m1", "m2"]).unwrap();

    let n = inherit_names(&mut roots, &entries, Some(tree_json), Some(ids_json));
    assert_eq!(n, 0);
    assert_eq!(roots[0].name.as_deref(), Some("Existing"));
}

#[test]
fn inherit_best_match_wins() {
    // Old: "Alpha"={m1,m2}, "Beta"={m3,m4,m5}
    // New A={m1,m2,m3} best→Alpha(0.67), New B={m3,m4,m5} best→Beta(1.0)
    // B gets Beta first (higher score), A gets Alpha
    let entries = vec![
        make_entry("m1"),
        make_entry("m2"),
        make_entry("m3"),
        make_entry("m4"),
        make_entry("m5"),
    ];
    let mut roots = vec![
        make_leaf("kb1", None, vec![0, 1, 2]),
        make_leaf("kb2", None, vec![2, 3, 4]),
    ];

    let tree_json = old_tree_json(vec![
        make_leaf("old1", Some("Alpha"), vec![0, 1]),
        make_leaf("old2", Some("Beta"), vec![2, 3, 4]),
    ]);
    let ids_json = serde_json::to_string(&vec!["m1", "m2", "m3", "m4", "m5"]).unwrap();

    let n = inherit_names(&mut roots, &entries, Some(tree_json), Some(ids_json));
    assert_eq!(n, 2);
    assert_eq!(roots[0].name.as_deref(), Some("Alpha"));
    assert_eq!(roots[1].name.as_deref(), Some("Beta"));
}

// ── count_unnamed ─────────────────────────────────────────────────────

#[test]
fn count_unnamed_mixed() {
    let roots = vec![
        make_leaf("kb1", Some("Named"), vec![0]),
        make_leaf("kb2", None, vec![1]),
        make_leaf("kb3", Some("Also"), vec![2]),
        make_leaf("kb4", None, vec![3]),
    ];
    assert_eq!(count_unnamed_leaves(&roots), 2);
}

// ── fallback names ────────────────────────────────────────────────────

#[test]
fn fallback_uses_most_common_tag() {
    let entries = vec![
        make_entry_with_tags("m1", &["deploy-flow", "ops"]),
        make_entry_with_tags("m2", &["deploy-flow"]),
        make_entry_with_tags("m3", &["ops"]),
    ];
    let mut roots = vec![make_leaf("kb1", None, vec![0, 1, 2])];
    let n = assign_fallback_names(&mut roots, &entries);
    assert_eq!(n, 1);
    let name = roots[0].name.as_deref().unwrap();
    assert!(
        name == "Deploy Flow" || name == "Ops",
        "expected tag-based name, got: {name}"
    );
    assert!(!roots[0].dirty);
}

#[test]
fn fallback_no_tags_uses_topic_id() {
    let entries = vec![make_entry("m1"), make_entry("m2")];
    let mut roots = vec![make_leaf("kb1", None, vec![0, 1])];
    let n = assign_fallback_names(&mut roots, &entries);
    assert_eq!(n, 1);
    assert_eq!(roots[0].name.as_deref(), Some("Topic kb1"));
}

#[test]
fn fallback_skips_already_named() {
    let entries = vec![make_entry("m1")];
    let mut roots = vec![make_leaf("kb1", Some("Existing"), vec![0])];
    let n = assign_fallback_names(&mut roots, &entries);
    assert_eq!(n, 0);
    assert_eq!(roots[0].name.as_deref(), Some("Existing"));
}
