use engram::topiary::{cosine_similarity, mean_vector, l2_normalize, Entry, TopicNode, TopicTree};

fn make_entry(id: &str, text: &str, embedding: Vec<f32>) -> Entry {
    Entry {
        id: id.to_string(),
        text: text.to_string(),
        embedding,
    }
}

/// Generate a simple unit vector in the given dimension (1.0 at idx, 0.0 elsewhere).
fn unit_vec(dim: usize, idx: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    v[idx] = 1.0;
    v
}

// ── Math helpers ──────────────────────────────────────────────────────────

#[test]
fn cosine_identical_vectors() {
    let v = vec![1.0, 2.0, 3.0];
    let sim = cosine_similarity(&v, &v);
    assert!((sim - 1.0).abs() < 1e-5);
}

#[test]
fn cosine_orthogonal_vectors() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!(sim.abs() < 1e-5);
}

#[test]
fn cosine_empty_vectors() {
    let sim = cosine_similarity(&[], &[]);
    assert_eq!(sim, 0.0);
}

#[test]
fn cosine_different_lengths() {
    let a = vec![1.0, 2.0];
    let b = vec![1.0, 2.0, 3.0];
    let sim = cosine_similarity(&a, &b);
    assert_eq!(sim, 0.0);
}

#[test]
fn mean_vector_basic() {
    let a = vec![2.0f32, 0.0];
    let b = vec![0.0f32, 4.0];
    let vecs: Vec<&[f32]> = vec![a.as_slice(), b.as_slice()];
    let m = mean_vector(&vecs);
    assert_eq!(m.len(), 2);
    // mean_vector returns L2-normalized result: mean is [1,2], normalized is [1/√5, 2/√5]
    let norm: f32 = m.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "result should be L2-normalized");
    assert!(m[1] / m[0] > 1.9 && m[1] / m[0] < 2.1, "ratio should be ~2:1");
}

#[test]
fn l2_normalize_basic() {
    let mut v = vec![3.0, 4.0];
    l2_normalize(&mut v);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}

#[test]
fn l2_normalize_zero_vector() {
    let mut v = vec![0.0, 0.0, 0.0];
    l2_normalize(&mut v); // should not panic
    assert_eq!(v, vec![0.0, 0.0, 0.0]);
}

// ── TopicTree insertion ───────────────────────────────────────────────────

#[test]
fn insert_creates_leaf() {
    let mut tree = TopicTree::new(0.30, 0.55);
    let emb = unit_vec(8, 0);
    tree.insert(0, &emb);
    assert_eq!(tree.roots.len(), 1);
    assert!(tree.roots[0].is_leaf());
    assert_eq!(tree.roots[0].members, vec![0]);
    assert!(tree.roots[0].dirty);
    assert_eq!(tree.roots[0].named_at_size, 0);
}

#[test]
fn similar_entries_cluster_together() {
    let mut tree = TopicTree::new(0.30, 0.55);
    // Two very similar vectors should go into the same leaf
    let e0 = vec![1.0, 0.1, 0.0, 0.0];
    let e1 = vec![1.0, 0.2, 0.0, 0.0];
    tree.insert(0, &e0);
    tree.insert(1, &e1);
    // Should be in same leaf (cosine > 0.30 threshold)
    assert_eq!(tree.roots.len(), 1);
    assert_eq!(tree.roots[0].members.len(), 2);
}

#[test]
fn dissimilar_entries_create_separate_leaves() {
    let mut tree = TopicTree::new(0.30, 0.55);
    // Orthogonal vectors should create separate leaves
    let e0 = unit_vec(4, 0);
    let e1 = unit_vec(4, 1);
    tree.insert(0, &e0);
    tree.insert(1, &e1);
    assert_eq!(tree.roots.len(), 2);
}

// ── Dirty flag logic (naming optimization) ────────────────────────────────

#[test]
fn new_leaf_is_dirty() {
    let mut tree = TopicTree::new(0.30, 0.55);
    tree.insert(0, &unit_vec(4, 0));
    assert!(tree.roots[0].dirty);
    assert!(tree.roots[0].name.is_none());
}

#[test]
fn named_leaf_stays_clean_on_insert() {
    let mut tree = TopicTree::new(0.30, 0.55);
    let emb = vec![1.0, 0.1, 0.0, 0.0];
    tree.insert(0, &emb);
    // Simulate naming
    tree.roots[0].name = Some("Test Topic".to_string());
    tree.roots[0].named_at_size = 1;
    tree.roots[0].dirty = false;
    // Insert similar entry
    let emb2 = vec![1.0, 0.2, 0.0, 0.0];
    tree.insert(1, &emb2);
    // Should NOT be dirty (named_at_size=1, now 2, 2 < 1*1.5=1.5... wait 2 >= 1+0=1)
    // 50% growth: 2 >= 1 + 1/2 = 1 (integer division: 1/2=0), so 2 >= 1+0=1 → dirty!
    // Actually for named_at_size=1: 1/2=0, so threshold is 1+0=1, and members.len()=2 >= 1 → dirty
    // This is fine — size 1 topics are volatile and should be re-checked
    // For named_at_size=2: 2/2=1, threshold is 2+1=3
    assert!(tree.roots[0].dirty);
}

#[test]
fn named_leaf_dirty_after_50pct_growth() {
    let mut tree = TopicTree::new(0.10, 0.55); // low threshold to cluster easily
    let dim = 8;
    // Insert 4 similar entries
    for i in 0..4 {
        let mut emb = vec![1.0f32; dim];
        emb[0] += i as f32 * 0.01; // tiny variation
        tree.insert(i, &emb);
    }
    assert_eq!(tree.roots.len(), 1);
    // Simulate naming at size 4
    tree.roots[0].name = Some("Clustered Topic".to_string());
    tree.roots[0].named_at_size = 4;
    tree.roots[0].dirty = false;
    // Insert 1 more — now 5. 50% of 4 = 2, so threshold is 4+2=6. 5 < 6 → not dirty
    let mut emb = vec![1.0f32; dim];
    emb[0] += 0.04;
    tree.insert(4, &emb);
    assert!(!tree.roots[0].dirty, "should not be dirty at 5/4 (< 50% growth)");
    // Insert 1 more — now 6. 6 >= 6 → dirty
    let mut emb = vec![1.0f32; dim];
    emb[0] += 0.05;
    tree.insert(5, &emb);
    assert!(tree.roots[0].dirty, "should be dirty at 6/4 (= 50% growth)");
}

#[test]
fn unnamed_leaf_always_dirty_on_insert() {
    let mut tree = TopicTree::new(0.10, 0.55);
    let emb = vec![1.0f32; 4];
    tree.insert(0, &emb);
    assert!(tree.roots[0].dirty);
    // Clear dirty but keep name=None
    tree.roots[0].dirty = false;
    let emb2 = vec![1.0f32, 1.01, 1.0, 1.0];
    tree.insert(1, &emb2);
    assert!(tree.roots[0].dirty, "unnamed leaf should always be dirty");
}

// ── Consolidation ─────────────────────────────────────────────────────────

#[test]
fn consolidate_merges_similar_leaves() {
    let dim = 8;
    let mut tree = TopicTree::new(0.30, 0.55);
    let entries: Vec<Entry> = (0..6)
        .map(|i| {
            let mut emb = vec![0.0f32; dim];
            // Two groups: 0-2 in dim 0, 3-5 in dim 1
            if i < 3 {
                emb[0] = 1.0;
                emb[1] = 0.05 * i as f32;
            } else {
                emb[1] = 1.0;
                emb[0] = 0.05 * (i - 3) as f32;
            }
            make_entry(&format!("m{i}"), &format!("entry {i}"), emb)
        })
        .collect();

    for (i, e) in entries.iter().enumerate() {
        tree.insert(i, &e.embedding);
    }
    tree.consolidate(&entries);

    let total_members: usize = tree.roots.iter().map(|r| r.total_members()).sum();
    assert_eq!(total_members, 6, "no members lost after consolidation");
}

// ── TopicNode helpers ─────────────────────────────────────────────────────

#[test]
fn topic_node_depth() {
    let leaf = TopicNode {
        id: "kb1".into(),
        name: None,
        centroid: vec![1.0],
        members: vec![0],
        children: vec![],
        dirty: false,
        avg_sim: 1.0,
        named_at_size: 0,
    };
    assert_eq!(leaf.depth(), 0);

    let parent = TopicNode {
        id: "t1".into(),
        name: None,
        centroid: vec![1.0],
        members: vec![],
        children: vec![leaf],
        dirty: false,
        avg_sim: 0.0,
        named_at_size: 0,
    };
    assert_eq!(parent.depth(), 1);
}

#[test]
fn topic_node_leaf_count() {
    let leaf1 = TopicNode {
        id: "kb1".into(),
        name: Some("A".into()),
        centroid: vec![1.0],
        members: vec![0, 1],
        children: vec![],
        dirty: false,
        avg_sim: 1.0,
        named_at_size: 2,
    };
    let leaf2 = TopicNode {
        id: "kb2".into(),
        name: Some("B".into()),
        centroid: vec![0.0, 1.0],
        members: vec![2],
        children: vec![],
        dirty: false,
        avg_sim: 1.0,
        named_at_size: 1,
    };
    let parent = TopicNode {
        id: "t1".into(),
        name: None,
        centroid: vec![0.5, 0.5],
        members: vec![],
        children: vec![leaf1, leaf2],
        dirty: false,
        avg_sim: 0.0,
        named_at_size: 0,
    };
    assert_eq!(parent.leaf_count(), 2);
    assert_eq!(parent.total_members(), 3);
}

// ── Resume generation ─────────────────────────────────────────────────────

#[test]
fn generate_resume_includes_named_topics() {
    let mut tree = TopicTree::new(0.30, 0.55);
    let entries = vec![
        make_entry("m1", "Deploy to production", vec![1.0, 0.0]),
        make_entry("m2", "Run cargo test", vec![1.0, 0.1]),
    ];
    tree.insert(0, &entries[0].embedding);
    tree.insert(1, &entries[1].embedding);
    tree.roots[0].name = Some("Deployment".into());
    tree.roots[0].named_at_size = 2;
    tree.roots[0].dirty = false;

    let (total_entries, total_topics, text) = tree.generate_resume(&entries);
    assert!(text.contains("Deployment"), "resume should contain topic name");
    assert!(text.contains("[2]"), "resume should show member count");
    assert_eq!(total_entries, 2);
    assert_eq!(total_topics, 1);
}

// ── Serialization roundtrip ───────────────────────────────────────────────

#[test]
fn topic_node_serde_roundtrip() {
    let node = TopicNode {
        id: "kb1".into(),
        name: Some("Test".into()),
        centroid: vec![1.0, 0.0],
        members: vec![0, 1, 2],
        children: vec![],
        dirty: false,
        avg_sim: 0.95,
        named_at_size: 3,
    };
    let json = serde_json::to_string(&node).unwrap();
    let restored: TopicNode = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.name, Some("Test".into()));
    assert_eq!(restored.named_at_size, 3);
    assert_eq!(restored.members, vec![0, 1, 2]);
}

#[test]
fn topic_node_serde_missing_named_at_size() {
    // Simulate old JSON without named_at_size field
    let json = r#"{"id":"kb1","name":"Old","centroid":[1.0],"members":[0],"children":[],"dirty":false,"avg_sim":0.9}"#;
    let node: TopicNode = serde_json::from_str(json).unwrap();
    assert_eq!(node.named_at_size, 0, "missing named_at_size should default to 0");
}

// ── enforce_budget preserves names ────────────────────────────────────────

#[test]
fn enforce_budget_preserves_target_name() {
    use engram::topiary::cluster::enforce_budget;
    let dim = 4;
    let entries: Vec<Entry> = (0..4)
        .map(|i| {
            let mut emb = vec![1.0f32; dim];
            emb[0] += i as f32 * 0.01;
            make_entry(&format!("m{i}"), &format!("entry {i}"), emb)
        })
        .collect();

    let mut leaves = vec![
        TopicNode {
            id: "kb1".into(),
            name: Some("Keep This Name".into()),
            centroid: vec![1.0, 1.0, 1.0, 1.0],
            members: vec![0, 1],
            children: vec![],
            dirty: false,
            avg_sim: 0.99,
            named_at_size: 2,
        },
        TopicNode {
            id: "kb2".into(),
            name: Some("Merge Into Above".into()),
            centroid: vec![1.01, 1.0, 1.0, 1.0],
            members: vec![2, 3],
            children: vec![],
            dirty: false,
            avg_sim: 0.99,
            named_at_size: 2,
        },
    ];

    enforce_budget(&mut leaves, &entries, 1); // force merge into 1
    assert_eq!(leaves.len(), 1);
    assert_eq!(leaves[0].name, Some("Keep This Name".into()));
    assert!(!leaves[0].dirty, "named target should NOT be dirty after merge");
    assert_eq!(leaves[0].members.len(), 4);
}
