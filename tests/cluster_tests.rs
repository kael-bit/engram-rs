use engram::consolidate::{
    batch_clusters, cluster_memories, combined_similarity, generate_label, tag_jaccard,
    MemoryCluster,
};
use engram::db::{Layer, Memory};

fn make_memory(id: &str, content: &str, tags: &[&str], importance: f64) -> Memory {
    Memory {
        id: id.to_string(),
        content: content.to_string(),
        layer: Layer::Working,
        importance,
        created_at: 1000,
        last_accessed: 2000,
        access_count: 5,
        repetition_count: 0,
        decay_rate: 1.0,
        source: "test".to_string(),
        tags: tags.iter().map(|s| s.to_string()).collect(),
        namespace: "default".to_string(),
        embedding: None,
        kind: "semantic".to_string(),
        modified_at: 1500,
    }
}

#[test]
fn test_tag_jaccard_identical() {
    let a = vec!["lesson".into(), "deploy".into()];
    let b = vec!["deploy".into(), "lesson".into()];
    assert!((tag_jaccard(&a, &b) - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_tag_jaccard_disjoint() {
    let a = vec!["lesson".into()];
    let b = vec!["config".into()];
    assert!((tag_jaccard(&a, &b) - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_tag_jaccard_partial_overlap() {
    let a = vec!["lesson".into(), "deploy".into()];
    let b = vec!["deploy".into(), "config".into()];
    // intersection={deploy}, union={lesson,deploy,config} → 1/3
    assert!((tag_jaccard(&a, &b) - 1.0 / 3.0).abs() < 0.001);
}

#[test]
fn test_tag_jaccard_both_empty() {
    let a: Vec<String> = vec![];
    let b: Vec<String> = vec![];
    assert!((tag_jaccard(&a, &b) - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_combined_similarity_tag_boost() {
    // Two orthogonal embeddings (cosine ≈ 0) but identical tags
    let mut emb_a = vec![0.0_f32; 128];
    emb_a[0] = 1.0;
    let mut emb_b = vec![0.0_f32; 128];
    emb_b[1] = 1.0;
    let tags = vec!["lesson".into(), "deploy".into()];
    let sim = combined_similarity(&emb_a, &emb_b, &tags, &tags);
    // cosine ≈ 0, jaccard = 1.0 → 0*0.7 + 1.0*0.3 = 0.3
    assert!((sim - 0.3).abs() < 0.01);
}

#[test]
fn test_combined_similarity_pure_cosine() {
    // Identical embeddings, no tags
    let emb = vec![0.5_f32; 128];
    let no_tags: Vec<String> = vec![];
    let sim = combined_similarity(&emb, &emb, &no_tags, &no_tags);
    // cosine = 1.0, jaccard = 0.0 → 1.0*0.7 + 0*0.3 = 0.7
    assert!((sim - 0.7).abs() < 0.01);
}

#[test]
fn test_cluster_no_embeddings() {
    let memories = vec![
        make_memory("aaa", "hello world", &["test"], 0.5),
        make_memory("bbb", "foo bar", &["test"], 0.6),
    ];
    let clusters = cluster_memories(&memories, &[], 0.50);
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].memories.len(), 2);
    assert!(clusters[0].label.contains("unclustered"));
}

#[test]
fn test_cluster_identical_embeddings_same_tags() {
    let memories = vec![
        make_memory("aaa", "hello world", &["greet"], 0.5),
        make_memory("bbb", "hello world again", &["greet"], 0.6),
    ];
    let emb = vec![0.1_f32; 128];
    let embeddings = vec![
        ("aaa".to_string(), emb.clone()),
        ("bbb".to_string(), emb.clone()),
    ];
    let clusters = cluster_memories(&memories, &embeddings, 0.50);
    // Identical embeddings + same tags → combined ≈ 1.0 → same cluster
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].memories.len(), 2);
    assert!(!clusters[0].similarities.is_empty());
}

#[test]
fn test_cluster_orthogonal_embeddings_no_tag_overlap() {
    let memories = vec![
        make_memory("aaa", "topic A", &["a"], 0.5),
        make_memory("bbb", "topic B", &["b"], 0.6),
    ];
    let mut emb_a = vec![0.0_f32; 128];
    emb_a[0] = 1.0;
    let mut emb_b = vec![0.0_f32; 128];
    emb_b[1] = 1.0;
    let embeddings = vec![
        ("aaa".to_string(), emb_a),
        ("bbb".to_string(), emb_b),
    ];
    let clusters = cluster_memories(&memories, &embeddings, 0.50);
    // Orthogonal + different tags → combined ≈ 0 → separate clusters
    assert_eq!(clusters.len(), 2);
}

#[test]
fn test_cluster_tag_boost_groups_weak_cosine() {
    // Weak cosine (orthogonal) but strong tag overlap
    let memories = vec![
        make_memory("aaa", "部署出错了", &["lesson", "deploy"], 0.5),
        make_memory("bbb", "deploy went wrong", &["lesson", "deploy"], 0.6),
    ];
    let mut emb_a = vec![0.0_f32; 128];
    emb_a[0] = 1.0;
    let mut emb_b = vec![0.0_f32; 128];
    emb_b[1] = 1.0;
    let embeddings = vec![
        ("aaa".to_string(), emb_a),
        ("bbb".to_string(), emb_b),
    ];
    // cosine ≈ 0, tag_jaccard = 1.0 → combined = 0*0.7 + 1.0*0.3 = 0.3
    // With threshold 0.25, they cluster together
    let clusters = cluster_memories(&memories, &embeddings, 0.25);
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].memories.len(), 2);

    // With threshold 0.50, they don't (0.3 < 0.50)
    let clusters = cluster_memories(&memories, &embeddings, 0.50);
    assert_eq!(clusters.len(), 2);
}

#[test]
fn test_cluster_size_capped() {
    // Create 15 memories with identical embeddings — would form one giant
    // cluster without the cap.
    let emb = vec![0.1_f32; 128];
    let mut memories = Vec::new();
    let mut embeddings = Vec::new();
    for i in 0..15 {
        let id = format!("mem-{:03}", i);
        memories.push(make_memory(&id, &format!("content {}", i), &["common"], 0.5));
        embeddings.push((id, emb.clone()));
    }
    let clusters = cluster_memories(&memories, &embeddings, 0.50);
    // Every cluster should be <= MAX_CLUSTER_SIZE (10)
    for c in &clusters {
        assert!(
            c.memories.len() <= 10,
            "cluster '{}' has {} memories, exceeds cap of 10",
            c.label,
            c.memories.len()
        );
    }
    // All 15 memories should still be present across clusters
    let total: usize = clusters.iter().map(|c| c.memories.len()).sum();
    assert_eq!(total, 15);
    // Should have at least 2 clusters
    assert!(clusters.len() >= 2);
}

#[test]
fn test_batch_clusters_simple() {
    let clusters = vec![
        MemoryCluster {
            memories: vec![make_memory("a", "short", &[], 0.5)],
            label: "test".to_string(),
            similarities: vec![],
        },
        MemoryCluster {
            memories: vec![make_memory("b", "short", &[], 0.5)],
            label: "test2".to_string(),
            similarities: vec![],
        },
    ];
    let batches = batch_clusters(&clusters, 100_000);
    // Both should fit in one batch
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].len(), 2);
}

#[test]
fn test_generate_label_common_tag() {
    let memories = vec![
        make_memory("a", "foo", &["deploy", "lesson"], 0.5),
        make_memory("b", "bar", &["deploy", "config"], 0.6),
    ];
    let label = generate_label(&memories);
    assert_eq!(label, "deploy");
}
