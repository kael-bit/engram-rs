//! Semantic clustering of memories by combined similarity (cosine + tag Jaccard).
//!
//! Groups memories that are semantically related using single-linkage clustering.
//! Clusters are capped at MAX_CLUSTER_SIZE to prevent chain drift (A≈B, B≈C but A≠C).
//! Oversized clusters are split by picking the two most distant members as seeds.

use crate::ai::cosine_similarity;
use crate::db::Memory;
use std::collections::{HashMap, HashSet};

/// Weight for cosine similarity in the combined clustering score.
const COSINE_WEIGHT: f64 = 0.7;
/// Weight for tag Jaccard similarity in the combined clustering score.
const TAG_WEIGHT: f64 = 0.3;
/// Maximum memories per cluster. Prevents chain drift in single-linkage.
const MAX_CLUSTER_SIZE: usize = 10;

/// A cluster of semantically related memories.
#[derive(Debug, Clone)]
pub struct MemoryCluster {
    /// Memories in this cluster.
    pub memories: Vec<Memory>,
    /// Human-readable label (most common tag or first words of top memory).
    pub label: String,
    /// Pairwise similarity scores within the cluster: (id1, id2, combined_score).
    pub similarities: Vec<(String, String, f64)>,
}

/// Compute Jaccard similarity between two tag sets.
/// Returns intersection / union, or 0.0 if both are empty.
fn tag_jaccard(tags_a: &[String], tags_b: &[String]) -> f64 {
    let set_a: HashSet<&str> = tags_a.iter().map(|s| s.as_str()).collect();
    let set_b: HashSet<&str> = tags_b.iter().map(|s| s.as_str()).collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Compute combined similarity: `cosine * 0.7 + tag_jaccard * 0.3`.
///
/// Tags carry strong semantic signal that embeddings miss, especially for CJK
/// where cosine similarity is weak (~0.55-0.60 for paraphrases). Two memories
/// both tagged `[lesson, deploy]` get a tag boost even if their content wording
/// differs.
fn combined_similarity(emb_a: &[f32], emb_b: &[f32], tags_a: &[String], tags_b: &[String]) -> f64 {
    let cosine = cosine_similarity(emb_a, emb_b);
    let jaccard = tag_jaccard(tags_a, tags_b);
    cosine * COSINE_WEIGHT + jaccard * TAG_WEIGHT
}

/// Cluster memories by combined similarity (cosine + tag Jaccard) using
/// single-linkage clustering.
///
/// Two memories are linked if `cosine * 0.7 + tag_jaccard * 0.3 > threshold`.
/// Memories without embeddings are grouped into a single "unclustered" group.
/// Clusters exceeding MAX_CLUSTER_SIZE are split to prevent chain drift.
///
/// # Arguments
/// * `memories` - All memories to cluster
/// * `embeddings` - Pairs of (memory_id, embedding_vector)
/// * `threshold` - Combined similarity threshold for linking (e.g. 0.55)
pub fn cluster_memories(
    memories: &[Memory],
    embeddings: &[(String, Vec<f32>)],
    threshold: f64,
) -> Vec<MemoryCluster> {
    // Build embedding lookup
    let emb_map: HashMap<&str, &Vec<f32>> = embeddings
        .iter()
        .map(|(id, emb)| (id.as_str(), emb))
        .collect();

    // Separate memories with/without embeddings
    let (with_emb, without_emb): (Vec<&Memory>, Vec<&Memory>) = memories
        .iter()
        .partition(|m| emb_map.contains_key(m.id.as_str()));

    // Compute pairwise combined similarities for memories with embeddings
    let mut sim_pairs: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..with_emb.len() {
        for j in (i + 1)..with_emb.len() {
            if let (Some(emb_a), Some(emb_b)) = (
                emb_map.get(with_emb[i].id.as_str()),
                emb_map.get(with_emb[j].id.as_str()),
            ) {
                let sim = combined_similarity(
                    emb_a, emb_b,
                    &with_emb[i].tags, &with_emb[j].tags,
                );
                if sim > threshold {
                    sim_pairs.push((i, j, sim));
                }
            }
        }
    }

    // Union-Find for single-linkage clustering
    let n = with_emb.len();
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while parent[r] != r {
            r = parent[r];
        }
        // Path compression
        let mut c = x;
        while parent[c] != r {
            let next = parent[c];
            parent[c] = r;
            c = next;
        }
        r
    }

    for &(i, j, _) in &sim_pairs {
        let ri = find(&mut parent, i);
        let rj = find(&mut parent, j);
        if ri != rj {
            parent[ri] = rj;
        }
    }

    // Group by root
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(i);
    }

    // Build clusters, splitting oversized ones to prevent chain drift
    let mut clusters: Vec<MemoryCluster> = Vec::new();

    for (_root, indices) in &groups {
        if indices.len() <= MAX_CLUSTER_SIZE {
            // Small enough — build directly
            let cluster_mems: Vec<Memory> = indices.iter().map(|&i| with_emb[i].clone()).collect();
            let index_set: HashSet<usize> = indices.iter().cloned().collect();
            let mut sims: Vec<(String, String, f64)> = sim_pairs
                .iter()
                .filter(|(i, j, _)| index_set.contains(i) && index_set.contains(j))
                .map(|(i, j, s)| (with_emb[*i].id.clone(), with_emb[*j].id.clone(), *s))
                .collect();
            sims.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            let label = generate_label(&cluster_mems);
            clusters.push(MemoryCluster {
                memories: cluster_mems,
                label,
                similarities: sims,
            });
        } else {
            // Oversized — split by picking two most distant members as seeds,
            // then assign each member to the nearest seed.
            let sub_clusters = split_oversized(indices, &with_emb, &emb_map, &sim_pairs);
            clusters.extend(sub_clusters);
        }
    }

    // Add unclustered group if any
    if !without_emb.is_empty() {
        clusters.push(MemoryCluster {
            memories: without_emb.into_iter().cloned().collect(),
            label: "unclustered (no embeddings)".to_string(),
            similarities: vec![],
        });
    }

    // Sort clusters: largest first, then by highest importance memory
    clusters.sort_by(|a, b| {
        b.memories
            .len()
            .cmp(&a.memories.len())
            .then_with(|| {
                let max_a = a
                    .memories
                    .iter()
                    .map(|m| (m.importance * 1000.0) as i64)
                    .max()
                    .unwrap_or(0);
                let max_b = b
                    .memories
                    .iter()
                    .map(|m| (m.importance * 1000.0) as i64)
                    .max()
                    .unwrap_or(0);
                max_b.cmp(&max_a)
            })
    });

    clusters
}

/// Split an oversized cluster by picking the two most distant members as seeds,
/// then assigning each member to the nearest seed (k=2 split). If a resulting
/// sub-cluster is still oversized, recurse.
fn split_oversized(
    indices: &[usize],
    with_emb: &[&Memory],
    emb_map: &HashMap<&str, &Vec<f32>>,
    sim_pairs: &[(usize, usize, f64)],
) -> Vec<MemoryCluster> {
    // Build a full similarity matrix for this group
    let n = indices.len();

    // Precompute pairwise similarities within this group
    let mut pair_sims: HashMap<(usize, usize), f64> = HashMap::new();
    let index_set: HashSet<usize> = indices.iter().cloned().collect();
    for &(i, j, s) in sim_pairs {
        if index_set.contains(&i) && index_set.contains(&j) {
            let (lo, hi) = if i < j { (i, j) } else { (j, i) };
            pair_sims.insert((lo, hi), s);
        }
    }

    // Also compute sims for pairs that weren't above threshold (needed for distance)
    for a in 0..n {
        for b in (a + 1)..n {
            let ia = indices[a];
            let ib = indices[b];
            let (lo, hi) = if ia < ib { (ia, ib) } else { (ib, ia) };
            if pair_sims.contains_key(&(lo, hi)) {
                continue;
            }
            if let (Some(emb_a), Some(emb_b)) = (
                emb_map.get(with_emb[ia].id.as_str()),
                emb_map.get(with_emb[ib].id.as_str()),
            ) {
                let sim = combined_similarity(
                    emb_a, emb_b,
                    &with_emb[ia].tags, &with_emb[ib].tags,
                );
                pair_sims.insert((lo, hi), sim);
            }
        }
    }

    // Find the two most distant members (lowest similarity)
    let mut min_sim = f64::MAX;
    let mut seed_a = 0usize;
    let mut seed_b = 1usize;
    for a in 0..n {
        for b in (a + 1)..n {
            let ia = indices[a];
            let ib = indices[b];
            let (lo, hi) = if ia < ib { (ia, ib) } else { (ib, ia) };
            let sim = pair_sims.get(&(lo, hi)).copied().unwrap_or(0.0);
            if sim < min_sim {
                min_sim = sim;
                seed_a = a;
                seed_b = b;
            }
        }
    }

    // Assign each member to the nearest seed
    let mut group_a: Vec<usize> = vec![indices[seed_a]];
    let mut group_b: Vec<usize> = vec![indices[seed_b]];

    for (pos, &idx) in indices.iter().enumerate() {
        if pos == seed_a || pos == seed_b {
            continue;
        }
        let ia = indices[seed_a];
        let ib = indices[seed_b];
        let (lo_a, hi_a) = if idx < ia { (idx, ia) } else { (ia, idx) };
        let (lo_b, hi_b) = if idx < ib { (idx, ib) } else { (ib, idx) };
        let sim_to_a = pair_sims.get(&(lo_a, hi_a)).copied().unwrap_or(0.0);
        let sim_to_b = pair_sims.get(&(lo_b, hi_b)).copied().unwrap_or(0.0);
        if sim_to_a >= sim_to_b {
            group_a.push(idx);
        } else {
            group_b.push(idx);
        }
    }

    // Recursively split if still oversized
    let mut result = Vec::new();
    for group in [group_a, group_b] {
        if group.len() > MAX_CLUSTER_SIZE {
            result.extend(split_oversized(&group, with_emb, emb_map, sim_pairs));
        } else {
            let cluster_mems: Vec<Memory> = group.iter().map(|&i| with_emb[i].clone()).collect();
            let grp_set: HashSet<usize> = group.iter().cloned().collect();
            let mut sims: Vec<(String, String, f64)> = sim_pairs
                .iter()
                .filter(|(i, j, _)| grp_set.contains(i) && grp_set.contains(j))
                .map(|(i, j, s)| (with_emb[*i].id.clone(), with_emb[*j].id.clone(), *s))
                .collect();
            sims.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            let label = generate_label(&cluster_mems);
            result.push(MemoryCluster {
                memories: cluster_mems,
                label,
                similarities: sims,
            });
        }
    }

    result
}

/// Generate a human-readable label for a cluster.
/// Uses the most common tag, or first few words of the highest-importance memory.
fn generate_label(memories: &[Memory]) -> String {
    // Count tags
    let mut tag_counts: HashMap<&str, usize> = HashMap::new();
    for m in memories {
        for t in &m.tags {
            // Skip generic tags
            if matches!(
                t.as_str(),
                "auto-extract" | "auto-distilled" | "distilled" | "session"
            ) {
                continue;
            }
            *tag_counts.entry(t.as_str()).or_default() += 1;
        }
    }

    // Find most common tag (that appears in >1 memory, or the only one)
    if let Some((&tag, &count)) = tag_counts.iter().max_by_key(|(_, &c)| c) {
        if count > 1 || memories.len() == 1 {
            return tag.to_string();
        }
    }

    // Fallback: first few words of highest-importance memory
    if let Some(top) = memories.iter().max_by(|a, b| {
        a.importance
            .partial_cmp(&b.importance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        let words: Vec<&str> = top.content.split_whitespace().take(6).collect();
        return format!("{}…", words.join(" "));
    }

    "unnamed".to_string()
}

/// Batch clusters into groups where each group's formatted prompt fits within
/// `max_chars`. Returns indices into the clusters slice for each batch.
pub fn batch_clusters(clusters: &[MemoryCluster], max_chars: usize) -> Vec<Vec<usize>> {
    let mut batches: Vec<Vec<usize>> = Vec::new();
    let mut current_batch: Vec<usize> = Vec::new();
    let mut current_size: usize = 0;

    for (i, cluster) in clusters.iter().enumerate() {
        let cluster_size = estimate_cluster_chars(cluster);

        // If a single cluster exceeds max_chars, it gets its own batch
        if cluster_size >= max_chars {
            if !current_batch.is_empty() {
                batches.push(std::mem::take(&mut current_batch));
                current_size = 0;
            }
            batches.push(vec![i]);
            continue;
        }

        if current_size + cluster_size > max_chars && !current_batch.is_empty() {
            batches.push(std::mem::take(&mut current_batch));
            current_size = 0;
        }

        current_batch.push(i);
        current_size += cluster_size;
    }

    if !current_batch.is_empty() {
        batches.push(current_batch);
    }

    // If no batches (empty input), return one empty batch
    if batches.is_empty() {
        batches.push(vec![]);
    }

    batches
}

/// Estimate the character count for a formatted cluster.
fn estimate_cluster_chars(cluster: &MemoryCluster) -> usize {
    let header = 60; // "## Cluster N: label (M memories)\n\n"
    let per_memory: usize = cluster
        .memories
        .iter()
        .map(|m| m.content.len() + 120) // metadata line + content
        .sum();
    let sims = cluster.similarities.len() * 40; // "  id1↔id2 = 0.XX\n"
    header + per_memory + sims
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{Layer, Memory};

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
                c.label, c.memories.len()
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
}
