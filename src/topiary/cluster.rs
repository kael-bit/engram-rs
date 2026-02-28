//! K-means clustering, split/merge passes, hierarchy building.

use super::{cosine_similarity, mean_vector, Entry, TopicNode, TopicTree};
use crate::thresholds;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand::{RngExt, SeedableRng};

impl TopicTree {
    /// Split oversized or low-coherence leaves.
    pub(super) fn split_pass(&mut self, all_entries: &[Entry]) -> usize {
        let mut splits = 0;
        let mut new_roots = Vec::new();

        for root in self.roots.drain(..) {
            let results = split_recursive(
                root,
                all_entries,
                self.max_leaf_size,
                self.min_internal_sim,
                &mut self.next_id,
                0,
            );
            for n in results {
                if n.children.is_empty() && n.members.is_empty() {
                    continue;
                }
                if !n.children.is_empty() {
                    splits += 1;
                }
                new_roots.push(n);
            }
        }

        self.roots = new_roots;
        splits
    }

    /// Merge similar leaf topics.
    pub(super) fn merge_pass(&mut self, all_entries: &[Entry]) -> usize {
        let mut leaf_ids: Vec<String> = Vec::new();
        let mut leaf_centroids: Vec<Vec<f32>> = Vec::new();
        let mut leaf_sizes: Vec<usize> = Vec::new();

        for root in &self.roots {
            for leaf in root.leaves() {
                leaf_ids.push(leaf.id.clone());
                leaf_centroids.push(leaf.centroid.clone());
                leaf_sizes.push(leaf.members.len());
            }
        }

        let mut merge_pairs: Vec<(String, String)> = Vec::new();
        let mut merged_set = std::collections::HashSet::new();

        for i in 0..leaf_ids.len() {
            if merged_set.contains(&leaf_ids[i]) {
                continue;
            }
            for j in (i + 1)..leaf_ids.len() {
                if merged_set.contains(&leaf_ids[j]) {
                    continue;
                }
                if leaf_sizes[i] + leaf_sizes[j] > self.max_leaf_size {
                    continue;
                }
                let sim = cosine_similarity(&leaf_centroids[i], &leaf_centroids[j]);
                if sim >= self.merge_threshold {
                    merge_pairs.push((leaf_ids[i].clone(), leaf_ids[j].clone()));
                    merged_set.insert(leaf_ids[i].clone());
                    merged_set.insert(leaf_ids[j].clone());
                    break;
                }
            }
        }

        let merge_count = merge_pairs.len();

        for (keep_id, remove_id) in &merge_pairs {
            let removed_members = self.collect_and_remove_leaf(remove_id);
            if let Some(keep_leaf) = super::find_leaf_mut(&mut self.roots, keep_id) {
                keep_leaf.members.extend(removed_members);
                let vecs: Vec<&[f32]> = keep_leaf
                    .members
                    .iter()
                    .map(|&idx| all_entries[idx].embedding.as_slice())
                    .collect();
                keep_leaf.centroid = mean_vector(&vecs);
            }
        }

        self.roots.retain(|n| n.total_members() > 0);
        merge_count
    }

    /// Top-down hierarchy building via k-means clustering of root-level topics.
    pub(super) fn build_hierarchy(&mut self) {
        let root_count = self.roots.len();
        if root_count < 4 {
            return;
        }

        let centroids: Vec<Vec<f32>> = self.roots.iter().map(|r| r.centroid.clone()).collect();
        let centroid_refs: Vec<&[f32]> = centroids.iter().map(|c| c.as_slice()).collect();

        let k = ((root_count as f64).sqrt().round() as usize).clamp(3, 12);

        let assignments = kmeans_vectors(&centroid_refs, k);

        let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (i, &cluster_id) in assignments.iter().enumerate() {
            clusters[cluster_id].push(i);
        }

        let old_roots: Vec<TopicNode> = self.roots.drain(..).collect();
        let mut slots: Vec<Option<TopicNode>> = old_roots.into_iter().map(Some).collect();

        for cluster in &clusters {
            if cluster.len() >= 2 {
                let mut children: Vec<TopicNode> = Vec::new();
                for &idx in cluster {
                    if let Some(node) = slots[idx].take() {
                        children.push(node);
                    }
                }
                let child_centroids: Vec<&[f32]> =
                    children.iter().map(|c| c.centroid.as_slice()).collect();
                let parent_centroid = mean_vector(&child_centroids);
                let parent_id = self.next_id();
                let mut parent = TopicNode {
                    id: parent_id,
                    name: None,
                    centroid: parent_centroid,
                    members: Vec::new(),
                    children,
                    dirty: false,
                    avg_sim: 0.0,
                    named_at_size: 0,
                };
                subdivide(&mut parent, &mut self.next_id, 1);
                self.roots.push(parent);
            } else if cluster.len() == 1 {
                if let Some(node) = slots[cluster[0]].take() {
                    self.roots.push(node);
                }
            }
        }

        for slot in slots {
            if let Some(node) = slot {
                self.roots.push(node);
            }
        }
    }
}

// ── Splitting ─────────────────────────────────────────────

fn split_recursive(
    mut node: TopicNode,
    all_entries: &[Entry],
    max_size: usize,
    _min_sim: f32,
    next_id: &mut u32,
    depth: usize,
) -> Vec<TopicNode> {
    if !node.is_leaf() {
        let old_children = std::mem::take(&mut node.children);
        for child in old_children {
            let results =
                split_recursive(child, all_entries, max_size, _min_sim, next_id, depth + 1);
            node.children.extend(results);
        }
        return vec![node];
    }

    if depth >= thresholds::TOPIARY_SPLIT_MAX_DEPTH {
        return vec![node];
    }

    let need_split = if node.members.len() <= 2 {
        false
    } else if node.members.len() > max_size {
        internal_similarity(&node, all_entries) < thresholds::TOPIARY_SPLIT_LARGE
    } else if node.members.len() >= 5 {
        internal_similarity(&node, all_entries) < thresholds::TOPIARY_SPLIT_MEDIUM
    } else {
        internal_similarity(&node, all_entries) < thresholds::TOPIARY_SPLIT_SMALL
    };

    if !need_split {
        return vec![node];
    }

    let k = if node.members.len() > 60 { 3 } else { 2 };
    let clusters = kmeans_split(&node.members, all_entries, k, next_id);

    if clusters.len() <= 1 {
        return vec![node];
    }

    let mut final_children = Vec::new();
    for child in clusters {
        let sub = split_recursive(child, all_entries, max_size, _min_sim, next_id, depth + 1);
        final_children.extend(sub);
    }

    node.children = final_children;
    node.members.clear();
    vec![node]
}

fn internal_similarity(node: &TopicNode, all_entries: &[Entry]) -> f32 {
    if node.members.len() < 2 {
        return 1.0;
    }
    let mut rng = StdRng::seed_from_u64(42);
    let sample_size = node.members.len().min(20);
    let sample: Vec<usize> = node
        .members
        .sample(&mut rng, sample_size)
        .copied()
        .collect();

    let mut total = 0.0f32;
    let mut count = 0u32;
    for i in 0..sample.len() {
        for j in (i + 1)..sample.len() {
            if let (Some(a), Some(b)) = (all_entries.get(sample[i]), all_entries.get(sample[j])) {
                total += cosine_similarity(&a.embedding, &b.embedding);
                count += 1;
            }
        }
    }
    if count == 0 {
        1.0
    } else {
        total / count as f32
    }
}

fn kmeans_split(
    members: &[usize],
    all_entries: &[Entry],
    k: usize,
    next_id: &mut u32,
) -> Vec<TopicNode> {
    if members.len() < k {
        return Vec::new();
    }

    let mut rng = StdRng::seed_from_u64(42);

    let mut centroid_indices: Vec<usize> = Vec::new();
    while centroid_indices.len() < k {
        let idx = members[rng.random_range(0..members.len())];
        if !centroid_indices.contains(&idx) {
            centroid_indices.push(idx);
        }
    }
    let mut centroids: Vec<Vec<f32>> = centroid_indices
        .iter()
        .filter_map(|&i| all_entries.get(i).map(|e| e.embedding.clone()))
        .collect();

    let mut assignments = vec![0usize; members.len()];

    for _iter in 0..20 {
        let mut changed = false;
        for (mi, &mem_idx) in members.iter().enumerate() {
            let emb = match all_entries.get(mem_idx) {
                Some(e) => &e.embedding,
                None => continue,
            };
            let mut best_k = 0;
            let mut best_sim = -1.0f32;
            for ki in 0..centroids.len() {
                let sim = cosine_similarity(emb, &centroids[ki]);
                if sim > best_sim {
                    best_sim = sim;
                    best_k = ki;
                }
            }
            if assignments[mi] != best_k {
                assignments[mi] = best_k;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        for ki in 0..centroids.len() {
            let vecs: Vec<&[f32]> = members
                .iter()
                .zip(assignments.iter())
                .filter(|(_, &a)| a == ki)
                .filter_map(|(&m, _)| all_entries.get(m).map(|e| e.embedding.as_slice()))
                .collect();
            if !vecs.is_empty() {
                centroids[ki] = mean_vector(&vecs);
            }
        }
    }

    let mut clusters = Vec::new();
    for ki in 0..k {
        let cluster_members: Vec<usize> = members
            .iter()
            .zip(assignments.iter())
            .filter(|(_, &a)| a == ki)
            .map(|(&m, _)| m)
            .collect();
        if cluster_members.is_empty() {
            continue;
        }
        *next_id += 1;
        let vecs: Vec<&[f32]> = cluster_members
            .iter()
            .filter_map(|&m| all_entries.get(m).map(|e| e.embedding.as_slice()))
            .collect();
        let centroid = mean_vector(&vecs);
        let avg_sim = if cluster_members.len() > 1 {
            let sum: f32 = cluster_members
                .iter()
                .filter_map(|&m| all_entries.get(m).map(|e| cosine_similarity(&e.embedding, &centroid)))
                .sum();
            sum / cluster_members.len() as f32
        } else {
            1.0
        };
        clusters.push(TopicNode {
            id: format!("t{}", next_id),
            name: None,
            centroid,
            members: cluster_members,
            children: Vec::new(),
            dirty: true,
            avg_sim,
            named_at_size: 0,
        });
    }

    clusters
}

// ── K-means on raw vectors ───────────────────────────────

pub(super) fn kmeans_vectors(vectors: &[&[f32]], k: usize) -> Vec<usize> {
    let n = vectors.len();
    if n <= k {
        return (0..n).collect();
    }

    let mut rng = StdRng::seed_from_u64(42);

    // k-means++ initialization
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    centroids.push(vectors[rng.random_range(0..n)].to_vec());

    for _ in 1..k {
        let dists: Vec<f32> = vectors
            .iter()
            .map(|v| {
                centroids
                    .iter()
                    .map(|c| 1.0 - cosine_similarity(v, c))
                    .fold(f32::MAX, f32::min)
            })
            .collect();
        let total: f32 = dists.iter().sum();
        if total <= 0.0 {
            centroids.push(vectors[rng.random_range(0..n)].to_vec());
            continue;
        }
        let threshold = rng.random::<f32>() * total;
        let mut cumsum = 0.0f32;
        let mut chosen = 0;
        for (i, &d) in dists.iter().enumerate() {
            cumsum += d;
            if cumsum >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.push(vectors[chosen].to_vec());
    }

    let mut assignments = vec![0usize; n];

    for _iter in 0..30 {
        let mut changed = false;
        for i in 0..n {
            let mut best_k = 0;
            let mut best_sim = -1.0f32;
            for ki in 0..k {
                let sim = cosine_similarity(vectors[i], &centroids[ki]);
                if sim > best_sim {
                    best_sim = sim;
                    best_k = ki;
                }
            }
            if assignments[i] != best_k {
                assignments[i] = best_k;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        for ki in 0..k {
            let vecs: Vec<&[f32]> = (0..n)
                .filter(|&i| assignments[i] == ki)
                .map(|i| vectors[i])
                .collect();
            if !vecs.is_empty() {
                centroids[ki] = mean_vector(&vecs);
            }
        }
    }

    assignments
}

// ── Hierarchy subdivision ─────────────────────────────────

fn subdivide(node: &mut TopicNode, next_id: &mut u32, depth: usize) {
    if depth >= thresholds::TOPIARY_HIERARCHY_MAX_DEPTH || node.children.len() <= thresholds::TOPIARY_HIERARCHY_MAX_CHILDREN {
        return;
    }

    let child_count = node.children.len();
    let k = ((child_count as f64).sqrt().round() as usize).clamp(2, 6);

    let centroids: Vec<Vec<f32>> = node.children.iter().map(|c| c.centroid.clone()).collect();
    let centroid_refs: Vec<&[f32]> = centroids.iter().map(|c| c.as_slice()).collect();

    let assignments = kmeans_vectors(&centroid_refs, k);

    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &cluster_id) in assignments.iter().enumerate() {
        clusters[cluster_id].push(i);
    }

    if clusters.iter().filter(|c| c.len() >= 2).count() == 0 {
        return;
    }

    let old_children: Vec<TopicNode> = node.children.drain(..).collect();
    let mut slots: Vec<Option<TopicNode>> = old_children.into_iter().map(Some).collect();

    for cluster in &clusters {
        if cluster.len() >= 2 {
            let mut sub_children: Vec<TopicNode> = Vec::new();
            for &idx in cluster {
                if let Some(child) = slots[idx].take() {
                    sub_children.push(child);
                }
            }
            let child_centroids: Vec<&[f32]> =
                sub_children.iter().map(|c| c.centroid.as_slice()).collect();
            *next_id += 1;
            let mut sub_parent = TopicNode {
                id: format!("t{}", next_id),
                name: None,
                centroid: mean_vector(&child_centroids),
                members: Vec::new(),
                children: sub_children,
                dirty: false,
                avg_sim: 0.0,
                named_at_size: 0,
            };
            subdivide(&mut sub_parent, next_id, depth + 1);
            node.children.push(sub_parent);
        } else if cluster.len() == 1 {
            if let Some(child) = slots[cluster[0]].take() {
                node.children.push(child);
            }
        }
    }

    for slot in slots {
        if let Some(child) = slot {
            node.children.push(child);
        }
    }
}

// ── Budget enforcement ────────────────────────────────────

/// When leaf count exceeds budget, incrementally merge the two most similar
/// leaves until count <= budget.  This preserves existing topic IDs and names
/// instead of destructively re-clustering via k-means.
pub(super) fn enforce_budget(
    leaves: &mut Vec<TopicNode>,
    all_entries: &[Entry],
    budget: usize,
) {
    if leaves.len() <= budget {
        return;
    }

    tracing::debug!(
        leaves = leaves.len(),
        budget,
        "topiary enforce_budget: incremental merge"
    );

    while leaves.len() > budget && leaves.len() >= 2 {
        // Find the two most similar leaves by centroid cosine similarity
        let mut best_sim = f32::NEG_INFINITY;
        let mut best_i = 0;
        let mut best_j = 1;

        for i in 0..leaves.len() {
            for j in (i + 1)..leaves.len() {
                let sim = cosine_similarity(&leaves[i].centroid, &leaves[j].centroid);
                if sim > best_sim {
                    best_sim = sim;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        // Merge j into i (remove j first since j > i)
        let removed = leaves.remove(best_j);
        let target = &mut leaves[best_i];

        // Combine members
        target.members.extend(removed.members);

        // Recalculate centroid from all member embeddings
        let vecs: Vec<&[f32]> = target.members.iter()
            .filter_map(|&m| all_entries.get(m).map(|e| e.embedding.as_slice()))
            .collect();
        if !vecs.is_empty() {
            target.centroid = mean_vector(&vecs);
        }

        // Recalculate avg_sim
        if target.members.len() > 1 {
            let sum: f32 = target.members.iter()
                .filter_map(|&m| all_entries.get(m).map(|e| cosine_similarity(&e.embedding, &target.centroid)))
                .sum();
            target.avg_sim = sum / target.members.len() as f32;
        } else {
            target.avg_sim = 1.0;
        }

        // Keep target name if it has one; only mark dirty if unnamed
        if target.name.is_none() {
            target.dirty = true;
        }

        tracing::trace!(
            merged_into = %target.id,
            new_size = target.members.len(),
            similarity = best_sim,
            "merged two most similar leaves"
        );
    }
}
