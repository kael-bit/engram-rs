//! Topiary — hierarchical topic clustering for engram memories.
//!
//! Adapted from the standalone `topic-tree` project. Builds a tree of
//! topic clusters from memory embeddings, with LLM-powered naming.

mod cluster;
pub mod naming;
pub mod worker;

use serde::{Deserialize, Serialize};

// ── Entry: bridge between engram Memory and topiary's internal type ────────

/// Lightweight struct used internally by topiary. Constructed from engram
/// `db::Memory` by extracting content + embedding.
#[derive(Clone, Serialize, Deserialize)]
pub struct Entry {
    pub id: String,
    pub text: String,
    pub embedding: Vec<f32>,
}

// ── Self-contained f32 math (not importing from ai.rs which uses f64) ──────

/// Cosine similarity between two f32 vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        (dot / denom) as f32
    }
}

/// Element-wise mean of vectors, L2-normalized.
pub fn mean_vector(vectors: &[&[f32]]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    let dim = vectors[0].len();
    let mut result = vec![0.0f64; dim];
    for v in vectors {
        for (i, &val) in v.iter().enumerate() {
            result[i] += val as f64;
        }
    }
    let n = vectors.len() as f64;
    let mut out: Vec<f32> = result.iter().map(|x| (*x / n) as f32).collect();
    l2_normalize(&mut out);
    out
}

/// In-place L2 normalization.
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x = (*x as f64 * inv) as f32;
        }
    }
}

// ── TopicNode ──────────────────────────────────────────────────────────────

/// A node in the topic tree.
#[derive(Serialize, Deserialize)]
pub struct TopicNode {
    pub id: String,
    pub name: Option<String>,
    pub centroid: Vec<f32>,
    pub members: Vec<usize>, // indices into the global entries list
    pub children: Vec<TopicNode>,
    pub dirty: bool,
    pub avg_sim: f32,
}

impl TopicNode {
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub fn total_members(&self) -> usize {
        if self.is_leaf() {
            self.members.len()
        } else {
            self.children.iter().map(|c| c.total_members()).sum()
        }
    }

    pub fn depth(&self) -> usize {
        if self.is_leaf() {
            1
        } else {
            1 + self.children.iter().map(|c| c.depth()).max().unwrap_or(0)
        }
    }

    pub fn leaf_count(&self) -> usize {
        if self.is_leaf() {
            1
        } else {
            self.children.iter().map(|c| c.leaf_count()).sum()
        }
    }

    /// Collect all leaf nodes (flattened).
    pub fn leaves(&self) -> Vec<&TopicNode> {
        if self.is_leaf() {
            vec![self]
        } else {
            self.children.iter().flat_map(|c| c.leaves()).collect()
        }
    }
}

const LEAF_BUDGET: usize = 256;

// ── TopicTree ──────────────────────────────────────────────────────────────

/// The main topic tree.
#[derive(Serialize, Deserialize)]
pub struct TopicTree {
    pub roots: Vec<TopicNode>,
    assign_threshold: f32,
    merge_threshold: f32,
    max_leaf_size: usize,
    min_internal_sim: f32,
    next_id: u32,
}

impl TopicTree {
    pub fn new(assign_threshold: f32, merge_threshold: f32) -> Self {
        Self {
            roots: Vec::new(),
            assign_threshold,
            merge_threshold,
            max_leaf_size: 8,
            min_internal_sim: 0.35,
            next_id: 0,
        }
    }

    fn next_id(&mut self) -> String {
        self.next_id += 1;
        format!("t{}", self.next_id)
    }

    /// Insert a single entry into the tree.
    pub fn insert(&mut self, mem_idx: usize, embedding: &[f32]) {
        let mut best_sim = -1.0f32;
        let mut best_leaf_id: Option<String> = None;

        for root in &self.roots {
            for leaf in root.leaves() {
                let sim = cosine_similarity(embedding, &leaf.centroid);
                if sim > best_sim {
                    best_sim = sim;
                    best_leaf_id = Some(leaf.id.clone());
                }
            }
        }

        let accept = if best_sim >= self.assign_threshold {
            if let Some(ref leaf_id) = best_leaf_id {
                if let Some(leaf) = find_leaf(&self.roots, leaf_id) {
                    if leaf.members.len() < 3 {
                        true
                    } else {
                        best_sim >= leaf.avg_sim * 0.8
                    }
                } else {
                    true
                }
            } else {
                false
            }
        } else {
            false
        };

        if accept {
            let leaf_id = best_leaf_id.unwrap();
            if let Some(leaf) = find_leaf_mut(&mut self.roots, &leaf_id) {
                let prev_size = leaf.members.len();
                leaf.members.push(mem_idx);
                if leaf.name.is_none() || prev_size < 5 {
                    leaf.dirty = true;
                }
                let n = leaf.members.len() as f32;
                for (i, val) in embedding.iter().enumerate() {
                    if i < leaf.centroid.len() {
                        leaf.centroid[i] = leaf.centroid[i] * ((n - 1.0) / n) + val / n;
                    }
                }
                l2_normalize(&mut leaf.centroid);
                leaf.avg_sim = leaf.avg_sim * ((n - 1.0) / n) + best_sim / n;
            }
        } else {
            let id = self.next_id();
            let node = TopicNode {
                id,
                name: None,
                centroid: embedding.to_vec(),
                members: vec![mem_idx],
                children: Vec::new(),
                dirty: true,
                avg_sim: 1.0,
            };
            self.roots.push(node);
        }
    }

    /// Run consolidation cycles until stable, then enforce budget and build hierarchy.
    pub fn consolidate(&mut self, all_entries: &[Entry]) {
        let mut prev_leaves = 0usize;
        for cycle in 0..10 {
            let splits = self.split_pass(all_entries);
            let merges = self.merge_pass(all_entries);
            let cur_leaves: usize = self.roots.iter().map(|r| r.leaf_count()).sum();
            tracing::debug!(
                cycle, splits, merges, topics = cur_leaves,
                "topiary consolidation cycle"
            );
            if (splits == 0 && merges == 0) || cur_leaves == prev_leaves {
                tracing::debug!(cycles = cycle + 1, "topiary stable");
                break;
            }
            prev_leaves = cur_leaves;
        }

        // Enforce leaf budget
        {
            let flat_leaves = collect_all_leaves(std::mem::take(&mut self.roots));
            let mut leaves = flat_leaves;
            cluster::enforce_budget(&mut leaves, all_entries, LEAF_BUDGET);
            self.roots = leaves;
        }

        // Build hierarchy
        self.build_hierarchy();
        {
            let cur_roots = self.roots.len();
            let max_d: usize = self.roots.iter().map(|r| r.depth()).max().unwrap_or(0);
            tracing::debug!(roots = cur_roots, depth = max_d, "topiary hierarchy built");
        }

        // Absorb small leaves into nearest sibling
        let absorbed = self.absorb_small_leaves(all_entries);
        if absorbed > 0 {
            tracing::debug!(absorbed, "topiary absorbed small leaves");
        }

        // Flatten single-child chains
        let mut changed = true;
        while changed {
            changed = false;
            for root in &mut self.roots {
                if prune_single_child(root) {
                    changed = true;
                }
            }
            let mut new_roots = Vec::new();
            for root in self.roots.drain(..) {
                flatten_into(&mut new_roots, root);
            }
            self.roots = new_roots;
        }

        // Clean up empty nodes
        self.roots.retain(|n| n.total_members() > 0);
        for root in &mut self.roots {
            prune_empty_children(root);
        }

        self.reassign_leaf_ids();
    }

    /// Absorb small leaf topics (1-2 members) into their nearest neighbor leaf.
    fn absorb_small_leaves(&mut self, all_entries: &[Entry]) -> usize {
        let mut total = 0;
        for root in &mut self.roots {
            total += absorb_small_in_node(root, all_entries, self.max_leaf_size);
        }
        total
    }

    pub(crate) fn collect_and_remove_leaf(&mut self, id: &str) -> Vec<usize> {
        for root in &mut self.roots {
            if root.id == id && root.is_leaf() {
                return std::mem::take(&mut root.members);
            }
            if let Some(m) = remove_leaf_from_children(root, id) {
                return m;
            }
        }
        Vec::new()
    }

    /// Reassign all leaf IDs to sequential kb1, kb2, ...
    fn reassign_leaf_ids(&mut self) {
        let mut counter = 1u32;
        for root in &mut self.roots {
            reassign_ids(root, &mut counter);
        }
    }

    /// Generate resume-format summary of topics.
    pub fn generate_resume(&self, all_entries: &[Entry]) -> (usize, usize, String) {
        let total_entries: usize = self.roots.iter().map(|r| r.total_members()).sum();
        let total_topics: usize = self.roots.iter().map(|r| r.leaf_count()).sum();

        let mut leaves: Vec<&TopicNode> = Vec::new();
        for root in &self.roots {
            leaves.extend(root.leaves());
        }

        // Sort leaves by member count descending
        leaves.sort_by(|a, b| b.members.len().cmp(&a.members.len()));

        let mut out = String::new();
        for leaf in &leaves {
            let name = leaf.name.as_deref().unwrap_or("unnamed");
            let count = leaf.members.len();
            let content_chars: usize = leaf
                .members
                .iter()
                .map(|&i| all_entries.get(i).map_or(0, |e| e.text.len()))
                .sum();
            let content_tokens = (content_chars as f64 / 3.5).ceil() as usize;
            out.push_str(&format!(
                "{}: \"{}\" [{}][~{} tok]\n",
                leaf.id, name, count, content_tokens
            ));
        }

        (total_entries, total_topics, out)
    }

    /// Serialize tree to JSON for storage.
    pub fn to_json(&self, all_entries: &[Entry]) -> serde_json::Value {
        let topics: Vec<serde_json::Value> = self
            .roots
            .iter()
            .map(|root| node_to_json(root, all_entries))
            .collect();

        let total_leaves: usize = self.roots.iter().map(|r| r.leaf_count()).sum();
        serde_json::json!({
            "total_entries": all_entries.len(),
            "total_topics": total_leaves,
            "topics": topics,
        })
    }
}

// ── Helper functions ──────────────────────────────────────────────────────

fn reassign_ids(node: &mut TopicNode, counter: &mut u32) {
    if node.is_leaf() {
        node.id = format!("kb{}", counter);
        *counter += 1;
        if node.name.is_none() {
            node.dirty = true;
        }
    } else {
        for child in &mut node.children {
            reassign_ids(child, counter);
        }
    }
}

pub fn find_leaf<'a>(roots: &'a [TopicNode], id: &str) -> Option<&'a TopicNode> {
    for root in roots {
        if let Some(leaf) = find_in_node(root, id) {
            return Some(leaf);
        }
    }
    None
}

pub fn find_leaf_mut<'a>(roots: &'a mut [TopicNode], id: &str) -> Option<&'a mut TopicNode> {
    for root in roots {
        if let Some(leaf) = find_in_node_mut(root, id) {
            return Some(leaf);
        }
    }
    None
}

fn find_in_node<'a>(node: &'a TopicNode, id: &str) -> Option<&'a TopicNode> {
    if node.id == id && node.is_leaf() {
        return Some(node);
    }
    for child in &node.children {
        if let Some(found) = find_in_node(child, id) {
            return Some(found);
        }
    }
    None
}

fn find_in_node_mut<'a>(node: &'a mut TopicNode, id: &str) -> Option<&'a mut TopicNode> {
    if node.id == id && node.is_leaf() {
        return Some(node);
    }
    for child in &mut node.children {
        if let Some(found) = find_in_node_mut(child, id) {
            return Some(found);
        }
    }
    None
}

fn remove_leaf_from_children(node: &mut TopicNode, id: &str) -> Option<Vec<usize>> {
    let mut found_idx = None;
    for (i, child) in node.children.iter().enumerate() {
        if child.id == id && child.is_leaf() {
            found_idx = Some(i);
            break;
        }
    }
    if let Some(idx) = found_idx {
        let removed = node.children.remove(idx);
        return Some(removed.members);
    }
    for child in &mut node.children {
        if let Some(m) = remove_leaf_from_children(child, id) {
            return Some(m);
        }
    }
    None
}

fn prune_empty_children(node: &mut TopicNode) {
    node.children.retain(|c| c.total_members() > 0);
    for child in &mut node.children {
        prune_empty_children(child);
    }
}

fn absorb_small_in_node(
    node: &mut TopicNode,
    all_entries: &[Entry],
    max_leaf_size: usize,
) -> usize {
    const ABSORB_THRESHOLD: f32 = 0.40;
    const SMALL_SIZE: usize = 2;

    let mut absorbed = 0;
    for child in &mut node.children {
        if !child.is_leaf() {
            absorbed += absorb_small_in_node(child, all_entries, max_leaf_size);
        }
    }

    if node.children.len() < 2 {
        return absorbed;
    }

    loop {
        let mut best: Option<(usize, usize, f32)> = None;

        for i in 0..node.children.len() {
            if !node.children[i].is_leaf() || node.children[i].members.len() > SMALL_SIZE {
                continue;
            }
            for j in 0..node.children.len() {
                if i == j || !node.children[j].is_leaf() {
                    continue;
                }
                if node.children[j].members.len() + node.children[i].members.len() > max_leaf_size
                {
                    continue;
                }
                let sim =
                    cosine_similarity(&node.children[i].centroid, &node.children[j].centroid);
                if sim >= ABSORB_THRESHOLD {
                    if best.is_none() || sim > best.unwrap().2 {
                        best = Some((i, j, sim));
                    }
                }
            }
        }

        if let Some((small_idx, target_idx, _)) = best {
            let small_members = std::mem::take(&mut node.children[small_idx].members);
            node.children[target_idx].members.extend(small_members);
            node.children[target_idx].dirty = true;
            let vecs: Vec<&[f32]> = node.children[target_idx]
                .members
                .iter()
                .map(|&idx| all_entries[idx].embedding.as_slice())
                .collect();
            node.children[target_idx].centroid = mean_vector(&vecs);
            node.children.remove(small_idx);
            absorbed += 1;
        } else {
            break;
        }
    }

    absorbed
}

fn flatten_into(out: &mut Vec<TopicNode>, node: TopicNode) {
    if node.children.is_empty() {
        out.push(node);
    } else if node.children.len() == 1 {
        for child in node.children {
            flatten_into(out, child);
        }
    } else {
        out.push(node);
    }
}

fn prune_single_child(node: &mut TopicNode) -> bool {
    let mut changed = false;
    for child in &mut node.children {
        if prune_single_child(child) {
            changed = true;
        }
    }
    while node.children.len() == 1 {
        let child = node.children.remove(0);
        node.members = child.members;
        node.children = child.children;
        if node.name.is_none() {
            node.name = child.name;
        }
        changed = true;
    }
    changed
}

fn collect_all_leaves(roots: Vec<TopicNode>) -> Vec<TopicNode> {
    let mut leaves = Vec::new();
    for root in roots {
        collect_leaves(root, &mut leaves);
    }
    leaves
}

fn collect_leaves(node: TopicNode, out: &mut Vec<TopicNode>) {
    if node.is_leaf() {
        if !node.members.is_empty() {
            out.push(node);
        }
    } else {
        for child in node.children {
            collect_leaves(child, out);
        }
    }
}

fn node_to_json(node: &TopicNode, all_entries: &[Entry]) -> serde_json::Value {
    if node.is_leaf() {
        let samples: Vec<&str> = node
            .members
            .iter()
            .filter_map(|&i| all_entries.get(i).map(|e| e.text.as_str()))
            .collect();
        serde_json::json!({
            "id": node.id,
            "name": node.name,
            "member_count": node.members.len(),
            "samples": samples,
        })
    } else {
        let children: Vec<serde_json::Value> = node
            .children
            .iter()
            .map(|c| node_to_json(c, all_entries))
            .collect();
        serde_json::json!({
            "id": node.id,
            "name": node.name,
            "member_count": node.total_members(),
            "children": children,
        })
    }
}
