// ---------------------------------------------------------------------------
// Vector utilities (single source of truth for cosine/mean/normalize)
// ---------------------------------------------------------------------------

/// Cosine similarity between two f32 vectors (returns f64).
/// Internal computation uses f64 for precision.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        let (ai, bi) = (a[i] as f64, b[i] as f64);
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

/// Cosine similarity returning f32 (convenience for topiary/clustering).
#[inline]
pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    cosine_similarity(a, b) as f32
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

// ---------------------------------------------------------------------------
// String utilities
// ---------------------------------------------------------------------------

/// Safe ID prefix — never panics on non-ASCII or short strings.
#[inline]
pub fn short_id(id: &str) -> &str {
    let end = id.floor_char_boundary(8.min(id.len()));
    &id[..end]
}

/// Truncate a string to `max` characters, appending "…" if truncated.
/// Handles multi-byte (CJK) correctly via char boundary.
pub fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max).collect();
        format!("{truncated}…")
    }
}

