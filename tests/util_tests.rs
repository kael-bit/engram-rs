use engram::util::{cosine_similarity, cosine_similarity_f32, l2_normalize, mean_vector, truncate_chars};

// ── truncate_chars ─────────────────────────────────────────────────────────

#[test]
fn empty_string() {
    assert_eq!(truncate_chars("", 5), "");
}

// ── cosine_similarity ──────────────────────────────────────────────────────

#[test]
fn cosine_empty() {
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
fn cosine_f32_matches() {
    let a = vec![0.5, 0.3, 0.8];
    let b = vec![0.1, 0.9, 0.4];
    let f64_sim = cosine_similarity(&a, &b);
    let f32_sim = cosine_similarity_f32(&a, &b);
    assert!((f64_sim as f32 - f32_sim).abs() < 1e-7);
}

// ── mean_vector ────────────────────────────────────────────────────────────

#[test]
fn mean_empty() {
    let m = mean_vector(&[]);
    assert!(m.is_empty());
}

// ── l2_normalize ───────────────────────────────────────────────────────────

#[test]
fn l2_normalize_zero() {
    let mut v = vec![0.0, 0.0, 0.0];
    l2_normalize(&mut v);
    assert!(v.iter().all(|&x| x == 0.0));
}
