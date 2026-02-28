use engram::util::{cosine_similarity, cosine_similarity_f32, l2_normalize, mean_vector, truncate_chars};

// ── truncate_chars ─────────────────────────────────────────────────────────

#[test]
fn ascii_no_truncate() {
    assert_eq!(truncate_chars("hello", 10), "hello");
}

#[test]
fn ascii_truncate() {
    assert_eq!(truncate_chars("hello world", 5), "hello…");
}

#[test]
fn cjk_truncate() {
    assert_eq!(truncate_chars("你好世界测试", 4), "你好世界…");
}

#[test]
fn empty_string() {
    assert_eq!(truncate_chars("", 5), "");
}

// ── cosine_similarity ──────────────────────────────────────────────────────

#[test]
fn cosine_identical() {
    let v = vec![1.0, 2.0, 3.0];
    let sim = cosine_similarity(&v, &v);
    assert!((sim - 1.0).abs() < 1e-6);
}

#[test]
fn cosine_orthogonal() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!(sim.abs() < 1e-6);
}

#[test]
fn cosine_opposite() {
    let a = vec![1.0, 0.0];
    let b = vec![-1.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!((sim + 1.0).abs() < 1e-6);
}

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
fn mean_single_vector() {
    let v = vec![3.0, 4.0];
    let m = mean_vector(&[&v]);
    // should be L2-normalized: [0.6, 0.8]
    assert!((m[0] - 0.6).abs() < 1e-5);
    assert!((m[1] - 0.8).abs() < 1e-5);
}

#[test]
fn mean_empty() {
    let m = mean_vector(&[]);
    assert!(m.is_empty());
}

// ── l2_normalize ───────────────────────────────────────────────────────────

#[test]
fn l2_normalize_basic() {
    let mut v = vec![3.0, 4.0];
    l2_normalize(&mut v);
    assert!((v[0] - 0.6).abs() < 1e-5);
    assert!((v[1] - 0.8).abs() < 1e-5);
}

#[test]
fn l2_normalize_zero() {
    let mut v = vec![0.0, 0.0, 0.0];
    l2_normalize(&mut v);
    assert!(v.iter().all(|&x| x == 0.0));
}
