//! Unit tests for `is_code_noise` — the proxy extraction noise filter.
//!
//! The function uses a two-tier signal system:
//! - **Strong signals** (e.g. `pub fn`, `cargo test`, `mod.rs`): a single
//!   hit without a value marker is enough to classify as noise.
//! - **Weak signals** (e.g. `compile`, `refactor`, `rename`): need 2+ hits
//!   without a value marker.
//! - **Value markers** (e.g. `lesson`, `decision`, `never`, `always`):
//!   override all signals — content with a value marker is never noise.

use engram::proxy::extract::is_code_noise;

// ── Strong signals → noise with 1 hit ─────────────────────────────────────

#[test]
fn single_strong_signal_cargo_build_is_noise() {
    assert!(is_code_noise("ran cargo build successfully"));
}

#[test]
fn single_strong_signal_pub_fn_is_noise() {
    assert!(is_code_noise("added a pub fn to the module"));
}

#[test]
fn single_strong_signal_mod_rs_is_noise() {
    assert!(is_code_noise("updated mod.rs to re-export types"));
}

#[test]
fn single_strong_signal_dead_code_is_noise() {
    assert!(is_code_noise("removed dead code from the file"));
}

#[test]
fn single_strong_signal_clippy_is_noise() {
    assert!(is_code_noise("fixed all clippy warnings"));
}

#[test]
fn multiple_strong_signals_is_noise() {
    // "impl " + "struct " → 2 strong signals
    assert!(is_code_noise("added impl block for struct Foo"));
}

#[test]
fn strong_plus_weak_is_noise() {
    // "mod.rs" (strong) + "refactor" (weak)
    assert!(is_code_noise("refactor: moved helper out of mod.rs"));
}

// ── Weak signals → need 2+ hits ───────────────────────────────────────────

#[test]
fn single_weak_signal_is_not_noise() {
    // "compilation" alone is just 1 weak signal → not enough
    assert!(!is_code_noise("fixed the compilation issue in the module"));
}

#[test]
fn two_weak_signals_is_noise() {
    // "refactor" + "rename" → 2 weak signals → noise
    assert!(is_code_noise("refactor: rename function to new name"));
}

#[test]
fn two_weak_signals_compile_and_test_pass() {
    // "compilation" + "test pass" → 2 weak signals
    assert!(is_code_noise("compilation succeeded and all test pass"));
}

#[test]
fn codebase_and_implementation_is_noise() {
    // "codebase" + "implementation" → 2 weak signals
    assert!(is_code_noise("reviewed the codebase and implementation details"));
}

// ── Value markers override signals ────────────────────────────────────────

#[test]
fn lesson_overrides_strong_signal() {
    assert!(!is_code_noise("LESSON: always run cargo check before committing"));
}

#[test]
fn decision_overrides_weak_signals() {
    assert!(!is_code_noise("decision: refactor the auth module to use OAuth2"));
}

#[test]
fn never_overrides_strong_signal() {
    assert!(!is_code_noise("never force-push after cargo test passes"));
}

#[test]
fn always_overrides_strong_signal() {
    assert!(!is_code_noise("always run cargo build in release mode for benchmarks"));
}

#[test]
fn chinese_prohibition_overrides() {
    assert!(!is_code_noise("禁止直接修改 implementation 细节"));
}

#[test]
fn chinese_must_overrides() {
    assert!(!is_code_noise("必须先 cargo check 再部署"));
}

#[test]
fn dont_overrides() {
    assert!(!is_code_noise("don't use merge conflict resolution without review"));
}

// ── No signals at all → not noise ─────────────────────────────────────────

#[test]
fn plain_knowledge_is_not_noise() {
    assert!(!is_code_noise("User prefers dark mode and compact layouts"));
}

#[test]
fn empty_string_is_not_noise() {
    assert!(!is_code_noise(""));
}

#[test]
fn no_signals_at_all_is_not_noise() {
    assert!(!is_code_noise("the weather is nice today"));
}

#[test]
fn preference_is_not_noise() {
    assert!(!is_code_noise("sky prefers concise replies over verbose explanations"));
}
