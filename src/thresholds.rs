//! Centralised thresholds, weights, and tuning constants.
//!
//! Every magic number that controls engram's behavior lives here so it can
//! be audited and tuned in one place.  The rest of the codebase imports from
//! `crate::thresholds`.
//!
//! ## Runtime-overridable thresholds
//!
//! Some scoring thresholds benefit from runtime tuning without recompilation.
//! These use [`std::sync::LazyLock`] to read an environment variable **once**
//! at first access and cache the result for the lifetime of the process.
//!
//! | Env var | Default | Description |
//! |---------|---------|-------------|
//! | `ENGRAM_IDF_BOOST_ALPHA` | 0.5 | IDF term-boost scaling factor in recall scoring |
//! | `ENGRAM_IDF_MISS_PENALTY` | 0.85 | Relevance penalty when no rare query terms match |
//! | `ENGRAM_ORPHAN_QUERY_PENALTY` | 0.5 | Penalty when all query terms have zero corpus frequency |
//! | `ENGRAM_TAG_BOOST` | 0.2 | Relevance boost per matching hint tag |
//! | `ENGRAM_NO_FTS_PENALTY` | 0.85 | Penalty for semantic-only hits lacking any query-term overlap |

use std::sync::LazyLock;

// ── Cosine similarity ──────────────────────────────────────────────────────
// Higher = stricter (only very similar items match).
// Hierarchy: proxy (loose) < insert (moderate) < triage (tight) < merge (tightest)

/// Proxy extraction: aggressive dedup to avoid flooding buffer
pub const PROXY_DEDUP_SIM: f64 = 0.60;

/// Insert path: prevent storing near-duplicates on manual POST
pub const INSERT_DEDUP_SIM: f64 = 0.65;

/// Triage + distill: skip promoting if a similar memory already exists
pub const TRIAGE_DEDUP_SIM: f64 = 0.75;

/// Merge: only auto-merge entries that are near-identical
pub const MERGE_SIM: f64 = 0.78;

/// Recall: quick dup check uses the same threshold as merge
pub const RECALL_DEDUP_SIM: f64 = MERGE_SIM;

/// Reconcile window: related but not duplicate (between these two bounds)
pub const RECONCILE_MIN_SIM: f64 = 0.55;
pub const RECONCILE_MAX_SIM: f64 = MERGE_SIM;

/// Core overlap detection during consolidation
pub const CORE_OVERLAP_SIM: f64 = 0.70;

/// Insert path: merge existing content when very similar
pub const INSERT_MERGE_SIM: f64 = 0.80;

/// Cosine similarity threshold for dedup in find_near_duplicate when
/// embeddings are available.  Higher than INSERT_DEDUP_SIM because this
/// is the last-resort DB-level check (API-level semantic dedup is first).
pub const DEDUP_COSINE_SIM: f64 = 0.85;

/// Jaccard pre-filter: only compute cosine for candidates above this.
pub const DEDUP_JACCARD_PREFILTER: f64 = 0.5;

// ── Resume ─────────────────────────────────────────────────────────────────

/// Compact mode relevance thresholds
pub const RESUME_HIGH_RELEVANCE: f64 = 0.25;
pub const RESUME_LOW_RELEVANCE: f64 = 0.10;
pub const RESUME_CORE_THRESHOLD: f64 = 0.35;
pub const RESUME_WORKING_THRESHOLD: f64 = 0.20;

/// Default recent_epochs for resume Recent section
pub const RESUME_DEFAULT_RECENT_EPOCHS: i64 = 24;

// ── Recall scoring ─────────────────────────────────────────────────────────

/// Default minimum cosine similarity to include a recall result.
pub const RECALL_SIM_FLOOR: f64 = 0.3;

/// Below this relevance, auto-expand the query via LLM.
pub const AUTO_EXPAND_THRESHOLD: f64 = 0.25;

// ── Runtime-overridable recall scoring knobs ───────────────────────────────
// These are read from environment variables **once** at first access via
// `LazyLock`.  If the env var is unset or unparseable, the default is used.

/// Helper: parse an env var into `f64`, falling back to `default`.
fn env_f64(var: &str, default: f64) -> f64 {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(default)
}

/// IDF term-boost scaling factor (`ENGRAM_IDF_BOOST_ALPHA`).
/// Controls how much rare query-term matches boost relevance.
pub static IDF_BOOST_ALPHA: LazyLock<f64> =
    LazyLock::new(|| env_f64("ENGRAM_IDF_BOOST_ALPHA", 0.5));

/// Relevance penalty when no rare query terms match a memory
/// (`ENGRAM_IDF_MISS_PENALTY`).  Applied multiplicatively to relevance.
pub static IDF_MISS_PENALTY: LazyLock<f64> =
    LazyLock::new(|| env_f64("ENGRAM_IDF_MISS_PENALTY", 0.85));

/// Heavy penalty when *all* query terms have zero document frequency
/// (`ENGRAM_ORPHAN_QUERY_PENALTY`).  Every result is likely a false positive.
pub static ORPHAN_QUERY_PENALTY: LazyLock<f64> =
    LazyLock::new(|| env_f64("ENGRAM_ORPHAN_QUERY_PENALTY", 0.5));

/// Per-tag relevance boost for memories matching caller-supplied hint tags
/// (`ENGRAM_TAG_BOOST`).  Multiplied by the number of matching tags.
pub static TAG_BOOST: LazyLock<f64> =
    LazyLock::new(|| env_f64("ENGRAM_TAG_BOOST", 0.2));

/// Relevance penalty for semantic-only hits that contain none of the query
/// terms (`ENGRAM_NO_FTS_PENALTY`).  Mitigates embedding false positives.
pub static NO_FTS_PENALTY: LazyLock<f64> =
    LazyLock::new(|| env_f64("ENGRAM_NO_FTS_PENALTY", 0.85));

// ── Memory weight / scoring ────────────────────────────────────────────────

/// Repetition bonus: scale × count, capped at cap.
pub const REP_BONUS_SCALE: f64 = 0.1;
pub const REP_BONUS_CAP: f64 = 0.5;

/// Access bonus: scale × ln(1 + count), capped at cap.
pub const ACCESS_BONUS_SCALE: f64 = 0.1;
pub const ACCESS_BONUS_CAP: f64 = 0.3;

/// Kind boost multipliers (semantic is always 1.0).
pub const KIND_BOOST_PROCEDURAL: f64 = 1.3;
pub const KIND_BOOST_EPISODIC: f64 = 0.8;

/// Layer boost multipliers (Working is always 1.0).
pub const LAYER_BOOST_CORE: f64 = 1.2;
pub const LAYER_BOOST_BUFFER: f64 = 0.8;

// ── Decay ──────────────────────────────────────────────────────────────────

/// Base decay amount per consolidation epoch (episodic rate).
/// Semantic decays at 60% of this, procedural at 20%.
pub const DECAY_BASE_AMOUNT: f64 = 0.005;
pub const DECAY_SEMANTIC_RATIO: f64 = 0.6;
pub const DECAY_PROCEDURAL_RATIO: f64 = 0.2;

/// Importance floor — memories never decay below this.
pub const DECAY_FLOOR: f64 = 0.0;

// ── Consolidation ──────────────────────────────────────────────────────────

/// Buffer capacity cap. Excess entries are evicted (lowest weight first).
pub const BUFFER_CAP_DEFAULT: usize = 200;

/// Score thresholds for Buffer→Working promotion.
pub const BUFFER_PROMOTE_SCORE: f64 = 0.4;

/// Score + importance thresholds for Working→Core gate candidacy.
pub const WORKING_GATE_SCORE: f64 = 0.8;

/// Gate cooldown: epochs before retrying a rejected memory.
pub const GATE_RETRY_EPOCHS: i64 = 48;
/// Gate cooldown: epochs before retrying a twice-rejected memory.
pub const GATE_RETRY_2_EPOCHS: i64 = 144;
/// Gate pending cooldown: skip for this many epochs after LLM failure.
pub const GATE_PENDING_COOLDOWN_EPOCHS: i64 = 4;

/// Lesson/procedural auto-promote cooldown in epochs.
pub const LESSON_PROMOTE_COOLDOWN_EPOCHS: i64 = 4;

/// Stale gate-rejection cleanup: remove tag after this many epochs.
pub const GATE_STALE_EPOCH_DIFF: i64 = 48;

// ── Clustering (consolidate/cluster.rs) ────────────────────────────────────

/// Weight for cosine similarity in the combined clustering score.
pub const CLUSTER_COSINE_WEIGHT: f64 = 0.7;
/// Weight for tag Jaccard similarity in the combined clustering score.
pub const CLUSTER_TAG_WEIGHT: f64 = 0.3;
/// Maximum memories per cluster. Prevents chain drift in single-linkage.
pub const MAX_CLUSTER_SIZE: usize = 10;

// ── Distillation (consolidate/audit.rs) ────────────────────────────────────

/// Minimum leaf members before distillation kicks in.
pub const DISTILL_THRESHOLD: usize = 10;
/// Maximum topics to distill per consolidation cycle.
pub const DISTILL_MAX_PER_CYCLE: usize = 2;

// ── Topiary ────────────────────────────────────────────────────────────────

/// Cosine threshold for assigning an entry to a topic cluster.
pub const TOPIARY_ASSIGN_THRESHOLD: f32 = 0.30;
/// Cosine threshold for merging two topic clusters.
pub const TOPIARY_MERGE_THRESHOLD: f32 = 0.55;
/// Minimum internal similarity for new topic nodes (default).
pub const TOPIARY_MIN_INTERNAL_SIM: f32 = 0.35;

/// Maximum leaf topics in the tree (pruning budget).
pub const TOPIARY_LEAF_BUDGET: usize = 256;

/// Absorb small children: cosine threshold and size limit.
pub const TOPIARY_ABSORB_THRESHOLD: f32 = 0.30;
pub const TOPIARY_ABSORB_SMALL_SIZE: usize = 3;

/// Tiered split thresholds: large (>max_size), medium (5-8), small (3-4).
pub const TOPIARY_SPLIT_LARGE: f32 = 0.50;
pub const TOPIARY_SPLIT_MEDIUM: f32 = 0.40;
pub const TOPIARY_SPLIT_SMALL: f32 = 0.35;

/// Split recursion depth limit in split_pass.
pub const TOPIARY_SPLIT_MAX_DEPTH: usize = 64;
/// Hierarchy subdivision: max depth and max children per node.
pub const TOPIARY_HIERARCHY_MAX_DEPTH: usize = 3;
pub const TOPIARY_HIERARCHY_MAX_CHILDREN: usize = 8;

/// Topiary worker debounce after trigger (milliseconds).
pub const TOPIARY_DEBOUNCE_MS: u64 = 5_000;

/// Naming batch size (topics per LLM call).
pub const TOPIARY_NAMING_BATCH_SIZE: usize = 30;

// ── Proxy ──────────────────────────────────────────────────────────────────

/// Minimum importance for proxy-extracted memories to be stored.
pub const PROXY_MIN_IMPORTANCE: f64 = 0.5;
/// Jaccard similarity threshold for proxy dedup.
pub const PROXY_DEDUP_THRESHOLD: f64 = 0.5;

/// Sliding window limits for proxy conversation buffering.
pub const PROXY_WINDOW_MAX_TURNS: usize = 8;
pub const PROXY_WINDOW_MAX_CHARS: usize = 16000;
/// Debounce: don't flush until this many seconds of quiet.
pub const PROXY_FLUSH_QUIET_SECS: i64 = 30;

// ── Embed queue (lib.rs) ───────────────────────────────────────────────────

/// Maximum embeddings per batch.
pub const EMBED_BATCH_SIZE: usize = 50;
/// Window between batches (ms).
pub const EMBED_WINDOW_MS: u64 = 500;

// ── DB limits (db/mod.rs) ──────────────────────────────────────────────────

/// Max content length before truncation.
pub const MAX_CONTENT_LEN: usize = 8192;
/// Max source field length.
pub const MAX_SOURCE_LEN: usize = 64;
/// Max number of tags per memory.
pub const MAX_TAGS: usize = 20;
/// Max length per tag.
pub const MAX_TAG_LEN: usize = 32;
/// Max fact field length.
pub const MAX_FACT_FIELD_LEN: usize = 512;

// ── HNSW vector index (db/vec.rs) ──────────────────────────────────────────

pub const HNSW_MAX_NB_CONN: usize = 16;
pub const HNSW_EF_CONSTRUCTION: usize = 200;
pub const HNSW_MAX_LAYER: usize = 16;
pub const HNSW_INITIAL_CAPACITY: usize = 10_000;
/// Overfetch factor for namespace-filtered search.
pub const NS_OVERFETCH: usize = 5;
/// Batch size for vector store rebuild.
pub const HNSW_BATCH_SIZE: usize = 500;
