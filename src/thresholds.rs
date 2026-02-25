/// Cosine similarity thresholds for deduplication across components.
///
/// Higher = stricter (only very similar items match).
/// The hierarchy: proxy (loose) < insert (moderate) < distill/triage (tight) < merge (tightest)

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

/// Audit: similarity range for suggesting merges to the LLM auditor
pub const AUDIT_MERGE_MIN_SIM: f64 = 0.65;
pub const AUDIT_MERGE_MAX_SIM: f64 = MERGE_SIM;

/// Core overlap detection during consolidation
pub const CORE_OVERLAP_SIM: f64 = 0.70;

/// Insert path: merge existing content when very similar
pub const INSERT_MERGE_SIM: f64 = 0.80;

/// Cosine similarity threshold for dedup in find_near_duplicate when
/// embeddings are available.  Higher than INSERT_DEDUP_SIM because this
/// is the last-resort DB-level check (API-level semantic dedup is first).
pub const DEDUP_COSINE_SIM: f64 = 0.85;

/// Jaccard pre-filter: only compute cosine for candidates above this.
/// Low enough to let through semantically similar but textually different
/// content, while still filtering obvious non-matches.
pub const DEDUP_JACCARD_PREFILTER: f64 = 0.5;

/// Sandbox audit: minimum safety score to auto-apply operations
pub const SANDBOX_SAFETY_THRESHOLD: f64 = 0.70;

/// Sandbox: memories modified within this many hours are "recently modified"
pub const SANDBOX_RECENT_MOD_HOURS: f64 = 24.0;

/// Sandbox: memories younger than this are "new" (protected from deletion)
pub const SANDBOX_NEW_AGE_HOURS: f64 = 48.0;

/// Resume: relevance thresholds for compact mode filtering
pub const RESUME_HIGH_RELEVANCE: f64 = 0.25;
pub const RESUME_LOW_RELEVANCE: f64 = 0.10;
pub const RESUME_CORE_THRESHOLD: f64 = 0.35;

/// Buffer TTL rescue: minimum importance to keep expired buffer memories
pub const BUFFER_RESCUE_IMPORTANCE: f64 = 0.70;
