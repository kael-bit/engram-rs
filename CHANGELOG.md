# Changelog

## 0.16.0

### Recall Overhaul

- **IDF-weighted term scoring**: Query terms are weighted by corpus rarity (`ln(N/df)`). Rare terms boost relevance more than common ones — works for both keywords and natural language queries.
- **Multiplicative scoring formula**: `score = relevance × (1 + 0.4×weight + 0.2×recency)`. Relevance is now the gate — a Core memory with zero relevance scores zero, not 0.50.
- **Dynamic noise filtering**: Replaced 100+ hardcoded stopwords (English + Chinese) with `df/N > 0.8` auto-detection. Language-agnostic, zero maintenance. Small corpus guard skips filtering when `total_docs < 5`.
- **IDF miss penalty**: Memories that miss all rare query terms get `relevance × 0.85`.
- **Orphan query penalty**: When no query terms exist in the corpus at all, all candidates get `relevance × 0.5`.
- **Tags as soft boost**: Tag parameters no longer hard-filter results. Each matching tag multiplies relevance by 1.2 — more results, better ranked.
- **min_score default 0.30**: Applied at API layer to filter low-quality noise. Internal `recall()` defaults to 0.0 for testability.

### Topiary Clustering

- **Tag-aware merge/enforce**: Cluster merge and budget enforcement use combined similarity (`cosine × 0.7 + tag_jaccard × 0.3`). Prevents grouping semantically-near but conceptually-different memories (e.g. "GitHub credentials" vs "GitHub promotion").
- **Absorb stays pure cosine**: Fragment absorption ignores tag differences — small orphans should consolidate regardless of tagging.
- **Absorb threshold lowered**: `TOPIARY_ABSORB_THRESHOLD` 0.40 → 0.30 for easier fragment cleanup.

### Prompt Quality

- **Unified tag guidance**: All 7 LLM entry points (extract/triage/gate/proxy/distill/merge/naming) now include "name the SUBJECT, not meta-properties" with good/bad examples.
- **Strict kind definitions**: `procedural` restricted to reusable step-by-step workflows ONLY. Strategies, plans, and one-time decisions default to `semantic`.
- **SETUP.md templates updated**: HTTP and MCP agent templates include the same tag/kind guidance.

### Tests

- **Removed 38 trivial tests** (-594 lines, 3 files deleted): Pure formula recalculation, obvious status code mappings, textbook math. Remaining 239 tests all provide real protection value.

## 0.15.0

### New Features

- **Persistent embed cache**: Query embeddings are cached in SQLite (128-entry FIFO). Survives restarts — no cold-start API calls for repeated queries.
- **API request logging**: Every protected endpoint logs method, path, namespace, and elapsed time via tracing. Enables observability and debugging of agent access patterns.
- **Multi-agent concurrent access**: Documented and verified. SQLite WAL + connection pool + RwLock vector index supports multiple agents hitting one instance safely.

### Performance

- **Naming LLM call reduction**: Topics track `named_at_size` — only re-name when unnamed or members grew 50%+ since last naming. Reduced naming calls from ~19/day to near-zero on stable trees.
- **bytemuck zero-copy**: Embedding f32↔bytes conversion uses `bytemuck::cast_slice` instead of manual `to_le_bytes` loops.
- **backon structured retry**: HTTP retry logic uses `backon::ExponentialBuilder` instead of hand-rolled sleep loops.

### Refactoring

- **Unified vector utils**: `cosine_similarity`, `mean_vector`, `l2_normalize` consolidated into `src/util.rs` as single source of truth. `ai.rs` and `topiary/mod.rs` re-export instead of duplicating (~50 lines removed).
- **Consistent imports**: All `cosine_similarity` consumers import from `crate::util` instead of scattered `crate::ai` references.
- **Safe string slicing**: Manual `id[..8]` replaced with `util::short_id()` (CJK-safe, never panics).

### Tests

- **Vector util tests**: 11 new tests covering cosine similarity (identical, orthogonal, opposite, empty, mismatched lengths, f32/f64 consistency), mean_vector, and l2_normalize.
- **Scoring tests**: 136-line `scoring_tests.rs` covering `memory_weight()` formula across all kind/layer combinations.

## 0.14.0

### Breaking Changes

- **Rerank removed**: `rerank` field removed from `RecallRequest`. LLM re-ranking was default-off and unnecessary — FTS + semantic scoring is sufficient. Reduces recall latency.
- **Audit redesigned**: Old three-powers global audit replaced with **topic distillation** — condenses bloated topics (≥10 members) into fewer, richer entries. More focused, less wasteful.
- **Sandbox removed**: The audit sandbox (`RuleChecker`) is no longer needed with the simpler distillation model.

### New Features

- **Episodic Core block**: Episodic memories can never be promoted to Core. Enforced at both candidate collection and DB `promote()` level.
- **Web UI split**: Single 1300-line `index.html` split into `index.html` + `style.css` + `app.js`, served via `/ui`, `/ui/style.css`, `/ui/app.js`.
- **Sort by weight**: Web UI memory list supports 4 sort modes — Newest, Oldest, Weight ↓, Weight ↑. Weight uses the same `memory_weight()` formula as recall scoring.

### Improvements

- **Centralized prompts**: All LLM prompts and JSON tool schemas moved to `src/prompts.rs`.
- **Centralized thresholds**: All 60+ magic numbers from 15+ files moved to `src/thresholds.rs`.
- **Epoch-based decay**: Decay only runs during active consolidation cycles, not by wall-clock time. Idle agents preserve memories intact.
- **Buffer redesign**: All new memories enter Buffer. Promotion requires passing through consolidation.
- **Gate whitelist**: LLM gate uses batch evaluation for Working→Core promotion decisions.
- **Reconcile merge**: Detects same-topic memories where newer content supersedes older, merging them automatically.
- **Net code reduction**: ~1400 lines removed across refactoring and feature removal.

## 0.13.0

### Breaking Changes

- **Clean API responses**: `/recall` and `/triggers` now return `MemoryResult` — only `id` (8-char), `content`, `score`, `layer`, `tags`, `kind`. Internal fields (`decay_rate`, `access_count`, `repetition_count`, `last_accessed`, `modified_at`, `source`, `namespace`) removed from API output.
- **Working memories are permanent**: Working layer memories are never deleted or demoted to Buffer. Importance decays to 0 but the memory persists — findable via recall and topic tree.

### Unified Scoring

- New `memory_weight()` in `src/scoring.rs` — single function used across all ranking contexts.
- Formula: `(importance + rep_bonus + access_bonus) × kind_boost × layer_boost`
  - `rep_bonus`: `min(repetition × 0.1, 0.5)` — permanent reinforcement signal
  - `access_bonus`: `min(ln(1 + access_count) × 0.1, 0.3)` — diminishing returns
  - `kind_boost`: procedural 1.3×, others 1.0×
  - `layer_boost`: Core 1.2×, Working 1.0×, Buffer 0.8×
- Additive design: importance=0 memories with high repetition still rank (not zeroed out).
- Recall scoring: `0.5 × relevance + 0.3 × weight + 0.2 × recency`
- Applied to: recall, resume Core sorting, triggers, consolidation promotion, buffer eviction.

### Activity-Driven Consolidation

- Consolidation loop skips when no write activity since last run — zero waste during idle.
- Only memory writes (create/update/delete) count as activity; reads don't.
- Agent frozen = system frozen. No phantom decay while inactive.

### Gate Improvements

- Promotion gate batches all candidates into single LLM call (was per-item).
- `_gate-pending` cooldown tag prevents retry storms on LLM errors (2h TTL).
- Triage dedup via `_triaged` filter.

### Other Changes

- Importance floor: 0.3 → 0.0
- `reinforcement_score()` removed — replaced by `memory_weight()`
- `Layer::score_bonus()` removed — absorbed into `memory_weight()`
- Web UI: recall results use dedicated `recallCard()` renderer for `MemoryResult` format

## 0.12.2

### Fixes

- **Topiary name inheritance**: Rebuilds now compare new topics with cached tree by member overlap (Jaccard ≥ 0.5) and inherit existing names, avoiding unnecessary LLM calls. Only truly new/changed topics trigger naming.
- **Unnamed tree protection**: If all topics remain unnamed after naming (e.g. LLM timeout), the tree is not stored — preserving the last good cached tree.
- **`/topic` touch control**: Default behavior is `touch=true` (bumps access_count for agent retrieval). Web UI passes `?touch=false` to avoid polluting access metrics during browsing.

### Changes

- Resume default hours: 4 → 12
- Web: topic grid layout changed from multi-column grid to single-column list
- Templates: removed hardcoded `compact=true` and `hours` params (both default correctly)

## 0.12.0

### Highlights

- **Topiary — topic clustering for memory organization**: Background process automatically clusters all memories (Core + Working + Buffer) into a hierarchical topic tree using spherical k-means with LLM-powered naming. Rebuilds incrementally after embedding updates with 5-second debounce. Resume exposes a compact topic index; agents drill into specific topics via `POST /topic`.
- **Resume v3 — four-section format**: Resume output restructured into Core (full text, mixed-weight sorting), Recent (time-descending), Topics (topiary index), and Triggers (pre-action safety tags). Replaces the old five-section compressed format. No more LLM compression in resume path — all sections are deterministic.
- **API reorganization**: `recall_handlers.rs` (1181 lines) split into `recall.rs`, `resume.rs`, and `topiary_api.rs` for maintainability.

### Features

- `src/topiary/` module: `mod.rs` (tree structure), `cluster.rs` (k-means), `worker.rs` (debounced async), `naming.rs` (LLM batch naming)
- `POST /topic` endpoint: batch topic drill-down by ID (`{"ids": ["kb1", "kb3"]}`)
- Topiary worker: debounced rebuild triggered after embed queue flush and consolidation
- Topic naming via `ai::llm_tool_call` with `TOPIC_NAMING_SYSTEM` prompt
- Core sorting: `importance × kind_boost × (1 + repetition_count × 2.5)`, procedural kind_boost = 1.3
- Triggers section in resume: all `trigger:*` tags sorted by access count
- `topiary_trigger` channel in `AppState` for cross-module signaling

### Refactoring

- `api/recall_handlers.rs` → `api/recall.rs` + `api/resume.rs` + `api/topiary_api.rs`
- Resume no longer uses LLM compression — sections are budget-capped by character count
- Removed old five-section resume format (Core/Sessions/Working/Recent/Buffer)

### Documentation

- ARCHITECTURE.md: added §6 Topiary, updated §7 Resume, added invariant #11
- README: added Topic Clustering feature, updated mermaid diagram, updated Session Recovery
- MCP.md: added `engram_topic` tool
- SETUP.md: added `/topic` to HTTP and MCP templates

## 0.11.0

### Highlights

- **Interactive install scripts**: One-liner setup for macOS, Linux, and Windows. Walks through download, configuration (LLM, embeddings, port, database), MCP client setup, and startup.
- **Embed queue with time-window batching**: Embedding requests are batched using a 500ms time window with a 50-item cap — first item triggers the window, flush on expiry or cap hit. Eliminates per-memory API round-trips.
- **Embed-only mode**: Start engram with just `ENGRAM_EMBED_URL` — no LLM required. Semantic search works, consolidation falls back to heuristics.
- **HNSW memory optimization**: Incremental reconcile/merge (O(new×existing) instead of O(n²)), auto-rebuild at >20% ghost ratio, dynamic initial capacity, embeddings loaded from vec index instead of SQLite.

### Features

- `install.sh` — interactive installer for macOS/Linux (binary download, cargo, systemd, MCP config)
- `install.ps1` — interactive installer for Windows (binary download, cargo, MCP config)
- Embed queue: `MAX_BATCH=50`, `WINDOW_MS=500`, configurable in `src/lib.rs`
- Embed-only `AiConfig::from_env()` — starts with embedding support even without LLM credentials
- HNSW `last_reconcile_ts` tracking for incremental reconciliation
- HNSW auto-rebuild when ghost node ratio exceeds 20%
- Dynamic HNSW capacity: `max(count * 2, 1000)`
- Health endpoint now reports `embed_queue_pending` count

### Fixes

- Consolidation no longer causes 2x memory spike from full HNSW rebuild
- Resume output sorted by importance (descending) as final sort pass
- Fixed embed queue race condition on shutdown

## 0.10.0

### Highlights

- **Anthropic native API support**: Set `ENGRAM_LLM_PROVIDER=anthropic` to use Anthropic's `/v1/messages` format directly — no OpenAI-compatible proxy needed. Supports tool_use format with `input_schema`.
- **Per-component LLM routing**: Each component (gate, audit, merge, extract, expand, rerank, proxy, triage, summary) can have its own `_URL`, `_KEY`, `_MODEL`, `_PROVIDER`. Mix OpenAI for cheap text processing with Anthropic for quality judgment — no proxy required.
- **Importance-weighted buffer eviction**: Buffer overflow now evicts lowest-importance memories first (was FIFO). Lessons and procedurals remain exempt.
- **Consolidation dry-run**: `POST /consolidate?dry_run=true` previews what would be promoted, decayed, and demoted — without writing anything.
- **npm package renamed**: Published as `engram-rs-mcp` on npm (was `engram-mcp`, which was taken).

### Features

- `ENGRAM_LLM_PROVIDER` env var: `openai` (default) or `anthropic`/`claude`
- Per-component env vars: `ENGRAM_{GATE,AUDIT,...}_{URL,KEY,MODEL,PROVIDER}`
- `ResolvedConfig` fallback chain: component-specific → global defaults
- `ENGRAM_NO_FTS_PENALTY` env var (default 0.85) — tunable keyword affinity penalty
- `ENGRAM_HNSW_EF_SEARCH` env var (default 64) — tunable HNSW search breadth
- `ENGRAM_TRIAGE_BATCH` env var (default 20) — tunable triage batch size
- Dry-run response includes `would_promote`, `would_decay`, `would_demote` with ID + content preview
- macOS builds (x86_64 + aarch64) in CI

### Fixes

- Keyword affinity penalty relaxed from 0.7 to 0.85 — less aggressive on cross-concept semantic matches
- Initial importance for lessons and procedurals boosted from 0.5 to 0.75
- Audit interval reduced from 24h to 12h

### Refactoring

- `AiConfig` replaced 7 per-component model fields with `HashMap<String, ComponentConfig>`
- `add_auth`, `llm_chat_*`, `llm_tool_call_*` take `ResolvedConfig` instead of global `AiConfig`
- Single `component_config_from_env()` helper eliminates repeated env var reading

### Documentation

- SETUP.md: mixed-provider configuration examples
- README: updated LLM requirements to mention Anthropic support
- Mermaid diagram updated with audit cycle

## 0.9.0

### Highlights

- **ENGRAM_LLM_LEVEL=auto**: Heuristic-first consolidation — high-confidence decisions (obvious promotions, clear garbage) skip LLM calls entirely. Only uncertain cases go to LLM. `full` and `off` modes also available.
- **Namespace isolation with cross-namespace merging**: Each project gets its own memory space. Resume/recall automatically include the `default` namespace. Consolidation merges project→default but never the reverse.
- **Audit rework — three powers, no delete**: Audit can only schedule (promote/demote), adjust importance, and merge. Deletion is lifecycle-only (TTL, decay). Function calling replaces raw JSON output.
- **Cosine-based dedup**: Replaced Jaccard similarity with cosine similarity for insert dedup. More accurate, especially for CJK content.
- **Working capacity cap**: LRU eviction when Working exceeds `ENGRAM_WORKING_CAP` (default 30). Replaces time-based decay.
- **Buffer capacity cap**: FIFO eviction when Buffer exceeds `ENGRAM_BUFFER_CAP` (default 200).

### Features

- Buffer stuck fix: auto-reset decay on >48h stuck memories, delete >7d abandoned
- Stale tag cleanup: remove orphaned `gate-rejected` and `promotion` tags
- Audit cluster-based review with cosine merge hints
- Core overlap detection during consolidation
- Resume compression caching in `engram_meta`
- Directional namespace merging (project→default only)
- Accept JSON body without Content-Type header
- Compound indexes for namespace queries
- Recall: `budget_tokens=0` treated as unlimited

### Fixes

- Decoupled `sim_floor` from `min_score` in recall — short CJK queries no longer return 0 results
- Reconcile layer guard: lower-layer memories cannot absorb higher-layer ones
- Resume budget calculation uses `chars().count()` instead of byte length — CJK no longer consumes 3× budget
- Audit only counts successful deletions
- Sandbox protects lessons/procedurals from delete, blocks same-layer demote
- Triage prompt filters test infrastructure noise

### Refactoring

- Magic numbers extracted to `src/thresholds.rs`
- Proxy split into module directory
- All tag-based layer routing removed — every memory enters Buffer
- Storage principle inverted: "default store, short exclusion list"

### Documentation

- README rewritten: demo GIF, streamlined features, docs split out
- MCP docs moved to `docs/MCP.md` with `mcp-config.json` in repo root
- ARCHITECTURE.md as internal-only technical reference
- SETUP.md prompt templates aligned across MCP and HTTP

### Tests

- Comprehensive sandbox RuleChecker unit tests
- Buffer dedup, audit parse, scoring boundary tests
- Proxy watermark extraction tests

## 0.8.0

### Highlights

- **HNSW vector index**: Replaced brute-force cosine search with hierarchical navigable small world graph. Faster recall at scale.
- **Semantic dedup on insert**: Same-concept memories now reinforce (access_count bump) or LLM-merge when they carry different details. No more silent duplicates.
- **Resume v2**: Centroid-based relevance filtering for Core memories, proportional truncation across sections, configurable budget (default 16K chars). Core memories compete by relevance — no exemptions.
- **LLM usage tracking**: Full per-component, per-model call logging with web panel visualization and `DELETE /llm-usage` to clear stats.
- **Separate model routing**: `ENGRAM_AUDIT_MODEL` independent from gate, with fallback chain (audit → gate → LLM_MODEL). Three-tier model guide in docs.
- **Reconcile performance fix**: Downgraded from Sonnet to 4o-mini, added fingerprint-based skip and keep_both decision cache. Eliminated O(n²) repeated LLM calls.

### Features

- Semantic dedup on manual insert — same concept reinforces or LLM-merges
- Repetition signal flows through triage and gate decisions
- LLM-compressed Core summary for budget-constrained resume
- Audit sandbox with safety grading (`/audit` and `/audit/sandbox`)
- Audit auto-apply via sandbox: Good+Marginal ops applied, Bad skipped
- `modified_at` field on all memories
- Bilingual query expansion + semantic-gated FTS
- Multi-hop fact queries (knowledge graph)
- Recall pagination (`offset` parameter)
- Web panel: LLM usage stats page, sidebar reorganization, component descriptions
- `DELETE /llm-usage` endpoint to clear usage stats
- `ENGRAM_AUDIT_MODEL` env var with gate fallback
- `get_meta_prefix()` for batch meta key queries
- Message watermark extraction replaces strip_boilerplate

### Fixes

- Reconcile model downgrade (gate→merge) + fingerprint skip + keep_both cache
- Resume no longer touches memories — only real recall increments access
- Gate prompt rejects operational logs, plans/TODOs, system docs
- Total count respects tag/kind/layer filters in `/memories`
- CJK char boundary panic in core summary truncation
- FTS5 syntax error on uppercase NOT/AND/OR
- Rerank score monotonicity
- Proxy always forwards upstream key, strips caller auth
- All DB calls wrapped in `spawn_blocking` (async safety)
- Security + correctness fixes from code audit (7 items)
- Proper error handling with `EngramError` throughout
- `json_each()` replaces LIKE for tag filtering

### Performance

- f64→f32 embeddings, parking_lot mutex, SQLite busy handler
- LRU crate replaces hand-rolled cache
- Deploy profile for 4x faster incremental builds

### Refactoring

- Split `consolidate.rs` into module directory
- Split remaining inline tests to separate files
- Integration tests rewritten with clean test data
- Removed injection detection (safety theater)
- Removed strip_boilerplate, replaced with watermark extraction
- Auth made optional — local deployments need no API key

### Documentation

- Model routing guide (judgment / light judgment / text processing)
- Post-compaction resume guidance in README, SETUP.md, all agent templates
- Recall instructions strengthened — mandatory before non-trivial tasks
- Auth made opt-in across all docs

## 0.7.0

### Highlights

- **Zero-downtime deploys via systemd socket activation**: Socket held by systemd, service restarts queue connections in kernel. Uses `listenfd` crate.
- **Proxy debounce flush**: Replaced fixed 120s extraction timer with 30s silence-based debounce. Flushes after conversation pauses, not on a clock.
- **Buffer→Working/Core reconciliation**: Buffer memories that update existing higher-layer memories are detected (cosine 0.55-0.78, >1h gap), judged by LLM, and promoted to replace the old version. Fixes the "no iterative updates" gap.
- **Audit restart storm fix**: `engram_meta` KV table persists `last_audit_ms` across restarts. Previously, 20 restarts = 20 audit runs.
- **Score cap at 1.0**: Dual-hit FTS boost × layer bonus could push scores to ~1.3; now capped.

### Changes

- **Lesson/procedural auto-promote**: Memories tagged `lesson` or kind `procedural` auto-promote to Working after 2h in Buffer, bypassing normal `promote_threshold`.
- **Resume touches Core/Working**: `/resume` increments `access_count` and bumps importance +0.02 for Core/Working memories, preventing Working decay.
- **X-Engram-Extract header**: `X-Engram-Extract: false` disables proxy memory extraction per-request. Content heuristic fallback for sub-agent detection.
- **Keyword affinity penalty**: Semantic-only results (no FTS match) that lack query terms get 0.7× relevance penalty. Mitigates text-embedding-3-small CJK weakness where unrelated content scores high cosine.
- **Namespace query param alias**: `GET /memories?namespace=X` works alongside `?ns=X`.
- **Namespace SQL-layer filter**: `GET /memories` namespace filter pushed from Rust `retain()` to SQL `WHERE` clause for efficiency.
- **Audit prompt softened**: No longer "aggressive about merges"; protects memories updated within 24h.
- **User message truncation**: Increased from 2000→6000 chars for proxy extraction context.
- **Kind alignment**: `kind` field consistent across all surfaces (prompt, schema, README, AGENTS.md, MCP).
- **Function calling for extraction**: Proxy extraction uses structured function calling instead of raw JSON output.
- **Proxy CJK panic fix**: `&m.content[..120]` byte slice on CJK → `chars().take(120)`.

### Tests

- 145 tests (unit + integration), clippy clean, 0 warnings.
- `engram_meta` persistence test added.

## 0.6.0

### Post-release fixes

- **Recall scoring rebalance**: relevance weight 0.45→0.6, importance 0.3→0.2, core bonus 1.3→1.1. Fixes high-importance memories dominating unrelated queries. Top-1 accuracy up significantly.
- **POST /repair endpoint**: auto-fixes FTS index (orphan cleanup + missing entry rebuild). Idempotent, safe to run anytime.
- **Merge observability**: logs winner ID, absorbed content previews, skip reasons. `ConsolidateResponse.merged_ids` returns winner IDs.
- **Merge length fix**: comparison changed from max→sum of inputs, fixing false rejections where combined input was shorter than individual max.
- **conn() returns Result**: database pool exhaustion returns errors instead of panicking. Non-Result methods (search_fts, search_semantic, etc.) gracefully degrade with empty results.
- **Merge hard cap 800 chars**: prevents information-dense mega-memories that hurt recall precision. Bloat detection rejects merges that expand beyond the longest input.
- **Recall quality test suite**: regression tests covering Top-1/Top-3 accuracy, time-windowed recall, and min_score filtering.
- **Test coverage**: 91 tests total (error.rs, builder chain, namespace delete, import roundtrip, export embeddings, FTS repair, scoring rebalance).

### Release

- **Sync embedding**: `sync_embed: true` on store blocks until embedding is generated. Eliminates 1.2s async race window for critical memories.
- **Export with embeddings**: `GET /export?embed=true` includes 1536-dim vectors. Import preserves them — full migration without re-embedding.
- **Recall min_score**: `min_score: 0.3` drops results below a relevance threshold.
- **Dual-hit boost**: memories found by both semantic AND keyword search get a 30% relevance boost, improving precision for specific-term queries.
- **Improved merge prompt**: enforces conciseness (500 char target, 2k hard cap), avoids vague summaries.
- **Import fixes**: namespace override, id reminting on collision, serde defaults for forward compatibility.
- **Batch store sync_embed**: works on batch endpoint too.
- **Body limit raised to 32MB** for embedding-rich imports.
- **API refactor**: `blocking()` helper replaces 14 spawn_blocking boilerplate chains.
- **Stats/batch-delete namespace filtering**: `?ns=` param and `X-Namespace` header support.
- **73 tests total** (up from 61), 0 clippy warnings. 24/24 smoke tests passing.

## 0.5.0

- **Connection pool**: replaced `Mutex<Connection>` with r2d2 pool (8 connections). Concurrent reads no longer block each other. 702 recalls/s, 956 stats/s under load.
- **Namespace isolation**: memories now have a `namespace` field for multi-agent use. Set via JSON body or `X-Namespace` header. Filter with `?ns=` on list/recent, `namespace` field on recall.
- **Supersede**: `POST /memories` accepts `supersedes: [id, ...]` to atomically replace old memories.
- **Query embedding cache**: LRU cache (128 entries) cuts repeated semantic recalls from 1.1s to 22ms.
- **Merge improvements**: consolidation now picks the newest memory as winner (not highest importance) and preserves max importance from the cluster.
- **API tests**: 13 integration tests covering auth, CRUD, recall, namespace, batch delete.
- **61 tests total**, 0 clippy warnings.

## 0.4.0

- **Body size limit**: 64KB cap on all requests (tower-http)
- **Batch embedding**: extract endpoint now generates all embeddings in a single API call
- **Tests**: consolidated and recall core logic now have dedicated unit tests (42 total)
- **Clippy clean**: zero warnings on stable toolchain
- **README**: updated with merge, rerank, time-filtered recall, and /recent docs

## 0.3.0

- **LLM-powered merge**: consolidation finds semantically similar memories (cosine > 0.68) and merges them via LLM
- **LLM recall rerank**: `rerank: true` in recall requests re-ranks results using the LLM for better relevance
- **Episodic memory**: time-filtered recall (`since`/`until`), `sort_by` parameter, `/recent` endpoint
- **CJK dedup fix**: near-duplicate detection now uses bigram tokenization for Chinese/Japanese/Korean text
- **Batch update**: `update_fields` uses a single query instead of multiple roundtrips
- **Embedding optimization**: `row_to_memory` skips deserializing embedding blobs unless explicitly needed

## 0.2.1

- **CJK bigram FTS**: full-text search works for Chinese/Japanese/Korean via bigram indexing at insert and query time
- **Near-duplicate detection**: Jaccard similarity check on insert prevents storing redundant memories
- **Tag filtering**: list and recall support `tag` parameter
- **Quick search**: `GET /search?q=term` for simple keyword lookup
- **Export/import**: `GET /export` and `POST /import` for backup and migration
- **Auth**: optional Bearer token with constant-time comparison
- **Consolidation improvements**: age-based promotion, access-count promotion, decay-based cleanup
- **Error handling**: structured error types with proper HTTP status codes
- **Extract**: LLM-powered memory extraction from raw text
- **Dockerfile + CI**: containerized deployment, GitHub Actions workflow

## 0.1.0

- Initial release
- Three-layer memory model (Buffer → Working → Core)
- Hybrid recall (semantic embeddings + FTS5 keyword search)
- Budget-aware retrieval with composite scoring
- Configurable decay rates per layer
- OpenAI-compatible embedding and LLM backends
