# Changelog

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
