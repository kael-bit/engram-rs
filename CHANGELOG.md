# Changelog

## 0.6.0

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
