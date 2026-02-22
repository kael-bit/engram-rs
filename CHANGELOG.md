# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-02-23

### Added
- `GET /search?q=&limit=` — lightweight keyword search without recall scoring
- `GET /memories?tag=X` — filter memory list by tag
- `GET /export` / `POST /import` — full backup and restore
- Near-duplicate detection on insert (Jaccard similarity > 0.8 auto-merges)
- CJK bigram indexing — Chinese/Japanese/Korean full-text search actually works now
- Age-based Working→Core promotion in consolidation (7 days + importance ≥ 0.5)
- Configurable buffer TTL and working age thresholds for consolidation
- Consolidation response includes promoted/dropped memory ids
- `engram_search` tool in MCP server
- 11 new integration tests (60 total)

### Fixed
- CJK characters were not indexed by FTS5 unicode61 tokenizer — fixed via bigram preprocessing
- `budget_tokens=0` now correctly returns empty results (was returning 1)
- Recall `touch()` only fires for genuinely relevant results (relevance > 0.2)
- Core memory fallback uses low baseline relevance instead of 0 to avoid inflated access counts
- Extract prompt preserves input language and skips trivial metadata

### Changed
- FTS management moved from SQLite triggers to application-level (required for bigram preprocessing)
- FTS index auto-rebuilds on startup only when CJK bigrams are missing
- Constant-time Bearer token comparison (via `subtle` crate)
- Simplified error enum: merged specific validation variants into `Validation(String)`

## [0.2.1] - 2026-02-23

Initial public release.

### Features
- Three-layer hierarchical memory: Buffer → Working → Core
- Exponential decay with layer-specific rates
- Budget-aware recall with composite scoring (importance + recency + relevance)
- Hybrid search: semantic (cosine similarity) + keyword (FTS5 BM25)
- LLM-based memory extraction from raw text
- CJK tokenization support
- REST API with full CRUD
- Optional Bearer token auth
- Optional AI integration (OpenAI-compatible embeddings + chat)
- MCP server for Claude/Cursor integration
- SQLite WAL storage, single ~4 MB binary
