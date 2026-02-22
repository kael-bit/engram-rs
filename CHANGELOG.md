# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
