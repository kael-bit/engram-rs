# engram

Hierarchical memory engine for AI agents.

## Why?

Most agent memory solutions dump everything into a vector database and call it a day. That doesn't map well to how memory actually works — some things should be forgotten, some things should stick, and retrieval should be context-aware.

engram uses a three-layer model based on [Atkinson-Shiffrin memory theory](https://en.wikipedia.org/wiki/Atkinson%E2%80%93Shiffrin_memory_model):

| Layer | Role | Decay |
|-------|------|-------|
| **Buffer** (L1) | Transient context | Fast (~1 day half-life) |
| **Working** (L2) | Active knowledge | Moderate (~5 day half-life) |
| **Core** (L3) | Identity & principles | Near-zero |

Memories promote upward through access frequency (Ebbinghaus-style reinforcement), and decay naturally when neglected. You don't need to decide what layer to write to — store everything as buffer, and the system promotes what sticks:

- **Buffer → Working**: recalled ≥2 times, or accessed at least once before TTL expires
- **Working → Core**: recalled ≥3 times with sufficient importance, or survived 7+ days with any access
- Each recall bumps importance by 0.05 (capped at 1.0)

Retrieval is budget-aware: you specify a token limit and get the optimal selection.

## Features

- Three-layer memory with configurable decay and automatic promotion
- Hybrid search: semantic (embeddings) + BM25 keyword matching
- LLM-powered memory merge: consolidation detects similar memories and merges them via LLM
- LLM re-ranking: optionally re-rank recall results using an LLM for better relevance
- LLM extraction: feed raw text, get structured memories
- Time-windowed recall: `since`/`until` filters, `/recent` endpoint
- Budget-aware recall with composite scoring
- CJK tokenization support (bigram indexing for Chinese/Japanese/Korean)
- Near-duplicate detection on insert
- Supersede: replace outdated memories by id
- Sync embedding: `sync_embed: true` blocks until embedding is ready (no async race window)
- Trigger memories: tag with `trigger:action-name`, fetch via `GET /triggers/:action` before risky operations
- Multi-agent namespace isolation
- Query expansion: LLM bridges abstract queries to concrete terms
- Connection pool (r2d2) for concurrent access
- Optional Bearer token auth
- AI-optional: works without AI (pure FTS), gains semantic search + LLM features with it
- Single binary, ~4 MB, <10 MB RSS

## Getting started

```bash
# Build from source
cargo build --release
# Binary: target/release/engram

# Run without AI (keyword search only)
./target/release/engram

# Run with AI backends
ENGRAM_LLM_URL=http://localhost:4000/v1/chat/completions \
ENGRAM_LLM_KEY=sk-xxx \
./target/release/engram --port 3917
```

Requires Rust 1.75+.

## API

### Store a memory

```bash
curl -X POST http://localhost:3917/memories \
  -H 'Content-Type: application/json' \
  -d '{"content": "prefers dark mode and vim keybindings", "importance": 0.8, "layer": 2}'

# Replace old memories with updated info
curl -X POST http://localhost:3917/memories \
  -H 'Content-Type: application/json' \
  -d '{"content": "now uses v0.5.0", "supersedes": ["<old-memory-id>"]}'

# Store in a namespace (multi-agent isolation)
curl -X POST http://localhost:3917/memories \
  -H 'Content-Type: application/json' \
  -H 'X-Namespace: agent-a' \
  -d '{"content": "agent-a private context"}'
```

### Namespace isolation

Each agent can store and query memories in its own namespace. Set via `X-Namespace` header or `namespace` field in the request body. Queries only return memories from the specified namespace.

```bash
# List only agent-a's memories
curl 'http://localhost:3917/memories?ns=agent-a'

# Recall within a namespace
curl -X POST http://localhost:3917/recall \
  -H 'Content-Type: application/json' \
  -H 'X-Namespace: agent-a' \
  -d '{"query": "what am I working on?"}'
```

### Recall

```bash
# Basic recall
curl -X POST http://localhost:3917/recall \
  -H 'Content-Type: application/json' \
  -d '{"query": "user preferences", "budget_tokens": 500}'

# With LLM re-ranking
curl -X POST http://localhost:3917/recall \
  -H 'Content-Type: application/json' \
  -d '{"query": "user preferences", "budget_tokens": 500, "rerank": true}'

# Time-filtered (last 4 hours)
curl -X POST http://localhost:3917/recall \
  -H 'Content-Type: application/json' \
  -d '{"query": "recent work", "budget_tokens": 500, "since": 1708600000000}'

# With query expansion (bridges abstract → concrete terms)
curl -X POST http://localhost:3917/recall \
  -H 'Content-Type: application/json' \
  -d '{"query": "observability stack", "expand": true}'
```

Scoring: `(0.6 × relevance + 0.2 × importance + 0.2 × recency) × layer_bonus`. Core memories get a slight scoring bonus.

### Session recovery

```bash
# One call to get identity, recent activity, and session context
curl http://localhost:3917/resume?hours=4
```

Returns `identity` (high-importance core memories), `recent` (time-windowed), and `sessions` (source=session). Designed for agent wake-up.

### Recent memories

```bash
curl 'http://localhost:3917/recent?hours=4&limit=20&source=session'
```

### Extract from text

```bash
curl -X POST http://localhost:3917/extract \
  -H 'Content-Type: application/json' \
  -d '{"text": "Moved to Tokyo, timezone JST, always reply in Japanese."}'
```

### Consolidate

```bash
# Basic: promote/decay only
curl -X POST http://localhost:3917/consolidate

# With LLM merge: detect and merge semantically similar memories
curl -X POST http://localhost:3917/consolidate \
  -H 'Content-Type: application/json' \
  -d '{"merge": true}'
```

Promotes frequently-accessed important memories upward, drops decayed entries. With `merge: true`, uses embedding similarity + LLM to intelligently merge duplicates.

### Full endpoint list

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health + stats |
| `GET` | `/stats` | Layer counts |
| `POST` | `/memories` | Create |
| `GET` | `/memories` | List (`?layer=N&tag=X&limit=50&offset=0`) |
| `GET` | `/memories/:id` | Get |
| `PATCH` | `/memories/:id` | Update |
| `DELETE` | `/memories/:id` | Delete |
| `POST` | `/recall` | Hybrid search (semantic + keyword, budget-aware, optional rerank) |
| `GET` | `/search` | Quick keyword search (`?q=term&limit=10`) |
| `GET` | `/recent` | Recent memories (`?hours=2&limit=20&layer=N&source=X`) |
| `GET` | `/resume` | Session recovery (`?hours=4`) |
| `POST` | `/consolidate` | Maintenance cycle (optional `{"merge": true}`) |
| `POST` | `/extract` | LLM extraction (requires AI) |
| `POST` | `/repair` | Fix FTS index (idempotent) |
| `GET` | `/health` | Detailed health (uptime, RSS, cache stats) |
| `GET` | `/triggers/:action` | Pre-action recall (memories tagged `trigger:action`) |
| `GET` | `/export` | Export all memories as JSON (`?embed=true` includes vectors) |
| `POST` | `/import` | Import memories (skips existing ids) |
| `POST` | `/memories/batch` | Batch create |
| `DELETE` | `/memories/batch` | Batch delete (`{"ids": [...]}`) |

## MCP

Included MCP server for Claude Desktop, Cursor, etc:

```bash
cd mcp && npm install && npm run build
```

```json
{
  "mcpServers": {
    "engram": {
      "command": "node",
      "args": ["/path/to/engram/mcp/dist/index.js"],
      "env": { "ENGRAM_URL": "http://localhost:3917" }
    }
  }
}
```

Tools: `engram_store`, `engram_recall`, `engram_recent`, `engram_resume`, `engram_search`, `engram_stats`, `engram_extract`, `engram_consolidate`, `engram_repair`, `engram_health`, `engram_triggers`.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_PORT` | `3917` | Server port |
| `ENGRAM_DB` | `engram.db` | Database path |
| `ENGRAM_API_KEY` | — | Bearer token auth (optional) |
| `ENGRAM_LLM_URL` | — | Chat completions endpoint |
| `ENGRAM_LLM_KEY` | — | LLM API key |
| `ENGRAM_LLM_MODEL` | `gpt-4o-mini` | Extraction model |
| `ENGRAM_EMBED_URL` | *(derived from LLM)* | Embeddings endpoint |
| `ENGRAM_EMBED_KEY` | *(same as LLM)* | Embeddings API key |
| `ENGRAM_EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `ENGRAM_CONSOLIDATE_MINS` | `30` | Auto-consolidation interval (0 to disable) |
| `ENGRAM_AUTO_MERGE` | `false` | Enable LLM merge in auto-consolidation |
| `RUST_LOG` | `info` | Log level |

## License

MIT
