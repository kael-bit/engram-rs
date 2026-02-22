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

Memories promote upward through access frequency and importance, and decay naturally when neglected. Retrieval is budget-aware: you specify a token limit and get the optimal selection.

## Features

- Three-layer memory with configurable decay and automatic promotion
- Hybrid search: semantic (embeddings) + BM25 keyword matching
- LLM extraction: feed raw text, get structured memories
- Budget-aware recall with composite scoring
- CJK tokenization support
- Optional Bearer token auth
- AI-optional: works without AI (pure FTS), gains semantic search with it
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

Requires Rust 1.85+.

## API

### Store a memory

```bash
curl -X POST http://localhost:3917/memories \
  -H 'Content-Type: application/json' \
  -d '{"content": "prefers dark mode and vim keybindings", "importance": 0.8, "layer": 2}'
```

### Recall

```bash
curl -X POST http://localhost:3917/recall \
  -H 'Content-Type: application/json' \
  -d '{"query": "user preferences", "budget_tokens": 500}'
```

Scoring: `(0.4 × importance + 0.3 × recency + 0.3 × relevance) × layer_bonus`. Core memories always included.

### Extract from text

```bash
curl -X POST http://localhost:3917/extract \
  -H 'Content-Type: application/json' \
  -d '{"text": "Moved to Tokyo, timezone JST, always reply in Japanese."}'
```

### Consolidate

```bash
curl -X POST http://localhost:3917/consolidate
```

Promotes frequently-accessed important memories upward, drops decayed entries.

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
| `POST` | `/recall` | Hybrid search (semantic + keyword, budget-aware) |
| `GET` | `/search` | Quick keyword search (`?q=term&limit=10`) |
| `POST` | `/consolidate` | Maintenance cycle |
| `POST` | `/extract` | LLM extraction (requires AI) |

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

Tools: `engram_store`, `engram_recall`, `engram_extract`, `engram_consolidate`.

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
| `RUST_LOG` | `info` | Log level |

## License

MIT
