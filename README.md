# engram

A biologically-inspired memory engine for AI agents — store, forget, and recall like a brain.

## Why?

Most agent memory solutions dump everything into a vector database and call it a day. That doesn't map well to how memory actually works — some things should be forgotten, some things should stick, and retrieval should be context-aware.

engram uses a three-layer model based on [Atkinson-Shiffrin memory theory](https://en.wikipedia.org/wiki/Atkinson%E2%80%93Shiffrin_memory_model):

| Layer | Role | Decay |
|-------|------|-------|
| **Buffer** (L1) | Transient context | Fast (~1 day half-life) |
| **Working** (L2) | Active knowledge | Moderate (~5 day half-life) |
| **Core** (L3) | Identity & principles | Near-zero |

Memories promote upward through access frequency (Ebbinghaus-style reinforcement), and decay naturally when neglected. You don't manage layers — store everything as buffer, and the system promotes what sticks:

- **Buffer → Working**: recalled ≥2 times, or accessed before TTL expires
- **Working → Core**: recalled ≥3 times with sufficient importance, or survived 7+ days with any access
- Each recall bumps importance by 0.05 (capped at 1.0)

## Quick Start

```bash
# 1. Build
cargo build --release

# 2. Run (works without AI — keyword search out of the box)
./target/release/engram

# 3. Store and recall
curl -X POST http://localhost:3917/memories \
  -H 'Content-Type: application/json' \
  -d '{"content": "prefers dark mode and vim keybindings"}'

curl -X POST http://localhost:3917/recall \
  -H 'Content-Type: application/json' \
  -d '{"query": "editor preferences", "budget_tokens": 500}'
```

To enable semantic search and LLM features, add AI backends:

```bash
ENGRAM_LLM_URL=http://localhost:4000/v1/chat/completions \
ENGRAM_LLM_KEY=sk-xxx \
./target/release/engram
```

Requires Rust 1.75+. Single binary, ~4 MB, <10 MB RSS.

## Features

- Three-layer memory with configurable decay and automatic promotion
- Hybrid search: semantic (embeddings) + BM25 keyword matching
- LLM-powered memory merge: consolidation detects similar memories and merges them via LLM
- LLM re-ranking: optionally re-rank recall results using an LLM for better relevance
- LLM extraction: feed raw text, get structured memories
- Query expansion: LLM bridges abstract queries to concrete terms
- Time-windowed recall: `since`/`until` filters, `/recent` endpoint
- Budget-aware recall with composite scoring
- Transparent LLM proxy: auto-extract memories from any LLM conversation
- Trigger memories: tag with `trigger:action-name`, fetch before risky operations
- Session recovery: one-call agent wake-up (`/resume`)
- CJK tokenization support (bigram indexing)
- Near-duplicate detection on insert
- Multi-agent namespace isolation
- Optional Bearer token auth
- Web dashboard (`/ui`)
- AI-optional: works without AI (pure FTS), gains semantic search + LLM features with it

## API

### Store

```bash
curl -X POST http://localhost:3917/memories \
  -H 'Content-Type: application/json' \
  -d '{"content": "prefers dark mode and vim keybindings", "importance": 0.8}'

# Replace outdated memories
curl -X POST http://localhost:3917/memories \
  -H 'Content-Type: application/json' \
  -d '{"content": "now uses v0.5.0", "supersedes": ["<old-memory-id>"]}'

# Batch create
curl -X POST http://localhost:3917/memories/batch \
  -H 'Content-Type: application/json' \
  -d '[{"content": "fact one"}, {"content": "fact two"}]'
```

### Recall

```bash
# Basic recall (budget-aware)
curl -X POST http://localhost:3917/recall \
  -H 'Content-Type: application/json' \
  -d '{"query": "user preferences", "budget_tokens": 500}'

# With LLM re-ranking for better relevance
curl -X POST http://localhost:3917/recall \
  -H 'Content-Type: application/json' \
  -d '{"query": "user preferences", "budget_tokens": 500, "rerank": true}'

# With query expansion (bridges abstract → concrete terms)
curl -X POST http://localhost:3917/recall \
  -H 'Content-Type: application/json' \
  -d '{"query": "observability stack", "expand": true}'
```

Scoring formula: `(0.6 * relevance + 0.2 * importance + 0.2 * recency) * layer_bonus`. Core memories get a slight scoring bonus. If the top result scores below 0.4 and an LLM is configured, query expansion kicks in automatically.

### Session Recovery

One call to restore agent context on wake-up:

```bash
curl http://localhost:3917/resume?hours=4
```

Returns `identity` (high-importance core memories), `recent` (time-windowed), `sessions` (source=session), and `next_actions` (tagged `next-action`).

### Extract from Text

```bash
curl -X POST http://localhost:3917/extract \
  -H 'Content-Type: application/json' \
  -d '{"text": "Moved to Tokyo, timezone JST, always reply in Japanese."}'
```

Requires AI. Parses unstructured text into structured memories automatically.

### Consolidate

```bash
# Promote/decay only
curl -X POST http://localhost:3917/consolidate

# With LLM merge: detect and merge semantically similar memories
curl -X POST http://localhost:3917/consolidate \
  -H 'Content-Type: application/json' \
  -d '{"merge": true}'
```

### Namespace Isolation

Each agent can store and query memories in its own namespace via the `X-Namespace` header or `namespace` field:

```bash
curl -X POST http://localhost:3917/memories \
  -H 'Content-Type: application/json' \
  -H 'X-Namespace: agent-a' \
  -d '{"content": "agent-a private context"}'

curl -X POST http://localhost:3917/recall \
  -H 'Content-Type: application/json' \
  -H 'X-Namespace: agent-a' \
  -d '{"query": "what am I working on?"}'
```

### Triggers

Pre-action safety recall. Tag memories with `trigger:action-name`, then query before risky operations:

```bash
# Store a lesson
curl -X POST http://localhost:3917/memories \
  -H 'Content-Type: application/json' \
  -d '{"content": "never commit .env files", "tags": ["trigger:git-push"]}'

# Before pushing, check lessons
curl http://localhost:3917/triggers/git-push
```

### Full Endpoint List

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health + stats |
| `GET` | `/health` | Detailed health (uptime, RSS, cache stats) |
| `GET` | `/stats` | Layer counts |
| `POST` | `/memories` | Create memory |
| `POST` | `/memories/batch` | Batch create |
| `GET` | `/memories` | List (`?layer=N&tag=X&ns=X&limit=50&offset=0`) |
| `GET` | `/memories/:id` | Get by id |
| `PATCH` | `/memories/:id` | Update |
| `DELETE` | `/memories/:id` | Delete |
| `DELETE` | `/memories` | Batch delete (`{"ids": [...]}` or `{"namespace": "x"}`) |
| `POST` | `/recall` | Hybrid search (semantic + keyword, budget-aware) |
| `GET` | `/search` | Quick keyword search (`?q=term&limit=10`) |
| `GET` | `/recent` | Recent memories (`?hours=2&limit=20&layer=N&source=X`) |
| `GET` | `/resume` | Session recovery (`?hours=4`) |
| `GET` | `/triggers/:action` | Pre-action recall |
| `POST` | `/consolidate` | Maintenance cycle (optional `{"merge": true}`) |
| `POST` | `/extract` | LLM extraction (requires AI) |
| `POST` | `/repair` | Fix FTS index + backfill embeddings |
| `POST` | `/vacuum` | Reclaim disk space (`?full=true` for full vacuum) |
| `GET` | `/export` | Export all memories as JSON (`?embed=true` includes vectors) |
| `POST` | `/import` | Import memories (skips existing ids) |
| `ANY` | `/proxy/*` | Transparent LLM proxy |
| `GET` | `/ui` | Web dashboard |

## LLM Proxy

engram can sit between your tools and any LLM API, automatically extracting memories from conversations. Works with OpenAI, Anthropic, Gemini, Ollama, or any provider.

```bash
# Enable the proxy
export ENGRAM_PROXY_UPSTREAM=https://api.openai.com

# Point your tools at engram instead of the provider:
# Before: https://api.openai.com/v1/chat/completions
# After:  http://localhost:3917/proxy/v1/chat/completions

curl http://localhost:3917/proxy/v1/chat/completions \
  -H "Authorization: Bearer sk-your-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}'
```

How it works:
1. Forwards request and headers to upstream verbatim
2. Streams the response back with no added latency
3. After the exchange completes, asynchronously extracts key facts/decisions/preferences via LLM
4. Stores extracted memories in the buffer layer (source: `proxy`, tagged `auto-extract`)

The proxy is selective — it only extracts user preferences, concrete decisions, hard-won lessons, and critical facts. Routine task details are ignored. You can use a cheaper model for proxy extraction via `ENGRAM_PROXY_MODEL`.

Three ways to build memory — use any combination:
- **Proxy**: automatic, no agent changes needed
- **API**: `POST /memories` for explicit storage
- **MCP**: `engram_store` tool for Claude Desktop, Cursor, etc.

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
| `ENGRAM_LLM_URL` | — | Chat completions endpoint (enables AI features) |
| `ENGRAM_LLM_KEY` | — | LLM API key |
| `ENGRAM_LLM_MODEL` | `gpt-4o-mini` | Default LLM model for all components |
| `ENGRAM_MERGE_MODEL` | *(LLM_MODEL)* | Model for memory merge during consolidation |
| `ENGRAM_EXTRACT_MODEL` | *(LLM_MODEL)* | Model for `/extract` endpoint |
| `ENGRAM_RERANK_MODEL` | *(LLM_MODEL)* | Model for recall re-ranking |
| `ENGRAM_EXPAND_MODEL` | *(LLM_MODEL)* | Model for query expansion |
| `ENGRAM_PROXY_MODEL` | *(LLM_MODEL)* | Model for proxy auto-extraction |
| `ENGRAM_EMBED_URL` | *(derived from LLM_URL)* | Embeddings endpoint |
| `ENGRAM_EMBED_KEY` | *(LLM_KEY)* | Embeddings API key |
| `ENGRAM_EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `ENGRAM_CONSOLIDATE_MINS` | `30` | Auto-consolidation interval (0 to disable) |
| `ENGRAM_AUTO_MERGE` | `false` | Enable LLM merge in auto-consolidation |
| `ENGRAM_PROXY_UPSTREAM` | — | LLM proxy upstream URL (enables `/proxy/*`) |
| `ENGRAM_PROXY_KEY` | — | Fallback API key for proxy (if client omits auth) |
| `RUST_LOG` | `info` | Log level |

Per-component model overrides let you use a cheaper model for high-volume operations (proxy extraction, query expansion) while keeping a stronger model for consolidation merges or re-ranking.

## License

MIT
