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

- **Buffer → Working**: reinforcement score ≥ 5.0 (access + repetition × 2.5), OR lesson/procedural memories auto-promote after 2h cooldown
- **Working → Core**: reinforcement score ≥ 3.0, importance ≥ 0.6, passes LLM quality gate (also auto-classifies kind)
- **Buffer TTL**: 24 hours — unaccessed buffer entries are dropped; rescued to Working if access score ≥ half threshold OR importance ≥ 0.7
- **Procedural** memories and **lessons** (`tag=lesson`) are exempt from TTL — they persist indefinitely
- **Session notes** (`source=session`) and `ephemeral`-tagged memories never promote to Core
- **Session distillation**: when 3+ session notes accumulate, consolidation synthesizes a project status snapshot as a regular memory that can promote normally — so project context reaches Core even though session notes can't
- **Gate retry**: memories rejected by the Core promotion gate get a second chance after 7 days
- Each recall bumps importance by 0.02 (capped at 1.0); `/resume` touches Core/Working memories
- Near-duplicate insertions count as repetition (2.5× weight)

## Quick Start

### Install

```bash
# From source
git clone https://github.com/kael-bit/engram && cd engram
cargo build --release
# Binary at ./target/release/engram (~9 MB)
```

### Run

```bash
# Minimal — keyword search works out of the box, no AI needed
./target/release/engram

# With semantic search + LLM features
ENGRAM_LLM_URL=https://api.openai.com/v1/chat/completions \
ENGRAM_LLM_KEY=sk-xxx \
ENGRAM_EMBED_URL=https://api.openai.com/v1/embeddings \
ENGRAM_EMBED_KEY=sk-xxx \
./target/release/engram
```

### Try it

```bash
# Store
curl -X POST http://localhost:3917/memories \
  -H 'Content-Type: application/json' \
  -d '{"content": "prefers dark mode and vim keybindings"}'

# Recall
curl -X POST http://localhost:3917/recall \
  -H 'Content-Type: application/json' \
  -d '{"query": "editor preferences"}'

# Dashboard
open http://localhost:3917/ui
```

Single binary, ~9 MB, ~65 MB RSS (includes jieba dictionary for CJK segmentation). No Docker, no Python, no external database.

## Real-World Example

Here's what a typical agent session looks like with engram:

```
# Agent wakes up — first thing, restore context
$ curl -s localhost:3917/resume?hours=6&compact=true | jq .

core: [
  {"content": "I'm Atlas, a memory-augmented assistant"},
  {"content": "Lesson: never commit internal docs to public repos"},
  {"content": "Lesson: deployment must be atomic — stop && cp && start in one command"}
]
working: [
  {"content": "Currently iterating on engram proxy + recall quality"}
]
next_actions: [
  {"content": "Next: test proxy extraction with real conversations"}
]

# Agent knows who it is, what it was doing, and what to do next.

# During work — user says something important
> User: "from now on, always use Rust instead of Python for new projects"

$ curl -sX POST localhost:3917/memories \
  -H 'Content-Type: application/json' \
  -d '{"content": "user preference: always use Rust over Python for new projects", "importance": 0.9}'

# Before deploying — check safety triggers
$ curl -s localhost:3917/triggers/deploy
[{"content": "Lesson: deployment must be atomic — stop && cp && start in one command"}]

# Session ending — store continuity
$ curl -sX POST localhost:3917/memories \
  -H 'Content-Type: application/json' \
  -d '{"content": "Session: added fact triple API, all tests pass. Next: write contradiction resolution tests.", "tags": ["session"]}'
```

No setup, no config files, no framework integration. Just HTTP calls.

## Setup Guides

### Claude Code

Claude Code can use engram in two ways — pick one or combine both.

#### Option A: MCP Tools (recommended)

Claude Code calls engram through MCP tools — explicit store/recall under your control.

```bash
# 1. Build the MCP server
cd mcp && npm install && npm run build && cd ..

# 2. Add to Claude Code
claude mcp add engram -- node /path/to/engram/mcp/dist/index.js

# Or manually create .mcp.json in your project root:
```

```json
{
  "mcpServers": {
    "engram": {
      "command": "node",
      "args": ["/path/to/engram/mcp/dist/index.js"],
      "env": {
        "ENGRAM_URL": "http://localhost:3917",
        "ENGRAM_API_KEY": "your-key-here"
      }
    }
  }
}
```

Then add to your project's `CLAUDE.md` (or system prompt):

```markdown
## Memory

You have persistent memory via engram MCP tools.

### Every session start
Call `engram_resume` with hours=6 to restore context. Read the core and working sections — that's who you are and what you were doing.

### During conversation
- Decisions, preferences, lessons learned → `engram_store` immediately. Don't wait.
- Workflows, procedures, how-to steps → `engram_store` with kind="procedural" (these never decay).
- "Remember this" from the user → `engram_store` with importance=0.9.
- Need context → `engram_recall` with the question as query.
- Before risky operations (git push, deploy, send message) → `engram_triggers` with the action name.

### Before session ends
Store a session summary via `engram_store`:
- What you did this session
- What was decided
- What to do next (critical — without this you'll wake up lost)
Tags: "session". This is how you maintain continuity across sessions.

### What NOT to store
- Routine task details ("ran cargo build, it passed")
- Things already in project files
- Transient operational status
```

#### Option B: LLM Proxy (automatic capture)

Route Claude Code's API calls through engram — memories are extracted automatically from every conversation. No prompt changes needed.

```bash
# 1. Start engram with proxy pointing at your real API
ENGRAM_PROXY_UPSTREAM=https://api.anthropic.com \
ENGRAM_LLM_URL=https://api.openai.com/v1/chat/completions \
ENGRAM_LLM_KEY=sk-xxx \
./target/release/engram

# 2. Tell Claude Code to use engram as its API endpoint
ANTHROPIC_BASE_URL=http://localhost:3917/proxy claude
# Or set it in your shell profile for all sessions
```

Now every conversation flows through engram. It forwards requests transparently (no latency), then asynchronously extracts key facts, decisions, and preferences into memory.

> **Note:** The proxy only handles **writing** — it captures memories from conversations automatically. For the agent to **read** memories back (resume, recall, search), you need either MCP tools (Option A) or prompt instructions that call the HTTP API directly (e.g., `curl` in CLAUDE.md). **The proxy alone does not give the agent memory recall.**

**Controlling extraction:** Set the `X-Engram-Extract: false` header on requests where you don't want memory extraction (e.g., sub-agent or tool-use traffic). The proxy also auto-detects common sub-agent patterns as a fallback.

#### Option A + B Combined (recommended)

Automatic extraction from all conversations, plus explicit tools for recall and precise control:

```bash
# Start with proxy
ENGRAM_PROXY_UPSTREAM=https://api.anthropic.com \
ENGRAM_LLM_URL=https://api.openai.com/v1/chat/completions \
ENGRAM_LLM_KEY=sk-xxx \
./target/release/engram
```

```bash
# Claude Code uses proxy for conversations
export ANTHROPIC_BASE_URL=http://localhost:3917/proxy
```

MCP config as above. Now Claude Code can explicitly recall/store via MCP tools while all conversations are also passively captured by the proxy.

### OpenClaw

OpenClaw routes LLM requests through configurable providers. Point the Anthropic provider at engram's proxy:

```jsonc
// ~/.openclaw/openclaw.json
{
  "models": {
    "providers": {
      "anthropic": {
        "baseUrl": "http://127.0.0.1:3917/proxy",
        "apiKey": "your-engram-api-key",
        "api": "anthropic-messages",
        "models": [
          { "id": "claude-opus-4-6", "name": "Claude Opus" },
          { "id": "claude-sonnet-4-20250514", "name": "Claude Sonnet" }
        ]
      }
    }
  }
}
```

Restart the gateway after editing: `openclaw gateway restart`

All conversations between OpenClaw agents and Anthropic models now flow through engram. Memories are extracted automatically.

For explicit memory operations, add instructions to your workspace files:

```markdown
<!-- AGENTS.md or HEARTBEAT.md -->
## Memory

engram is your memory. Without it, every session starts from zero.

### Every wake-up / heartbeat
Restore context first — nothing else matters until you know who you are:
  curl -s http://localhost:3917/resume?hours=6&compact=true

Read the core section (identity, constraints, lessons) and working section (active context).
If there are next_actions, that's your todo list.

### Storing memories
Important facts, decisions, lessons:
  curl -X POST http://localhost:3917/memories \
    -H 'Content-Type: application/json' \
    -d '{"content": "user prefers direct answers, no filler", "tags": ["preference"]}'

Lessons from mistakes (with trigger for pre-action recall):
  curl -X POST http://localhost:3917/memories \
    -H 'Content-Type: application/json' \
    -d '{"content": "never force-push to main", "tags": ["lesson","trigger:git-push"]}'

Workflows and procedures (these never decay):
  curl -X POST http://localhost:3917/memories \
    -H 'Content-Type: application/json' \
    -d '{"content": "deploy flow: cargo test && cargo build --release && systemctl stop && cp && systemctl start", "tags": ["deploy"], "kind": "procedural"}'

User explicitly says "remember this":
  curl -X POST http://localhost:3917/memories \
    -H 'Content-Type: application/json' \
    -d '{"content": "...", "importance": 0.9}'

### Before risky operations
  curl -s http://localhost:3917/triggers/git-push
  curl -s http://localhost:3917/triggers/deploy

### Before session ends / compaction
Store what you did and what comes next:
  curl -X POST http://localhost:3917/memories \
    -H 'Content-Type: application/json' \
    -d '{"content": "Session: refactored auth module, decided to use JWT. Next: write integration tests.", "tags": ["session"]}'

Without the "next" part, you'll wake up not knowing what to do.

### Don't store
- Routine ops ("ran tests, passed")
- Things already in code/config files
- Transient status that won't matter tomorrow
```

### Cursor / Windsurf / Other Editors

Any editor that supports MCP can use engram:

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

For editors without MCP, use the proxy approach — set the editor's LLM API endpoint to `http://localhost:3917/proxy/...`.

### systemd (Production)

```ini
# /etc/systemd/system/engram.socket
[Unit]
Description=engram memory engine socket

[Socket]
ListenStream=3917
NoDelay=true

[Install]
WantedBy=sockets.target
```

```ini
# /etc/systemd/system/engram.service
[Unit]
Description=engram memory engine
After=network.target
Requires=engram.socket

[Service]
ExecStart=/usr/local/bin/engram
Environment=ENGRAM_DB=/var/lib/engram/engram.db
Environment=ENGRAM_API_KEY=your-secret-key
Environment=ENGRAM_LLM_URL=https://api.openai.com/v1/chat/completions
Environment=ENGRAM_LLM_KEY=sk-xxx
Environment=ENGRAM_PROXY_UPSTREAM=https://api.anthropic.com
Restart=always
RestartSec=1

[Install]
WantedBy=multi-user.target
```

Socket activation gives zero-downtime deploys — the socket stays open during restarts, queuing connections in the kernel.

```bash
sudo systemctl enable --now engram.socket engram
```

## Features

### Memory Types

Three kinds of memory with different decay behavior:

| Kind | Decay | Use Case |
|------|-------|----------|
| `semantic` | Normal | Facts, preferences, knowledge (default) |
| `episodic` | Normal | Events, experiences, time-bound context |
| `procedural` | Near-zero | Workflows, instructions, how-to — persists indefinitely |

```bash
curl -X POST http://localhost:3917/memories \
  -H 'Content-Type: application/json' \
  -d '{"content": "deploy with: cargo build --release && scp ...", "kind": "procedural"}'
```

### Fact Triples & Contradiction Resolution

engram extracts structured facts as (subject, predicate, object) triples from memories, enabling relationship queries and automatic contradiction detection:

```bash
# Store facts explicitly
curl -X POST http://localhost:3917/facts \
  -H 'Content-Type: application/json' \
  -d '{"facts": [{"subject": "alice", "predicate": "role", "object": "engineer"}]}'

# Query by entity
curl 'http://localhost:3917/facts?entity=alice'

# When a contradicting fact is inserted, the old one is auto-superseded
curl -X POST http://localhost:3917/facts \
  -H 'Content-Type: application/json' \
  -d '{"facts": [{"subject": "alice", "predicate": "role", "object": "manager"}]}'
# Response includes {"resolved": 1} — the old "engineer" fact is now timestamped as superseded

# View fact history
curl 'http://localhost:3917/facts/history?subject=alice&predicate=role'
```

Facts are also extracted automatically when using `/extract` or the LLM proxy.

### Injection Protection

Memories get injected into LLM context windows. Malicious content stored as a memory could act as a prompt injection. engram protects against this:

- **On insert**: content is scored for injection risk. High-risk memories (score ≥ 0.7) are tagged `suspicious`.
- **On recall**: suspicious memories are automatically downranked in scoring.
- **On output**: special tokens (`<|im_start|>`, `[INST]`, `<<SYS>>`, etc.) are stripped from recall/resume responses.

Detection covers instruction overrides, role-play injection, model-specific control tokens, and XML/tag escaping attempts.

### Hybrid Search

Recall combines multiple signals:

- **Semantic search**: cosine similarity on embeddings (requires AI)
- **FTS5 keyword search**: BM25 with jieba segmentation for Chinese and bigram tokenization for Japanese/Korean
- **Fact lookup**: exact entity matching via the facts table
- **Composite scoring**: `(0.6 × relevance + 0.2 × importance + 0.2 × recency) × layer_bonus`
- **Keyword affinity**: semantic-only results that lack any query terms get a 0.7× penalty — mitigates embedding model weaknesses with short CJK queries
- **Stop word filtering**: common words (的、是、了、the、is、etc.) are excluded from FTS queries to reduce noise
- **FTS relevance floor**: keyword matches require minimum relevance to contribute to dual-hit boosting
- **Score cap**: final scores are capped at 1.0 to prevent threshold confusion from stacked bonuses

Results are deduplicated across all search modes. When both semantic and keyword search find the same memory, relevance gets a boost.

### LLM Proxy

Sit between your tools and any LLM API. Memories are extracted automatically — no code changes needed.

```bash
ENGRAM_PROXY_UPSTREAM=https://api.openai.com \
./target/release/engram

# Point your tools at engram:
# Before: https://api.openai.com/v1/chat/completions
# After:  http://localhost:3917/proxy/v1/chat/completions
```

How it works:
1. Forwards requests and headers to upstream verbatim
2. Streams the response back with zero added latency
3. After the exchange completes, asynchronously extracts key facts/decisions/preferences
4. Stores extracted memories in the buffer layer (source: `proxy`, tagged `auto-extract`)

The proxy is selective — routine task details are skipped. Only user preferences, concrete decisions, lessons learned, and identity-defining facts are extracted.

### Session Recovery

One call to restore agent context on wake-up:

```bash
# Full resume
curl http://localhost:3917/resume?hours=4

# Token-efficient (recommended for agents)
curl 'http://localhost:3917/resume?hours=4&compact=true&budget=8000'
```

Returns structured sections: `core` (identity/permanent), `working` (active context), `recent` (time-windowed), `sessions` (session notes), `next_actions` (tagged next-action).

`compact=true` strips metadata, `budget=N` caps total output characters with priority-based truncation (core first, buffer last).

### Namespace Isolation

Multi-agent support — each agent gets its own memory space:

```bash
curl -X POST http://localhost:3917/memories \
  -H 'X-Namespace: agent-alpha' \
  -H 'Content-Type: application/json' \
  -d '{"content": "agent-alpha private context"}'
```

### Triggers

Pre-action safety recall. Tag memories with `trigger:action-name`, then query before risky operations:

```bash
# Store a lesson
curl -X POST http://localhost:3917/memories \
  -H 'Content-Type: application/json' \
  -d '{"content": "never force-push to main", "tags": ["trigger:git-push"]}'

# Before pushing, check
curl http://localhost:3917/triggers/git-push
```

### Background Maintenance

engram runs autonomously — no cron or external scheduler needed:

- **Auto-consolidation** every 30 minutes: promotes active memories, decays neglected ones, reconciles buffer updates against Working/Core
- **Proxy debounce flush**: extracts memories from buffered conversations after 30 seconds of silence (not a fixed timer — waits for natural conversation pauses)
- **Auto-audit** every 24 hours: LLM reviews all Working/Core memories for quality, merges duplicates, demotes stale entries
- **Graceful shutdown**: flushes all pending proxy windows before exit

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health + endpoint list |
| `GET` | `/health` | Detailed health (uptime, RSS, cache stats) — same as `/` |
| `GET` | `/stats` | Layer counts |
| `POST` | `/memories` | Create memory |
| `POST` | `/memories/batch` | Batch create |
| `GET` | `/memories` | List (`?layer=N&tag=X&ns=X&limit=50&offset=0`) |
| `GET` | `/memories/:id` | Get by id (prefix match supported) |
| `PATCH` | `/memories/:id` | Update content/tags/importance/kind |
| `DELETE` | `/memories/:id` | Delete |
| `DELETE` | `/memories` | Batch delete (`{"ids": [...]}` or `{"namespace": "x"}`) |
| `GET` | `/trash` | List soft-deleted memories (`?limit=100&offset=0`) |
| `POST` | `/trash/:id/restore` | Restore a memory from trash |
| `DELETE` | `/trash` | Permanently purge all trash |
| `POST` | `/recall` | Hybrid search (semantic + keyword + facts) |
| `GET` | `/search` | Quick keyword search (`?q=term&limit=10`) |
| `GET` | `/recent` | Recent memories (`?hours=2&limit=20`) |
| `GET` | `/resume` | Session recovery (`?hours=4&compact=true&budget=8000`) |
| `GET` | `/triggers/:action` | Pre-action recall |
| `POST` | `/consolidate` | Maintenance cycle (`{"merge": true}` for LLM merge) |
| `POST` | `/audit` | LLM-powered memory reorganization (requires AI) |
| `POST` | `/sanitize` | Check text for injection risk (`{"content": "..."}`) |
| `POST` | `/extract` | LLM text → structured memories |
| `POST` | `/repair` | Fix FTS index + backfill embeddings |
| `POST` | `/vacuum` | Reclaim disk space |
| `GET` | `/export` | Export as JSON (`?embed=true` includes vectors) |
| `POST` | `/import` | Import memories (skips existing ids) |
| `POST` | `/facts` | Insert fact triples |
| `GET` | `/facts` | Query facts by entity (`?entity=alice`) |
| `GET` | `/facts/all` | List all facts |
| `GET` | `/facts/conflicts` | Check conflicts (`?subject=X&predicate=Y`) |
| `GET` | `/facts/history` | Fact history (`?subject=X&predicate=Y`) |
| `DELETE` | `/facts/:id` | Delete fact |
| `ANY` | `/proxy/*` | Transparent LLM proxy |
| `POST` | `/proxy/flush` | Flush buffered proxy conversations for extraction |
| `GET` | `/ui` | Web dashboard |

## MCP Tools

| Tool | Description |
|------|-------------|
| `engram_store` | Store a memory with optional importance, tags, kind |
| `engram_recall` | Hybrid search with budget, time filters, reranking |
| `engram_recent` | List recent memories by time |
| `engram_resume` | Full session recovery bootstrap |
| `engram_search` | Quick keyword search |
| `engram_extract` | LLM-powered text → memories |
| `engram_consolidate` | Run promotion/decay/merge cycle |
| `engram_stats` | Layer counts and status |
| `engram_health` | Detailed health check |
| `engram_triggers` | Pre-action safety recall |
| `engram_repair` | Fix indexes and backfill embeddings |
| `engram_delete` | Delete a memory by ID |
| `engram_update` | Update importance, tags, or layer |
| `engram_trash` | List soft-deleted memories |
| `engram_restore` | Restore a memory from trash |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_PORT` | `3917` | Server port |
| `ENGRAM_DB` | `engram.db` | SQLite database path |
| `ENGRAM_API_KEY` | — | Bearer token auth (optional) |
| `ENGRAM_LLM_URL` | — | Chat completions endpoint (enables AI features) |
| `ENGRAM_LLM_KEY` | — | LLM API key |
| `ENGRAM_LLM_MODEL` | `gpt-4o-mini` | Default LLM model |
| `ENGRAM_MERGE_MODEL` | *(LLM_MODEL)* | Model for consolidation merge |
| `ENGRAM_EXTRACT_MODEL` | *(LLM_MODEL)* | Model for `/extract` |
| `ENGRAM_RERANK_MODEL` | *(LLM_MODEL)* | Model for recall re-ranking |
| `ENGRAM_EXPAND_MODEL` | *(LLM_MODEL)* | Model for query expansion |
| `ENGRAM_PROXY_MODEL` | *(LLM_MODEL)* | Model for proxy extraction |
| `ENGRAM_GATE_MODEL` | *(LLM_MODEL)* | Model for Core promotion gate + audit |
| `ENGRAM_EMBED_URL` | *(from LLM_URL)* | Embeddings endpoint |
| `ENGRAM_EMBED_KEY` | *(LLM_KEY)* | Embeddings API key |
| `ENGRAM_EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `ENGRAM_CONSOLIDATE_MINS` | `30` | Auto-consolidation interval (0 = off) |
| `ENGRAM_AUTO_MERGE` | `false` | Enable LLM merge in auto-consolidation |
| `ENGRAM_PROXY_UPSTREAM` | — | Upstream LLM URL (enables proxy) |
| `ENGRAM_PROXY_KEY` | — | Fallback API key for proxy |
| `ENGRAM_WORKSPACE` | — | Default workspace tag for `/resume` |
| `ENGRAM_AUDIT_HOURS` | `24` | Background audit interval (0 = off) |
| `RUST_LOG` | `info` | Log level (`debug` for verbose) |

Per-component model overrides let you use cheap models for high-volume operations (proxy extraction, query expansion) while keeping stronger models for consolidation and audit.

## License

MIT
