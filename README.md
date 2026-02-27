# engram

Persistent memory for AI agents — organized by time and space. Important memories get promoted, noise fades away, and related knowledge clusters into a browsable topic tree. All automatic.

<p align="center">
  <img src="docs/engram-quickstart.gif" alt="engram demo — store, context reset, recall" width="720">
</p>

[中文](README_CN.md)

## Quick Start

```bash
# Install and start
curl -fsSL https://raw.githubusercontent.com/kael-bit/engram-rs/main/install.sh | bash

# Store a memory
curl -X POST http://localhost:3917/memories \
  -d '{"content": "Always run tests before deploying", "tags": ["deploy"]}'

# Recall by meaning
curl -X POST http://localhost:3917/recall \
  -d '{"query": "deployment checklist"}'

# Restore full context (session start)
curl http://localhost:3917/resume
```

## What Makes engram Different

Most agent memory tools give you a vector store with search. engram adds a **lifecycle** — memories aren't just stored, they're managed over time.

### LLM-Powered Quality Gate

New memories land in Buffer. To reach Working or Core, they pass through an LLM quality gate that evaluates whether the content is worth keeping long-term. This isn't keyword matching — the LLM reads the memory in context and makes a judgment call.

```
Buffer → [LLM gate: "Is this a decision, lesson, or preference?"] → Working
Working → [sustained access + LLM gate] → Core (permanent)
```

### Semantic Dedup & Merge

When two memories say the same thing differently, engram detects and merges them:

```
Memory A: "use PostgreSQL for auth"
Memory B: "auth service runs on Postgres"
→ After consolidation: single merged memory with both contexts
```

The merge is LLM-powered — it understands meaning, not just string similarity.

### Automatic Decay

Memories that aren't accessed lose importance over time. But not all memories decay equally:

| Kind | Decay | Use case |
|------|-------|----------|
| `semantic` | Normal | Knowledge, preferences, decisions (default) |
| `episodic` | Normal | Events, experiences, time-bound context |
| `procedural` | Never | Workflows, instructions, how-to — persists indefinitely |

Working memories are never deleted — their importance can drop to zero, but they remain searchable. Buffer memories get evicted when they decay below threshold. Core memories are permanent.

### Self-Organizing Topic Tree

Vector clustering groups related memories automatically. The tree is hierarchical, with LLM-generated names:

```
Memory Architecture
├── Three-layer lifecycle [4]
├── Embedding pipeline [3]
└── Consolidation logic [5]
Deploy & Ops
├── CI/CD procedures [3]
└── Production incidents [2]
User Preferences [6]
```

The tree rebuilds automatically when memories change. On session start, the agent gets a topic index as a table of contents. Drill into any topic with `POST /topic {"ids": ["kb3"]}` to get the full memories in that cluster.

### Trigger System

Tag a memory with `trigger:deploy`, and it auto-surfaces when the agent checks `/triggers/deploy` before deploying. Lessons learned from past mistakes become guardrails for future actions.

```bash
# Store a lesson
curl -X POST http://localhost:3917/memories \
  -d '{"content": "LESSON: always backup DB before migration", "tags": ["trigger:deploy", "lesson"]}'

# Before deploying, agent checks:
curl http://localhost:3917/triggers/deploy
# → returns all memories tagged trigger:deploy, ranked by access count
```

## Architecture

Memory is organized along two dimensions — **time** and **space**:

```
         Time (lifecycle)                    Space (topic tree)
┌─────────────────────────────┐    ┌──────────────────────────────┐
│                             │    │ Auth Architecture            │
│  Buffer → Working → Core    │    │ ├── OAuth2 migration [3]     │
│    ↓         ↓        ↑     │    │ └── Token handling [2]       │
│  evict    decay    permanent│    │ Deploy & Ops                 │
│           only      + gate  │    │ ├── CI/CD procedures [3]     │
│                             │    │ └── Rollback lessons [2]     │
└─────────────────────────────┘    │ User Preferences [6]         │
                                   └──────────────────────────────┘
```

**Time** — a three-layer lifecycle inspired by the [Atkinson–Shiffrin memory model](https://en.wikipedia.org/wiki/Atkinson%E2%80%93Shiffrin_memory_model):

| Layer | Role | Behavior |
|-------|------|----------|
| **Buffer** | Short-term intake | All new memories land here. Unaccessed entries decay and get evicted |
| **Working** | Active knowledge | Promoted via repeated access or lesson/procedure tags. Never deleted — importance decays but memory persists |
| **Core** | Long-term identity | Promoted through sustained usage + LLM quality gate. Permanent |

**Space** — a self-organizing topic tree built from embedding vectors. Related memories cluster by semantic similarity, and an LLM names each cluster:

| Mechanism | Purpose |
|-----------|---------|
| **Vector clustering** | Groups semantically similar memories into topics by cosine similarity |
| **Hierarchy** | Related topics nest under shared parents, forming a multi-level tree |
| **LLM naming** | Automatically generates human-readable names for each cluster |
| **Auto-rebuild** | Tree updates automatically when memories change — no manual maintenance |

The problem topic trees solve: vector search requires you to guess the right query. Topic trees let the agent browse by subject — scan the directory first, then drill into the relevant branch.

## Session Recovery

One call restores full context after a restart or context compaction:

```
GET /resume →

=== Core (24) ===
deploy: test → build → stop → start (procedural)
LESSON: never force-push to main
...

=== Recent (12h) ===
[02-27 14:15] switched auth to OAuth2
[02-27 11:01] published API docs

=== Topics (Core: 24, Working: 57, Buffer: 7) ===
kb1: "Deploy Procedures" [5]
kb2: "Auth Architecture" [3]
kb3: "Memory Design" [8]
...

=== Triggers ===
deploy, git-push, database-migration
```

Four sections, each serving a different purpose:

| Section | What it gives you | Budget |
|---------|-------------------|--------|
| **Core** | Full text of permanent rules and identity — never truncated | ~2k tokens |
| **Recent** | Memories changed in the last 12 hours, for short-term continuity | ~1k tokens |
| **Topics** | Named topic index — the table of contents for all your memories | Leaf list |
| **Triggers** | Pre-action safety tags — auto-surface relevant lessons | Tag list |

The agent reads the topic index, spots something relevant, and drills in with `POST /topic`. This avoids dumping the entire memory store into context.

## Search & Retrieval

Hybrid retrieval: semantic embeddings + BM25 keyword search (with [jieba](https://github.com/messense/jieba-rs) for CJK). Results ranked by relevance, memory importance, and recency.

```bash
# Semantic search with budget control
curl -X POST http://localhost:3917/recall \
  -d '{"query": "how do we handle auth", "budget_tokens": 2000}'

# Pre-action safety check
curl http://localhost:3917/triggers/deploy

# Topic drill-down
curl -X POST http://localhost:3917/topic \
  -d '{"ids": ["kb3"]}'
```

## Background Maintenance

Fully autonomous. Activity-driven — skips cycles when there's been no write activity:

### Consolidation (every 30 minutes)

Each cycle runs these steps in order:

1. **Decay** — reduce importance of unaccessed memories
2. **Dedup** — detect and merge near-identical memories (cosine > 0.78)
3. **Triage** — LLM categorizes new Buffer memories for promotion
4. **Gate** — LLM evaluates promotion candidates (batch, single call)
5. **Reconcile** — LLM resolves ambiguous similar pairs, cached to avoid repeat calls
6. **Topic tree rebuild** — re-cluster and name any new or dirty topics

### Audit (every 12 hours)

Full-store review by LLM:

- Promote under-valued memories, demote stale ones
- Merge duplicates that escaped real-time dedup
- Adjust importance scores based on global context

The audit sees all Core and Working memories at once, so it can catch cross-topic redundancy that per-memory heuristics miss.

## Namespace Isolation

One engram instance, multiple projects. Each namespace gets its own isolated memory space:

```bash
# Project-specific memories
curl -X POST http://localhost:3917/memories \
  -H "X-Namespace: my-project" \
  -d '{"content": "API uses OAuth2 bearer tokens"}'

# Cross-project knowledge lives in the default namespace
curl -X POST http://localhost:3917/memories \
  -d '{"content": "Always use UTC for timestamps"}'
```

## Install

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/kael-bit/engram-rs/main/install.sh | bash

# Windows (PowerShell)
irm https://raw.githubusercontent.com/kael-bit/engram-rs/main/install.ps1 | iex
```

### Docker

```bash
docker run -d --name engram \
  -p 3917:3917 \
  -v engram-data:/data \
  -e ENGRAM_LLM_URL=https://api.openai.com/v1 \
  -e ENGRAM_LLM_KEY=sk-... \
  ghcr.io/kael-bit/engram-rs:latest
```

### LLM Configuration

engram works without an LLM — keyword search and heuristic lifecycle only. Add one to unlock smart features (triage, gate, merge, topic naming, audit):

```bash
export ENGRAM_LLM_URL=https://api.openai.com/v1
export ENGRAM_LLM_KEY=sk-...

# Or any OpenAI-compatible API (Ollama, LM Studio, etc.)
export ENGRAM_LLM_URL=http://localhost:11434/v1

# Anthropic native
export ENGRAM_LLM_PROVIDER=anthropic
export ENGRAM_LLM_KEY=sk-ant-...
```

Two-tier model configuration — use a strong model for judgment tasks (gate, audit) and a cheap model for text processing (merge, triage):

```bash
ENGRAM_GATE_MODEL=gpt-4o          # quality decisions
ENGRAM_LLM_MODEL=gpt-4o-mini     # text processing
```

### For AI Agents

Paste this into your agent's system prompt or session:

```
Set up engram (persistent memory) by following the guide at:
https://raw.githubusercontent.com/kael-bit/engram-rs/main/docs/SETUP.md
```

## Integration

Works with Claude Code, Cursor, Windsurf, OpenClaw, and any MCP-compatible tool.

17 MCP tools — see [MCP docs](docs/MCP.md). Full HTTP API — see [Setup guide](docs/SETUP.md).

```bash
# MCP (Claude Code)
npx engram-rs-mcp

# MCP (Cursor / Windsurf / generic)
# Add to your MCP config:
{"mcpServers": {"engram": {"command": "npx", "args": ["-y", "engram-rs-mcp"]}}}
```

### Web Dashboard

Built-in web UI at `http://localhost:3917` — browse memories, view the topic tree, monitor LLM usage, inspect consolidation history.

## Specs

| | |
|---|---|
| Binary size | ~10 MB |
| Memory usage | ~100 MB RSS in production |
| Storage | SQLite, no external database |
| Language | Rust |
| Platforms | Linux, macOS, Windows (x86_64 + aarch64) |
| License | MIT |

## License

MIT
