# engram

Persistent memory for AI agents — organized by time and space. Important memories get promoted, noise decays naturally, and related knowledge clusters into a browsable topic tree. Fully automatic.

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

## Core Features

Most agent memory tools provide a vector store with search. engram adds a **lifecycle** — memories are not just stored, they are continuously managed.

### LLM Quality Gate

New memories enter the Buffer layer. Promotion to Working or Core requires passing an LLM quality gate — the LLM evaluates each memory in context and determines whether it warrants long-term retention.

```
Buffer → [LLM gate: "Is this a decision, lesson, or preference?"] → Working
Working → [sustained access + LLM gate] → Core
```

### Semantic Dedup & Merge

When two memories express the same concept in different words, engram detects and merges them:

```
Memory A: "use PostgreSQL for auth"
Memory B: "auth service runs on Postgres"
→ After consolidation: single merged memory preserving both contexts
```

Merging is LLM-powered — based on semantic understanding, not string similarity.

### Automatic Decay

Decay is epoch-based — it only occurs during active consolidation cycles, not by wall-clock time. If the agent is idle for a week, memories remain intact.

| Kind | Decay rate | Use case |
|------|-----------|----------|
| `episodic` | Fast | Events, experiences, time-bound context |
| `semantic` | Slow | Knowledge, preferences, lessons (default) |
| `procedural` | Slowest | Workflows, instructions, how-to |

Working and Core memories are never deleted. In the Working layer, importance decreases gradually but memories remain searchable. Buffer serves as a temporary staging area where all kinds may be evicted.

### Self-Organizing Topic Tree

Vector clustering automatically groups related memories. The tree is hierarchical, with LLM-generated names:

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

The tree rebuilds automatically when memories change. At session start, the agent receives a topic index as a table of contents. Use `POST /topic {"ids": ["kb3"]}` to retrieve all memories within a specific cluster.

### Triggers

Tag a memory with `trigger:deploy`, and it surfaces automatically when the agent queries `/triggers/deploy` before executing a deployment.

```bash
# Store a lesson
curl -X POST http://localhost:3917/memories \
  -d '{"content": "LESSON: always backup DB before migration", "tags": ["trigger:deploy", "lesson"]}'

# Pre-deployment check
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
│  evict     decay    gate    │    │ Deploy & Ops                 │
│                             │    │ ├── CI/CD procedures [3]     │
│                             │    │ └── Rollback lessons [2]     │
└─────────────────────────────┘    │ User Preferences [6]         │
                                   └──────────────────────────────┘
```

**Time** — a three-layer lifecycle inspired by the [Atkinson–Shiffrin memory model](https://en.wikipedia.org/wiki/Atkinson%E2%80%93Shiffrin_memory_model):

| Layer | Role | Behavior |
|-------|------|----------|
| **Buffer** | Short-term staging | All new memories enter here. Evicted when they fall below threshold |
| **Working** | Active knowledge | Promoted by consolidation. Never deleted — importance decays at different rates by kind |
| **Core** | Long-term identity | Promoted through LLM quality gate. Never deleted |

**Space** — a self-organizing topic tree built from embedding vectors. Related memories cluster by semantic similarity, with LLM-generated names for each cluster:

| Mechanism | Description |
|-----------|-------------|
| **Vector clustering** | Groups semantically similar memories into topics via cosine similarity |
| **Hierarchy** | Related topics nest under shared parent nodes, forming a multi-level tree |
| **LLM naming** | Generates human-readable names for each cluster automatically |
| **Auto-rebuild** | Tree updates when memories change — no manual maintenance required |

Topic trees address a fundamental limitation of vector search: it requires the right query to find the right memory. Topic trees allow the agent to browse by subject — scan the directory, then drill into the relevant branch.

## Session Recovery

A single call restores full context, intended for session start or post-compaction recovery:

```
GET /resume →

=== Core (24) ===
deploy: test → build → stop → start (procedural)
LESSON: never force-push to main
...

=== Recent ===
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

Four sections, each serving a distinct purpose:

| Section | Content | Budget |
|---------|---------|--------|
| **Core** | Full text of permanent rules and identity — never truncated | ~2k tokens |
| **Recent** | Memories changed since last consolidation window, for short-term continuity | ~1k tokens |
| **Topics** | Named topic index — structured directory of all memories | Leaf list |
| **Triggers** | Pre-action safety tags for automatic lesson recall | Tag list |

The agent reads the topic index, identifies relevant topics, and drills in via `POST /topic` on demand. This avoids loading the entire memory store into context.

## Search & Retrieval

Hybrid retrieval combining semantic embeddings and BM25 keyword search (with [jieba](https://github.com/messense/jieba-rs) for CJK tokenization). Results are ranked by relevance, memory importance, and recency.

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

Fully autonomous and activity-driven — cycles are skipped when there has been no write activity:

### Consolidation (every 30 minutes)

Each cycle executes the following steps in order:

1. **Decay** — reduce importance of unaccessed memories
2. **Dedup** — detect and merge near-identical memories (cosine > 0.78)
3. **Triage** — LLM categorizes new Buffer memories for promotion
4. **Gate** — LLM evaluates promotion candidates (batched, single call)
5. **Reconcile** — LLM resolves ambiguous similar pairs; results are cached to avoid redundant calls
6. **Topic tree rebuild** — re-cluster and name new or changed topics

### Topic Distillation

When a topic cluster grows too large (10+ memories), engram condenses overlapping memories into fewer, richer entries — preserving all specific details while reducing redundancy. Up to 2 topics are distilled per consolidation cycle.

## Multi-Agent & Namespace Isolation

A single engram instance can serve multiple agents concurrently. SQLite WAL mode, a connection pool, and an `RwLock`-protected vector index make concurrent reads and writes safe out of the box.

Use the `X-Namespace` header to give each agent (or project) its own isolated memory space:

```bash
# Project-specific memories
curl -X POST http://localhost:3917/memories \
  -H "X-Namespace: my-project" \
  -d '{"content": "API uses OAuth2 bearer tokens"}'

# Cross-project knowledge in the default namespace
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

engram functions without an LLM — providing keyword search and rule-based lifecycle only. Adding an LLM enables intelligent features (triage, gate, merge, topic naming, audit):

```bash
export ENGRAM_LLM_URL=https://api.openai.com/v1
export ENGRAM_LLM_KEY=sk-...

# Any OpenAI-compatible API (Ollama, LM Studio, etc.)
export ENGRAM_LLM_URL=http://localhost:11434/v1

# Anthropic native
export ENGRAM_LLM_PROVIDER=anthropic
export ENGRAM_LLM_KEY=sk-ant-...
```

Two-tier model configuration — strong model for judgment tasks, lightweight model for text processing:

```bash
ENGRAM_GATE_MODEL=gpt-4o          # quality decisions
ENGRAM_LLM_MODEL=gpt-4o-mini     # text processing
```

### For AI Agents

Add this to your agent's system prompt or session:

```
Set up engram (persistent memory) by following the guide at:
https://raw.githubusercontent.com/kael-bit/engram-rs/main/docs/SETUP.md
```

## Integration

Compatible with Claude Code, Cursor, Windsurf, OpenClaw, and any MCP-compatible tool.

17 MCP tools — see [MCP docs](docs/MCP.md). Full HTTP API — see [Setup guide](docs/SETUP.md).

```bash
# MCP (Claude Code)
npx engram-rs-mcp

# MCP (Cursor / Windsurf / generic)
# Add to your MCP config:
{"mcpServers": {"engram": {"command": "npx", "args": ["-y", "engram-rs-mcp"]}}}
```

### Web Dashboard

Built-in web UI at `http://localhost:3917` for browsing memories, viewing the topic tree, monitoring LLM usage, and inspecting consolidation history.

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

<a href="https://glama.ai/mcp/servers/@kael-bit/engram-rs">
  <img width="380" height="200" src="https://glama.ai/mcp/servers/@kael-bit/engram-rs/badge" />
</a>
