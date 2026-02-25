# MCP Server

engram's MCP (Model Context Protocol) server lets Claude Code, Cursor, Windsurf, OpenClaw, and other MCP-compatible editors use engram as persistent memory.

## Quick Setup

Copy `mcp-config.json` from the repo root into your editor's MCP config:

```json
{
  "mcpServers": {
    "engram": {
      "command": "npx",
      "args": ["-y", "engram-mcp"],
      "env": {
        "ENGRAM_URL": "http://localhost:3917",
        "ENGRAM_API_KEY": "",
        "ENGRAM_NAMESPACE": ""
      }
    }
  }
}
```

Or run from source:

```bash
cd mcp && npm install && npm run build
ENGRAM_URL=http://localhost:3917 node dist/index.js
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_URL` | `http://localhost:3917` | engram server URL |
| `ENGRAM_API_KEY` | — | Bearer token auth |
| `ENGRAM_NAMESPACE` | — | Default namespace for project isolation |

## Tools

| Tool | HTTP Equivalent | Description |
|------|----------------|-------------|
| `engram_store` | `POST /memories` | Store a memory with optional importance, tags, kind |
| `engram_recall` | `POST /recall` | Hybrid search with budget, time filters, reranking |
| `engram_recent` | `GET /recent` | List recent memories by time window |
| `engram_resume` | `GET /resume` | Session recovery (core + working + recent + actions) |
| `engram_search` | `GET /search` | Quick keyword search |
| `engram_extract` | `POST /extract` | LLM-powered text → structured memories |
| `engram_consolidate` | `POST /consolidate` | Run promotion/decay/merge cycle |
| `engram_stats` | `GET /stats` | Layer counts and status |
| `engram_health` | `GET /health` | Health check (uptime, RSS, cache, integrity) |
| `engram_triggers` | `GET /triggers/:action` | Pre-action safety recall |
| `engram_repair` | `POST /repair` | Fix FTS index + backfill embeddings |
| `engram_delete` | `DELETE /memories/:id` | Delete a memory by ID |
| `engram_update` | `PATCH /memories/:id` | Update importance, tags, or layer |
| `engram_trash` | `GET /trash` | List soft-deleted memories |
| `engram_restore` | `POST /trash/:id/restore` | Restore from trash |
| `engram_facts` | `POST /facts` | Insert fact triples |

## Editor-Specific Setup

### Claude Code

```bash
claude mcp add -s user engram -- npx -y engram-mcp
```

Set env vars in `~/.claude/settings.json` under the engram server entry.

### Cursor / Windsurf

Add the `mcp-config.json` content to your workspace `.cursor/mcp.json` or equivalent config file.

### OpenClaw

Add to your workspace MCP config or use the HTTP API integration via [SETUP.md](SETUP.md).
