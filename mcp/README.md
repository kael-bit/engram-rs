# engram-rs-mcp

MCP server for [engram](https://github.com/kael-bit/engram-rs) — persistent, brain-like memory for AI agents.

Works with Claude Code, Cursor, Windsurf, OpenClaw, and any MCP-compatible editor.

## Quick Setup

```json
{
  "mcpServers": {
    "engram": {
      "command": "npx",
      "args": ["-y", "engram-rs-mcp"],
      "env": {
        "ENGRAM_URL": "http://localhost:3917",
        "ENGRAM_API_KEY": "",
        "ENGRAM_NAMESPACE": ""
      }
    }
  }
}
```

### Claude Code

```bash
claude mcp add -s user engram -- npx -y engram-rs-mcp
```

### Cursor / Windsurf

Add the config above to `.cursor/mcp.json` or equivalent.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_URL` | `http://localhost:3917` | engram server URL |
| `ENGRAM_API_KEY` | — | Bearer token for auth |
| `ENGRAM_NAMESPACE` | — | Namespace for project isolation |

## Tools

| Tool | Description |
|------|-------------|
| `engram_store` | Store a memory with optional importance, tags, kind |
| `engram_recall` | Hybrid search with budget, time filters, reranking |
| `engram_resume` | Session recovery (core + working + recent + actions) |
| `engram_recent` | List recent memories by time window |
| `engram_search` | Quick keyword search |
| `engram_extract` | LLM-powered text → structured memories |
| `engram_consolidate` | Run promotion/decay/merge cycle |
| `engram_triggers` | Pre-action safety recall |
| `engram_stats` | Layer counts and status |
| `engram_health` | Health check |
| `engram_delete` | Delete a memory by ID |
| `engram_update` | Update importance, tags, or layer |
| `engram_trash` | List soft-deleted memories |
| `engram_restore` | Restore from trash |
| `engram_facts` | Insert fact triples |

## What is engram?

Persistent memory that works like a brain — three-layer model (Buffer → Working → Core) with automatic decay, promotion, dedup, and LLM-powered quality gates. Single binary, ~9 MB.

See the [main repo](https://github.com/kael-bit/engram-rs) for full documentation.

## License

MIT
