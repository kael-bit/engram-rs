# engram MCP server

MCP (Model Context Protocol) wrapper for [engram](../README.md), the hierarchical memory engine.

## Setup

```bash
cd mcp
npm install
npm run build
```

## Usage

The server communicates over stdio. Point it at a running engram instance via `ENGRAM_URL`:

```bash
ENGRAM_URL=http://localhost:3917 npm start
```

If `ENGRAM_URL` is not set, it defaults to `http://localhost:3917`.

## Claude Desktop / Cline config

Add to your MCP settings (e.g. `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "engram": {
      "command": "node",
      "args": ["/absolute/path/to/engram-rs/mcp/dist/index.js"],
      "env": {
        "ENGRAM_URL": "http://localhost:3917"
      }
    }
  }
}
```

## Tools

| Tool | Description |
|------|-------------|
| `engram_store` | Store a memory (content, importance, layer, tags) |
| `engram_recall` | Budget-aware hybrid search (query, budget_tokens, layers, etc.) |
| `engram_extract` | LLM-powered memory extraction from raw text |
| `engram_consolidate` | Run decay/promotion maintenance cycle |

## Requirements

- Node.js 18+
- A running engram server
