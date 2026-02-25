# engram Setup Guide

You're an AI agent helping your user set up engram — a persistent memory system that gives you long-term memory across sessions.

After setup, you'll be able to store facts, recall context, and resume where you left off.

## Step 0: Ask the user

Before doing anything, ask:

**How do you want memory integration?**
- `mcp` — MCP tools (recommended for Claude Code and MCP-compatible editors)
- `http` — HTTP API via curl in your system prompt (works everywhere, no build step)

## Step 1: Install engram

```bash
# Linux / macOS
curl -L https://github.com/kael-bit/engram-rs/releases/latest/download/engram-linux-x86_64 -o engram
chmod +x engram
sudo mv engram /usr/local/bin/

# Windows (PowerShell)
Invoke-WebRequest -Uri https://github.com/kael-bit/engram-rs/releases/latest/download/engram-windows-x86_64.exe -OutFile engram.exe
```

<details>
<summary>Build from source (optional)</summary>

```bash
git clone https://github.com/kael-bit/engram-rs.git
cd engram-rs
cargo build --release
# Binary at ./target/release/engram (~9MB)
```
</details>

## Step 2: Start engram

```bash
# Minimal — just memory engine, no AI features
./engram
```

With LLM features, engram needs three classes of models:

| Role | What it does | Env var | Budget pick | Quality pick |
|------|-------------|---------|-------------|-------------|
| **Judgment** | Decides what memories are permanent (gate), reviews memory quality (audit) | `ENGRAM_GATE_MODEL`, `ENGRAM_AUDIT_MODEL` | Claude Haiku | Claude Sonnet |
| **Light judgment** | Extracts memories from conversations (proxy) | `ENGRAM_PROXY_MODEL` | GPT-4o-mini | Gemini Flash |
| **Text processing** | Merges text, expands queries, reranks (merge, rerank, expand) | `ENGRAM_LLM_MODEL` | GPT-4o-mini | GPT-4o-mini |

You can use one model for everything, or split by role to save cost:

```bash
# Single model (simple, works fine)
ENGRAM_LLM_URL=https://api.openai.com/v1/chat/completions \
ENGRAM_LLM_KEY=sk-xxx \
ENGRAM_EMBED_URL=https://api.openai.com/v1/embeddings \
ENGRAM_EMBED_KEY=sk-xxx \
./engram

# Split by role (recommended — saves cost, better quality where it matters)
ENGRAM_LLM_URL=https://api.openai.com/v1/chat/completions \
ENGRAM_LLM_KEY=sk-xxx \
ENGRAM_LLM_MODEL=gpt-4o-mini \
ENGRAM_GATE_MODEL=claude-sonnet-4-6 \
ENGRAM_PROXY_MODEL=gemini-3-flash \
ENGRAM_EMBED_URL=https://api.openai.com/v1/embeddings \
ENGRAM_EMBED_KEY=sk-xxx \
./engram
```

> **Why split?** The gate decides what gets promoted to permanent Core memory — it needs to distinguish "never force-push to main" (lesson, keep forever) from "fixed the build, 198 tests pass" (changelog, don't keep). Cheap models fail at this. Text merging and query expansion don't need that judgment.

```bash
# Optional: enable auth (for remote/shared deployments)
ENGRAM_API_KEY=your-secret-key ./engram
```

Engram is now running on `http://localhost:3917`.

Verify: `curl -s http://localhost:3917/health`


## Step 3: Configure your tool

There are two integration methods. Pick one:

| Method | How it works | Best for |
|--------|-------------|----------|
| **MCP** | Agent calls engram via MCP tool protocol | Claude Code, editors with MCP support |
| **HTTP** | Agent runs curl commands from system prompt | Everything else (OpenClaw, Codex, Cursor, etc.) |

---

### Option A: MCP

Build the MCP server:

```bash
cd engram/mcp && npm install && npm run build && cd ..
```

**Claude Code** — register via CLI or `.mcp.json`:

```bash
claude mcp add engram -- node /absolute/path/to/engram/mcp/dist/index.js
```

Or create `.mcp.json` in the project root:

```json
{
  "mcpServers": {
    "engram": {
      "command": "node",
      "args": ["/absolute/path/to/engram/mcp/dist/index.js"],
      "env": {
        "ENGRAM_URL": "http://localhost:3917"
      }
    }
  }
}
```

**Other MCP-compatible editors** — follow the editor's MCP server registration docs using the same command/args above.

Then add the prompt template below to your project instructions (`CLAUDE.md`, `AGENTS.md`, or equivalent):

````markdown
## Memory

You have persistent memory via engram MCP tools.

### Every session start or after context compaction
Restore context first. Read the core and working sections.
- Call `engram_resume` with hours=6, compact=true

**After compaction** (you see a `<summary>` block or "Compacted" system message): resume immediately, before responding to any user message. Compaction summaries are lossy.

### When to store what

| What | Tags | Kind | Example |
|------|------|------|---------|
| Design decisions | topic tags | *(default)* | "API uses REST, auth via Bearer token" |
| Lessons from mistakes | `lesson` + topic | *(default)* | "LESSON: never force-push to main" |
| Step-by-step workflows | `procedure` + topic | `procedural` | "Deploy: test → build → stop → cp → start" |
| User preferences | `preference` | *(default)* | "User prefers concise Chinese replies" |
| Session recap | `session` | *(default)* | "Did X, decided Y. Next: Z" |

**Don't store**: routine output, things already in code/config files, transient status.

### Storing memories
- Decisions, preferences, lessons → store immediately with topic tags
- Workflows, procedures → store with kind="procedural" (never decay)
- "Remember this" from the user → store with importance=0.9
- Lessons from mistakes → store with tags=["lesson", "trigger:action-name"]

How: `engram_store`

### Recalling memories
Before acting on any non-trivial task, recall first. Don't assume you remember — check.
Default recall is fast (~30ms cached). For short/vague queries, add expand=true (+1-2s).
Before risky operations, check triggers for relevant lessons.

Key parameters:
- `query` (required) — search query
- `limit` — max results (default 20)
- `budget_tokens` — token budget for results (default 2000, 0 = unlimited)
- `expand` — LLM query expansion for short/vague queries (+1-2s)
- `rerank` — LLM reranking for better ordering (+2-4s)
- `tags` — filter by tags
- `since` / `until` — ISO 8601 time range filter
- `min_score` — minimum relevance threshold (0.0-1.0)

How: `engram_recall` with the topic as query. `engram_triggers` for pre-action checks.

### Before session ends
Store what you did + what was decided + what to do next. Tag as "session".
Without the "next" part, you'll wake up not knowing what to do.
````

---

### Option B: HTTP

No build step. Add the prompt template below to your project instructions (`CLAUDE.md`, `AGENTS.md`, or equivalent):

````markdown
## Memory

You have persistent memory via engram at http://localhost:3917

### Every session start or after context compaction
Restore context first. Read the core and working sections.
```bash
curl -sf "http://localhost:3917/resume?hours=6&compact=true"
```

**After compaction** (you see a `<summary>` block or "Compacted" system message): resume immediately, before responding to any user message. Compaction summaries are lossy.

### When to store what

| What | Tags | Kind | Example |
|------|------|------|---------|
| Design decisions | topic tags | *(default)* | "API uses REST, auth via Bearer token" |
| Lessons from mistakes | `lesson` + topic | *(default)* | "LESSON: never force-push to main" |
| Step-by-step workflows | `procedure` + topic | `procedural` | "Deploy: test → build → stop → cp → start" |
| User preferences | `preference` | *(default)* | "User prefers concise Chinese replies" |
| Session recap | `session` | *(default)* | "Did X, decided Y. Next: Z" |

**Don't store**: routine output, things already in code/config files, transient status.

### Storing memories
- Decisions, preferences, lessons → store immediately with topic tags
- Workflows, procedures → store with kind="procedural" (never decay)
- "Remember this" from the user → store with importance=0.9
- Lessons from mistakes → store with tags=["lesson", "trigger:action-name"]

How:
```bash
curl -sf -X POST http://localhost:3917/memories \
  -H 'Content-Type: application/json' \
  -d '{"content": "...", "tags": ["topic"]}'

# Lessons with trigger
curl -sf -X POST http://localhost:3917/memories \
  -H 'Content-Type: application/json' \
  -d '{"content": "LESSON: never force-push to main", "tags": ["lesson","trigger:git-push"]}'

# Procedures (never decay)
curl -sf -X POST http://localhost:3917/memories \
  -H 'Content-Type: application/json' \
  -d '{"content": "deploy: test → build → stop → cp → start", "tags": ["deploy"], "kind": "procedural"}'
```

### Recalling memories
Before acting on any non-trivial task, recall first. Don't assume you remember — check.
Default recall is fast (~30ms cached). For short/vague queries, add expand=true (+1-2s).
Before risky operations, check triggers for relevant lessons.

Key parameters:
- `query` (required) — search query
- `limit` — max results (default 20)
- `budget_tokens` — token budget for results (default 2000, 0 = unlimited)
- `expand` — LLM query expansion for short/vague queries (+1-2s)
- `rerank` — LLM reranking for better ordering (+2-4s)
- `tags` — filter by tags
- `since` / `until` — ISO 8601 time range filter
- `min_score` — minimum relevance threshold (0.0-1.0)

How:
```bash
curl -sf -X POST http://localhost:3917/recall \
  -H 'Content-Type: application/json' \
  -d '{"query": "how do we deploy"}'

# Pre-action trigger check
curl -sf http://localhost:3917/triggers/deploy
```

### Before session ends
Store what you did + what was decided + what to do next. Tag as "session".
Without the "next" part, you'll wake up not knowing what to do.
````

---

## Extra: LLM Proxy (automatic capture)

On top of either setup, you can route API calls through engram's proxy to automatically extract memories from every conversation — no prompt changes needed.

```bash
# Start engram with proxy pointing at your real API
ENGRAM_PROXY_UPSTREAM=https://api.anthropic.com \
./engram

# Point your tools at engram:
# Before: https://api.anthropic.com/v1/messages
# After:  http://localhost:3917/proxy/v1/messages
```

The proxy forwards requests transparently, then asynchronously extracts facts and decisions into memory.

> **Note:** The proxy only captures memories passively. For the agent to **read** memories back (resume, recall), you still need either MCP or HTTP prompt instructions above.

Set `X-Engram-Extract: false` header on requests where you don't want extraction.

## Step 4: Verify

```bash
# Check engram is running
curl -s http://localhost:3917/health

# Store a test memory
curl -sf -X POST http://localhost:3917/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "Test memory from setup", "tags": ["test"]}'

# Recall it
curl -sf -X POST http://localhost:3917/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "test setup"}'
```

Start a new session in your tool. The agent should call resume at startup and see the test memory.

## Optional: Production (systemd)

```bash
sudo tee /etc/systemd/system/engram.service << 'EOF'
[Unit]
Description=engram memory engine
After=network.target

[Service]
ExecStart=/usr/local/bin/engram
# Optional: enable auth for remote/shared deployments
# Environment=ENGRAM_API_KEY=your-secret-key

# Embeddings
Environment=ENGRAM_EMBED_URL=https://api.openai.com/v1/embeddings
Environment=ENGRAM_EMBED_KEY=sk-xxx

# LLM — default model (text processing: merge, rerank, expand)
Environment=ENGRAM_LLM_URL=https://api.openai.com/v1/chat/completions
Environment=ENGRAM_LLM_KEY=sk-xxx
Environment=ENGRAM_LLM_MODEL=gpt-4o-mini

# Judgment — Core promotion gate + audit (needs strong model)
Environment=ENGRAM_GATE_MODEL=claude-sonnet-4-6

# Light judgment — proxy extraction (if using LLM proxy)
# Environment=ENGRAM_PROXY_MODEL=gemini-3-flash
# Environment=ENGRAM_PROXY_UPSTREAM=https://api.anthropic.com
# Environment=ENGRAM_PROXY_KEY=sk-xxx

Restart=always
RestartSec=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now engram
```

For multi-agent setups, add `X-Namespace: agent-name` header to all requests to isolate each agent's memory.
