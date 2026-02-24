# engram Setup Guide

You're an AI agent helping your user set up engram — a persistent memory system that gives you long-term memory across sessions.

After setup, you'll be able to store facts, recall context, and resume where you left off.

## Step 0: Ask the user

Before doing anything, ask:

1. **Which coding tool are you using?**
   - `claude-code` — Claude Code (Anthropic)
   - `openclaw` — OpenClaw
   - `codex` — OpenAI Codex CLI
   - `other` — Cursor, Windsurf, etc.

2. **How do you want memory integration?**
   - `mcp` — MCP tools (recommended for Claude Code — explicit store/recall via tool calls)
   - `http` — HTTP API via curl in your system prompt (works everywhere, no setup)

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
# Minimal — just memory engine
ENGRAM_API_KEY=your-secret-key ./engram

# With LLM features (consolidation, extraction, reranking)
ENGRAM_API_KEY=your-secret-key \
ENGRAM_LLM_URL=https://api.openai.com/v1/chat/completions \
ENGRAM_LLM_KEY=sk-xxx \
ENGRAM_EMBED_URL=https://api.openai.com/v1/embeddings \
ENGRAM_EMBED_KEY=sk-xxx \
./engram
```

Engram is now running on `http://localhost:3917`.

Verify: `curl -s http://localhost:3917/health`

## Step 3: Configure your tool

Follow the section matching the user's answers from Step 0.

---

### Claude Code + MCP

Build the MCP server:

```bash
cd engram/mcp && npm install && npm run build && cd ..
```

Register with Claude Code:

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
        "ENGRAM_URL": "http://localhost:3917",
        "ENGRAM_API_KEY": "your-secret-key"
      }
    }
  }
}
```

Then add to the project's `CLAUDE.md`:

````markdown
## Memory

You have persistent memory via engram MCP tools.

### Every session start or after context compaction
Call `engram_resume` with hours=6 to restore context. Read the core and working sections — that's who you are and what you were doing.

Compaction summaries are lossy — they drop details, lessons, and recent decisions.
After any compaction event, immediately call `engram_resume` before continuing.

### When to store what

| What | Tags | Kind | Example |
|------|------|------|---------|
| Design decisions | topic tags | *(default)* | "API uses REST, auth via Bearer token" |
| Lessons from mistakes | `lesson` + topic | *(default)* | "LESSON: never force-push to main" |
| Step-by-step workflows | `procedure` + topic | `procedural` | "Deploy: test → build → stop → cp → start" |
| User preferences | `preference` | *(default)* | "User prefers concise Chinese replies" |
| Session recap | `session` | *(default)* | "Did X, decided Y. Next: Z" |

**Don't store**: routine output ("tests passed"), things in code files, transient status.

### During conversation
- Decisions, preferences, lessons → `engram_store` immediately
- Workflows, procedures → `engram_store` with kind="procedural" (never decay)
- "Remember this" from the user → `engram_store` with importance=0.9
- Before risky operations → `engram_triggers` with the action name

### Recalling memories
Before acting on any non-trivial task, recall first. Don't assume you remember — check:
- `engram_recall` with the topic as query

Recall when:
- Starting work on a topic you've touched before
- The user asks about past decisions, preferences, or history
- Before making architectural or design decisions
- You're unsure about a convention or workflow

### Before session ends
Store a summary: what you did + what was decided + what to do next.
Tag as "session". Without the "next" part, you'll wake up lost.
````

**Done.** The agent now has persistent memory via MCP tool calls.

---

### Claude Code + HTTP (no MCP)

No build step needed. Add to the project's `CLAUDE.md`:

````markdown
## Memory

You have persistent memory via engram at http://localhost:3917
Auth: Bearer YOUR_API_KEY

### Every session start or after context compaction (DO THIS FIRST)
```bash
curl -sf -H "Authorization: Bearer YOUR_API_KEY" "http://localhost:3917/resume?hours=6&compact=true"
```

Compaction summaries are lossy. After any compaction event, call resume again to restore details.

### When to store what

| What | Tags | Kind | Example |
|------|------|------|---------|
| Design decisions | topic tags | *(default)* | "API uses REST, auth via Bearer token" |
| Lessons from mistakes | `lesson` + topic | *(default)* | "LESSON: never force-push to main" |
| Step-by-step workflows | `procedure` + topic | `procedural` | "Deploy: test → build → stop → cp → start" |
| User preferences | `preference` | *(default)* | "User prefers concise Chinese replies" |
| Session recap | `session` | *(default)* | "Did X, decided Y. Next: Z" |

**Don't store**: routine output, things in code files, transient status.

### Recalling memories
Before acting on any non-trivial task, recall first. Don't assume you remember — check:
```bash
curl -sf -X POST -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" -d '{"query": "your question"}' http://localhost:3917/recall
```
Recall when:
- Starting work on a topic you've touched before
- The user asks about past decisions, preferences, or history
- Before making architectural or design decisions
- You're unsure about a convention or workflow

### Store
```bash
# Fact or decision
curl -sf -X POST -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" -d '{"content": "...", "tags": ["topic"]}' http://localhost:3917/memories

# Lesson (survives longer)
curl -sf -X POST -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" -d '{"content": "LESSON: ...", "tags": ["lesson", "topic"]}' http://localhost:3917/memories

# Procedure (never expires)
curl -sf -X POST -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" -d '{"content": "Steps: 1. ... 2. ...", "tags": ["procedure"], "kind": "procedural"}' http://localhost:3917/memories
```

### Before risky operations
```bash
curl -sf -H "Authorization: Bearer YOUR_API_KEY" http://localhost:3917/triggers/deploy
```

### End of session
Store what you did + what to do next. Tag as "session".
````

**Done.** The agent uses curl to talk to engram directly.

---

### OpenClaw

Add to `AGENTS.md` in the workspace:

````markdown
## Memory

engram is your memory. Without it, every session starts from zero.

### Every wake-up / heartbeat / post-compaction
Restore context first — nothing else matters until you know who you are:
  curl -s http://localhost:3917/resume?hours=6&compact=true

Read the core section (identity, constraints, lessons) and working section (active context).
If there are next_actions, that's your todo list.

Compaction summaries are lossy — they drop details, lessons, and recent decisions.
After any compaction event, resume from engram before continuing the conversation.

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
```bash
# Important facts, decisions, lessons
curl -X POST http://localhost:3917/memories \
  -H 'Content-Type: application/json' \
  -d '{"content": "user prefers direct answers, no filler", "tags": ["preference"]}'

# Lessons from mistakes (with trigger for pre-action recall)
curl -X POST http://localhost:3917/memories \
  -H 'Content-Type: application/json' \
  -d '{"content": "never force-push to main", "tags": ["lesson","trigger:git-push"]}'

# Workflows and procedures (these never decay)
curl -X POST http://localhost:3917/memories \
  -H 'Content-Type: application/json' \
  -d '{"content": "deploy flow: cargo test && cargo build --release && systemctl stop && cp && systemctl start", "tags": ["deploy"], "kind": "procedural"}'
```

### Recalling memories
Before acting on any non-trivial task, recall first. Don't assume you remember — check:
```bash
curl -sf -X POST http://localhost:3917/recall \
  -H 'Content-Type: application/json' \
  -d '{"query": "how do we deploy"}'
```
Recall when:
- Starting work on a topic you've touched before
- The user asks about past decisions, preferences, or history
- Before making architectural or design decisions
- You're unsure about a convention or workflow

### Before risky operations
```bash
curl -s http://localhost:3917/triggers/git-push
curl -s http://localhost:3917/triggers/deploy
```

### Before session ends / compaction
Store what you did and what comes next:
```bash
curl -X POST http://localhost:3917/memories \
  -H 'Content-Type: application/json' \
  -d '{"content": "Session: refactored auth module, decided to use JWT. Next: write integration tests.", "tags": ["session"]}'
```

Without the "next" part, you'll wake up not knowing what to do.
````

**Done.** OpenClaw agents use curl with heartbeat-driven memory cycles.

---

### Codex / Other Editors

Use the HTTP approach. Add the curl-based instructions from **Claude Code + HTTP** above to your editor's system prompt or project instructions file.

For multi-agent setups, add `X-Namespace: agent-name` header to isolate each agent's memory.

---

## Extra: LLM Proxy (automatic capture)

On top of any setup above, you can route API calls through engram's proxy to automatically extract memories from every conversation — no prompt changes needed.

```bash
# Start engram with proxy pointing at your real API
ENGRAM_PROXY_UPSTREAM=https://api.anthropic.com \
ENGRAM_LLM_URL=https://api.openai.com/v1/chat/completions \
ENGRAM_LLM_KEY=sk-xxx \
./target/release/engram

# Tell your tool to use engram as its API endpoint
ANTHROPIC_BASE_URL=http://localhost:3917/proxy claude  # Claude Code
```

The proxy forwards requests transparently (no latency), then asynchronously extracts facts, decisions, and preferences into memory.

> **Note:** The proxy only captures memories passively. For the agent to **read** memories back (resume, recall), you still need either MCP or HTTP prompt instructions above.

Set `X-Engram-Extract: false` header on requests where you don't want extraction (e.g., sub-agent traffic).

## Step 4: Verify

After setup, test the integration:

```bash
# Check engram is running
curl -s http://localhost:3917/health

# Store a test memory
curl -sf -X POST -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content": "Test memory from setup", "tags": ["test"]}' \
  http://localhost:3917/memories

# Recall it
curl -sf -X POST -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "test setup"}' \
  http://localhost:3917/recall
```

Then start a new session in your tool. The agent should call `resume` at startup and see the test memory.

## Optional: Production (systemd)

```bash
sudo tee /etc/systemd/system/engram.service << 'EOF'
[Unit]
Description=engram memory engine
After=network.target

[Service]
ExecStart=/usr/local/bin/engram
Environment=ENGRAM_API_KEY=your-secret-key
Environment=ENGRAM_LLM_URL=https://api.openai.com/v1/chat/completions
Environment=ENGRAM_LLM_KEY=sk-xxx
Environment=ENGRAM_EMBED_URL=https://api.openai.com/v1/embeddings
Environment=ENGRAM_EMBED_KEY=sk-xxx
Restart=always
RestartSec=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now engram
```
