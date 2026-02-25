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

With LLM features, engram needs two classes of models:

| Role | What it does | Env var | Budget pick | Quality pick |
|------|-------------|---------|-------------|-------------|
| **Judgment** | Decides what memories are permanent (gate), reviews memory quality (audit) | `ENGRAM_GATE_MODEL`, `ENGRAM_AUDIT_MODEL` | Claude Haiku | Claude Sonnet |
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
ENGRAM_EMBED_URL=https://api.openai.com/v1/embeddings \
ENGRAM_EMBED_KEY=sk-xxx \
./engram
```

> **Why split?** The gate decides what gets promoted to permanent Core memory — it needs to distinguish "never force-push to main" (lesson, keep forever) from "fixed the build, 198 tests pass" (changelog, don't keep). Cheap models fail at this. Text merging and query expansion don't need that judgment.

**LLM usage level:** Set `ENGRAM_LLM_LEVEL` to control how much consolidation relies on LLMs:
- `auto` (default) — heuristics handle obvious cases, LLM only for uncertain ones
- `full` — always use LLM for triage and gate decisions
- `off` — zero LLM calls in consolidation (pure heuristics, merge/reconcile skipped)

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
        "ENGRAM_URL": "http://localhost:3917",
        "ENGRAM_NAMESPACE": "my-project"
      }
    }
  }
}
```

**Other MCP-compatible editors** — follow the editor's MCP server registration docs using the same command/args above.

Then add the prompt template below to your project instructions (`CLAUDE.md`, `AGENTS.md`, or equivalent):

````markdown
## Memory System

You have persistent memory via engram MCP tools.

### 1. Context Restoration (Tool: `engram_resume`)

**Trigger:** Every session start OR after context compaction.

- Call `engram_resume` with hours=6 and compact=true.
- **Rule:** If you see a `<summary>` block or "Compacted" system message, you MUST resume immediately, before responding to any user message. Compaction summaries are lossy; read the core and working sections returned by the tool.

### 2. When & What to Store (Tool: `engram_store`)

**Default: store it.** If information came from the user (not from command output), store it. Engram handles dedup and decay — you won't create clutter. The cost of forgetting is always higher than the cost of storing.

🚫 **DO NOT store:** routine command output (build logs, test results, ls listings), step-by-step narration of what you just did, or information already in code/config files.

**Everything else — store immediately.** Especially:
- Anything the user says about themselves (name, role, timezone, background)
- Preferences, even casual ones ("I like concise answers")
- Corrections and criticism
- Decisions, even small ones ("let's use X")
- Rules and constraints ("always do Y before Z")
- Lessons from mistakes (tag: `lesson` + `trigger:kebab-verb`)
- Workflows / ordering constraints (kind: `procedural`)
- Milestone recaps: decisions made + what to do next (tag: `session`)

🌐 **Cross-Project Knowledge**

Memories default to your project namespace. For knowledge that applies across ALL projects (user identity, preferences, universal lessons), store to namespace `"default"` explicitly: `engram_store(content="...", namespace="default")`.

🔄 **Memory Reinforcement (Repetition is Good!)**

The engram system uses repetition to strengthen memory weights. If an important rule, user preference, or lesson comes up again — store it again. Do not skip storing a critical insight just because you think "I already saved this before." Engram handles dedup and reinforcement automatically.

⚠️ **HIGHEST PRIORITY: User Feedback**

Corrections, criticism, preferences, and rules from the user are NEVER automatically captured. You must store them yourself immediately.

**CRITICAL RULE:** When the user corrects you, criticizes, or states a new rule → call `engram_store` FIRST, then reply. No store = no reply.

**Storage modifiers:**
- Workflows/procedures → set kind="procedural" (never decay).
- "Remember this" from user → set importance=0.9.
- Lessons from mistakes → include tags like `["lesson", "trigger:kebab-case-verb"]` (e.g., `trigger:git-push`, `trigger:deploy`).

### 3. Recalling Memories (Tools: `engram_recall` & `engram_triggers`)

Before acting on any non-trivial task, you **MUST** check your memory. Do not assume you remember.

- **Pre-action check:** Before risky operations (e.g., deploying, pushing code), call `engram_triggers` to check for relevant lessons.
- **General search:** Call `engram_recall` with the topic as the query.
  - `query` (required): Search string.
  - `expand` (boolean): Set to true for short/vague queries (+1-2s).
  - `limit` (default 20), `budget_tokens` (default 2000), `tags`, `min_score`.

### 4. Milestone Recaps

You cannot predict when a session ends. Instead, proactively store a recap immediately after completing any major task or reaching a decision point.

- **Content:** Decisions made + Lessons learned + What to do next. Skip operational details ("deployed X", "ran Y") — only store *why* and *what's next*.
- **Tags:** Include `session`.
- *Note: Without the "what to do next" part, you will wake up in the next session not knowing what to do.*
````

---

### Option B: HTTP

No build step. Add the prompt template below to your project instructions (`CLAUDE.md`, `AGENTS.md`, or equivalent):

````markdown
## Memory System

You have persistent memory via engram at http://localhost:3917

### 1. Context Restoration

**Trigger:** Every session start OR after context compaction.

```bash
curl -sf "http://localhost:3917/resume?hours=6&compact=true"
```

**Rule:** If you see a `<summary>` block or "Compacted" system message, you MUST resume immediately, before responding to any user message. Compaction summaries are lossy; read the core and working sections returned.

### 2. When & What to Store

**Default: store it.** If information came from the user (not from command output), store it. Engram handles dedup and decay — you won't create clutter. The cost of forgetting is always higher than the cost of storing.

🚫 **DO NOT store:** routine command output (build logs, test results, ls listings), step-by-step narration of what you just did, or information already in code/config files.

**Everything else — store immediately.** Especially:
- Anything the user says about themselves (name, role, timezone, background)
- Preferences, even casual ones ("I like concise answers")
- Corrections and criticism
- Decisions, even small ones ("let's use X")
- Rules and constraints ("always do Y before Z")
- Lessons from mistakes (tag: `lesson` + `trigger:kebab-verb`)
- Workflows / ordering constraints (kind: `procedural`)
- Milestone recaps: decisions made + what to do next (tag: `session`)

🌐 **Cross-Project Knowledge**

Memories default to your project namespace. For knowledge that applies across ALL projects (user identity, preferences, universal lessons), store to namespace `"default"` explicitly:

```bash
curl -sf -X POST http://localhost:3917/memories -H "X-Namespace: default" \\
  -d '{"content": "User prefers concise replies in Chinese"}'
```

🔄 **Memory Reinforcement (Repetition is Good!)**

The engram system uses repetition to strengthen memory weights. If an important rule, user preference, or lesson comes up again — store it again. Do not skip storing a critical insight just because you think "I already saved this before." Engram handles dedup and reinforcement automatically.

⚠️ **HIGHEST PRIORITY: User Feedback**

Corrections, criticism, preferences, and rules from the user are NEVER automatically captured. You must store them yourself immediately.

**CRITICAL RULE:** When the user corrects you, criticizes, or states a new rule → store the lesson FIRST, then reply. No store = no reply.

**Storage modifiers:**
- Workflows/procedures → set kind="procedural" (never decay).
- "Remember this" from user → set importance=0.9.
- Lessons from mistakes → include tags like `["lesson", "trigger:kebab-case-verb"]` (e.g., `trigger:git-push`, `trigger:deploy`).

```bash
curl -sf -X POST http://localhost:3917/memories \
  -d '{"content": "...", "tags": ["topic"]}'

# Procedures (never decay)
curl -sf -X POST http://localhost:3917/memories \
  -d '{"content": "deploy: test → build → stop → start", "tags": ["deploy"], "kind": "procedural"}'

# Lessons with trigger
curl -sf -X POST http://localhost:3917/memories \
  -d '{"content": "LESSON: never force-push to main", "tags": ["lesson","trigger:git-push"]}'
```

### 3. Recalling Memories

Before acting on any non-trivial task, you **MUST** check your memory. Do not assume you remember.

- **Pre-action check:** Before risky operations (e.g., deploying, pushing code), check triggers for relevant lessons.
- **General search:** Query with the topic.
  - `query` (required): Search string.
  - `expand` (boolean): Set to true for short/vague queries (+1-2s).
  - `limit` (default 20), `budget_tokens` (default 2000), `tags`, `min_score`.

```bash
curl -sf -X POST http://localhost:3917/recall \
  -d '{"query": "how do we deploy"}'

# Pre-action trigger check
curl -sf http://localhost:3917/triggers/deploy
```

### 4. Milestone Recaps

You cannot predict when a session ends. Instead, proactively store a recap immediately after completing any major task or reaching a decision point.

- **Content:** Decisions made + Lessons learned + What to do next. Skip operational details ("deployed X", "ran Y") — only store *why* and *what's next*.
- **Tags:** Include `session`.
- *Note: Without the "what to do next" part, you will wake up in the next session not knowing what to do.*
````

---

## Step 4: Verify

```bash
# Check engram is running
curl -s http://localhost:3917/health

# Store a test memory
curl -sf -X POST http://localhost:3917/memories \
  -d '{"content": "Test memory from setup", "tags": ["test"]}'

# Recall it
curl -sf -X POST http://localhost:3917/recall \
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

Restart=always
RestartSec=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now engram
```

## Project Isolation (Namespace)

Set `ENGRAM_NAMESPACE` to isolate memories per project. One engram instance, multiple workspaces:

- **MCP**: Set `ENGRAM_NAMESPACE` in `.mcp.json` env (see Step 3 above)
- **HTTP**: Set env var `ENGRAM_NAMESPACE=my-project` or pass `-H "X-Namespace: my-project"` on all requests

Resume and recall automatically include the `default` namespace alongside your project namespace, so cross-project knowledge (user identity, preferences, universal lessons) is always available.

To store cross-project knowledge explicitly, override the namespace to `default`:
- **MCP**: `engram_store(content="...", namespace="default")`
- **HTTP**: `curl -X POST ... -H "X-Namespace: default" -d '{"content":"..."}'`

Consolidation merge is directional: project memories can be absorbed into `default`, but `default` memories are never pulled into a project namespace.
