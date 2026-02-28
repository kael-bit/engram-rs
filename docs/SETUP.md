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

engram **requires** an embedding provider — without it, semantic recall doesn't work and the server will refuse to start. At minimum, set `ENGRAM_EMBED_URL`:

```bash
# Minimal — embedding only, no LLM features (dedup, consolidation, gating)
ENGRAM_EMBED_URL=https://api.openai.com/v1/embeddings \
ENGRAM_EMBED_KEY=sk-xxx \
./engram
```

With LLM features, engram needs two classes of models:

| Role | What it does | Env var | Budget pick | Quality pick |
|------|-------------|---------|-------------|-------------|
| **Judgment** | Decides what memories are permanent (gate), reviews memory quality (audit) | `ENGRAM_GATE_MODEL`, `ENGRAM_AUDIT_MODEL` | Claude Haiku | Claude Sonnet |
| **Text processing** | Merges text, expands queries (merge, expand) | `ENGRAM_LLM_MODEL` | GPT-4o-mini | GPT-4o-mini |

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

# Anthropic native API (no OpenAI-compatible proxy needed)
ENGRAM_LLM_URL=https://api.anthropic.com/v1/messages \
ENGRAM_LLM_KEY=sk-ant-xxx \
ENGRAM_LLM_MODEL=claude-sonnet-4-6-20250514 \
ENGRAM_LLM_PROVIDER=anthropic \
ENGRAM_EMBED_URL=https://api.openai.com/v1/embeddings \
ENGRAM_EMBED_KEY=sk-xxx \
./engram

# Mix providers per component (no proxy needed!)
# Text processing → OpenAI (cheap), judgment → Anthropic (quality)
ENGRAM_LLM_URL=https://api.openai.com/v1/chat/completions \
ENGRAM_LLM_KEY=sk-xxx \
ENGRAM_LLM_MODEL=gpt-4o-mini \
ENGRAM_GATE_URL=https://api.anthropic.com/v1/messages \
ENGRAM_GATE_KEY=sk-ant-xxx \
ENGRAM_GATE_MODEL=claude-sonnet-4-6-20250514 \
ENGRAM_GATE_PROVIDER=anthropic \
ENGRAM_AUDIT_URL=https://api.anthropic.com/v1/messages \
ENGRAM_AUDIT_KEY=sk-ant-xxx \
ENGRAM_AUDIT_MODEL=claude-sonnet-4-6-20250514 \
ENGRAM_AUDIT_PROVIDER=anthropic \
ENGRAM_EMBED_URL=https://api.openai.com/v1/embeddings \
ENGRAM_EMBED_KEY=sk-xxx \
./engram
```

Every component (`gate`, `audit`, `merge`, `extract`, `expand`, `proxy`, `triage`, `summary`) supports `_URL`, `_KEY`, `_MODEL`, `_PROVIDER` overrides. Unset fields fall back to `ENGRAM_LLM_*` globals.

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

### 1. Context Restoration — ⚠️ MANDATORY FIRST ACTION (Tool: `engram_resume`)

**Your VERY FIRST tool call in every session must be `engram_resume`.** No exceptions. Do this before reading files, before answering the user, before anything else.

**Why:** Resume returns your topic index — the ONLY way to access your Working memories. Without it you are operating blind. A compaction summary is NOT a substitute; it lacks your topic index and cannot be drilled into.

**Resume returns 4 sections:**
- **Core** — permanent rules/identity (full text, never truncated)
- **Recent** — recently changed memories for short-term continuity
- **Topics** — a named topic index of ALL your memories (e.g. `kb1: "Deploy procedures" [5]`). Use `engram_topic` with IDs to drill into any topic. **This is your memory's table of contents.**
- **Triggers** — pre-action safety tags for reflex recall

### 2. When & What to Store (Tool: `engram_store`)

**🧠 Store everything valuable, store it immediately, never batch.** Engram handles dedup and decay automatically — over-storing costs nothing, forgetting costs everything.

**⚠️ CRITICAL:** When the user corrects you or sets a rule → store FIRST, then reply. When a conversation produces a decision, discovery, lesson, or strategy insight → store it in your very next tool call. If you think "I'll store this later", you're already wrong. If the same info comes up again, store it again — repetition strengthens memory weight.

✅ **Store:** user identity, preferences, decisions, constraints, lessons, milestone recaps ("what was decided" + "what to do next"). Use `episodic` for time-bound events, `semantic` for lasting knowledge, `procedural` for permanent workflows.
**🚫 Don't store:** command output, step-by-step narration, info already in code/config files.

**Tags:** Name the SUBJECT, not meta-properties. Good: `deploy-flow`, `ssh-config`, `competitor-mem0`. Bad: `session`, `distilled`, `engram-rs` (too generic). Trigger tags (`trigger:git-push`) are the exception — they enable reflex recall.

**Kind:**
- `semantic` (default) — facts, preferences, lessons, constraints, strategies. **When in doubt, use this.**
- `episodic` — specific dated events: "decided to use X", "v1.0 released", "found bug in Y"
- `procedural` — reusable step-by-step workflows ONLY (deploy steps, build process). **NOT:** strategies, plans, guidelines, one-time decisions. If it has an end condition ("until X", "for now"), it's NOT procedural.

**Other modifiers:**
- Lessons → `tags=["lesson", "trigger:kebab-verb"]` (e.g. `trigger:git-push`, `trigger:deploy-app`)
- Cross-project knowledge → `namespace="default"`

### 3. Recalling Memories (Tools: `engram_recall` & `engram_triggers` & `engram_topic`)

Before acting on any non-trivial task, you **MUST** check your memory. Do not assume you remember.

- **Pre-action check:** Before risky operations (e.g., deploying, pushing code), call `engram_triggers` to check for relevant lessons.
- **General search:** Call `engram_recall` with the topic as the query.
  - `query` (required): Search string.
  - `expand` (boolean): Set to true for short/vague queries (+1-2s).
  - `limit` (default 20), `budget_tokens` (default 2000), `tags`, `min_score` (default 0.30; set to 0.0 for all results).
- **Topic drill-down:** Resume includes a topic index (e.g. `kb3: "Memory architecture" [8]`). To explore a topic's full memories, call `engram_topic` with `ids=["kb3"]`. Use this when you need all memories about a subject, not just the top search results.

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

### 1. Context Restoration — ⚠️ MANDATORY FIRST ACTION

**Your VERY FIRST tool call in every session must be `curl -sf http://localhost:3917/resume`.** No exceptions. Do this before reading files, before answering the user, before anything else.

**Why:** Resume returns your topic index — the ONLY way to access your Working memories. Without it you are operating blind. A compaction summary is NOT a substitute; it lacks your topic index and cannot be drilled into.

**Resume returns 4 sections:**
- **Core** — permanent rules/identity (full text, never truncated)
- **Recent** — recently changed memories for short-term continuity
- **Topics** — a named topic index of ALL your memories (e.g. `kb1: "Deploy procedures" [5]`). Use `POST /topic {"ids":["kb1"]}` to drill into any topic. **This is your memory's table of contents.**
- **Triggers** — pre-action safety tags for reflex recall

### 2. When & What to Store

**🧠 Store everything valuable, store it immediately, never batch.** Engram handles dedup and decay automatically — over-storing costs nothing, forgetting costs everything.

**⚠️ CRITICAL:** When the user corrects you or sets a rule → store FIRST, then reply. When a conversation produces a decision, discovery, lesson, or strategy insight → store it in your very next tool call. If you think "I'll store this later", you're already wrong. If the same info comes up again, store it again — repetition strengthens memory weight.

✅ **Store:** user identity, preferences, decisions, constraints, lessons, milestone recaps ("what was decided" + "what to do next"). Use `episodic` for time-bound events, `semantic` for lasting knowledge, `procedural` for permanent workflows.
**🚫 Don't store:** command output, step-by-step narration, info already in code/config files.

**Tags:** Name the SUBJECT, not meta-properties. Good: `deploy-flow`, `ssh-config`, `competitor-mem0`. Bad: `session`, `distilled`, `engram-rs` (too generic). Trigger tags (`trigger:git-push`) are the exception — they enable reflex recall.

**Kind:**
- `semantic` (default) — facts, preferences, lessons, constraints, strategies. **When in doubt, use this.**
- `episodic` — specific dated events: "decided to use X", "v1.0 released", "found bug in Y"
- `procedural` — reusable step-by-step workflows ONLY (deploy steps, build process). **NOT:** strategies, plans, guidelines, one-time decisions. If it has an end condition ("until X", "for now"), it's NOT procedural.

**Other modifiers:**
- Lessons → `"tags": ["lesson", "trigger:kebab-verb"]` (e.g. `trigger:git-push`, `trigger:deploy-app`)
- Cross-project knowledge → `-H "X-Namespace: default"`

```bash
curl -sf -X POST http://localhost:3917/memories \
  -d '{"content": "...", "tags": ["ssh-config"]}'

# Procedures (step-by-step only)
'{"content": "deploy: test → build → stop → start", "tags": ["deploy-flow"], "kind": "procedural"}'
# Lessons with trigger
'{"content": "LESSON: never force-push to main", "tags": ["lesson","trigger:git-push"]}'
# Events/decisions
'{"content": "Decided to switch from Postgres to SQLite for zero-dep deployment", "tags": ["db-choice"], "kind": "episodic"}'
# Cross-project knowledge
curl -sf -X POST http://localhost:3917/memories -H "X-Namespace: default" \
  -d '{"content": "User prefers concise replies, no filler", "tags": ["communication-style"]}'
```

### 3. Recalling Memories

Before acting on any non-trivial task, you **MUST** check your memory. Do not assume you remember.

- **Pre-action check:** Before risky operations (e.g., deploying, pushing code), check triggers for relevant lessons.
- **Topic drill-down:** Resume includes a topic index (e.g. `kb3: "Memory architecture" [8]`). To explore a topic's full memories, use `POST /topic {"ids":["kb3"]}`. Use this when you need all memories about a subject.
- **General search:** Query with the topic.
  - `query` (required): Search string.
  - `expand` (boolean): Set to true for short/vague queries (+1-2s).
  - `limit` (default 20), `budget_tokens` (default 2000), `tags`, `min_score` (default 0.30; set to 0.0 for all results).

```bash
curl -sf -X POST http://localhost:3917/recall \
  -d '{"query": "how do we deploy"}'

# Pre-action trigger check
curl -sf http://localhost:3917/triggers/deploy

# Drill into specific topics from resume index
curl -sf -X POST http://localhost:3917/topic \
  -d '{"ids": ["kb1", "kb3"]}'
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

# LLM — default model (text processing: merge, expand)
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
