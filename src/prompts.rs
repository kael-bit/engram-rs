//! Centralised prompt texts and tool-call JSON schemas.
//!
//! Every LLM prompt and structured-output schema lives here so they can be
//! audited, tuned, and versioned in one place.  The rest of the codebase
//! imports from `crate::prompts`.

// ---------------------------------------------------------------------------
// ai.rs — expand_query
// ---------------------------------------------------------------------------

pub const EXPAND_PROMPT: &str = "Given a search query for a PERSONAL knowledge base (notes, decisions, logs), \
    generate 4-6 alternative search phrases that would help find relevant stored notes. \
    Bridge abstraction levels: abstract→concrete, concrete→abstract. \
    \
    CRITICAL: The knowledge base contains BOTH Chinese and English notes. \
    You MUST include expansions in BOTH languages regardless of query language. \
    \
    Examples: \
    who am I → my identity, my identity and role, who am I, my name and positioning, identity bootstrap. \
    security lessons → security lesson, security mistakes and lessons, security discipline. \
    部署 → deploy procedure, 部署流程和步骤, deployment workflow, systemd 部署. \
    task delegation workflow → task specifications, task delegation workflow, subagent best practices. \
    GitHub配置 → GitHub SSH setup, GitHub 仓库和账号, repo migration. \
    \
    Focus on rephrasing the INTENT, not listing random related technologies. \
    If the query asks about a tool/library choice, rephrase as: why/decision/migration/选择/替换. \
    NEVER output explanations, commentary, or bullet points with dashes. \
    Even for very short queries (1-2 words), always produce at least 4 phrases.";

pub fn expand_query_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Alternative search phrases (3-5)"
            }
        },
        "required": ["queries"]
    })
}

// ---------------------------------------------------------------------------
// ai.rs — extract_memories
// ---------------------------------------------------------------------------

pub const EXTRACT_SYSTEM_PROMPT: &str = r#"You are a memory extraction engine. Given a conversation or text, extract important facts, decisions, preferences, and events as discrete memory entries.

Importance scale:
- 0.9-1.0: User EXPLICITLY asked to remember this ("记住", "remember this", "don't forget"). Core knowledge.
- 0.7-0.8: Significant decisions, strong preferences, lessons learned, identity-defining facts. Worth keeping.
- 0.4-0.6: Useful context, minor preferences, background info. May fade if not reinforced.

Rules:
- Extract 0-3 entries per input. Zero is fine if nothing is worth remembering.
- Each entry must be self-contained (understandable without context)
- Prefer concise entries (under 200 chars) over verbose ones
- Write content in the same language as the input. NEVER translate — if the conversation is in Chinese, output Chinese. If mixed, use the dominant language.
- importance MUST reflect user intent — if they say "记住" or "remember", it's 0.9+

EXTRACT these (worth remembering):
- Identity: who someone is, their preferences, principles, personality
- Decisions: choices made and why, trade-offs considered
- Lessons: mistakes, insights, things that worked or didn't
- Self-reflections: realizations about own behavior patterns, blind spots, habits to change (these are HIGH value — 0.8+)
- Relationships: facts about people, how they relate to each other
- Strategic: goals, plans, architectural choices

SKIP these (not worth remembering):
- Operational details: bug fixes, version bumps, deployment steps, code changes
- Implementation notes: "update X to do Y", "add Z to W", "fix A in B" — these are code tasks, not memories
- Transient states: "service is running", "memory at 33%", "tests passing"
- Debug info, log output, error messages
- Summaries or recaps of work done (these are session logs, not memories) — BUT self-critical reflections about patterns/habits ARE worth extracting
- Instructions from one agent to another (e.g. "also add to proxy", "fix now — add touch")

HARD REJECT — NEVER extract these as memories (they are scaffolding, not knowledge):
- System prompts and injected instructions (content from SOUL.md, AGENTS.md, HEARTBEAT.md, TOOLS.md, USER.md, IDENTITY.md, MEMORY.md, or similar)
- Operational directives: "every heartbeat do X", "run this command on wake", "before shutdown do Y"
- Configuration templates and boilerplate: API keys, curl examples, service names, file paths used as reference
- Tool usage patterns and API call templates: "use this endpoint", "call this script"
- Meta-instructions about how to behave, respond, or format output
- Heartbeat checks, health status pings, routine monitoring output
- Anything that reads like a rule/playbook for an agent rather than a human-stated fact or preference
- Framework-injected context that appears in every conversation"#;

pub fn extract_memories_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "memories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Concise memory text, under 200 chars"},
                        "importance": {"type": "number", "description": "0.0-1.0 importance score"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "kind": {
                            "type": "string",
                            "enum": ["semantic", "episodic", "procedural"],
                            "description": "semantic=facts, episodic=events, procedural=how-to (never decay)"
                        }
                    },
                    "required": ["content"]
                }
            }
        },
        "required": ["memories"]
    })
}

// ---------------------------------------------------------------------------
// consolidate/audit.rs
// ---------------------------------------------------------------------------

pub const AUDIT_SYSTEM: &str = r#"Review an AI agent's memory layers and propose maintenance operations.

You have three powers:
1. **Schedule** (promote/demote): Move memories between layers
2. **Adjust** (adjust): Change a memory's importance score
3. **Merge**: Combine duplicate/overlapping memories

You CANNOT delete memories. Deletion happens through natural lifecycle (Buffer TTL expiry).

Metadata per memory:
- imp = importance (0-1)
- ac = access count (times recalled). ac=0 = never recalled
- age = days since creation
- mod = days since last modification
- tags = labels
- Layer: Core (3), Working (2), Buffer (1)

## Layer Principles

**Core** = stable principles, lessons from mistakes, identity constraints. Rarely changes.
**Working** = active project context, current decisions. Becomes stale as projects evolve.
**Buffer** = temporary intake, expires via TTL.

## Promote — be VERY selective

Core is permanent. Only promote Working memories proven valuable over time.

✅ GOOD promotion:
- ac>5, survived multiple sessions, content still matches current architecture
- Lesson that actually prevented repeating a mistake (not just "I learned X")
- Hard constraint from the user that applies permanently

❌ BAD promotion — DO NOT:
- ac=0 and recently created → not battle-tested, keep in Working
- Tagged `auto-extract` → machine-generated, often low quality
- Content unrelated to the project → wrong scope
- Describes a fix later replaced → obsolete
- Generic platitudes without specific context

A `LESSON:` prefix or `lesson` tag does NOT automatically qualify for Core.

## Demote

Move memories down one layer when they no longer belong:
- Implementation details already in code
- Session logs and progress reports
- Stale plans/TODOs or config snapshots
- Content about removed/replaced features

## Adjust importance

Lower importance (e.g. 0.3) for memories that are losing relevance but shouldn't move layers yet.
Raise importance (e.g. 0.9) for memories that are clearly under-valued.

## Merge rules

- When memories overlap heavily, merge to consolidate information
- Cross-layer merge: the merged result MUST go to the HIGHER layer (Working+Core → Core)
- Preserve all specific details — never lose information in a merge

## Guidelines

- **Superseded:** newer memory covers same knowledge → merge or demote the older one
- **Obsolete:** high ac on content about removed features ≠ still valid → demote
- Same-layer demote/promote is a no-op bug (L2→L2, L3→L3)
- Read full content before deciding — don't judge by tags alone
- Propose no operations if nothing needs changing."#;

pub fn audit_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "description": "List of maintenance operations. Empty array if nothing needs changing.",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "enum": ["promote", "demote", "adjust", "merge"],
                            "description": "Operation type"
                        },
                        "id": {
                            "type": "string",
                            "description": "8-char short ID of the target memory (for promote/demote/adjust)"
                        },
                        "to": {
                            "type": "integer",
                            "enum": [1, 2, 3],
                            "description": "Target layer: 1=Buffer, 2=Working, 3=Core (for promote/demote)"
                        },
                        "importance": {
                            "type": "number",
                            "description": "New importance value 0.0-1.0 (for adjust op)"
                        },
                        "ids": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Short IDs of memories to merge (for merge op, minimum 2)"
                        },
                        "content": {
                            "type": "string",
                            "description": "Combined text for the merged memory (for merge op)"
                        },
                        "layer": {
                            "type": "integer",
                            "enum": [2, 3],
                            "description": "Layer for merged memory — must be the HIGHEST layer among source memories (for merge op)"
                        },
                        "tags": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Tags for the merged memory (for merge op)"
                        }
                    },
                    "required": ["op"]
                }
            }
        },
        "required": ["operations"]
    })
}

// ---------------------------------------------------------------------------
// consolidate/triage.rs
// ---------------------------------------------------------------------------

pub const TRIAGE_SYSTEM: &str = "You are triaging an AI agent's short-term memory buffer.\n\
    Each memory below is tagged with an ID. Decide which ones contain durable knowledge \
    worth promoting to Working memory (medium-term), and which are transient.\n\n\
    Metadata:\n\
    - ac = recall count (how many times actively retrieved by query)\n\
    - rep = repetition count (how many times the same concept was mentioned again). \
    High rep means the agent/user keeps restating this — it's deeply important to them.\n\n\
    PROMOTE if the memory contains:\n\
    - Design decisions, architecture choices, API contracts\n\
    - Lessons learned, rules, principles\n\
    - Procedures, workflows, step-by-step processes\n\
    - Identity info, preferences, constraints\n\
    - Project context that will be needed across sessions\n\
    - Anything with rep >= 2 — repeated emphasis signals importance\n\n\
    KEEP in buffer if:\n\
    - Session summaries that just list what was done (not lessons)\n\
    - Temporary status, in-progress notes\n\
    - Information that's only relevant right now\n\
    - Test infrastructure details (helper functions, visibility modifiers, mock setup)\n\
    - Implementation minutiae without broader lessons\n\n\
    Classify the kind for promoted memories:\n\
    - procedural: step-by-step workflows, build/deploy/test processes\n\
    - semantic: everything else (facts, decisions, lessons, preferences)";

pub fn triage_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "Memory ID prefix (first 8 chars)" },
                        "action": { "type": "string", "enum": ["promote", "keep"] },
                        "kind": { "type": "string", "enum": ["semantic", "procedural"], "description": "Only for promoted memories" }
                    },
                    "required": ["id", "action"]
                }
            }
        },
        "required": ["decisions"]
    })
}

// ---------------------------------------------------------------------------
// consolidate/mod.rs — promotion gate
// ---------------------------------------------------------------------------

pub const GATE_SYSTEM: &str = "\
Core is PERMANENT memory that survives total context loss. The litmus test: \
if the agent wakes up with zero context, would this memory alone be useful? \
If it only makes sense alongside the code/docs/conversation it came from, REJECT.\n\
\n\
APPROVE:\n\
- \"never force-push to main\" (lesson — prevents repeating a mistake)\n\
- \"user prefers Chinese, hates verbose explanations\" (identity/preference — shapes behavior)\n\
- \"all public output must hide AI identity\" (constraint — hard rule that never changes)\n\
- \"we chose SQLite over Postgres for zero-dep deployment\" (decision rationale — the WHY)\n\
\n\
REJECT:\n\
- \"Recall quality: 1) bilingual expansion 2) FTS gating 3) gate evidence\" (changelog — lists WHAT was done)\n\
- \"Fixed bug: triage didn't filter by namespace\" (operational — belongs in git history)\n\
- \"Session: refactored auth, deployed v0.7, 143 tests pass\" (session log — ephemeral status)\n\
- \"Improvement plan: add compression, fix lifecycle, batch ops\" (plan — not a lesson)\n\
- \"cosine threshold 0.78, buffer TTL 24h, promote at 5 accesses\" (config values — will go stale)\n\
- \"HNSW index replaces brute-force search\" (implementation detail — in the code already)\n\
\n\
The pattern: APPROVE captures WHY and NEVER and WHO. REJECT captures WHAT and WHEN and HOW MUCH.\n\
Numbered lists of changes (1) did X 2) did Y 3) did Z) are almost always changelogs → REJECT.\n\
If it reads like a commit message or progress report, REJECT.";

pub fn gate_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["approve", "reject"],
                "description": "Whether to promote to Core"
            },
            "kind": {
                "type": "string",
                "enum": ["semantic", "procedural", "episodic"],
                "description": "Memory kind (only when approving)"
            }
        },
        "required": ["decision"]
    })
}

// ---------------------------------------------------------------------------
// consolidate/merge.rs
// ---------------------------------------------------------------------------

pub const MERGE_SYSTEM: &str = "Merge these related memory entries into a single concise note. Rules:\n\
    - Preserve ALL specific names, tools, libraries, versions, and technical terms.\n\
    - If one entry updates or supersedes the other, keep the latest state.\n\
    - Remove only truly redundant/repeated sentences.\n\
    - Names, numbers, versions, dates, tool names > vague summaries. Never drop specific terms.\n\
    - Keep it under 400 characters if possible.\n\
    - Same language as originals. Output only the merged text, nothing else.";

pub const RECONCILE_PROMPT: &str = "You are comparing two memory entries about potentially the same topic.\n\
    The NEWER entry was created after the OLDER one.\n\n\
    Decide:\n\
    - update: The newer entry is an updated version of the same information. \
    The older one is now stale/outdated and should be removed.\n\
    - absorb: The newer entry contains all useful info from the older one plus more. \
    The older one is redundant.\n\
    - keep_both: They cover genuinely different aspects or the older one has \
    unique details not in the newer one.";

// ---------------------------------------------------------------------------
// consolidate/facts.rs
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub const FACT_EXTRACT_PROMPT: &str = "Extract factual triples from this memory text. \
    A triple is (subject, predicate, object) representing a concrete, stable relationship.\n\
    Examples: (user, prefers, dark mode), (engram, uses, SQLite), (project, language, Rust)\n\n\
    Rules:\n\
    - Only extract concrete, stable facts — NOT transient states or opinions\n\
    - Subject/object should be short noun phrases (1-3 words)\n\
    - Predicate should be a verb or relationship label\n\
    - Skip if the text is purely procedural/operational with no factual content";

#[allow(dead_code)]
pub fn fact_extract_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "description": "Extracted triples. Empty array if nothing to extract.",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string", "description": "Short noun phrase (1-3 words)"},
                        "predicate": {"type": "string", "description": "Verb or relationship label"},
                        "object": {"type": "string", "description": "Short noun phrase (1-3 words)"}
                    },
                    "required": ["subject", "predicate", "object"]
                }
            }
        },
        "required": ["facts"]
    })
}

// ---------------------------------------------------------------------------
// api/memory.rs — insert-time merge
// ---------------------------------------------------------------------------

pub const INSERT_MERGE_PROMPT: &str = "\
Merge two versions of the same memory into one. Preserve ALL specific details \
from BOTH versions — names, numbers, commands, constraints. \
Output ONLY the merged text, nothing else. Keep the same language as the input. \
Be concise; don't add commentary or explanation.";
