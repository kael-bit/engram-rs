//! Centralised prompt texts and tool-call JSON schemas.
//!
//! Every LLM prompt and structured-output schema lives here so they can be
//! audited, tuned, and versioned in one place.  The rest of the codebase
//! imports from `crate::prompts`.

// ---------------------------------------------------------------------------
// ai.rs — expand_query
// ---------------------------------------------------------------------------

pub const EXPAND_PROMPT: &str = r#"Given a search query for a PERSONAL knowledge base (notes, decisions, logs), generate 3-5 alternative search phrases that would help find relevant stored notes. Bridge abstraction levels: abstract→concrete, concrete→abstract.

CRITICAL: The knowledge base contains BOTH Chinese and English notes. You MUST include expansions in BOTH languages regardless of query language.

Examples:
who am I → my identity, my identity and role, who am I, my name and positioning, identity bootstrap.
security lessons → security lesson, security mistakes and lessons, security discipline.
部署 → deploy procedure, 部署流程和步骤, deployment workflow, systemd 部署.
task delegation workflow → task specifications, task delegation workflow, subagent best practices.
GitHub配置 → GitHub SSH setup, GitHub 仓库和账号, repo migration.

Focus on rephrasing the INTENT, not listing random related technologies.
If the query asks about a tool/library choice, rephrase as: why/decision/migration/选择/替换.
NEVER output explanations, commentary, or bullet points with dashes.
Even for very short queries (1-2 words), always produce at least 3 phrases."#;

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

// NOTE: "under 200 chars" matches extract_memories_schema() description
pub const EXTRACT_SYSTEM_PROMPT: &str = r#"You are a memory extraction engine. Given a conversation or text, extract important facts, decisions, preferences, and events as discrete memory entries.

Importance levels:
- "critical": User EXPLICITLY asked to remember this ("记住", "remember this", "don't forget"). Core knowledge.
- "high": Significant decisions, lessons learned, identity-defining facts.
- "medium": Useful context, minor preferences. May fade if not reinforced.
- "low": Background info, probably not worth keeping long-term.

Rules:
- Extract 0-3 entries per input. Zero is fine if nothing is worth remembering.
- Each entry must be self-contained (understandable without context)
- Prefer concise entries (1-2 concise sentences) over verbose ones
- Write content in the same language as the input. NEVER translate — if the conversation is in Chinese, output Chinese. If mixed, use the dominant language.
- importance MUST reflect user intent — if they say "记住" or "remember", it's "critical"

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
- System/agent configuration: persona definitions, workflow templates, framework-injected instructions
- Operational directives: scheduled tasks, health checks, monitoring routines, API call templates
- Meta-instructions: behavior rules, response formatting, tool usage patterns"#;

pub fn extract_memories_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "memories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Concise memory text, 1-2 concise sentences"},
                        "importance": {"type": "string", "enum": ["low", "medium", "high", "critical"], "description": "Importance level"},
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

// NOTE: "ac>5" is a prompt heuristic, not from thresholds.rs
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

Demote ONE layer at a time — Core→Working or Working→Buffer. Never skip layers.

## Adjust importance

Use adjust when a memory belongs in its current layer but its importance score is wrong:
- Lower (e.g. 0.3): Memory is correct but rarely useful — low importance makes it decay faster in Working
- Raise (e.g. 0.9): Memory is clearly under-valued relative to its actual importance
- Do NOT use adjust as a substitute for demote — if the content doesn't belong in this layer, demote it

## Merge rules

- When memories overlap heavily, merge to consolidate information
- Cross-layer merge: the merged result MUST go to the HIGHER layer (Working+Core → Core)
- Preserve all specific details — never lose information in a merge

## Input Format

Each memory is labeled with a letter alias (A, B, C, ... Z, AA, AB, ...). Use these aliases when referencing memories in operations.

Memories are grouped into semantic clusters. Similar memories appear together with merge similarity hints when applicable. Use these hints to identify merge candidates.

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
                            "description": "Letter alias from the memory listing (A, B, C...) for promote/demote/adjust"
                        },
                        "to": {
                            "type": "integer",
                            "enum": [1, 2, 3],
                            "description": "Target layer. Demote by one layer only: Core(3)→Working(2) or Working(2)→Buffer(1)"
                        },
                        "importance": {
                            "type": "number",
                            "description": "New importance value 0.0-1.0 (for adjust op)"
                        },
                        "ids": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Letter aliases from the memory listing (A, B, C...) for merge op, minimum 2"
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

// NOTE: "rep >= 2" is a prompt heuristic for triage decisions
pub const TRIAGE_SYSTEM: &str = r#"You are triaging an AI agent's short-term memory buffer.
Each memory below is tagged with an ID. Decide which ones contain durable knowledge worth promoting to Working memory (medium-term), and which are transient.

Metadata:
- ac = recall count (how many times actively retrieved by query)
- rep = repetition count (how many times the same concept was mentioned again). High rep means the agent/user keeps restating this — it's deeply important to them.

PROMOTE if the memory contains:
- Design decisions, architecture choices, API contracts
- Lessons learned, rules, principles
- Procedures, workflows, step-by-step processes
- Identity info, preferences, constraints
- Project context that will be needed across sessions
- Anything with rep >= 2 — repeated emphasis signals importance

KEEP in buffer if:
- Session summaries that just list what was done (not lessons)
- Temporary status, in-progress notes
- Information that's only relevant right now
- Test infrastructure details (helper functions, visibility modifiers, mock setup)
- Implementation minutiae without broader lessons

Classify the kind for promoted memories:
- procedural: step-by-step workflows, build/deploy/test processes
- semantic: facts, decisions, lessons, preferences
- episodic: specific events, dated occurrences, one-time incidents"#;

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
                        "kind": { "type": "string", "enum": ["semantic", "procedural", "episodic"], "description": "Only for promoted memories" }
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

pub const GATE_SYSTEM: &str = r#"Core is PERMANENT memory that survives total context loss. The litmus test: if the agent wakes up with zero context, would this memory alone be useful? If it only makes sense alongside the code/docs/conversation it came from, REJECT.

APPROVE:
- "never force-push to main" (lesson — prevents repeating a mistake)
- "user prefers Chinese, hates verbose explanations" (identity/preference — shapes behavior)
- "all public output must hide AI identity" (constraint — hard rule that never changes)
- "we chose SQLite over Postgres for zero-dep deployment" (decision rationale — the WHY)

REJECT:
- "Recall quality: 1) bilingual expansion 2) FTS gating 3) gate evidence" (changelog — lists WHAT was done)
- "Fixed bug: triage didn't filter by namespace" (operational — belongs in git history)
- "Session: refactored auth, deployed v0.7, 143 tests pass" (session log — ephemeral status)
- "Improvement plan: add compression, fix lifecycle, batch ops" (plan — not a lesson)
- "cosine threshold 0.78, buffer TTL 24h, promote at 5 accesses" (config values — will go stale)
- "HNSW index replaces brute-force search" (implementation detail — in the code already)

The pattern: APPROVE captures WHY and NEVER and WHO. REJECT captures WHAT and WHEN and HOW MUCH.
Numbered lists of changes (1) did X 2) did Y 3) did Z) are almost always changelogs → REJECT.
If it reads like a commit message or progress report, REJECT."#;

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
                "enum": ["semantic", "procedural"],
                "description": "Memory kind (only when approving)"
            }
        },
        "required": ["decision"]
    })
}

// ---------------------------------------------------------------------------
// consolidate/merge.rs
// ---------------------------------------------------------------------------

pub const MERGE_SYSTEM: &str = r#"Merge these related memory entries into a single concise note. Rules:
- Preserve ALL specific names, tools, libraries, versions, and technical terms.
- If one entry updates or supersedes the other, keep the latest state.
- Remove only truly redundant/repeated sentences.
- Names, numbers, versions, dates, tool names > vague summaries. Never drop specific terms.
- Keep it to 2-3 concise sentences if possible.
- Same language as originals. Output only the merged text, nothing else."#;

pub const RECONCILE_PROMPT: &str = r#"You are comparing two memory entries about potentially the same topic.
The NEWER entry was created after the OLDER one.

Decide:
- update: The newer entry is an updated version of the same information. The older one is now stale/outdated and should be removed.
- absorb: The newer entry contains all useful info from the older one plus more. The older one is redundant.
- keep_both: They cover genuinely different aspects or the older one has unique details not in the newer one."#;

// ---------------------------------------------------------------------------
// consolidate/facts.rs
// ---------------------------------------------------------------------------

// Disabled: facts auto-extraction is currently off. Kept for potential re-enablement.
#[allow(dead_code)]
pub const FACT_EXTRACT_PROMPT: &str = r#"Extract factual triples from this memory text. A triple is (subject, predicate, object) representing a concrete, stable relationship.
Examples: (user, prefers, dark mode), (engram, uses, SQLite), (project, language, Rust)

Rules:
- Only extract concrete, stable facts — NOT transient states or opinions
- Subject/object should be short noun phrases (1-3 words)
- Predicate should be a verb or relationship label
- Skip if the text is purely procedural/operational with no factual content"#;

// Disabled: facts auto-extraction is currently off. Kept for potential re-enablement.
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

pub const INSERT_MERGE_PROMPT: &str = r#"Merge two versions of the same memory into one. Preserve ALL specific details from BOTH versions — names, numbers, commands, constraints. Output ONLY the merged text, nothing else. Keep the same language as the input. Be concise; don't add commentary or explanation. Keep it to 2-3 concise sentences if possible."#;
