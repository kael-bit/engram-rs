//! Centralised prompt texts and tool-call JSON schemas.
//!
//! Every LLM prompt and structured-output schema lives here so they can be
//! audited, tuned, and versioned in one place.  The rest of the codebase
//! imports from `crate::prompts`.

// ---------------------------------------------------------------------------
// ai.rs — expand_query
// ---------------------------------------------------------------------------

pub const EXPAND_PROMPT: &str = r#"Given a search query for a PERSONAL knowledge base (notes, decisions, logs), generate 3-5 alternative search phrases that would help find relevant stored notes. Bridge abstraction levels: abstract→concrete, concrete→abstract.

CRITICAL: If the query contains CJK characters, include expansions in BOTH the original language AND English. If the query is in English, include English expansions only.

Examples:
deploy → deployment procedure, deploy steps, systemd restart, CI/CD pipeline.
auth bug → authentication fix, login issue, token validation error.
user preferences → settings, config choices, UI defaults, what the user likes.
project architecture → system design, module structure, how components connect.

Focus on rephrasing the INTENT, not listing random related technologies.
If the query asks about a tool/library choice, rephrase as: why/decision/migration/alternatives.
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
- "critical": User EXPLICITLY asked to remember this ("remember this", "don't forget"). Core knowledge.
- "high": Significant decisions, lessons learned, identity-defining facts.
- "medium": Useful context, minor preferences. May fade if not reinforced.
- "low": Background info, probably not worth keeping long-term.

Rules:
- Extract 0-3 entries per input. Zero is fine if nothing is worth remembering.
- Each entry must be self-contained (understandable without context)
- Prefer concise entries (1-2 concise sentences) over verbose ones
- Write content in the same language as the input. NEVER translate — if the conversation is in Chinese, output Chinese. If mixed, use the dominant language.
- importance MUST reflect user intent — if they say "remember" or "don't forget", it's "critical"

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
- Instructions from one agent to another (task delegation, implementation orders)

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
// (old global-scan audit prompts removed — replaced by topic distillation in audit.rs)

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

pub const GATE_SYSTEM: &str = r#"Core is PERMANENT memory — it survives total context loss and never decays.
Whitelist: ONLY these four categories may enter Core. Everything else → REJECT.

1. LESSON — "never do X" / "always Y before Z"
   A specific past mistake and how to avoid repeating it.
   Must name the concrete mistake. Generic advice ("test before deploy") → REJECT.

2. IDENTITY — who the user is, who the agent is, relationship dynamics
   Preferences, personality traits, communication style, roles.
   Must be about a person, not a system.

3. CONSTRAINT — a hard rule that applies permanently
   Security policies, approval workflows, communication boundaries.
   Must be unconditional. If it has "for now" / "until" / "currently" → REJECT.

4. DECISION RATIONALE — why we chose X over Y
   The reasoning behind a choice. Must contain the WHY.
   The outcome alone ("we use SQLite") without reasoning → REJECT.

Default: REJECT. When in doubt → REJECT.
If it reads like a changelog, progress report, bug fix, config snapshot,
step-by-step recipe, algorithm formula, or implementation detail → REJECT.

Kind assignment (only when approving):
- procedural: permanently true process with no end condition
- semantic: fact about identity, preference, or constraint
- episodic: time-bound decision or event that may lose relevance"#;

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
                "description": "Memory kind (only when approving): procedural=permanent process, semantic=identity/preference/constraint, episodic=time-bound decision"
            }
        },
        "required": ["decision"]
    })
}

pub fn gate_batch_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "Memory ID prefix (first 8 chars)" },
                        "decision": { "type": "string", "enum": ["approve", "reject"] },
                        "kind": { "type": "string", "enum": ["semantic", "procedural", "episodic"], "description": "Memory kind (only when approving)" }
                    },
                    "required": ["id", "decision"]
                }
            }
        },
        "required": ["decisions"]
    })
}

// ---------------------------------------------------------------------------
// consolidate/merge.rs — reconcile
// ---------------------------------------------------------------------------

pub fn reconcile_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["update", "absorb", "keep_both"],
                "description": "How to reconcile the two memories"
            },
            "merged_content": {
                "type": "string",
                "description": "Merged text preserving ALL details from both entries. Required when decision is update or absorb."
            }
        },
        "required": ["decision"]
    })
}

// ---------------------------------------------------------------------------
// consolidate/merge.rs — merge
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
- update: The newer entry is an updated version of the same information. The older one is now stale/outdated.
- absorb: The two entries overlap significantly and should be combined into one.
- keep_both: They cover genuinely different aspects or the older one has unique details not in the newer one.

If you choose "update" or "absorb", you MUST also provide "merged_content": a single merged text that preserves ALL specific details from BOTH entries — names, numbers, commands, constraints, lessons, reinforcement language. Never drop details. Same language as originals. 2-4 concise sentences."#;

// ---------------------------------------------------------------------------
// consolidate/facts.rs
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// api/memory.rs — insert-time merge
// ---------------------------------------------------------------------------

pub const INSERT_MERGE_PROMPT: &str = r#"Merge two versions of the same memory into one. Preserve ALL specific details from BOTH versions — names, numbers, commands, constraints. Output ONLY the merged text, nothing else. Keep the same language as the input. Be concise; don't add commentary or explanation. Keep it to 2-3 concise sentences if possible."#;

// ---------------------------------------------------------------------------
// consolidate/distill.rs — session note distillation
// ---------------------------------------------------------------------------

pub const DISTILL_SYSTEM_PROMPT: &str = "You synthesize session notes into a concise project status snapshot.\n\
Focus on: what exists now, current version/state, key capabilities, what's in progress.\n\
Skip: lessons learned, past bugs, how things were built.\n\
Output a single paragraph, 2-4 sentences, under 250 chars. Same language as input.\n\
No preamble, no markdown headers — just the status text.";

// ---------------------------------------------------------------------------
// consolidate/audit.rs — topic distillation
// ---------------------------------------------------------------------------

pub const DISTILL_TOPIC_SYSTEM: &str = r#"You condense overlapping memories within a topic into fewer, richer entries.

Given N memories that belong to the same semantic topic, find groups that overlap
and merge each group into one comprehensive entry. Leave unique memories alone.

Rules:
1. **Preserve ALL specific details** — names, numbers, IDs, dates, commands, constraints. Never generalize away specifics.
2. **Only merge memories that genuinely overlap** — same subject, redundant info. Don't merge unrelated memories just because they're in the same topic.
3. **Each distilled entry must list its source IDs** (first 8 chars of the memory ID). A source can only appear in ONE distilled entry.
4. **A distilled entry must replace 2+ sources** — don't output single-source "merges".
5. **If nothing overlaps, return an empty distilled array** — don't force merges.
6. **Kind**: choose the most appropriate kind for the merged result (semantic/episodic/procedural).
7. **Same language as input** — don't translate.
8. **Content length**: the merged text should be roughly as long as the longest source, not a summary. You're consolidating, not summarizing."#;

pub fn distill_topic_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "distilled": {
                "type": "array",
                "description": "List of condensed memory entries",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The condensed memory text. Must preserve ALL specific details from sources."
                        },
                        "source_ids": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "IDs (first 8 chars) of memories this entry replaces"
                        },
                        "kind": {
                            "type": "string",
                            "enum": ["semantic", "episodic", "procedural"],
                            "description": "Memory kind for the distilled entry"
                        }
                    },
                    "required": ["content", "source_ids"]
                }
            }
        },
        "required": ["distilled"]
    })
}

// ---------------------------------------------------------------------------
// recall.rs — rerank
// ---------------------------------------------------------------------------

pub const RERANK_SYSTEM: &str = "\
You rerank memory search results by relevance to the user's query.
Think step-by-step about what the user ACTUALLY needs:
- \"X是谁\" / \"who is X\" → identity/relationship answers first
- \"X怎么用\" / \"how to X\" → workflows, procedures, role descriptions first; lessons/caveats second
- \"X怎么设计\" / \"how is X designed\" → architecture/design answers first
Prefer results that DIRECTLY answer the question over tangential mentions or meta-commentary.
A result describing what something does and how to use it beats a cautionary principle about it.
Return ONLY the numbers, most relevant first, comma-separated. No explanation.
Example: 3,1,5,2";

// ---------------------------------------------------------------------------
// proxy/extract.rs — proxy memory extraction
// ---------------------------------------------------------------------------

pub const PROXY_EXTRACT_SYSTEM: &str = "You extract long-term memories from a multi-turn conversation between a user and their AI assistant.\n\
    Most windows produce nothing — that's fine. But don't miss real signals.\n\n\
    EXTRACT:\n\
    - Decisions: 'we chose X over Y', 'switching to native binaries'\n\
    - Constraints: 'no Docker', 'code must look human-written', 'never mention X in public'\n\
    - Lessons: mistakes pointed out, corrections, 'don't do X again', 'X was wrong because Y'\n\
    - User feedback that shapes behavior: criticism, preferences, rules stated in frustration\n\
      (e.g. 'you're too expensive, use haiku for this' → extract the model routing decision)\n\
    - Infrastructure/tooling changes that persist\n\
    - Gotchas discovered through experience\n\n\
    Extract from EITHER side — user corrections AND assistant realizations both count.\n\
    User anger often contains the most important signals.\n\n\
    NEVER extract:\n\
    - Routine code changes, refactors, test results\n\
    - Bug fixes (unless there's a reusable lesson)\n\
    - The assistant explaining how things work (teaching ≠ memory)\n\
    - Anything already in ALREADY IN MEMORY below\n\n\
    LANGUAGE: Match the user's language.\n\n\
    DEDUP: Skip if it overlaps with ALREADY IN MEMORY.";

pub fn proxy_extract_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "One sentence, concrete, under 150 chars"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "1-4 relevant tags"
                        },
                        "kind": {
                            "type": "string",
                            "enum": ["semantic", "episodic", "procedural"],
                            "description": "semantic=facts/decisions/lessons/preferences (most memories are this). episodic=specific dated events. procedural=reusable step-by-step workflows ONLY (e.g. 'deploy: test→build→stop→copy→start'). Code changes, prompt edits, bug fixes are NOT procedural — they are semantic."
                        }
                    },
                    "required": ["content", "tags", "kind"]
                },
                "maxItems": 3
            }
        },
        "required": ["items"]
    })
}

// ---------------------------------------------------------------------------
// topiary/naming.rs — topic naming
// ---------------------------------------------------------------------------

pub const TOPIC_NAMING_SYSTEM: &str = r#"You are naming memory topics for an AI agent's retrieval index.
For each cluster, write a name the agent can use to decide whether to open this topic.

Rules:
- 2-4 words, English only
- Name must cover ALL samples, not just the most prominent one
- If samples share a clear action/scenario, use verb phrase (e.g. "Fix scoring errors")
- If samples are thematically related but diverse, use noun phrase (e.g. "Proxy and tunnel setup")
- Avoid: filler words (notes, details, lessons, info, overview, misc), comma-joined unrelated concepts"#;

pub fn topic_naming_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "Topic ID (e.g. kb1)" },
                        "name": { "type": "string", "description": "2-4 word name" }
                    },
                    "required": ["id", "name"]
                }
            }
        },
        "required": ["topics"]
    })
}
