# engram-rs Internal Architecture

> Audience: AI auditing/optimizing prompts in `src/prompts.rs`.
> Describes what the code DOES as of the current source. All thresholds from `src/thresholds.rs`.
> All LLM prompts and tool schemas live in `src/prompts.rs`.
> **This document must be updated whenever the architecture changes.**

---

## 1. Memory Model

Each memory has: `id` (UUID), `content` (≤8192 chars), `layer` (1/2/3), `importance` (0.0–1.0),
`created_at`, `last_accessed`, `access_count`, `repetition_count`, `decay_rate`, `source`,
`tags` (JSON array, ≤20 tags, ≤32 chars each), `namespace`, `embedding` (BLOB), `kind`
(`semantic`|`episodic`|`procedural`), `modified_at`.

### Layers

| Layer | Enum | Score Bonus | Semantics |
|-------|------|-------------|-----------|
| **Buffer** | 1 | 0.9× | Intake. All new memories land here via API. Evicted by epoch-based decay or capacity cap (200). |
| **Working** | 2 | 1.0× | Active knowledge. Promoted from Buffer by consolidation. Never deleted — importance decays by kind. |
| **Core** | 3 | 1.1× | Long-term identity. Promoted from Working through LLM gate. Never deleted. |

**Decay rates by kind** (epoch-based, per active consolidation cycle):

| Kind | Decay per epoch | Use case |
|------|----------------|----------|
| `episodic` | −0.005 (full) | Events, experiences, time-bound context |
| `semantic` | −0.003 (60%) | Knowledge, preferences, lessons (default) |
| `procedural` | −0.001 (20%) | Workflows, instructions, how-to |

Decay only happens during active consolidation — no decay while the agent is idle. Working and Core memories are never deleted regardless of importance. Buffer memories below `decay_drop_threshold` (0.01) are evicted.

**Layer entry:** All API writes enter Buffer regardless of any `layer` parameter. Three layers each have a single entry point: Buffer ← API, Working ← consolidation promotion, Core ← LLM gate. The agent cannot specify which layer a memory goes to.

---

## 2. Memory Lifecycle

```
API POST /memories
    │
    ▼
[Validation] → content ≤8192, tags ≤20, source ≤64
    │
    ▼
[Dedup Check] ─── duplicate found ──→ touch() + increment repetition_count → return existing
    │ no dup
    ▼
[Insert into Buffer]
    │
    ▼
[Async: generate embedding, index FTS, update vec index]
    │
    ╔══════════════════════════════════════════════════════════╗
    ║  CONSOLIDATION CYCLE (every 30 min via cron/API call)   ║
    ╠══════════════════════════════════════════════════════════╣
    ║                                                          ║
    ║  1. cleanup_stale_tags()                                 ║
    ║  2. Buffer capacity cap                                   ║
    ║  3. Working quality (never delete, never demote)           ║
    ║  4. Buffer → Working (mechanical: score ≥ 5 OR kind)     ║
    ║  5. Buffer capacity cap enforcement                      ║
    ║  6. Working → Core candidates collected                  ║
    ║  7. LLM Gate: review candidates → approve/reject         ║
    ║  8. Reconcile: LLM detects same-topic updates            ║
    ║  9. Session distillation: 3+ session notes → summary     ║
    ║ 10. Buffer dedup: cosine-based, no LLM                   ║
    ║ 11. Buffer triage: LLM evaluates buffer → promote/keep   ║
    ║ 12. Merge: LLM combines near-duplicates (if enabled)     ║
    ║ 13. FTS repair                                           ║
    ║ 14. Core summary update                                  ║
    ║ 15. Importance decay (epoch-based, kind-differentiated)    ║
    ║ 16. Drop fully-decayed buffer memories                   ║
    ╚══════════════════════════════════════════════════════════╝
```

### 2.1 Insert & Dedup

On `POST /memories`, the API handler:

1. Validates input (content length, tag count).
2. **Layer override:** Ignores any `layer` parameter — all writes go to Buffer.
2. **API-level semantic dedup** (`quick_semantic_dup`): if AI is configured, embeds content and
   checks cosine similarity against stored memories. Threshold depends on source:
   - Proxy extraction: `PROXY_DEDUP_SIM = 0.60`
   - Normal insert: `INSERT_DEDUP_SIM = 0.65`
3. If a duplicate is found: calls `touch()` (bumps `access_count`, updates `last_accessed`)
   and increments `repetition_count`. Returns the existing memory.
4. If no duplicate: calls `db.insert()`.

**DB-level dedup** (`find_near_duplicate` in `db/memory.rs`):
- Runs on insert when `skip_dedup` is not set.
- If embedding is provided: **Jaccard pre-filter** (tokenize both texts, require Jaccard > `DEDUP_JACCARD_PREFILTER = 0.50`) → then **cosine similarity** (> `DEDUP_COSINE_SIM = 0.85`).
- If no embedding: Jaccard-only check.
- On duplicate hit with very high similarity (> `INSERT_MERGE_SIM = 0.80`): calls LLM to merge
  content of old + new, using `INSERT_MERGE_PROMPT`. Updates existing memory's content in-place.

### 2.2 Buffer → Working Promotion (Mechanical)

In `consolidate_sync()`, two paths:

**By reinforcement score:**
- `reinforcement_score = access_count + repetition_count × 2.5`
- Promote if score ≥ `buffer_threshold` (default: `max(promote_threshold, 5)` = 5).

**By kind (cooldown-gated):**
- Lessons (`tag: lesson`) or procedurals (`kind: procedural`) auto-promote after 4 consolidation
  epochs in buffer (~2h of active consolidation). Distilled sessions excluded.

### 2.3 Buffer Triage (LLM)

After mechanical promotion, the LLM evaluates remaining buffer memories:

- **Eligibility:** >10 min old, not `distilled`/`ephemeral`, not already promoted.
- **Batch:** up to 20 per consolidation cycle, grouped by namespace.
- **LLM call:** `gate` model, `TRIAGE_SYSTEM` prompt. Decides `promote` or `keep` per memory.
- Promoted memories move to Working; can also set `kind: procedural`.
- `keep` decisions add `_triaged` tag (prevents re-triage on next cycle).

### 2.4 Buffer Eviction

Buffer eviction is epoch-based, consistent with the decay model — no wall-clock TTL.

**Eviction by decay:** Buffer memories decay through epoch-based importance reduction during active
consolidation cycles. When importance falls below `decay_drop_threshold` (0.01), the memory is deleted.
Idle periods cause zero decay, so memories survive agent downtime intact.

**Capacity cap** (default 200, env `ENGRAM_BUFFER_CAP`): when buffer exceeds cap, lowest-weight
entries are evicted. This prevents unbounded growth when the agent writes faster than consolidation
promotes.

### 2.5 Working → Core Promotion (LLM Gate)

**Candidate selection** (in `consolidate_sync`):
- `reinforcement_score ≥ promote_threshold (default 3)` AND `importance ≥ 0.6`, OR
- Age > `working_age_promote_secs (default 7 days)` AND score > 0 AND importance ≥ 0.6.
- Session notes, ephemeral, auto-distilled, and distilled memories are blocked.
- `gate-rejected-final` memories never retry.
- `gate-rejected` retries after 48 consolidation epochs; `gate-rejected-2` after 144 epochs.
  Epoch-based: idle time does not count toward cooldown.

**LLM Gate** (`llm_promotion_gate`):
- Model: `gate` tier.
- Prompt: `GATE_SYSTEM` — **whitelist approach**: only 4 categories are allowed into Core:
  1. **LESSON** — hard-won insights from mistakes or experience
  2. **IDENTITY** — who the user/agent is, preferences, constraints
  3. **CONSTRAINT** — unconditional rules (if it has "for now" / "temporary", reject)
  4. **DECISION RATIONALE** — why a permanent architectural/design choice was made
- Everything outside these categories is rejected by default.
- Context injected: high access count (≥30) or repetition count (≥2) noted.
- Returns `approve` (with kind) or `reject`.
- Rejection tagging escalates: `gate-rejected` → `gate-rejected-2` → `gate-rejected-final`.

**Working layer protection** (run before main consolidation):
- Working memories are never deleted or demoted. Gate-rejected Working memories only lose importance
  through natural decay — they are not deleted, demoted, or removed on any schedule.

### 2.6 Demotion

- Session notes (`source: session` or `tag: session`) in Core → demoted to Working.
- `ephemeral` tagged Core memories → demoted to Working.
- Working memories are never demoted to Buffer. Demotion only applies to Core → Working.
- Audit can demote via LLM decision (see §5).

### 2.7 Importance Decay

Every consolidation cycle (epoch-based, not time-based): `decay_importance(cycle_start, 0.005, floor=0.0)`.
All kinds decay but at different rates: episodic ×1.0, semantic ×0.6, procedural ×0.2. Decay only happens during active consolidation cycles — idle periods cause zero decay. Floor of 0.0 — memories with high repetition or access stay discoverable even at zero importance.

---

## 3. Consolidation Pipeline

Triggered by `POST /consolidate` or cron. Runs `consolidate()`:

### Phase 1: Synchronous (`consolidate_sync`)

1. **Tag cleanup:** Remove stale `gate-rejected` tags from Working after sufficient epochs without access.
   Remove orphaned `promotion` tags from Working/Core.
2. **Buffer capacity cap:** Evict lowest-weight entries if >200 buffer entries.
3. **Working quality:** Working memories are never deleted or demoted.
4. **Core hygiene:** Demote session notes and ephemeral from Core.
5. **Core overlap scan:** O(n²) pairwise cosine on Core embeddings. Flag pairs with
   similarity > `CORE_OVERLAP_SIM = 0.70`. Stored in `engram_meta` to avoid re-flagging.
6. **Working → Core candidates:** Collect based on reinforcement score + importance.
7. **Buffer → Working:** Mechanical promotion (score ≥ 5 or lesson/procedural after 2h).
8. **Buffer capacity cap:** FIFO eviction if >200 entries.
9. **Buffer capacity cap:** FIFO eviction if >200 entries.
10. **Drop decayed:** Delete Buffer memories below `decay_drop_threshold = 0.01`. Working/Core are never deleted.
11. **Importance decay:** epoch-based, −0.005/cycle (episodic), floor 0.0.

### Phase 2: Async (LLM-dependent)

12. **LLM Gate:** Review Working→Core candidates. Approve or reject with escalating epoch-based cooldowns.
13. **Reconcile:** Detect same-topic updates across all layers (see §3.2).
14. **Session distillation:** Synthesize 3+ session notes into project status snapshot (see §3.3).
15. **Buffer dedup:** Cosine-based duplicate removal within buffer (no LLM).
16. **Buffer triage:** LLM evaluates buffer memories for promotion.
17. **Merge** (if `merge=true`): LLM combines near-duplicate memories (see §3.1).
18. **FTS repair:** Fix orphaned FTS entries after merges/deletes.
19. **Core summary:** Update compressed summary in `engram_meta` for resume.

### 3.1 Merge (`merge_similar`)

In `consolidate/merge.rs`:

1. Fetch all memories with embeddings.
2. Group by namespace. Within each namespace, find cosine-similar pairs:
   - Threshold: `MERGE_SIM = 0.78`
3. For each pair, call LLM (`merge` model) with `MERGE_SYSTEM` prompt.
4. Winner (more important or more accessed) absorbs loser:
   - Winner's content replaced with merged text.
   - Tags merged (union, cap 20).
   - Importance = max of both.
   - Access counts summed.
   - Loser deleted.
5. Cross-layer merge: result goes to the **higher** layer.
6. Dedup key stored in `engram_meta` to prevent re-merging same pair.

### 3.2 Reconcile (`reconcile_updates`)

Detects when a newer memory supersedes an older one on the same topic.

1. Scan Working + Core memories with embeddings.
2. Find pairs in similarity window: `RECONCILE_MIN_SIM (0.55) < sim < RECONCILE_MAX_SIM (0.78)`.
3. Require time gap: newer must be ≥1h newer than older.
4. **Single LLM call** (`merge` model) with `RECONCILE_PROMPT` — returns both decision and merged content:
   - `update` + `merged_content`: older is stale → write merged content to newer, delete older.
   - `absorb` + `merged_content`: overlap detected → write merged content to newer, delete older.
   - `keep_both`: different aspects → no action.
5. Merged content preserves ALL specific details from both entries (names, numbers, constraints,
   reinforcement language). Falls back to newer's content if `merged_content` is empty.
6. Winner inherits: max importance, summed access counts, union of tags.
7. Dedup key stored per pair to prevent re-evaluation.

### 3.3 Session Distillation (`distill_sessions`)

Groups undistilled session notes by namespace. When 3+ notes accumulate:

1. Take latest 10 notes.
2. LLM (`gate` model) synthesizes a project status snapshot (≤250 chars).
3. Check for near-duplicate existing distillation (threshold: `TRIAGE_DEDUP_SIM = 0.75`).
4. Insert result as Working memory with tags `[project-status, auto-distilled]`.
5. Source notes tagged `distilled` to prevent reprocessing.
6. `auto-distilled` tag blocks Core promotion (project status is too volatile).

---

## 4. Recall & Scoring

### 4.1 Hybrid Retrieval (`recall()` in `src/recall.rs`)

Three retrieval channels run in parallel:

**FTS5 keyword search:**
- Pre-processed with jieba (Chinese segmentation) + bigrams (Japanese/Korean).
- Returns BM25-scored results.

**Semantic search:**
- Query embedded via `text-embedding-3-small`.
- If enough FTS+fact candidates exist (≥ `fetch_limit × 2`): restrict semantic search to
  those candidate IDs only (filtered mode). Otherwise: full corpus scan.
- Floor: `DEFAULT_SIM_FLOOR = 0.30` (or `min_score` if set).

**Fact graph:**
- Query entities against `facts` table (subject/object FTS).
- Multi-hop traversal (up to 2 hops) for richer context.
- Fact-linked memories get `relevance = 1.0`.

### 4.2 Scoring Formula

```
score = (W_importance × importance + W_recency × recency + W_relevance × relevance) × layer_bonus
```

Where:
- `W_importance = 0.20`, `W_recency = 0.20`, `W_relevance = 0.60`
- `recency = exp(-decay_rate × age_hours / 168)` (168h = 1 week half-life)
- `layer_bonus`: Buffer=0.9, Working=1.0, Core=1.1
- Score capped at 1.0.

### 4.3 Dual-Hit Boost

Memories found by **both** semantic and FTS get boosted:
- Normal queries: `relevance *= (1 + fts_rel × 0.3 × sem_gate)` where `sem_gate = min(1, relevance/0.45)`.
- Short CJK queries (<10 chars): `fts_boost = 0.6`, higher FTS floor (`0.50 + fts_rel × 0.35`).
  This compensates for `text-embedding-3-small`'s poor discrimination on short Chinese text.

**FTS-only hits** (no semantic confirmation):
- Normal: capped at `fts_rel × 0.25`.
- Short CJK: capped at `fts_rel × 0.40`.

**Keyword affinity penalty:**
Semantic-only hits (no FTS match) that lack any query terms in content get `relevance *= 0.7`.

### 4.4 Query Expansion

**Explicit** (`expand=true`): LLM generates 4–6 alternative search phrases via `EXPAND_PROMPT`.
Bilingual (Chinese + English) expansion required.

**Auto-expand**: If `expand` not set AND top result has `relevance < AUTO_EXPAND_THRESHOLD (0.25)`,
automatically expands and retries. Uses expanded result only if it beats original.

### 4.5 Re-ranking & Touch

When `rerank=true`: LLM (`rerank` model, `RERANK_SYSTEM`) re-orders results. Scores interpolated.
Only memories with `relevance > 0.5` get `touch()`-ed. Dry queries (`dry=true`) skip touch.

---

## 5. Audit System

### 5.1 Three Powers

The audit LLM has exactly three powers:

| Power | Operations | Description |
|-------|-----------|-------------|
| **Schedule** | `promote`, `demote` | Move memories between layers |
| **Adjust** | `adjust` | Change importance score (0.0–1.0) |
| **Merge** | `merge` | Combine 2+ memories into one |

**Audit CANNOT delete memories.** Deletion only happens through natural lifecycle (epoch-based decay, capacity cap).
Demote is restricted to one layer at a time (Core→Working or Working→Buffer, never Core→Buffer).

### 5.2 Audit Flow (`audit_memories` in `consolidate/audit.rs`)

1. Fetch all Core + Working memories (meta only, no embeddings).
2. Group into semantic clusters with merge similarity hints (cosine `AUDIT_MERGE_MIN_SIM`–`AUDIT_MERGE_MAX_SIM`).
3. Format with metadata: `[short_id] L{layer} imp={importance} ac={access} age={days}d mod={days}d tags=[...] (kind) content`.
4. LLM call (`audit` model, falls back to `gate` model) with `AUDIT_SYSTEM` prompt.
   Uses function calling with `audit_tool_schema()` for structured output.
5. Returns `Vec<RawAuditOp>` — operations use 8-char short IDs.
6. Resolve short IDs to full UUIDs (`resolve_audit_ops`).
7. **Sandbox check** before execution.

### 5.3 Sandbox (`sandbox_audit` in `consolidate/sandbox.rs`)

Every audit operation passes through `RuleChecker` before execution:

**Rules checked per operation:**

| Rule | Check | Fail condition |
|------|-------|---------------|
| Delete blocked | Audit has no delete power | Any delete op → always fail |
| Target exists | ID must resolve to a real memory | ID not found |
| No same-layer demote | Demote target must be lower than current | `current_layer <= target` |
| Importance bounds | Adjust value must be in 0.0–1.0 | Out of range |
| Merge minimum | Merge needs ≥2 source IDs | <2 IDs |
| Merge direction | Merged result must go to highest source layer | `target_layer < max(source_layers)` |
| Merge content length | Merged text must preserve information | `output_len < shortest_input / 2` |

**Grading:**
- Each operation gets `Good` (pass) or `Bad` (blocked).
- Only `Good` operations execute. Bad operations are skipped and logged.
- Overall score = `good_count / total_count`. Threshold: `SANDBOX_SAFETY_THRESHOLD (0.70)`.

---

## 6. Topiary — Topic Clustering

### 6.1 Overview

Topiary builds a hierarchical topic tree from memory embeddings. It organizes all memories (Core,
Working, and Buffer) into named topic clusters, providing an index for resume and drill-down recall
via `POST /topic`.

### 6.2 Architecture

```
memory write → embed queue → 500ms batch → embeddings stored
    → topiary trigger → 5s debounce → rebuild tree
    → name dirty topics (LLM) → cache in engram_meta

consolidation completes → topiary trigger → same flow
```

**Module structure:** `src/topiary/`

| File | Lines | Purpose |
|------|-------|---------|
| `mod.rs` | ~600 | TopicNode, TopicTree, insert, consolidate, hierarchy, helpers |
| `cluster.rs` | ~560 | k-means (spherical, seeded), split/merge passes, enforce_budget |
| `worker.rs` | ~220 | Debounced async background worker |
| `naming.rs` | ~210 | LLM batch naming via `ai::llm_tool_call` |

### 6.3 Data Flow

1. **Entry bridge:** engram `db::Memory` → `topiary::Entry { id, text, embedding: Vec<f32> }`.
   Only memories with embeddings are included.
2. **Insert:** Each entry assigned to closest leaf (cosine > `assign_threshold`) or creates a new
   singleton leaf. Centroid updated incrementally with L2 normalization.
3. **Consolidate:** Up to 10 split/merge cycles until stable, then `enforce_budget` (k-means if
   leaves exceed `LEAF_BUDGET=256`), hierarchy construction, small-leaf absorption, single-child
   pruning. Leaf IDs reassigned to sequential `kb1, kb2, ...`.
4. **Naming:** Dirty leaves (new/changed) batched in groups of 30, sent to LLM via `ai::llm_tool_call`
   with `TOPIC_NAMING_SYSTEM` prompt. Existing names provided as dedup context.
5. **Storage:** Two forms cached in `engram_meta`:
   - `topiary_tree`: JSON summary (topic IDs, names, member counts, sample texts)
   - `topiary_tree_full`: Full serialized `TopicTree` (with centroids, for future incremental updates)
   - `topiary_entry_ids`: Ordered entry ID list for index→ID resolution

### 6.4 Worker

`topiary::worker::spawn_worker(db, ai, trigger_rx)` — tokio task.

- **Trigger sources:** EmbedQueue flush completion, post-consolidation
- **Debounce:** 5 second quiet window (drains signals, resets on each new signal)
- **Startup:** If no `topiary_tree` in meta, immediately rebuilds
- **Input:** All Core + Working + Buffer memories with embeddings
- **Output:** Cached tree in `engram_meta`
- **Safety net:** If any topic remains unnamed after LLM naming, the tree is not saved — the
  previous cached tree is preserved instead. This prevents partially-named trees from replacing
  good cached state.

### 6.5 Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `assign_threshold` | 0.30 | Min cosine similarity to assign entry to existing leaf |
| `merge_threshold` | 0.55 | Min similarity between leaf centroids to merge |
| `max_leaf_size` | 8 | Split threshold for oversized leaves |
| `min_internal_sim` | 0.35 | Minimum intra-cluster similarity |
| `LEAF_BUDGET` | 256 | Max leaf count before k-means enforcement |
| `ABSORB_THRESHOLD` | 0.40 | Min similarity to absorb small (≤2 member) leaves |
| Debounce | 5s | Quiet window before rebuild |
| Naming batch | 30 | Max topics per LLM naming call |

### 6.6 POST /topic Endpoint

```
POST /topic
Body: {"ids": ["kb1", "kb3"]}
Response: {
  "kb1": {"name": "Memory architecture", "memories": [...]},
  "kb3": {"name": "User preferences", "memories": [...]}
}
```

Loads `topiary_tree_full` from meta, finds leaf by ID, resolves member indices to entry IDs
via `topiary_entry_ids`, fetches full memories from DB.

---

## 7. LLM Call Sites

**LLM Level** (`ENGRAM_LLM_LEVEL`, default `auto`): Controls when consolidation calls LLMs.
In `auto` mode, high-confidence decisions use heuristics; only uncertain cases invoke LLMs.
In `off` mode, triage/gate use pure heuristics and merge/reconcile are skipped.

Every place the system calls an LLM, with model tier and purpose:

| Call Site | Model Tier | Component Log | Purpose | Prompt |
|-----------|-----------|---------------|---------|--------|
| **Triage** | `gate` | `triage` | Evaluate buffer memories → promote/keep | `TRIAGE_SYSTEM` |
| **Gate** | `gate` | `gate` | Approve/reject Working→Core promotion | `GATE_SYSTEM` |
| **Merge** | `merge` | `merge` | Combine near-duplicate content | `MERGE_SYSTEM` |
| **Reconcile** | `merge` | `reconcile` | Judge UPDATE/ABSORB/KEEP + merge content in one call | `RECONCILE_PROMPT` |
| **Audit** | `audit` (→`gate`) | `audit` | Review all Core+Working, propose ops | `AUDIT_SYSTEM` |
| **Expand** | `expand` | `query_expand` | Generate alternative search queries | `EXPAND_PROMPT` |
| **Extract** | `extract` | `extract` | Extract memories from conversation text | `EXTRACT_SYSTEM_PROMPT` |
| **Fact extract** | `extract` | `fact_extract` | Extract (subject, predicate, object) triples | `FACT_EXTRACT_PROMPT` (disabled) |
| **Resume compress** | `resume` (→default) | `resume_compress` | Summarize section for context budget | Inline system prompt |
| **Insert merge** | `merge` | `insert_merge` | Merge on insert when high similarity | `INSERT_MERGE_PROMPT` |
| **Rerank** | `rerank` | `rerank` | Re-order recall results by relevance | `RERANK_SYSTEM` |
| **Distill** | `gate` | `distill` | Synthesize session notes into status | Inline system prompt |
| **Naming** | `naming` (→default) | `naming` | Name topic clusters for resume index | `TOPIC_NAMING_SYSTEM` |

Model tier resolution (`AiConfig::model_for`): each component checks its env var
(e.g. `ENGRAM_GATE_MODEL`), falls back to `ENGRAM_LLM_MODEL`, defaults to `gpt-4o-mini`.
`audit` falls back to `gate` model if no dedicated audit model.

All LLM calls are logged in `llm_usage` table with component, model, token counts, and duration.

---

## 7. Resume System

`GET /resume` provides session recovery context. Four sections, fixed budget.

### Namespace Merging

When a project namespace is set (e.g. `ns=engram-rs`), resume fetches from **both** the project
namespace and `default`. This ensures cross-project knowledge (user identity, preferences, universal
lessons stored in `default`) is always available alongside project-specific context.

Recall follows the same rule: queries with a namespace filter also include `default` results.

**Directional merge rule:** Consolidation merge and reconcile allow cross-namespace operations only
when one side is `default`. The merged result always stays in `default` — project-level memories can
be absorbed into `default`, but `default` memories are never pulled into a project namespace.

### Output Format

```
=== Core (N) ===
[full text, no truncation, up to ~2k tokens]

=== Recent (Nh) ===
[recently modified/created non-Core memories, time descending, up to ~1k tokens]

=== Topics (Working: N, Buffer: N) ===
kb1: "Topic name" [8]
kb2: "Another topic" [5]
...

Triggers: git-push, deploy, memory-store, ...
```

### Section Details

| Section | Content | Budget | Sort |
|---------|---------|--------|------|
| **Core** | Permanent rules/identity, full text | ~2k tokens | `importance × kind_boost × (1 + rep × 2.5)` |
| **Recent** | Non-Core memories modified in last N consolidation epochs | ~1k tokens | Time descending |
| **Topics** | Flat leaf index from cached topiary tree | ~300-500 tokens | Member count descending |
| **Triggers** | All `trigger:*` tags | ~100-200 tokens | Access count descending |

**Core sorting:** `kind_boost` = 1.3 for procedural, 1.0 for semantic, 0.8 for episodic. Procedural memories naturally
rank higher because they decay slowest AND get a 1.3× boost.

**Topics:** Read from `topiary_tree` in `engram_meta`. If no tree cached, section is omitted.
Agent can drill into any topic via `POST /topic {"ids": ["kb1", "kb3"]}`.

**Triggers:** Collected from all memories with tags matching `trigger:*` pattern, deduplicated,
sorted by aggregate access count.

---

## 8. Algorithms & Thresholds Reference

### All Cosine Similarity Thresholds

| Constant | Value | Used In |
|----------|-------|---------|
| `PROXY_DEDUP_SIM` | 0.60 | Proxy extraction dedup |
| `INSERT_DEDUP_SIM` | 0.65 | API insert dedup |
| `RECONCILE_MIN_SIM` | 0.55 | Reconcile: lower bound (related but not duplicate) |
| `AUDIT_MERGE_MIN_SIM` | 0.65 | Audit: merge suggestion lower bound |
| `CORE_OVERLAP_SIM` | 0.70 | Core overlap detection |
| `TRIAGE_DEDUP_SIM` | 0.75 | Triage + distill dedup, buffer dedup |
| `MERGE_SIM` | 0.78 | Auto-merge threshold, reconcile upper bound, audit merge upper bound |
| `RECONCILE_MAX_SIM` | 0.78 | = MERGE_SIM |
| `AUDIT_MERGE_MAX_SIM` | 0.78 | = MERGE_SIM |
| `INSERT_MERGE_SIM` | 0.80 | Insert-time content merge |
| `DEDUP_COSINE_SIM` | 0.85 | DB-level last-resort dedup |
| `RECALL_DEDUP_SIM` | 0.78 | = MERGE_SIM, used in quick_semantic_dup |

### Other Thresholds

| Constant | Value | Used In |
|----------|-------|---------|
| `DEDUP_JACCARD_PREFILTER` | 0.50 | Pre-filter before cosine in DB dedup |
| `SANDBOX_SAFETY_THRESHOLD` | 0.70 | Min safety score to execute audit ops |
| `SANDBOX_RECENT_MOD_HOURS` | 24.0h | "Recently modified" guard in audit sandbox |
| `SANDBOX_NEW_AGE_HOURS` | 48.0h | "New memory" protection in audit sandbox |
| `BUFFER_CAP` | 200 (env) | Max buffer entries before capacity-based eviction |
| `RESUME_HIGH_RELEVANCE` | 0.25 | Identity/constraint boost in resume |
| `RESUME_LOW_RELEVANCE` | 0.10 | Lesson/procedural boost in resume |
| `RESUME_CORE_THRESHOLD` | 0.35 | Min relevance to include Core in resume |
| `RESUME_WORKING_THRESHOLD` | 0.20 | Min relevance to include Working in resume |

### Recall Scoring Weights

| Weight | Value |
|--------|-------|
| `WEIGHT_RELEVANCE` | 0.60 |
| `WEIGHT_IMPORTANCE` | 0.20 |
| `WEIGHT_RECENCY` | 0.20 |
| `AUTO_EXPAND_THRESHOLD` | 0.25 |
| `DEFAULT_SIM_FLOOR` | 0.30 |

### Consolidation Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `promote_threshold` | 3 | Reinforcement score for Working→Core candidacy |
| `promote_min_importance` | 0.6 | Min importance for any promotion |
| `decay_drop_threshold` | 0.01 | Below this → delete (Buffer only; Working/Core never deleted) |
| `BUFFER_CAP` | 200 (env) | Max buffer entries before FIFO eviction |
| `working_age_promote_secs` | 604800 (7d) | Age-based Working→Core candidacy |
| `buffer_threshold` | max(promote_threshold, 5) = 5 | Reinforcement score for Buffer→Working |
| `BUFFER_CAP` | 200 (env) | Max buffer entries before FIFO eviction |
| `importance_decay` | 0.005/epoch, floor 0.0 | Epoch-based importance reduction (per active consolidation cycle) |
| `REPETITION_WEIGHT` | 2.5 | Repetition counts 2.5× more than access in reinforcement |

---

## 9. Indexing

### FTS5

- SQLite FTS5 with `unicode61` tokenizer.
- CJK text pre-processed: Chinese via jieba segmentation, Japanese/Korean via bigrams.
- Content and tags indexed. Rebuilt on startup.
- Manual insert/delete/update (no triggers).

### Vector Index (HNSW)

- In-memory `VecIndex` using `instant-distance` HNSW.
- Loaded from SQLite embedding BLOBs on startup.
- Supports filtered search (`search_semantic_by_ids`) and full scan (`search_semantic_ns`).
- Embeddings stored as f32 arrays serialized to bytes in SQLite BLOB column.

### Fact Graph

- `facts` table: `(subject, predicate, object, memory_id)`.
- FTS on subject/object for entity lookup.
- Multi-hop traversal (`query_multihop`) follows entity→fact→entity chains up to N hops.
- Fact extraction currently disabled in consolidation (low quality), but API available.

---

## 10. Key Invariants

1. **Memories only move up layers through LLM review** (triage or gate). Mechanical promotion
   exists for Buffer→Working only (high reinforcement score or lesson/procedural kind).
2. **Audit cannot delete** — only schedule (promote/demote), adjust importance, or merge.
   Natural lifecycle (epoch-based decay, capacity cap) handles cleanup.
3. **Demotions are one-layer-at-a-time** — Core→Working or Working→Buffer, never skip.
4. **Working memories are never deleted** — importance decays at different rates by kind, but memories
   remain searchable even at zero importance. Only Buffer entries can be evicted.
5. **Repetition > access for importance signal** — `REPETITION_WEIGHT = 2.5` means being restated
   counts 2.5× more than being recalled.
6. **Session notes never promote to Core** — they're episodic by nature. Instead, they're distilled
   into project status snapshots that can promote normally.
7. **All LLM decisions are logged** — component, model, token usage, and duration tracked in `llm_usage`.
8. **Gate rejections escalate** — 3 chances with increasing epoch-based cooldowns (48, 144, never),
   preventing infinite retry loops while ensuring idle time doesn't auto-grant retries.
9. **Resume doesn't touch memories** — only recall (query-driven) increments access counts.
10. **Merge direction is always upward** — cross-layer merge result goes to the highest source layer.
11. **Topiary is eventually consistent** — tree rebuilds are debounced and async. Resume reads a
    cached snapshot; it may lag a few seconds behind the latest writes. This is intentional — resume
    must be fast, and the tree is an index, not a source of truth.
