#!/usr/bin/env node

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const ENGRAM_URL = process.env.ENGRAM_URL || "http://localhost:3917";
const ENGRAM_API_KEY = process.env.ENGRAM_API_KEY || "";

async function engramFetch(path: string, body?: unknown): Promise<unknown> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (ENGRAM_API_KEY) {
    headers["Authorization"] = `Bearer ${ENGRAM_API_KEY}`;
  }

  const method = body !== undefined ? "POST" : "GET";
  const resp = await fetch(`${ENGRAM_URL}${path}`, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

  const text = await resp.text();
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    data = text;
  }

  if (!resp.ok) {
    const msg = typeof data === "object" && data && "error" in data
      ? (data as { error: string }).error
      : text;
    throw new Error(`engram ${path} failed (${resp.status}): ${msg}`);
  }

  return data;
}

async function engramMethod(method: string, path: string, body?: unknown): Promise<unknown> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (ENGRAM_API_KEY) {
    headers["Authorization"] = `Bearer ${ENGRAM_API_KEY}`;
  }
  const resp = await fetch(`${ENGRAM_URL}${path}`, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const text = await resp.text();
  let data: unknown;
  try { data = JSON.parse(text); } catch { data = text; }
  if (!resp.ok) {
    const msg = typeof data === "object" && data && "error" in data
      ? (data as { error: string }).error : text;
    throw new Error(`engram ${method} ${path} failed (${resp.status}): ${msg}`);
  }
  return data;
}


const server = new McpServer({
  name: "engram",
  version: "0.7.0",
});

server.tool(
  "engram_store",
  "Store a memory. All memories start in Buffer and promote to Working/Core through " +
    "access frequency and LLM quality gating. Procedural memories and lessons (tag=lesson) " +
    "auto-promote to Working after 2h. Use supersedes to replace outdated memories by their ids.",
  {
    content: z.string().describe("Memory content text"),
    importance: z.number().min(0).max(1).optional().describe("Importance 0-1 (default 0.5, bumps +0.02 on each recall)"),
    kind: z.enum(["semantic", "episodic", "procedural"]).optional().describe("Memory type: semantic (facts, default), episodic (events), procedural (how-to, persists indefinitely)"),
    tags: z.array(z.string()).optional().describe("Tags for categorization"),
    source: z.string().optional().describe("Source identifier"),
    supersedes: z.array(z.string()).optional().describe("IDs of old memories this one replaces"),
    skip_dedup: z.boolean().optional().describe("Skip near-duplicate detection"),
    namespace: z.string().optional().describe("Namespace for multi-agent isolation"),
    sync_embed: z.boolean().optional().describe("Wait for embedding generation before returning (default: false)"),
  },
  async ({ content, importance, kind, tags, source, supersedes, skip_dedup, namespace, sync_embed }) => {
    const body: Record<string, unknown> = { content };
    if (importance !== undefined) body.importance = importance;
    if (kind !== undefined) body.kind = kind;
    if (tags !== undefined) body.tags = tags;
    if (source !== undefined) body.source = source;
    if (supersedes !== undefined) body.supersedes = supersedes;
    if (skip_dedup !== undefined) body.skip_dedup = skip_dedup;
    if (namespace !== undefined) body.namespace = namespace;
    if (sync_embed !== undefined) body.sync_embed = sync_embed;

    const result = await engramFetch("/memories", body);
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  }
);

server.tool(
  "engram_recall",
  "Hybrid semantic + keyword search with budget-aware retrieval. " +
    "Supports time filtering, source/tag filtering, and LLM re-ranking.",
  {
    query: z.string().describe("Search query"),
    budget_tokens: z.number().int().positive().optional().describe("Max token budget"),
    layers: z.array(z.number().int().min(1).max(3)).optional().describe("Filter by layers"),
    min_importance: z.number().min(0).max(1).optional().describe("Min importance threshold"),
    limit: z.number().int().positive().optional().describe("Max results"),
    since: z.number().int().optional().describe("Only memories created after this unix ms timestamp"),
    until: z.number().int().optional().describe("Only memories created before this unix ms timestamp"),
    sort_by: z.enum(["score", "recent", "accessed"]).optional().describe("Sort order (default: score)"),
    rerank: z.boolean().optional().describe("Re-rank results using LLM"),
    expand: z.boolean().optional().describe("Expand query with LLM-generated synonyms for better recall"),
    source: z.string().optional().describe("Filter by source (e.g. session, extract, api)"),
    tags: z.array(z.string()).optional().describe("Filter by tags (must have ALL specified)"),
    namespace: z.string().optional().describe("Filter by namespace"),
    min_score: z.number().min(0).max(1).optional().describe("Drop results below this score (0-1)"),
    dry: z.boolean().optional().describe("Skip touch/reinforcement on results (for background queries)"),
  },
  async ({ query, budget_tokens, layers, min_importance, limit, since, until, sort_by, rerank, expand, source, tags, namespace, min_score, dry }) => {
    const body: Record<string, unknown> = { query };
    if (budget_tokens !== undefined) body.budget_tokens = budget_tokens;
    if (layers !== undefined) body.layers = layers;
    if (min_importance !== undefined) body.min_importance = min_importance;
    if (limit !== undefined) body.limit = limit;
    if (since !== undefined) body.since = since;
    if (until !== undefined) body.until = until;
    if (sort_by !== undefined) body.sort_by = sort_by;
    if (rerank !== undefined) body.rerank = rerank;
    if (expand !== undefined) body.expand = expand;
    if (source !== undefined) body.source = source;
    if (tags !== undefined) body.tags = tags;
    if (namespace !== undefined) body.namespace = namespace;
    if (min_score !== undefined) body.min_score = min_score;
    if (dry !== undefined) body.dry = dry;

    const result = await engramFetch("/recall", body);
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  }
);

server.tool(
  "engram_recent",
  "List recent memories by creation time. Good for session context recovery.",
  {
    hours: z.number().positive().optional().describe("Look back N hours (default 2)"),
    limit: z.number().int().min(1).max(100).optional().describe("Max results (default 20)"),
    layer: z.number().int().min(1).max(3).optional().describe("Filter by layer"),
    source: z.string().optional().describe("Filter by source (e.g. session)"),
    min_importance: z.number().min(0).max(1).optional().describe("Minimum importance threshold"),
    namespace: z.string().optional().describe("Filter by namespace"),
  },
  async ({ hours, limit, layer, source, min_importance, namespace }) => {
    const params = new URLSearchParams();
    if (hours !== undefined) params.set("hours", String(hours));
    if (limit !== undefined) params.set("limit", String(limit));
    if (layer !== undefined) params.set("layer", String(layer));
    if (source !== undefined) params.set("source", source);
    if (min_importance !== undefined) params.set("min_importance", String(min_importance));
    if (namespace !== undefined) params.set("ns", namespace);
    const qs = params.toString();
    const result = await engramFetch(`/recent${qs ? "?" + qs : ""}`);
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  }
);

server.tool(
  "engram_resume",
  "Full memory bootstrap for session recovery. Returns core (permanent knowledge), " +
    "working (ongoing context/decisions), buffer (transient), recent activity, and " +
    "session notes. Use workspace tags to filter by current work context. " +
    "Compact mode (default) minimizes token usage.",
  {
    hours: z.number().positive().optional().describe("Look back N hours for recent/sessions (default 4)"),
    namespace: z.string().optional().describe("Filter by namespace"),
    workspace: z.string().optional().describe("Comma-separated workspace tags to filter core/working memories (e.g. 'engram,rust')"),
    compact: z.boolean().optional().describe("Compact output: content+tags only (default true)"),
    budget: z.number().optional().describe("Max chars across all sections (default 8000, 0=unlimited)"),
    limit: z.number().optional().describe("Max memories per layer (default 100)"),
  },
  async ({ hours, namespace, workspace, compact, budget, limit }) => {
    const params = new URLSearchParams();
    if (hours !== undefined) params.set("hours", String(hours));
    if (namespace !== undefined) params.set("ns", namespace);
    if (workspace !== undefined) params.set("workspace", workspace);
    if (compact !== undefined) params.set("compact", String(compact));
    if (budget !== undefined) params.set("budget", String(budget));
    if (limit !== undefined) params.set("limit", String(limit));
    const qs = params.toString();
    const result = await engramFetch(`/resume${qs ? "?" + qs : ""}`);
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  }
);

server.tool(
  "engram_extract",
  "Extract structured memories from raw text using LLM. " +
    "Feed conversation logs or notes and get individual memories.",
  {
    text: z.string().describe("Raw text to extract memories from"),
    auto_embed: z.boolean().optional().describe("Generate embeddings (default true)"),
  },
  async ({ text, auto_embed }) => {
    const body: Record<string, unknown> = { text };
    if (auto_embed !== undefined) body.auto_embed = auto_embed;

    const result = await engramFetch("/extract", body);
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  }
);

server.tool(
  "engram_search",
  "Quick keyword search. Lighter than recall — no scoring or budget logic.",
  {
    q: z.string().describe("Search query"),
    limit: z.number().int().min(1).max(50).optional().describe("Max results (default 10)"),
    namespace: z.string().optional().describe("Filter by namespace"),
  },
  async ({ q, limit, namespace }) => {
    const params = new URLSearchParams({ q });
    if (limit !== undefined) params.set("limit", String(limit));
    if (namespace !== undefined) params.set("ns", namespace);
    const result = await engramFetch(`/search?${params}`);
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  }
);

server.tool(
  "engram_consolidate",
  "Run a memory consolidation cycle. Promotes important memories upward, " +
    "drops decayed entries. With merge=true, uses LLM to merge similar memories.",
  {
    merge: z.boolean().optional().describe("Use LLM to merge similar memories"),
    promote_threshold: z.number().int().positive().optional().describe("Min access count to promote (default 3)"),
    promote_min_importance: z.number().min(0).max(1).optional().describe("Min importance to promote (default 0.6)"),
    decay_drop_threshold: z.number().min(0).max(1).optional().describe("Drop below this recency (default 0.01)"),
  },
  async ({ merge, promote_threshold, promote_min_importance, decay_drop_threshold }) => {
    const body: Record<string, unknown> = {};
    if (merge !== undefined) body.merge = merge;
    if (promote_threshold !== undefined) body.promote_threshold = promote_threshold;
    if (promote_min_importance !== undefined) body.promote_min_importance = promote_min_importance;
    if (decay_drop_threshold !== undefined) body.decay_drop_threshold = decay_drop_threshold;

    const result = await engramFetch("/consolidate", body);
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  }
);

server.tool(
  "engram_stats",
  "Get memory statistics: counts per layer, AI status, version.",
  {},
  async () => {
    const result = await engramFetch("/stats");
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  }
);

server.tool(
  "engram_repair",
  "Repair FTS search index. Removes orphaned entries and rebuilds missing ones. " +
    "Safe to run anytime — idempotent.",
  {},
  async () => {
    const result = await engramFetch("/repair", {});
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  }
);

server.tool(
  "engram_health",
  "Detailed health check: uptime, RSS memory, embed cache stats, AI config status.",
  {},
  async () => {
    const result = await engramFetch("/health");
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  }
);

server.tool(
  "engram_triggers",
  "Fetch trigger memories for a specific action. Call before performing an action (e.g. git-push, deploy) to recall relevant lessons and rules.",
  { action: z.string().describe("Action name (e.g. git-push, deploy, send-message)") },
  async ({ action }: { action: string }) => {
    const result = await engramFetch(`/triggers/${encodeURIComponent(action)}`);
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  }
);

server.tool(
  "engram_delete",
  "Delete a memory by ID. Use when a memory is outdated, incorrect, or redundant.",
  { id: z.string().describe("Memory ID to delete") },
  async ({ id }: { id: string }) => {
    const result = await engramMethod("DELETE", `/memories/${encodeURIComponent(id)}`);
    return {
      content: [{ type: "text", text: JSON.stringify(result) }],
    };
  }
);

server.tool(
  "engram_update",
  "Update a memory's content, metadata (importance, tags, layer, kind). Use supersedes on store to replace with history.",
  {
    id: z.string().describe("Memory ID to update"),
    content: z.string().optional().describe("New content text"),
    importance: z.number().min(0).max(1).optional().describe("New importance value"),
    tags: z.array(z.string()).optional().describe("Replace tags"),
    layer: z.number().int().min(1).max(3).optional().describe("Move to layer: 1=buffer, 2=working, 3=core"),
    kind: z.enum(["semantic", "episodic", "procedural"]).optional().describe("Memory type"),
  },
  async ({ id, content, importance, tags, layer, kind }: { id: string; content?: string; importance?: number; tags?: string[]; layer?: number; kind?: string }) => {
    const body: Record<string, unknown> = {};
    if (content !== undefined) body.content = content;
    if (importance !== undefined) body.importance = importance;
    if (tags !== undefined) body.tags = tags;
    if (layer !== undefined) body.layer = layer;
    if (kind !== undefined) body.kind = kind;
    const result = await engramMethod("PATCH", `/memories/${encodeURIComponent(id)}`, body);
    return {
      content: [{ type: "text", text: JSON.stringify(result) }],
    };
  }
);

server.tool(
  "engram_trash",
  "List soft-deleted memories. Recently deleted items can be restored.",
  {
    limit: z.number().int().min(1).max(200).optional().describe("Max results (default 100)"),
  },
  async ({ limit }) => {
    const params = new URLSearchParams();
    if (limit !== undefined) params.set("limit", String(limit));
    const qs = params.toString();
    const result = await engramFetch(`/trash${qs ? "?" + qs : ""}`);
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  }
);

server.tool(
  "engram_restore",
  "Restore a soft-deleted memory from trash back to its original layer.",
  { id: z.string().describe("Memory ID to restore from trash") },
  async ({ id }: { id: string }) => {
    const result = await engramMethod("POST", `/trash/${encodeURIComponent(id)}/restore`);
    return {
      content: [{ type: "text", text: JSON.stringify(result) }],
    };
  }
);


async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("engram MCP server running (engram at %s)", ENGRAM_URL);
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
