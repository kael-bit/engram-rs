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


const server = new McpServer({
  name: "engram",
  version: "0.1.0",
});

// -- engram_store --

server.tool(
  "engram_store",
  "Store a memory in the engram hierarchical memory engine. " +
    "Layer 1 = buffer (transient), 2 = working (active), 3 = core (permanent).",
  {
    content: z.string().describe("Memory content text"),
    importance: z
      .number()
      .min(0)
      .max(1)
      .optional()
      .describe("Importance score 0-1 (default ~0.5)"),
    layer: z
      .number()
      .int()
      .min(1)
      .max(3)
      .optional()
      .describe("Memory layer: 1=buffer, 2=working, 3=core"),
    tags: z
      .array(z.string())
      .optional()
      .describe("Optional tags for categorization"),
    source: z.string().optional().describe("Source identifier"),
  },
  async ({ content, importance, layer, tags, source }) => {
    const body: Record<string, unknown> = { content };
    if (importance !== undefined) body.importance = importance;
    if (layer !== undefined) body.layer = layer;
    if (tags !== undefined) body.tags = tags;
    if (source !== undefined) body.source = source;

    const result = await engramFetch("/memories", body);
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  }
);

// -- engram_recall --

server.tool(
  "engram_recall",
  "Recall memories from engram using hybrid semantic + keyword search. " +
    "Supports budget-aware retrieval — specify a token budget and get the " +
    "most relevant memories that fit.",
  {
    query: z.string().describe("Search query"),
    budget_tokens: z
      .number()
      .int()
      .positive()
      .optional()
      .describe("Max token budget for returned memories"),
    layers: z
      .array(z.number().int().min(1).max(3))
      .optional()
      .describe("Filter by layers (e.g. [2, 3])"),
    min_importance: z
      .number()
      .min(0)
      .max(1)
      .optional()
      .describe("Minimum importance threshold"),
    limit: z
      .number()
      .int()
      .positive()
      .optional()
      .describe("Max number of memories to return"),
  },
  async ({ query, budget_tokens, layers, min_importance, limit }) => {
    const body: Record<string, unknown> = { query };
    if (budget_tokens !== undefined) body.budget_tokens = budget_tokens;
    if (layers !== undefined) body.layers = layers;
    if (min_importance !== undefined) body.min_importance = min_importance;
    if (limit !== undefined) body.limit = limit;

    const result = await engramFetch("/recall", body);
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  }
);

// -- engram_extract --

server.tool(
  "engram_extract",
  "Extract structured memories from raw text using LLM. " +
    "Feed conversation logs or notes, and engram will identify and store " +
    "individual memories with appropriate importance and layer assignments.",
  {
    text: z.string().describe("Raw text to extract memories from"),
    auto_embed: z
      .boolean()
      .optional()
      .describe("Generate embeddings for extracted memories (default true)"),
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

// -- engram_search --

server.tool(
  "engram_search",
  "Quick keyword search across memories. Lighter than recall — " +
    "no scoring, no budget logic, just find memories matching a term.",
  {
    q: z.string().describe("Search query"),
    limit: z.number().int().min(1).max(50).optional().describe("Max results (default 10)"),
  },
  async ({ q, limit }) => {
    const params = new URLSearchParams({ q });
    if (limit !== undefined) params.set("limit", String(limit));
    const result = await engramFetch(`/search?${params}`);
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  }
);

// -- engram_consolidate --

server.tool(
  "engram_consolidate",
  "Run a memory consolidation cycle. Promotes frequently-accessed important " +
    "memories from working (L2) to core (L3), and cleans up decayed buffer entries.",
  {
    promote_threshold: z
      .number()
      .int()
      .positive()
      .optional()
      .describe("Min access count to promote L2→L3 (default 3)"),
    promote_min_importance: z
      .number()
      .min(0)
      .max(1)
      .optional()
      .describe("Min importance to promote (default 0.6)"),
    decay_drop_threshold: z
      .number()
      .min(0)
      .max(1)
      .optional()
      .describe("Drop memories with recency score below this (default 0.01)"),
  },
  async ({ promote_threshold, promote_min_importance, decay_drop_threshold }) => {
    const body: Record<string, unknown> = {};
    if (promote_threshold !== undefined) body.promote_threshold = promote_threshold;
    if (promote_min_importance !== undefined)
      body.promote_min_importance = promote_min_importance;
    if (decay_drop_threshold !== undefined)
      body.decay_drop_threshold = decay_drop_threshold;

    const hasBody = Object.keys(body).length > 0;
    const result = await engramFetch("/consolidate", hasBody ? body : undefined);
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
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
