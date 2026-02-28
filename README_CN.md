# engram

为 AI agent 提供持久化记忆，沿时间与空间两个维度自动组织。高价值记忆逐层提升，低价值记忆自然淘汰，相关知识自动聚类为可浏览的主题树。全程无需人工干预。

<p align="center">
  <img src="docs/engram-quickstart.gif" alt="engram 演示 — 存储、上下文重置、检索" width="720">
</p>

## 快速开始

```bash
# 安装并启动
curl -fsSL https://raw.githubusercontent.com/kael-bit/engram-rs/main/install.sh | bash

# 写入一条记忆
curl -X POST http://localhost:3917/memories \
  -d '{"content": "部署前必须跑测试", "tags": ["deploy"]}'

# 语义检索
curl -X POST http://localhost:3917/recall \
  -d '{"query": "部署流程"}'

# 恢复完整上下文（用于会话启动）
curl http://localhost:3917/resume
```

## 核心特性

多数 agent 记忆方案提供的是带搜索的向量存储。engram 在此基础上引入了**生命周期管理**——记忆不仅被存储，还会随使用情况持续演化。

### LLM 质量门控

新记忆首先进入 Buffer 层。晋升至 Working 或 Core 层需要通过 LLM 质量门控——LLM 会结合上下文判断该记忆是否具有长期保留价值，而非简单的关键词匹配。

```
Buffer → [LLM 门控："属于决策、教训还是偏好？"] → Working
Working → [持续访问 + LLM 门控] → Core
```

### 语义去重与合并

当两条记忆表达相同含义但措辞不同时，engram 能够自动检测并合并：

```
记忆 A："认证使用 PostgreSQL"
记忆 B："认证服务跑在 Postgres 上"
→ 整理后：合并为一条，保留双方上下文
```

合并过程由 LLM 驱动，基于语义理解而非字符串比较。

### 自动衰减

衰减基于活动周期（epoch）触发，仅在系统存在写入活动的 consolidation 周期中执行。系统空闲期间不产生任何衰减。

| 类型 | 衰减速率 | 适用场景 |
|------|---------|---------|
| `episodic` | 较快 | 事件、经历等时效性内容 |
| `semantic` | 较慢 | 知识、偏好、经验教训（默认类型） |
| `procedural` | 最慢 | 工作流、操作规范 |

Working 和 Core 层的记忆均不会被删除。Working 层中记忆的重要性会逐步降低，但始终可通过检索访问。Buffer 层为临时暂存区，其中的记忆可被正常淘汰。

### 自组织主题树

基于向量聚类，自动将语义相关的记忆归入同一主题。主题树为层次结构，由 LLM 自动命名：

```
Memory Architecture
├── Three-layer lifecycle [4]
├── Embedding pipeline [3]
└── Consolidation logic [5]
Deploy & Ops
├── CI/CD procedures [3]
└── Production incidents [2]
User Preferences [6]
```

当记忆发生变更时主题树自动重建。会话启动时，agent 获取主题索引作为知识目录，可通过 `POST /topic {"ids": ["kb3"]}` 展开特定主题下的全部记忆。

### 触发器

为记忆添加 `trigger:deploy` 标签后，agent 在执行部署前调用 `/triggers/deploy` 即可自动召回相关经验教训。

```bash
# 写入一条经验教训
curl -X POST http://localhost:3917/memories \
  -d '{"content": "LESSON: 迁移前必须备份数据库", "tags": ["trigger:deploy", "lesson"]}'

# 部署前查询相关教训
curl http://localhost:3917/triggers/deploy
# → 返回所有标记为 trigger:deploy 的记忆，按访问频次排序
```

## 架构

记忆沿**时间**和**空间**两个维度组织：

```
         Time (lifecycle)                    Space (topic tree)
┌─────────────────────────────┐    ┌──────────────────────────────┐
│                             │    │ Auth Architecture            │
│  Buffer → Working → Core    │    │ ├── OAuth2 migration [3]     │
│    ↓         ↓        ↑     │    │ └── Token handling [2]       │
│  evict     decay    gate    │    │ Deploy & Ops                 │
│                             │    │ ├── CI/CD procedures [3]     │
│                             │    │ └── Rollback lessons [2]     │
└─────────────────────────────┘    │ User Preferences [6]         │
                                   └──────────────────────────────┘
```

**时间维度** — 参考 [Atkinson–Shiffrin 记忆模型](https://en.wikipedia.org/wiki/Atkinson%E2%80%93Shiffrin_memory_model)设计的三层生命周期：

| 层级 | 定位 | 行为 |
|------|------|------|
| **Buffer** | 短期暂存 | 所有新记忆的入口。低于阈值时被淘汰 |
| **Working** | 活跃知识 | 由 consolidation 从 Buffer 提升。不会被删除，重要性按类型以不同速率衰减 |
| **Core** | 长期身份 | 通过 LLM 质量门控从 Working 提升。不会被删除 |

**空间维度** — 基于 embedding 向量构建的自组织主题树。语义相近的记忆自动聚类，由 LLM 为各聚类生成可读名称：

| 机制 | 说明 |
|------|------|
| **向量聚类** | 按余弦相似度将语义相关的记忆归入同一主题 |
| **层次结构** | 相关主题嵌套于共同父节点下，形成多级树形结构 |
| **LLM 命名** | 自动为每个聚类生成人类可读的名称 |
| **自动重建** | 记忆变更时主题树自动更新 |

主题树解决的问题：向量搜索依赖精确的查询词，而主题树提供了按主题浏览的能力——先查看目录，再展开目标分支。

## 会话恢复

单次调用即可恢复完整上下文，适用于启动或上下文压缩后的场景：

```
GET /resume →

=== Core (24) ===
部署流程：测试 → 构建 → 停服 → 启动 (procedural)
LESSON: 不要 force-push 到 main
...

=== Recent ===
[02-27 14:15] 认证切换到 OAuth2
[02-27 11:01] 发布了 API 文档

=== Topics (Core: 24, Working: 57, Buffer: 7) ===
kb1: "部署流程" [5]
kb2: "认证架构" [3]
kb3: "记忆系统设计" [8]
...

=== Triggers ===
deploy, git-push, database-migration
```

四个区块各有侧重：

| 区块 | 内容 | 预算 |
|------|------|------|
| **Core** | 永久性规则与身份信息的完整文本，不做截断 | ~2k tokens |
| **Recent** | 最近活跃周期内变更的记忆，用于短期上下文连续性 | ~1k tokens |
| **Topics** | 主题索引，即全部记忆的结构化目录 | 叶子列表 |
| **Triggers** | 操作前置检查标签，自动召回相关经验教训 | 标签列表 |

Agent 通过主题索引定位相关主题，再调用 `POST /topic` 按需展开，无需将整个记忆库加载到上下文中。

## 搜索与检索

混合检索机制：语义 embedding + BM25 关键词搜索（通过 [jieba](https://github.com/messense/jieba-rs) 支持中日韩分词）。结果综合相关度、记忆重要性和时效性进行排序。

```bash
# 语义搜索，支持 token 预算控制
curl -X POST http://localhost:3917/recall \
  -d '{"query": "认证怎么做的", "budget_tokens": 2000}'

# 操作前置安全检查
curl http://localhost:3917/triggers/deploy

# 主题展开
curl -X POST http://localhost:3917/topic \
  -d '{"ids": ["kb3"]}'
```

## 后台维护

全自动运行，按活动驱动——无写入操作时自动跳过当前周期：

### Consolidation（每 30 分钟）

每个周期依次执行以下步骤：

1. **衰减** — 降低未被访问记忆的重要性
2. **去重** — 检测并合并近似记忆（cosine > 0.78）
3. **分类** — LLM 对新进入 Buffer 的记忆进行分类和提升决策
4. **门控** — LLM 批量评估晋升候选（单次调用）
5. **调和** — LLM 处理模糊相似的记忆对，结果缓存以避免重复计算
6. **主题树重建** — 重新聚类并为新增或变更的主题命名

### 主题蒸馏

当某个主题聚类的记忆过多（≥10 条）时，engram 会将重叠的记忆合并为更少、更丰富的条目——保留所有具体细节的同时减少冗余。每个整合周期最多处理 2 个主题。

## 命名空间隔离

单个 engram 实例支持多项目。每个命名空间拥有独立的记忆空间：

```bash
# 项目专属记忆
curl -X POST http://localhost:3917/memories \
  -H "X-Namespace: my-project" \
  -d '{"content": "API 使用 OAuth2 Bearer Token"}'

# 跨项目通用知识存入默认命名空间
curl -X POST http://localhost:3917/memories \
  -d '{"content": "时间戳统一使用 UTC"}'
```

## 安装

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/kael-bit/engram-rs/main/install.sh | bash

# Windows (PowerShell)
irm https://raw.githubusercontent.com/kael-bit/engram-rs/main/install.ps1 | iex
```

### Docker

```bash
docker run -d --name engram \
  -p 3917:3917 \
  -v engram-data:/data \
  -e ENGRAM_LLM_URL=https://api.openai.com/v1 \
  -e ENGRAM_LLM_KEY=sk-... \
  ghcr.io/kael-bit/engram-rs:latest
```

### LLM 配置

engram 可在无 LLM 的环境下运行，仅支持关键词搜索和基于规则的生命周期管理。接入 LLM 后可解锁智能特性（分类、门控、合并、主题命名、审计）：

```bash
export ENGRAM_LLM_URL=https://api.openai.com/v1
export ENGRAM_LLM_KEY=sk-...

# 支持任意 OpenAI 兼容的 API（Ollama、LM Studio 等）
export ENGRAM_LLM_URL=http://localhost:11434/v1

# Anthropic 原生接入
export ENGRAM_LLM_PROVIDER=anthropic
export ENGRAM_LLM_KEY=sk-ant-...
```

支持双模型配置——判断类任务使用强模型，文本处理任务使用轻量模型：

```bash
ENGRAM_GATE_MODEL=gpt-4o          # 质量判断
ENGRAM_LLM_MODEL=gpt-4o-mini     # 文本处理
```

### 接入 AI Agent

将以下内容添加到 agent 的系统提示词或会话中：

```
Set up engram (persistent memory) by following the guide at:
https://raw.githubusercontent.com/kael-bit/engram-rs/main/docs/SETUP.md
```

## 集成

支持 Claude Code、Cursor、Windsurf、OpenClaw 及所有 MCP 兼容工具。

提供 17 个 MCP 工具（详见 [MCP 文档](docs/MCP.md)）和完整的 HTTP API（详见[配置指南](docs/SETUP.md)）。

```bash
# MCP (Claude Code)
npx engram-rs-mcp

# MCP (Cursor / Windsurf / 通用)
# 在 MCP 配置文件中添加：
{"mcpServers": {"engram": {"command": "npx", "args": ["-y", "engram-rs-mcp"]}}}
```

### Web 控制台

内置 Web UI，访问 `http://localhost:3917`，可浏览记忆、查看主题树、监控 LLM 调用量及整理历史。

## 技术规格

| | |
|---|---|
| 二进制大小 | ~10 MB |
| 内存占用 | 生产环境约 100 MB RSS |
| 存储 | SQLite，无外部数据库依赖 |
| 语言 | Rust |
| 平台 | Linux、macOS、Windows（x86_64 + aarch64） |
| 协议 | MIT |

## License

MIT

<a href="https://glama.ai/mcp/servers/@kael-bit/engram-rs">
  <img width="380" height="200" src="https://glama.ai/mcp/servers/@kael-bit/engram-rs/badge" />
</a>
