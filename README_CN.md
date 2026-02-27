# engram

AI agent 的持久记忆——沿时间与空间两个维度自组织。重要的记忆被提升，噪音自然衰减，相关知识聚类成可浏览的主题树。全自动。

<p align="center">
  <img src="docs/engram-quickstart.gif" alt="engram 演示 — 存储、上下文重置、检索" width="720">
</p>

## 快速开始

```bash
# 安装启动
curl -fsSL https://raw.githubusercontent.com/kael-bit/engram-rs/main/install.sh | bash

# 存一条记忆
curl -X POST http://localhost:3917/memories \
  -d '{"content": "部署前必须跑测试", "tags": ["deploy"]}'

# 按语义检索
curl -X POST http://localhost:3917/recall \
  -d '{"query": "部署流程"}'

# 恢复完整上下文（会话开始时调）
curl http://localhost:3917/resume
```

## engram 做了什么不一样的事

大多数 agent 记忆工具给你的是一个带搜索的向量库。engram 在这之上加了**生命周期**——记忆不只是被存储，还会随时间被管理。

### LLM 质量门控

新记忆先进 Buffer。要升到 Working 或 Core，需要通过 LLM 质量门控——不是关键词匹配，LLM 会读上下文判断这条记忆值不值得长期留着。

```
Buffer → [LLM 门控："这是决策、教训还是偏好？"] → Working
Working → [持续访问 + LLM 门控] → Core（永久）
```

### 语义去重与合并

两条记忆说的是同一件事但措辞不同，engram 会检测并合并：

```
记忆 A："用 PostgreSQL 做认证"
记忆 B："认证服务跑在 Postgres 上"
→ 整理后：合并成一条，保留两边的上下文
```

合并由 LLM 驱动——理解含义，不只比较文本。

### 自动衰减

不被访问的记忆重要性会随时间降低。但不是一刀切：

| 类型 | 衰减 | 用途 |
|------|------|------|
| `semantic` | 正常 | 知识、偏好、决策（默认） |
| `episodic` | 正常 | 事件、经历、时效性上下文 |
| `procedural` | 永不 | 工作流、指令、操作手册——永久保留 |

Working 记忆永不删除——重要性可以降到零，但记忆本身始终可搜索。Buffer 记忆低于阈值会被淘汰。Core 记忆永久保留。

### 自组织主题树

向量聚类自动把相关记忆归到一起。树是分层的，名字由 LLM 生成：

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

记忆变化时主题树自动重建。会话开始时 agent 拿到主题索引作为知识目录。看到相关的主题，用 `POST /topic {"ids": ["kb3"]}` 展开该聚类下的全部记忆。

### 触发器系统

给记忆打上 `trigger:deploy` 标签，agent 执行部署前查一下 `/triggers/deploy`，过去的教训就自动浮现。犯过的错变成未来的防线。

```bash
# 存一条教训
curl -X POST http://localhost:3917/memories \
  -d '{"content": "LESSON: 迁移前必须备份数据库", "tags": ["trigger:deploy", "lesson"]}'

# 部署前，agent 检查：
curl http://localhost:3917/triggers/deploy
# → 返回所有带 trigger:deploy 标签的记忆，按访问次数排序
```

## 架构

记忆沿两个维度组织——**时间**和**空间**：

```
         Time (lifecycle)                    Space (topic tree)
┌─────────────────────────────┐    ┌──────────────────────────────┐
│                             │    │ Auth Architecture            │
│  Buffer → Working → Core    │    │ ├── OAuth2 migration [3]     │
│    ↓         ↓        ↑     │    │ └── Token handling [2]       │
│  evict    decay    permanent│    │ Deploy & Ops                 │
│           only      + gate  │    │ ├── CI/CD procedures [3]     │
│                             │    │ └── Rollback lessons [2]     │
└─────────────────────────────┘    │ User Preferences [6]         │
                                   └──────────────────────────────┘
```

**时间** — 受 [Atkinson–Shiffrin 记忆模型](https://en.wikipedia.org/wiki/Atkinson%E2%80%93Shiffrin_memory_model)启发的三层生命周期：

| 层级 | 角色 | 行为 |
|------|------|------|
| **Buffer** | 短期缓冲 | 所有新记忆先进这里。不被访问的会衰减淘汰 |
| **Working** | 工作记忆 | 通过反复访问或标记为教训/流程后提升进来。永不删除——重要性会降但记忆始终保留 |
| **Core** | 长期身份 | 经过持续使用 + LLM 质量门控后提升。永久保留 |

**空间** — 基于 embedding 向量的自组织主题树。相关记忆按语义相似度自动聚类，LLM 为每个聚类生成名字：

| 机制 | 作用 |
|------|------|
| **向量聚类** | 基于 embedding 余弦相似度，将语义相近的记忆归入同一主题 |
| **层次结构** | 相关主题嵌套在共同父节点下，形成多层树 |
| **LLM 命名** | 自动为每个聚类生成人类可读的主题名 |
| **自动重建** | 记忆变化时主题树自动更新，无需手动维护 |

主题树解决的问题：向量搜索要求你猜对搜索词才能找到东西。主题树让 agent 能按主题浏览——先看目录，再展开感兴趣的分支。

## 会话恢复

一次调用恢复全部上下文，用于启动或上下文压缩后：

```
GET /resume →

=== Core (24) ===
部署流程：测试 → 构建 → 停服 → 启动 (procedural)
LESSON: 不要 force-push 到 main
...

=== Recent (12h) ===
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

四个区块，各有用途：

| 区块 | 内容 | 预算 |
|------|------|------|
| **Core** | 永久规则和身份的完整文本——不截断 | ~2k tokens |
| **Recent** | 最近 12 小时内变更的记忆，保持短期连续性 | ~1k tokens |
| **Topics** | 主题索引——所有记忆的目录 | 叶子列表 |
| **Triggers** | 操作前安全标签——自动浮现相关教训 | 标签列表 |

Agent 读主题索引，发现相关的就用 `POST /topic` 展开。不需要把整个记忆库塞进上下文。

## 搜索与检索

混合检索：语义 embedding + BM25 关键词搜索（通过 [jieba](https://github.com/messense/jieba-rs) 支持中文分词）。结果按相关度、记忆重要性和时效性综合排序。

```bash
# 语义搜索，可控制返回预算
curl -X POST http://localhost:3917/recall \
  -d '{"query": "认证怎么做的", "budget_tokens": 2000}'

# 操作前安全检查
curl http://localhost:3917/triggers/deploy

# 主题展开
curl -X POST http://localhost:3917/topic \
  -d '{"ids": ["kb3"]}'
```

## 后台维护

全自动，按活动驱动——没有写操作时自动跳过：

### 整理（每 30 分钟）

每个周期按顺序执行：

1. **衰减** — 降低未访问记忆的重要性
2. **去重** — 检测并合并几乎相同的记忆（cosine > 0.78）
3. **分类** — LLM 对新 Buffer 记忆分类，决定是否提升
4. **门控** — LLM 评估提升候选（批量，单次调用）
5. **调和** — LLM 处理模糊相似对，结果缓存避免重复调用
6. **主题树重建** — 重新聚类，命名新的或变更的主题

### 审计（每 12 小时）

LLM 全局审查：

- 提升被低估的记忆，降级过时的
- 合并逃过实时去重的重复项
- 根据全局上下文调整重要性分数

审计同时看到所有 Core 和 Working 记忆，能发现逐条处理发现不了的跨主题冗余。

## 命名空间隔离

一个 engram 实例，多个项目。每个命名空间有独立的记忆空间：

```bash
# 项目专属记忆
curl -X POST http://localhost:3917/memories \
  -H "X-Namespace: my-project" \
  -d '{"content": "API 使用 OAuth2 Bearer Token"}'

# 跨项目知识放在默认命名空间
curl -X POST http://localhost:3917/memories \
  -d '{"content": "时间戳统一用 UTC"}'
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

没有 LLM 也能用——只有关键词搜索和规则驱动的生命周期。加一个 LLM 解锁智能功能（分类、门控、合并、主题命名、审计）：

```bash
export ENGRAM_LLM_URL=https://api.openai.com/v1
export ENGRAM_LLM_KEY=sk-...

# 或者任何 OpenAI 兼容 API（Ollama、LM Studio 等）
export ENGRAM_LLM_URL=http://localhost:11434/v1

# Anthropic 原生
export ENGRAM_LLM_PROVIDER=anthropic
export ENGRAM_LLM_KEY=sk-ant-...
```

两级模型配置——判断类任务用强模型，文本处理用便宜模型：

```bash
ENGRAM_GATE_MODEL=gpt-4o          # 质量判断
ENGRAM_LLM_MODEL=gpt-4o-mini     # 文本处理
```

### 给 AI Agent 配置

把这段贴到 agent 的系统提示词或会话里：

```
Set up engram (persistent memory) by following the guide at:
https://raw.githubusercontent.com/kael-bit/engram-rs/main/docs/SETUP.md
```

## 集成

支持 Claude Code、Cursor、Windsurf、OpenClaw 及所有 MCP 兼容工具。

17 个 MCP 工具——见 [MCP 文档](docs/MCP.md)。完整 HTTP API——见 [配置指南](docs/SETUP.md)。

```bash
# MCP (Claude Code)
npx engram-rs-mcp

# MCP (Cursor / Windsurf / 通用)
# 添加到 MCP 配置文件：
{"mcpServers": {"engram": {"command": "npx", "args": ["-y", "engram-rs-mcp"]}}}
```

### Web 控制台

内置 Web UI，访问 `http://localhost:3917`——浏览记忆、查看主题树、监控 LLM 用量、查看整理历史。

## 规格

| | |
|---|---|
| 二进制大小 | ~10 MB |
| 内存占用 | 生产环境 ~100 MB RSS |
| 存储 | SQLite，无外部数据库依赖 |
| 语言 | Rust |
| 平台 | Linux、macOS、Windows（x86_64 + aarch64） |
| 协议 | MIT |

## License

MIT
