# engram

AI agent 的记忆引擎。沿两个轴组织记忆：**时间轴**（三层衰减与晋升）和**空间轴**（自组织主题树）。重要的记忆自动提升，噪音自然淡出，相关知识自动聚类。

混合搜索，LLM 驱动的记忆整理，单个 Rust 二进制，一个 SQLite 文件。

多数 agent 记忆方案是平坦的存储——什么都往里塞，靠关键词搜出来。没有遗忘，没有组织，没有生命周期。engram 补上了让记忆真正有用的那部分：忘掉不重要的，浮现重要的。

<p align="center">
  <img src="docs/engram-quickstart.gif" alt="engram 演示 — 存储、上下文重置、检索" width="720">
</p>

[English](README.md)

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

## 它做了什么

### 三层生命周期

参考 [Atkinson–Shiffrin 记忆模型](https://en.wikipedia.org/wiki/Atkinson%E2%80%93Shiffrin_memory_model)，记忆按重要性分三层管理：

```
Buffer（短期）→ Working（活跃知识）→ Core（长期身份）
   ↓                  ↓                    ↑
 过期淘汰           重要性衰减         LLM 质量门控晋升
```

- **Buffer**：所有新记忆的入口。临时暂存，低于阈值时被淘汰
- **Working**：经过 consolidation 提升。不会被删除，重要性按类型以不同速率衰减
- **Core**：通过 LLM 质量门控从 Working 提升。永不删除

### LLM 质量门控

晋升不是靠规则拍脑袋——LLM 会结合上下文判断这条记忆是不是真的值得长期保留。

```
Buffer → [LLM 门控："属于决策、教训还是偏好？"] → Working
Working → [持续访问 + LLM 门控] → Core
```

### 自动衰减

衰减基于活动周期触发，系统空闲期间不衰减。不同类型衰减速率不同：

| 类型 | 衰减速率 | 适用场景 |
|------|---------|---------|
| `episodic` | 较快 | 事件、经历等时效性内容 |
| `semantic` | 较慢 | 知识、偏好、经验教训（默认） |
| `procedural` | 最慢 | 工作流、操作规范 |

### 语义去重与合并

两条记忆说的是同一件事但措辞不同？自动检测并合并：

```
"认证使用 PostgreSQL" + "认证服务跑在 Postgres 上"
→ 合并为一条，保留双方上下文
```

### 自组织主题树

向量聚类自动把相关记忆归到一起，LLM 自动命名。不需要手动打标签整理：

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

主题树解决的问题：向量搜索依赖你问对问题，而主题树让 agent 能**按主题浏览**——先看目录，再展开目标分支。

### 触发器

给记忆打 `trigger:deploy` 标签，agent 执行部署前调用 `/triggers/deploy` 就能自动回忆相关教训：

```bash
curl -X POST http://localhost:3917/memories \
  -d '{"content": "LESSON: 迁移前必须备份数据库", "tags": ["trigger:deploy", "lesson"]}'

# 部署前查询
curl http://localhost:3917/triggers/deploy
```

## 会话恢复

Agent 醒来时一个 `GET /resume` 拿回所有上下文，不用翻文件：

```
=== Core (24) ===
部署流程：测试 → 构建 → 停服 → 启动 (procedural)
LESSON: 不要 force-push 到 main
...

=== Recent ===
认证切换到 OAuth2
发布了 API 文档

=== Topics (Core: 24, Working: 57, Buffer: 7) ===
kb1: "部署流程" [5]
kb2: "认证架构" [3]
kb3: "记忆系统设计" [8]
...

Triggers: deploy, git-push, database-migration
```

| 区块 | 内容 | 作用 |
|------|------|------|
| **Core** | 永久性规则与身份信息完整文本 | 不可遗忘的东西 |
| **Recent** | 最近变更的记忆 | 短期连续性 |
| **Topics** | 主题索引（目录） | 按需展开，不全量加载 |
| **Triggers** | 操作前置标签 | 自动召回经验教训 |

Agent 读目录，找到相关主题，调 `POST /topic` 按需展开。

## 搜索与检索

语义 embedding + BM25 关键词搜索，支持中日韩分词（[jieba](https://github.com/messense/jieba-rs)）。IDF 加权评分——稀有词权重更高，常见词自动降权，不需要维护停用词表。

```bash
# 语义搜索
curl -X POST http://localhost:3917/recall \
  -d '{"query": "认证怎么做的", "budget_tokens": 2000}'

# 主题展开
curl -X POST http://localhost:3917/topic \
  -d '{"ids": ["kb3"]}'
```

## 后台维护

全自动，按活动驱动——没有写入就跳过：

**Consolidation（每 30 分钟）**

1. **衰减** — 降低未访问记忆的重要性
2. **去重** — 合并近似记忆（cosine > 0.78）
3. **分类** — LLM 对 Buffer 记忆进行分类
4. **门控** — LLM 批量评估晋升候选
5. **调和** — 处理模糊相似的记忆对（结果缓存）
6. **主题树重建** — 重新聚类和命名

**主题蒸馏** — 主题记忆过多时（≥10 条），自动合并重叠内容为更精炼的条目。

## 命名空间隔离

单实例支持多项目，用 `X-Namespace` 隔离：

```bash
curl -X POST http://localhost:3917/memories \
  -H "X-Namespace: my-project" \
  -d '{"content": "API 使用 OAuth2 Bearer Token"}'
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

不接 LLM 也能用——关键词搜索和规则生命周期照常工作。接入 LLM 解锁智能特性（分类、门控、合并、主题命名）：

```bash
export ENGRAM_LLM_URL=https://api.openai.com/v1
export ENGRAM_LLM_KEY=sk-...

# 兼容任何 OpenAI API（Ollama、LM Studio 等）
export ENGRAM_LLM_URL=http://localhost:11434/v1

# Anthropic 原生接入
export ENGRAM_LLM_PROVIDER=anthropic
export ENGRAM_LLM_KEY=sk-ant-...
```

双模型配置——判断任务用强模型，文本处理用轻量模型：

```bash
ENGRAM_GATE_MODEL=gpt-4o          # 质量判断
ENGRAM_LLM_MODEL=gpt-4o-mini     # 文本处理
```

### 接入 AI Agent

将以下内容添加到 agent 的系统提示词中：

```
Set up engram (persistent memory) by following the guide at:
https://raw.githubusercontent.com/kael-bit/engram-rs/main/docs/SETUP.md
```

## 集成

支持 Claude Code、Cursor、Windsurf、OpenClaw 及所有 MCP 兼容工具。

17 个 MCP 工具（[MCP 文档](docs/MCP.md)）| 完整 HTTP API（[配置指南](docs/SETUP.md)）

```bash
# MCP (Claude Code)
npx engram-rs-mcp

# MCP (Cursor / Windsurf / 通用)
{"mcpServers": {"engram": {"command": "npx", "args": ["-y", "engram-rs-mcp"]}}}
```

### Web 控制台

内置 Web UI：`http://localhost:3917`，可浏览记忆、查看主题树、监控 LLM 调用量。

## 技术规格

| | |
|---|---|
| 二进制 | ~10 MB |
| 内存 | 生产环境约 100 MB RSS |
| 存储 | SQLite，无外部依赖 |
| 语言 | Rust |
| 平台 | Linux、macOS、Windows（x86_64 + aarch64） |
| 协议 | MIT |

## License

MIT

<a href="https://glama.ai/mcp/servers/@kael-bit/engram-rs">
  <img width="380" height="200" src="https://glama.ai/mcp/servers/@kael-bit/engram-rs/badge" />
</a>
