# StudySteward

[中文版](README.md) | [English](README.en.md)

StudySteward 是一个本地优先的 Agentic Research CLI。它把文档导入、语义检索和
RAG 问答组合成快速路径，并为复杂目标提供有边界、可追踪的 Research Agent。

## 核心能力

- 导入 PDF、Markdown、文本和网页，支持目录扫描与交互式批量选择。
- 使用 BGE + ChromaDB 做本地语义检索，SQLite 保存文档、chunk 和任务元数据。
- 普通提问走低延迟 RAG；`/research` 才启动最多 8 步的规划与工具选择循环。
- Agent 只开放只读工具：`list_documents`、`search_kb`、`inspect_document`、`finish`。
- 每一步的工具、参数、观察和耗时均持久化，可用 `/trace` 审计。
- 报告使用稳定的 `[D{id}]` 引用，可通过 `/info D{id}` 回溯来源。

## 快速开始

```bash
uv sync
uv run study
```

首次使用先运行 `/setup` 下载嵌入模型，然后导入资料：

```text
/ingest D:\notes
/reindex
/research 比较知识库中的 Linux 与 Windows 提权方法，说明适用前提、共同点、差异和资料缺口，并给出来源引用
/trace
/info D16
```

`/reindex` 用于修复旧版本生成的向量映射，不会重新计算 embedding。

## 两条执行路径

```text
普通问题 -> 检索 Top-K -> 流式回答

复杂研究目标 -> 规划下一步 -> 选择只读工具 -> 观察结果
             -> 修正检索策略 -> 生成带引用报告
             -> SQLite execution trace
```

Research Agent 的循环受最大步数约束；达到上限时会根据已有证据收敛，并明确资料缺口，
而不是无限执行。向量元数据使用稳定的 `kb_doc_id` 与 SQLite chunk 映射，删除文档时会
同步删除对应向量；检索层还会隔离旧版本遗留的孤儿向量。

## 配置

复制 `.env.example` 并配置 AI Provider。项目支持 Anthropic Messages API 与
OpenAI-compatible Chat Completions API。知识库数据默认位于项目配置指定的
`.studysteward` 目录。

## 测试

```bash
uv run python -m unittest discover -s tests -v
```
