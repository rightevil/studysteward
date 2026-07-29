# StudySteward

[中文](README.md) | [English](README.en.md)

StudySteward is a local-first RAG and agentic research CLI. Simple questions use
the low-latency RAG path, while complex goals can invoke a bounded Research Agent
that plans, selects read-only tools, observes evidence, and produces cited reports.

## Features

- Interactive file, directory, PDF, Markdown, text, and web ingestion.
- Local BGE embeddings, ChromaDB vectors, and SQLite metadata.
- Stable `[D{id}]` citations backed by explicit document-to-chunk mappings.
- A maximum eight-step research loop with four read-only tools.
- Persistent tool arguments, observations, timings, and final reports.
- Auditable traces through `/trace` and source lookup through `/info D{id}`.

## Quick Start

```bash
uv sync
uv run study
```

```text
/ingest D:\notes
/reindex
/research Compare Linux and Windows privilege escalation prerequisites and evidence gaps
/trace
/info D16
```

`/reindex` repairs vector mappings created by older releases without recomputing
embeddings. Run the test suite with:

```bash
uv run python -m unittest discover -s tests -v
```
