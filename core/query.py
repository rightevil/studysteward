from core.config import Config
from core.kb_manager import KBManager

# NOTE: LlamaIndex 0.12+ requires explicit import of response synthesizer
from llama_index.core.response_synthesizers import CompactAndRefine


def search(kb: KBManager, query: str, tag: str | None = None,
           top_k: int = 10) -> list[dict]:
    """Semantic search across the knowledge base."""
    retriever = kb.index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    results = []
    for node in nodes:
        results.append({
            "id": node.node_id,
            "text": node.text,
            "distance": 1.0 - (node.score or 0),
            "doc_title": node.metadata.get("title", "Unknown"),
            "doc_type": node.metadata.get("type", ""),
        })
    return results


def ask(kb: KBManager, config: Config, question: str,
        doc_id: int | None = None):
    """RAG Q&A with streaming via LlamaIndex. Yields text tokens."""
    from ai.provider import create_provider_from_env

    provider = create_provider_from_env({
        "provider": config.ai_provider,
        "api_key": config.ai_api_key,
        "model": config.ai_model,
        "base_url": config.ai_base_url,
    })

    retriever = kb.index.as_retriever(similarity_top_k=10)
    nodes = retriever.retrieve(question)

    if not nodes:
        yield "No relevant materials found in the knowledge base."
        return

    # Build context with full source citations
    ctx_parts = []
    chunks = []
    for node in nodes:
        title = node.metadata.get("title", "Unknown")
        source = node.metadata.get("source", "")

        # Look up source URL and file path from SQLite
        source_url = source if source.startswith("http") else ""
        source_path = ""
        # Try to find matching document in SQLite
        docs = kb.sqlite.list_documents()
        for d in docs:
            if d["title"] == title or source in (d.get("source_url", ""), d.get("source_path", "")):
                source_url = d.get("source_url", "") or source_url
                source_path = d.get("source_path", "")
                break

        # Citation: Title — url
        if source_url:
            citation = f"{title} — {source_url}"
        elif source_path:
            citation = f"{title} — {source_path}"
        else:
            citation = f"{title}"

        ctx_parts.append(f"{citation}\n{node.text}")
        chunks.append(type("Chunk", (), {
            "doc_id": 0,
            "content": node.text,
            "chunk_index": 0,
            "embedding_id": node.node_id,
            "doc_title": citation,
        })())

    ctx_text = "\n\n".join(ctx_parts)
    yield from provider.ask_stream(question, chunks)
