from core.config import Config
from core.kb_manager import KBManager
from ai.base import Chunk

# NOTE: LlamaIndex 0.12+ requires explicit import of response synthesizer
from llama_index.core.response_synthesizers import CompactAndRefine


def search(kb: KBManager, query: str, tag: str | None = None,
           top_k: int = 10) -> list[dict]:
    """Semantic search across the knowledge base."""
    retriever = kb.index.as_retriever(similarity_top_k=top_k * 3)
    nodes = retriever.retrieve(query)
    valid_doc_ids = {document["id"] for document in kb.sqlite.list_documents()}

    results = []
    for node in nodes:
        raw_doc_id = node.metadata.get("kb_doc_id")
        doc_id = int(raw_doc_id) if raw_doc_id not in (None, "") else None
        if doc_id not in valid_doc_ids:
            continue
        results.append({
            "id": node.node_id,
            "doc_id": doc_id,
            "text": node.text,
            "distance": 1.0 - (node.score or 0),
            "doc_title": node.metadata.get("title", "Unknown"),
            "doc_type": node.metadata.get("doc_type", ""),
            "source": node.metadata.get("source", ""),
        })
    return results[:top_k]


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

    retriever = kb.index.as_retriever(similarity_top_k=30)
    nodes = retriever.retrieve(question)
    documents = kb.sqlite.list_documents()
    docs_by_id = {document["id"]: document for document in documents}
    nodes = [
        node
        for node in nodes
        if node.metadata.get("kb_doc_id") not in (None, "")
        and int(node.metadata["kb_doc_id"]) in docs_by_id
    ][:10]
    if not nodes:
        yield "No relevant materials found in the knowledge base."
        return

    # Build context with full source citations
    chunks = []
    for node in nodes:
        title = node.metadata.get("title", "Unknown")
        source = node.metadata.get("source", "")
        raw_doc_id = node.metadata.get("kb_doc_id")
        doc_id = int(raw_doc_id)
        document = docs_by_id.get(doc_id)

        source_url = document.get("source_url", "") or (
            source if source.startswith("http") else ""
        )
        source_path = document.get("source_path", "")

        # Citation: Title — url
        if source_url:
            citation = f"[D{doc_id}] {title} — {source_url}"
        elif source_path:
            citation = f"[D{doc_id}] {title} — {source_path}"
        else:
            citation = f"[D{doc_id}] {title}"

        chunks.append(
            Chunk(
                doc_id=doc_id,
                content=node.text,
                chunk_index=int(node.metadata.get("chunk_index", 0)),
                embedding_id=node.node_id,
                doc_title=citation,
            )
        )

    yield from provider.ask_stream(question, chunks)
