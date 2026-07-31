import time
from dataclasses import dataclass, field
from typing import Iterable

from ai.base import Chunk
import core.query as query_module


@dataclass(frozen=True)
class RAGContext:
    chunks: tuple[Chunk, ...]
    results: tuple[dict, ...] = ()
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RAGAnswer:
    question: str
    text: str
    context: RAGContext
    generation_ms: float


def retrieve_context(
    kb,
    question: str,
    *,
    top_k: int = 10,
    strategy: str = "hybrid_rerank",
    require_reranker: bool = False,
) -> RAGContext:
    """Retrieve and format the exact context used by the answer model."""
    diagnostics = {}
    results = query_module.search(
        kb,
        question,
        top_k=top_k,
        strategy=strategy,
        diagnostics=diagnostics,
        require_reranker=require_reranker,
    )
    return context_from_results(kb, results, diagnostics=diagnostics)


def context_from_evidence(
    kb,
    evidence: Iterable[tuple[int, int]],
) -> RAGContext:
    """Build an oracle context from stable document/chunk references."""
    documents = {
        int(document["id"]): document
        for document in kb.sqlite.list_documents()
    }
    chunks_by_document: dict[int, dict[int, dict]] = {}
    results = []
    seen = set()
    for raw_doc_id, raw_chunk_index in evidence:
        reference = (int(raw_doc_id), int(raw_chunk_index))
        if reference in seen:
            continue
        seen.add(reference)
        doc_id, chunk_index = reference
        document = documents.get(doc_id)
        if document is None:
            raise ValueError(f"Oracle evidence references missing document D{doc_id}")
        if doc_id not in chunks_by_document:
            chunks_by_document[doc_id] = {
                int(chunk["chunk_index"]): chunk
                for chunk in kb.sqlite.get_chunks(doc_id)
            }
        chunk = chunks_by_document[doc_id].get(chunk_index)
        if chunk is None:
            raise ValueError(
                f"Oracle evidence references missing chunk D{doc_id}:C{chunk_index}"
            )
        results.append(
            {
                "id": chunk["embedding_id"],
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "text": chunk["content"],
                "doc_title": document["title"],
                "doc_type": document.get("type", ""),
                "source": (
                    document.get("source_url")
                    or document.get("source_path")
                    or ""
                ),
                "retrieval_sources": ["oracle"],
                "retrieval_ranks": {"oracle": len(results) + 1},
            }
        )
    return context_from_results(
        kb,
        results,
        diagnostics={"context_mode": "oracle"},
    )


def context_from_results(
    kb,
    results: list[dict],
    *,
    diagnostics: dict | None = None,
) -> RAGContext:
    """Convert retrieval results to cited model context chunks."""
    documents = {
        int(document["id"]): document
        for document in kb.sqlite.list_documents()
    }
    chunks = []
    for result in results:
        doc_id = int(result["doc_id"])
        document = documents.get(doc_id, {})
        title = result.get("doc_title", "Unknown")
        source = result.get("source", "")
        source_url = document.get("source_url", "") or (
            source if str(source).startswith("http") else ""
        )
        source_path = document.get("source_path", "")

        if source_url:
            citation = f"[D{doc_id}] {title} — {source_url}"
        elif source_path:
            citation = f"[D{doc_id}] {title} — {source_path}"
        else:
            citation = f"[D{doc_id}] {title}"

        chunks.append(
            Chunk(
                doc_id=doc_id,
                content=result["text"],
                chunk_index=int(result.get("chunk_index", 0)),
                embedding_id=str(result["id"]),
                doc_title=citation,
            )
        )
    return RAGContext(
        chunks=tuple(chunks),
        results=tuple(dict(result) for result in results),
        diagnostics=dict(diagnostics or {}),
    )


def answer_from_context(provider, question: str, context: RAGContext) -> RAGAnswer:
    """Generate one non-streaming answer from an explicit context."""
    started = time.perf_counter()
    answer = provider.ask(question, list(context.chunks))
    generation_ms = round((time.perf_counter() - started) * 1000, 2)
    return RAGAnswer(
        question=question,
        text=str(answer),
        context=context,
        generation_ms=generation_ms,
    )
