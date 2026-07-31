import time

from core.config import Config
from core.kb_manager import KBManager
from core.retrieval import reciprocal_rank_fusion, rerank_candidates
from ai.base import Chunk


def search(
    kb: KBManager,
    query: str,
    tag: str | None = None,
    top_k: int = 10,
    strategy: str = "hybrid_rerank",
    diagnostics: dict | None = None,
    require_reranker: bool = False,
) -> list[dict]:
    """Search with dense, hybrid, or optional CrossEncoder reranking."""
    if strategy not in {"dense", "hybrid", "hybrid_rerank"}:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")

    timings: dict[str, float | bool | str] = {}
    candidate_limit = max(top_k * 3, 15 if strategy == "hybrid_rerank" else top_k)
    started = time.perf_counter()
    dense_results = _dense_search(kb, query, candidate_limit)
    timings["dense_ms"] = _elapsed_ms(started)
    if strategy == "dense":
        _finish_diagnostics(diagnostics, timings, reranker_used=False)
        return dense_results[:top_k]

    started = time.perf_counter()
    lexical_results = kb.lexical_index.search(query, candidate_limit)
    timings["lexical_ms"] = _elapsed_ms(started)
    fusion_limit = max(top_k, 15) if strategy == "hybrid_rerank" else top_k
    started = time.perf_counter()
    fused = reciprocal_rank_fusion(
        [("dense", dense_results), ("lexical", lexical_results)],
        limit=fusion_limit,
    )
    timings["fusion_ms"] = _elapsed_ms(started)
    if strategy == "hybrid":
        _finish_diagnostics(diagnostics, timings, reranker_used=False)
        return fused

    started = time.perf_counter()
    try:
        from ai.reranker import get_reranker

        reranker = get_reranker(
            kb.config.reranker_model,
            local_files_only=True,
        )
        batch_size = 8
        results = rerank_candidates(
            query,
            fused,
            reranker,
            limit=top_k,
            batch_size=batch_size,
        )
    except Exception as exc:
        timings["reranker_error"] = type(exc).__name__
        if require_reranker:
            raise
        results = fused[:top_k]
        _finish_diagnostics(diagnostics, timings, reranker_used=False)
        return results

    timings["rerank_ms"] = _elapsed_ms(started)
    timings["reranker_device"] = str(reranker.device)
    timings["reranker_batch_size"] = batch_size
    _finish_diagnostics(diagnostics, timings, reranker_used=True)
    return results


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _finish_diagnostics(
    diagnostics: dict | None,
    timings: dict,
    *,
    reranker_used: bool,
) -> None:
    if diagnostics is None:
        return
    diagnostics.update(timings)
    diagnostics["reranker_used"] = reranker_used


def _dense_search(kb: KBManager, query: str, limit: int) -> list[dict]:
    retriever = kb.index.as_retriever(similarity_top_k=limit)
    nodes = retriever.retrieve(query)
    valid_doc_ids = {document["id"] for document in kb.sqlite.list_documents()}

    results = []
    for node in nodes:
        raw_doc_id = node.metadata.get("kb_doc_id")
        doc_id = int(raw_doc_id) if raw_doc_id not in (None, "") else None
        if doc_id not in valid_doc_ids:
            continue
        score = float(node.score or 0)
        results.append({
            "id": node.node_id,
            "doc_id": doc_id,
            "chunk_index": int(node.metadata.get("chunk_index", 0)),
            "text": node.text,
            "score": score,
            "distance": 1.0 - score,
            "doc_title": node.metadata.get("title", "Unknown"),
            "doc_type": node.metadata.get("doc_type", ""),
            "source": node.metadata.get("source", ""),
            "retrieval_sources": ["dense"],
            "retrieval_ranks": {"dense": len(results) + 1},
        })
    return results[:limit]


def ask(kb: KBManager, config: Config, question: str,
        doc_id: int | None = None):
    """RAG Q&A with streaming via LlamaIndex. Yields text tokens."""
    from ai.provider import create_provider_from_env
    from core.rag import retrieve_context

    provider = create_provider_from_env({
        "provider": config.ai_provider,
        "api_key": config.ai_api_key,
        "model": config.ai_model,
        "base_url": config.ai_base_url,
    })

    context = retrieve_context(
        kb,
        question,
        top_k=10,
        strategy="hybrid_rerank",
    )
    if not context.chunks:
        yield "No relevant materials found in the knowledge base."
        return
    yield from provider.ask_stream(question, list(context.chunks))
