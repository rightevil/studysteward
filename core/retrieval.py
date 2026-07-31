import math
import re
from collections import Counter
from dataclasses import dataclass


_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z0-9_.:+/-]*|\d+|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """Tokenize technical identifiers exactly and Chinese text as bigrams."""
    tokens: list[str] = []
    for match in _TOKEN.finditer(text.casefold()):
        value = match.group()
        if "\u4e00" <= value[0] <= "\u9fff":
            if len(value) == 1:
                tokens.append(value)
            else:
                tokens.extend(value[index:index + 2] for index in range(len(value) - 1))
            continue
        tokens.append(value)
    return tokens


@dataclass(frozen=True)
class LexicalDocument:
    embedding_id: str
    doc_id: int
    chunk_index: int
    text: str
    title: str
    doc_type: str
    source: str


class BM25Index:
    """Small in-memory BM25 index over persisted knowledge-base chunks."""

    def __init__(
        self,
        documents: list[LexicalDocument],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.documents = tuple(documents)
        self.k1 = k1
        self.b = b
        self._term_frequencies: list[Counter[str]] = []
        self._lengths: list[int] = []
        document_frequencies: Counter[str] = Counter()

        for document in self.documents:
            # Repeating the title gives section-poor chunks a small document signal.
            terms = tokenize(f"{document.title} {document.title} {document.text}")
            frequencies = Counter(terms)
            self._term_frequencies.append(frequencies)
            self._lengths.append(len(terms))
            document_frequencies.update(frequencies.keys())

        count = len(self.documents)
        self._average_length = (
            sum(self._lengths) / count if count else 0.0
        )
        self._idf = {
            term: math.log(
                1.0 + (count - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in document_frequencies.items()
        }

    @classmethod
    def from_kb(cls, kb) -> "BM25Index":
        documents = []
        for document in kb.sqlite.list_documents():
            source = document.get("source_url") or document.get("source_path") or ""
            for chunk in kb.sqlite.get_chunks(document["id"]):
                documents.append(
                    LexicalDocument(
                        embedding_id=chunk["embedding_id"],
                        doc_id=document["id"],
                        chunk_index=chunk["chunk_index"],
                        text=chunk["content"],
                        title=document["title"],
                        doc_type=document["type"],
                        source=source,
                    )
                )
        return cls(documents)

    def search(self, query: str, limit: int) -> list[dict]:
        query_terms = Counter(tokenize(query))
        if not query_terms or not self.documents:
            return []

        ranked = []
        average_length = self._average_length or 1.0
        for index, document in enumerate(self.documents):
            frequencies = self._term_frequencies[index]
            length = self._lengths[index]
            score = 0.0
            for term, query_frequency in query_terms.items():
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / average_length
                )
                score += (
                    self._idf.get(term, 0.0)
                    * frequency
                    * (self.k1 + 1.0)
                    / denominator
                    * query_frequency
                )
            if score <= 0:
                continue
            ranked.append(
                {
                    "id": document.embedding_id,
                    "doc_id": document.doc_id,
                    "chunk_index": document.chunk_index,
                    "text": document.text,
                    "score": score,
                    "distance": 1.0,
                    "doc_title": document.title,
                    "doc_type": document.doc_type,
                    "source": document.source,
                }
            )
        ranked.sort(key=lambda item: (-item["score"], item["id"]))
        return ranked[:limit]


def reciprocal_rank_fusion(
    rankings: list[tuple[str, list[dict]]],
    *,
    limit: int,
    rank_constant: int = 60,
) -> list[dict]:
    """Fuse result lists without requiring their raw scores to be comparable."""
    fused_scores: Counter[str] = Counter()
    results_by_id: dict[str, dict] = {}
    sources_by_id: dict[str, set[str]] = {}
    ranks_by_id: dict[str, dict[str, int]] = {}

    for source, results in rankings:
        seen: set[str] = set()
        for rank, result in enumerate(results, start=1):
            result_id = str(result["id"])
            if result_id in seen:
                continue
            seen.add(result_id)
            fused_scores[result_id] += 1.0 / (rank_constant + rank)
            results_by_id.setdefault(result_id, result)
            sources_by_id.setdefault(result_id, set()).add(source)
            ranks_by_id.setdefault(result_id, {})[source] = rank

    ranked_ids = sorted(
        fused_scores,
        key=lambda result_id: (-fused_scores[result_id], result_id),
    )[:limit]
    if not ranked_ids:
        return []
    maximum = fused_scores[ranked_ids[0]]
    fused = []
    for result_id in ranked_ids:
        result = dict(results_by_id[result_id])
        normalized_score = fused_scores[result_id] / maximum
        result.update(
            {
                "score": normalized_score,
                "distance": 1.0 - normalized_score,
                "retrieval_sources": sorted(sources_by_id[result_id]),
                "retrieval_ranks": ranks_by_id[result_id],
            }
        )
        fused.append(result)
    return fused


def rerank_candidates(
    query: str,
    candidates: list[dict],
    model,
    *,
    limit: int,
    batch_size: int = 8,
) -> list[dict]:
    """Rerank fused candidates with a query-document CrossEncoder."""
    if limit < 1 or not candidates:
        return []

    pairs = [(query, candidate.get("text", "")) for candidate in candidates]
    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    scored = []
    for original_rank, (candidate, raw_score) in enumerate(
        zip(candidates, scores),
        start=1,
    ):
        raw_score = float(raw_score)
        if raw_score >= 0:
            normalized_score = 1.0 / (1.0 + math.exp(-raw_score))
        else:
            exponential = math.exp(raw_score)
            normalized_score = exponential / (1.0 + exponential)

        result = dict(candidate)
        result.update(
            {
                "pre_rerank_rank": original_rank,
                "pre_rerank_score": float(candidate.get("score", 0.0)),
                "reranker_score": raw_score,
                "score": normalized_score,
                "distance": 1.0 - normalized_score,
            }
        )
        scored.append(result)

    scored.sort(key=lambda item: (-item["reranker_score"], item["pre_rerank_rank"]))
    return scored[:limit]
