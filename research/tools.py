import json

from core.query import search


class ResearchTools:
    """Read-only knowledge tools exposed to the research runtime."""

    _DESCRIPTIONS = {
        "list_documents": "List available documents once. args: {}",
        "search_kb": (
            "Semantic search returning deduplicated candidate documents. "
            "args: {query: str, top_k?: int}"
        ),
        "inspect_document": (
            "Read compact excerpts from one identified document. args: {doc_id: int}"
        ),
        "finish": "Return the final cited report. args: {report: str}",
    }

    def __init__(self, kb):
        self.kb = kb

    def descriptions(self) -> str:
        return "\n".join(
            f"- {name}: {description}"
            for name, description in self._DESCRIPTIONS.items()
        )

    def execute(self, name: str, args: dict) -> str:
        try:
            if name == "list_documents":
                return self._list_documents()
            if name == "search_kb":
                return self._search(args)
            if name == "inspect_document":
                return self._inspect(args)
            return f"Tool error: unknown tool '{name}'."
        except Exception as exc:
            return f"Tool error: {exc}"

    def _list_documents(self) -> str:
        documents = self.kb.list_documents()
        if not documents:
            return "No documents are available."
        payload = {
            "document_count": len(documents),
            "documents": [
                {
                    "citation": f"[D{doc['id']}]",
                    "title": doc["title"],
                    "type": doc["type"],
                    "summary": self._excerpt(doc.get("summary") or "", 160),
                }
                for doc in documents
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    def _search(self, args: dict) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return "Tool error: query is required."
        top_k = max(1, min(int(args.get("top_k", 5)), 8))
        results = search(self.kb, query, top_k=10)
        if not results:
            return json.dumps(
                {"query": query, "result_count": 0, "results": []},
                ensure_ascii=False,
            )
        rows = []
        seen_doc_ids = set()
        for result in results:
            doc_id = result.get("doc_id")
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            rows.append(
                {
                    "citation": f"[D{doc_id}]",
                    "title": result["doc_title"],
                    "score": round(1 - result["distance"], 3),
                    "excerpt": self._excerpt(result["text"], 900),
                }
            )
            if len(rows) >= top_k:
                break
        return json.dumps(
            {"query": query, "result_count": len(rows), "results": rows},
            ensure_ascii=False,
        )

    def _inspect(self, args: dict) -> str:
        doc_id = int(args.get("doc_id", 0))
        document = self.kb.get_document(doc_id)
        if not document:
            return f"Document [D{doc_id}] was not found."
        chunks = self.kb.sqlite.get_chunks(doc_id)
        payload = {
            "citation": f"[D{doc_id}]",
            "title": document["title"],
            "type": document["type"],
            "summary": document.get("summary", ""),
            "source": document.get("source_url") or document.get("source_path", ""),
            "excerpts": [
                {
                    "chunk_index": chunk["chunk_index"],
                    "text": self._excerpt(chunk["content"], 900),
                }
                for chunk in chunks[:5]
            ],
        }
        if not chunks:
            payload["note"] = "Legacy document has no chunk mapping; rebuild its index."
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _excerpt(text: str, limit: int) -> str:
        compact = " ".join(str(text).split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."
