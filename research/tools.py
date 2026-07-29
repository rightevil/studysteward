import json

from core.query import search


class ResearchTools:
    """Read-only knowledge tools exposed to the research runtime."""

    _DESCRIPTIONS = {
        "list_documents": "List available documents. args: {}",
        "search_kb": "Semantic search. args: {query: str, top_k?: int}",
        "inspect_document": "Read metadata and indexed chunks. args: {doc_id: int}",
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
        return "\n".join(
            f"[D{doc['id']}] {doc['title']} | {doc['type']} | "
            f"{doc.get('summary') or '(no summary)'}"
            for doc in documents
        )

    def _search(self, args: dict) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return "Tool error: query is required."
        top_k = max(1, min(int(args.get("top_k", 5)), 10))
        results = search(self.kb, query, top_k=top_k)
        if not results:
            return f"No evidence found for query: {query}"
        rows = []
        for result in results:
            doc_id = result.get("doc_id")
            citation = f"[D{doc_id}]" if doc_id else "[legacy source]"
            rows.append(
                f"{citation} {result['doc_title']} "
                f"(score={1 - result['distance']:.3f})\n{result['text']}"
            )
        return "\n\n".join(rows)

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
            "chunks": [chunk["content"] for chunk in chunks[:8]],
        }
        if not chunks:
            payload["note"] = "Legacy document has no chunk mapping; rebuild its index."
        return json.dumps(payload, ensure_ascii=False)
