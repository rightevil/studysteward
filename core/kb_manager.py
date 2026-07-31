import json
from pathlib import Path
from llama_index.core import VectorStoreIndex, Settings
from core.config import Config
from storage.sqlite import SQLiteStore
from storage.chroma import get_vector_store
from storage.files import FileStore
from ai.embedding import get_embedder


class KBManager:
    def __init__(self, config: Config):
        self.config = config
        self.data_dir = config.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite = SQLiteStore(self.data_dir / "kb.db")
        self.sqlite.init_schema()
        self.files = FileStore(self.data_dir)
        self._embedder = None
        self._index = None
        self._vector_store = None
        self._lexical_index = None

    @property
    def embedder(self):
        if self._embedder is None:
            from ai.embedding import get_embedder
            from llama_index.core import Settings
            self._embedder = get_embedder()
            Settings.embed_model = self._embedder
        return self._embedder

    @property
    def vector_store(self):
        if self._vector_store is None:
            from storage.chroma import get_vector_store
            self._vector_store = get_vector_store(self.data_dir)
        return self._vector_store

    @property
    def index(self):
        if self._index is None:
            self._index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                embed_model=self.embedder,
            )
        return self._index

    @property
    def lexical_index(self):
        if self._lexical_index is None:
            from core.retrieval import BM25Index

            self._lexical_index = BM25Index.from_kb(self)
        return self._lexical_index

    def invalidate_retrieval_cache(self):
        self._lexical_index = None

    def get_document(self, doc_id: int) -> dict | None:
        doc = self.sqlite.get_document(doc_id)
        if doc:
            doc["tags"] = self.sqlite.get_tags_for_doc(doc_id)
        return doc

    def list_documents(self, tag: str | None = None) -> list[dict]:
        docs = self.sqlite.list_documents(tag=tag)
        for doc in docs:
            doc["tags"] = self.sqlite.get_tags_for_doc(doc["id"])
        return docs

    def delete_document(self, doc_id: int):
        doc = self.sqlite.get_document(doc_id)
        if not doc:
            return
        embedding_ids = [
            chunk["embedding_id"] for chunk in self.sqlite.get_chunks(doc_id)
        ]
        if embedding_ids:
            self.vector_store.delete_nodes(embedding_ids)
        self.sqlite.delete_document(doc_id)
        self.invalidate_retrieval_cache()
        if doc.get("file_hash"):
            path = self.files.path_for_hash(doc["file_hash"])
            if path:
                self.files.delete(path)

    def get_tag_tree(self) -> list[dict]:
        return self.sqlite.get_tag_tree()

    def set_tags(self, doc_id: int, tags: list[str]):
        self.sqlite.set_document_tags(doc_id, tags)

    def _chroma_collection(self):
        import chromadb

        return chromadb.PersistentClient(
            path=str(self.data_dir / "chroma")
        ).get_or_create_collection("chunks")

    def repair_legacy_index(self) -> dict:
        """Attach SQLite IDs to legacy vectors without recomputing embeddings."""
        collection = self._chroma_collection()
        payload = collection.get(include=["documents", "metadatas"])
        documents = self.sqlite.list_documents()
        by_source: dict[str, list[dict]] = {}
        by_title: dict[str, list[dict]] = {}
        for document in documents:
            for source in (document.get("source_path"), document.get("source_url")):
                if source:
                    by_source.setdefault(source, []).append(document)
            by_title.setdefault(document["title"], []).append(document)

        chunks_by_doc: dict[int, list[tuple[str, int, str]]] = {}
        unmatched = 0
        repaired = 0
        for node_id, content, raw_metadata in zip(
            payload["ids"],
            payload["documents"],
            payload["metadatas"],
        ):
            metadata = dict(raw_metadata or {})
            node_payload = {}
            if metadata.get("_node_content"):
                try:
                    node_payload = json.loads(metadata["_node_content"])
                except (TypeError, json.JSONDecodeError):
                    node_payload = {}
            node_metadata = node_payload.get("metadata", {})
            source = metadata.get("source") or node_metadata.get("source")
            title = metadata.get("title") or node_metadata.get("title")

            raw_kb_doc_id = metadata.get("kb_doc_id") or node_metadata.get("kb_doc_id")
            document = None
            if raw_kb_doc_id not in (None, ""):
                document = next(
                    (item for item in documents if item["id"] == int(raw_kb_doc_id)),
                    None,
                )
            if document is None and source and len(by_source.get(source, [])) == 1:
                document = by_source[source][0]
            if document is None and title and len(by_title.get(title, [])) == 1:
                document = by_title[title][0]
            if document is None:
                unmatched += 1
                continue

            doc_id = document["id"]
            chunk_index = len(chunks_by_doc.setdefault(doc_id, []))
            chunks_by_doc[doc_id].append((content or "", chunk_index, node_id))
            if metadata.get("kb_doc_id") != doc_id:
                metadata["kb_doc_id"] = doc_id
                metadata["chunk_index"] = chunk_index
                if node_payload:
                    node_payload.setdefault("metadata", {}).update(
                        {"kb_doc_id": doc_id, "chunk_index": chunk_index}
                    )
                    metadata["_node_content"] = json.dumps(
                        node_payload, ensure_ascii=False
                    )
                collection.update(ids=[node_id], metadatas=[metadata])
                repaired += 1

        for doc_id, chunks in chunks_by_doc.items():
            self.sqlite.delete_chunks(doc_id)
            self.sqlite.add_chunks(doc_id, chunks)
        self.invalidate_retrieval_cache()

        return {
            "scanned": len(payload["ids"]),
            "repaired": repaired,
            "mapped": sum(len(chunks) for chunks in chunks_by_doc.values()),
            "unmatched": unmatched,
            "documents": len(chunks_by_doc),
        }
