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

    @property
    def embedder(self):
        if self._embedder is None:
            from ai.embedding import get_embedder
            from llama_index.core import Settings
            self._embedder = get_embedder()
            Settings.embed_model = self._embedder
        return self._embedder

    @property
    def index(self):
        if self._index is None:
            from storage.chroma import get_vector_store
            vector_store = get_vector_store(self.data_dir)
            self._index = VectorStoreIndex.from_vector_store(vector_store, embed_model=self.embedder)
        return self._index

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
        # Delete from vector store
        # NOTE: LlamaIndex ChromaDB doesn't support per-document deletion easily.
        # For now, delete metadata only. Full rebuild would be needed for vector cleanup.
        self.sqlite.delete_document(doc_id)
        if doc.get("file_hash"):
            path = self.files.path_for_hash(doc["file_hash"])
            if path:
                self.files.delete(path)

    def get_tag_tree(self) -> list[dict]:
        return self.sqlite.get_tag_tree()

    def set_tags(self, doc_id: int, tags: list[str]):
        self.sqlite.set_document_tags(doc_id, tags)
