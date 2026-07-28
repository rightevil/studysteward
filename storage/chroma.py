from pathlib import Path
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from ai.embedding import get_embedder


def get_vector_store(data_dir: Path) -> ChromaVectorStore:
    """Get or create the ChromaDB vector store for the knowledge base."""
    chroma_path = str(data_dir / "chroma")
    db = chromadb.PersistentClient(path=chroma_path)
    collection = db.get_or_create_collection("chunks")
    return ChromaVectorStore(
        chroma_collection=collection,
        embed_model=get_embedder(),
    )
