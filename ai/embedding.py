import os
from pathlib import Path
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Explicit cache path: same as huggingface hub default
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"

_embedder: HuggingFaceEmbedding | None = None


def get_embedder(*, local_files_only: bool = True) -> HuggingFaceEmbedding:
    """Load the embedding model, using the local cache during normal operation."""
    global _embedder
    if _embedder is None:
        import io, sys
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            _embedder = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-zh",
                embed_batch_size=16,
                cache_folder=str(CACHE_DIR),
                local_files_only=local_files_only,
            )
        finally:
            sys.stderr = old_stderr
    return _embedder


def is_model_installed() -> bool:
    """Check if model is cached without loading it."""
    try:
        HuggingFaceEmbedding(
            model_name="BAAI/bge-small-zh",
            embed_batch_size=16,
            cache_folder=str(CACHE_DIR),
            local_files_only=True,
        )
        return True
    except Exception:
        return False
