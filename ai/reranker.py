import os
from pathlib import Path

from ai.embedding import CACHE_DIR


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class RerankerUnavailableError(RuntimeError):
    """Raised when the optional local reranker cannot be loaded."""


_rerankers: dict[str, object] = {}
_local_load_errors: dict[str, Exception] = {}


def get_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL,
    *,
    local_files_only: bool = True,
):
    """Load the optional CrossEncoder, without network access by default."""
    if model_name in _rerankers:
        return _rerankers[model_name]
    if local_files_only and model_name in _local_load_errors:
        raise RerankerUnavailableError(
            f"{model_name} is not available in the local cache"
        ) from _local_load_errors[model_name]

    try:
        reranker = _create_reranker(
            model_name,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        if local_files_only:
            _local_load_errors[model_name] = exc
        raise RerankerUnavailableError(
            f"Unable to load reranker {model_name}"
        ) from exc

    _local_load_errors.pop(model_name, None)
    _rerankers[model_name] = reranker
    return reranker


def _create_reranker(
    model_name: str,
    *,
    local_files_only: bool,
):
    import torch
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        model_name,
        local_files_only=local_files_only,
        max_length=512,
        device="cpu",
        activation_fn=torch.nn.Identity(),
        model_kwargs={"torch_dtype": torch.float32},
    )


def is_reranker_installed(model_name: str = DEFAULT_RERANKER_MODEL) -> bool:
    """Check the Hugging Face cache without loading model weights."""
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id=model_name,
                cache_dir=str(CACHE_DIR),
                local_files_only=True,
            )
        )
    except Exception:
        return False

    has_weights = any(
        (snapshot / filename).is_file()
        for filename in ("model.safetensors", "pytorch_model.bin")
    )
    has_tokenizer = any(
        (snapshot / filename).is_file()
        for filename in ("tokenizer.json", "sentencepiece.bpe.model", "vocab.txt")
    )
    return (snapshot / "config.json").is_file() and has_weights and has_tokenizer
