from pathlib import Path

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

from ai.embedding import get_embedder
from parser.formats import is_supported
from parser.mineru import parse_file, parse_url
from storage.files import compute_hash


MAX_FILE_SIZE = 50 * 1024 * 1024


def _make_pipeline(chunk_size: int, chunk_overlap: int) -> IngestionPipeline:
    return IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            get_embedder(),
        ],
    )


def _index_document(kb, config, text: str, metadata: dict, document_fields: dict) -> int:
    """Persist one document and keep SQLite and Chroma mappings consistent."""
    doc_id = kb.sqlite.add_document(**document_fields)
    nodes = []
    try:
        doc = Document(text=text, metadata={**metadata, "kb_doc_id": doc_id})
        nodes = _make_pipeline(config.chunk_size, config.chunk_overlap).run(
            documents=[doc]
        )
        for index, node in enumerate(nodes):
            node.metadata.update(
                {
                    **metadata,
                    "kb_doc_id": doc_id,
                    "chunk_index": index,
                }
            )
        kb.index.insert_nodes(nodes)
        kb.sqlite.add_chunks(
            doc_id,
            [
                (node.get_content(), index, node.node_id)
                for index, node in enumerate(nodes)
            ],
        )
        return doc_id
    except Exception:
        if nodes:
            kb.vector_store.delete_nodes([node.node_id for node in nodes])
        kb.sqlite.delete_document(doc_id)
        raise


def run_ingest(kb_manager, config, source: str, auto_summarize: bool = True, on_progress=None) -> int:
    """Run the document ingest pipeline and return the resulting document ID."""
    is_url = source.startswith(("http://", "https://"))

    if is_url:
        import hashlib

        file_hash = hashlib.sha256(source.encode()).hexdigest()
    else:
        source_path = Path(source)
        if not is_supported(source_path):
            raise ValueError(f"Unsupported format: {source_path.suffix}")
        file_size = source_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"File too large: {file_size / 1024 / 1024:.1f}MB")
        file_hash = compute_hash(source_path)

    kb = kb_manager
    if kb.sqlite.document_exists_by_hash(file_hash):
        docs = kb.sqlite.conn.execute(
            "SELECT id FROM documents WHERE file_hash=?", (file_hash,)
        ).fetchone()
        if docs:
            if on_progress:
                on_progress("completed")
            return docs[0]

    # Stage 1: Parse
    if on_progress:
        on_progress("parsing")
    if is_url:
        text, doc_type = parse_url(source)
        title = source
    else:
        text, doc_type = parse_file(source_path)
        title = source_path.name

    # Stage 2: AI summarize
    summary = ""
    tags: list[str] = []
    if auto_summarize and config.ai_api_key:
        if on_progress:
            on_progress("summarizing")
        try:
            from ai.provider import create_provider_from_env

            provider = create_provider_from_env(
                {
                    "provider": config.ai_provider,
                    "api_key": config.ai_api_key,
                    "model": config.ai_model,
                    "base_url": config.ai_base_url,
                }
            )
            result = provider.summarize(text)
            title = result.title
            summary = result.summary
            tags = provider.suggest_tags(text)
        except Exception:
            pass

    # Stage 3: Chunk + embed
    if on_progress:
        on_progress("embedding")
    # Stage 4: Index + metadata
    if on_progress:
        on_progress("indexing")
    doc_id = _index_document(
        kb,
        config,
        text,
        {"title": title, "source": source, "doc_type": doc_type},
        {
            "title": title,
            "source_path": source,
            "source_url": source if is_url else None,
            "doc_type": doc_type,
            "summary": summary,
            "file_hash": file_hash,
        },
    )
    if not is_url:
        kb.files.store(source_path)
    if tags:
        kb.sqlite.set_document_tags(doc_id, tags)

    if on_progress:
        on_progress("completed")
    return doc_id


def ingest_text(kb, config, source: str, text: str, doc_type: str, auto_summarize: bool = True, on_progress=None) -> int:
    """Run the ingest pipeline on pre-parsed text and return the document ID."""
    import hashlib
    from datetime import date

    file_hash = hashlib.sha256(source.encode()).hexdigest()

    if kb.sqlite.document_exists_by_hash(file_hash):
        docs = kb.sqlite.conn.execute(
            "SELECT id FROM documents WHERE file_hash=?", (file_hash,)
        ).fetchone()
        if docs:
            if on_progress:
                on_progress("completed")
            return docs[0]

    title = source
    summary = ""
    tags = []

    if auto_summarize and config.ai_api_key:
        if on_progress:
            on_progress("summarizing")
        try:
            from ai.provider import create_provider_from_env

            provider = create_provider_from_env(
                {
                    "provider": config.ai_provider,
                    "api_key": config.ai_api_key,
                    "model": config.ai_model,
                    "base_url": config.ai_base_url,
                }
            )
            result = provider.summarize(text)
            title = result.title
            summary = result.summary
            tags = provider.suggest_tags(text)
        except Exception:
            pass

    safe_title = "".join(c if c.isalnum() or c in "._- " else "_" for c in title)[:50]
    raw_dir = kb.files.raw_dir / date.today().isoformat()
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{file_hash[:12]}_{safe_title}.md"
    raw_path.write_text(text, encoding="utf-8")

    if on_progress:
        on_progress("embedding")
    if on_progress:
        on_progress("indexing")
    doc_id = _index_document(
        kb,
        config,
        text,
        {"title": title, "source": source, "doc_type": doc_type},
        {
            "title": title,
            "source_path": str(raw_path),
            "source_url": source if source.startswith("http") else None,
            "doc_type": doc_type,
            "summary": summary,
            "file_hash": file_hash,
        },
    )
    if tags:
        kb.sqlite.set_document_tags(doc_id, tags)

    if on_progress:
        on_progress("completed")
    return doc_id
