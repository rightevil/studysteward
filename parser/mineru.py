from pathlib import Path
from llama_index.readers.mineru import MinerUReader
from parser.formats import HTML_FORMATS, MINERU_FORMATS, TEXT_FORMATS
from parser.text import parse_text_file

_reader: MinerUReader | None = None
_precision_reader: MinerUReader | None = None


def _get_reader() -> MinerUReader:
    global _reader
    if _reader is None:
        _reader = MinerUReader()
    return _reader


def _get_precision_reader() -> MinerUReader:
    """Get a precision-mode reader (supports HTML, requires token)."""
    global _precision_reader
    if _precision_reader is None:
        from core.config import load_config
        cfg = load_config()
        _precision_reader = MinerUReader(mode="precision", token=cfg.mineru_token)
    return _precision_reader


def parse_file(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()

    if ext in TEXT_FORMATS:
        return parse_text_file(path)

    if ext in MINERU_FORMATS or ext in HTML_FORMATS:
        # HTML files need precision mode
        if ext in HTML_FORMATS:
            reader = _get_precision_reader()
        else:
            reader = _get_reader()
        documents = reader.load_data(str(path))
        text = "\n\n".join(d.text for d in documents)

        doc_type = "web" if ext in HTML_FORMATS else (
            "image" if ext in {".png", ".jpg", ".jpeg", ".jp2", ".webp", ".gif", ".bmp"}
            else "office" if ext in {".docx", ".pptx", ".xls", ".xlsx"}
            else "pdf"
        )
        return text, doc_type

    raise ValueError(f"Unsupported format: {ext}")


def parse_url(url: str, on_progress=None) -> tuple[str, str]:
    """Parse a web page URL via LlamaIndex Web Reader. Returns (markdown, 'web')."""
    from llama_index.readers.web import SimpleWebPageReader
    if on_progress:
        on_progress("Fetching page...")
    documents = SimpleWebPageReader(html_to_text=True).load_data([url])
    text = "\n\n".join(d.text for d in documents)
    return text, "web"
