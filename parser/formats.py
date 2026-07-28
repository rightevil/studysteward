from pathlib import Path


MINERU_FORMATS = {
    ".pdf", ".docx", ".pptx", ".xls", ".xlsx",
    ".png", ".jpg", ".jpeg", ".jp2", ".webp", ".gif", ".bmp",
}
TEXT_FORMATS = {".txt", ".md", ".markdown", ".rst", ".csv"}
HTML_FORMATS = {".html", ".htm"}
SUPPORTED_FORMATS = MINERU_FORMATS | TEXT_FORMATS | HTML_FORMATS


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_FORMATS
