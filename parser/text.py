from pathlib import Path

def parse_text_file(path: Path) -> tuple[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.read(), "note"
