import hashlib
import shutil
from datetime import date
from pathlib import Path

def compute_hash(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

class FileStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.raw_dir = self.root / "raw"

    def store(self, file_path: Path) -> Path:
        h = compute_hash(file_path)
        today = date.today().isoformat()
        dest_dir = self.raw_dir / today
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{h[:12]}_{file_path.name}"
        shutil.copy2(file_path, dest)
        return dest

    def delete(self, stored_path: Path):
        if stored_path.exists():
            stored_path.unlink()

    def path_for_hash(self, file_hash: str) -> Path | None:
        if not self.raw_dir.exists():
            return None
        prefix = file_hash[:12]
        for f in self.raw_dir.rglob(f"{prefix}_*"):
            if compute_hash(f) == file_hash:
                return f
        return None
