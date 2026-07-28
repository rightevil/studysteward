from dataclasses import dataclass, field
from pathlib import Path
import os

@dataclass
class Config:
    ai_provider: str = "claude"
    ai_api_key: str = ""
    ai_model: str = ""
    ai_base_url: str = ""
    mineru_token: str = ""
    data_dir: Path = Path.home() / ".studysteward"
    chunk_size: int = 500
    chunk_overlap: int = 50

BUNDLED_ENV = Path(__file__).parent.parent / ".env"

def _read_env(path: Path) -> dict[str, str]:
    env = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env

def load_config(path: str | Path | None = None) -> Config:
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.append(Path.home() / ".studysteward" / ".env")
    candidates.append(BUNDLED_ENV)

    env = {}
    for p in candidates:
        if p.exists():
            env = _read_env(p)
            break

    return Config(
        ai_provider=env.get("STUDYSTEWARD_AI_PROVIDER", "claude"),
        ai_api_key=env.get("STUDYSTEWARD_AI_API_KEY", ""),
        ai_model=env.get("STUDYSTEWARD_AI_MODEL", ""),
        ai_base_url=env.get("STUDYSTEWARD_AI_BASE_URL", ""),
        mineru_token=env.get("MINERU_API_TOKEN", ""),
        data_dir=Path(os.path.expanduser(env.get("STUDYSTEWARD_DATA_DIR", "~/.studysteward"))),
        chunk_size=int(env.get("STUDYSTEWARD_CHUNK_SIZE", "500")),
        chunk_overlap=int(env.get("STUDYSTEWARD_CHUNK_OVERLAP", "50")),
    )
