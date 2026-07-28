"""
StudySteward REPL — thin entry point.
"""
import os
from pathlib import Path


def _load_dotenv():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())


def _check_model():
    model_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-zh" / "snapshots"
    if model_dir.exists():
        for snap in model_dir.iterdir():
            if (snap / "pytorch_model.bin").exists() or (snap / "model.safetensors").exists():
                return True
    return False


def main():
    _load_dotenv()

    if not _check_model():
        from rich.console import Console
        Console().print("[yellow]Model not installed. Run: study setup[/yellow]")
        return

    import ai.provider  # noqa
    import cli.commands  # noqa

    from cli.app import run
    run()
