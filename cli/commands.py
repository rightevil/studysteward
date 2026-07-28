"""
Command handlers - all /commands register here.
Imported at startup to register with the command registry.
"""
import io
import sys
import threading
from pathlib import Path

from cli.events import emit
from core.commands import Command, get_all, register
from prompt_toolkit.utils import get_cwidth


def _chat(msg: str):
    emit("chat.append", text=msg)


def _fit_display_width(value: str, width: int) -> str:
    """Truncate and pad text using terminal display-cell width."""
    result = []
    used = 0
    for char in str(value).replace("\n", " "):
        char_width = max(0, get_cwidth(char))
        if used + char_width > width:
            break
        result.append(char)
        used += char_width
    return "".join(result) + " " * (width - used)


def _load_documents() -> list[dict]:
    """Read document metadata without loading the vector/embedding stack."""
    from core.config import load_config
    from storage.sqlite import SQLiteStore

    config = load_config()
    store = SQLiteStore(config.data_dir / "kb.db")
    store.init_schema()
    return store.list_documents()


def _cmd_help(args: str):
    lines = ["Available commands:", "-" * 40]
    for cmd in get_all():
        aliases = f" ({', '.join(cmd.aliases)})" if cmd.aliases else ""
        lines.append(f"  /{cmd.name:<12} {cmd.help}{aliases}")
    for line in lines:
        _chat(line)


def _cmd_ingest(args: str):
    from core.config import load_config
    from core.kb_manager import KBManager
    from core.task_queue import TaskQueue

    cfg = load_config()
    kb = KBManager(cfg)
    source = args.strip().strip('"')
    if not source:
        _chat("Usage: /ingest <file|url>")
        return

    is_url = source.startswith(("http://", "https://"))
    if not is_url:
        path = Path(source)
        if not path.exists():
            _chat(f"'{source}' not found.")
            return

    task_id = TaskQueue(kb.sqlite).enqueue(source)
    emit("task.add", task_id=task_id, source=source)
    _chat(f"Task #{task_id}: {source[:60]}")

    if is_url:
        from workers.ingest_worker import ingest_url

        threading.Thread(target=ingest_url, args=(task_id, cfg, source), daemon=True).start()
    else:
        from workers.ingest_worker import ingest_file

        threading.Thread(target=ingest_file, args=(task_id, cfg, kb, source), daemon=True).start()


def _cmd_list(args: str):
    docs = _load_documents()
    _chat("Documents:")
    _chat(f"  {'ID':>4}  {'Title':<44}  Type")
    _chat(f"  {'-' * 4}  {'-' * 44}  {'-' * 10}")
    if docs:
        for doc in docs:
            title = _fit_display_width(doc["title"], 44)
            _chat(f"  {doc['id']:>4}  {title}  {doc['type']}")
    else:
        _chat("  No documents.")
    _chat("")


def _cmd_search(args: str):
    from core.config import load_config
    from core.kb_manager import KBManager
    from core.query import search

    query = args.strip()
    if not query:
        _chat("Usage: /search <query>")
        return

    for idx, result in enumerate(search(KBManager(load_config()), query), 1):
        _chat(f"  {idx}. {result.get('doc_title', '?')} ({1 - result.get('distance', 0):.2f})")


def _cmd_tags(args: str):
    from core.config import load_config
    from core.kb_manager import KBManager

    kb = KBManager(load_config())
    tree = kb.get_tag_tree()
    if not tree:
        _chat("No tags.")
        return

    def _print(nodes, indent=0):
        for node in nodes:
            _chat(f"{'  ' * indent}- {node['name']}")
            if node.get("children"):
                _print(node["children"], indent + 1)

    _print(tree)


def _cmd_config(args: str):
    from core.config import load_config

    cfg = load_config()
    _chat(f"  AI: {cfg.ai_provider}/{cfg.ai_model or 'default'}")
    _chat(f"  MinerU: {'configured' if cfg.mineru_token else 'not set'}")


def _cmd_setup(args: str):
    _chat("Downloading BAAI/bge-small-zh...")
    old_stderr, old_stdout = sys.stderr, sys.stdout
    sys.stderr = io.StringIO()
    sys.stdout = io.StringIO()
    try:
        from ai.embedding import get_embedder

        get_embedder(local_files_only=False)
    finally:
        sys.stderr = old_stderr
        sys.stdout = old_stdout
    _chat("Done.")


def _cmd_rm(args: str):
    from core.config import load_config
    from core.kb_manager import KBManager

    try:
        doc_id = int(args.strip())
    except Exception:
        _chat("Usage: /rm <doc_id>")
        return

    kb = KBManager(load_config())
    doc = kb.get_document(doc_id)
    if not doc:
        _chat(f"#{doc_id} not found.")
        return
    kb.delete_document(doc_id)
    _chat(f"Deleted #{doc_id}")


def _cmd_status(args: str):
    from core.config import load_config
    from core.kb_manager import KBManager
    from core.task_queue import TaskQueue

    tasks = TaskQueue(KBManager(load_config()).sqlite).get_all()
    if not tasks:
        _chat("No tasks.")
        return

    icon_map = {
        "pending": ".",
        "queued": ".",
        "running": "*",
        "completed": "ok",
        "done": "ok",
        "failed": "x",
    }
    for task in tasks[:10]:
        icon = icon_map.get(task["status"], "?")
        _chat(f"  [{icon}] #{task['id']} {task['status']} {task['source'][:50]}")


def _cmd_info(args: str):
    from core.config import load_config
    from core.kb_manager import KBManager

    try:
        doc_id = int(args.strip())
    except Exception:
        _chat("Usage: /info <doc_id>")
        return

    doc = KBManager(load_config()).get_document(doc_id)
    if not doc:
        _chat(f"#{doc_id} not found.")
        return

    _chat(f"  Title: {doc['title']}")
    _chat(f"  Type: {doc['type']}")
    _chat(f"  Tags: {', '.join(doc.get('tags', [])) or '(none)'}")


register(Command(name="help", help="Show commands", handler=_cmd_help, aliases=["h"], priority=1))
register(Command(name="setup", help="Download embedding model", handler=_cmd_setup, priority=2))
register(Command(name="ingest", help="Import file/URL into KB", handler=_cmd_ingest, aliases=["i"], priority=10))
register(Command(name="search", help="Semantic search", handler=_cmd_search, aliases=["s"], priority=11))
register(Command(name="list", help="List documents", handler=_cmd_list, aliases=["ls"], priority=20))
register(Command(name="info", help="Document details", handler=_cmd_info, priority=21))
register(Command(name="tags", help="Tag tree", handler=_cmd_tags, priority=22))
register(Command(name="rm", help="Delete document", handler=_cmd_rm, aliases=["delete"], priority=23))
register(Command(name="status", help="Task queue", handler=_cmd_status, aliases=["st"], priority=30))
register(Command(name="config", help="Show config", handler=_cmd_config, aliases=["cfg"], priority=40))
