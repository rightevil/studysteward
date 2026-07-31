"""
Command handlers - all /commands register here.
Imported at startup to register with the command registry.
"""
import io
import json
import sys
import threading
from pathlib import Path

from cli.confirmations import request_confirmation
from cli.events import emit
from core.commands import Command, get_all, register
from parser.formats import is_supported
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


def _with_display_numbers(documents: list[dict]) -> list[dict]:
    """Add stable-for-this-list, gapless numbers without changing database IDs."""
    total = len(documents)
    return [
        {**document, "display_no": total - index}
        for index, document in enumerate(documents)
    ]


def _resolve_document_number(kb, display_no: int) -> dict | None:
    """Resolve the number shown by /list to a document with its real ID."""
    documents = _with_display_numbers(kb.list_documents())
    return next(
        (document for document in documents if document["display_no"] == display_no),
        None,
    )


def _cmd_help(args: str):
    lines = ["Available commands:", "-" * 40]
    for cmd in get_all():
        aliases = f" ({', '.join(cmd.aliases)})" if cmd.aliases else ""
        lines.append(f"  /{cmd.name:<12} {cmd.help}{aliases}")
    for line in lines:
        _chat(line)


def _scan_directory(directory: Path, recursive: bool) -> tuple[list[Path], int]:
    """Return supported files and the number of unsupported files."""
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    files = sorted(
        (path for path in candidates if path.is_file()),
        key=lambda path: str(path).casefold(),
    )
    supported = [path for path in files if is_supported(path)]
    return supported, len(files) - len(supported)


def _batch_ingest_files(files: list[Path]):
    """Create tasks and import files sequentially in one background worker."""
    from core.config import load_config
    from core.kb_manager import KBManager
    from core.task_queue import TaskQueue
    from workers.ingest_worker import ingest_file

    try:
        cfg = load_config()
        kb = KBManager(cfg)
        queue = TaskQueue(kb.sqlite)
        jobs = []
        for path in files:
            source = str(path)
            task_id = queue.enqueue(source)
            jobs.append((task_id, source))
            emit("task.add", task_id=task_id, source=source)

        emit("chat.append", text=f"Queued {len(jobs)} files for sequential import.")
        completed = 0
        for task_id, source in jobs:
            if ingest_file(task_id, cfg, kb, source) is not None:
                completed += 1
        emit(
            "chat.append",
            text=f"Batch finished: {completed} succeeded, {len(jobs) - completed} failed.",
        )
    except Exception as exc:
        emit("chat.append", text=f"Batch import failed: {exc}")


def _start_batch_import(files: list[Path], ignored: int, recursive: bool):
    if not files:
        scope = "directory tree" if recursive else "directory"
        _chat(f"No supported files found in the {scope}. Ignored {ignored} files.")
        return

    scope = "recursively" if recursive else "from this directory"
    _chat(f"Starting batch import: {len(files)} files {scope}; ignored {ignored} files.")
    threading.Thread(target=_batch_ingest_files, args=(files,), daemon=True).start()


def _open_directory_selection(
    directory: Path,
    files: list[Path],
    ignored: int,
    recursive: bool,
):
    """Ask the UI to select documents before starting a batch import."""
    if not files:
        scope = "directory tree" if recursive else "directory"
        _chat(f"No supported files found in the {scope}. Ignored {ignored} files.")
        return

    emit(
        "directory.select",
        root=directory,
        files=files,
        recursive=recursive,
        on_confirm=lambda selected: _start_batch_import(selected, ignored, recursive),
        on_cancel=lambda: _chat("Directory import cancelled."),
    )


def _prompt_directory_import(directory: Path):
    direct_files, direct_ignored = _scan_directory(directory, recursive=False)
    recursive_files, recursive_ignored = _scan_directory(directory, recursive=True)

    if not recursive_files:
        _chat(f"No supported files found under '{directory}'. Ignored {recursive_ignored} files.")
        return

    question = (
        f"Directory found: {len(direct_files)} supported files here, "
        f"{len(recursive_files)} including subdirectories "
        f"({recursive_ignored} ignored). Import recursively? [y/N]"
    )
    request_confirmation(
        question,
        on_yes=lambda: _open_directory_selection(
            directory, recursive_files, recursive_ignored, True
        ),
        on_no=lambda: _open_directory_selection(
            directory, direct_files, direct_ignored, False
        ),
    )


def _cmd_ingest(args: str):
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
        if path.is_dir():
            _prompt_directory_import(path.resolve())
            return
        if not is_supported(path):
            _chat(f"Unsupported format: {path.suffix or '(no extension)'}")
            return

    from core.config import load_config
    from core.kb_manager import KBManager
    from core.task_queue import TaskQueue

    cfg = load_config()
    kb = KBManager(cfg)

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
    docs = _with_display_numbers(_load_documents())
    _chat("Documents:")
    _chat(f"  {'No.':>4}  {'Title':<44}  Type")
    _chat(f"  {'-' * 4}  {'-' * 44}  {'-' * 10}")
    if docs:
        for doc in docs:
            title = _fit_display_width(doc["title"], 44)
            _chat(f"  {doc['display_no']:>4}  {title}  {doc['type']}")
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


def _run_research(goal: str):
    from ai.provider import create_provider_from_env
    from core.config import load_config
    from core.kb_manager import KBManager
    from research.agent import ResearchAgent
    from research.model import ProviderResearchModel
    from research.tools import ResearchTools

    progress: list[str] = []
    kb = None
    try:
        config = load_config()
        kb = KBManager(config)
        provider = create_provider_from_env(
            {
                "provider": config.ai_provider,
                "api_key": config.ai_api_key,
                "model": config.ai_model,
                "base_url": config.ai_base_url,
            }
        )

        def show_progress(message: str):
            progress.append(message)
            emit("chat.stream", text="\n".join(progress[-8:]))

        agent = ResearchAgent(
            ProviderResearchModel(provider),
            ResearchTools(kb),
            trace_store=kb.sqlite,
            max_steps=8,
        )
        result = agent.run(goal, on_event=show_progress)
        footer = (
            f"\n\nResearch run #{result.run_id} | "
            f"{len(result.steps)} steps | {result.status}"
        )
        emit("chat.commit", text=result.report + footer)
    except Exception as exc:
        emit("chat.commit", text=f"Research failed: {exc}")
    finally:
        if kb is not None:
            kb.sqlite.close()
        emit("agent.status", state="idle")


def _cmd_research(args: str):
    goal = args.strip()
    if not goal:
        _chat("Usage: /research <goal>")
        return
    emit("agent.status", state="researching")
    emit("chat.stream", text="_Planning research..._")
    threading.Thread(target=_run_research, args=(goal,), daemon=True).start()


def _cmd_trace(args: str):
    from core.config import load_config
    from storage.sqlite import SQLiteStore

    raw_id = args.strip()
    if raw_id:
        try:
            run_id = int(raw_id)
        except ValueError:
            _chat("Usage: /trace [research_run_id]")
            return
    else:
        run_id = None

    config = load_config()
    store = SQLiteStore(config.data_dir / "kb.db")
    store.init_schema()
    run = store.get_research_run(run_id)
    if not run:
        store.close()
        _chat("No research run found.")
        return
    steps = store.get_research_steps(run["id"])
    store.close()
    _chat(f"Research run #{run['id']}: {run['goal']}")
    _chat(f"Status: {run['status']}")
    for step in steps:
        arguments = json.loads(step["args_json"])
        observation = step["observation"].replace("\n", " ")
        if len(observation) > 240:
            observation = observation[:237] + "..."
        _chat(
            f"  {step['step_no']}. {step['tool']} "
            f"{json.dumps(arguments, ensure_ascii=False)} "
            f"({step['duration_ms']} ms)"
        )
        _chat(f"     {observation}")
    _chat("")


def _run_reindex():
    from core.config import load_config
    from core.kb_manager import KBManager

    kb = None
    try:
        kb = KBManager(load_config())
        result = kb.repair_legacy_index()
        emit(
            "chat.commit",
            text=(
                "Index repair completed.\n"
                f"  Vectors scanned: {result['scanned']}\n"
                f"  Vectors mapped: {result['mapped']}\n"
                f"  Metadata repaired: {result['repaired']}\n"
                f"  Documents mapped: {result['documents']}\n"
                f"  Unmatched vectors: {result['unmatched']}"
            ),
        )
    except Exception as exc:
        emit("chat.commit", text=f"Index repair failed: {exc}")
    finally:
        if kb is not None:
            kb.sqlite.close()


def _cmd_reindex(args: str):
    emit("chat.stream", text="_Repairing legacy index metadata..._")
    threading.Thread(target=_run_reindex, daemon=True).start()


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
    from ai.reranker import is_reranker_installed
    from core.config import load_config

    cfg = load_config()
    _chat(f"  AI: {cfg.ai_provider}/{cfg.ai_model or 'default'}")
    _chat(f"  MinerU: {'configured' if cfg.mineru_token else 'not set'}")
    status = "installed" if is_reranker_installed(cfg.reranker_model) else "not installed"
    _chat(f"  Reranker: {cfg.reranker_model} ({status})")


def _cmd_setup(args: str):
    from core.config import load_config

    target = args.strip().casefold() or "embedding"
    if target not in {"embedding", "reranker", "all"}:
        _chat("Usage: /setup [embedding|reranker|all]")
        return

    models = []
    if target in {"embedding", "all"}:
        models.append(("BAAI/bge-small-zh", "embedding"))
    if target in {"reranker", "all"}:
        models.append((load_config().reranker_model, "reranker"))

    old_stderr, old_stdout = sys.stderr, sys.stdout
    sys.stderr = io.StringIO()
    sys.stdout = io.StringIO()
    try:
        for model_name, model_type in models:
            _chat(f"Downloading {model_name}...")
            if model_type == "embedding":
                from ai.embedding import get_embedder

                get_embedder(local_files_only=False)
            else:
                from ai.reranker import get_reranker

                get_reranker(model_name, local_files_only=False)
    finally:
        sys.stderr = old_stderr
        sys.stdout = old_stdout
    _chat("Done.")


def _cmd_rm(args: str):
    from core.config import load_config
    from core.kb_manager import KBManager

    try:
        display_no = int(args.strip())
    except Exception:
        _chat("Usage: /rm <document_no>")
        return

    kb = KBManager(load_config())
    doc = _resolve_document_number(kb, display_no)
    if not doc:
        _chat(f"Document No. {display_no} not found. Run /list to see current numbers.")
        return
    kb.delete_document(doc["id"])
    _chat(f"Deleted document No. {display_no}: {doc['title']}")


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

    raw_number = args.strip()
    try:
        if raw_number.lower().startswith("d"):
            stable_id = int(raw_number[1:])
            display_no = None
        else:
            stable_id = None
            display_no = int(raw_number)
    except ValueError:
        _chat("Usage: /info <document_no|D-id>")
        return

    kb = KBManager(load_config())
    doc = (
        kb.get_document(stable_id)
        if stable_id is not None
        else _resolve_document_number(kb, display_no)
    )
    if not doc:
        reference = f"D{stable_id}" if stable_id is not None else f"No. {display_no}"
        _chat(f"Document {reference} not found. Run /list to see current documents.")
        return

    _chat(f"  Citation: [D{doc['id']}]")
    _chat(f"  Title: {doc['title']}")
    _chat(f"  Type: {doc['type']}")
    _chat(f"  Tags: {', '.join(doc.get('tags', [])) or '(none)'}")


register(Command(name="help", help="Show commands", handler=_cmd_help, aliases=["h"], priority=1))
register(Command(name="setup", help="Download embedding/reranker model", handler=_cmd_setup, priority=2))
register(Command(name="ingest", help="Import file/URL into KB", handler=_cmd_ingest, aliases=["i"], priority=10))
register(Command(name="search", help="Semantic search", handler=_cmd_search, aliases=["s"], priority=11))
register(Command(name="research", help="Run bounded research agent", handler=_cmd_research, aliases=["r"], priority=12))
register(Command(name="trace", help="Show a research execution trace", handler=_cmd_trace, priority=13))
register(Command(name="list", help="List documents", handler=_cmd_list, aliases=["ls"], priority=20))
register(Command(name="info", help="Document details by No. or D-id", handler=_cmd_info, priority=21))
register(Command(name="tags", help="Tag tree", handler=_cmd_tags, priority=22))
register(Command(name="rm", help="Delete document by list number", handler=_cmd_rm, aliases=["delete"], priority=23))
register(Command(name="status", help="Task queue", handler=_cmd_status, aliases=["st"], priority=30))
register(Command(name="reindex", help="Repair legacy vector mappings", handler=_cmd_reindex, priority=31))
register(Command(name="config", help="Show config", handler=_cmd_config, aliases=["cfg"], priority=40))
