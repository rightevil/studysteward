"""
prompt_toolkit Application - main event loop and UI.
"""
import asyncio
import threading
import time

from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import Condition, has_completions
from prompt_toolkit.formatted_text import FormattedText, to_formatted_text
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth

from cli.events import emit, poll
from cli.confirmations import consume_confirmation
from cli.layout import build_layout
from cli.widgets.chat_panel import chat_panel
from cli.widgets.directory_selector import directory_selector
from cli.widgets.live_output import live_output
from cli.widgets.task_panel import task_panel
from core.commands import get as get_cmd, get_all

# Agent status (shared with layout)
_agent_state_val = "idle"
_selector_active = Condition(lambda: directory_selector.active)


def _agent_state():
    return _agent_state_val


class _CmdCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/"):
            prefix = text[1:]
            for cmd in get_all():
                for candidate in [cmd.name, *cmd.aliases]:
                    if candidate.startswith(prefix):
                        meta = cmd.help if candidate == cmd.name else f"{cmd.help} -> /{cmd.name}"
                        yield Completion(
                            candidate,
                            start_position=-len(prefix),
                            display=f"/{candidate}",
                            display_meta=meta,
                        )


def _execute(raw: str):
    """Execute a command or chat message."""
    raw = raw.strip()
    if not raw:
        return
    emit("chat.user", text=raw)
    if consume_confirmation(raw):
        return

    if raw.startswith("/"):
        parts = raw[1:].split(maxsplit=1)
        name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        if not name:
            emit("chat.append", text="Type /help for commands.")
            return

        cmd = get_cmd(name)
        if cmd:
            try:
                cmd.handler(args)
            except Exception as e:
                emit("chat.append", text=f"Error: {e}")
        else:
            emit("chat.append", text=f"Unknown '/{name}'. Type /help.")
    else:
        emit("chat.stream", text="_Thinking..._")
        threading.Thread(target=_chat_thread, args=(raw,), daemon=True).start()


def _chat_thread(question: str):
    """Run RAG Q&A in background and stream results via the event bus."""
    global _agent_state_val
    from core.config import load_config
    from core.kb_manager import KBManager
    from core.query import ask

    _agent_state_val = "retrieving"
    emit("agent.status", state="retrieving")
    cfg = load_config()
    kb = KBManager(cfg)
    accumulated = ""
    last_update = time.time()

    try:
        for token in ask(kb, cfg, question):
            accumulated += token
            now = time.time()
            if now - last_update > 0.2:
                emit("chat.stream", text=accumulated)
                _agent_state_val = "answering"
                emit("agent.status", state="answering")
                last_update = now
        emit("chat.commit", text=accumulated)
    except Exception as e:
        emit("chat.commit", text=f"Error: {e}")
    finally:
        _agent_state_val = "idle"
        emit("agent.status", state="idle")


def _print_transcript(app: Application, messages: list[tuple[str, str]]):
    """Print completed messages above the live UI into terminal scrollback."""
    width = app.output.get_size().columns
    fragments = []
    for kind, text in messages:
        if kind == "user":
            line = f"  > {text}"
            padding = " " * max(0, width - get_cwidth(line))
            fragments.append(("class:user-input", line + padding + "\n"))
        else:
            fragments.extend(
                to_formatted_text(chat_panel.format_messages(text + "\n", width=width))
            )
    formatted = FormattedText(fragments)
    run_in_terminal(lambda: app.print_text(formatted))


def _process_events(app: Application | None = None):
    """Apply UI updates from the event bus in the main loop."""
    global _agent_state_val
    transcript: list[tuple[str, str]] = []
    for ev in poll():
        if ev.type == "chat.user":
            text = ev.payload["text"]
            chat_panel.append(text)
            transcript.append(("user", text))
        elif ev.type == "chat.append":
            text = ev.payload["text"]
            chat_panel.append(text)
            transcript.append(("message", text))
        elif ev.type == "chat.stream":
            text = ev.payload["text"]
            chat_panel.append_streaming(text)
            live_output.update(text)
        elif ev.type == "chat.commit":
            text = ev.payload["text"]
            chat_panel.commit_streaming(text)
            live_output.clear()
            transcript.append(("message", text))
        elif ev.type == "task.add":
            task_panel.add(ev.payload["task_id"], ev.payload["source"])
        elif ev.type == "task.update":
            payload = dict(ev.payload)
            task_id = payload.pop("task_id")
            task_panel.update(task_id, **payload)
        elif ev.type == "task.done":
            task_panel.update(ev.payload["task_id"], status="done", progress=None, phase="completed")
            text = "  Done - document saved"
            chat_panel.append(text)
            transcript.append(("message", text))
        elif ev.type == "task.failed":
            task_panel.update(ev.payload["task_id"], status="failed")
            text = f"  Failed: {ev.payload['error']}"
            chat_panel.append(text)
            transcript.append(("message", text))
        elif ev.type == "agent.status":
            _agent_state_val = ev.payload.get("state", "idle")
        elif ev.type == "directory.select":
            directory_selector.open(
                root=ev.payload["root"],
                files=ev.payload["files"],
                recursive=ev.payload["recursive"],
                on_confirm=ev.payload["on_confirm"],
                on_cancel=ev.payload["on_cancel"],
            )
    if transcript and app is not None and app.is_running:
        _print_transcript(app, transcript)


async def _event_loop(app: Application, interval: float = 0.25):
    """Consume UI events for as long as the application is running."""
    while app.is_running:
        _process_events(app)
        app.invalidate()
        await asyncio.sleep(interval)


def _submit(input_buffer: Buffer):
    """Store and execute the current non-empty input."""
    text = input_buffer.text
    input_buffer.reset(append_to_history=bool(text.strip()))
    _execute(text)


def create_app() -> Application:
    """Build and return the prompt_toolkit Application."""
    input_buffer = Buffer(
        completer=_CmdCompleter(),
        complete_while_typing=True,
        history=InMemoryHistory(),
        multiline=False,
    )

    kb = KeyBindings()

    @kb.add("enter", filter=~_selector_active)
    def _(event):
        _submit(input_buffer)

    @kb.add("c-c")
    def _(event):
        event.app.exit()

    @kb.add("up", filter=~has_completions & ~_selector_active)
    def _(event):
        input_buffer.history_backward()

    @kb.add("down", filter=~has_completions & ~_selector_active)
    def _(event):
        input_buffer.history_forward()

    @kb.add("up", filter=_selector_active)
    def _(event):
        directory_selector.move(-1)
        event.app.invalidate()

    @kb.add("down", filter=_selector_active)
    def _(event):
        directory_selector.move(1)
        event.app.invalidate()

    @kb.add(" ", filter=_selector_active)
    def _(event):
        directory_selector.toggle()
        event.app.invalidate()

    @kb.add("enter", filter=_selector_active)
    def _(event):
        directory_selector.confirm()
        event.app.invalidate()

    @kb.add("e", filter=_selector_active)
    def _(event):
        directory_selector.enter_directory()
        event.app.invalidate()

    @kb.add("q", filter=_selector_active)
    def _(event):
        directory_selector.back()
        event.app.invalidate()

    app = Application(
        layout=build_layout(input_buffer),
        key_bindings=kb,
        full_screen=False,
        mouse_support=False,
        refresh_interval=0.5,
        style=Style.from_dict({"user-input": "bg:#3a3a3a #ffffff"}),
    )
    return app


def run():
    """Start the terminal application."""
    from core.config import load_config

    cfg = load_config()
    print(
        "StudySteward  |  Embedding: BAAI/bge-small-zh"
        f"  |  AI: {cfg.ai_provider}/{cfg.ai_model or 'default'}"
    )
    print("Type /help for commands, or just ask a question.")
    print()
    print()
    app = create_app()
    app.run(pre_run=lambda: app.create_background_task(_event_loop(app)))
