"""
prompt_toolkit Application - main event loop and UI.
"""
import asyncio
import threading
import time

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import Condition, has_completions
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings

from cli.events import emit, poll
from cli.layout import build_layout
from cli.widgets.chat_panel import chat_panel
from cli.widgets.task_panel import task_panel
from core.commands import get as get_cmd, get_all

# Agent status (shared with layout)
_agent_state_val = "idle"
_mouse_support_enabled = True


def _agent_state():
    return _agent_state_val


def _toggle_mouse_support() -> bool:
    """Toggle terminal mouse tracking and return the new state."""
    global _mouse_support_enabled
    _mouse_support_enabled = not _mouse_support_enabled
    return _mouse_support_enabled


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
        emit("chat.append", text=f"**[You]** {raw}")
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


def _process_events():
    """Apply UI updates from the event bus in the main loop."""
    global _agent_state_val
    for ev in poll():
        if ev.type == "chat.append":
            chat_panel.append(ev.payload["text"])
        elif ev.type == "chat.stream":
            chat_panel.append_streaming(ev.payload["text"])
        elif ev.type == "chat.commit":
            chat_panel.commit_streaming(ev.payload["text"])
        elif ev.type == "task.add":
            task_panel.add(ev.payload["task_id"], ev.payload["source"])
        elif ev.type == "task.update":
            payload = dict(ev.payload)
            task_id = payload.pop("task_id")
            task_panel.update(task_id, **payload)
        elif ev.type == "task.done":
            task_panel.update(ev.payload["task_id"], status="done", progress=None, phase="completed")
            chat_panel.append(f"  Done - document #{ev.payload['doc_id']}")
        elif ev.type == "task.failed":
            task_panel.update(ev.payload["task_id"], status="failed")
            chat_panel.append(f"  Failed: {ev.payload['error']}")
        elif ev.type == "agent.status":
            _agent_state_val = ev.payload.get("state", "idle")


async def _event_loop(app: Application, interval: float = 0.25):
    """Consume UI events for as long as the application is running."""
    while app.is_running:
        _process_events()
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

    @kb.add("enter")
    def _(event):
        _submit(input_buffer)

    @kb.add("c-c")
    def _(event):
        event.app.exit()

    @kb.add("up", filter=~has_completions)
    def _(event):
        input_buffer.history_backward()

    @kb.add("down", filter=~has_completions)
    def _(event):
        input_buffer.history_forward()

    @kb.add("f2")
    def _(event):
        enabled = _toggle_mouse_support()
        mode = "scroll" if enabled else "select"
        emit("chat.append", text=f"Mouse mode: {mode} (F2 to switch)")
        event.app.invalidate()

    app = Application(
        layout=build_layout(input_buffer),
        key_bindings=kb,
        full_screen=False,
        mouse_support=Condition(lambda: _mouse_support_enabled),
        refresh_interval=0.5,
    )
    return app


def run():
    """Start the terminal application."""
    app = create_app()
    app.run(pre_run=lambda: app.create_background_task(_event_loop(app)))
