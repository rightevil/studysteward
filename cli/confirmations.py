from dataclasses import dataclass
from typing import Callable

from cli.events import emit


@dataclass
class _Confirmation:
    question: str
    on_yes: Callable[[], None]
    on_no: Callable[[], None]


_pending: _Confirmation | None = None


def request_confirmation(
    question: str,
    on_yes: Callable[[], None],
    on_no: Callable[[], None],
):
    """Request a yes/no answer from the next input line."""
    global _pending
    _pending = _Confirmation(question=question, on_yes=on_yes, on_no=on_no)
    emit("chat.append", text=question)


def consume_confirmation(answer: str) -> bool:
    """Consume input when a confirmation is pending."""
    global _pending
    if _pending is None:
        return False

    normalized = answer.strip().lower()
    if normalized in {"y", "yes"}:
        confirmation = _pending
        _pending = None
        confirmation.on_yes()
    elif normalized in {"", "n", "no"}:
        confirmation = _pending
        _pending = None
        confirmation.on_no()
    elif normalized in {"c", "cancel", "/cancel"}:
        _pending = None
        emit("chat.append", text="Directory import cancelled.")
    else:
        emit("chat.append", text="Please answer y/yes, n/no, or cancel.")
    return True


def clear_confirmation():
    """Clear any pending prompt, primarily for application reset and tests."""
    global _pending
    _pending = None
