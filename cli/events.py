"""
Event bus for UI-worker communication.
Workers emit events, and the UI loop applies them on the main thread.
"""
from dataclasses import dataclass, field
from queue import Empty, Queue


@dataclass
class Event:
    type: str
    payload: dict = field(default_factory=dict)


_bus = Queue()
_interval_hooks = []  # [(interval_sec, callback)]


def emit(event_type: str, **payload):
    """Thread-safe: push an event to the shared bus."""
    _bus.put(Event(type=event_type, payload=payload))


def poll() -> list[Event]:
    """Drain all pending events. Called from the main UI loop."""
    events = []
    while True:
        try:
            events.append(_bus.get_nowait())
        except Empty:
            break
    return events


def on_interval(seconds: float, callback):
    """Register a callback to run every N seconds in the main loop."""
    _interval_hooks.append((seconds, callback))
    _interval_hooks.sort(key=lambda x: x[0])
