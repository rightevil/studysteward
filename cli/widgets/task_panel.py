"""
Task panel widget — shows active tasks with phase, not just percentages.
"""
import time
import threading
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout import Window


class TaskPanel:
    """Displays task queue as a Window in the layout."""

    def __init__(self):
        self.tasks = []
        self._lock = threading.Lock()
        self.control = FormattedTextControl(text=self._render)
        self.window = Window(content=self.control, height=5, wrap_lines=False)

    def add(self, task_id: int, source: str):
        with self._lock:
            self.tasks.append(
                {
                    "id": task_id,
                    "source": source,
                    "status": "queued",
                    "progress": None,
                    "phase": "waiting",
                }
            )
            self.tasks[:] = self.tasks[-20:]

    def update(self, task_id: int, **kw):
        with self._lock:
            for t in self.tasks:
                if t["id"] == task_id:
                    t.update(kw)
                    if kw.get("status") == "done":
                        t["done_time"] = time.time()
                    break

    def _render(self) -> str:
        with self._lock:
            now = time.time()
            active = [t for t in self.tasks if t["status"] in ("queued", "running")][:2]
            done = [t for t in self.tasks if t["status"] == "done" and now - t.get("done_time", 0) < 5][:1]

        if not active and not done:
            return "  No active tasks"

        lines = []
        for t in active + done:
            tag = "#" + str(t["id"])
            src = t["source"][:40]
            status = t["status"]
            phase = t.get("phase", "")
            pct = t.get("progress")

            if status == "done":
                lines.append(f"  Done  {tag} {src}")
            elif status == "queued":
                lines.append(f"  Wait  {tag} {src}")
            elif status == "running":
                progress = f" ({pct}%)" if pct is not None else ""
                lines.append(f"  Run   {tag} {phase}{progress}  {src}")
            else:
                lines.append(f"  ?     {tag} {status}")
        return "\n".join(lines)


# Singleton
task_panel = TaskPanel()
