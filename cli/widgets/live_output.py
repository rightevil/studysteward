import threading

from prompt_toolkit.application.current import get_app
from prompt_toolkit.layout import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.utils import get_cwidth


class LiveOutput:
    """Shows only the response currently being streamed."""

    def __init__(self):
        self.text = ""
        self._lock = threading.Lock()
        self.control = FormattedTextControl(text=self._render)
        self.window = Window(
            content=self.control,
            height=lambda: self.line_count(),
            wrap_lines=True,
        )

    def update(self, text: str):
        with self._lock:
            self.text = text

    def clear(self):
        self.update("")

    def _render(self) -> str:
        with self._lock:
            return self.text

    def line_count(self, width: int | None = None) -> int:
        with self._lock:
            text = self.text
        if not text:
            return 0
        if width is None:
            width = max(1, get_app().output.get_size().columns)

        count = 0
        for line in text.split("\n"):
            cells = get_cwidth(line)
            count += max(1, (cells + width - 1) // width)
        return count


live_output = LiveOutput()
