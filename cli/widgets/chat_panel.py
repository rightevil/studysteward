"""
Chat panel — conversation history with prompt_toolkit HTML rendering.
"""
import re
import threading
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout import Window
from prompt_toolkit.formatted_text import HTML


class _ChatWindow(Window):
    """Window that reports mouse scrolling back to its chat panel."""

    def __init__(self, panel, **kwargs):
        self._panel = panel
        super().__init__(**kwargs)

    def _scroll_up(self):
        self._panel._follow_tail = False
        super()._scroll_up()

    def _scroll_down(self):
        super()._scroll_down()
        if self._panel._is_at_bottom():
            self._panel._follow_tail = True


class ChatPanel:
    """Stores and renders conversation history."""

    def __init__(self):
        self.lines = ["Type /help for commands, or just ask a question."]
        self._streaming = False
        self._follow_tail = True
        self._lock = threading.Lock()
        self.control = FormattedTextControl(
            text=self._render,
            show_cursor=True,
            get_cursor_position=self._cursor_pos,
        )
        self.window = _ChatWindow(
            self,
            content=self.control,
            wrap_lines=True,
        )

    def _cursor_pos(self):
        """Return a Point at end of text."""
        text = self._render_text()
        lines = text.split("\n")
        y = max(0, len(lines) - 1)
        x = len(lines[-1]) if lines else 0
        if not self._follow_tail:
            y = min(self.window.vertical_scroll, y)
            x = 0
        from prompt_toolkit.layout.screen import Point
        return Point(x=x, y=y)

    def _render_text(self) -> str:
        """Return plain text to compute cursor position."""
        with self._lock:
            return "\n".join(self.lines)

    def _is_at_bottom(self) -> bool:
        render_info = self.window.render_info
        if render_info is None:
            return True
        max_scroll = max(0, render_info.content_height - render_info.window_height)
        return self.window.vertical_scroll >= max_scroll

    def _follow_new_content(self, was_at_bottom: bool):
        if was_at_bottom:
            self._follow_tail = True
            self.window.vertical_scroll = 10**9

    def scroll_page_up(self):
        self._follow_tail = False
        render_info = self.window.render_info
        page_size = max(1, render_info.window_height - 1) if render_info else 10
        self.window.vertical_scroll = max(0, self.window.vertical_scroll - page_size)

    def scroll_page_down(self):
        render_info = self.window.render_info
        page_size = max(1, render_info.window_height - 1) if render_info else 10
        max_scroll = (
            max(0, render_info.content_height - render_info.window_height)
            if render_info
            else self.window.vertical_scroll + page_size
        )
        self.window.vertical_scroll = min(max_scroll, self.window.vertical_scroll + page_size)
        if self.window.vertical_scroll >= max_scroll:
            self._follow_tail = True

    def append(self, msg: str):
        was_at_bottom = self._is_at_bottom()
        with self._lock:
            self._streaming = False
            self.lines.append(msg)
            self.lines[:] = self.lines[-200:]
        self._follow_new_content(was_at_bottom)

    def append_streaming(self, text: str):
        was_at_bottom = self._is_at_bottom()
        with self._lock:
            if self._streaming and self.lines:
                self.lines[-1] = text
            else:
                self._streaming = True
                self.lines.append(text)
        self._follow_new_content(was_at_bottom)

    def commit_streaming(self, final: str):
        was_at_bottom = self._is_at_bottom()
        with self._lock:
            self._streaming = False
            if self.lines:
                self.lines[-1] = final
        self._follow_new_content(was_at_bottom)

    @staticmethod
    def _format_line(text: str) -> str:
        t = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"^#{1,3} (.+)$", r"<b><u>\1</u></b>", t, flags=re.MULTILINE)
        t = re.sub(r"`([^`]+)`", r"<i>\1</i>", t)
        return t

    def _render(self):
        with self._lock:
            html_lines = [self._format_line(line) for line in self.lines]
        return HTML("\n".join(html_lines))

    @classmethod
    def format_messages(cls, text: str) -> HTML:
        """Format completed transcript text for terminal output."""
        return HTML("\n".join(cls._format_line(line) for line in text.split("\n")))


chat_panel = ChatPanel()
