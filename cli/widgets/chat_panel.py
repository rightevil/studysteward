"""
Chat panel — conversation history with prompt_toolkit HTML rendering.
"""
import re
import threading
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout import Window
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.utils import get_cwidth


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

    @staticmethod
    def _fit_cell(value: str, width: int) -> str:
        value = re.sub(r"[*_`]", "", value.strip())
        result = []
        used = 0
        for char in value:
            char_width = max(0, get_cwidth(char))
            if used + char_width > width:
                break
            result.append(char)
            used += char_width
        if used < get_cwidth(value) and width > 1:
            while result and used + 1 > width:
                used -= max(0, get_cwidth(result.pop()))
            result.append("…")
            used += 1
        return "".join(result) + " " * max(0, width - used)

    @classmethod
    def _render_table(cls, rows: list[list[str]], width: int) -> list[str]:
        column_count = max(len(row) for row in rows)
        normalized = [row + [""] * (column_count - len(row)) for row in rows]
        natural = [
            max(get_cwidth(re.sub(r"[*_`]", "", row[index].strip())) for row in normalized)
            for index in range(column_count)
        ]
        separators = 3 * (column_count - 1)
        available = max(column_count * 6, width - 4 - separators)
        if sum(natural) <= available:
            widths = natural
        else:
            per_column = max(6, available // column_count)
            widths = [min(size, per_column) for size in natural]

        rendered = []
        for row_index, row in enumerate(normalized):
            rendered.append(
                "  "
                + " │ ".join(
                    cls._fit_cell(cell, widths[index])
                    for index, cell in enumerate(row)
                ).rstrip()
            )
            if row_index == 0:
                rendered.append(
                    "  " + "─┼─".join("─" * column_width for column_width in widths)
                )
        return rendered

    @classmethod
    def _prepare_markdown(cls, text: str, width: int) -> str:
        lines = text.split("\n")
        output = []
        index = 0
        in_code = False
        while index < len(lines):
            line = lines[index]
            if line.strip().startswith("```"):
                in_code = not in_code
                index += 1
                continue
            if in_code:
                output.append(f"    {line}")
                index += 1
                continue

            if (
                line.strip().startswith("|")
                and index + 1 < len(lines)
                and re.match(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$", lines[index + 1])
            ):
                table_rows = []
                table_index = index
                while (
                    table_index < len(lines)
                    and lines[table_index].strip().startswith("|")
                ):
                    if table_index != index + 1:
                        table_rows.append(
                            [
                                cell.strip()
                                for cell in lines[table_index]
                                .strip()
                                .strip("|")
                                .split("|")
                            ]
                        )
                    table_index += 1
                output.extend(cls._render_table(table_rows, width))
                index = table_index
                continue

            if re.match(r"^\s*-\s+", line):
                line = re.sub(r"^\s*-\s+", "  • ", line)
            elif re.match(r"^\s*\d+\.\s+", line):
                line = "  " + line.lstrip()
            elif line.startswith("> "):
                line = "  │ " + line[2:]
            elif re.match(r"^\s*---+\s*$", line):
                line = "─" * min(max(20, width - 4), 100)
            output.append(line)
            index += 1
        return "\n".join(output)

    def _render(self):
        with self._lock:
            text = "\n".join(self.lines)
        prepared = self._prepare_markdown(text, 120)
        return HTML("\n".join(self._format_line(line) for line in prepared.split("\n")))

    @classmethod
    def format_messages(cls, text: str, width: int = 120) -> HTML:
        """Format completed transcript text for terminal output."""
        prepared = cls._prepare_markdown(text, width)
        return HTML("\n".join(cls._format_line(line) for line in prepared.split("\n")))


chat_panel = ChatPanel()
