from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from prompt_toolkit.layout import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.screen import Point


@dataclass(frozen=True)
class _Entry:
    kind: Literal["all", "directory", "file"]
    label: str
    path: Path | None = None


class DirectorySelector:
    """Interactive state for selecting files from a scanned directory tree."""

    _CONTENT_START = 5

    def __init__(self):
        self.active = False
        self.root = Path()
        self.current_dir = Path()
        self.recursive = False
        self.files: tuple[Path, ...] = ()
        self.selected: set[Path] = set()
        self.cursor = 0
        self.message = ""
        self._on_confirm: Callable[[list[Path]], None] | None = None
        self._on_cancel: Callable[[], None] | None = None
        self.control = FormattedTextControl(
            text=self._render,
            get_cursor_position=self._cursor_position,
            show_cursor=False,
        )
        self.window = Window(content=self.control, wrap_lines=False)

    def open(
        self,
        root: Path,
        files: list[Path],
        recursive: bool,
        on_confirm: Callable[[list[Path]], None],
        on_cancel: Callable[[], None],
    ):
        self.root = root.resolve()
        self.current_dir = self.root
        self.recursive = recursive
        self.files = tuple(sorted((path.resolve() for path in files), key=lambda p: str(p).casefold()))
        self.selected = set()
        self.cursor = 0
        self.message = ""
        self._on_confirm = on_confirm
        self._on_cancel = on_cancel
        self.active = True

    def close(self):
        self.active = False
        self._on_confirm = None
        self._on_cancel = None

    def _entries(self) -> list[_Entry]:
        entries = [_Entry("all", f"Select all ({len(self.files)} files)")]
        if self.recursive:
            directories = set()
            for path in self.files:
                try:
                    relative = path.relative_to(self.current_dir)
                except ValueError:
                    continue
                if len(relative.parts) > 1:
                    directories.add(self.current_dir / relative.parts[0])
            entries.extend(
                _Entry("directory", f"{path.name}/", path)
                for path in sorted(directories, key=lambda p: p.name.casefold())
            )

        entries.extend(
            _Entry("file", path.name, path)
            for path in self.files
            if path.parent == self.current_dir
        )
        return entries

    def move(self, delta: int):
        entries = self._entries()
        self.cursor = max(0, min(self.cursor + delta, len(entries) - 1))
        self.message = ""

    def toggle(self):
        entry = self._entries()[self.cursor]
        if entry.kind == "all":
            if len(self.selected) == len(self.files):
                self.selected.clear()
            else:
                self.selected = set(self.files)
        elif entry.kind == "file" and entry.path is not None:
            if entry.path in self.selected:
                self.selected.remove(entry.path)
            else:
                self.selected.add(entry.path)
        else:
            self.message = "Press E to enter a directory."

    def enter_directory(self):
        entry = self._entries()[self.cursor]
        if self.recursive and entry.kind == "directory" and entry.path is not None:
            self.current_dir = entry.path
            self.cursor = 0
            self.message = ""
        else:
            self.message = "Move to a directory and press E."

    def back(self):
        if self.current_dir != self.root:
            self.current_dir = self.current_dir.parent
            self.cursor = 0
            self.message = ""
            return

        callback = self._on_cancel
        self.close()
        if callback:
            callback()

    def confirm(self):
        if not self.selected:
            self.message = "Select at least one document before confirming."
            return

        files = sorted(self.selected, key=lambda path: str(path).casefold())
        callback = self._on_confirm
        self.close()
        if callback:
            callback(files)

    def _selection_mark(self, entry: _Entry) -> str:
        if entry.kind == "directory":
            return "DIR"
        if entry.kind == "all":
            if not self.selected:
                return " "
            return "x" if len(self.selected) == len(self.files) else "-"
        return "x" if entry.path in self.selected else " "

    def _render(self) -> str:
        try:
            current = self.current_dir.relative_to(self.root)
            current_text = "." if not current.parts else str(current)
        except ValueError:
            current_text = str(self.current_dir)

        lines = [
            "Directory import selection",
            f"Root: {self.root}",
            f"Current: {current_text}  |  Selected: {len(self.selected)}/{len(self.files)}",
            self.message,
            "",
        ]
        for index, entry in enumerate(self._entries()):
            pointer = ">" if index == self.cursor else " "
            lines.append(f"{pointer} [{self._selection_mark(entry)}] {entry.label}")
        return "\n".join(lines)

    def _cursor_position(self) -> Point:
        return Point(x=0, y=self._CONTENT_START + self.cursor)


directory_selector = DirectorySelector()
