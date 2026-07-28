"""
Layout builder — prompt_toolkit Application layout.
Header + Tasks + Chat + Input at bottom.
"""
from prompt_toolkit.layout import Dimension, Layout, HSplit, Window, VSplit
from prompt_toolkit.layout.controls import FormattedTextControl, BufferControl
from prompt_toolkit.application.current import get_app
from prompt_toolkit.filters import Condition, has_completions
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.menus import CompletionsMenuControl
from prompt_toolkit.buffer import Buffer
from cli.widgets.task_panel import task_panel
from cli.widgets.directory_selector import directory_selector
from cli.widgets.live_output import live_output


_selector_active = Condition(lambda: directory_selector.active)
_tasks_visible = Condition(task_panel.has_visible_tasks)


def _rule() -> Window:
    return Window(
        height=1,
        width=lambda: get_app().output.get_size().columns,
        char="─",
    )


def _completion_height() -> int:
    state = get_app().current_buffer.complete_state
    return min(8, len(state.completions)) if state else 0


def build_layout(input_buffer: Buffer) -> Layout:
    """Create the full application layout."""
    content = HSplit([
            ConditionalContainer(live_output.window, filter=~_selector_active),
            ConditionalContainer(directory_selector.window, filter=_selector_active),
            ConditionalContainer(
                HSplit([
                    _rule(),
                    task_panel.window,
                ]),
                filter=_tasks_visible,
            ),
            _rule(),
            ConditionalContainer(
                VSplit([
                    Window(content=FormattedTextControl(text="  > "), width=4, height=1, dont_extend_width=True),
                    Window(content=BufferControl(buffer=input_buffer), height=1),
                ]),
                filter=~_selector_active,
            ),
            ConditionalContainer(
                Window(
                    content=FormattedTextControl(
                        text="  Up/Down move  Space select  Enter import  E open directory  Q back/cancel"
                    ),
                    height=1,
                ),
                filter=_selector_active,
            ),
            _rule(),
            ConditionalContainer(
                HSplit([
                    VSplit([
                        Window(
                            content=CompletionsMenuControl(),
                            width=Dimension(min=8),
                            height=_completion_height,
                            dont_extend_width=True,
                            style="class:completion-menu",
                        ),
                        Window(),
                    ]),
                    Window(),
                ]),
                filter=has_completions & ~_selector_active,
            ),
        ])
    return Layout(content)
