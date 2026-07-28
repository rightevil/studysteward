"""
Layout builder — prompt_toolkit Application layout.
Header + Tasks + Chat + Input at bottom.
"""
from prompt_toolkit.layout import Layout, HSplit, Window, VSplit
from prompt_toolkit.layout.controls import FormattedTextControl, BufferControl
from prompt_toolkit.layout.containers import Float, FloatContainer, WindowAlign
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.buffer import Buffer
from cli.widgets.task_panel import task_panel
from cli.widgets.chat_panel import chat_panel


def _header_text():
    from core.config import load_config
    from cli.app import _agent_state
    cfg = load_config()
    state = _agent_state()
    state_map = {"idle": "", "retrieving": "retrieving...", "answering": "thinking...", "reading": "reading..."}
    status = state_map.get(state, "")
    return (
        f"StudySteward  |  Embedding: BAAI/bge-small-zh  |  AI: {cfg.ai_provider}/{cfg.ai_model or 'default'}"
        f"{'  |  ' + status if status else ''}"
    )


def build_layout(input_buffer: Buffer) -> Layout:
    """Create the full application layout."""
    header = Window(
        content=FormattedTextControl(text=_header_text),
        height=1, align=WindowAlign.CENTER,
    )
    content = HSplit([
            header,
            Window(height=1, char="─"),
            task_panel.window,
            Window(height=1, char="─"),
            chat_panel.window,
            Window(height=1, char="─"),
            VSplit([
                Window(content=FormattedTextControl(text="  > "), width=4, height=1, dont_extend_width=True),
                Window(content=BufferControl(buffer=input_buffer), height=1),
            ]),
        ])
    return Layout(
        FloatContainer(
            content=content,
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(
                        max_height=8,
                        scroll_offset=1,
                        display_arrows=True,
                    ),
                )
            ],
        )
    )
