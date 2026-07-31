import re


_FENCED_CODE = re.compile(r"(```[^\n]*\n.*?(?:```|$))", re.DOTALL)
_HEADING = re.compile(r"^#{1,6}[ \t]+\S")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


def _repair_collapsed_syntax(text: str) -> str:
    """Repair common Markdown boundaries omitted by model responses."""
    # A model may emit "...title## Next" with no line break.
    text = re.sub(r"(?<=[^\s#])(#{1,6}[ \t]+)", r"\n\n\1", text)

    # Split a heading from a table header or first list item glued to it.
    text = re.sub(
        r"(?m)^(#{1,6}[ \t]+[^|\n]+?)(\|[^\n]+\|)\n(?=\|[-: |]+\|)",
        r"\1\n\n\2\n",
        text,
    )
    text = re.sub(
        r"(?m)^(#{1,6}[ \t]+.+?)(?<!\s)(-[ \t]+)",
        r"\1\n\n\2",
        text,
    )

    # Once a bullet line starts, recover later bullets glued to its prose.
    repaired_lines = []
    for line in text.split("\n"):
        if line.lstrip().startswith("- "):
            line = re.sub(
                r"(?<=\S)-[ \t]+(?=(?:\[[^\]\n]+\]|[*_]{1,2}|"
                r"[\u4e00-\u9fffA-Z]))",
                "\n- ",
                line,
            )
        repaired_lines.append(line.rstrip())
    return "\n".join(repaired_lines)


def _normalize_spacing(text: str) -> str:
    lines = text.split("\n")
    output: list[str] = []
    in_table = False

    for line in lines:
        is_heading = bool(_HEADING.match(line))
        is_table = bool(_TABLE_ROW.match(line))

        if is_heading and output and output[-1] != "":
            output.append("")
        if is_table and not in_table and output and output[-1] != "":
            output.append("")
        if in_table and not is_table and line and output and output[-1] != "":
            output.append("")

        output.append(line)
        if is_heading:
            output.append("")
        in_table = is_table

    normalized = "\n".join(output)
    return re.sub(r"\n{3,}", "\n\n", normalized).strip()


def normalize_markdown(text: str) -> str:
    """Normalize model-generated Markdown without modifying fenced code."""
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    code_blocks: list[str] = []

    def preserve_code(match: re.Match) -> str:
        token = f"STUDYSTEWARD_CODE_BLOCK_{len(code_blocks)}"
        code_blocks.append(match.group(0))
        return f"\n\n{token}\n\n"

    protected = _FENCED_CODE.sub(preserve_code, text)
    normalized = _normalize_spacing(_repair_collapsed_syntax(protected))
    for index, code_block in enumerate(code_blocks):
        normalized = normalized.replace(
            f"STUDYSTEWARD_CODE_BLOCK_{index}",
            code_block,
        )
    return normalized.strip()
