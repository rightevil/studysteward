import re
import json
from dataclasses import dataclass


EVIDENCE_TOOLS = {"inspect_document"}
_CITATION = re.compile(r"\[D(\d+)\]")
_LEGACY_CITATION = re.compile(r"\[(?:\d+\s*,?\s*)+\]")
_GAP_LANGUAGE = re.compile(
    r"证据不足|资料不足|资料缺|缺少|缺失|未找到|无法(?:完成|验证|比较)|"
    r"knowledge gap|insufficient evidence|no evidence|not found",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ReportValidation:
    valid: bool
    errors: tuple[str, ...]
    allowed_doc_ids: frozenset[int]


def evidence_doc_ids(steps) -> set[int]:
    doc_ids: set[int] = set()
    for step in steps:
        if step.action.tool not in EVIDENCE_TOOLS:
            continue
        try:
            payload = json.loads(step.observation)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not payload.get("excerpts"):
            continue
        citation = str(payload.get("citation", ""))
        doc_ids.update(int(value) for value in _CITATION.findall(citation))
    return doc_ids


def validate_report(report: str, steps) -> ReportValidation:
    errors: list[str] = []
    allowed = evidence_doc_ids(steps)
    cited = {int(value) for value in _CITATION.findall(report)}
    unsupported = cited - allowed

    if _LEGACY_CITATION.search(report):
        errors.append("Citations must use [D<number>], not [number] or [1, 2].")
    if allowed and not cited:
        errors.append("The report contains no valid evidence citations.")
    if unsupported:
        values = ", ".join(f"D{value}" for value in sorted(unsupported))
        errors.append(f"The report cites documents not observed as evidence: {values}.")
    if len(report) > 8000:
        errors.append("The report exceeds the 8000 character safety limit.")

    longest_line = max((len(line) for line in report.splitlines()), default=0)
    if longest_line > 600:
        errors.append("The report contains a pathologically long line.")
    if re.search(r"(.{2,16})\1{5,}", report):
        errors.append("The report contains pathological repeated text.")

    uncited_claims = _uncited_claim_lines(report)
    if uncited_claims:
        if allowed:
            errors.append(
                f"{len(uncited_claims)} factual lines lack an evidence citation; "
                f"first: {uncited_claims[0][:120]}"
            )
        else:
            errors.append(
                "No evidence was observed, but the report contains factual claims; "
                f"first: {uncited_claims[0][:120]}"
            )

    return ReportValidation(not errors, tuple(errors), frozenset(allowed))


def _uncited_claim_lines(report: str) -> list[str]:
    lines = report.splitlines()
    uncited = []
    in_code = False
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line or line.startswith("#") or _CITATION.search(line):
            continue
        if _GAP_LANGUAGE.search(line):
            continue
        if re.match(r"^\|?(?:\s*:?-{3,}:?\s*\|)+$", line):
            continue
        if line.startswith("|") and index + 1 < len(lines):
            if re.match(r"^\|?(?:\s*:?-{3,}:?\s*\|)+$", lines[index + 1].strip()):
                continue

        is_list_or_table = bool(
            re.match(r"^(?:[-*]|\d+\.)\s+", line) or line.startswith("|")
        )
        plain = re.sub(r"[*_`>|]", "", line).strip()
        if is_list_or_table or len(plain) >= 60:
            uncited.append(plain)
    return uncited


def safe_fallback_report(goal: str, validation: ReportValidation, steps) -> str:
    inspected = []
    for step in steps:
        if step.action.tool != "inspect_document":
            continue
        try:
            payload = json.loads(step.observation)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not payload.get("excerpts"):
            continue
        citation = payload.get("citation")
        title = payload.get("title")
        if citation and title and (citation, title) not in inspected:
            inspected.append((citation, title))

    if inspected:
        sources = "\n".join(
            f"- {citation} {title}" for citation, title in inspected
        )
    else:
        sources = "- 没有完成可验证的文档检查。"

    return (
        "# 研究证据不足\n\n"
        f"研究目标：{goal}\n\n"
        "## 已检查来源\n\n"
        f"{sources}\n\n"
        "## 结论\n\n"
        "- 当前知识库证据不足以可靠完成该研究目标。\n"
        "- 模型草稿未通过引用或输出质量校验，系统已拒绝展示未经验证的内容。\n"
        "- 建议补充缺失主题的资料后重新运行研究。"
    )
