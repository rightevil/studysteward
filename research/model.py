import json

from research.agent import ResearchAction, ResearchStep
from research.validation import evidence_doc_ids


ACTION_PROMPT = """You are an evidence-driven research agent.
Goal: {goal}

Available tools:
{tools}

Previous steps:
{history}

Choose exactly one next action. Use observations to revise your next query.
Never repeat the same tool call. Call list_documents at most once.
If searches do not reveal evidence for a requested subject, finish and report the gap.
search_kb only discovers candidates. After every successful search_kb call, you
MUST inspect one returned document before searching again.
Only inspect_document observations are citable evidence.
Do not invent evidence. Every factual claim in the final report must cite [D<number>].
The final report must be valid Markdown:
- Put every heading on its own line.
- Leave a blank line before and after headings and tables.
- Put every list item on its own line.
- Never concatenate a heading, list item, or table row with adjacent text.
Return ONLY JSON:
{{"thought":"short reason","tool":"tool_name","args":{{...}}}}

Use tool "finish" with args {{"report":"..."}} only when evidence is sufficient.
Step {step} of {max_steps}.
"""


FINAL_PROMPT = """Write the best possible research report for this goal using only
the observations below. Mark unsupported gaps explicitly. Cite sources as [D<number>].
Allowed evidence citations: {allowed_citations}
Never use your own knowledge to fill missing evidence.
Never mention a technique, command, prerequisite, or comparison unless it appears
in the observations. If one side of a requested comparison lacks evidence, say that
the comparison cannot be completed reliably and describe the missing material.
Inventory and search results are discovery data, not factual evidence. Only claims
present in inspect_document observations may appear as document-backed facts.
Return valid Markdown with headings, list items, and table rows on separate lines.
Leave blank lines around headings and tables.

Goal: {goal}
Observations:
{history}
"""

REVISE_PROMPT = """Rewrite the draft so it passes every validation error.
Use only the observations and allowed [D<number>] citations. Remove unsupported
claims instead of replacing them with model knowledge. If evidence is missing,
state the gap plainly. Return only valid Markdown.

Goal: {goal}
Allowed citations: {allowed_citations}
Validation errors:
{errors}

Observations:
{history}

Draft:
{draft}
"""


class ProviderResearchModel:
    def __init__(self, provider):
        self.provider = provider

    @staticmethod
    def _history(steps: list[ResearchStep]) -> str:
        if not steps:
            return "(none)"
        rows = []
        seen_observations = set()
        for step in steps:
            observation = step.observation[:3000]
            if observation in seen_observations:
                observation = "(duplicate observation omitted)"
            else:
                seen_observations.add(observation)
            rows.append(
                f"{step.number}. {step.action.tool} "
                f"{json.dumps(step.action.args, ensure_ascii=False)}\n"
                f"Observation: {observation}"
            )
        return "\n\n".join(rows)[-12000:]

    def next_action(self, goal, steps, tools, step, max_steps) -> ResearchAction:
        prompt = ACTION_PROMPT.format(
            goal=goal,
            tools=tools,
            history=self._history(steps),
            step=step,
            max_steps=max_steps,
        )
        try:
            payload = self.provider.complete_json(
                prompt,
                required_keys=("tool", "args"),
            )
        except ValueError:
            payload = self.provider.complete_json(
                prompt
                + "\nYour previous response was invalid. Return exactly one JSON "
                "object with thought, tool, and args. Do not add prose or a second object.",
                required_keys=("tool", "args"),
            )
        return ResearchAction(
            thought=str(payload.get("thought", "")).strip() or "Continue research",
            tool=str(payload.get("tool", "")).strip(),
            args=payload.get("args") if isinstance(payload.get("args"), dict) else {},
        )

    def finalize(self, goal: str, steps: list[ResearchStep]) -> str:
        allowed = " ".join(
            f"[D{doc_id}]" for doc_id in sorted(evidence_doc_ids(steps))
        ) or "(none)"
        return self.provider.complete(
            FINAL_PROMPT.format(
                goal=goal,
                history=self._history(steps),
                allowed_citations=allowed,
            ),
            max_tokens=2048,
        )

    def revise_report(
        self,
        goal: str,
        steps: list[ResearchStep],
        draft: str,
        errors: tuple[str, ...],
        allowed_doc_ids: frozenset[int],
    ) -> str:
        allowed = " ".join(
            f"[D{doc_id}]" for doc_id in sorted(allowed_doc_ids)
        ) or "(none)"
        return self.provider.complete(
            REVISE_PROMPT.format(
                goal=goal,
                allowed_citations=allowed,
                errors="\n".join(f"- {error}" for error in errors),
                history=self._history(steps),
                draft=draft[:8000],
            ),
            max_tokens=2048,
        )
