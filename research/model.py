import json

from research.agent import ResearchAction, ResearchStep


ACTION_PROMPT = """You are an evidence-driven research agent.
Goal: {goal}

Available tools:
{tools}

Previous steps:
{history}

Choose exactly one next action. Use observations to revise your next query.
Do not invent evidence. Every factual claim in the final report must cite [D<number>].
Return ONLY JSON:
{{"thought":"short reason","tool":"tool_name","args":{{...}}}}

Use tool "finish" with args {{"report":"..."}} only when evidence is sufficient.
Step {step} of {max_steps}.
"""


FINAL_PROMPT = """Write the best possible research report for this goal using only
the observations below. Mark unsupported gaps explicitly. Cite sources as [D<number>].

Goal: {goal}
Observations:
{history}
"""


class ProviderResearchModel:
    def __init__(self, provider):
        self.provider = provider

    @staticmethod
    def _history(steps: list[ResearchStep]) -> str:
        if not steps:
            return "(none)"
        rows = []
        for step in steps:
            observation = step.observation[:4000]
            rows.append(
                f"{step.number}. {step.action.tool} "
                f"{json.dumps(step.action.args, ensure_ascii=False)}\n"
                f"Observation: {observation}"
            )
        return "\n\n".join(rows)[-12000:]

    def next_action(self, goal, steps, tools, step, max_steps) -> ResearchAction:
        payload = self.provider.complete_json(
            ACTION_PROMPT.format(
                goal=goal,
                tools=tools,
                history=self._history(steps),
                step=step,
                max_steps=max_steps,
            )
        )
        return ResearchAction(
            thought=str(payload.get("thought", "")).strip() or "Continue research",
            tool=str(payload.get("tool", "")).strip(),
            args=payload.get("args") if isinstance(payload.get("args"), dict) else {},
        )

    def finalize(self, goal: str, steps: list[ResearchStep]) -> str:
        return self.provider.complete(
            FINAL_PROMPT.format(goal=goal, history=self._history(steps)),
            max_tokens=2048,
        )
