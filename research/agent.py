import json
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class ResearchAction:
    thought: str
    tool: str
    args: dict


@dataclass(frozen=True)
class ResearchStep:
    number: int
    action: ResearchAction
    observation: str
    duration_ms: int


@dataclass(frozen=True)
class ResearchResult:
    goal: str
    report: str
    status: str
    run_id: int | None
    steps: tuple[ResearchStep, ...] = field(default_factory=tuple)


class ResearchAgent:
    """Bounded ReAct runtime for evidence-driven research."""

    def __init__(self, model, tools, trace_store=None, max_steps: int = 8):
        self.model = model
        self.tools = tools
        self.trace_store = trace_store
        self.max_steps = max_steps

    def run(
        self,
        goal: str,
        on_event: Callable[[str], None] | None = None,
    ) -> ResearchResult:
        goal = goal.strip()
        if not goal:
            raise ValueError("Research goal cannot be empty")

        emit = on_event or (lambda message: None)
        run_id = self.trace_store.create_research_run(goal) if self.trace_store else None
        steps: list[ResearchStep] = []
        status = "completed"
        report = ""

        try:
            for number in range(1, self.max_steps + 1):
                action = self.model.next_action(
                    goal,
                    steps,
                    self.tools.descriptions(),
                    number,
                    self.max_steps,
                )
                emit(f"Step {number}/{self.max_steps}: {action.thought} [{action.tool}]")
                started = time.perf_counter()

                if action.tool == "finish":
                    report = str(action.args.get("report", "")).strip()
                    if not report:
                        observation = "finish requires a non-empty report"
                    else:
                        observation = "Research report completed."
                else:
                    observation = self.tools.execute(action.tool, action.args)

                duration_ms = int((time.perf_counter() - started) * 1000)
                step = ResearchStep(number, action, observation, duration_ms)
                steps.append(step)
                if run_id is not None:
                    self.trace_store.add_research_step(
                        run_id,
                        number,
                        action.thought,
                        action.tool,
                        json.dumps(action.args, ensure_ascii=False),
                        observation,
                        duration_ms,
                    )
                if action.tool == "finish" and report:
                    break
            else:
                status = "step_limit"

            if not report:
                report = self.model.finalize(goal, steps)
            if run_id is not None:
                self.trace_store.finish_research_run(run_id, status, report)
            return ResearchResult(goal, report, status, run_id, tuple(steps))
        except Exception as exc:
            if run_id is not None:
                self.trace_store.finish_research_run(run_id, "failed", str(exc))
            raise
