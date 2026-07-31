import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from research.formatting import normalize_markdown
from research.validation import (
    safe_fallback_report,
    validate_report,
)


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

    _TOOL_BUDGETS = {
        "list_documents": 1,
        "search_kb": 3,
        "inspect_document": 4,
    }

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
        seen_actions: set[str] = set()
        tool_usage: Counter[str] = Counter()
        seen_candidate_ids: set[int] = set()
        pending_candidate_ids: set[int] = set()
        no_novelty_count = 0
        planning_attempts = 0
        consecutive_rejections = 0
        last_policy_feedback = ""

        try:
            while (
                len(steps) < self.max_steps
                and planning_attempts < self.max_steps * 2
            ):
                planning_attempts += 1
                number = len(steps) + 1
                policy_context = self._policy_context(
                    tool_usage,
                    pending_candidate_ids,
                    last_policy_feedback,
                    len(steps),
                )
                action = self.model.next_action(
                    goal,
                    steps,
                    f"{self.tools.descriptions()}\n\n{policy_context}",
                    number,
                    self.max_steps,
                )
                policy_rejection = self._policy_rejection_reason(
                    action,
                    seen_actions,
                    tool_usage,
                    pending_candidate_ids,
                )
                if policy_rejection:
                    consecutive_rejections += 1
                    last_policy_feedback = policy_rejection
                    emit(f"Policy - {policy_rejection}; replanning.")
                    if consecutive_rejections >= 2:
                        status = "evidence_exhausted"
                        emit("Policy - repeated invalid plans; stopping research.")
                        break
                    continue

                consecutive_rejections = 0
                last_policy_feedback = ""
                emit(self._event_label(number, action))
                started = time.perf_counter()

                if action.tool == "finish":
                    report = normalize_markdown(action.args.get("report", ""))
                    if not report:
                        observation = "finish requires a non-empty report"
                    else:
                        observation = "Research report completed."
                else:
                    observation = self.tools.execute(action.tool, action.args)
                    seen_actions.add(self._action_signature(action))
                    tool_usage[action.tool] += 1

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

                if action.tool == "search_kb":
                    candidates = self._candidate_doc_ids(observation)
                    pending_candidate_ids = candidates
                    if candidates - seen_candidate_ids:
                        no_novelty_count = 0
                    else:
                        no_novelty_count += 1
                    seen_candidate_ids.update(candidates)
                    if no_novelty_count >= 2:
                        status = "evidence_exhausted"
                        emit("Evidence - two searches produced no new candidates.")
                        break
                elif action.tool == "inspect_document":
                    pending_candidate_ids.clear()

            if len(steps) >= self.max_steps and not report:
                status = "step_limit"
            elif planning_attempts >= self.max_steps * 2 and not report:
                status = "evidence_exhausted"

            if not report:
                emit("Synthesis - generating a report from inspected evidence.")
                report = normalize_markdown(self.model.finalize(goal, steps))
            validation = validate_report(report, steps)
            if not validation.valid and hasattr(self.model, "revise_report"):
                emit("Validation - draft rejected; attempting one constrained revision.")
                report = normalize_markdown(
                    self.model.revise_report(
                        goal,
                        steps,
                        report,
                        validation.errors,
                        validation.allowed_doc_ids,
                    )
                )
                validation = validate_report(report, steps)
            if not validation.valid:
                status = "validation_failed"
                emit("Validation - revision rejected; returning an evidence-gap report.")
                report = safe_fallback_report(goal, validation, steps)
            if run_id is not None:
                self.trace_store.finish_research_run(run_id, status, report)
            return ResearchResult(goal, report, status, run_id, tuple(steps))
        except Exception as exc:
            if run_id is not None:
                self.trace_store.finish_research_run(run_id, "failed", str(exc))
            raise

    @staticmethod
    def _action_signature(action: ResearchAction) -> str:
        args = json.dumps(action.args, ensure_ascii=False, sort_keys=True)
        return f"{action.tool}:{args}"

    def _policy_rejection_reason(
        self,
        action: ResearchAction,
        seen_actions: set[str],
        tool_usage: Counter[str],
        pending_candidate_ids: set[int],
    ) -> str | None:
        if action.tool == "finish":
            return None
        if action.tool not in self._TOOL_BUDGETS:
            return f"unknown tool {action.tool}"
        if pending_candidate_ids:
            candidates = ", ".join(
                f"D{doc_id}" for doc_id in sorted(pending_candidate_ids)
            )
            if action.tool != "inspect_document":
                return f"inspect one of {candidates} before another discovery action"
            try:
                requested_doc_id = int(action.args.get("doc_id", 0))
            except (TypeError, ValueError):
                requested_doc_id = 0
            if requested_doc_id not in pending_candidate_ids:
                return f"inspect_document must select one of {candidates}"
        signature = self._action_signature(action)
        if signature in seen_actions:
            return f"repeated action {action.tool} with identical arguments"
        budget = self._TOOL_BUDGETS.get(action.tool)
        if budget is not None and tool_usage[action.tool] >= budget:
            return f"{action.tool} reached its budget of {budget}"
        return None

    def _policy_context(
        self,
        tool_usage: Counter[str],
        pending_candidate_ids: set[int],
        last_policy_feedback: str,
        executed_steps: int,
    ) -> str:
        remaining = {
            tool: max(0, budget - tool_usage[tool])
            for tool, budget in self._TOOL_BUDGETS.items()
        }
        rows = [
            "Runtime policy state:",
            f"- Executed steps: {executed_steps}/{self.max_steps}",
            f"- list_documents remaining: {remaining['list_documents']}",
            f"- search_kb remaining: {remaining['search_kb']}",
            f"- inspect_document remaining: {remaining['inspect_document']}",
            "- search_kb only discovers candidates; only inspect_document creates "
            "citable evidence.",
        ]
        if pending_candidate_ids:
            candidates = ", ".join(
                f"D{doc_id}" for doc_id in sorted(pending_candidate_ids)
            )
            rows.append(
                f"- You must inspect one of {candidates} or finish before searching again."
            )
        if last_policy_feedback:
            rows.append(f"- Previous plan was rejected: {last_policy_feedback}")
        return "\n".join(rows)

    @staticmethod
    def _candidate_doc_ids(observation: str) -> set[int]:
        try:
            payload = json.loads(observation)
        except (TypeError, json.JSONDecodeError):
            return set()
        return {
            int(match)
            for item in payload.get("results", [])
            for match in re.findall(r"\[D(\d+)\]", str(item.get("citation", "")))
        }

    def _event_label(self, number: int, action: ResearchAction) -> str:
        detail = ""
        if action.tool == "search_kb":
            detail = str(action.args.get("query", "")).strip()
        elif action.tool == "inspect_document":
            detail = f"D{action.args.get('doc_id', '?')}"
        if detail:
            detail = re.sub(r"\s+", " ", detail)
            detail = f" - {detail[:72]}"
        return f"Step {number}/{self.max_steps} - {action.tool}{detail}"
