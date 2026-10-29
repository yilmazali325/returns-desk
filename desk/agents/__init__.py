"""The agent layer: Resolution (customer-facing) and Operations (back office) loops.

    toolspecs.py   tool schemas per role (byte-stable constants)
    prompts.py     static system prompts and the per-turn message builders
    resolution.py  the Resolution loop; asks the orchestrator what each call means this turn
    operations.py  the Operations loop for one handoff, plus the code-built report

This module holds what both loops share: `Limits`, the per-turn budgets in one place (SPEC §6.7),
and `StepRunner`. Every LLM call of a turn goes through the runner, so the call cap, the telemetry
sums and the audit entry for each step are handled in exactly one place. Audit entries for LLM
steps carry the parsed tool call, token counts and timings, never raw model text or reasoning.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from typing import Any

from desk.audit import AuditLog
from desk.llm import LLMError, Step, ToolSpec



@dataclass(frozen=True)
class Limits:
    """Every per-turn budget of the agent loops. Change a limit here and nowhere else."""

    llm_calls_per_turn: int = 12  # all LLM steps of one turn, both agents together
    resolution_steps: int = 5  # Resolution steps per turn
    ops_steps: int = 4  # Operations steps per handoff
    handoffs_per_turn: int = 2  # a handoff that reached no decision may be retried once
    gate_rejections: int = 2  # rejected handoffs before only a question is allowed
    tool_step_tokens: int = 160  # max_tokens for a step that may call a tool
    reply_step_tokens: int = 220  # max_tokens for a step that may reply to the customer


LIMITS = Limits()

# Names other modules (adapter configuration, bench report) read; derived from LIMITS.
MAX_LLM_CALLS_PER_TURN = LIMITS.llm_calls_per_turn
TOOL_STEP_TOKENS = LIMITS.tool_step_tokens
REPLY_STEP_TOKENS = LIMITS.reply_step_tokens


@dataclass
class Telemetry:
    """Sums over LLM steps and gateway calls (one per turn, merged into one per case)."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    llm_latency_ms: float = 0.0
    tool_ms: float = 0.0
    protocol_errors: int = 0
    dropped_calls: int = 0
    llm_errors: int = 0

    def add(self, other: Telemetry) -> None:
        for item in fields(self):
            setattr(self, item.name, getattr(self, item.name) + getattr(other, item.name))

    @property
    def prefill_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.cached_tokens)


def protocol_code(error: str | None) -> str | None:
    """'INVALID_ARGS: args.x ...' -> 'INVALID_ARGS'."""
    return error.split(":", 1)[0].strip() if error else None


class StepRunner:
    """Runs LLM steps for one turn within the call budget; records telemetry and audit entries."""

    def __init__(self, llm: Any, audit: AuditLog, max_calls: int = LIMITS.llm_calls_per_turn) -> None:
        self.llm = llm
        self.audit = audit
        self.max_calls = max_calls
        self.telemetry = Telemetry()

    @property
    def remaining(self) -> int:
        return self.max_calls - self.telemetry.llm_calls

    def step(self, role: str, messages: Sequence[dict[str, Any]], tools: Sequence[ToolSpec],
             allowed: Sequence[str], max_tokens: int) -> Step | None:
        """One LLM call, or None when the turn's budget is used up. LLMError propagates."""
        if self.remaining <= 0:
            return None
        self.telemetry.llm_calls += 1
        try:
            step = self.llm.step(role, list(messages), list(tools), list(allowed), max_tokens)
        except LLMError as exc:
            self.telemetry.llm_errors += 1
            self.audit.record(kind="llm_step", agent=role, tool="(none)", arguments={},
                              error={"code": "LLM_ERROR", "message": str(exc)[:300]}, allowed=list(allowed))
            raise
        self._account(step)
        self.audit.record(
            kind="llm_step", agent=role, tool=step.tool or "(none)",
            arguments=dict(step.args) if step.tool else {},
            result=step.telemetry(),
            error=({"code": protocol_code(step.protocol_error), "message": step.protocol_error[:300]}
                   if step.protocol_error else None),
            allowed=list(allowed),
        )
        if step.dropped_calls:
            # Native mode may return several calls; only the first is used (SPEC §7).
            self.audit.record(kind="guard", agent="orchestrator", tool="dropped_calls",
                              arguments={"role": role}, result={"dropped": step.dropped_calls})
        return step

    def _account(self, step: Step) -> None:
        t = self.telemetry
        t.prompt_tokens += step.prompt_tokens
        t.cached_tokens += step.cached_tokens
        t.completion_tokens += step.completion_tokens
        t.prefill_ms += step.prompt_ms
        t.decode_ms += step.predicted_ms
        t.llm_latency_ms += step.latency_ms
        t.protocol_errors += 1 if step.protocol_error else 0
        t.dropped_calls += len(step.dropped_calls)

    def timed(self, call: Callable[[], Any]) -> Any:
        """Run a gateway call and add its wall time to tool_ms (telemetry only, never logic)."""
        started = time.perf_counter()
        try:
            return call()
        finally:
            self.telemetry.tool_ms += (time.perf_counter() - started) * 1000
