"""Resolution agent: the customer-facing loop for one turn.

The loop itself is deliberately dumb. Each step it asks the orchestrator (through `ResolutionHooks`)
which tools are allowed, runs one LLM step, and hands the call back to the orchestrator to decide
what it means this turn: a lookup, a handoff to validate and run, or a reply to accept or send
back once with feedback. The agent decides; the orchestrator can only block with a reason. That
split keeps every business rule (grounding checks, the one-time clarify veto, confirmation routing)
in orchestrator.py, where it is tested without a model, while this file only owns the mechanics:
append-only history, protocol errors fed back as tool results, the step cap, and a final step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from desk.agents import LIMITS, Limits, StepRunner
from desk.agents.prompts import desk_note_message
from desk.agents.toolspecs import FIND_ORDER_LINES, REPLY, RESOLUTION_TOOLS
from desk.llm import assistant_message, tool_message

MAX_RESOLUTION_STEPS = LIMITS.resolution_steps  # read by the adapter configuration and the bench report


@dataclass(frozen=True)
class ReplyVerdict:
    """The orchestrator's answer to a reply: accepted, or rejected with feedback for the agent."""

    accepted: bool
    feedback: dict[str, Any] | None = None


class ResolutionHooks(Protocol):
    """What the loop asks the orchestrator. Implemented per turn in orchestrator.py."""

    def allowed(self, final: bool) -> list[str]:
        """The tools the agent may call at this step (`final`: the last step of the turn)."""

    def before_step(self, step_no: int, final: bool) -> str | None:
        """Called before every step; may return a note for the agent (it never acts for the agent)."""

    def lookup(self, args: dict[str, Any]) -> dict[str, Any]: ...

    def handoff(self, args: dict[str, Any]) -> dict[str, Any]: ...

    def reply(self, message: str) -> ReplyVerdict: ...


def run_resolution(runner: StepRunner, messages: list[dict[str, Any]], hooks: ResolutionHooks,
                   limits: Limits = LIMITS) -> str | None:
    """Run Resolution steps until a reply is accepted. Returns the reply text, or None when the
    step or call budget ran out (the orchestrator then answers with its template)."""
    for step_no in range(1, limits.resolution_steps + 1):
        final = step_no == limits.resolution_steps or runner.remaining <= 1
        note = hooks.before_step(step_no, final)
        if note:
            messages.append(desk_note_message(note))
        allowed = hooks.allowed(final)
        max_tokens = limits.reply_step_tokens if REPLY in allowed else limits.tool_step_tokens
        step = runner.step("resolution", messages, RESOLUTION_TOOLS, allowed, max_tokens)
        if step is None:
            return None
        messages.append(assistant_message(step))
        if not step.ok:
            messages.append(tool_message(step.tool or "error", {"error": step.protocol_error, "allowed_now": allowed}))
            continue
        if step.tool == REPLY:
            verdict = hooks.reply(step.args["message"])
            if verdict.accepted:
                return step.args["message"]
            messages.append(tool_message(REPLY, verdict.feedback or {}))
            continue
        result = hooks.lookup(step.args) if step.tool == FIND_ORDER_LINES else hooks.handoff(step.args)
        messages.append(tool_message(step.tool, result))
    return None
