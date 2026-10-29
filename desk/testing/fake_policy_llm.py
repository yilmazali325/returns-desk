"""FakePolicyLLM: a deterministic, rule-based stand-in for the local model, for tests and E2E only.

Why it exists: the UI's end-to-end tests must drive the REAL orchestrator, gateway, store and
guards, but must not depend on a 4B model's wording or on a GPU being free. This double reads
exactly what the model reads (the prompt messages), grounds the customer's words with
desk.grounding (the analyzer behind the orchestrator's own gates), and answers with one plausible
tool call per step. Its answers go through `desk.llm.interpret_calls`, the same validation a real
model answer gets, so a wrong call is a protocol error here too.

What it does per role:
  resolution   confirm control used -> handoff execute_confirmed; a grounded line + remedy -> the
               matching propose handoff, citing the customer's words (evidence) when several lines
               could fit; a question about rules or status, or anything unclear -> a short
               question. After the Operations report it replies with a sentence for the outcome.
  operations   follows the brief's intent in the documented tool order, and reports the outcome
               it read from the tool results.

It is never the default: the UI server uses it only with DESK_LLM=fake-policy. It reads no clock
and uses no randomness. Token counts are rough size estimates (characters / 4) so the telemetry
widgets have numbers to show; they are not measurements, decode time is 0 (so no tokens/s is
shown) and the engine is labelled as a fake.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from desk.agents.toolspecs import (
    CHECK_STATUS,
    CREATE_PROPOSAL,
    EXECUTE_CONFIRMED,
    EXECUTE_PROPOSAL,
    FIND_ORDER_LINES,
    GET_REQUEST_STATUS,
    HANDOFF,
    LIST_EXCHANGE_OPTIONS,
    PROPOSE_EXCHANGE,
    PROPOSE_REFUND,
    REPLY,
    REPORT,
)
from desk.agents.prompts import (
    ACCOUNT_CARD_HEADER,
    BRIEF_HEADER,
    CONFIRMATION_PREFIX,
    CONVERSATION_HEADER,
    DESK_NOTE_PREFIX,
    DESK_PREFIX,
    INTENT_PREFIX,
)
from desk.domain import EXCHANGE, FAILURE, GATE, ORDER_NOT_FOUND, REFUND, REFUSAL, TIMEOUT, code_class
from desk.grounding import Analysis, analyze, check_evidence
from desk.llm import RawCall, Step, ToolSpec, interpret_calls
from desk.orchestrator import ALREADY_COMPLETED_NOTE, CANCEL_NOTE, REPLAY_NOTE
from desk.sanitize import UNTRUSTED_LABEL

ENGINE_LABEL = "fake-policy (deterministic test double, no model)"
CHARS_PER_TOKEN = 4

# Replies are written to pass the reply guard: no amounts, ids, sizes or SKUs (the desk appends
# the exact facts), no success words unless the outcome is completed, and questions end with "?".
REPLIES = {
    "completed": "Thank you for waiting, your request is all set.",
    "replay": "This request was already taken care of earlier, so nothing new was charged or moved.",
    "proposed": "Here is what I can offer. Please review the details and use the confirm button if you want to go ahead.",
    "denied": "I am sorry, but this request does not meet our return rules.",
    "needs_info": "Which size would you like instead?",
    "failed": "Something went wrong on our side, so I am checking the records before anything else happens.",
    "cancel": "No problem, nothing will go ahead. Is there anything else I can help with?",
    "already_completed": "This request was already used for another item, so I cannot start a new one under it.",
    "not_found": "I could not find that order on your account. Could you check the order number?",
    "which_remedy": "Would you like a refund, or an exchange for a different size?",
    "which_item": "Which item from your order do you mean?",
    "inquiry": "Happy to help. Which item would you like to return, and would you prefer a refund or an exchange?",
}
SUMMARIES = {
    "proposed": "A proposal is ready for the customer to confirm.",
    "completed": "The confirmed request is complete.",
    "denied": "The rules or the current stock refused the request.",
    "needs_info": "The customer still has to choose something.",
    "failed": "A technical problem stopped the request.",
}

_CARD_LINE_RE = re.compile(
    r"^- order (?P<order>.+?), line (?P<line>.+?): (?P<sku>\S+) = product (?P<pid>[^\s,]+)"
    r"(?: \((?P<aliases>[^)]*)\))?(?:, size (?P<size>[^\s,]+)(?: \([^)]*\))?)?(?:, paid \$(?P<paid>[\d,]+\.\d\d))?"
    r"(?:, delivered (?P<delivered>\S+))?")
_HISTORY_RE = re.compile(r'^\d+\. customer: (".*") -> desk decision: \w+$')
_DESK_CALLED_RE = re.compile(r"^" + re.escape(DESK_PREFIX) + r"the desk called (\w+) for you: (\{.*\})$", re.DOTALL)
_TODAY_RE = re.compile(r"today is (\d{4}-\d{2}-\d{2})")
MAX_EVIDENCE_WORDS = 6


# ------------------------------------------------------------------------------------------------
# Reading the prompt (the same text the real model gets)


@dataclass
class ResolutionView:
    """What the Resolution prompt says this step."""

    customer_messages: list[str] = field(default_factory=list)
    lines: list[dict[str, Any]] | None = None  # None: the account card is not loaded yet
    as_of: str | None = None  # "today" from the account card
    confirmed: bool = False  # the explicit confirm control was used and the grant stands
    desk_note: str | None = None
    handoffs: int = 0
    report: dict[str, Any] | None = None  # the Operations report, once one arrived
    last_tool: str | None = None
    last_result: dict[str, Any] = field(default_factory=dict)


def read_resolution(messages: Sequence[Mapping[str, Any]]) -> ResolutionView:
    view = ResolutionView()
    for message in messages:
        content = str(message.get("content") or "")
        role = message.get("role")
        if role == "user" and content.startswith(ACCOUNT_CARD_HEADER) and view.lines is None:
            view.lines = card_lines(content)
            today = _TODAY_RE.search(content)
            view.as_of = today.group(1) if today else None
        elif role == "user" and CONVERSATION_HEADER in content:
            _read_turn_block(view, content)
        elif role == "assistant" and message.get("tool_name") == HANDOFF:
            view.handoffs += 1
        elif role == "tool":
            _read_resolution_result(view, str(message.get("tool_name")), _json(content))
    return view


def _read_turn_block(view: ResolutionView, block: str) -> None:
    for row in block.splitlines():
        match = _HISTORY_RE.match(row)
        if match:
            view.customer_messages.append(json.loads(match.group(1)))
        elif row.startswith(CONFIRMATION_PREFIX):
            view.confirmed = "explicit confirm control for" in row
        elif row.startswith(DESK_NOTE_PREFIX):
            view.desk_note = row.removeprefix(DESK_NOTE_PREFIX)
    view.customer_messages.append(current_message(block))


def _read_resolution_result(view: ResolutionView, tool: str, result: dict[str, Any]) -> None:
    view.last_tool, view.last_result = tool, result
    if tool == FIND_ORDER_LINES:
        answer = result.get("result") if result.get("already_done") else result  # a repeat nests the cached answer
        view.lines = list(((answer or {}).get("data") or {}).get("matches") or [])
    elif tool == HANDOFF and "outcome" in result:
        view.report = result


def current_message(block: str) -> str:
    """The customer's message from its untrusted block (sanitize turned any quotes inside it to ''')."""
    opening = f'{UNTRUSTED_LABEL}:\n"""\n'
    start = block.find(opening)
    if start < 0:
        return ""
    return block[start + len(opening):].split('\n"""', 1)[0]


def card_lines(card: str) -> list[dict[str, Any]]:
    """Line summaries back from the account card text (desk.agents.prompts.line_label)."""
    lines = []
    for row in card.splitlines():
        match = _CARD_LINE_RE.match(row)
        if match is None:
            continue
        aliases = [a.strip() for a in (match.group("aliases") or "").split(",") if a.strip()]
        paid = match.group("paid")
        lines.append({"order_id": match.group("order"), "line_id": match.group("line"), "sku": match.group("sku"),
                      "product_id": match.group("pid"), "size": match.group("size"), "aliases": aliases,
                      "paid_cents": int(paid.replace(",", "").replace(".", "")) if paid else None,
                      "delivered_on": match.group("delivered")})
    return lines


def _json(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


@dataclass
class OperationsView:
    """What the Operations prompt says this step: the brief's intent and every tool result so far."""

    intent: str = ""
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def called(self, tool: str) -> bool:
        return any(name == tool for name, _ in self.events)

    def last(self, tool: str) -> dict[str, Any]:
        return next((payload for name, payload in reversed(self.events) if name == tool), {})


def read_operations(messages: Sequence[Mapping[str, Any]]) -> OperationsView:
    view = OperationsView()
    for message in messages:
        content = str(message.get("content") or "")
        if message.get("role") == "user" and content.startswith(BRIEF_HEADER):
            view.intent = next((r.removeprefix(INTENT_PREFIX) for r in content.splitlines()
                                if r.startswith(INTENT_PREFIX)), "")
        elif message.get("role") == "user" and (called := _DESK_CALLED_RE.match(content)):
            view.events.append((called.group(1), _json(called.group(2))))
        elif message.get("role") == "tool":
            payload = _json(content)
            if payload.get("already_done"):
                continue  # a cached repeat adds no new fact
            view.events.append((str(message.get("tool_name")), payload))
    return view


# ------------------------------------------------------------------------------------------------
# The policies


def grounded_handoff(analysis: Analysis, lines: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The propose handoff the customer's words justify, or None when a question is the answer."""
    if analysis.unique_target is None or len(analysis.requested_sizes) > 1 or analysis.inquiry:
        return None
    order_id, line_id = analysis.unique_target
    evidence = cited_words(analysis, (order_id, line_id))
    cite = {"evidence": evidence} if evidence else {}
    if analysis.remedy_cue == REFUND:
        return {"intent": PROPOSE_REFUND, "order_id": order_id, "line_id": line_id, **cite}
    if analysis.remedy_cue != EXCHANGE:
        return None
    own = next((x.get("product_id") for x in lines if (x.get("order_id"), x.get("line_id")) == (order_id, line_id)), None)
    words = analysis.requested_product_words
    wanted = analysis.requested_product or (words[0] if len(words) == 1 else None)
    return {"intent": PROPOSE_EXCHANGE, "order_id": order_id, "line_id": line_id,
            "requested_size": analysis.requested_size or "UNSPECIFIED",
            "requested_product": wanted if wanted and wanted != own else "SAME", **cite}


def cited_words(analysis: Analysis, key: tuple[str, str]) -> str | None:
    """The shortest run of the customer's words that the desk accepts as picking this line, or
    None when the typed ids (or the account) already leave one line. Like a careful agent, the
    double quotes the customer instead of paraphrasing."""
    if len(analysis.id_candidates) <= 1:
        return None
    for size in range(1, MAX_EVIDENCE_WORDS + 1):
        for text in reversed(analysis.texts):
            words = text.split()
            for start in range(len(words) - size + 1):
                phrase = " ".join(words[start: start + size])
                if check_evidence(analysis, key, phrase) is None:
                    return phrase
    return None


def resolution_call(view: ResolutionView, allowed: Sequence[str]) -> tuple[str, dict[str, Any]]:
    """One Resolution step: (tool, args)."""
    if view.report is not None or allowed == [REPLY]:
        return REPLY, {"message": reply_text(view)}
    error = view.last_result.get("error") if isinstance(view.last_result.get("error"), dict) else {}
    if error.get("code") == "HANDOFF_SUGGESTED" and isinstance(error.get("handoff"), dict):
        return HANDOFF, dict(error["handoff"])  # the desk showed what the conversation names; we agree
    if (view.confirmed and view.handoffs == 0) or error.get("code") == "CONFIRMATION_PENDING":
        return HANDOFF, {"intent": EXECUTE_CONFIRMED}
    if view.last_tool == HANDOFF:  # our handoff was rejected by a gate: ask instead of guessing
        return REPLY, {"message": reply_text(view)}
    if view.lines is None and FIND_ORDER_LINES in allowed:
        return FIND_ORDER_LINES, {"order_ref": "ALL"}
    handoff = grounded_handoff(analyze(view.customer_messages, view.lines or [], [], view.as_of), view.lines or [])
    if handoff is not None and HANDOFF in allowed and view.handoffs == 0:
        return HANDOFF, handoff
    return REPLY, {"message": reply_text(view)}


def reply_text(view: ResolutionView) -> str:
    """A guard-safe sentence for the report's outcome, the desk note, or the open question."""
    if view.report is not None:
        outcome = str(view.report.get("outcome"))
        if outcome == "completed" and view.report.get("already_processed_earlier"):
            return REPLIES["replay"]
        return REPLIES.get(outcome, REPLIES["failed"])
    notes = {REPLAY_NOTE: "replay", ALREADY_COMPLETED_NOTE: "already_completed", CANCEL_NOTE: "cancel"}
    if view.desk_note in notes:
        return REPLIES[notes[view.desk_note]]
    analysis = analyze(view.customer_messages, view.lines or [], [])
    if analysis.inquiry:
        return REPLIES["inquiry"]
    if analysis.not_found_code == ORDER_NOT_FOUND and analysis.unique_target is None:
        return REPLIES["not_found"]
    if analysis.unique_target is None:
        return REPLIES["which_item"]
    if analysis.remedy_cue not in (REFUND, EXCHANGE):
        return REPLIES["which_remedy"]
    return REPLIES["needs_info"]


def operations_call(view: OperationsView, allowed: Sequence[str], free_text: bool) -> tuple[str, dict[str, Any]]:
    """One Operations step: the next tool of the intent's documented order, or the report."""
    tool = next_operations_tool(view)
    if tool not in allowed:
        tool = REPORT if REPORT in allowed else allowed[0]
    if tool == CREATE_PROPOSAL:
        remedy = EXCHANGE if view.intent == PROPOSE_EXCHANGE else REFUND
        args: dict[str, Any] = {"remedy": remedy}
        requested = (view.last(LIST_EXCHANGE_OPTIONS).get("data") or {}).get("requested_sku")
        if remedy == EXCHANGE and requested:
            args["replacement_sku"] = requested
        return tool, args
    if tool == REPORT:
        outcome = observed_outcome(view)
        return tool, {"outcome": outcome, **({"summary": SUMMARIES[outcome]} if free_text else {})}
    return tool, {}


def next_operations_tool(view: OperationsView) -> str:
    if timeout_pending(view.events):
        return GET_REQUEST_STATUS
    if view.intent == EXECUTE_CONFIRMED:
        return REPORT if view.called(EXECUTE_PROPOSAL) else EXECUTE_PROPOSAL
    if view.intent == CHECK_STATUS:
        return REPORT if view.called(GET_REQUEST_STATUS) else GET_REQUEST_STATUS
    if view.called(CREATE_PROPOSAL):
        return REPORT
    if view.intent == PROPOSE_EXCHANGE and not view.called(LIST_EXCHANGE_OPTIONS):
        return LIST_EXCHANGE_OPTIONS
    return CREATE_PROPOSAL


def timeout_pending(events: Sequence[tuple[str, dict[str, Any]]]) -> bool:
    pending = False
    for tool, payload in events:
        if tool == EXECUTE_PROPOSAL and _code(payload) == TIMEOUT:
            pending = True
        elif tool == GET_REQUEST_STATUS:
            pending = False
    return pending


def observed_outcome(view: OperationsView) -> str:
    """The outcome the tool results show (the desk compares it with its own code-built report)."""
    outcome = "needs_info"
    for tool, payload in view.events:
        data, code = payload.get("data") or {}, _code(payload)
        if tool == GET_REQUEST_STATUS and data:
            outcome = "completed" if data.get("status") == "completed" else outcome
        elif tool in (CREATE_PROPOSAL, EXECUTE_PROPOSAL) and payload.get("ok"):
            outcome = "completed" if "transaction" in data else "proposed"
        elif code == TIMEOUT:
            outcome = "failed"
        elif code is not None and tool != LIST_EXCHANGE_OPTIONS:
            outcome = {REFUSAL: "denied", GATE: "needs_info", FAILURE: "failed"}.get(code_class(code), outcome)
    return outcome


def _code(payload: Mapping[str, Any]) -> str | None:
    error = payload.get("error")
    return str(error.get("code")) if isinstance(error, dict) and error.get("code") else None


# ------------------------------------------------------------------------------------------------
# The LLM interface (same `step()` signature as desk.llm.LLMClient)


class FakePolicyLLM:
    """Deterministic stand-in for LLMClient. `step_delay_s` slows each step down so a browser can
    watch the live trace (telemetry only; it never changes an answer)."""

    protocol = "fake-policy"
    engine = ENGINE_LABEL

    def __init__(self, step_delay_s: float = 0.0) -> None:
        self.step_delay_s = step_delay_s
        self.calls = 0

    def step(self, role: str, messages: Sequence[Mapping[str, Any]], tools: Sequence[ToolSpec],
             allowed: Sequence[str], max_tokens: int) -> Step:
        allowed = list(allowed)
        if role == "operations":
            free_text = any(t.name == REPORT and "summary" in t.parameters.get("properties", {}) for t in tools)
            name, args = operations_call(read_operations(messages), allowed, free_text)
        else:
            name, args = resolution_call(read_resolution(messages), allowed)
        if self.step_delay_s > 0:
            time.sleep(self.step_delay_s)
        self.calls += 1
        step = interpret_calls([RawCall(name, args)], tools, allowed)
        step.role = role
        step.text = json.dumps({"tool": name, "args": args}, sort_keys=True)
        step.prompt_tokens = sum(len(str(m.get("content") or "")) for m in messages) // CHARS_PER_TOKEN
        step.completion_tokens = max(1, len(step.text) // CHARS_PER_TOKEN)
        step.latency_ms = round(self.step_delay_s * 1000, 1)  # no decoding happens, so predicted_ms stays 0
        step.finish_reason = "stop"
        step.http_attempts = 1
        return step
