"""Operations agent: the back-office steps for one handoff, inside bounds enforced by code.

The Operations LLM chooses the order of its tools, maps the requested size to a SKU from the
option table, creates the proposal and, after the customer's explicit confirmation, executes it.
Only this agent starts those business actions. Code keeps the guarantees (SPEC §6.0, §6.5-6.6):

- Per-step `allowed` lists: after a TIMEOUT the only allowed tool is get_request_status, after a
  decisive result only the report, and the last step offers only the terminal tool.
- A repeated (tool, arguments) call gets the cached answer instead of running again.
- When the agent ends without a decisive call, code may only READ: the read-only `check_proposal`
  dry run (a refusal is then reported as a deny; a would-be proposal is NOT created) and, after a
  TIMEOUT the agent left unreconciled, `get_request_status`. Both are audited as `kind:
  "fallback"`. Code never creates or executes a proposal for the agent.
- The report Resolution receives is built by code from the gateway results (`HandoffFacts`); the
  LLM's claimed outcome is only compared with it.

Operations never sees customer text: its context is the static prompt plus a structured brief.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from desk import policy
from desk.agents import LIMITS, Limits, StepRunner
from desk.agents.prompts import desk_note_message, ops_brief
from desk.agents.toolspecs import (
    CHECK_STATUS,
    CREATE_PROPOSAL,
    EXECUTE_CONFIRMED,
    EXECUTE_PROPOSAL,
    GET_REQUEST_STATUS,
    LIST_EXCHANGE_OPTIONS,
    PROPOSE_EXCHANGE,
    PROPOSE_REFUND,
    REPORT,
)
from desk.domain import (
    EXCHANGE,
    FAILURE,
    GATE,
    NO_DECISION,
    NOT_EXECUTED,
    REFUND,
    REFUSAL,
    SIZE_UNSPECIFIED,
    TIMEOUT,
    OpsReport,
    ToolResult,
    code_class,
)
from desk.llm import LLMError, ToolSpec, assistant_message, tool_message
from desk.tools import BusinessTools

MAX_OPS_STEPS = LIMITS.ops_steps  # read by the adapter configuration
CHECK_PROPOSAL = "check_proposal"  # the gateway's read-only dry run of create_proposal (code only)
PROPOSE_INTENTS = (PROPOSE_REFUND, PROPOSE_EXCHANGE)
# Keys removed from everything an agent reads: the gateway scopes identity, the model never needs it.
HIDDEN_KEYS = frozenset({"customer_id", "request_id", "binding_hash"})
SIZE_CODES = frozenset({"OUT_OF_STOCK", "PRICE_MISMATCH", "SIZE_NOT_OFFERED", "SAME_SIZE"})


@dataclass(frozen=True)
class OpsBrief:
    """What Operations is asked to do. `target` is a line summary; `confirmed` a proposal dict."""

    intent: str
    target: dict[str, Any] | None = None
    requested_size: str | None = None
    requested_product: str | None = None
    confirmed: dict[str, Any] | None = None

    @property
    def remedy(self) -> str:
        return EXCHANGE if self.intent == PROPOSE_EXCHANGE else REFUND


@dataclass
class ToolEvent:
    tool: str
    result: ToolResult
    by: str  # "operations" or "orchestrator"


@dataclass
class OpsRun:
    """Everything that happened in one Operations handoff, in order."""

    events: list[ToolEvent] = field(default_factory=list)
    claimed_outcome: str | None = None
    claimed_summary: str | None = None
    steps: int = 0

    def attempted(self, tool: str) -> bool:
        return any(event.tool == tool for event in self.events)

    @property
    def timeout_pending(self) -> bool:
        """A TIMEOUT from execute with no status check after it."""
        pending = False
        for event in self.events:
            if event.tool == EXECUTE_PROPOSAL and event.result.code == TIMEOUT:
                pending = True
            elif event.tool == GET_REQUEST_STATUS:
                pending = False
        return pending

    @property
    def proposal_decided(self) -> bool:
        """create_proposal gave an answer about the customer's request (not an agent slip)."""
        for event in self.events:
            if event.tool != CREATE_PROPOSAL:
                continue
            code = event.result.code
            if event.result.ok or code == SIZE_UNSPECIFIED or code_class(code) in (REFUSAL, FAILURE):
                return True
        return False


def agent_view(value: Any) -> Any:
    """A copy without identity fields, for prompts (customer_id/request_id never reach a model)."""
    if isinstance(value, dict):
        return {k: agent_view(v) for k, v in value.items() if k not in HIDDEN_KEYS}
    if isinstance(value, list):
        return [agent_view(item) for item in value]
    return value


class OperationsAgent:
    """Runs the Operations loop for one handoff."""

    def __init__(self, runner: StepRunner, tools: BusinessTools, specs: tuple[ToolSpec, ...],
                 system_prompt: str, limits: Limits = LIMITS) -> None:
        self.runner = runner
        self.tools = tools
        self.specs = specs
        self.system_prompt = system_prompt
        self.limits = limits
        self.max_steps = limits.ops_steps

    def run(self, brief: OpsBrief, run: OpsRun | None = None) -> OpsRun:
        """Run the loop, recording into `run` (pass your own to keep the events if an LLMError
        escapes). After an LLMError only a read-only reconcile is made: no new mutation."""
        run = run if run is not None else OpsRun()
        first_allowed = allowed_tools(brief, run, last_step=self.max_steps == 1)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": ops_brief(brief.intent, brief.target, brief.requested_size,
                                                  brief.requested_product, agent_view(brief.confirmed),
                                                  first_allowed)},
        ]
        try:
            self._loop(brief, run, messages)
        except LLMError:
            if run.timeout_pending:
                self._fallback(run, GET_REQUEST_STATUS, {}, "TIMEOUT not reconciled (LLM unavailable)", messages)
            raise
        self._read_after_the_loop(brief, run, messages)
        return run

    def _loop(self, brief: OpsBrief, run: OpsRun, messages: list[dict[str, Any]]) -> None:
        cache: dict[str, Any] = {}
        for step_no in range(1, self.max_steps + 1):
            allowed = allowed_tools(brief, run, last_step=step_no == self.max_steps)
            step = self.runner.step("operations", messages, self.specs, allowed, self.limits.tool_step_tokens)
            if step is None:
                break
            run.steps += 1
            messages.append(assistant_message(step))
            if not step.ok:
                messages.append(tool_message(step.tool or "error", {"error": step.protocol_error, "allowed_now": allowed}))
                if run.timeout_pending:  # SPEC §6.5: after TIMEOUT the next tool must be the status check
                    self._fallback(run, GET_REQUEST_STATUS, {}, "TIMEOUT not reconciled by the agent", messages)
                continue
            if step.tool == REPORT:
                run.claimed_outcome = step.args.get("outcome")
                run.claimed_summary = step.args.get("summary")
                break
            key = json.dumps([step.tool, step.args], sort_keys=True)
            next_allowed = allowed_tools(brief, run, last_step=step_no + 1 >= self.max_steps)
            if key in cache:
                self.tools.audit.record(kind="guard", agent="orchestrator", tool="duplicate_call",
                                        arguments={"tool": step.tool, "args": step.args})
                messages.append(tool_message(step.tool, {"already_done": True, "result": cache[key],
                                                         "allowed_now": next_allowed}))
                continue
            result = self._call(step.tool, step.args, "operations")
            run.events.append(ToolEvent(step.tool, result, "operations"))
            cache[key] = agent_view(result.to_dict())
            next_allowed = allowed_tools(brief, run, last_step=step_no + 1 >= self.max_steps)
            messages.append(tool_message(step.tool, {**cache[key], **decide_hint(step.tool, result),
                                                     "allowed_now": next_allowed}))

    # --- code may READ after the loop, never act -------------------------------------------

    def _read_after_the_loop(self, brief: OpsBrief, run: OpsRun, messages: list[dict[str, Any]]) -> None:
        """Read-only safety nets for an agent that stopped early (SPEC §6.0).

        A confirmed proposal the agent did not execute stays unexecuted (NOT_EXECUTED): no code
        path executes on the agent's behalf. Measured with scripted FakeLLM runs, a correct agent
        never needs it (tests/unit/test_agents.py), so the old execute fallback is gone."""
        if run.timeout_pending:
            self._fallback(run, GET_REQUEST_STATUS, {}, "TIMEOUT not reconciled by the agent", messages)
        if brief.intent in PROPOSE_INTENTS and not run.proposal_decided:
            self._fallback(run, CHECK_PROPOSAL, {"remedy": brief.remedy},
                           "no decision by the agent; read-only check (a refusal may be reported, no proposal is made)",
                           messages)
        if brief.intent == CHECK_STATUS and not run.attempted(GET_REQUEST_STATUS):
            self._fallback(run, GET_REQUEST_STATUS, {}, "status not checked by the agent", messages)

    def _fallback(self, run: OpsRun, tool: str, args: dict[str, Any], reason: str,
                  messages: list[dict[str, Any]]) -> None:
        self.tools.audit.record(kind="fallback", agent="orchestrator", tool=tool, arguments=args,
                                result={"reason": reason})
        result = self._call(tool, args, "orchestrator")
        run.events.append(ToolEvent(tool, result, "orchestrator"))
        messages.append(desk_note_message(f"the desk called {tool} for you: "
                                          + json.dumps(agent_view(result.to_dict()), sort_keys=True)))

    def _call(self, tool: str, args: dict[str, Any], actor: str) -> ToolResult:
        """Dispatch to the gateway. Ops tools take no IDs: the target and grant are injected."""
        tools = self.tools
        if tool == LIST_EXCHANGE_OPTIONS:
            return self.runner.timed(lambda: tools.list_exchange_options(actor=actor))
        if tool == CREATE_PROPOSAL:
            return self.runner.timed(lambda: tools.create_proposal(args.get("remedy"), args.get("replacement_sku"), actor=actor))
        if tool == EXECUTE_PROPOSAL:
            return self.runner.timed(lambda: tools.execute_proposal(actor=actor))
        if tool == GET_REQUEST_STATUS:
            return self.runner.timed(lambda: tools.get_request_status(actor=actor))
        if tool == CHECK_PROPOSAL:
            return self.runner.timed(lambda: tools.check_proposal(args.get("remedy"), actor=actor))
        raise ValueError(f"not an Operations gateway tool: {tool!r}")


def decide_hint(tool: str, result: ToolResult) -> dict[str, str]:
    """After the option table, name the call that decides. Measured: with only the table, the model
    often reported "denied" for an unavailable size without asking create_proposal (11 of 16 such
    handoffs on the variant suite), so the refusal came from code's read-only check instead."""
    if tool != LIST_EXCHANGE_OPTIONS or not result.ok:
        return {}
    sku = result.data.get("requested_sku")
    call = {"remedy": EXCHANGE, **({"replacement_sku": sku} if sku else {})}
    return {"next": f"create_proposal {json.dumps(call)} decides, also when that option is not valid: "
                    "it then gives the exact reason."}


def allowed_tools(brief: OpsBrief, run: OpsRun, *, last_step: bool) -> list[str]:
    """Which Operations tools make sense now. The tool list itself never changes."""
    if run.timeout_pending:
        return [GET_REQUEST_STATUS]
    if last_step:
        return [REPORT]
    if brief.intent == EXECUTE_CONFIRMED:
        return [REPORT] if run.attempted(EXECUTE_PROPOSAL) else [EXECUTE_PROPOSAL, GET_REQUEST_STATUS, REPORT]
    if brief.intent == CHECK_STATUS:
        return [REPORT] if run.attempted(GET_REQUEST_STATUS) else [GET_REQUEST_STATUS, REPORT]
    if run.proposal_decided:
        return [REPORT]
    if brief.intent == PROPOSE_REFUND:
        return [CREATE_PROPOSAL, REPORT]
    return [LIST_EXCHANGE_OPTIONS, CREATE_PROPOSAL, REPORT]


# --- The code-built report --------------------------------------------------------------------


@dataclass
class HandoffFacts:
    """What one handoff established, read from gateway results only (never from LLM claims)."""

    intent: str
    proposal: dict[str, Any] | None = None
    reused: bool = False
    transaction: dict[str, Any] | None = None
    replay: bool = False
    reconciled: bool = False
    refusals: list[dict[str, Any]] = field(default_factory=list)  # reason dicts: {"code", ...details}
    gates: list[dict[str, Any]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    alternatives: list[dict[str, Any]] = field(default_factory=list)

    @property
    def decided(self) -> bool:
        """Did the handoff reach an answer for the customer? A NO_DECISION / NOT_EXECUTED handoff
        did not, so Resolution may hand off again within the turn's budget."""
        open_codes = {NO_DECISION, NOT_EXECUTED}
        return not any(r["code"] in open_codes for r in self.gates) and NOT_EXECUTED not in self.failures

    @property
    def outcome(self) -> str:
        if self.transaction is not None:
            return "completed"
        if self.proposal is not None:
            return "proposed"
        if self.refusals:
            return "denied"
        if self.failures:
            return "failed"
        return "needs_info"

    def report(self) -> OpsReport:
        codes = [r["code"] for r in self.refusals + self.gates] + list(self.failures)
        return OpsReport(intent=self.intent, outcome=self.outcome, reason_codes=codes, proposal=self.proposal,
                         transaction=self.transaction, replay=self.replay, reconciled=self.reconciled,
                         alternatives=list(self.alternatives))


def handoff_facts(brief: OpsBrief, run: OpsRun, catalog: list[dict[str, Any]]) -> HandoffFacts:
    """Classify every gateway result of the handoff with domain.CODE_CLASS.

    The informational tool (list_exchange_options) never refuses by itself; create_proposal and
    execute_proposal give authoritative answers (SPEC §6 decision table), and so does code's
    read-only check_proposal, but only with a refusal: its "would propose" becomes NO_DECISION.
    Format codes (agent slips such as INVALID_ARGS) are ignored.
    """
    facts = HandoffFacts(intent=brief.intent)
    timed_out = False
    for event in run.events:
        result = event.result
        if event.tool == EXECUTE_PROPOSAL and result.code == TIMEOUT:
            timed_out = True
        elif event.tool == GET_REQUEST_STATUS:
            completed = result.ok and result.data.get("status") == "completed"
            if completed and (timed_out or brief.intent == CHECK_STATUS):
                facts.transaction = result.data["record"]
                facts.reconciled = timed_out
                facts.replay = not timed_out
        elif event.tool == CHECK_PROPOSAL and result.ok:
            facts.gates.append({"code": NO_DECISION})  # valid, but only the agent may create it
        elif event.tool in (EXECUTE_PROPOSAL, CREATE_PROPOSAL) and result.ok:
            _absorb_success(facts, result.data)
        elif event.tool in (EXECUTE_PROPOSAL, CREATE_PROPOSAL, CHECK_PROPOSAL):
            _absorb_error(facts, brief, result, catalog)
        elif result.code is not None and code_class(result.code) == FAILURE:
            facts.failures.append(result.code)  # e.g. INTERNAL from an informational tool
    if timed_out and facts.transaction is None:
        facts.failures.append(TIMEOUT)
    if brief.intent == EXECUTE_CONFIRMED and not run.attempted(EXECUTE_PROPOSAL) and facts.transaction is None:
        facts.failures.append(NOT_EXECUTED)
    return facts


def _absorb_success(facts: HandoffFacts, data: dict[str, Any]) -> None:
    if data.get("replay"):
        facts.transaction, facts.replay = data["transaction"], True
    elif "transaction" in data:
        facts.transaction = data["transaction"]
    else:
        facts.proposal, facts.reused = data["proposal"], bool(data.get("reused"))


def _absorb_error(facts: HandoffFacts, brief: OpsBrief, result: ToolResult, catalog: list[dict[str, Any]]) -> None:
    error = result.error or {}
    details = error.get("details") or {}
    kind = code_class(result.code)
    if details.get("alternatives"):
        facts.alternatives = list(details["alternatives"])
    if kind == REFUSAL:
        facts.refusals.extend(reason_details(code, details, brief, catalog) for code in details.get("reasons") or [result.code])
    elif kind == GATE and result.code == SIZE_UNSPECIFIED:
        facts.gates.append({"code": result.code})
    elif kind == FAILURE:
        facts.failures.append(result.code)


def reason_details(code: str, details: dict[str, Any], brief: OpsBrief, catalog: list[dict[str, Any]]) -> dict[str, Any]:
    """A reason dict render.py can word precisely (days for the window, size/SKU for exchanges)."""
    reason: dict[str, Any] = {"code": code}
    eligibility = details.get("eligibility") or {}
    for key in ("days_since_delivery", "window_days"):
        if eligibility.get(key) is not None:
            reason[key] = eligibility[key]
    if code in SIZE_CODES:
        sku = details.get("requested_sku") or (brief.confirmed or {}).get("replacement_sku")
        product = policy.find_product(catalog, sku) if sku else None
        if product is not None:
            reason.update(sku=sku, size=product.get("size"))
        elif brief.requested_size and brief.requested_size not in ("UP", "DOWN"):
            reason["size"] = brief.requested_size
    return reason
