"""Desk: the per-turn pipeline (SPEC §6). Agents say yes; code can say no.

The agents decide the interpretation: which line, which remedy, which size, whether to ask and
what to say. Only an agent starts a business action (Resolution hands off -> Operations creates
the proposal -> Operations executes it after the customer's explicit confirmation). Code verifies
facts and may only BLOCK (the reason goes back to the agent as a tool error), READ (reconcile after
a timeout, a read-only check) or REPLAY (the stored result of a completed request). It never makes
a handoff, a proposal or a commit itself, and never swaps the agent's reading for a regex reading.

`Desk.handle_turn(message, confirm)` runs, in order:

1. Faults for this turn (tools.begin_turn), then token extraction over every customer message.
2. Confirmation. confirm "last" resolves to the proposal of the latest EARLIER turn output that
   carried one. The grant is withheld when this message really cancels or changes the request.
3. Idempotency preamble. If the request already completed: the same request is a deterministic
   replay, anything else is denied. Resolution only replies.
4. Route the Resolution agent:
     reply-only   preamble, or a pure cancel ("No, cancel that")
     grant        the agent hands off {intent: execute_confirmed}; Operations executes
     normal       handoffs are checked against the facts (grounding.check_*); a question the
                  conversation already answers is sent back ONCE with what is known, then the
                  agent's choice stands
5. The decision comes from gateway results only (`decide`, using domain.CODE_CLASS through
   operations.handoff_facts), verified against the store. Code adds a deny on its own only for
   facts: an order the customer typed that is not theirs, or a line they named by its ids that
   policy refuses.
6. Response = the LLM's sentence(s) + the code facts line, or the template if the guard fires.

Every block or read by code writes an audit entry of kind "gate", "guard" or "fallback", and those
are counted in the metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from desk import policy
from desk.agents import LIMITS, Limits, StepRunner, Telemetry
from desk.agents.operations import (
    PROPOSE_INTENTS,
    HandoffFacts,
    OperationsAgent,
    OpsBrief,
    OpsRun,
    agent_view,
    handoff_facts,
)
from desk.agents.prompts import (
    CONFIRMATION_DONE,
    CONFIRMATION_NONE,
    CONFIRMATION_WITHHELD,
    NO_CARD_TEXT,
    RESOLUTION_SYSTEM,
    account_card_text,
    confirmation_granted,
    line_label,
    operations_system,
    turn_block,
)
from desk.agents.resolution import ReplyVerdict, run_resolution
from desk.agents.toolspecs import (
    CHECK_STATUS,
    EXECUTE_CONFIRMED,
    FIND_ORDER_LINES,
    HANDOFF,
    PROPOSE_EXCHANGE,
    PROPOSE_REFUND,
    REPLY,
    operations_tools,
)
from desk.domain import (
    AMBIGUOUS_TARGET,
    BUDGET_EXHAUSTED,
    CLARIFY,
    COMPLETED,
    CONFIRMATION_REQUIRED,
    DENY,
    ERROR,
    EXCHANGE,
    INTERNAL,
    LINE_NOT_FOUND,
    LLM_ERROR,
    OPEN,
    ORDER_NOT_FOUND,
    PROPOSE,
    REFUND,
    REMEDY_MISMATCH,
    REMEDY_UNCLEAR,
    REPLACEMENT_NOT_REQUESTED,
    REQUEST_ALREADY_COMPLETED,
    SIZE_UNSPECIFIED,
    TARGET_MISMATCH,
    TARGET_NOT_GROUNDED,
    TOOL_NOT_ALLOWED,
    TurnOutput,
)
from desk.grounding import (
    BOTH,
    BY_CUSTOMER_IDS,
    BY_ONLY_LINE,
    NONE,
    SAME,
    Analysis,
    analyze,
    canonical_size,
    check_remedy,
    changes_the_open_offer,
    check_requested_size,
    check_target,
    newer_size_in_latest_words,
)
from desk.llm import LLMError
from desk.profiles import Profile
from desk.render import compose_response, money, reason_text, template_reply
from desk.sanitize import sanitize
from desk.tools import BusinessTools, normalize_ref

REPLY_ONLY, GRANT, NORMAL = "reply_only", "grant", "normal"
MAX_HANDOFFS = LIMITS.handoffs_per_turn  # read by the adapter configuration
MAX_LISTED_LINES = 8
ALL_LINES = normalize_ref("ALL")  # the lookup key of find_order_lines("ALL")
CONFIRMATION_CANCELLED = "CONFIRMATION_CANCELLED"  # wording only (render.REASON_TEXT); not a gateway code
UNREAD_SIZE = "mentions a size that is neither the item's nor the proposed one"
UNSPECIFIED_SIZES = {"", "UNSPECIFIED", "NONE", "ANY"}

REPLAY_NOTE = ("this request was already completed earlier, so nothing new will be processed. "
               "Tell the customer it is already done.")
ALREADY_COMPLETED_NOTE = ("this request was already completed for a different item, so nothing else can be "
                          "processed under it. Tell the customer politely.")
CANCEL_NOTE = ("the customer is cancelling. Do not hand off anything. Acknowledge it, say nothing was "
               "processed, and ask whether they need anything else.")
CLOSED_NOTE = "the request still does not identify one line and remedy. Ask the customer one short question now."


@dataclass(frozen=True)
class Handoff:
    """A checked handoff: the agent's target (a line summary of this account) and its reading."""

    intent: str
    target: dict[str, Any] | None = None
    requested_size: str | None = None
    requested_product: str | None = None
    grounded_by: str | None = None
    evidence: str | None = None

    def as_args(self) -> dict[str, Any]:
        """The handoff call these facts suggest (shown in the one-time clarify feedback)."""
        args: dict[str, Any] = {"intent": self.intent}
        if self.target is not None:
            args.update(order_id=self.target["order_id"], line_id=self.target["line_id"])
        if self.intent == PROPOSE_EXCHANGE:
            args.update(requested_size=self.requested_size or "UNSPECIFIED", requested_product=self.requested_product or "SAME")
        return args


@dataclass
class Turn:
    """Everything the desk knows and decides during one turn."""

    number: int
    message: str  # sanitized
    analysis: Analysis  # all customer messages so far
    last: Analysis  # this message alone (which IDs does it name?)
    lines: list[dict[str, Any]]
    catalog: list[dict[str, Any]]
    runner: StepRunner
    mode: str = NORMAL
    confirmed_id: str | None = None  # what the flag resolved to (even if withheld)
    grant: dict[str, Any] | None = None  # the confirmed proposal, when the grant stands
    withheld: str | None = None  # why the flag did not execute (cancel or change wording)
    pure_cancel: bool = False
    desk_note: str | None = None
    preamble: HandoffFacts | None = None
    handoffs: list[HandoffFacts] = field(default_factory=list)
    handoff_count: int = 0
    acted: bool = False  # a handoff reached a decision this turn (proposal, refusal, commit)
    status_checked: bool = False  # the agent asked Operations for the request status
    target: dict[str, Any] | None = None
    rejections: int = 0  # handoffs the fact checks blocked
    closed: bool = False  # too many blocked handoffs: only a question is allowed now
    vetoed: bool = False
    final_step: bool = False  # the agent's current step is its last one this turn
    code_denied: bool = False  # the deny came from code's own fact check, not from Operations
    eligibility: dict[tuple[str, str], list[dict[str, Any]]] = field(default_factory=dict)  # read cache
    lookups: dict[str, Any] = field(default_factory=dict)
    reply_text: str | None = None
    llm_error: str | None = None

    def line(self, order_id: str, line_id: str) -> dict[str, Any] | None:
        return next((x for x in self.lines if x["order_id"] == order_id and x["line_id"] == line_id), None)


@dataclass
class TurnFacts:
    """What this turn established, from gateway results and the store (input of `decide`)."""

    transaction: dict[str, Any] | None = None
    replay: bool = False
    reconciled: bool = False
    proposal: dict[str, Any] | None = None
    re_presented: bool = False
    refusals: list[dict[str, Any]] = field(default_factory=list)
    gates: list[dict[str, Any]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    alternatives: list[dict[str, Any]] = field(default_factory=list)


def decide(facts: TurnFacts) -> str:
    """SPEC §6 decision table, strongest first: completed > propose > deny > error > clarify."""
    if facts.transaction is not None:
        return COMPLETED
    if facts.proposal is not None:
        return PROPOSE
    if facts.refusals:
        return DENY
    if facts.failures:
        return ERROR
    return CLARIFY


class Desk:
    """One conversation's orchestrator. Construct once per session; call handle_turn per message."""

    def __init__(self, tools: BusinessTools, llm: Any, profile: Profile, limits: Limits = LIMITS) -> None:
        self.tools = tools
        self.llm = llm
        self.profile = profile
        self.limits = limits
        self.ops_specs = operations_tools(profile.structured_report)
        self.ops_system = operations_system(profile.structured_report)
        self.history: list[dict[str, Any]] = []  # {message, decision, proposal_id} per earlier turn
        self.telemetry = Telemetry()
        self.turn_latency_ms: list[float] = []
        self._card: str | None = None

    # === the pipeline ===========================================================================

    def handle_turn(self, message: str, confirm: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()  # telemetry only
        turn = self._start_turn(message, confirm)
        try:
            self._route(turn)
        except LLMError as exc:
            turn.llm_error = str(exc)[:300]
        facts = self._turn_facts(turn)
        decision = decide(facts)
        output = self._compose(turn, facts, decision)
        self.history.append({"message": turn.message, "decision": decision,
                             "proposal_id": output.proposal["proposal_id"] if output.proposal else None})
        self.telemetry.add(turn.runner.telemetry)
        self.turn_latency_ms.append(round((time.perf_counter() - started) * 1000, 1))
        return output.to_dict()

    def _start_turn(self, message: str, confirm: str | None) -> Turn:
        number = len(self.history) + 1
        self.tools.begin_turn(number)  # external faults apply before anything reads state
        lines = self.tools.customer_lines()
        state = self.tools.final_state()
        catalog = state.get("products", [])
        text = sanitize(message)
        analysis = analyze([h["message"] for h in self.history] + [text], lines, catalog, state.get("as_of"))
        turn = Turn(number=number, message=text, analysis=analysis, last=analyze([text], lines, catalog),
                    lines=lines, catalog=catalog, runner=StepRunner(self.llm, self.tools.audit, self.limits.llm_calls_per_turn))
        # A cancel with nothing new in it ("No, cancel that") is a pure cancel. With a new remedy,
        # size or item ("an L instead", "hmm, I meant line 2") it changes the request instead.
        turn.pure_cancel = (analysis.cancel_cue and analysis.last_remedy_cue == NONE
                            and analysis.last_requested_size is None and not analysis.last_references)
        self.tools.audit.record(kind="grounding", agent="orchestrator", tool="analyze", arguments={"turn": number},
                                result=analysis.to_dict())
        if confirm:
            self._resolve_confirmation(turn, confirm)
        return turn

    def _resolve_confirmation(self, turn: Turn, confirm: str) -> None:
        presented = [h["proposal_id"] for h in self.history if h["proposal_id"]]
        proposal_id = (presented[-1] if presented else None) if confirm == "last" else (confirm if confirm in presented else None)
        proposal = self.tools.get_proposal(proposal_id) if proposal_id else None
        if proposal is None:
            self.tools.audit.record(kind="confirmation", agent="orchestrator", tool="confirmation_unresolved",
                                    arguments={"confirm_proposal": confirm}, result={"presented": presented})
            return
        turn.confirmed_id = proposal_id
        conflict = self._grant_conflict(turn, proposal)
        if conflict:
            turn.withheld = conflict
            self.tools.audit.record(kind="guard", agent="orchestrator", tool="grant_withheld",
                                    arguments={"proposal_id": proposal_id}, result={"reason": conflict})
            return
        turn.grant = proposal
        self.tools.set_grant(proposal_id)

    def _grant_conflict(self, turn: Turn, proposal: dict[str, Any]) -> str | None:
        """Why the flag must not execute this proposal: the message really cancels or changes it
        (CLARIFICATIONS #9). Weak words ("I'll post the old one back") and negated cancel words
        ("please don't cancel") are not changes: any other wording executes the bound proposal."""
        a, last = turn.analysis, turn.last
        if a.cancel_cue:
            return "cancel wording"
        if a.last_strong_remedy == BOTH or a.last_remedy_cue == BOTH:
            return "both remedies mentioned"
        if a.last_strong_remedy in (REFUND, EXCHANGE) and a.last_strong_remedy != proposal["remedy"]:
            return f"asks for a {a.last_strong_remedy} instead"
        if a.last_requested_size is not None:
            sku, _code = policy.resolve_size(a.last_requested_size, proposal["original_sku"], turn.catalog)
            if sku != proposal["replacement_sku"]:
                return f"asks for size {a.last_requested_size}"
        if self._mentions_unbound_size(turn, proposal):
            return UNREAD_SIZE
        if last.unknown_order_refs or last.unknown_line_refs:
            return "names an item that is not on the account"
        if (last.order_refs and proposal["order_id"] not in last.order_refs) or (
                last.line_refs and proposal["line_id"] not in last.line_refs):
            return "names a different line"
        bound = turn.line(proposal["order_id"], proposal["line_id"])
        if last.product_refs and bound is not None and bound["product_id"] not in last.product_refs:
            return "names a different item"
        return None

    def _mentions_unbound_size(self, turn: Turn, proposal: dict[str, Any]) -> bool:
        """A size that is neither the item's nor the replacement's. For an exchange the size IS the
        binding, so any such size means the customer is still choosing. For a refund it matters
        only when another of the customer's lines has that size (they may mean that line)."""
        bound = {canonical_size(s) for s in (_sku_size(turn.catalog, proposal["original_sku"]),
                                             _sku_size(turn.catalog, proposal["replacement_sku"])) if s}
        unbound = {s for s in turn.analysis.last_sizes if s not in bound}
        if proposal["remedy"] == EXCHANGE:
            return bool(unbound)
        others = {canonical_size(x["size"]) for x in turn.lines if x["size"] is not None
                  and (x["order_id"], x["line_id"]) != (proposal["order_id"], proposal["line_id"])}
        return bool(unbound & others)

    def _route(self, turn: Turn) -> None:
        record = self.tools.completed_record()
        if record is not None:
            self._replay_preamble(turn, record)
            turn.mode = REPLY_ONLY
        elif turn.grant is not None:
            turn.mode = GRANT
            turn.target = turn.line(turn.grant["order_id"], turn.grant["line_id"])
        elif turn.pure_cancel:
            turn.mode = REPLY_ONLY
            turn.desk_note = CANCEL_NOTE
        turn.reply_text = run_resolution(turn.runner, self._resolution_messages(turn), _TurnHooks(self, turn), self.limits)

    def _replay_preamble(self, turn: Turn, record: dict[str, Any]) -> None:
        """SPEC §6.2: once completed, the same request replays (no effects) and anything else is denied."""
        turn.runner.timed(lambda: self.tools.get_request_status(actor="orchestrator"))
        if turn.grant is not None:
            same = record.get("proposal_id") == turn.grant["proposal_id"]
        else:
            same = record_matches_text(turn.analysis, record)
        facts = HandoffFacts(intent="idempotency_preamble")
        if same:
            facts.transaction, facts.replay = record, True
            turn.desk_note = REPLAY_NOTE
        else:
            facts.refusals.append({"code": REQUEST_ALREADY_COMPLETED})
            turn.desk_note = ALREADY_COMPLETED_NOTE
            if turn.analysis.unique_target:
                turn.target = turn.line(*turn.analysis.unique_target)
        turn.preamble = facts
        self.tools.audit.record(kind="route", agent="orchestrator", tool="idempotency_preamble",
                                arguments={"turn": turn.number},
                                result={"replay": same, "transaction_id": record.get("transaction_id")})

    # === Resolution context and hooks ===========================================================

    def _resolution_messages(self, turn: Turn) -> list[dict[str, Any]]:
        if turn.preamble is not None:
            confirmation = CONFIRMATION_DONE if turn.confirmed_id else CONFIRMATION_NONE
        elif turn.grant is not None:
            confirmation = confirmation_granted(turn.grant)
        else:
            confirmation = CONFIRMATION_WITHHELD if turn.confirmed_id else CONFIRMATION_NONE
        open_proposal = self.tools.open_proposal()
        block = turn_block(self.history, agent_view(open_proposal) if open_proposal else None, confirmation,
                           turn.desk_note, turn.message, self._allowed(turn, final=False))
        return [{"role": "system", "content": RESOLUTION_SYSTEM},
                {"role": "user", "content": self._account_card(turn)},
                {"role": "user", "content": block}]

    def _account_card(self, turn: Turn) -> str:
        """Prefetched once per conversation (it hides policy fields, so it never changes)."""
        if not self.profile.prefetch:
            return NO_CARD_TEXT
        if self._card is None:
            result = turn.runner.timed(lambda: self.tools.account_card())
            self._card = account_card_text(result.data or {}, turn.analysis.as_of)
        return self._card

    def _allowed(self, turn: Turn, final: bool) -> list[str]:
        if turn.mode == REPLY_ONLY:
            return [REPLY]
        if turn.mode == GRANT:  # even on the last step the agent may still carry out the confirmation
            return [REPLY] if turn.acted or turn.handoff_count >= self.limits.handoffs_per_turn else [HANDOFF, REPLY]
        if final or turn.acted or turn.closed or turn.handoff_count >= self.limits.handoffs_per_turn:
            return [REPLY]
        # Once every line was listed there is nothing left to look up. Measured: the native baseline
        # otherwise repeated find_order_lines("ALL") until the step budget ran out (an error turn).
        if self.profile.prefetch or ALL_LINES in turn.lookups:
            return [HANDOFF, REPLY]
        return [FIND_ORDER_LINES, HANDOFF, REPLY]

    def _before_step(self, turn: Turn, step_no: int, final: bool) -> str | None:
        """A note for the agent between steps. Code never acts for the agent here."""
        turn.final_step = final
        if turn.mode == NORMAL and not turn.acted and not turn.closed and turn.rejections >= self.limits.gate_rejections:
            turn.closed = True
            self.tools.audit.record(kind="guard", agent="orchestrator", tool="handoffs_closed",
                                    result={"reason": f"{turn.rejections} rejected handoffs; only a question is allowed now"})
            return CLOSED_NOTE
        return None

    def _lookup(self, turn: Turn, args: dict[str, Any]) -> dict[str, Any]:
        order_ref = str(args.get("order_ref") or "ALL")
        key = normalize_ref(order_ref)
        if key in turn.lookups:
            self.tools.audit.record(kind="guard", agent="orchestrator", tool="duplicate_call",
                                    arguments={"tool": FIND_ORDER_LINES, "args": args})
            return {"already_done": True, "result": turn.lookups[key], "allowed_now": self._allowed(turn, False)}
        result = turn.runner.timed(lambda: self.tools.find_order_lines(order_ref))
        turn.lookups[key] = agent_view(result.to_dict())
        return {**turn.lookups[key], "allowed_now": self._allowed(turn, False)}

    def _handoff(self, turn: Turn, args: dict[str, Any]) -> dict[str, Any]:
        intent = args.get("intent")
        if turn.mode == GRANT and not turn.acted and self._reads_a_change(turn, args):
            self._withdraw_grant(turn, "the agent read a changed request")
        if turn.mode == GRANT:
            if intent != EXECUTE_CONFIRMED or turn.acted or turn.handoff_count >= self.limits.handoffs_per_turn:
                return self._reject(turn, TOOL_NOT_ALLOWED, "This turn carries the customer's explicit confirmation. "
                                    'The only allowed handoff is {"intent": "execute_confirmed"}.')
            return self._run_handoff(turn, Handoff(EXECUTE_CONFIRMED))
        if turn.mode == REPLY_ONLY or turn.acted or turn.closed or turn.handoff_count >= self.limits.handoffs_per_turn:
            return self._reject(turn, TOOL_NOT_ALLOWED, "No handoff now. Call reply_to_customer.")
        if intent == EXECUTE_CONFIRMED:
            return self._reject(turn, CONFIRMATION_REQUIRED, "The customer did not use the explicit confirm control this "
                                "turn, so nothing can be executed. Re-present the open proposal with its propose intent.")
        if intent == CHECK_STATUS:
            turn.status_checked = True
            return self._run_handoff(turn, Handoff(CHECK_STATUS))
        handoff, rejection = self._check_propose(turn, args)
        if rejection is not None:
            turn.rejections += 1
            self.tools.audit.record(kind="gate", agent="orchestrator", tool=HANDOFF, arguments=args, error=rejection)
            return {"error": rejection}
        return self._run_handoff(turn, handoff)

    def _reads_a_change(self, turn: Turn, args: dict[str, Any]) -> bool:
        """On a confirmed turn the agent proposes something other than the bound proposal: its
        reading is that the message changes the request ("Yes, refund the t-shirt" on a jeans
        proposal, "yes, but make it an 11"). Saying no to execution is the agent's call; a
        proposal for the same thing is not a change and still only allows execute_confirmed."""
        bound, intent = turn.grant, args.get("intent")
        if intent not in (PROPOSE_REFUND, PROPOSE_EXCHANGE):
            return False
        if (args.get("order_id"), args.get("line_id")) != (bound["order_id"], bound["line_id"]):
            return True
        if (intent == PROPOSE_EXCHANGE) != (bound["remedy"] == EXCHANGE):
            return True
        size = _size_arg(args.get("requested_size"))
        if intent != PROPOSE_EXCHANGE or size is None:
            return False
        sku, _code = policy.resolve_size(size, bound["original_sku"], turn.catalog)
        return sku != bound["replacement_sku"]

    def _withdraw_grant(self, turn: Turn, reason: str) -> None:
        """The flag does not execute: the turn is handled as unconfirmed (CLARIFICATIONS #9)."""
        self.tools.audit.record(kind="guard", agent="orchestrator", tool="grant_withheld",
                                arguments={"proposal_id": turn.grant["proposal_id"]}, result={"reason": reason})
        turn.withheld, turn.grant, turn.target, turn.mode = reason, None, None, NORMAL
        self.tools.set_grant(None)

    def _check_propose(self, turn: Turn, args: dict[str, Any]) -> tuple[Handoff | None, dict[str, Any] | None]:
        """Facts can block the agent's propose handoff; they never replace its reading (SPEC §6.0)."""
        a = turn.analysis
        intent = args["intent"]
        verdict = check_target(a, args.get("order_id"), args.get("line_id"), args.get("evidence"))
        if not verdict.ok:
            return None, self._gate_feedback(turn, verdict.code, verdict.reason)
        ok, code = check_remedy(a, intent)
        if not ok:
            return None, self._gate_feedback(turn, code)
        target = turn.line(*verdict.key)
        if intent == PROPOSE_REFUND:
            return Handoff(PROPOSE_REFUND, target, grounded_by=verdict.grounded_by, evidence=verdict.evidence), None
        offer, size = self.tools.open_proposal(), args.get("requested_size")
        moves_offer = (offer is not None and offer["remedy"] == EXCHANGE and (offer["order_id"], offer["line_id"]) == verdict.key
                       and changes_the_open_offer(turn.last, size, target["sku"], offer["replacement_sku"], turn.catalog))
        if _product_arg(args.get("requested_product"), target) is None and (
                moves_offer or newer_size_in_latest_words(turn.last, size, target["sku"], turn.catalog)):
            ok, code = True, None
        else:
            ok, code = check_requested_size(a, args.get("requested_size"), target["sku"], turn.catalog,
                                            args.get("requested_product"))
        if not ok:
            return None, self._gate_feedback(turn, code)
        return Handoff(PROPOSE_EXCHANGE, target, _size_arg(args.get("requested_size")),
                       _product_arg(args.get("requested_product"), target), verdict.grounded_by, verdict.evidence), None

    def _gate_feedback(self, turn: Turn, code: str, reason: str | None = None) -> dict[str, Any]:
        a = turn.analysis
        asked = a.requested_sizes[0] if len(a.requested_sizes) == 1 else None
        hints = {
            ORDER_NOT_FOUND: "That order is not on this customer's account. Tell the customer; do not hand off.",
            LINE_NOT_FOUND: "That line is not in the order. Ask the customer which line they mean.",
            AMBIGUOUS_TARGET: ("More than one line could fit. If the customer's words pick one, hand off again with "
                               "evidence set to their exact words; otherwise ask which line they mean."),
            TARGET_MISMATCH: "That line does not fit what the customer typed. Use the line they named, or ask.",
            REMEDY_UNCLEAR: "The customer mentioned both a refund and an exchange. Ask which one they want.",
            REMEDY_MISMATCH: f"The customer's latest words ask for a {a.stated_remedy}, not that.",
            SIZE_UNSPECIFIED: ("The customer named more than one size. Ask which one they want." if len(a.requested_sizes) > 1
                               else 'Use the size in the customer\'s own words. If they named none, hand off with requested_size '
                               '"UNSPECIFIED": Operations then proposes the only possible size or lists the sizes to offer.'),
            REPLACEMENT_NOT_REQUESTED: ("The customer asked for the same size they already have." if asked == SAME
                                        else f"The customer asked for size {asked}."),
        }
        feedback: dict[str, Any] = {"code": code, "message": reason_text(code), "hint": hints[code]}
        if reason:
            feedback["why"] = reason
        listed = a.id_candidates if code == AMBIGUOUS_TARGET else a.candidates
        feedback["candidates"] = [line_label(c) for c in listed[:MAX_LISTED_LINES]]
        return feedback

    def _reject(self, turn: Turn, code: str, message: str) -> dict[str, Any]:
        """A handoff this turn does not allow (protocol, not facts): no rejection is counted, so a
        later valid handoff is still possible within the step budget."""
        self.tools.audit.record(kind="gate", agent="orchestrator", tool=HANDOFF, error={"code": code, "message": message})
        return {"error": {"code": code, "message": message, "allowed_now": self._allowed(turn, False)}}

    def _reply(self, turn: Turn, message: str) -> ReplyVerdict:
        """Accept the reply, or send a question back ONCE when the conversation already answers it."""
        if turn.mode == GRANT and not turn.acted and turn.handoff_count < self.limits.handoffs_per_turn:
            return ReplyVerdict(False, {"error": {"code": "CONFIRMATION_PENDING", "message": (
                'The customer confirmed with the explicit control. Hand off {"intent": "execute_confirmed"} first.')}})
        if turn.mode != NORMAL or turn.acted or turn.vetoed or turn.final_step or turn.closed:
            return ReplyVerdict(True)  # the agent could not act on the feedback any more
        suggestion = self._clarify_veto(turn)
        if suggestion is None:
            return ReplyVerdict(True)
        turn.vetoed = True
        self.tools.audit.record(kind="guard", agent="orchestrator", tool="clarify_veto", arguments=suggestion.as_args(),
                                result={"reason": "the conversation already names one line and the remedy"})
        return ReplyVerdict(False, {"error": {"code": "HANDOFF_SUGGESTED", "message": self._veto_text(suggestion),
                                              "handoff": suggestion.as_args()}})

    def _veto_text(self, suggestion: Handoff) -> str:
        """Why the agent's question looks unnecessary, in the words the agent needs to act on it."""
        target = suggestion.target
        open_proposal = self.tools.open_proposal()
        if open_proposal is not None and (open_proposal["order_id"], open_proposal["line_id"]) == (
                target["order_id"], target["line_id"]):
            return (f"Proposal {open_proposal['proposal_id']} for this line is still open: nothing is done until the "
                    "customer uses the confirm control. Hand off the same intent again to re-present it. If the "
                    "customer really asked something else, reply again and your message is sent.")
        remedy = "an exchange" if suggestion.intent == PROPOSE_EXCHANGE else "a refund"
        sizes = (' With no size named, hand off with requested_size "UNSPECIFIED": Operations proposes the only '
                 "possible size, or tells you which sizes to offer, so do not ask about sizes first."
                 if suggestion.intent == PROPOSE_EXCHANGE and suggestion.requested_size is None else "")
        return (f"You asked a question, but the customer's words already name one line ({line_label(target)}) "
                f"and ask for {remedy}. Hand off: Operations checks eligibility, sizes and stock.{sizes} If you "
                "still need the customer's answer, reply again and your question is sent.")

    def _clarify_veto(self, turn: Turn) -> Handoff | None:
        """The handoff the conversation suggests, when the agent's question looks unnecessary.

        Feedback only: the agent decides. Never for a question about rules or status, after the
        agent checked the status, or when both remedies (or no remedy) were mentioned
        (CLARIFICATIONS #4). Exchange without a size follows CLARIFICATIONS #5: with 2+ valid
        sizes asking is right, unless policy already refuses the line (then Operations must say so).
        """
        a = turn.analysis
        if a.unique_target is None or a.inquiry or turn.status_checked or turn.withheld == UNREAD_SIZE:
            return None
        target = turn.line(*a.unique_target)
        if a.remedy_cue == REFUND:
            return Handoff(PROPOSE_REFUND, target)
        if a.remedy_cue != EXCHANGE or len(a.requested_sizes) > 1:
            return None
        product = a.requested_product if a.requested_product != target.get("product_id") else None
        if a.requested_size is None and product is None and len(self._valid_options(target)) >= 2 \
                and not self._ineligibility(turn, target):
            return None
        return Handoff(PROPOSE_EXCHANGE, target, a.requested_size, product)

    def _valid_options(self, target: dict[str, Any]) -> list[dict[str, Any]]:
        options = policy.valid_options(policy.options_for_sku(self.tools.final_state(), target["sku"]))
        return [{"sku": o.sku, "size": o.size} for o in options]

    def _ineligibility(self, turn: Turn, target: dict[str, Any]) -> list[dict[str, Any]]:
        """Policy's refusal reasons for a line (audited read through the gateway; cached per turn)."""
        key = (target["order_id"], target["line_id"])
        if key not in turn.eligibility:
            result = turn.runner.timed(lambda: self.tools.check_eligibility(key[0], key[1], actor="orchestrator"))
            data = result.data or {}
            turn.eligibility[key] = [] if not result.ok or data.get("eligible") or data.get("replay") else [
                {"code": code, "days_since_delivery": data.get("days_since_delivery"),
                 "window_days": data.get("window_days")} for code in data.get("reasons") or []]
        return turn.eligibility[key]

    # === Operations handoff =====================================================================

    def _run_handoff(self, turn: Turn, handoff: Handoff) -> dict[str, Any]:
        """Inject the agent's checked target, run Operations, keep the code-built facts."""
        turn.handoff_count += 1
        if handoff.intent in PROPOSE_INTENTS:
            target = handoff.target
            self.tools.set_target(target["order_id"], target["line_id"], handoff.intent,
                                  requested_size=handoff.requested_size, requested_product=handoff.requested_product,
                                  grounded_by=handoff.grounded_by, evidence=handoff.evidence)
            turn.target = target
        brief = OpsBrief(handoff.intent, handoff.target, handoff.requested_size, handoff.requested_product,
                         confirmed=turn.grant if handoff.intent == EXECUTE_CONFIRMED else None)
        run = OpsRun()
        try:
            OperationsAgent(turn.runner, self.tools, self.ops_specs, self.ops_system, self.limits).run(brief, run)
        finally:  # even after an LLMError, what the gateway did this turn must reach the decision
            facts = handoff_facts(brief, run, turn.catalog)
            turn.handoffs.append(facts)
            turn.acted = turn.acted or (handoff.intent != CHECK_STATUS and facts.decided)
            self.tools.audit.record(kind="ops_report", agent="orchestrator", tool="report_to_resolution",
                                    arguments={"intent": handoff.intent}, result=facts.report().to_dict())
        if run.claimed_outcome is not None and run.claimed_outcome != facts.outcome:
            self.tools.audit.record(kind="guard", agent="orchestrator", tool="report_mismatch",
                                    arguments={"claimed": run.claimed_outcome}, result={"actual": facts.outcome})
        return self._report_view(run, facts)

    def _report_view(self, run: OpsRun, facts: HandoffFacts) -> dict[str, Any]:
        """What Resolution reads: the code-built report (F) or the Ops agent's own summary (B)."""
        if self.profile.structured_report:
            view = structured_report(facts)
        else:
            view = {"outcome": run.claimed_outcome or facts.outcome, "summary": run.claimed_summary or ""}
        view["next"] = REPLY if facts.decided else f"{HANDOFF} again, or {REPLY}"
        return view

    # === decision and response ==================================================================

    def _turn_facts(self, turn: Turn) -> TurnFacts:
        facts = TurnFacts()
        for source in ([turn.preamble] if turn.preamble else []) + turn.handoffs:
            if source.transaction is not None:
                facts.transaction = source.transaction
                facts.replay = facts.replay or source.replay
                facts.reconciled = facts.reconciled or source.reconciled
            if source.proposal is not None:
                facts.proposal, facts.re_presented = source.proposal, source.reused
            facts.refusals += source.refusals
            facts.gates += source.gates
            facts.failures += source.failures
            facts.alternatives = source.alternatives or facts.alternatives
        a = turn.analysis
        # An explicitly named order that is not on the account, with nothing else grounded, is an
        # authoritative "not found" computed by code (identical for foreign and nonexistent orders).
        if turn.mode == NORMAL and a.not_found_code == ORDER_NOT_FOUND and a.unique_target is None and not turn.acted:
            facts.refusals.append(named_reason(ORDER_NOT_FOUND, a.unknown_order_refs))
        if turn.llm_error:
            facts.failures.append(LLM_ERROR)
        elif turn.reply_text is None:
            facts.failures.append(BUDGET_EXHAUSTED)
        self._verify_against_store(facts)
        if decide(facts) == CLARIFY:
            self._deny_named_ineligible_line(turn, facts)
        return facts

    def _deny_named_ineligible_line(self, turn: Turn, facts: TurnFacts) -> None:
        """Code may say no on facts: the customer named the line by its ids (or owns only that line),
        asked for a refund or exchange, and policy refuses the line. Then a question is pointless
        and asking about sizes would advertise an exchange that cannot happen (review B3)."""
        a = turn.analysis
        if turn.mode != NORMAL or turn.acted or a.inquiry or a.remedy_cue == NONE or len(a.id_candidates) != 1:
            return
        key = a.id_candidate_keys[0]
        verdict = check_target(a, *key)
        if not (verdict.ok and verdict.grounded_by in (BY_CUSTOMER_IDS, BY_ONLY_LINE)):
            return
        target = turn.line(*key)
        reasons = self._ineligibility(turn, target)
        if reasons:
            facts.refusals += reasons
            turn.target, turn.code_denied = target, True
            self.tools.audit.record(kind="guard", agent="orchestrator", tool="code_deny",
                                    arguments={"order_id": key[0], "line_id": key[1]},
                                    result={"reasons": [r["code"] for r in reasons]})

    def _verify_against_store(self, facts: TurnFacts) -> None:
        """completed only if the store holds that transaction; propose only if the proposal is open."""
        if facts.transaction is not None:
            record = self.tools.completed_record()
            if record is None or record.get("transaction_id") != facts.transaction.get("transaction_id"):
                facts.transaction = None
                facts.failures.append(INTERNAL)
        if facts.proposal is not None:
            current = self.tools.get_proposal(facts.proposal["proposal_id"])
            facts.proposal = current if current is not None and current["status"] == OPEN else None

    def _compose(self, turn: Turn, facts: TurnFacts, decision: str) -> TurnOutput:
        details: dict[str, Any] = {"catalog": turn.catalog}
        if decision == PROPOSE:
            details.update(proposal=facts.proposal,
                           reasons=[{"code": CONFIRMATION_REQUIRED}] if facts.re_presented else [])
        elif decision == COMPLETED:
            details.update(transaction=facts.transaction, proposal=self.tools.get_proposal(facts.transaction.get("proposal_id", "")),
                           replay=facts.replay, reconciled=facts.reconciled, target=turn.target)
        elif decision == DENY:
            details.update(target=turn.target, reasons=facts.refusals, alternatives=facts.alternatives)
        elif decision == CLARIFY:
            details.update(self._clarify_details(turn, facts))
        else:
            details.update(reasons=[{"code": code} for code in dict.fromkeys(facts.failures)])
        known_skus = [str(p.get("sku")) for p in turn.catalog if p.get("sku")]
        if turn.code_denied:  # the agent's text was a question, so the reply comes from the facts
            response, violations = template_reply(decision, **details), []
        else:
            response, violations = compose_response(turn.reply_text, decision, known_skus=known_skus, **details)
        if turn.reply_text is not None and violations:
            self.tools.audit.record(kind="guard", agent="orchestrator", tool="reply_guard",
                                    arguments={"decision": decision}, result={"violations": violations})
        return TurnOutput(decision=decision, response=response,
                          proposal=facts.proposal if decision == PROPOSE else None,
                          transaction=facts.transaction if decision == COMPLETED else None,
                          confirmed_proposal_id=turn.grant["proposal_id"] if turn.grant else None)

    def _clarify_details(self, turn: Turn, facts: TurnFacts) -> dict[str, Any]:
        """Reason, candidates and sizes for a clarify, from code facts only."""
        a = turn.analysis
        if turn.pure_cancel:
            open_proposal = self.tools.open_proposal()
            target = turn.line(open_proposal["order_id"], open_proposal["line_id"]) if open_proposal else None
            return {"reasons": [{"code": CONFIRMATION_CANCELLED}], "target": target}
        if facts.gates:
            return {"reasons": facts.gates, "alternatives": facts.alternatives, "target": turn.target}
        if a.not_found_code == LINE_NOT_FOUND:
            return {"reasons": [named_reason(LINE_NOT_FOUND, a.unknown_line_refs)], "candidates": self._candidates(turn)}
        if a.remedy_cue == NONE and not _names_an_item(a):
            return {}  # small talk or a general question: no item and no remedy to talk about (RT-8)
        if a.unique_target is None:
            return {"reasons": [{"code": TARGET_NOT_GROUNDED}], "candidates": self._candidates(turn)}
        target = turn.line(*a.unique_target)
        if a.remedy_cue in (NONE, BOTH):
            return {"reasons": [{"code": REMEDY_UNCLEAR}], "target": target}
        if a.remedy_cue == EXCHANGE and a.requested_size is None:
            # Never list exchange sizes for a line policy refuses (review B3).
            sizes = [] if self._ineligibility(turn, target) else self._valid_options(target)
            return {"reasons": [{"code": SIZE_UNSPECIFIED}], "alternatives": sizes, "target": target}
        return {"candidates": [target]}

    def _candidates(self, turn: Turn) -> list[dict[str, Any]]:
        """The lines the customer may mean: the analysis candidates, else the named orders' lines."""
        a = turn.analysis
        lines = a.candidates or [x for x in turn.lines if x["order_id"] in a.order_refs] or turn.lines
        return lines[:MAX_LISTED_LINES]


class _TurnHooks:
    """Connects the Resolution loop to this turn's rules (implements resolution.ResolutionHooks)."""

    def __init__(self, desk: Desk, turn: Turn) -> None:
        self.desk = desk
        self.turn = turn

    def allowed(self, final: bool) -> list[str]:
        return self.desk._allowed(self.turn, final)

    def before_step(self, step_no: int, final: bool) -> str | None:
        return self.desk._before_step(self.turn, step_no, final)

    def lookup(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.desk._lookup(self.turn, args)

    def handoff(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.desk._handoff(self.turn, args)

    def reply(self, message: str) -> ReplyVerdict:
        return self.desk._reply(self.turn, message)


# === small pure helpers =========================================================================


def record_matches_text(analysis: Analysis, record: dict[str, Any]) -> bool:
    """Is the conversation still about the completed transaction? A different named line or order,
    or strong wording for the other remedy ("now exchange it"), is a different request (review B4)."""
    action = record.get("action")
    if action and analysis.stated_remedy in (REFUND, EXCHANGE) and analysis.stated_remedy != action:
        return False
    if analysis.unique_target is not None:
        return analysis.unique_target == (record.get("order_id"), record.get("line_id"))
    if analysis.not_found_code:
        return False
    if analysis.order_refs and record.get("order_id") not in analysis.order_refs:
        return False
    return not (analysis.line_refs and record.get("line_id") not in analysis.line_refs)


def named_reason(code: str, named: list[str]) -> dict[str, Any]:
    """A not-found reason that repeats the one id the customer typed ("order ORD-99"), so the reply
    may echo it. Same wording whether the order does not exist or is another customer's, so nothing
    leaks. With several unknown ids we cannot tell which one is the order: generic wording."""
    return {"code": code, "named": named[0]} if len(named) == 1 else {"code": code}


def _names_an_item(analysis: Analysis) -> bool:
    return bool(analysis.order_refs or analysis.line_refs or analysis.product_refs or analysis.unknown_order_refs
                or analysis.unknown_line_refs or analysis.unmatched_product_words or analysis.mentioned_sizes)


def _sku_size(catalog: list[dict[str, Any]], sku: str | None) -> str | None:
    product = policy.find_product(catalog, sku) if sku else None
    return product.get("size") if product else None


def _size_arg(value: Any) -> str | None:
    """The agent's requested size, or None for "UNSPECIFIED"."""
    text = str(value).strip() if value is not None else ""
    return None if text.upper() in UNSPECIFIED_SIZES else text


def _product_arg(value: Any, target: dict[str, Any]) -> str | None:
    """The product the agent says the customer asks for, if it is not the line's own product."""
    text = str(value).strip() if value is not None else ""
    if not text or text.upper() == SAME or normalize_ref(text) == normalize_ref(target.get("product_id")):
        return None
    return text


def structured_report(facts: HandoffFacts) -> dict[str, Any]:
    """Compact, code-built report for Resolution (no identity fields)."""
    view: dict[str, Any] = {"outcome": facts.outcome}
    if facts.proposal is not None:
        p = facts.proposal
        view["proposal"] = {"proposal_id": p["proposal_id"], "remedy": p["remedy"], "order_id": p["order_id"],
                            "line_id": p["line_id"], "item": p["original_sku"], "replacement": p["replacement_sku"],
                            "amount": money(p["amount_cents"])}
        view["customer_must_confirm"] = True
    if facts.transaction is not None:
        t = facts.transaction
        result = t.get("result") or {}
        view["transaction"] = {k: result.get(k) for k in ("transaction_id", "refund_id", "exchange_id") if result.get(k)}
        if isinstance(t.get("amount_cents"), int):
            view["transaction"]["amount"] = money(t["amount_cents"])
        view["already_processed_earlier"] = facts.replay
        view["recovered_from_timeout"] = facts.reconciled
    if facts.refusals:
        view["reasons"] = [reason_text(r["code"], r) for r in facts.refusals]
    if facts.gates:
        view["needs"] = [reason_text(r["code"], r) for r in facts.gates]
    if facts.alternatives:
        view["alternatives"] = [f"{o.get('sku')} (size {o.get('size')})" for o in facts.alternatives]
    if facts.failures:
        view["problems"] = list(facts.failures)
    return view
