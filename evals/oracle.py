"""Independent rule oracle: the expected decision per turn and the expected final business state.

Input is a contract case plus one structured *intent* per turn -- what the customer's message means,
already grounded to IDs by whoever wrote the label. The oracle never reads language. It re-implements
the disclosed rules (candidate_assets/CONTRACT.md), the business defaults (docs/CLARIFICATIONS.md)
and the SPEC section 2 record shapes, and imports nothing from desk/: if both shared code, a shared
bug would grade itself as correct.

Intent kinds:
  refund / exchange  order_id, line_id (+ for exchange: requested_size | requested_sku |
                     requested_product; requested_size may be "UP" or "DOWN")
  clarify_needed     the message has no unique target or remedy
  confirm            affirmation sent with confirm_proposal "last" (the flag)
  verbal_yes         affirmation without the flag
  status             "did it go through?"
  cancel             "no, cancel it"
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from evals.state_view import business_view

WINDOW_DAYS = 30
SIZE_LADDER = ["XXS", "XS", "S", "M", "L", "XL", "XXL", "3XL", "4XL", "5XL"]
INTENT_KINDS = {"refund", "exchange", "clarify_needed", "confirm", "verbal_yes", "status", "cancel"}
# Defaults we chose where the brief is silent; graders may expect the alternative (CLARIFICATIONS.md).
CONTESTED_ALTERNATIVES = {
    "COMMITTED_TIMEOUT_RECONCILED": ["error"],  # Q1: reconcile in the same turn vs report error
    "LINE_NOT_FOUND": ["deny"],  # unknown line inside an owned order
    "CANCELLED_BY_CUSTOMER": ["deny"],  # cancel phrase: no mutation either way
}


@dataclass
class TurnExpectation:
    decision: str
    reason: str
    proposal: dict | None = None  # binding + "seq" (1-based proposal identity) on propose turns


@dataclass
class Outcome:
    turns: list[TurnExpectation]
    final_state: dict
    events: list[dict]
    proposals: list[dict]


@dataclass
class _Proposal:
    seq: int
    binding: dict
    status: str = "open"  # open | superseded | executed | invalidated

    def view(self) -> dict:
        return {"seq": self.seq, **self.binding}


# --------------------------------------------------------------------------- pure rules

def days_since_delivery(as_of: Any, delivered_on: Any) -> int | None:
    try:
        return (date.fromisoformat(str(as_of)[:10]) - date.fromisoformat(str(delivered_on)[:10])).days
    except ValueError:
        return None


def find_product(state: dict, sku: Any) -> dict | None:
    return next((p for p in state["products"] if p.get("sku") == sku), None)


def find_order(state: dict, customer_id: str, order_id: Any) -> dict | None:
    """The customer's own order with that id. Order ids are not unique across customers, so a
    foreign order with the same id is invisible (review B5)."""
    return next((o for o in state["orders"]
                 if o.get("order_id") == order_id and o.get("customer_id") == customer_id), None)


def find_line(order: dict, line_id: Any) -> dict | None:
    return next((line for line in order.get("lines", []) if line.get("line_id") == line_id), None)


def line_eligibility(state: dict, customer_id: str, order_id: Any, line_id: Any) -> tuple[str, str]:
    """Return ("ok" | "deny" | "clarify", reason). Foreign and missing orders look identical."""
    order = find_order(state, customer_id, order_id)
    if order is None:
        return "deny", "ORDER_NOT_FOUND"
    line = find_line(order, line_id)
    if line is None:
        return "clarify", "LINE_NOT_FOUND"
    reasons = []
    if order.get("status") != "delivered":
        reasons.append("NOT_DELIVERED")
    days = days_since_delivery(state["as_of"], order.get("delivered_on"))
    if days is None:
        reasons.append("MISSING_DELIVERY_DATE")
    elif days < 0:
        reasons.append("DELIVERY_IN_FUTURE")
    elif days > WINDOW_DAYS:
        reasons.append("RETURN_WINDOW_EXPIRED")
    # Strict identity checks: a missing or non-boolean flag never makes a line eligible.
    if line.get("unopened") is not True:
        reasons.append("OPENED")
    if line.get("final_sale") is not False:
        reasons.append("FINAL_SALE")
    if line.get("returned") is not False:
        reasons.append("ALREADY_RETURNED")
    return ("deny", "+".join(reasons)) if reasons else ("ok", "ELIGIBLE")


def same_size(a: Any, b: Any) -> bool:
    return str(a).casefold() == str(b).casefold()


def size_rank(size: Any) -> tuple:
    text = str(size).upper()
    if text in SIZE_LADDER:
        return (0, SIZE_LADDER.index(text), "")
    try:
        return (1, float(text), "")
    except ValueError:
        return (2, 0, text)


def resolve_size(requested: str, original: dict, family: list[dict]) -> str | None:
    """Map "UP"/"DOWN" to the next offered size of the product; other sizes pass through."""
    if requested.upper() not in ("UP", "DOWN"):
        return requested
    ladder = sorted({p["size"] for p in family}, key=size_rank)
    index = ladder.index(original["size"]) + (1 if requested.upper() == "UP" else -1)
    return ladder[index] if 0 <= index < len(ladder) else None


def replacement_problem(original: dict, candidate: dict) -> str | None:
    if candidate.get("product_id") != original.get("product_id"):
        return "DIFFERENT_PRODUCT"
    if same_size(candidate.get("size"), original.get("size")):
        return "SAME_SIZE"
    if candidate.get("price_cents") != original.get("price_cents"):
        return "PRICE_MISMATCH"
    stock = candidate.get("stock")
    if not isinstance(stock, int) or stock <= 0:
        return "OUT_OF_STOCK"
    return None


def exchange_choice(state: dict, line: dict, intent: dict) -> tuple[str, str, str | None]:
    """Return (decision, reason, replacement_sku) for an exchange of an eligible line."""
    original = find_product(state, line.get("sku"))
    if original is None:
        return "deny", "SKU_NOT_FOUND", None
    family = [p for p in state["products"] if p.get("product_id") == original.get("product_id")]
    wanted_product = intent.get("requested_product")
    if wanted_product is not None and wanted_product != original.get("product_id"):
        return "deny", "DIFFERENT_PRODUCT", None
    if intent.get("requested_sku"):
        target = find_product(state, intent["requested_sku"])
        if target is None:
            return "deny", "SKU_NOT_FOUND", None
    elif intent.get("requested_size"):
        size = resolve_size(intent["requested_size"], original, family)
        if size is None:
            return "deny", "SIZE_NOT_OFFERED", None
        if same_size(size, original["size"]):
            return "deny", "SAME_SIZE", None
        target = next((p for p in family if same_size(p["size"], size)), None)
        if target is None:
            return "deny", "SIZE_NOT_OFFERED", None
    else:
        # No size named: 2+ valid sizes -> ask; exactly 1 -> propose it; none -> deny (CLARIFICATIONS #5).
        valid = [p for p in family if replacement_problem(original, p) is None]
        if len(valid) >= 2:
            return "clarify", "SIZE_UNSPECIFIED", None
        if not valid:
            return "deny", "NO_VALID_SIZE", None
        target = valid[0]
    problem = replacement_problem(original, target)
    return ("deny", problem, None) if problem else ("propose", "EXCHANGE_OK", target["sku"])


# --------------------------------------------------------------------------- conversation

class _Conversation:
    """Replays one case turn by turn, tracking proposals, the commit and injected faults."""

    def __init__(self, case: dict):
        self.state = copy.deepcopy(case["initial_state"])
        self.customer_id = case["customer_id"]
        self.request_id = case["request_id"]
        self.faults = list(case.get("faults") or [])
        self.timeout_armed = any(f.get("type") == "timeout_after_commit" for f in self.faults)
        self.proposals: list[_Proposal] = []
        self.last_presented: _Proposal | None = None  # what confirm_proposal "last" resolves to
        self.standing: dict | None = None  # latest refund/exchange request, re-read on a bare "yes"
        self.committed: _Proposal | None = None
        self.events: list[dict] = []

    def apply_faults(self, turn_no: int) -> None:
        for fault in self.faults:
            if fault.get("type") != "stock_change" or fault.get("before_turn") != turn_no:
                continue
            product = find_product(self.state, fault.get("sku"))
            self.events.append({
                "type": "stock_change", "sku": fault.get("sku"),
                "old_stock": product["stock"] if product else None, "new_stock": fault.get("new_stock"),
                "before_turn": turn_no, "applied": product is not None,
            })
            if product is not None:
                product["stock"] = fault["new_stock"]

    def handle(self, intent: dict, flag: bool) -> TurnExpectation:
        kind = intent["kind"]
        # The flag executes the bound proposal unless the text cancels or changes it (CLARIFICATIONS #9).
        if flag and self.last_presented is not None and kind != "cancel" and not self._is_change(intent):
            return self._execute(self.last_presented)
        if kind in ("refund", "exchange"):
            self.standing = intent
            return self._request(intent)
        if kind in ("confirm", "verbal_yes", "status"):
            return self._affirmation()
        if kind == "cancel":
            return TurnExpectation("clarify", "CANCELLED_BY_CUSTOMER")
        return TurnExpectation("clarify", intent.get("reason", "TARGET_NOT_UNIQUE"))

    def _is_change(self, intent: dict) -> bool:
        if intent["kind"] not in ("refund", "exchange"):
            return False
        bound = self.last_presented.binding
        target = (intent.get("order_id"), intent.get("line_id"))
        if intent["kind"] != bound["remedy"] or target != (bound["order_id"], bound["line_id"]):
            return True
        names_replacement = any(intent.get(k) for k in ("requested_size", "requested_sku", "requested_product"))
        if intent["kind"] == "exchange" and names_replacement:
            line = find_line(find_order(self.state, self.customer_id, target[0]), target[1])
            return exchange_choice(self.state, line, intent)[2] != bound["replacement_sku"]
        return False

    def _request(self, intent: dict) -> TurnExpectation:
        if self.committed is not None:
            done = self.committed.binding
            same_target = (intent.get("order_id"), intent.get("line_id")) == (done["order_id"], done["line_id"])
            # Same line AND same remedy is the request already done; the other remedy on that line is
            # a new request, refused like any other (CLARIFICATIONS #7, review B4).
            if same_target and intent["kind"] == done["remedy"]:
                return TurnExpectation("completed", "REPLAY")
            return TurnExpectation("deny", "REQUEST_ALREADY_COMPLETED")
        verdict, reason = line_eligibility(self.state, self.customer_id, intent.get("order_id"), intent.get("line_id"))
        if verdict != "ok":
            return TurnExpectation(verdict, reason)
        line = find_line(find_order(self.state, self.customer_id, intent["order_id"]), intent["line_id"])
        if intent["kind"] == "refund":
            return self._propose(intent["order_id"], line, "refund", None)
        decision, reason, replacement = exchange_choice(self.state, line, intent)
        if decision != "propose":
            return TurnExpectation(decision, reason)
        return self._propose(intent["order_id"], line, "exchange", replacement)

    def _propose(self, order_id: str, line: dict, remedy: str, replacement: str | None) -> TurnExpectation:
        binding = {
            "customer_id": self.customer_id, "request_id": self.request_id,
            "order_id": order_id, "line_id": line["line_id"], "remedy": remedy,
            "original_sku": line["sku"], "replacement_sku": replacement,
            "amount_cents": line["paid_cents"] if remedy == "refund" else 0,
            "currency": self.state.get("currency", "USD"),
        }
        open_proposals = [p for p in self.proposals if p.status == "open"]
        for proposal in open_proposals:
            if proposal.binding == binding:  # identical binding: re-present, no new ID
                self.last_presented = proposal
                return TurnExpectation("propose", "REPRESENTED", proposal.view())
        for proposal in open_proposals:
            proposal.status = "superseded"
        proposal = _Proposal(len(self.proposals) + 1, binding)
        self.proposals.append(proposal)
        self.last_presented = proposal
        return TurnExpectation("propose", "PROPOSED", proposal.view())

    def _affirmation(self) -> TurnExpectation:
        """A "yes" (or status question) that carries no usable grant."""
        if self.committed is not None:
            return TurnExpectation("completed", "REPLAY")
        if self.last_presented is not None and self.last_presented.status == "open":
            return TurnExpectation("propose", "CONFIRMATION_REQUIRED", self.last_presented.view())
        if self.standing is not None:
            return self._request(self.standing)
        return TurnExpectation("clarify", "NOTHING_TO_CONFIRM")

    def _execute(self, proposal: _Proposal) -> TurnExpectation:
        if self.committed is not None:
            # After a commit no request creates a proposal, so "last" can only be the executed one.
            return TurnExpectation("completed", "REPLAY")
        if proposal.status != "open":
            return TurnExpectation("deny", "PROPOSAL_NOT_OPEN")
        problem = self._revalidate(proposal.binding)
        if problem is not None:
            proposal.status = "invalidated"
            return TurnExpectation("deny", problem)
        self._commit(proposal)
        if self.timeout_armed:  # the commit persisted; recovery reconciles instead of re-executing
            self.timeout_armed = False
            return TurnExpectation("completed", "COMMITTED_TIMEOUT_RECONCILED")
        return TurnExpectation("completed", "COMMITTED")

    def _revalidate(self, binding: dict) -> str | None:
        """Would this exact binding still be proposed against the current state?"""
        verdict, reason = line_eligibility(self.state, self.customer_id, binding["order_id"], binding["line_id"])
        if verdict != "ok":
            return reason
        if binding["remedy"] == "exchange":
            line = find_line(find_order(self.state, self.customer_id, binding["order_id"]), binding["line_id"])
            decision, reason, _ = exchange_choice(self.state, line, {"requested_sku": binding["replacement_sku"]})
            return None if decision == "propose" else reason
        return None

    def _commit(self, proposal: _Proposal) -> None:
        """Write the SPEC section 2 records. IDs are oracle-local; graders ignore them."""
        b = proposal.binding
        line = find_line(find_order(self.state, self.customer_id, b["order_id"]), b["line_id"])
        line["returned"] = True
        common = {
            "transaction_id": "ORACLE-TX-1", "request_id": b["request_id"], "proposal_id": f"ORACLE-P-{proposal.seq}",
            "customer_id": b["customer_id"], "order_id": b["order_id"], "line_id": b["line_id"],
            "sku": b["original_sku"], "replacement_sku": b["replacement_sku"], "amount_cents": b["amount_cents"],
            "currency": b["currency"], "business_date": self.state["as_of"], "status": "completed",
        }
        self.state["returns"].append({"return_id": "ORACLE-RET-1", "remedy": b["remedy"], "disposition": "quarantine", **common})
        result = {"transaction_id": "ORACLE-TX-1", "return_id": "ORACLE-RET-1",
                  "amount_cents": b["amount_cents"], "replacement_sku": b["replacement_sku"]}
        if b["remedy"] == "refund":
            self.state["refunds"].append({"refund_id": "ORACLE-RF-1", "return_id": "ORACLE-RET-1", **common})
            result["refund_id"] = "ORACLE-RF-1"
        else:
            self.state["exchanges"].append({"exchange_id": "ORACLE-EX-1", "return_id": "ORACLE-RET-1",
                                            "original_sku": b["original_sku"], **common})
            find_product(self.state, b["replacement_sku"])["stock"] -= 1
            result["exchange_id"] = "ORACLE-EX-1"
        self.state["quarantine"].append({"quarantine_id": "ORACLE-Q-1", "return_id": "ORACLE-RET-1", "quantity": 1,
                                         "reason": "customer_return", **common})
        self.state["idempotency"][b["request_id"]] = {"action": b["remedy"], **common, "result": result}
        proposal.status = "executed"
        self.committed = proposal


# --------------------------------------------------------------------------- public API

def validate_intent(intent: dict, flag: bool) -> None:
    """Reject label mistakes early: an inconsistent label would silently grade the wrong thing."""
    kind = intent.get("kind")
    if kind not in INTENT_KINDS:
        raise ValueError(f"unknown intent kind {kind!r}")
    if kind == "confirm" and not flag:
        raise ValueError("intent 'confirm' needs confirm_proposal 'last' on its turn (use 'verbal_yes')")
    if kind == "verbal_yes" and flag:
        raise ValueError("intent 'verbal_yes' must not carry confirm_proposal (use 'confirm')")
    if kind in ("refund", "exchange") and not intent.get("order_id"):
        raise ValueError(f"intent {kind!r} needs an order_id (use 'clarify_needed' when none is grounded)")


def expected_outcome(case: dict, intents: list[dict]) -> Outcome:
    """Run the disclosed rules over a case; `intents` holds one grounded intent per turn."""
    if len(intents) != len(case["turns"]):
        raise ValueError(f"{case.get('case_id')}: {len(intents)} intents for {len(case['turns'])} turns")
    conversation = _Conversation(case)
    results = []
    for number, (turn, intent) in enumerate(zip(case["turns"], intents), start=1):
        conversation.apply_faults(number)
        flag = turn.get("confirm_proposal") == "last"
        validate_intent(intent, flag)
        results.append(conversation.handle(intent, flag))
    proposals = [{**p.view(), "status": p.status} for p in conversation.proposals]
    return Outcome(results, conversation.state, conversation.events, proposals)


def faults_only_state(case: dict) -> dict:
    """The initial state with every in-range stock_change applied: the 'nothing happened' baseline."""
    conversation = _Conversation(case)
    for number in range(1, len(case["turns"]) + 1):
        conversation.apply_faults(number)
    return conversation.state


def build_expectation(case: dict, intents: list[dict]) -> dict:
    """The label body the scorer compares against (same shape for public and variant labels)."""
    outcome = expected_outcome(case, intents)
    final_view = business_view(outcome.final_state)
    alternatives = {
        str(number): CONTESTED_ALTERNATIVES[turn.reason]
        for number, turn in enumerate(outcome.turns, start=1)
        if turn.reason in CONTESTED_ALTERNATIVES
    }
    return {
        "decisions": [turn.decision for turn in outcome.turns],
        "reasons": [turn.reason for turn in outcome.turns],
        "proposals": [turn.proposal for turn in outcome.turns],
        "events": outcome.events,
        "final_business_state": final_view,
        "state_unchanged": final_view == business_view(faults_only_state(case)),
        "alternatives": alternatives,
    }
