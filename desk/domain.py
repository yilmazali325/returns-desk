"""Shared vocabulary of the desk: reason codes, their decision class, and the value types
that flow between policy, store, tools, agents and the orchestrator.

Everything here is plain data (frozen dataclasses with ``to_dict()``) so it serialises to
JSON for the audit, the adapter output and the UI without any custom encoder.

``CODE_CLASS`` is the single place that says what a code *means* for the turn decision
(SPEC §6): a ``refusal`` is an authoritative business "no" (deny), a ``gate`` is missing
or ambiguous information (clarify), a ``failure`` is something broken (error) and a
``format`` code is an agent mistake that the agent can correct (never changes the label).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any

# --- Remedies, actors, decisions -------------------------------------------------------

REFUND = "refund"
EXCHANGE = "exchange"
REMEDIES = (REFUND, EXCHANGE)

ACTORS = ("resolution", "operations", "orchestrator")

PROPOSE, CLARIFY, DENY, COMPLETED, ERROR = "propose", "clarify", "deny", "completed", "error"
DECISIONS = (PROPOSE, CLARIFY, DENY, COMPLETED, ERROR)

# Proposal lifecycle (SPEC §4).
OPEN, SUPERSEDED, EXECUTED, INVALIDATED = "open", "superseded", "executed", "invalidated"
PROPOSAL_STATUSES = (OPEN, SUPERSEDED, EXECUTED, INVALIDATED)

WINDOW_DAYS = 30

# --- Reason codes ----------------------------------------------------------------------

# Eligibility (policy.eligibility), in the order they are reported.
ORDER_NOT_FOUND = "ORDER_NOT_FOUND"
LINE_NOT_FOUND = "LINE_NOT_FOUND"
NOT_DELIVERED = "NOT_DELIVERED"
MISSING_DELIVERY_DATE = "MISSING_DELIVERY_DATE"
DELIVERY_IN_FUTURE = "DELIVERY_IN_FUTURE"
RETURN_WINDOW_EXPIRED = "RETURN_WINDOW_EXPIRED"
OPENED = "OPENED"
FINAL_SALE = "FINAL_SALE"
ALREADY_RETURNED = "ALREADY_RETURNED"

# Exchange validity (policy.validate_replacement / resolve_size).
SKU_NOT_FOUND = "SKU_NOT_FOUND"  # the ORIGINAL sku is not in the catalog
SIZE_NOT_OFFERED = "SIZE_NOT_OFFERED"
DIFFERENT_PRODUCT = "DIFFERENT_PRODUCT"
SAME_SIZE = "SAME_SIZE"
PRICE_MISMATCH = "PRICE_MISMATCH"
OUT_OF_STOCK = "OUT_OF_STOCK"
NO_EXCHANGE_AVAILABLE = "NO_EXCHANGE_AVAILABLE"  # no size requested and no valid option exists

# Proposal / execution flow.
PROPOSAL_NOT_OPEN = "PROPOSAL_NOT_OPEN"
REQUEST_ALREADY_COMPLETED = "REQUEST_ALREADY_COMPLETED"
BINDING_MISMATCH = "BINDING_MISMATCH"
CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
EXECUTE_LIMIT = "EXECUTE_LIMIT"  # max one execute per turn
TIMEOUT = "TIMEOUT"

# Grounding gates (orchestrator / grounding.py) and argument checks in the gateway.
TARGET_NOT_GROUNDED = "TARGET_NOT_GROUNDED"
TARGET_MISMATCH = "TARGET_MISMATCH"
AMBIGUOUS_TARGET = "AMBIGUOUS_TARGET"  # 2+ lines fit and the agent cited no words that pick one
NO_DECISION = "NO_DECISION"  # Operations ended without creating the (valid) proposal; nothing was made
SIZE_UNSPECIFIED = "SIZE_UNSPECIFIED"
REMEDY_UNCLEAR = "REMEDY_UNCLEAR"
REMEDY_MISMATCH = "REMEDY_MISMATCH"
REPLACEMENT_NOT_REQUESTED = "REPLACEMENT_NOT_REQUESTED"
NO_TARGET = "NO_TARGET"
TOOL_NOT_ALLOWED = "TOOL_NOT_ALLOWED"
INVALID_ARGS = "INVALID_ARGS"
UNKNOWN_TOOL = "UNKNOWN_TOOL"
DUPLICATE_CALL = "DUPLICATE_CALL"

# Infrastructure.
NOT_EXECUTED = "NOT_EXECUTED"  # a confirmed proposal was not carried out by the agents this turn
INTERNAL = "INTERNAL"
LLM_ERROR = "LLM_ERROR"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"

ELIGIBILITY_CODES = (
    ORDER_NOT_FOUND,
    LINE_NOT_FOUND,
    NOT_DELIVERED,
    MISSING_DELIVERY_DATE,
    DELIVERY_IN_FUTURE,
    RETURN_WINDOW_EXPIRED,
    OPENED,
    FINAL_SALE,
    ALREADY_RETURNED,
)

REFUSAL, GATE, FAILURE, FORMAT = "refusal", "gate", "failure", "format"

CODE_CLASS: dict[str, str] = {
    # refusal -> deny (only when computed by code for the grounded request)
    ORDER_NOT_FOUND: REFUSAL,
    NOT_DELIVERED: REFUSAL,
    MISSING_DELIVERY_DATE: REFUSAL,
    DELIVERY_IN_FUTURE: REFUSAL,
    RETURN_WINDOW_EXPIRED: REFUSAL,
    OPENED: REFUSAL,
    FINAL_SALE: REFUSAL,
    ALREADY_RETURNED: REFUSAL,
    SKU_NOT_FOUND: REFUSAL,
    SIZE_NOT_OFFERED: REFUSAL,
    DIFFERENT_PRODUCT: REFUSAL,
    SAME_SIZE: REFUSAL,
    PRICE_MISMATCH: REFUSAL,
    OUT_OF_STOCK: REFUSAL,
    NO_EXCHANGE_AVAILABLE: REFUSAL,
    PROPOSAL_NOT_OPEN: REFUSAL,
    REQUEST_ALREADY_COMPLETED: REFUSAL,
    # gate -> clarify (we lack a unique, grounded request)
    LINE_NOT_FOUND: GATE,
    TARGET_NOT_GROUNDED: GATE,
    TARGET_MISMATCH: GATE,
    AMBIGUOUS_TARGET: GATE,
    NO_DECISION: GATE,
    SIZE_UNSPECIFIED: GATE,
    REMEDY_UNCLEAR: GATE,
    REMEDY_MISMATCH: GATE,
    # failure -> error
    TIMEOUT: FAILURE,
    BINDING_MISMATCH: FAILURE,
    NOT_EXECUTED: FAILURE,
    INTERNAL: FAILURE,
    LLM_ERROR: FAILURE,
    BUDGET_EXHAUSTED: FAILURE,
    # format -> the agent called something wrongly; ignored once the turn ends normally
    CONFIRMATION_REQUIRED: FORMAT,
    EXECUTE_LIMIT: FORMAT,
    REPLACEMENT_NOT_REQUESTED: FORMAT,
    NO_TARGET: FORMAT,
    TOOL_NOT_ALLOWED: FORMAT,
    INVALID_ARGS: FORMAT,
    UNKNOWN_TOOL: FORMAT,
    DUPLICATE_CALL: FORMAT,
}


def code_class(code: str | None) -> str:
    """Decision class of a code. Unknown codes are failures: an unexplained code must
    surface as an error, never as a wrong deny."""
    if code is None:
        return FAILURE
    return CODE_CLASS.get(code, FAILURE)


# --- Value types ------------------------------------------------------------------------


@dataclass(frozen=True)
class LineSummary:
    """What Resolution may see about a line: identity and product, no policy fields
    (status, dates, unopened, final_sale, returned stay with Operations)."""

    order_id: str
    line_id: str
    sku: str
    product_id: str | None
    size: str | None
    aliases: list[str]
    paid_cents: int
    delivered_on: str | None = None  # lets the agent resolve a delivery-date reference; no verdict

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    reasons: list[str]
    order_id: str
    line_id: str
    sku: str | None
    product_id: str | None
    size: str | None
    paid_cents: int | None
    delivered_on: str | None
    as_of: str
    days_since_delivery: int | None
    window_days: int = WINDOW_DAYS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExchangeOption:
    sku: str
    size: str
    price_cents: Any
    stock: Any
    valid: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroundedTarget:
    """The order line Operations acts on. Set by the orchestrator from a validated
    handoff; agents never pass these IDs to Ops tools themselves.

    ``grounded_by`` says why the line was accepted (see grounding.TargetCheck) and
    ``evidence`` carries the customer's words the agent cited, so the audit shows whose
    reading picked the item."""

    order_id: str
    line_id: str
    intent: str  # "refund" | "exchange"
    requested_sku: str | None = None
    requested_size: str | None = None
    requested_product: str | None = None
    grounded_by: str | None = None
    evidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_binding(
    *,
    customer_id: str,
    order_id: str,
    line_id: str,
    remedy: str,
    amount_cents: int,
    replacement_sku: str | None,
    request_id: str,
) -> dict[str, Any]:
    """The facts a confirmation binds (CONTRACT: customer, line, remedy, amount, replacement)."""
    return {
        "customer_id": customer_id,
        "order_id": order_id,
        "line_id": line_id,
        "remedy": remedy,
        "amount_cents": amount_cents,
        "replacement_sku": replacement_sku,
        "request_id": request_id,
    }


def binding_hash(binding: dict[str, Any]) -> str:
    """sha256 of canonical JSON, so the same facts always give the same hash."""
    canonical = json.dumps(binding, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    request_id: str
    customer_id: str
    order_id: str
    line_id: str
    remedy: str
    original_sku: str
    replacement_sku: str | None
    amount_cents: int
    currency: str
    eligibility: dict[str, Any]
    binding_hash: str
    status: str
    created_turn: int

    def binding(self) -> dict[str, Any]:
        return make_binding(
            customer_id=self.customer_id,
            order_id=self.order_id,
            line_id=self.line_id,
            remedy=self.remedy,
            amount_cents=self.amount_cents,
            replacement_sku=self.replacement_sku,
            request_id=self.request_id,
        )

    def with_status(self, status: str) -> Proposal:
        if status not in PROPOSAL_STATUSES:
            raise ValueError(f"unknown proposal status {status!r}")
        return replace(self, status=status)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Proposal:
        return cls(**data)


@dataclass(frozen=True)
class ToolResult:
    """What every gateway call returns to an agent: ``data`` on success, otherwise
    ``error = {code, message, details?}``."""

    ok: bool
    data: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    @classmethod
    def success(cls, data: dict[str, Any]) -> ToolResult:
        return cls(ok=True, data=data, error=None)

    @classmethod
    def failure(cls, code: str, message: str, details: dict[str, Any] | None = None) -> ToolResult:
        error: dict[str, Any] = {"code": code, "message": message}
        if details:
            error["details"] = details
        return cls(ok=False, data=None, error=error)

    @property
    def code(self) -> str | None:
        return self.error["code"] if self.error else None

    @property
    def is_replay(self) -> bool:
        return bool(self.ok and self.data and self.data.get("replay"))

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "data": self.data, "error": self.error}


@dataclass(frozen=True)
class OpsReport:
    """Operations' result for one handoff, built by code from tool results (SPEC §6.6)."""

    intent: str
    outcome: str
    reason_codes: list[str] = field(default_factory=list)
    proposal: dict[str, Any] | None = None
    transaction: dict[str, Any] | None = None
    replay: bool = False
    reconciled: bool = False
    alternatives: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TurnOutput:
    """One entry of ``result.turns[]``. A proposal object is present iff the decision is
    ``propose`` (CLARIFICATIONS #10); the constructor refuses anything else so a wiring
    bug fails loudly in tests instead of shipping a wrong label."""

    decision: str
    response: str
    proposal: dict[str, Any] | None = None
    transaction: dict[str, Any] | None = None
    confirmed_proposal_id: str | None = None

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"unknown decision {self.decision!r}")
        if (self.decision == PROPOSE) != (self.proposal is not None):
            raise ValueError("proposal must be an object iff decision == 'propose'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
