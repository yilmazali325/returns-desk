"""Building blocks for hand-written held-out cases: catalog, orders, intents, turns and effects.

A case spec holds the contract input (state, turns, faults) next to the answer a human decided
for it (decision, reason and proposal per turn, and the final money/stock effect), so a reviewer
can challenge any label without leaving the case definition.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CURRENCY = "USD"


# --------------------------------------------------------------------------- contract state

def product(sku: str, product_id: str, size: str, price_cents: int, stock: int) -> dict:
    return {"sku": sku, "product_id": product_id, "size": size, "price_cents": price_cents, "stock": stock}


def sized(product_id: str, price_cents: int, stock_by_size: dict[str, int]) -> list[dict]:
    """One product family at a single catalog price; SKUs follow the public `<PRODUCT>-<SIZE>` style."""
    return [product(f"{product_id}-{size}", product_id, size, price_cents, stock)
            for size, stock in stock_by_size.items()]


def line(line_id: str, sku: str, paid_cents: int, *, opened: bool = False,
         final_sale: bool = False, returned: bool = False) -> dict:
    return {"line_id": line_id, "sku": sku, "paid_cents": paid_cents, "unopened": not opened,
            "final_sale": final_sale, "returned": returned}


def order(order_id: str, customer_id: str, delivered_on: str | None, lines: list[dict],
          *, status: str = "delivered") -> dict:
    return {"order_id": order_id, "customer_id": customer_id, "status": status,
            "delivered_on": delivered_on, "lines": lines}


def state(as_of: str, products: list[dict], orders: list[dict]) -> dict:
    """A resettable initial state in the contract schema; transaction records start empty."""
    return {"as_of": as_of, "currency": CURRENCY, "products": products, "orders": orders,
            "returns": [], "refunds": [], "exchanges": [], "quarantine": [], "idempotency": {}}


def stock_change(before_turn: int, sku: str, new_stock: int) -> dict:
    return {"type": "stock_change", "before_turn": before_turn, "sku": sku, "new_stock": new_stock}


TIMEOUT = {"type": "timeout_after_commit"}


# --------------------------------------------------------------------------- intents (evals/oracle.py vocabulary)

def refund(order_id: str, line_id: str) -> dict:
    return {"kind": "refund", "order_id": order_id, "line_id": line_id}


def exchange(order_id: str, line_id: str, *, size: str | None = None, product_id: str | None = None) -> dict:
    """`size` may be a concrete size, "UP"/"DOWN" (relative), or omitted ("a different size")."""
    intent = {"kind": "exchange", "order_id": order_id, "line_id": line_id}
    if size is not None:
        intent["requested_size"] = size
    if product_id is not None:
        intent["requested_product"] = product_id
    return intent


def unclear(reason: str) -> dict:
    """The message has no unique target or remedy (or is not a request at all)."""
    return {"kind": "clarify_needed", "reason": reason}


CONFIRM = {"kind": "confirm"}        # affirmation sent WITH confirm_proposal "last"
VERBAL_YES = {"kind": "verbal_yes"}  # affirmation WITHOUT the flag
STATUS = {"kind": "status"}          # "did it go through?" / a question about the open proposal
CANCEL = {"kind": "cancel"}


# --------------------------------------------------------------------------- hand labels

@dataclass(frozen=True)
class Proposal:
    """What a propose turn must bind. `seq` is proposal identity: equal seq = same proposal ID."""
    seq: int
    order_id: str
    line_id: str
    remedy: str
    replacement_sku: str | None
    amount_cents: int


def refund_offer(seq: int, order_id: str, line_id: str, amount_cents: int) -> Proposal:
    return Proposal(seq, order_id, line_id, "refund", None, amount_cents)


def exchange_offer(seq: int, order_id: str, line_id: str, replacement_sku: str) -> Proposal:
    return Proposal(seq, order_id, line_id, "exchange", replacement_sku, 0)


@dataclass(frozen=True)
class Turn:
    message: str
    intent: dict
    decision: str
    reason: str
    confirm: bool = False  # True = the turn carries confirm_proposal "last"
    proposal: Proposal | None = None
    alternatives: tuple[str, ...] = ()  # decisions also accepted leniently (contested defaults)


def say(message: str, intent: dict, decision: str, reason: str, proposal: Proposal | None = None,
        alt: tuple[str, ...] = ()) -> Turn:
    """A customer turn without the confirmation flag."""
    return Turn(message, intent, decision, reason, False, proposal, alt)


def flagged(message: str, intent: dict, decision: str, reason: str, proposal: Proposal | None = None,
            alt: tuple[str, ...] = ()) -> Turn:
    """A customer turn that carries confirm_proposal "last"."""
    return Turn(message, intent, decision, reason, True, proposal, alt)


@dataclass(frozen=True)
class Effect:
    """The single committed transaction a case must end with."""
    remedy: str
    order_id: str
    line_id: str
    amount_cents: int = 0
    replacement_sku: str | None = None


def refunded(order_id: str, line_id: str, amount_cents: int) -> Effect:
    return Effect("refund", order_id, line_id, amount_cents)


def exchanged(order_id: str, line_id: str, replacement_sku: str) -> Effect:
    return Effect("exchange", order_id, line_id, 0, replacement_sku)


@dataclass
class CaseSpec:
    case_id: str
    family: str
    note: str
    customer_id: str
    initial_state: dict
    turns: list[Turn]
    faults: list[dict] = field(default_factory=list)
    effect: Effect | None = None  # None = no business change may remain (besides stock faults)
    stock_after: dict[str, int] = field(default_factory=dict)  # every SKU whose final stock differs
    request_id: str | None = None  # defaults to the public pattern REQ-<case_id>

    def resolved_request_id(self) -> str:
        return self.request_id or f"REQ-{self.case_id}"

    def contract_case(self) -> dict:
        """The case exactly as the adapter receives it (same keys and order as public_cases.json)."""
        turns = []
        for turn in self.turns:
            entry = {"message": turn.message}
            if turn.confirm:
                entry["confirm_proposal"] = "last"
            turns.append(entry)
        return {"case_id": self.case_id, "customer_id": self.customer_id,
                "request_id": self.resolved_request_id(), "initial_state": self.initial_state,
                "turns": turns, "faults": self.faults}
