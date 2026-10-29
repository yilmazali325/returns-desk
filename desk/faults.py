"""Fault injection at the business-tool boundary (CONTRACT "Faults", SPEC §4).

Faults change the world the tools see; they never tell an agent what to do:

- ``stock_change`` replaces a SKU's stock immediately before its 1-based ``before_turn``,
  by plain assignment (even to a negative number: validity checks, not the fault, decide
  what that means) and is recorded as an external event.
- ``timeout_after_commit`` lets the first commit persist and then reports a simulated
  TIMEOUT instead of the result. It fires exactly once per case.

Bad fault entries never crash a case: an unknown SKU produces an event with
``applied: false``; an out-of-range turn, a malformed entry or an unknown type is skipped
and listed in ``FaultPlan.skipped`` for the audit.
"""

from __future__ import annotations

import copy
from typing import Any

from desk import policy
from desk.domain import SKU_NOT_FOUND
from desk.store import Store

STOCK_CHANGE = "stock_change"
TIMEOUT_AFTER_COMMIT = "timeout_after_commit"

BEFORE_TURN_OUT_OF_RANGE = "BEFORE_TURN_OUT_OF_RANGE"
UNKNOWN_FAULT_TYPE = "UNKNOWN_FAULT_TYPE"
INVALID_FAULT = "INVALID_FAULT"


class FaultPlan:
    """The case's ``faults[]``, sorted into what will apply and when."""

    def __init__(self, faults: list[Any] | None = None, turn_count: int | None = None) -> None:
        self.turn_count = turn_count
        self.timeout_armed = False
        self.timeout_fired = False
        self.skipped: list[dict[str, Any]] = []
        self._stock_changes: list[dict[str, Any]] = []
        for index, fault in enumerate(copy.deepcopy(list(faults or []))):
            self._classify(index, fault)

    def _classify(self, index: int, fault: Any) -> None:
        kind = fault.get("type") if isinstance(fault, dict) else None
        if kind == TIMEOUT_AFTER_COMMIT:
            self.timeout_armed = True
        elif kind != STOCK_CHANGE:
            reason = UNKNOWN_FAULT_TYPE if isinstance(fault, dict) else INVALID_FAULT
            self.skipped.append({"index": index, "fault": fault, "reason": reason})
        elif not (isinstance(fault.get("sku"), str) and policy.is_int(fault.get("new_stock"))):
            self.skipped.append({"index": index, "fault": fault, "reason": INVALID_FAULT})
        elif not self._turn_in_range(fault.get("before_turn")):
            self.skipped.append({"index": index, "fault": fault, "reason": BEFORE_TURN_OUT_OF_RANGE})
        else:
            self._stock_changes.append(fault)

    def _turn_in_range(self, before_turn: Any) -> bool:
        if not policy.is_int(before_turn) or before_turn < 1:
            return False
        return self.turn_count is None or before_turn <= self.turn_count

    def stock_changes_for(self, turn_no: int) -> list[dict[str, Any]]:
        """Stock changes due before this turn, in input list order."""
        return [copy.deepcopy(f) for f in self._stock_changes if f["before_turn"] == turn_no]

    def consume_timeout(self) -> bool:
        """True exactly once: the first commit after arming. Only real commits call this,
        so replays and refused executions never use up the fault."""
        if self.timeout_armed and not self.timeout_fired:
            self.timeout_fired = True
            return True
        return False

    def arm_timeout(self) -> None:
        """Arm (or re-arm) the timeout for the next commit; used by the UI's fault button."""
        self.timeout_armed = True
        self.timeout_fired = False


def apply_stock_change(store: Store, sku: str, new_stock: int, before_turn: int | None) -> dict[str, Any]:
    """Assign the stock in one transaction and record the external event."""
    with store.transaction() as tx:
        found, old_stock = tx.set_stock(sku, new_stock)
        event: dict[str, Any] = {
            "event_id": tx.next_id("EV"),
            "type": STOCK_CHANGE,
            "sku": sku,
            "old_stock": old_stock,
            "new_stock": new_stock,
            "before_turn": before_turn,
            "applied": found,
            "source": "external",
        }
        if not found:
            event["reason"] = SKU_NOT_FOUND
        tx.add_event(event)
    return event
