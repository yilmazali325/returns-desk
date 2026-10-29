"""Fault plan tests: what applies when, list order, unknown SKUs and types, out-of-range
turns, and the once-only timeout."""

from __future__ import annotations

import pytest

from desk.domain import SKU_NOT_FOUND
from desk.faults import (
    BEFORE_TURN_OUT_OF_RANGE,
    INVALID_FAULT,
    UNKNOWN_FAULT_TYPE,
    FaultPlan,
    apply_stock_change,
)
from desk.store import Store


def _stock(sku: str, new_stock: int, before_turn: object = 2) -> dict:
    return {"type": "stock_change", "sku": sku, "new_stock": new_stock, "before_turn": before_turn}


# --- classification ---------------------------------------------------------------------


def test_valid_stock_changes_are_scheduled_in_list_order():
    plan = FaultPlan([_stock("A", 1, 2), _stock("B", 2, 1), _stock("A", 0, 2)], turn_count=3)
    assert plan.stock_changes_for(2) == [_stock("A", 1, 2), _stock("A", 0, 2)]
    assert plan.stock_changes_for(1) == [_stock("B", 2, 1)]
    assert plan.stock_changes_for(3) == [] and plan.skipped == []


@pytest.mark.parametrize("before_turn", [0, -1, 4, "2", True, None, 2.0])
def test_out_of_range_before_turn_never_applies(before_turn):
    plan = FaultPlan([_stock("A", 1, before_turn)], turn_count=3)
    assert all(plan.stock_changes_for(n) == [] for n in range(-1, 6))
    assert [s["reason"] for s in plan.skipped] == [BEFORE_TURN_OUT_OF_RANGE]


def test_missing_before_turn_never_applies():
    fault = {"type": "stock_change", "sku": "A", "new_stock": 1}
    assert FaultPlan([fault], turn_count=2).skipped[0]["reason"] == BEFORE_TURN_OUT_OF_RANGE


def test_without_a_turn_count_any_positive_turn_is_in_range():
    assert FaultPlan([_stock("A", 1, 99)]).stock_changes_for(99) == [_stock("A", 1, 99)]


@pytest.mark.parametrize(
    "fault",
    [
        {"type": "stock_change", "sku": "A", "new_stock": "0", "before_turn": 1},
        {"type": "stock_change", "sku": "A", "new_stock": False, "before_turn": 1},
        {"type": "stock_change", "new_stock": 0, "before_turn": 1},
        {"type": "stock_change", "sku": 7, "new_stock": 0, "before_turn": 1},
    ],
)
def test_malformed_stock_changes_are_skipped(fault):
    plan = FaultPlan([fault], turn_count=2)
    assert plan.stock_changes_for(1) == [] and plan.skipped[0]["reason"] == INVALID_FAULT


def test_unknown_types_and_non_objects_are_recorded_and_ignored():
    plan = FaultPlan([{"type": "price_change", "sku": "A"}, {"sku": "A"}, "boom", None], turn_count=2)
    assert [s["reason"] for s in plan.skipped] == [UNKNOWN_FAULT_TYPE, UNKNOWN_FAULT_TYPE, INVALID_FAULT, INVALID_FAULT]
    assert [s["index"] for s in plan.skipped] == [0, 1, 2, 3]
    assert not plan.timeout_armed


def test_plan_is_isolated_from_the_input_list():
    faults = [_stock("A", 1, 1)]
    plan = FaultPlan(faults, turn_count=1)
    faults[0]["new_stock"] = 50
    plan.stock_changes_for(1)[0]["new_stock"] = 60
    assert plan.stock_changes_for(1)[0]["new_stock"] == 1


def test_no_faults():
    plan = FaultPlan()
    assert plan.stock_changes_for(1) == [] and plan.skipped == [] and plan.consume_timeout() is False


# --- timeout ------------------------------------------------------------------------------


def test_timeout_fires_exactly_once():
    plan = FaultPlan([{"type": "timeout_after_commit"}], turn_count=3)
    assert plan.timeout_armed and not plan.timeout_fired
    assert [plan.consume_timeout() for _ in range(3)] == [True, False, False]
    assert plan.timeout_fired


def test_duplicate_timeout_entries_still_fire_once():
    plan = FaultPlan([{"type": "timeout_after_commit"}, {"type": "timeout_after_commit"}])
    assert [plan.consume_timeout(), plan.consume_timeout()] == [True, False]


def test_arm_timeout_rearms_for_the_ui():
    plan = FaultPlan()
    assert plan.consume_timeout() is False
    plan.arm_timeout()
    assert [plan.consume_timeout(), plan.consume_timeout()] == [True, False]
    plan.arm_timeout()
    assert plan.consume_timeout() is True


# --- applying a stock change -------------------------------------------------------------------


def test_apply_stock_change_assigns_and_records_an_external_event(state, lookup):
    store = Store(state)
    event = apply_stock_change(store, "HOODIE-L", 0, 2)
    assert event == {
        "event_id": "EV-0001",
        "type": "stock_change",
        "sku": "HOODIE-L",
        "old_stock": 3,
        "new_stock": 0,
        "before_turn": 2,
        "applied": True,
        "source": "external",
    }
    assert lookup.product(store.export(), "HOODIE-L")["stock"] == 0
    assert store.events() == [event] and store.ledger() == []  # external, not a business action
    assert store.version() == 2


def test_negative_stock_from_a_fault_is_assigned_as_given(state, lookup):
    store = Store(state)
    apply_stock_change(store, "HOODIE-L", -3, 1)
    assert lookup.product(store.export(), "HOODIE-L")["stock"] == -3


def test_unknown_sku_is_recorded_as_not_applied(state):
    store = Store(state)
    event = apply_stock_change(store, "NOPE-1", 5, 1)
    assert (event["applied"], event["reason"], event["old_stock"]) == (False, SKU_NOT_FOUND, None)
    assert store.export() == state and store.version() == 1 and store.events() == [event]


# --- through the gateway ------------------------------------------------------------------------


def test_multiple_faults_before_one_turn_apply_in_list_order(make_tools, lookup):
    faults = [_stock("HOODIE-L", 0, 2), _stock("HOODIE-S", 9, 2), _stock("HOODIE-L", 7, 2)]
    tools = make_tools(faults=faults, turn_count=2)
    assert tools.begin_turn(1) == []
    events = tools.begin_turn(2)
    assert [(e["sku"], e["old_stock"], e["new_stock"]) for e in events] == [
        ("HOODIE-L", 3, 0),
        ("HOODIE-S", 2, 9),
        ("HOODIE-L", 0, 7),
    ]
    assert [e["event_id"] for e in tools.external_events()] == ["EV-0001", "EV-0002", "EV-0003"]
    assert lookup.product(tools.final_state(), "HOODIE-L")["stock"] == 7
    fault_entries = [e for e in tools.audit.entries() if e["kind"] == "fault"]
    assert [e["turn"] for e in fault_entries] == [2, 2, 2]
    assert all(e["visible_to_agent"] is False and e["agent"] == "external" for e in fault_entries)


def test_a_turn_applies_its_faults_only_once(make_tools, lookup):
    tools = make_tools(faults=[_stock("HOODIE-L", 1, 1)], turn_count=1)
    assert len(tools.begin_turn(1)) == 1
    assert tools.begin_turn(1) == [] and tools.apply_turn_faults(1) == []
    assert len(tools.external_events()) == 1


def test_unknown_sku_through_the_gateway(make_tools, make_state):
    tools = make_tools(faults=[_stock("NOPE-1", 0, 1)], turn_count=1)
    (event,) = tools.begin_turn(1)
    assert event["applied"] is False and tools.final_state() == make_state()


def test_skipped_faults_and_armed_timeout_are_noted_in_the_audit_before_turn_1(make_tools):
    faults = [_stock("HOODIE-L", 0, 5), {"type": "price_change"}, "junk", {"type": "timeout_after_commit"}]
    tools = make_tools(faults=faults, turn_count=2)
    entries = tools.audit.entries()
    assert [(e["turn"], e["tool"], (e["error"] or {}).get("code")) for e in entries] == [
        (0, "stock_change", BEFORE_TURN_OUT_OF_RANGE),
        (0, "price_change", UNKNOWN_FAULT_TYPE),
        (0, "invalid_fault", INVALID_FAULT),
        (0, "timeout_after_commit", None),
    ]
    assert entries[-1]["result"] == {"armed": True}
    assert tools.begin_turn(1) == [] and tools.begin_turn(2) == []  # out-of-range fault never applies
