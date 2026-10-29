"""Gateway tests: identity injection, the idempotency short-circuit, proposal binding and
supersession, execute with grant/binding/revalidation, duplicate prevention, the
timeout-after-commit fault, the audit trail, and the shared domain types."""

from __future__ import annotations

import copy
import json

import pytest

from desk import domain, policy
from desk.audit import AuditLog
from desk.domain import (
    ALREADY_RETURNED,
    BINDING_MISMATCH,
    CONFIRMATION_REQUIRED,
    DIFFERENT_PRODUCT,
    EXECUTE_LIMIT,
    FAILURE,
    INTERNAL,
    INVALID_ARGS,
    LINE_NOT_FOUND,
    NO_EXCHANGE_AVAILABLE,
    NO_TARGET,
    ORDER_NOT_FOUND,
    OUT_OF_STOCK,
    PRICE_MISMATCH,
    PROPOSAL_NOT_OPEN,
    REMEDY_MISMATCH,
    REPLACEMENT_NOT_REQUESTED,
    REQUEST_ALREADY_COMPLETED,
    RETURN_WINDOW_EXPIRED,
    SAME_SIZE,
    SIZE_NOT_OFFERED,
    SIZE_UNSPECIFIED,
    SKU_NOT_FOUND,
    TIMEOUT,
    TOOL_NOT_ALLOWED,
    OpsReport,
    Proposal,
    ToolResult,
    TurnOutput,
    binding_hash,
    code_class,
)
from desk.store import Store
from desk.tools import BusinessTools, default_aliases, normalize_ref

TIMEOUT_FAULT = {"type": "timeout_after_commit"}


def _propose(tools: BusinessTools, remedy: str = "refund", line_id: str = "X9", order_id: str = "ORD-55", **requested):
    tools.set_target(order_id, line_id, remedy, **requested)
    return tools.create_proposal(remedy)


def _confirm(tools: BusinessTools, turn: int, proposal_id: str) -> ToolResult:
    tools.begin_turn(turn)
    tools.set_grant(proposal_id)
    return tools.execute_proposal()


def _last_audit(tools: BusinessTools, tool: str) -> dict:
    return [e for e in tools.audit.entries() if e["tool"] == tool][-1]


# === domain types ===========================================================================


def test_every_reason_code_has_a_decision_class():
    codes = [value for name, value in vars(domain).items() if name.isupper() and isinstance(value, str) and value == name]
    assert len(codes) > 30
    assert [code for code in codes if code not in domain.CODE_CLASS] == []
    assert set(domain.CODE_CLASS.values()) == {"refusal", "gate", "failure", "format"}


@pytest.mark.parametrize(
    ("code", "klass"),
    [
        (RETURN_WINDOW_EXPIRED, "refusal"),
        (ORDER_NOT_FOUND, "refusal"),
        (OUT_OF_STOCK, "refusal"),
        (REQUEST_ALREADY_COMPLETED, "refusal"),
        (LINE_NOT_FOUND, "gate"),
        (SIZE_UNSPECIFIED, "gate"),
        (TIMEOUT, "failure"),
        (BINDING_MISMATCH, "failure"),
        (CONFIRMATION_REQUIRED, "format"),
        (REPLACEMENT_NOT_REQUESTED, "format"),
        ("SOMETHING_NEW", FAILURE),
        (None, FAILURE),
    ],
)
def test_code_class(code, klass):
    assert code_class(code) == klass


def test_tool_result_helpers():
    ok = ToolResult.success({"replay": True})
    bad = ToolResult.failure("X", "msg")
    detailed = ToolResult.failure("Y", "msg", {"reasons": ["Y"]})
    assert (ok.code, ok.is_replay, bad.code, bad.is_replay) == (None, True, "X", False)
    assert bad.to_dict() == {"ok": False, "data": None, "error": {"code": "X", "message": "msg"}}
    assert detailed.error["details"] == {"reasons": ["Y"]}
    assert ToolResult.success({}).is_replay is False


def test_turn_output_enforces_the_proposal_rule():
    proposal = {"proposal_id": "P-0001"}
    assert TurnOutput("propose", "ok?", proposal).to_dict()["proposal"] == proposal
    assert TurnOutput("completed", "done", None, {"transaction_id": "TX-0001"}, "P-0001").to_dict()["confirmed_proposal_id"] == "P-0001"
    with pytest.raises(ValueError, match="iff"):
        TurnOutput("deny", "no", proposal)
    with pytest.raises(ValueError, match="iff"):
        TurnOutput("propose", "ok?", None)
    with pytest.raises(ValueError, match="unknown decision"):
        TurnOutput("maybe", "?")


def test_ops_report_and_proposal_round_trip():
    report = OpsReport(intent="refund", outcome="proposed", reason_codes=[], alternatives=[{"sku": "A"}])
    assert report.to_dict()["alternatives"] == [{"sku": "A"}] and report.to_dict()["replay"] is False
    with pytest.raises(ValueError, match="unknown proposal status"):
        _sample_proposal().with_status("done")
    assert Proposal.from_dict(_sample_proposal().to_dict()) == _sample_proposal()


def test_binding_hash_is_canonical():
    a = {"customer_id": "C7", "order_id": "O", "line_id": "L", "remedy": "refund", "amount_cents": 1, "replacement_sku": None, "request_id": "R"}
    b = dict(reversed(list(a.items())))
    assert binding_hash(a) == binding_hash(b) and len(binding_hash(a)) == 64
    assert binding_hash({**a, "amount_cents": 2}) != binding_hash(a)


def _sample_proposal() -> Proposal:
    return Proposal("P-1", "R", "C", "O", "L", "refund", "S", None, 1, "USD", {}, "h", "open", 1)


# === audit log ==============================================================================


def test_audit_entries_are_immutable_snapshots_and_listeners_get_copies():
    audit = AuditLog()
    seen: list[dict] = []
    unsubscribe = audit.subscribe(seen.append)
    arguments = {"x": [1]}
    entry = audit.record(kind="tool", agent="operations", tool="t", arguments=arguments)
    arguments["x"].append(2)
    entry["tool"] = "changed"
    seen[0]["tool"] = "changed too"
    audit.entries()[0]["tool"] = "changed again"
    assert audit.entries()[0]["arguments"] == {"x": [1]} and audit.entries()[0]["tool"] == "t"
    assert entry["seq"] == 1 and entry["committed_effects"] == [] and entry["visible_to_agent"] is True
    unsubscribe()
    unsubscribe()  # idempotent
    audit.record(kind="tool", agent="operations", tool="t2")
    assert len(seen) == 1 and len(audit.entries()) == 2


def test_a_failing_listener_never_breaks_the_business_call(make_tools):
    audit = AuditLog()

    def broken(_entry: dict) -> None:
        raise RuntimeError("viewer crashed")

    audit.subscribe(broken)
    tools = make_tools(audit=audit)
    assert tools.find_order_lines().ok and audit.listener_errors == 1


def test_audit_entries_carry_the_turn(make_tools):
    tools = make_tools()
    tools.begin_turn(3)
    tools.find_order_lines()
    assert _last_audit(tools, "find_order_lines")["turn"] == 3


# === Resolution tools =======================================================================


def test_find_order_lines_returns_only_this_customers_lines_without_policy_fields(make_tools):
    result = make_tools().find_order_lines()
    assert result.ok and result.data["count"] == 2
    assert result.data["matches"][0] == {
        "order_id": "ORD-55",
        "line_id": "X9",
        "sku": "HOODIE-M",
        "product_id": "HOODIE",
        "size": "M",
        "aliases": ["hoodie"],
        "paid_cents": 4200,
        "delivered_on": "2026-03-01",  # a fact for "the one I got last month"; no eligibility verdict
    }
    assert {m["line_id"] for m in result.data["matches"]} == {"X9", "X10"}


def test_line_summaries_normalise_timestamps_and_hide_bad_dates(make_tools, make_state):
    state = make_state()
    state["orders"][0]["delivered_on"] = "2026-03-01T22:10:00Z"
    assert make_tools(state).customer_lines()[0]["delivered_on"] == "2026-03-01"
    state["orders"][0]["delivered_on"] = "yesterday"
    assert make_tools(state).customer_lines()[0]["delivered_on"] is None


@pytest.mark.parametrize("ref", ["ORD-55", "#ord 55", "ord_55", "Ord55"])
def test_find_order_lines_normalises_the_order_reference(make_tools, ref):
    assert make_tools().find_order_lines(ref).data["count"] == 2


@pytest.mark.parametrize("ref", [None, "", "all", "ALL", "*"])
def test_find_order_lines_all(make_tools, ref):
    assert make_tools().find_order_lines(ref).data["count"] == 2


def test_foreign_order_lookup_is_identical_to_a_missing_one(make_tools):
    tools = make_tools()
    foreign, missing = tools.find_order_lines("ORD-66"), tools.find_order_lines("ORD-404")
    assert foreign.data == missing.data == {"matches": [], "count": 0, "order_not_on_account": True}


def test_aliases_come_from_the_injected_function(make_tools, state):
    state["orders"][0]["lines"][1]["sku"] = "GONE-1"
    tools = make_tools(state, alias_fn=lambda pid: [pid.lower(), f"{pid.lower()}s"])
    x9, x10 = tools.find_order_lines().data["matches"]
    assert x9["aliases"] == ["hoodie", "hoodies"]
    assert (x10["product_id"], x10["size"], x10["aliases"]) == (None, None, [])
    assert default_aliases("TEE") == ["tee"] and normalize_ref("#O-1_0 0") == "o100"


def test_account_card_is_an_audited_orchestrator_prefetch(make_tools):
    tools = make_tools()
    card = tools.account_card()
    assert card.data["count"] == 2 and len(card.data["lines"]) == 2
    assert [p["sku"] for p in card.data["catalog"]] == ["HOODIE-XS", "HOODIE-S", "HOODIE-M", "HOODIE-L", "HOODIE-XL", "CAP-M"]
    assert set(card.data["catalog"][0]) == {"sku", "product_id", "size"}  # no price or stock
    entry = _last_audit(tools, "prefetch_account_card")
    assert entry["agent"] == "orchestrator" and entry["result"] == card.data


def test_actor_is_validated_and_recorded(make_tools):
    tools = make_tools()
    tools.find_order_lines(actor="orchestrator")
    assert _last_audit(tools, "find_order_lines")["agent"] == "orchestrator"
    with pytest.raises(ValueError, match="unknown actor"):
        tools.find_order_lines(actor="llm")


# === targets and grants =====================================================================


def test_set_target_accepts_handoff_intent_names_and_rejects_others(make_tools):
    tools = make_tools()
    target = tools.set_target("ORD-55", "X9", "propose_exchange", requested_size="L")
    assert (target.intent, tools.target, target.requested_size) == ("exchange", target, "L")
    assert _last_audit(tools, "set_target")["arguments"]["intent"] == "exchange"
    with pytest.raises(ValueError, match="intent"):
        tools.set_target("ORD-55", "X9", "cancel")
    cited = tools.set_target("ORD-55", "X9", "refund", grounded_by="agent", evidence="my hoodie")
    assert _last_audit(tools, "set_target")["arguments"] | {} == {**cited.to_dict()}
    assert (cited.grounded_by, cited.evidence) == ("agent", "my hoodie")


def test_begin_turn_resets_target_and_grant(make_tools):
    tools = make_tools()
    tools.set_target("ORD-55", "X9", "refund")
    tools.set_grant("P-0001")
    tools.begin_turn(2)
    assert (tools.target, tools.grant, tools.turn) == (None, None, 2)
    tools.set_grant(None)
    assert [e["tool"] for e in tools.audit.entries()].count("set_grant") == 1


# === check_eligibility / list_exchange_options =============================================


def test_check_eligibility_needs_a_target(make_tools):
    assert make_tools().check_eligibility().code == NO_TARGET


def test_check_eligibility_is_informational(make_tools, state):
    state["orders"][0]["delivered_on"] = "2026-01-01"
    tools = make_tools(state)
    tools.set_target("ORD-55", "X9", "refund")
    result = tools.check_eligibility()
    assert result.ok and result.data["eligible"] is False and result.data["reasons"] == [RETURN_WINDOW_EXPIRED]


def test_list_exchange_options_reports_options_and_the_requested_size(make_tools):
    tools = make_tools()
    tools.set_target("ORD-55", "X9", "exchange", requested_size="large")
    data = tools.list_exchange_options().data
    assert (data["original_sku"], data["valid_count"], data["requested_sku"], data["requested_code"]) == ("HOODIE-M", 2, "HOODIE-L", None)
    assert [o["sku"] for o in data["options"]] == ["HOODIE-XS", "HOODIE-S", "HOODIE-L", "HOODIE-XL"] and data["reasons"] == []


def test_list_exchange_options_edge_cases(make_tools, state):
    state["orders"][0]["lines"][0]["sku"] = "GONE-1"
    tools = make_tools(state)
    assert tools.list_exchange_options().code == NO_TARGET
    tools.set_target("ORD-55", "X9", "exchange", requested_size="L")
    data = tools.list_exchange_options().data
    assert (data["options"], data["reasons"], data["requested_code"]) == ([], [SKU_NOT_FOUND], SKU_NOT_FOUND)
    tools.set_target("ORD-55", "X404", "exchange")
    assert tools.list_exchange_options().data == {"original_sku": None, "options": [], "valid_count": 0, "reasons": [LINE_NOT_FOUND]}
    tools.set_target("ORD-66", "Z1", "exchange")
    assert tools.list_exchange_options().data["reasons"] == [ORDER_NOT_FOUND]


# === create_proposal: refund =================================================================


def test_refund_proposal_binds_paid_cents_and_the_session_identity(make_tools):
    tools = make_tools()
    tools.begin_turn(1)
    result = _propose(tools)
    proposal = result.data["proposal"]
    assert result.ok and result.data["reused"] is False and result.data["superseded"] is None
    assert proposal == {
        "proposal_id": "P-0001",
        "request_id": "REQ-T1",
        "customer_id": "C7",
        "order_id": "ORD-55",
        "line_id": "X9",
        "remedy": "refund",
        "original_sku": "HOODIE-M",
        "replacement_sku": None,
        "amount_cents": 4200,
        "currency": "USD",
        "eligibility": {"delivered_on": "2026-03-01", "as_of": "2026-03-15", "days_since_delivery": 14, "window_days": 30},
        "binding_hash": Proposal.from_dict(proposal).binding_hash,
        "status": "open",
        "created_turn": 1,
    }
    assert binding_hash(Proposal.from_dict(proposal).binding()) == proposal["binding_hash"]
    assert tools.open_proposal() == proposal and tools.get_proposal("P-0001") == proposal and tools.get_proposal("P-9") is None


def test_identical_binding_returns_the_same_open_proposal(make_tools):
    tools = make_tools()
    first, again = _propose(tools), _propose(tools)
    assert again.data["reused"] is True and again.data["proposal"] == first.data["proposal"]
    assert [p["proposal_id"] for p in tools.proposals()] == ["P-0001"]


def test_a_different_binding_supersedes_the_open_proposal(make_tools):
    tools = make_tools()
    _propose(tools)
    second = _propose(tools, "exchange", requested_size="L")
    assert second.data["superseded"] == "P-0001" and second.data["proposal"]["proposal_id"] == "P-0002"
    assert [(p["proposal_id"], p["status"]) for p in tools.proposals()] == [("P-0001", "superseded"), ("P-0002", "open")]


@pytest.mark.parametrize(
    ("setup", "remedy", "replacement", "code"),
    [
        (None, "refund", None, NO_TARGET),
        (("ORD-55", "X9", "refund"), "Refund", None, INVALID_ARGS),
        (("ORD-55", "X9", "exchange"), "refund", None, REMEDY_MISMATCH),
        (("ORD-55", "X9", "refund"), "refund", "HOODIE-L", INVALID_ARGS),
        (("ORD-55", "X404", "refund"), "refund", None, LINE_NOT_FOUND),
    ],
)
def test_create_proposal_argument_errors(make_tools, setup, remedy, replacement, code):
    tools = make_tools()
    if setup:
        tools.set_target(*setup)
    result = tools.create_proposal(remedy, replacement)
    assert result.code == code and tools.proposals() == []


def test_ineligible_line_is_refused_with_every_reason(make_tools, state):
    state["orders"][0]["delivered_on"] = "2026-01-01"
    state["orders"][0]["lines"][0]["unopened"] = False
    result = _propose(make_tools(state))
    assert result.code == RETURN_WINDOW_EXPIRED
    assert result.error["details"]["reasons"] == [RETURN_WINDOW_EXPIRED, "OPENED"]
    assert result.error["details"]["eligibility"]["days_since_delivery"] == 73


def test_foreign_and_missing_orders_get_the_same_refusal(make_tools):
    tools = make_tools()
    foreign = _propose(tools, order_id="ORD-66", line_id="Z1")
    missing = _propose(tools, order_id="ORD-404", line_id="Z1")
    assert foreign.code == missing.code == ORDER_NOT_FOUND
    assert foreign.error["message"] == missing.error["message"]
    strip = lambda e: {**e["details"]["eligibility"], "order_id": None}  # noqa: E731
    assert strip(foreign.error) == strip(missing.error)
    assert "5000" not in json.dumps(foreign.to_dict()) and "HOODIE-S" not in json.dumps(foreign.to_dict())


def test_no_new_proposal_on_a_confirmation_turn(make_tools):
    tools = make_tools()
    tools.set_grant("P-0001")
    assert _propose(tools).code == TOOL_NOT_ALLOWED and tools.proposals() == []


# === create_proposal: exchange =================================================================


@pytest.mark.parametrize(
    ("requested", "replacement"),
    [
        ({"requested_sku": "HOODIE-L"}, None),
        ({"requested_size": "L"}, None),
        ({"requested_size": "large"}, "HOODIE-L"),
        ({"requested_size": "UP"}, None),
        ({"requested_size": "L", "requested_product": "hoodie"}, None),
    ],
)
def test_exchange_proposal_for_the_requested_size(make_tools, requested, replacement):
    tools = make_tools()
    tools.set_target("ORD-55", "X9", "exchange", **requested)
    proposal = tools.create_proposal("exchange", replacement).data["proposal"]
    assert (proposal["remedy"], proposal["replacement_sku"], proposal["amount_cents"]) == ("exchange", "HOODIE-L", 0)


@pytest.mark.parametrize(
    ("requested", "code"),
    [
        ({"requested_size": "M"}, SAME_SIZE),
        ({"requested_size": "M", "requested_product": "CAP"}, DIFFERENT_PRODUCT),
        ({"requested_sku": "CAP-M"}, DIFFERENT_PRODUCT),
        ({"requested_size": "XL"}, PRICE_MISMATCH),
        ({"requested_size": "XS"}, PRICE_MISMATCH),  # equals the paid price, not the catalog price
        ({"requested_size": "XXL"}, SIZE_NOT_OFFERED),
    ],
)
def test_impossible_requested_exchanges_are_refused_with_alternatives(make_tools, requested, code):
    tools = make_tools()
    tools.set_target("ORD-55", "X9", "exchange", **requested)
    result = tools.create_proposal("exchange")
    assert result.code == code and domain.code_class(code) == "refusal"
    assert result.error["details"]["alternatives"] == [{"sku": "HOODIE-S", "size": "S"}, {"sku": "HOODIE-L", "size": "L"}]
    assert tools.proposals() == []


def test_requested_size_out_of_stock(make_tools, state, lookup):
    lookup.product(state, "HOODIE-L")["stock"] = 0
    tools = make_tools(state)
    result = _propose(tools, "exchange", requested_size="L")
    assert result.code == OUT_OF_STOCK and result.error["details"]["alternatives"] == [{"sku": "HOODIE-S", "size": "S"}]


def test_agent_cannot_pick_a_size_the_customer_did_not_ask_for(make_tools):
    tools = make_tools()
    tools.set_target("ORD-55", "X9", "exchange", requested_size="L")
    result = tools.create_proposal("exchange", "HOODIE-S")
    assert result.code == REPLACEMENT_NOT_REQUESTED and domain.code_class(result.code) == "format"


def test_exchange_without_a_size_and_several_options_needs_clarification(make_tools):
    tools = make_tools()
    tools.set_target("ORD-55", "X9", "exchange")
    result = tools.create_proposal("exchange", "HOODIE-L")  # the agent guessing does not help
    assert result.code == SIZE_UNSPECIFIED and len(result.error["details"]["alternatives"]) == 2


def test_exchange_without_a_size_and_exactly_one_option_proposes_it(make_tools, state, lookup):
    lookup.product(state, "HOODIE-S")["stock"] = 0
    tools = make_tools(state)
    assert _propose(tools, "exchange").data["proposal"]["replacement_sku"] == "HOODIE-L"
    tools.set_target("ORD-55", "X9", "exchange")
    assert tools.create_proposal("exchange", "HOODIE-S").code == REPLACEMENT_NOT_REQUESTED


def test_exchange_without_a_size_and_no_option_is_refused(make_tools, state, lookup):
    for sku in ("HOODIE-S", "HOODIE-L"):
        lookup.product(state, sku)["stock"] = 0
    assert _propose(make_tools(state), "exchange").code == NO_EXCHANGE_AVAILABLE


def test_unknown_original_sku_blocks_exchange_but_not_refund(make_tools, state):
    state["orders"][0]["lines"][0]["sku"] = "GONE-1"
    tools = make_tools(state)
    assert _propose(tools, "exchange").code == SKU_NOT_FOUND
    assert _propose(tools, "exchange", requested_size="L").code == SKU_NOT_FOUND
    refund = _propose(tools).data["proposal"]
    assert (refund["original_sku"], refund["amount_cents"]) == ("GONE-1", 4200)


# === execute_proposal ===========================================================================


def test_execute_requires_an_explicit_grant(make_tools, make_state):
    tools = make_tools()
    _propose(tools)
    result = tools.execute_proposal()
    assert result.code == CONFIRMATION_REQUIRED and tools.final_state() == make_state()


def test_execute_unknown_or_superseded_proposal_is_not_open(make_tools, make_state):
    tools = make_tools()
    _propose(tools)
    _propose(tools, "exchange", requested_size="L")  # P-0001 superseded
    assert _confirm(tools, 2, "P-0404").error["details"]["status"] is None
    superseded = _confirm(tools, 3, "P-0001")
    assert superseded.code == PROPOSAL_NOT_OPEN and superseded.error["details"]["status"] == "superseded"
    assert tools.final_state() == make_state()


def test_confirmed_refund_commits_exactly_once(make_tools, lookup, make_state):
    tools = make_tools()
    _propose(tools)
    result = _confirm(tools, 2, "P-0001")
    transaction = result.data["transaction"]
    assert result.ok and result.data["replay"] is False and result.data["status"] == "executed"
    final = tools.final_state()
    assert final["idempotency"]["REQ-T1"] == transaction == tools.completed_record()
    assert [r["amount_cents"] for r in final["refunds"]] == [4200]
    assert lookup.line(final, "ORD-55", "X9")["returned"] is True
    assert final["products"] == make_state()["products"]  # quarantine never restocks
    assert tools.get_proposal("P-0001")["status"] == "executed" and tools.open_proposal() is None
    entry = _last_audit(tools, "execute_proposal")
    assert len(entry["committed_effects"]) == 5 and entry["visible_to_agent"] is True


def test_confirmed_exchange_takes_one_replacement_unit(make_tools, lookup):
    tools = make_tools()
    _propose(tools, "exchange", requested_size="L")
    _confirm(tools, 2, "P-0001")
    _confirm(tools, 3, "P-0001")  # replay
    final = tools.final_state()
    assert lookup.product(final, "HOODIE-L")["stock"] == 2 and lookup.product(final, "HOODIE-M")["stock"] == 1
    assert len(final["exchanges"]) == 1 and final["refunds"] == []


def test_double_execute_replays_with_zero_effects(make_tools):
    tools = make_tools()
    _propose(tools)
    first = _confirm(tools, 2, "P-0001")
    after_first = tools.final_state()
    ledger_rows = len(tools.store.ledger())
    same_turn = tools.execute_proposal()
    next_turn = _confirm(tools, 3, "P-0001")
    for replay in (same_turn, next_turn):
        assert replay.is_replay and replay.data["transaction"] == first.data["transaction"]
    assert tools.final_state() == after_first and len(tools.store.ledger()) == ledger_rows
    executes = [e for e in tools.audit.entries() if e["tool"] == "execute_proposal"]
    assert [len(e["committed_effects"]) for e in executes] == [5, 0, 0]


def test_one_execute_attempt_per_turn(make_tools, state, lookup):
    tools = make_tools(state, faults=[{"type": "stock_change", "sku": "HOODIE-L", "new_stock": 0, "before_turn": 2}], turn_count=2)
    tools.begin_turn(1)
    _propose(tools, "exchange", requested_size="L")
    assert _confirm(tools, 2, "P-0001").code == OUT_OF_STOCK
    assert tools.execute_proposal().code == EXECUTE_LIMIT


def test_revalidation_after_a_stock_change_invalidates_and_changes_nothing_else(make_tools, make_state, lookup):
    faults = [{"type": "stock_change", "sku": "HOODIE-L", "new_stock": 0, "before_turn": 2}]
    tools = make_tools(faults=faults, turn_count=2)
    tools.begin_turn(1)
    assert _propose(tools, "exchange", requested_size="L").ok
    result = _confirm(tools, 2, "P-0001")
    assert result.code == OUT_OF_STOCK
    assert result.error["details"]["proposal_status"] == "invalidated"
    assert result.error["details"]["alternatives"] == [{"sku": "HOODIE-S", "size": "S"}]
    expected = make_state()
    lookup.product(expected, "HOODIE-L")["stock"] = 0
    assert tools.final_state() == expected and tools.completed_record() is None
    assert tools.get_proposal("P-0001")["status"] == "invalidated"
    assert _last_audit(tools, "execute_proposal")["committed_effects"] == []


def test_negative_stock_from_a_fault_is_out_of_stock_and_never_decremented(make_tools, lookup):
    tools = make_tools()
    _propose(tools, "exchange", requested_size="L")
    tools.inject_stock_change("HOODIE-L", -2)
    assert _confirm(tools, 2, "P-0001").code == OUT_OF_STOCK
    assert lookup.product(tools.final_state(), "HOODIE-L")["stock"] == -2


def test_revalidation_of_a_refund_uses_current_state(make_tools, state):
    """Another request returns the line between proposal and confirmation."""
    store = Store(state)
    first, second = make_tools(store=store, request_id="REQ-A"), make_tools(store=store, request_id="REQ-B")
    _propose(first)
    _propose(second, "exchange", requested_size="L")
    assert _confirm(first, 2, "P-0001").ok
    late = _confirm(second, 2, "P-0002")
    assert late.code == ALREADY_RETURNED and second.get_proposal("P-0002")["status"] == "invalidated"
    assert len(store.export()["returns"]) == 1


def test_duplicate_via_a_different_proposal_for_the_same_line_is_refused(make_tools):
    tools = make_tools()
    _propose(tools)  # P-0001 refund
    _propose(tools, "exchange", requested_size="L")  # P-0002 supersedes it
    assert _confirm(tools, 2, "P-0001").code == PROPOSAL_NOT_OPEN
    assert _confirm(tools, 3, "P-0002").ok
    late = _confirm(tools, 4, "P-0001")
    assert late.code == REQUEST_ALREADY_COMPLETED
    final = tools.final_state()
    assert len(final["returns"]) == 1 and final["refunds"] == [] and len(final["exchanges"]) == 1


def test_another_request_cannot_return_the_same_line_again(make_tools, state):
    store = Store(state)
    _propose(make_tools(store=store, request_id="REQ-A"))
    _confirm(make_tools(store=store, request_id="REQ-A"), 2, "P-0001")
    result = _propose(make_tools(store=store, request_id="REQ-B"))
    assert result.code == ALREADY_RETURNED and len(store.export()["refunds"]) == 1


def test_primary_key_guard_backs_up_policy(make_tools, state, monkeypatch):
    """Even if eligibility were wrong, the returned_lines key refuses the second return."""
    store = Store(state)
    a, b = make_tools(store=store, request_id="REQ-A"), make_tools(store=store, request_id="REQ-B")
    _propose(a)
    _propose(b)
    assert _confirm(a, 2, "P-0001").ok
    real = policy.eligibility

    def always_eligible(*args):
        report = real(*args)
        return domain.Eligibility(**{**report.to_dict(), "eligible": True, "reasons": []})

    monkeypatch.setattr(policy, "eligibility", always_eligible)
    result = _confirm(b, 2, "P-0002")
    assert result.code == ALREADY_RETURNED and result.error["details"]["proposal_status"] == "invalidated"
    assert b.get_proposal("P-0002")["status"] == "invalidated" and len(store.export()["refunds"]) == 1


def test_binding_mismatch_when_stored_facts_were_altered(make_tools, make_state):
    tools = make_tools()
    _propose(tools)
    with tools.store.transaction() as tx:
        tampered = tx.proposal("P-0001")
        tx.put_proposal(Proposal.from_dict({**tampered.to_dict(), "amount_cents": 9999}))
    assert _confirm(tools, 2, "P-0001").code == BINDING_MISMATCH
    assert tools.final_state() == make_state() and tools.get_proposal("P-0001")["status"] == "open"


@pytest.mark.parametrize(("customer_id", "request_id"), [("C7", "REQ-OTHER"), ("C8", "REQ-T1")])
def test_a_proposal_cannot_execute_under_another_identity(make_tools, state, customer_id, request_id):
    store = Store(state)
    _propose(make_tools(store=store))
    intruder = make_tools(store=store, customer_id=customer_id, request_id=request_id)
    assert _confirm(intruder, 2, "P-0001").code == BINDING_MISMATCH
    assert store.export()["refunds"] == []


# === idempotency short-circuit ==================================================================


def test_completed_request_replays_the_same_target_and_refuses_others(make_tools):
    tools = make_tools()
    _propose(tools, "exchange", requested_size="L")
    transaction = _confirm(tools, 2, "P-0001").data["transaction"]
    tools.begin_turn(3)
    same = [
        _propose(tools, "exchange", requested_size="large"),
        _propose(tools, "exchange"),  # no size given: nothing contradicts the record
        _propose(tools, "exchange", requested_sku="HOODIE-L"),
    ]
    assert all(r.is_replay and r.data["transaction"] == transaction for r in same)
    different = [
        _propose(tools, "exchange", requested_size="S"),
        _propose(tools),  # refund instead of exchange
        _propose(tools, "exchange", line_id="X10"),
    ]
    assert {r.code for r in different} == {REQUEST_ALREADY_COMPLETED}
    assert different[0].error["details"] == {"transaction_id": "TX-0001"}
    assert len(tools.proposals()) == 1


def test_every_ops_tool_short_circuits_after_completion(make_tools):
    tools = make_tools()
    _propose(tools)
    _confirm(tools, 2, "P-0001")
    tools.begin_turn(3)
    tools.set_target("ORD-55", "X9", "refund")
    eligibility = tools.check_eligibility()
    assert eligibility.data["completed_by_this_request"] is True and eligibility.data["reasons"] == [ALREADY_RETURNED]
    assert tools.list_exchange_options().is_replay and tools.create_proposal("refund").is_replay
    assert tools.execute_proposal().is_replay  # no grant needed to replay; nothing is written
    tools.set_target("ORD-55", "X10", "refund")
    assert tools.check_eligibility().code == REQUEST_ALREADY_COMPLETED
    assert tools.list_exchange_options().code == REQUEST_ALREADY_COMPLETED
    assert tools.check_proposal("refund").code == REQUEST_ALREADY_COMPLETED
    tools.begin_turn(4)  # no target injected: nothing to compare against
    assert tools.check_eligibility().is_replay


def test_grant_for_an_identically_bound_older_proposal_replays(make_tools):
    tools = make_tools()
    _propose(tools)  # P-0001 refund
    _propose(tools, "exchange", requested_size="L")  # P-0002
    _propose(tools)  # P-0003 refund again (same binding as P-0001)
    assert _confirm(tools, 2, "P-0003").ok
    assert _confirm(tools, 3, "P-0001").is_replay
    assert _confirm(tools, 4, "P-0002").code == REQUEST_ALREADY_COMPLETED
    assert _confirm(tools, 5, "P-0404").code == REQUEST_ALREADY_COMPLETED


def test_a_completed_record_supplied_in_the_input_replays(make_tools, state):
    state["idempotency"] = {"REQ-T1": {"status": "completed", "proposal_id": "P-OLD"}}
    tools = make_tools(state, faults=[TIMEOUT_FAULT])
    assert _propose(tools).is_replay
    assert _confirm(tools, 2, "P-OLD").is_replay
    assert tools.faults.timeout_fired is False and tools.faults.timeout_armed  # replays never consume it
    assert tools.final_state() == state


# === timeout_after_commit =========================================================================


def test_timeout_fires_once_after_a_durable_commit(make_tools):
    tools = make_tools(faults=[TIMEOUT_FAULT], turn_count=3)
    tools.begin_turn(1)
    _propose(tools)
    timeout = _confirm(tools, 2, "P-0001")
    assert timeout.code == TIMEOUT and timeout.data is None
    committed = tools.final_state()
    assert len(committed["refunds"]) == 1 and "REQ-T1" in committed["idempotency"]  # committed BEFORE the timeout

    entry = _last_audit(tools, "execute_proposal")
    assert entry["error"]["code"] == TIMEOUT and entry["visible_to_agent"] is False
    assert len(entry["committed_effects"]) == 5 and entry["result"]["transaction"] == committed["idempotency"]["REQ-T1"]

    status = tools.get_request_status()
    assert status.data == {"request_id": "REQ-T1", "status": "completed", "record": committed["idempotency"]["REQ-T1"]}

    replay = _confirm(tools, 3, "P-0001")
    assert replay.is_replay and replay.data["transaction"] == committed["idempotency"]["REQ-T1"]
    assert tools.final_state() == committed and _last_audit(tools, "execute_proposal")["committed_effects"] == []
    assert tools.faults.timeout_fired


def test_a_refused_execution_does_not_consume_the_timeout(make_tools):
    faults = [TIMEOUT_FAULT, {"type": "stock_change", "sku": "HOODIE-L", "new_stock": 0, "before_turn": 2}]
    tools = make_tools(faults=faults, turn_count=4)
    tools.begin_turn(1)
    _propose(tools, "exchange", requested_size="L")
    assert _confirm(tools, 2, "P-0001").code == OUT_OF_STOCK and not tools.faults.timeout_fired
    tools.begin_turn(3)
    tools.inject_stock_change("HOODIE-L", 5)
    assert _propose(tools, "exchange", requested_size="L").data["proposal"]["proposal_id"] == "P-0002"
    assert _confirm(tools, 4, "P-0002").code == TIMEOUT and tools.faults.timeout_fired


def test_get_request_status_before_completion(make_tools):
    assert make_tools().get_request_status().data == {"request_id": "REQ-T1", "status": "not_found"}


# === UI fault controls, read helpers, internal errors ================================================


def test_ui_fault_controls(make_tools, lookup):
    tools = make_tools()
    event = tools.inject_stock_change("HOODIE-L", 1)
    assert event["before_turn"] is None and lookup.product(tools.final_state(), "HOODIE-L")["stock"] == 1
    tools.arm_timeout()
    _propose(tools)
    assert _confirm(tools, 1, "P-0001").code == TIMEOUT
    assert [e["tool"] for e in tools.audit.entries() if e["kind"] == "fault"] == ["stock_change", "timeout_after_commit"]


def test_read_helpers_return_copies(make_tools):
    tools = make_tools()
    _propose(tools)
    tools.proposals()[0]["status"] = "hacked"
    tools.final_state()["products"].clear()
    assert tools.proposals()[0]["status"] == "open" and len(tools.final_state()["products"]) == 6
    assert tools.external_events() == [] and tools.completed_record() is None


def test_unexpected_exceptions_become_audited_internal_failures(make_tools, monkeypatch):
    tools = make_tools()

    def broken() -> dict:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(tools.store, "export", broken)
    result = tools.find_order_lines()
    assert result.code == INTERNAL and "disk on fire" in result.error["message"]
    assert _last_audit(tools, "find_order_lines")["error"]["code"] == INTERNAL


# === the public scenarios at the gateway level =========================================================


def _case(public_cases, case_id):
    case = copy.deepcopy(next(c for c in public_cases if c["case_id"] == case_id))
    return case, Store(case["initial_state"])


def _tools_for(case, store) -> BusinessTools:
    from desk.faults import FaultPlan

    return BusinessTools(store, case["customer_id"], case["request_id"], FaultPlan(case["faults"], len(case["turns"])))


def test_public_refund_at_the_30_day_boundary(public_cases, lookup):
    case, store = _case(public_cases, "public_01_refund_boundary")
    tools = _tools_for(case, store)
    tools.begin_turn(1)
    proposal = _propose(tools, order_id="O100", line_id="L1").data["proposal"]
    assert (proposal["amount_cents"], proposal["eligibility"]["days_since_delivery"]) == (3000, 30)
    assert _confirm(tools, 2, proposal["proposal_id"]).ok
    final = store.export()
    assert [r["amount_cents"] for r in final["refunds"]] == [3000] and lookup.product(final, "TEE-M")["stock"] == 2
    assert list(final) == list(case["initial_state"])


def test_public_expired_return_changes_nothing(public_cases):
    case, store = _case(public_cases, "public_03_expired")
    tools = _tools_for(case, store)
    tools.begin_turn(1)
    assert _propose(tools, order_id="O100", line_id="L1").code == RETURN_WINDOW_EXPIRED
    assert json.dumps(store.export()) == json.dumps(case["initial_state"])


def test_public_stale_stock_exchange(public_cases, lookup):
    case, store = _case(public_cases, "public_05_stale_stock")
    tools = _tools_for(case, store)
    tools.begin_turn(1)
    proposal = _propose(tools, "exchange", order_id="O100", line_id="L1", requested_size="L").data["proposal"]
    assert _confirm(tools, 2, proposal["proposal_id"]).code == OUT_OF_STOCK
    expected = copy.deepcopy(case["initial_state"])
    lookup.product(expected, "TEE-L")["stock"] = 0
    assert store.export() == expected and [e["new_stock"] for e in tools.external_events()] == [0]


def test_public_timeout_then_replay(public_cases):
    case, store = _case(public_cases, "public_06_timeout_replay")
    tools = _tools_for(case, store)
    tools.begin_turn(1)
    pid = _propose(tools, order_id="O100", line_id="L1").data["proposal"]["proposal_id"]
    assert _confirm(tools, 2, pid).code == TIMEOUT
    assert tools.get_request_status().data["status"] == "completed"
    assert _confirm(tools, 3, pid).is_replay
    final = store.export()
    assert (len(final["refunds"]), len(final["returns"]), len(final["quarantine"]), len(final["idempotency"])) == (1, 1, 1, 1)


def test_a_refund_proposal_that_lost_eligibility_is_invalidated_without_alternatives(make_tools, make_state):
    """Two requests propose a refund of the same line; once one commits, the other one's execute
    re-runs eligibility inside the commit lock and refuses (no exchange sizes for a refund)."""
    store = Store(make_state())
    first, second = make_tools(store=store, request_id="REQ-A"), make_tools(store=store, request_id="REQ-B")
    for tools in (first, second):
        tools.begin_turn(1)
        tools.set_target("ORD-55", "X9", "refund")
        assert tools.create_proposal("refund").ok
    p_first, p_second = first.open_proposal()["proposal_id"], second.open_proposal()["proposal_id"]
    first.begin_turn(2)
    first.set_grant(p_first)
    assert first.execute_proposal().ok
    second.begin_turn(2)
    second.set_grant(p_second)
    refused = second.execute_proposal()
    assert refused.code == ALREADY_RETURNED and refused.error["details"]["alternatives"] == []
    assert len(store.export()["refunds"]) == 1


def test_check_proposal_answers_like_create_but_stores_nothing(make_tools):
    tools = make_tools()
    tools.begin_turn(1)
    tools.set_target("ORD-55", "X9", "exchange", requested_size="L")
    dry = tools.check_proposal("exchange")
    assert dry.ok and dry.data == {"would_propose": {"remedy": "exchange", "replacement_sku": "HOODIE-L",
                                                     "amount_cents": 0}, "stored": False}
    assert tools.proposals() == [] and _last_audit(tools, "check_proposal")["agent"] == "orchestrator"
    tools.set_target("ORD-55", "X9", "exchange", requested_size="XL")
    assert tools.check_proposal("exchange").code == "PRICE_MISMATCH" and tools.proposals() == []
    tools.set_target("ORD-55", "X9", "refund")
    assert tools.check_proposal("refund").data["would_propose"]["amount_cents"] == 4200


def test_check_eligibility_of_a_named_line_needs_no_target(make_tools):
    tools = make_tools()
    tools.begin_turn(1)
    assert tools.check_eligibility("ORD-55", "X9").data["eligible"] is True
    assert tools.check_eligibility("ORD-66", "Z1").data["reasons"] == ["ORDER_NOT_FOUND"]  # another customer's
    assert _last_audit(tools, "check_eligibility")["arguments"] == {"order_id": "ORD-66", "line_id": "Z1"}
    assert tools.check_eligibility().code == "NO_TARGET"


def test_the_timeout_text_does_not_tell_the_agent_what_to_do(make_tools):
    """Contract review C9: the fault is raised at the tool boundary with a neutral message."""
    tools = make_tools(faults=[{"type": "timeout_after_commit"}])
    tools.begin_turn(1)
    tools.set_target("ORD-55", "X9", "refund")
    pid = tools.create_proposal("refund").data["proposal"]["proposal_id"]
    tools.begin_turn(2)
    tools.set_grant(pid)
    lost = tools.execute_proposal()
    assert lost.code == "TIMEOUT" and lost.error["message"] == "commit response lost (timeout)"
