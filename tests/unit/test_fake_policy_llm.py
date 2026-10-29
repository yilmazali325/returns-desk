"""desk/testing/fake_policy_llm.py: the deterministic double reads prompts built by desk.agents.prompts
(through the shared header constants, review C8) and acts like a careful agent: it cites the
customer's words when several lines could fit, follows a one-time desk suggestion, and asks
instead of proposing on a question about rules or status.
"""

from __future__ import annotations

import json
import time
from typing import Any

from desk.agents import prompts
from desk.agents.prompts import (
    CONFIRMATION_NONE,
    NO_CARD_TEXT,
    RESOLUTION_SYSTEM,
    account_card_text,
    confirmation_granted,
    ops_brief,
    turn_block,
)
from desk.agents.toolspecs import (
    CREATE_PROPOSAL,
    EXECUTE_PROPOSAL,
    FIND_ORDER_LINES,
    GET_REQUEST_STATUS,
    HANDOFF,
    LIST_EXCHANGE_OPTIONS,
    OPERATIONS_TOOLS_FREE_TEXT,
    OPERATIONS_TOOLS_STRUCTURED,
    REPLY,
    REPORT,
    RESOLUTION_TOOLS,
)
from desk.grounding import analyze
from desk.llm import tool_message
from desk.orchestrator import ALREADY_COMPLETED_NOTE, CANCEL_NOTE, REPLAY_NOTE
from desk.sanitize import sanitize
from desk.testing import fake_policy_llm as fp

LINES = [
    {"order_id": "ORD-7Q", "line_id": "LN-3", "sku": "SCARF-M", "product_id": "SCARF", "size": "M",
     "aliases": ["scarf", "scarves"], "paid_cents": 2500, "delivered_on": "2026-03-01"},
    {"order_id": "ORD-7Q", "line_id": "LN-4", "sku": "CAP-S", "product_id": "CAP", "size": "S",
     "aliases": ["cap"], "paid_cents": 900, "delivered_on": "2026-03-09"},
]


def prompt(message: str, *, lines: list[dict[str, Any]] = LINES, history: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    card = account_card_text({"lines": lines, "catalog": []}, "2026-03-15")
    return [{"role": "system", "content": RESOLUTION_SYSTEM}, {"role": "user", "content": card},
            {"role": "user", "content": turn_block(history or [], None, CONFIRMATION_NONE, None, sanitize(message),
                                                   [HANDOFF, REPLY])}]


def call(messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    return fp.resolution_call(fp.read_resolution(messages), [HANDOFF, REPLY])


def test_the_double_reads_a_prompt_built_by_the_prompt_module() -> None:
    view = fp.read_resolution(prompt("refund the cap", history=[{"message": "hi", "decision": "clarify"}]))
    assert view.as_of == "2026-03-15" and view.customer_messages == ["hi", "refund the cap"]
    assert [(x["line_id"], x["paid_cents"], x["delivered_on"]) for x in view.lines] == [
        ("LN-3", 2500, "2026-03-01"), ("LN-4", 900, "2026-03-09")]
    assert prompts.ACCOUNT_CARD_HEADER in prompt("x")[1]["content"]


def test_it_cites_the_customers_words_when_two_lines_could_fit() -> None:
    assert call(prompt("Please refund the cap.")) == (
        HANDOFF, {"intent": "propose_refund", "order_id": "ORD-7Q", "line_id": "LN-4", "evidence": "cap"})
    tool, args = call(prompt("Refund line LN-3 of ORD-7Q."))  # the ids already pin the line: no citation
    assert tool == HANDOFF and "evidence" not in args


def test_cited_words_none_when_no_phrase_passes() -> None:
    analysis = analyze(["refund something"], LINES, [], "2026-03-15")
    assert fp.cited_words(analysis, ("ORD-7Q", "LN-3")) is None


def test_it_follows_a_one_time_desk_suggestion() -> None:
    suggested = {"intent": "propose_refund", "order_id": "ORD-7Q", "line_id": "LN-3"}
    messages = prompt("Refund the scarf.") + [
        {"role": "assistant", "content": "", "tool_name": REPLY, "tool_args": {"message": "Which one?"}},
        tool_message(REPLY, {"error": {"code": "HANDOFF_SUGGESTED", "message": "...", "handoff": suggested}})]
    assert call(messages) == (HANDOFF, suggested)


def test_a_rule_or_status_question_gets_a_question_back() -> None:
    assert call(prompt("What's your return policy for scarves? Just asking.")) == (REPLY, {"message": fp.REPLIES["inquiry"]})


def test_an_exchange_for_a_product_nobody_owns_names_it() -> None:
    one = LINES[:1]
    tool, args = call(prompt("Can I swap my scarf from ORD-7Q for the jacket instead?", lines=one))
    assert tool == HANDOFF and args["requested_product"] == "jacket"


def test_a_replayed_report_gets_the_replay_sentence() -> None:
    messages = prompt("did it go through?") + [tool_message(HANDOFF, {"outcome": "completed", "already_processed_earlier": True})]
    assert call(messages) == (REPLY, {"message": fp.REPLIES["replay"]})


# --- ported from tests/unit/test_server.py so this module's tests live next to it ---------------


def full_prompt(message: str, *, lines: list[dict[str, Any]] | None = LINES, confirmation: str = CONFIRMATION_NONE,
                note: str | None = None) -> list[dict[str, Any]]:
    card = account_card_text({"lines": lines, "catalog": []}) if lines is not None else NO_CARD_TEXT
    return [{"role": "system", "content": RESOLUTION_SYSTEM}, {"role": "user", "content": card},
            {"role": "user", "content": turn_block([], None, confirmation, note, sanitize(message), [HANDOFF, REPLY])}]


def test_current_message_without_a_block_is_empty_and_card_rows_are_parsed_back() -> None:
    assert fp.current_message("no block here") == ""
    card = account_card_text({"lines": LINES, "catalog": []}) + "\n- not a line row"
    assert [x["line_id"] for x in fp.card_lines(card)] == ["LN-3", "LN-4"]


def test_resolution_asks_when_the_words_do_not_ground_a_request() -> None:
    assert call(full_prompt("I want a refund.")) == (REPLY, {"message": fp.REPLIES["which_item"]})
    one = LINES[:1]
    assert call(full_prompt("About my scarf.", lines=one)) == (REPLY, {"message": fp.REPLIES["which_remedy"]})
    assert call(full_prompt("Refund order ORD-999 please.", lines=one)) == (REPLY, {"message": fp.REPLIES["not_found"]})
    assert fp.resolution_call(fp.read_resolution(full_prompt("Exchange my scarf for something.", lines=one)), [REPLY]) == \
        (REPLY, {"message": fp.REPLIES["needs_info"]})
    assert call(full_prompt("Exchange my scarf for a small or a large.", lines=one))[0] == REPLY


def test_resolution_executes_only_with_the_confirm_control() -> None:
    messages = full_prompt("Yes go ahead.", confirmation=confirmation_granted({"proposal_id": "P-0007"}))
    assert call(messages) == (HANDOFF, {"intent": "execute_confirmed"})
    pending = messages + [{"role": "assistant", "content": "", "tool_name": REPLY, "tool_args": {"message": "ok"}},
                          tool_message(REPLY, {"error": {"code": "CONFIRMATION_PENDING", "message": "hand off first"}})]
    view = fp.read_resolution(pending)
    view.confirmed = False
    assert fp.resolution_call(view, [HANDOFF, REPLY]) == (HANDOFF, {"intent": "execute_confirmed"})


def test_a_gate_rejection_is_answered_with_a_question() -> None:
    handoff = {"intent": "propose_refund", "order_id": "ORD-7Q", "line_id": "LN-3"}
    rejected = full_prompt("Refund line LN-3 of ORD-7Q.") + [
        {"role": "assistant", "content": "", "tool_name": HANDOFF, "tool_args": handoff},
        tool_message(HANDOFF, {"error": {"code": "TARGET_MISMATCH", "message": "no"}})]
    assert call(rejected)[0] == REPLY


def test_resolution_looks_lines_up_when_no_card_is_loaded() -> None:
    messages = full_prompt("Refund line LN-3 of ORD-7Q.", lines=None)
    assert fp.resolution_call(fp.read_resolution(messages), [FIND_ORDER_LINES, HANDOFF, REPLY]) == \
        (FIND_ORDER_LINES, {"order_ref": "ALL"})
    looked_up = messages + [{"role": "assistant", "content": "", "tool_name": FIND_ORDER_LINES, "tool_args": {"order_ref": "ALL"}},
                            tool_message(FIND_ORDER_LINES, {"ok": True, "data": {"matches": LINES, "count": 2}})]
    assert fp.resolution_call(fp.read_resolution(looked_up), [FIND_ORDER_LINES, HANDOFF, REPLY])[1]["line_id"] == "LN-3"
    cached = messages + [tool_message(FIND_ORDER_LINES, {"already_done": True, "result": {"data": {"matches": LINES}}})]
    assert fp.read_resolution(cached).lines == LINES


def test_resolution_replies_from_the_report_and_the_desk_notes() -> None:
    base = full_prompt("Refund line LN-3 of ORD-7Q.")
    assert call(base + [tool_message(HANDOFF, {"outcome": "proposed"})]) == (REPLY, {"message": fp.REPLIES["proposed"]})
    assert call(base + [tool_message(HANDOFF, {"outcome": "weird"})]) == (REPLY, {"message": fp.REPLIES["failed"]})
    for desk_note, key in ((REPLAY_NOTE, "replay"), (CANCEL_NOTE, "cancel"), (ALREADY_COMPLETED_NOTE, "already_completed")):
        assert fp.resolution_call(fp.read_resolution(full_prompt("ok", note=desk_note)), [REPLY]) == \
            (REPLY, {"message": fp.REPLIES[key]})


def test_grounded_handoff_names_another_product_when_asked() -> None:
    catalog_lines = LINES[:1] + [{"order_id": "ORD-8", "line_id": "LN-9", "sku": "CAP-S", "product_id": "CAP", "size": "S",
                                  "aliases": ["cap"]}]
    analysis = analyze(["Exchange my scarf from ORD-7Q for the cap in size S."], catalog_lines, [])
    handoff = fp.grounded_handoff(analysis, catalog_lines)
    assert handoff is not None and handoff["requested_product"] == "CAP"


def ops_prompt(intent: str, *events: tuple[str, dict[str, Any]], desk_called: tuple[str, dict] | None = None) -> list[dict[str, Any]]:
    messages = [{"role": "system", "content": "ops"},
                {"role": "user", "content": ops_brief(intent, LINES[0], "L", None, None, ["x"])}]
    for tool, payload in events:
        messages.append({"role": "assistant", "content": "", "tool_name": tool, "tool_args": {}})
        messages.append(tool_message(tool, payload))
    if desk_called:
        messages.append({"role": "user", "content": f"DESK: the desk called {desk_called[0]} for you: " + json.dumps(desk_called[1])})
    return messages


def ops(messages: list[dict[str, Any]], allowed: list[str], free_text: bool = False) -> tuple[str, dict[str, Any]]:
    return fp.operations_call(fp.read_operations(messages), allowed, free_text)


ALL_OPS = [LIST_EXCHANGE_OPTIONS, CREATE_PROPOSAL, EXECUTE_PROPOSAL, GET_REQUEST_STATUS, REPORT]
PROPOSAL_OK = {"ok": True, "data": {"proposal": {"proposal_id": "P-1"}}, "error": None}


def test_operations_follows_the_documented_tool_order_for_an_exchange() -> None:
    assert ops(ops_prompt("propose_exchange"), ALL_OPS) == (LIST_EXCHANGE_OPTIONS, {})
    options = ("list_exchange_options", {"ok": True, "data": {"requested_sku": "SCARF-L"}, "error": None})
    assert ops(ops_prompt("propose_exchange", options), ALL_OPS) == (CREATE_PROPOSAL, {"remedy": "exchange", "replacement_sku": "SCARF-L"})
    no_size = ("list_exchange_options", {"ok": True, "data": {"requested_sku": None}, "error": None})
    assert ops(ops_prompt("propose_exchange", no_size), ALL_OPS) == (CREATE_PROPOSAL, {"remedy": "exchange"})
    done = ops_prompt("propose_exchange", options, ("create_proposal", PROPOSAL_OK))
    assert ops(done, ALL_OPS) == (REPORT, {"outcome": "proposed"})
    assert ops(done, ALL_OPS, free_text=True) == (REPORT, {"outcome": "proposed", "summary": fp.SUMMARIES["proposed"]})


def test_operations_reconciles_a_timeout_before_reporting() -> None:
    assert ops(ops_prompt("execute_confirmed"), ALL_OPS) == (EXECUTE_PROPOSAL, {})
    timeout = ("execute_proposal", {"ok": False, "data": None, "error": {"code": "TIMEOUT"}})
    assert ops(ops_prompt("execute_confirmed", timeout), ALL_OPS) == (GET_REQUEST_STATUS, {})
    assert fp.observed_outcome(fp.read_operations(ops_prompt("execute_confirmed", timeout))) == "failed"
    status = {"ok": True, "data": {"status": "completed", "record": {}}, "error": None}
    reconciled = ops_prompt("execute_confirmed", timeout, desk_called=("get_request_status", status))
    assert ops(reconciled, ALL_OPS) == (REPORT, {"outcome": "completed"})
    assert ops(ops_prompt("check_status"), ALL_OPS) == (GET_REQUEST_STATUS, {})
    not_found = ("get_request_status", {"ok": True, "data": {"status": "not_found"}, "error": None})
    assert ops(ops_prompt("check_status", not_found), ALL_OPS) == (REPORT, {"outcome": "needs_info"})


def test_operations_reports_what_the_results_show() -> None:
    def outcome(*events: tuple[str, dict[str, Any]]) -> str:
        return fp.observed_outcome(fp.read_operations(ops_prompt("propose_refund", *events)))

    def error(code: str) -> dict[str, Any]:
        return {"ok": False, "data": None, "error": {"code": code}}

    assert outcome(("create_proposal", error("RETURN_WINDOW_EXPIRED"))) == "denied"
    assert outcome(("create_proposal", error("SIZE_UNSPECIFIED"))) == "needs_info"
    assert outcome(("create_proposal", error("INTERNAL"))) == "failed"
    assert outcome(("create_proposal", PROPOSAL_OK), ("create_proposal", error("INVALID_ARGS"))) == "proposed"
    assert outcome(("list_exchange_options", error("REQUEST_ALREADY_COMPLETED"))) == "needs_info"
    assert outcome(("execute_proposal", {"ok": True, "data": {"transaction": {}}, "error": None})) == "completed"
    assert outcome(("create_proposal", {"ok": False, "error": "not a dict"})) == "needs_info"
    repeat = ops_prompt("propose_refund") + [tool_message(CREATE_PROPOSAL, {"already_done": True, "result": PROPOSAL_OK})]
    assert fp.read_operations(repeat).events == []


def test_operations_falls_back_to_an_allowed_tool() -> None:
    assert ops(ops_prompt("propose_refund"), [REPORT]) == (REPORT, {"outcome": "needs_info"})
    assert ops(ops_prompt("propose_refund"), [GET_REQUEST_STATUS]) == (GET_REQUEST_STATUS, {})


def test_step_validates_like_a_real_answer_and_records_size_estimates() -> None:
    llm = fp.FakePolicyLLM(step_delay_s=0.001)
    started = time.perf_counter()
    step = llm.step("resolution", full_prompt("Please refund line LN-3 of ORD-7Q."), RESOLUTION_TOOLS, [HANDOFF, REPLY], 160)
    assert time.perf_counter() - started >= 0.001
    assert step.ok and step.tool == HANDOFF and step.role == "resolution" and llm.calls == 1
    assert step.prompt_tokens > 100 and step.completion_tokens >= 1 and step.predicted_ms == 0 and step.latency_ms == 1.0
    ops_step = llm.step("operations", ops_prompt("propose_refund"), OPERATIONS_TOOLS_FREE_TEXT, [CREATE_PROPOSAL, REPORT], 160)
    assert ops_step.ok and ops_step.tool == CREATE_PROPOSAL
    report = llm.step("operations", ops_prompt("propose_refund", ("create_proposal", PROPOSAL_OK)), OPERATIONS_TOOLS_FREE_TEXT,
                      [REPORT], 160)
    assert report.args == {"outcome": "proposed", "summary": fp.SUMMARIES["proposed"]}
    structured = llm.step("operations", ops_prompt("propose_refund", ("create_proposal", PROPOSAL_OK)),
                          OPERATIONS_TOOLS_STRUCTURED, [REPORT], 160)
    assert structured.args == {"outcome": "proposed"}


def test_json_helper_never_raises() -> None:
    assert fp._json("not json") == {} and fp._json("[1, 2]") == {} and fp._json('{"a": 1}') == {"a": 1}
