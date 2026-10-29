"""E2E tests for the UI's error paths and less common audit entries.

Two techniques, both against the real page:
- Playwright route interception makes the server look broken (409, 500, a stream that stops early,
  an SSE `error` event, an unreachable server) so each actionable notice can be checked.
- The team view's event mapping is a pure function (`window.ReturnsDesk.describeEntry`); it is called
  in the page with every audit-entry shape the desk can record, including ones the deterministic
  fake model never provokes (vetoes, reply-guard overrides, fallbacks, skipped faults).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(180)]

JSON = "application/json"
SSE = "text/event-stream"


def sse(*events: tuple[str, Any]) -> str:
    return "".join(f"event: {kind}\ndata: {json.dumps(data)}\n\n" for kind, data in events) + ": keep-alive\n\n"


def notice_text(desk: Any) -> str:
    notice = desk.page.locator('[data-testid="notice"]')
    notice.wait_for()
    return notice.inner_text()


def dismiss(desk: Any) -> None:
    desk.page.locator('[data-testid="notice"] button').click()
    desk.page.wait_for_selector('[data-testid="notice"]', state="detached")


def send(desk: Any, text: str) -> None:
    desk.page.fill('[data-testid="composer-input"]', text)
    desk.page.click('[data-testid="send"]')


@pytest.mark.http_errors
def test_turn_failures_show_a_notice_and_reload_the_session(desk: Any) -> None:
    page = desk.page
    desk.open("refund_boundary")
    turns = "**/api/sessions/*/turns"
    page.route(turns, lambda route: route.fulfill(status=409, content_type=JSON, body=json.dumps({"detail": "a turn is still running"})))
    send(desk, "hello")
    assert "Not sent" in notice_text(desk) and "a turn is still running" in notice_text(desk)
    dismiss(desk)
    page.unroute(turns)
    page.route(turns, lambda route: route.fulfill(status=200, content_type=SSE, body=sse(("start", {"turn": 1, "message": "x", "confirm_proposal_id": None}))))
    send(desk, "hello")
    assert "the stream ended before the turn finished" in notice_text(desk)
    dismiss(desk)
    page.unroute(turns)
    page.route(turns, lambda route: route.fulfill(status=200, content_type=SSE, body=sse(
        ("start", {"turn": 1, "message": "x", "confirm_proposal_id": None}), ("error", {"turn": 1, "message": "RuntimeError: boom"}))))
    send(desk, "hello")
    assert "The desk hit an internal error" in notice_text(desk) and "RuntimeError: boom" in notice_text(desk)
    dismiss(desk)
    page.unroute(turns)
    page.route(turns, lambda route: route.fulfill(status=502, content_type="text/plain", body="bad gateway"))
    page.route("**/api/sessions/*/state", lambda route: route.fulfill(status=503, content_type="text/plain", body="down"))
    send(desk, "hello")
    page.wait_for_function("() => document.querySelector('[data-testid=notice]') && document.querySelector('[data-testid=notice]').innerText.includes('Could not reload the session')")
    assert "HTTP 503" in notice_text(desk)
    page.unroute(turns)
    page.unroute("**/api/sessions/*/state")
    assert desk.fill_and_send(1) == "propose"  # the session itself is fine


@pytest.mark.http_errors
def test_failure_lab_and_reset_failures_are_explained(desk: Any) -> None:
    page = desk.page
    desk.open("refund_boundary")
    for bad in ("-1", "1.5", ""):
        page.fill('[data-testid="stock-value"]', bad)
        page.click('[data-testid="stock-apply"]')
        assert "whole number of 0 or more" in notice_text(desk)
        dismiss(desk)
    page.route("**/api/sessions/*/faults", lambda route: route.fulfill(status=422, content_type=JSON, body=json.dumps(
        {"detail": [{"loc": ["body", "new_stock"], "msg": "Input should be a valid integer"}]})))
    page.fill('[data-testid="stock-value"]', "2")
    page.click('[data-testid="stock-apply"]')
    text = notice_text(desk)
    assert "could not apply that" in text and "new_stock: Input should be a valid integer" in text
    dismiss(desk)
    page.unroute("**/api/sessions/*/faults")
    page.route("**/api/sessions/*/reset", lambda route: route.fulfill(status=500, content_type="text/html", body="<h1>oops</h1>"))
    page.click('[data-testid="reset"]')
    assert "Reset failed" in notice_text(desk) and "HTTP 500" in notice_text(desk)
    dismiss(desk)
    page.unroute("**/api/sessions/*/reset")
    page.click('[data-testid="arm-timeout"]')
    page.wait_for_selector('[data-testid="fault-status"][data-state="armed"]')
    page.click('[data-testid="send"]')  # empty composer
    assert desk.page.locator('[data-testid="progress"]').inner_text() == "Type a message first."


@pytest.mark.http_errors
def test_an_unreachable_server_is_shown_in_the_header(desk: Any) -> None:
    desk.page.route("**/api/health", lambda route: route.abort())
    desk.page.route("**/api/telemetry", lambda route: route.abort())
    desk.page.wait_for_selector('#llm-dot[data-state="down"]', timeout=10_000)
    assert desk.page.locator("#model-text").inner_text() == "desk server not reachable"


@pytest.mark.http_errors
def test_start_up_failures_are_explained(make_desk: Any) -> None:
    broken = make_desk(before_goto=lambda page: page.route("**/api/meta", lambda route: route.fulfill(status=500, body="x")))
    assert "The desk server is not reachable" in notice_text(broken)
    no_session = make_desk(before_goto=lambda page: page.route(
        "**/api/sessions", lambda route: route.fulfill(status=500, body="x") if route.request.method == "POST" else route.continue_()))
    assert "Could not start the session" in notice_text(no_session)
    assert no_session.page.locator("#scenario-notes").inner_text() == ""
    stale = make_desk(before_goto=lambda page: page.add_init_script("localStorage.setItem('returns-desk.session', '\"S-9999\"')"))
    stale.page.wait_for_function("() => window.ReturnsDesk.app.session && window.ReturnsDesk.app.session.session_id !== 'S-9999'")


def test_blocked_browser_storage_does_not_break_the_page(make_desk: Any) -> None:
    desk = make_desk(before_goto=lambda page: page.add_init_script(
        "Object.defineProperty(window, 'localStorage', {get() { throw new Error('storage blocked'); }})"))
    desk.open("refund_boundary")
    desk.page.click('[data-testid="theater"]')  # stores a preference: must fail quietly
    desk.page.click('[data-testid="theater"]')  # back to the normal layout (theater hides the rail)
    assert desk.fill_and_send(1) == "propose"


def test_a_reload_restores_the_same_session(desk: Any) -> None:
    desk.open("exchange")
    assert desk.fill_and_send(1) == "propose"
    session_id = desk.page.evaluate("() => window.ReturnsDesk.app.session.session_id")
    desk.page.reload()
    desk.page.wait_for_function("id => window.ReturnsDesk.app.session && window.ReturnsDesk.app.session.session_id === id", arg=session_id)
    assert desk.decision(1) == "propose" and desk.page.locator('[data-testid="confirm"]').count() == 1


def test_tabs_audit_tools_and_blank_state_controls(desk: Any) -> None:
    page = desk.page
    desk.open("refund_boundary")
    assert desk.fill_and_send(1) == "propose"
    page.focus('[data-testid="tab-team"]')
    page.keyboard.press("ArrowRight")
    assert page.get_attribute('[data-testid="tab-audit"]', "aria-selected") == "true"
    page.keyboard.press("Home")  # any other key leaves the selection alone
    assert page.get_attribute('[data-testid="tab-audit"]', "aria-selected") == "true"
    page.click("#audit-expand")
    assert page.locator("details.audit-entry[open]").count() == page.locator("details.audit-entry").count()
    page.click("#audit-collapse")
    assert page.locator("details.audit-entry[open]").count() == 0
    page.check("#audit-hide-grounding")
    assert page.locator('details.audit-entry[data-kind="grounding"]').count() == 0
    page.focus('[data-testid="tab-audit"]')
    page.keyboard.press("ArrowLeft")
    assert page.get_attribute('[data-testid="tab-team"]', "aria-selected") == "true"
    page.select_option('[data-testid="scenario-select"]', "__blank__")
    page.wait_for_function("() => window.ReturnsDesk.app.session.case_id === null")
    first = page.evaluate("() => window.ReturnsDesk.app.session.session_id")
    page.select_option("#blank-state-select", index=2)
    page.wait_for_function("id => window.ReturnsDesk.app.session.session_id !== id", arg=first)
    assert page.locator('[data-testid="script-list"]').inner_text() == "No script for a blank conversation."


def test_copy_buttons_on_the_model_down_banner(make_desk: Any, ui_server_factory: Any) -> None:
    server = ui_server_factory(DESK_LLM="llama", DESK_LLAMA_URL=f"http://127.0.0.1:{ui_server_factory.free_port()}")
    granted = make_desk(server.url, permissions=["clipboard-read", "clipboard-write"])
    granted.page.locator('[data-testid="model-down"]').wait_for()
    button = granted.page.locator('[data-testid="model-down"] .cmd button').first
    button.click()
    assert button.inner_text() == "Copied"
    assert granted.page.evaluate("() => navigator.clipboard.readText()").startswith("scripts/run.sh")
    denied = make_desk(server.url, before_goto=lambda page: page.add_init_script(
        "navigator.clipboard.writeText = () => Promise.reject(new Error('denied'))"))
    denied.page.locator('[data-testid="model-down"]').wait_for()
    button = denied.page.locator('[data-testid="model-down"] .cmd button').first
    button.click()
    denied.page.wait_for_function("() => document.querySelector('[data-testid=model-down] .cmd button').textContent === 'Select and copy'")
    denied.page.click('[data-testid="model-down"] button:has-text("Check again")')
    assert denied.page.get_attribute("#llm-dot", "data-state") == "down"


def crafted_error_turn(base_url: str, session_id: str, entries: list[dict[str, Any]], decision: str = "error") -> str:
    """An SSE answer whose turn ended in `decision` with the given audit entries (UI rendering only)."""
    state = httpx.get(f"{base_url}/api/sessions/{session_id}/state").json()
    seq = len(state["action_audit"])
    entries = [{"seq": seq + i + 1, "turn": 1, "arguments": {}, "result": None, "committed_effects": [], "visible_to_agent": True, **e}
               for i, e in enumerate(entries)]
    output = {"decision": decision, "response": "The final step did not answer in time.", "proposal": None, "transaction": None,
              "confirmed_proposal_id": None}
    record = {"turn": 1, "message": "hello", "confirm_proposal_id": None, "output": output, "latency_ms": 12.0}
    state = {**state, "action_audit": state["action_audit"] + entries, "turns": [record]}
    return sse(("start", {"turn": 1, "message": "hello", "confirm_proposal_id": None}),
               *[("audit", e) for e in entries], ("turn", {**record, "state": state}))


@pytest.mark.parametrize(("entries", "expected"), [
    ([{"kind": "tool", "agent": "operations", "tool": "execute_proposal", "error": {"code": "TIMEOUT", "message": "lost"},
       "visible_to_agent": False}], "timed out and is not verified yet"),
    ([], "could not finish this turn"),
])
def test_error_notices_explain_what_happened(desk: Any, fake_ui: Any, entries: list[dict[str, Any]], expected: str) -> None:
    desk.open("refund_boundary")
    session_id = desk.page.evaluate("() => window.ReturnsDesk.app.session.session_id")
    body = crafted_error_turn(fake_ui.url, session_id, entries)
    desk.page.route("**/api/sessions/*/turns", lambda route: route.fulfill(status=200, content_type=SSE, body=body))
    send(desk, "hello")
    assert desk.wait_turn(1) == "error"
    assert expected in desk.message(1).locator('[data-testid="error-notice"]').inner_text()
    assert desk.page.locator('[data-decision="completed"]').count() == 0


def test_a_deny_that_code_decided_on_the_facts_names_the_reason(desk: Any, fake_ui: Any) -> None:
    """The desk may deny without any tool call (guard code_deny): the notice still names the rule."""
    desk.open("refund_boundary")
    session_id = desk.page.evaluate("() => window.ReturnsDesk.app.session.session_id")
    denied = {"kind": "guard", "agent": "orchestrator", "tool": "code_deny", "arguments": {"order_id": "O100", "line_id": "L1"},
              "result": {"reasons": ["FINAL_SALE"]}, "error": None}
    body = crafted_error_turn(fake_ui.url, session_id, [denied], decision="deny")
    desk.page.route("**/api/sessions/*/turns", lambda route: route.fulfill(status=200, content_type=SSE, body=body))
    send(desk, "hello")
    assert desk.wait_turn(1) == "deny"
    meta = httpx.get(f"{fake_ui.url}/api/meta").json()["reason_text"]
    notice = desk.message(1).locator('[data-testid="deny-notice"]').inner_text()
    assert "Not possible under the return rules" in notice and meta["FINAL_SALE"] in notice


def test_a_deny_reported_by_the_read_only_check_names_the_reason(desk: Any, fake_ui: Any) -> None:
    """Operations ended without a decision; code's read-only check_proposal dry run found the refusal."""
    desk.open("refund_boundary")
    session_id = desk.page.evaluate("() => window.ReturnsDesk.app.session.session_id")
    checked = {"kind": "tool", "agent": "orchestrator", "tool": "check_proposal", "arguments": {"remedy": "refund"},
               "result": None, "error": {"code": "OPENED", "message": "refused", "details": {"reasons": ["OPENED"]}}}
    body = crafted_error_turn(fake_ui.url, session_id, [checked], decision="deny")
    desk.page.route("**/api/sessions/*/turns", lambda route: route.fulfill(status=200, content_type=SSE, body=body))
    send(desk, "hello")
    assert desk.wait_turn(1) == "deny"
    meta = httpx.get(f"{fake_ui.url}/api/meta").json()["reason_text"]
    notice = desk.message(1).locator('[data-testid="deny-notice"]').inner_text()
    assert "Not possible under the return rules" in notice and meta["OPENED"] in notice


def entry(kind: str, tool: str, agent: str = "orchestrator", **fields: Any) -> dict[str, Any]:
    return {"seq": 1, "turn": 1, "kind": kind, "agent": agent, "tool": tool, "arguments": {}, "result": None, "error": None,
            "committed_effects": [], "visible_to_agent": True, **fields}


def err(code: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": code, **({"details": details} if details else {})}


MAPPING = [
    (entry("gate", "handoff_to_operations", arguments={"order_id": "ORD-99"}, error=err("ORDER_NOT_FOUND")), "grounding", "blocked", "unknown order ORD-99"),
    (entry("gate", "handoff_to_operations", arguments={"order_id": "ORD-7Q", "line_id": "LN-9"}, error=err("LINE_NOT_FOUND")), "grounding", "blocked", "unknown line LN-9 in ORD-7Q"),
    (entry("gate", "handoff_to_operations", error=err("SIZE_UNSPECIFIED")), "grounding", "blocked", "rejected handoff: We need to know which size"),
    (entry("guard", "clarify_veto", arguments={"intent": "propose_refund"}), "grounding", "blocked", "veto: line and remedy are clear, hand off propose_refund"),
    (entry("guard", "grant_withheld", result={"reason": "cancel wording"}), "approval", "blocked", "withheld: cancel wording"),
    (entry("guard", "grant_withheld"), "approval", "blocked", "withheld: the message changes the request"),
    (entry("guard", "reply_guard", arguments={"decision": "propose"}, result={"violations": ["success_claim:processed"]}), "replyguard", "blocked", 'model said "processed" on a propose; template used'),
    (entry("guard", "reply_guard", arguments={"decision": "clarify"}, result={"violations": ["no_question", "size:XL"]}), "replyguard", "blocked", "template used: no_question, size:XL"),
    (entry("guard", "report_mismatch", arguments={"claimed": "denied"}, result={"actual": "proposed"}), "replyguard", "blocked", 'Ops said "denied", records say "proposed"'),
    (entry("guard", "duplicate_call", arguments={"tool": "create_proposal"}), "orchestrator", "info", "repeat call answered from cache: create_proposal"),
    (entry("guard", "dropped_calls"), "orchestrator", "info", "extra tool calls in one answer were dropped"),
    (entry("guard", "something_new"), "orchestrator", "info", "guard something_new"),
    (entry("guard", "code_deny", result={"reasons": ["OPENED"]}), "policy", "blocked", "code said no on the facts: The item has been opened"),
    (entry("guard", "handoffs_closed", result={"reason": "3 rejected handoffs; only a question is allowed now"}), "orchestrator", "info",
     "handoffs closed: 3 rejected handoffs"),
    (entry("guard", "handoffs_closed"), "orchestrator", "info", "handoffs closed: only a question is allowed now"),
    (entry("confirmation", "confirmation_unresolved"), "approval", "blocked", "no earlier proposal to confirm"),
    (entry("route", "idempotency_preamble", result={"replay": False}), "recovery", "blocked", "request already completed for another item"),
    (entry("fallback", "check_proposal", result={"reason": "no decision by the agent; read-only check"}), "orchestrator", "info", "read-only check by code: check_proposal"),
    (entry("fallback", "get_request_status"), "orchestrator", "info", "read-only check by code: get_request_status"),
    (entry("fallback", "something_else"), "orchestrator", "info", "read-only check by code: something_else"),
    (entry("fault", "stock_change", agent="external", arguments={"sku": "NOPE-1"}, result={"applied": False}), "warehouse", "blocked", "unknown SKU NOPE-1: ignored"),
    (entry("fault", "power_cut", agent="external", error=err("UNKNOWN_FAULT_TYPE")), "warehouse", "blocked", "fault ignored: UNKNOWN_FAULT_TYPE"),
    (entry("brand_new_kind", "x"), "orchestrator", "info", "brand_new_kind · x"),
    (entry("tool", "find_order_lines", agent="resolution", result={"count": 0, "order_not_on_account": True}), "grounding", "info", "lookup: 0 lines (order not on this account)"),
    (entry("tool", "find_order_lines", agent="resolution", result={"count": 2}), "grounding", "info", "lookup: 2 lines"),
    (entry("tool", "check_eligibility", arguments={"order_id": "O100", "line_id": "L1"}, result={"eligible": True, "days_since_delivery": 3, "window_days": 30}), "policy", "ok", "eligible · day 3 of 30"),
    (entry("tool", "check_eligibility", arguments={"order_id": "O100", "line_id": "L1"}, result={"eligible": False, "reasons": ["OPENED"]}), "policy", "blocked", "The item has been opened"),
    (entry("tool", "check_eligibility", error=err("NO_TARGET")), "policy", "blocked", "We could not tell which item you mean."),
    (entry("tool", "check_proposal", arguments={"remedy": "refund"}, result={"eligible": True}), "policy", "info", "dry run: would be eligible · no proposal made"),
    (entry("tool", "check_proposal", arguments={"remedy": "refund"}, error=err("FINAL_SALE", reasons=["FINAL_SALE", "OPENED"])), "policy", "blocked", "dry run: The item was bought as final sale"),
    (entry("tool", "check_proposal", arguments={"remedy": "exchange"}, error=err("SIZE_UNSPECIFIED")), "policy", "info", "dry run: We need to know which size"),
    (entry("tool", "list_exchange_options", agent="operations", error=err("REQUEST_ALREADY_COMPLETED")), "policy", "info", "options: REQUEST_ALREADY_COMPLETED"),
    (entry("tool", "create_proposal", agent="operations", error=err("SIZE_UNSPECIFIED")), "approval", "blocked", "waiting for the customer"),
    (entry("tool", "create_proposal", agent="operations", error=err("REPLACEMENT_NOT_REQUESTED")), "gateway", "blocked", "refused REPLACEMENT_NOT_REQUESTED (agent slip"),
    (entry("tool", "get_request_status", agent="operations", result={"status": "not_found"}), "recovery", "info", "status: no completed record"),
    (entry("tool", "get_request_status", agent="orchestrator", result={"status": "completed", "record": {"transaction_id": "TX-9"}}), "recovery", "ok", "status: TX-9 already completed"),
    (entry("tool", "execute_proposal", agent="operations", result={"replay": True, "transaction": {"transaction_id": "TX-9"}}), "recovery", "ok", "replay: TX-9 already processed"),
    (entry("tool", "execute_proposal", agent="operations", error=err("CONFIRMATION_REQUIRED")), "approval", "blocked", "no explicit confirmation this turn"),
    (entry("tool", "execute_proposal", agent="operations", error=err("BINDING_MISMATCH")), "gateway", "blocked", "refused BINDING_MISMATCH"),
    (entry("tool", "brand_new_tool", agent="operations", error=err("INTERNAL")), "gateway", "blocked", "brand_new_tool INTERNAL"),
    (entry("tool", "brand_new_tool", agent="orchestrator"), "gateway", "info", "brand_new_tool"),
    (entry("llm_step", "create_proposal", agent="operations", error=err("INVALID_ARGS"), result={}), "operations", "blocked", "protocol error INVALID_ARGS"),
]


@pytest.mark.parametrize(("audit", "node", "status", "text"), MAPPING, ids=[f"{m[0]['kind']}-{m[0]['tool']}-{i}" for i, m in enumerate(MAPPING)])
def test_team_view_maps_every_kind_of_audit_entry(desk: Any, audit: dict[str, Any], node: str, status: str, text: str) -> None:
    described = desk.page.evaluate("e => window.ReturnsDesk.describeEntry(e, {handoff: null, opsActive: true, timedOut: false})", audit)
    main = [f for f in described["fx"] if not f.get("edgeOnly")][-1]
    assert (main["node"], main["status"]) == (node, status)
    assert text in main["text"]
    assert described["narration"]
    title = desk.page.evaluate("e => window.ReturnsDesk.entryTitle(e)", audit)
    assert title  # every entry gets a readable one-line title in the audit trail


def test_audit_titles_name_the_important_facts(desk: Any) -> None:
    titles = {
        "target": entry("target", "set_target", arguments={"order_id": "ORD-7Q", "line_id": "LN-3", "intent": "exchange", "requested_size": "L"}),
        "grant": entry("grant", "set_grant", arguments={"proposal_id": "P-0003"}),
        "route": entry("route", "idempotency_preamble", result={"replay": True, "transaction_id": "TX-0002"}),
        "fault": entry("fault", "stock_change", agent="external", result={"sku": "SCARF-L", "old_stock": 4, "new_stock": 0, "applied": True}),
        "armed": entry("fault", "timeout_after_commit", agent="external", result={"armed": True}),
        "guard": entry("guard", "reply_guard", result={"violations": ["success_claim:done"]}),
        "deny": entry("guard", "code_deny", result={"reasons": ["OPENED"]}),
        "unknown": entry("brand_new_kind", "x"),
    }
    got = {k: desk.page.evaluate("e => window.ReturnsDesk.entryTitle(e)", v) for k, v in titles.items()}
    assert got["target"] == "grounded target ORD-7Q/LN-3 · exchange (size L)"
    assert got["grant"] == "explicit confirmation of P-0003 (this turn only)"
    assert got["route"] == "already completed: replay TX-0002, no new effects"
    assert got["fault"] == "external stock change SCARF-L 4 → 0"
    assert got["armed"] == "timeout_after_commit: armed for the next commit"
    assert got["guard"] == "reply_guard: success_claim:done"
    assert got["deny"].startswith("code_deny: The item has been opened")
    assert got["unknown"] == "x"
