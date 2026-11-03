"""End-to-end UI tests: every public scenario through the browser, plus the failure and recovery paths.

Each test drives the real page against a real server process (fake-policy model, real desk). Business
outcomes are checked in the State inspector against the independent oracle's labels
(evals/expected/public_expected.json), not against what the server says about itself.

Screenshots of the key states are written to var/screenshots/ for the presentation and the README.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(180)]

SCENARIOS = ["refund_boundary", "exchange", "expired", "clarify", "stale_stock", "timeout_replay"]


def money(cents: int) -> str:
    return f"${cents // 100:,}.{cents % 100:02d}"


def assert_inspector_matches_labels(desk: Any) -> None:
    """Stock per SKU, returned flags and record counts in the inspector equal the oracle's final state."""
    expected = desk.label["final_business_state"]
    desk.open_inspector()
    for product in expected["products"]:
        assert desk.stock(product["sku"]) == product["stock"], product["sku"]
    for order in expected["orders"]:
        for line in order["lines"]:
            assert desk.returned(order["order_id"], line["line_id"]) is line["returned"]
    for collection in ("refunds", "exchanges", "returns", "quarantine"):
        assert desk.rows(f"records-{collection}") == len(expected[collection]), collection
    for refund in expected["refunds"]:
        assert money(refund["amount_cents"]) in desk.page.locator('[data-testid="records-refunds"]').inner_text()
    assert desk.rows("records-idempotency") == len(expected["idempotency"])


@pytest.mark.parametrize("suffix", SCENARIOS)
def test_every_public_scenario_end_to_end(desk: Any, suffix: str) -> None:
    desk.open(suffix)
    decisions = desk.run_script()
    assert decisions == desk.label["decisions"]
    assert_inspector_matches_labels(desk)
    # Never a success pill without a completed decision, and a completed turn shows the full stepper.
    for number, decision in enumerate(decisions, start=1):
        pill = desk.message(number).locator('[data-testid="decision"]').inner_text().strip().lower()
        assert pill == decision
        if decision == "completed":
            assert desk.message(number).locator('[data-step="completed"][data-state="done"]').count() == 1
        else:
            assert desk.message(number).locator('[data-step="completed"][data-state="done"]').count() == 0
    desk.shot(f"scenario_{suffix}.png")


def test_proposal_card_shows_amount_item_facts_and_binding(desk: Any) -> None:
    desk.open("refund_boundary")
    assert desk.fill_and_send(1) == "propose"
    card = desk.page.locator('[data-testid="proposal-card"]')
    line = desk.case["initial_state"]["orders"][0]["lines"][0]
    assert card.locator('[data-testid="proposal-amount"]').inner_text() == money(line["paid_cents"])
    text = card.inner_text()
    assert "day 30 of 30" in text and "unopened" in text and "not final sale" in text and line["sku"] in text
    assert "binding " in text
    assert card.locator('[data-testid="confirm"]').inner_text() == f"Confirm refund {money(line['paid_cents'])}"
    assert card.locator('[data-testid="proposal-status"]').inner_text().lower() == "open"
    desk.shot("01_proposal_card.png")
    assert desk.confirm(2) == "completed"
    assert card.locator('[data-testid="proposal-status"]').inner_text().lower() == "executed"
    assert desk.page.locator('[data-testid="confirm"]').count() == 0  # nothing left to approve
    desk.open_inspector()
    desk.shot("02_completed_with_inspector.png")


def test_clarification_flow_asks_then_proposes_the_named_line(desk: Any) -> None:
    desk.open("clarify")
    assert desk.fill_and_send(1) == "clarify"
    first = desk.message(1)
    assert first.locator('[data-testid="proposal-card"]').count() == 0
    assert "?" in first.locator('[data-testid="response"]').inner_text()
    for line in desk.case["initial_state"]["orders"][0]["lines"]:
        assert line["line_id"] in first.inner_text()  # the question lists the candidate lines
    desk.shot("03_clarify.png")
    assert desk.fill_and_send(2) == "propose"
    assert desk.page.locator('[data-testid="proposal-card"]').count() == 1
    assert desk.confirm(3) == "completed"


def test_policy_deny_for_an_expired_window(desk: Any) -> None:
    desk.open("expired")
    assert desk.fill_and_send(1) == "deny"
    notice = desk.message(1).locator('[data-testid="deny-notice"]')
    assert "30-day return window" in notice.inner_text()
    assert desk.page.locator('[data-testid="proposal-card"]').count() == 0
    desk.wait_team_caught_up()
    policy = desk.page.locator('[data-node="policy"]')
    assert policy.get_attribute("data-status") == "blocked" and "30-day" in desk.node_status("policy")
    assert_inspector_matches_labels(desk)
    desk.shot("04_expired_deny.png")


def test_stale_stock_from_the_failure_lab_is_denied_with_alternatives(desk: Any) -> None:
    desk.open("exchange")  # no scheduled fault: the stock change comes from the failure lab
    assert desk.fill_and_send(1) == "propose"
    replacement = desk.page.locator('[data-testid="proposal-replacement"]').inner_text()
    assert "3 in stock now" in desk.page.locator('[data-testid="replacement-stock"]').inner_text()
    desk.page.select_option('[data-testid="stock-sku"]', replacement)
    desk.page.fill('[data-testid="stock-value"]', "0")
    desk.page.click('[data-testid="stock-apply"]')
    desk.page.wait_for_selector('[data-testid="replacement-stock"]:has-text("0 in stock now")')  # the card shows live stock
    assert desk.confirm(2) == "deny"
    message = desk.message(2)
    assert message.locator('[data-step="revalidating"][data-state="failed"]').count() == 1
    notice = message.locator('[data-testid="deny-notice"]').inner_text()
    assert "out of stock" in notice and f"{replacement} 3 → 0" in notice
    alternatives = message.locator('[data-testid="alternatives"]').inner_text()
    assert "Exchange for size S instead" in alternatives and "size L" not in alternatives
    desk.open_inspector()
    assert desk.rows("records-exchanges") == 0 and desk.stock(replacement) == 0
    assert desk.rows("records-events") == 1
    desk.shot("05_stale_stock_deny.png")


def test_timeout_after_commit_recovers_and_a_repeat_replays_with_no_new_effects(desk: Any) -> None:
    desk.open("timeout_replay")
    assert "Armed" in desk.page.locator('[data-testid="fault-status"]').inner_text()
    assert desk.fill_and_send(1) == "propose"
    assert desk.confirm(2) == "completed"
    second = desk.message(2)
    assert second.locator('[data-step="recovered"][data-state="recovered"]').count() == 1
    assert "did not commit a second time" in second.locator('[data-testid="recovered-notice"]').inner_text()
    desk.wait_team_caught_up()
    recovery = desk.page.locator('[data-node="recovery"]')
    assert recovery.get_attribute("data-status") == "ok" and "reconciled: TX-0001 found, no second commit" in desk.node_status("recovery")
    desk.shot("team_recovery.png")
    assert desk.fill_and_send(3) == "completed"  # the scripted repeat carries the explicit confirmation
    third = desk.message(3)
    assert third.locator('[data-step="replay"]').count() == 1
    assert "No new refund" in third.locator('[data-testid="replay-notice"]').inner_text()
    assert desk.committed_entries() == 1  # one commit in the whole audit
    assert_inspector_matches_labels(desk)
    desk.shot("06_timeout_recovered_then_replay.png")


def test_double_click_on_confirm_commits_once(desk: Any) -> None:
    desk.open("refund_boundary")
    assert desk.fill_and_send(1) == "propose"
    proposal_id = desk.page.get_attribute('[data-testid="proposal-card"]', "data-proposal")
    desk.page.dblclick('[data-testid="confirm"]')
    assert desk.wait_turn(2) == "completed"
    assert desk.page.locator('[data-testid="desk-message"]').count() == 2  # the second click sent nothing
    # Even a second explicit approval reaching the server is a replay with no new effects.
    session_id = desk.page.evaluate("() => window.ReturnsDesk.app.session.session_id")
    replay = desk.page.evaluate(
        """async ([sid, pid]) => {
             const shown = window.ReturnsDesk.app.session;  // what this page displays: the card's binding
             const r = await fetch(`/api/sessions/${sid}/turns`, {method: "POST", headers: {"Content-Type": "application/json"},
               body: JSON.stringify({message: "Yes, I confirm this proposal.", confirm_proposal_id: pid,
                                     confirm_binding_hash: shown.cards[pid].binding_hash, generation: shown.generation})});
             const text = await r.text();
             const turn = text.split("\\n\\n").find((b) => b.startsWith("event: turn"));
             return JSON.parse(turn.split("\\ndata: ")[1]);
           }""", [session_id, proposal_id])
    assert replay["output"]["decision"] == "completed"
    assert len(replay["state"]["final_state"]["refunds"]) == 1
    assert sum(1 for e in replay["state"]["action_audit"] if e["committed_effects"]) == 1


def test_model_down_shows_the_start_command_and_never_a_success(make_desk: Any, ui_server_factory: Any) -> None:
    # A real-model server whose llama-server URL has nothing listening (the factory stops it afterwards).
    server = ui_server_factory(DESK_LLM="llama", DESK_LLAMA_URL=f"http://127.0.0.1:{ui_server_factory.free_port()}")
    desk = make_desk(server.url)
    banner = desk.page.locator('[data-testid="model-down"]')
    banner.wait_for()
    commands = banner.locator('[data-testid="start-command"]').all_inner_texts()
    assert commands[0].startswith("scripts/run.sh --llama-port ")
    assert commands[1].startswith("llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --port ")
    assert desk.page.get_attribute("#llm-dot", "data-state") == "down"
    desk.open("refund_boundary")
    assert desk.fill_and_send(1) == "error"
    notice = desk.message(1).locator('[data-testid="error-notice"]').inner_text()
    assert "did not answer" in notice and "nothing was changed" in notice
    assert desk.page.locator('[data-decision="completed"]').count() == 0
    assert desk.page.locator('[data-testid="proposal-card"]').count() == 0
    desk.open_inspector()
    assert desk.rows("records-refunds") == 0 and desk.rows("records-proposals") == 0
    desk.shot("08_model_down.png")


def test_not_now_cancels_and_nothing_moves(desk: Any) -> None:
    desk.open("refund_boundary")
    assert desk.fill_and_send(1) == "propose"
    desk.page.click('[data-testid="not-now"]')
    assert desk.wait_turn(2) == "clarify"
    assert "nothing was processed" in desk.message(2).locator('[data-testid="response"]').inner_text()
    desk.open_inspector()
    assert desk.rows("records-refunds") == 0 and desk.rows("records-idempotency") == 0
    assert desk.page.locator('[data-testid="proposal-status"]').inner_text().lower() == "open"  # still confirmable later
    assert desk.page.locator('[data-testid="confirm"]').count() == 1
    assert desk.confirm(3) == "completed"


def test_reset_restores_the_initial_state(desk: Any) -> None:
    desk.open("refund_boundary")
    assert desk.run_script() == ["propose", "completed"]
    desk.open_inspector()
    assert desk.rows("records-refunds") == 1
    desk.page.click('[data-testid="reset"]')
    desk.page.wait_for_function("() => window.ReturnsDesk.app.session.turns.length === 0 && window.ReturnsDesk.app.session.generation === 2")
    assert desk.page.locator('[data-testid="desk-message"]').count() == 0
    assert desk.rows("records-refunds") == 0 and desk.rows("records-quarantine") == 0
    line = desk.case["initial_state"]["orders"][0]["lines"][0]
    assert desk.returned(desk.case["initial_state"]["orders"][0]["order_id"], line["line_id"]) is False
    assert desk.page.get_attribute('[data-testid="script-list"] li[data-next="true"] button', "data-testid") == "fill-1"
    assert desk.run_script() == ["propose", "completed"]  # the same request runs again from scratch
    desk.shot("09_after_reset_and_rerun.png")


def test_new_request_keeps_the_data_and_starts_an_empty_chat(desk: Any) -> None:
    desk.open("refund_boundary")
    assert desk.run_script() == ["propose", "completed"]
    first = desk.page.evaluate("() => window.ReturnsDesk.app.session.request_id")
    desk.page.click('[data-testid="new-request"]')
    desk.page.wait_for_function("() => window.ReturnsDesk.app.session.turns.length === 0 && window.ReturnsDesk.app.session.generation === 2")
    assert desk.page.evaluate("() => window.ReturnsDesk.app.session.request_id") != first
    assert desk.page.locator('[data-testid="desk-message"]').count() == 0
    assert "kept" in desk.page.inner_text('[data-testid="progress"]')
    desk.open_inspector()
    order = desk.case["initial_state"]["orders"][0]
    assert desk.rows("records-refunds") == 1 and desk.returned(order["order_id"], order["lines"][0]["line_id"]) is True
    desk.page.fill('[data-testid="composer-input"]', desk.case["turns"][0]["message"])
    desk.page.click('[data-testid="send"]')
    assert desk.wait_turn(1) == "deny"  # already refunded under the first request
    assert desk.rows("records-refunds") == 1
    desk.shot("09b_new_request_keeps_the_data.png")


def test_keyboard_only_flow_and_live_regions(desk: Any) -> None:
    desk.open("refund_boundary")
    assert desk.page.get_attribute('[data-testid="progress"]', "aria-live") == "polite"
    assert desk.page.get_attribute('[data-testid="mission-now"]', "aria-live") == "polite"
    desk.page.focus('[data-testid="composer-input"]')
    desk.page.keyboard.type(desk.case["turns"][0]["message"])
    desk.page.keyboard.press("Enter")
    assert desk.wait_turn(1) == "propose"
    desk.page.focus('[data-testid="confirm"]')
    desk.page.keyboard.press("Enter")
    assert desk.wait_turn(2) == "completed"


def test_audit_entries_stream_in_while_the_turn_is_still_running(desk: Any) -> None:
    desk.open("refund_boundary")
    desk.page.click('[data-testid="fill-1"]')
    desk.page.click('[data-testid="send"]')
    # Entries are rendered before the decision exists: streamed, not batched at the end.
    desk.page.wait_for_function(
        "() => document.querySelector('[data-testid=desk-message][data-decision=working]') "
        "&& document.querySelectorAll('[data-turn-group=\"1\"] [data-testid=audit-entry]').length >= 3")
    assert desk.wait_turn(1) == "propose"


JARGON = re.compile(r"\b\w+_\w+\b|\w\(\)")  # snake_case names, CODE_NAMES and tool() calls


def test_every_audit_step_says_what_it_does_in_plain_words(desk: Any) -> None:
    desk.open("exchange")
    assert desk.fill_and_send(1) == "propose"
    desk.page.click('[data-testid="tab-audit"]')
    entries = desk.page.locator('[data-turn-group="1"] [data-testid="audit-entry"]')
    plain = [entries.nth(i).locator('[data-testid="entry-plain"]') for i in range(entries.count())]
    assert len(plain) >= 5
    for line in plain:
        text = line.inner_text()
        assert line.is_visible() and text.endswith(".") and not JARGON.search(text), text
    texts = " ".join(line.inner_text() for line in plain)
    assert "Resolution passes the case to Operations" in texts and "Operations reports back to Resolution" in texts


def test_audit_trail_groups_turns_and_highlights_effects(desk: Any) -> None:
    desk.open("timeout_replay")
    desk.run_script()
    desk.page.click('[data-testid="tab-audit"]')
    timeline = desk.page.locator('[data-testid="audit-timeline"]')
    assert timeline.locator("[data-turn-group]").count() == 4  # setup + 3 turns
    assert timeline.locator('[data-handoff="true"]').count() >= 4
    committed = timeline.locator('[data-effects="true"]')
    assert committed.count() == 1 and "hidden from agent" in committed.inner_text()
    committed.locator("summary").click()
    body = committed.inner_text()
    assert "refund RF-0001" in body and "quarantine Q-0001" in body and "TIMEOUT" in body
    llm_step = timeline.locator('[data-kind="llm_step"]').first
    assert " tok" in llm_step.inner_text() and " ms" in llm_step.inner_text()
    desk.shot("10_audit_trail.png")


def test_blank_conversation_for_a_customer_without_orders(desk: Any) -> None:
    desk.page.select_option('[data-testid="scenario-select"]', "__blank__")
    desk.page.select_option('[data-testid="customer-select"]', "GUEST-1")
    desk.page.wait_for_function("() => window.ReturnsDesk.app.session.customer_id === 'GUEST-1'")
    desk.page.fill('[data-testid="composer-input"]', "Please refund my order.")
    desk.page.click('[data-testid="send"]')
    assert desk.wait_turn(1) in ("clarify", "deny")
    assert desk.page.locator('[data-testid="proposal-card"]').count() == 0


def test_mobile_layout_has_no_horizontal_scroll(make_desk: Any) -> None:
    desk = make_desk(viewport={"width": 390, "height": 844})
    desk.open("refund_boundary")
    assert desk.fill_and_send(1) == "propose"
    overflow = desk.page.evaluate("() => document.documentElement.scrollWidth - window.innerWidth")
    assert overflow <= 0
    desk.page.locator('[data-testid="proposal-card"]').scroll_into_view_if_needed()
    desk.shot("11_mobile.png")


def test_the_narrowest_phone_has_no_horizontal_scroll(make_desk: Any) -> None:
    desk = make_desk(viewport={"width": 375, "height": 812})  # the reviewers' phone width
    desk.open("refund_boundary")
    assert desk.page.evaluate("() => document.documentElement.scrollWidth - window.innerWidth") <= 0


def test_dark_mode_uses_the_dark_tokens(make_desk: Any) -> None:
    desk = make_desk(color_scheme="dark")
    desk.open("exchange")
    assert desk.fill_and_send(1) == "propose"
    background = desk.page.evaluate("() => getComputedStyle(document.body).backgroundColor")
    assert background == "rgb(15, 17, 21)"
    desk.shot("12_dark_mode.png")


def test_theme_switch_overrides_the_system_setting_and_is_remembered(make_desk: Any) -> None:
    desk = make_desk(color_scheme="dark")
    pressed = '[data-testid="theme-switch"] [aria-pressed="true"]'
    background_is = "rgb => getComputedStyle(document.body).backgroundColor === rgb"
    assert desk.page.get_attribute(pressed, "data-theme-choice") == "dark"
    desk.page.click('[data-testid="theme-switch"] [data-theme-choice="light"]')
    desk.page.wait_for_function(background_is, arg="rgb(246, 247, 249)")
    assert desk.page.get_attribute(pressed, "data-theme-choice") == "light"
    desk.page.reload()
    desk.page.wait_for_function(background_is, arg="rgb(246, 247, 249)")
    assert desk.page.get_attribute(pressed, "data-theme-choice") == "light"
    desk.page.click('[data-testid="theme-switch"] [data-theme-choice="dark"]')
    desk.page.wait_for_function(background_is, arg="rgb(15, 17, 21)")


def test_fill_next_offers_the_next_scripted_turn_beside_send(desk: Any) -> None:
    desk.open("refund_boundary")
    assert desk.page.inner_text('[data-testid="empty-fill"]') == "Fill message 1"
    desk.page.click('[data-testid="empty-fill"]')
    assert desk.page.input_value("#composer-input") == desk.case["turns"][0]["message"]
    assert desk.page.is_hidden('[data-testid="fill-next"]')  # the box already holds a message
    desk.page.click('[data-testid="send"]')
    assert desk.wait_turn(1) == "propose"
    assert desk.page.inner_text('[data-testid="fill-next"]') == "Fill message 2"
    desk.page.click('[data-testid="fill-next"]')
    assert desk.page.input_value("#composer-input") == desk.case["turns"][1]["message"]
