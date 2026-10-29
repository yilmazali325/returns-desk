"""E2E tests for the review fixes in the UI (docs/review/security.md, contract.md):

- SEC-1     a stale tab's Confirm cannot approve a proposal it never displayed (409, nothing moves)
- NEW-1     the card's Confirm button sends a neutral confirmation, never the scenario's next script line
- NEW-2     the composer's confirmation box clears after every send and never moves to a newer proposal
- C10       no confirmation is offered for a dead proposal; a deny's alternatives are actionable chips
- C6        the Team view is readable at 1280x720 and 1440x900 (computed font sizes, no overflow, goal
            lines never cut), theater fits the viewport, phones get a card list; screenshots of the Team
            view at 1280x720 (normal and theater) go to var/e2e/shots/
- audit     entry kinds this page does not know yet, and agent-grounded entries, still render; a real
            fake-model turn that picks a line by citing the customer's words shows that evidence
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(180)]

CONFIRMATION = "Yes, I confirm this proposal."


def state(desk: Any) -> dict[str, Any]:
    return desk.page.evaluate("() => window.ReturnsDesk.app.session")


def customer_bubble(desk: Any, turn: int) -> str:
    return desk.page.locator(f'.msg-customer[data-turn="{turn}"] .bubble').inner_text()


# ================================================================================================
# SEC-1: two tabs on one session (they share it through localStorage)


@pytest.mark.http_errors
def test_a_stale_tab_cannot_confirm_a_proposal_it_never_displayed(make_desk: Any) -> None:
    tab_b = make_desk().open("clarify")
    tab_b.page.fill('[data-testid="composer-input"]', "Refund line L1 of order O100 please.")
    tab_b.page.click('[data-testid="send"]')
    assert tab_b.wait_turn(1) == "propose"
    assert tab_b.page.locator('[data-testid="confirm"]').inner_text() == "Confirm refund $30.00"
    session = state(tab_b)
    # Tab A (another window on the same session): reset, then a new P-0001 for the $90 jacket.
    with httpx.Client(base_url=tab_b.base_url, timeout=30) as api:
        assert api.post(f"/api/sessions/{session['session_id']}/reset", json={}).status_code == 200
        api.post(f"/api/sessions/{session['session_id']}/turns", json={"message": "Refund line L2 of order O100 please."})
        current = api.get(f"/api/sessions/{session['session_id']}/state").json()
    assert next(iter(current["cards"].values()))["line_id"] == "L2"
    tab_b.page.click('[data-testid="confirm"]')  # the stale $30.00 button
    notice = tab_b.page.locator('[data-testid="notice"]')
    notice.wait_for()
    assert "Not sent" in notice.inner_text() and "proposal changed" in notice.inner_text()
    tab_b.page.wait_for_function("() => !window.ReturnsDesk.app.busy")
    after = state(tab_b)
    assert after["final_state"]["refunds"] == [] and after["generation"] == session["generation"] + 1
    assert tab_b.page.locator('[data-testid="proposal-card"]').inner_text().count("$90.00") >= 1  # now shows what is real


def test_the_card_and_the_composer_send_the_binding_of_what_they_show(desk: Any) -> None:
    sent: list[dict[str, Any]] = []
    desk.page.on("request", lambda r: sent.append(r.post_data_json) if r.url.endswith("/turns") else None)
    desk.open("refund_boundary")
    assert desk.fill_and_send(1) == "propose"
    card = next(iter(state(desk)["cards"].values()))
    assert desk.confirm(2) == "completed"
    assert sent[-1] == {"message": CONFIRMATION, "confirm_proposal_id": card["proposal_id"],
                        "confirm_binding_hash": card["binding_hash"], "generation": state(desk)["generation"]}
    assert sent[0]["confirm_proposal_id"] is None and "confirm_binding_hash" not in sent[0]


# ================================================================================================
# NEW-1: the Confirm button never sends the script's wording for another remedy


def test_confirm_on_an_off_script_exchange_card_executes_the_exchange(desk: Any) -> None:
    desk.open("refund_boundary")  # the script would say "Yes, refund the proposed item and amount."
    desk.page.fill('[data-testid="composer-input"]', "Please exchange line L1 of order O100 for size L.")
    desk.page.click('[data-testid="send"]')
    assert desk.wait_turn(1) == "propose"
    assert desk.page.locator('[data-testid="confirm"]').inner_text() == "Confirm exchange to TEE-L"
    assert desk.confirm(2) == "completed"
    assert customer_bubble(desk, 2) == CONFIRMATION
    assert len(state(desk)["final_state"]["exchanges"]) == 1


# ================================================================================================
# NEW-2: one tick, one message, one proposal


def test_the_confirmation_box_clears_after_a_send_and_never_retargets(desk: Any) -> None:
    desk.open("refund_boundary")
    assert desk.fill_and_send(1) == "propose"
    box = desk.page.locator('[data-testid="confirm-attach"]')
    box.check()
    desk.page.fill('[data-testid="composer-input"]', "Please exchange it for size L.")
    desk.page.click('[data-testid="send"]')
    desk.wait_turn(2)
    assert not box.is_checked()  # cleared by the send, whatever the desk decided
    latest = state(desk)["turns"][-1]["output"]
    if latest["proposal"]:  # a new proposal replaced P-0001: the box names it, unticked
        assert desk.page.locator("#confirm-attach-id").inner_text() == latest["proposal"]["proposal_id"]
    desk.page.fill('[data-testid="composer-input"]', "When was my order delivered?")
    desk.page.click('[data-testid="send"]')
    desk.wait_turn(3)
    record = state(desk)["turns"][-1]
    assert record["confirm_proposal_id"] is None and record["output"]["decision"] != "completed"
    assert state(desk)["final_state"]["exchanges"] == [] and state(desk)["final_state"]["refunds"] == []


def test_a_tick_is_dropped_when_the_latest_proposal_changes(desk: Any) -> None:
    desk.open("refund_boundary")
    assert desk.fill_and_send(1) == "propose"
    box = desk.page.locator('[data-testid="confirm-attach"]')
    box.check()
    session_id = state(desk)["session_id"]
    with httpx.Client(base_url=desk.base_url, timeout=30) as api:  # another tab asks for an exchange instead
        api.post(f"/api/sessions/{session_id}/turns", json={"message": "Please exchange line L1 of order O100 for size L."})
    desk.page.click('[data-testid="arm-timeout"]')  # this page refreshes its state (the failure lab answers with it)
    desk.page.wait_for_function("() => window.ReturnsDesk.app.session.turns.length === 2")
    newest = state(desk)["turns"][-1]["output"]["proposal"]["proposal_id"]
    assert newest != "P-0001" and desk.page.locator("#confirm-attach-id").inner_text() == newest
    assert not box.is_checked()  # the tick was for P-0001; it never silently moves to the new proposal


# ================================================================================================
# C10: dead proposals and actionable alternatives


def test_no_confirmation_is_offered_for_an_executed_proposal_unless_the_script_asks(desk: Any) -> None:
    desk.open("timeout_replay")
    assert desk.fill_and_send(1) == "propose"
    assert desk.page.locator('[data-testid="confirm-attach-text"]').is_visible()
    assert desk.confirm(2) == "completed"
    assert desk.page.locator("#confirm-attach").is_hidden()
    assert desk.page.locator('[data-testid="confirm"]').count() == 0  # the executed card has no buttons
    desk.page.click('[data-testid="fill-3"]')  # the script repeats its approval of the executed proposal
    label = desk.page.locator('[data-testid="confirm-attach-text"]').inner_text()
    assert desk.page.locator('[data-testid="confirm-attach"]').is_checked() and "already executed" in label
    desk.page.click('[data-testid="send"]')
    assert desk.wait_turn(3) == "completed"
    assert len(state(desk)["final_state"]["refunds"]) == 1  # a replay: no second refund
    assert desk.page.locator('[data-testid="replay-notice"]').count() == 1


def test_after_a_revalidation_deny_the_alternative_chip_starts_a_new_proposal(desk: Any) -> None:
    desk.open("stale_stock")
    assert desk.run_script() == ["propose", "deny"]
    assert desk.page.locator("#confirm-attach").is_hidden()  # P-0001 is invalidated: nothing to confirm
    chip = desk.message(2).locator('[data-testid="alternative"]')
    assert chip.inner_text() == "Exchange for size S instead" and chip.get_attribute("data-sku") == "TEE-S"
    chip.click()
    assert desk.page.input_value('[data-testid="composer-input"]') == "Please exchange line L1 of order O100 for size S."
    assert not desk.page.locator('[data-testid="confirm-attach"]').is_checked()
    desk.page.click('[data-testid="send"]')
    assert desk.wait_turn(3) == "propose"
    assert desk.page.locator('[data-testid="proposal-replacement"]').last.inner_text() == "TEE-S"
    assert state(desk)["final_state"]["exchanges"] == []  # still nothing moved without a confirmation
    desk.shot("c10_alternative_chip.png")


def test_a_deny_for_a_size_out_of_stock_offers_the_other_sizes_as_chips(desk: Any) -> None:
    desk.open("refund_boundary")
    desk.page.select_option('[data-testid="stock-sku"]', "TEE-L")
    desk.page.fill('[data-testid="stock-value"]', "0")
    desk.page.click('[data-testid="stock-apply"]')
    desk.page.wait_for_function("() => window.ReturnsDesk.app.session.external_events.length === 1")
    desk.page.fill('[data-testid="composer-input"]', "Please exchange line L1 of order O100 for size L.")
    desk.page.click('[data-testid="send"]')
    assert desk.wait_turn(1) == "deny"
    chip = desk.message(1).locator('[data-testid="alternative"]')
    assert chip.count() >= 1 and "TEE-L" not in [chip.nth(i).get_attribute("data-sku") for i in range(chip.count())]
    chip.first.click()  # no proposal exists: the line comes from the target grounded this turn
    assert desk.page.input_value('[data-testid="composer-input"]').startswith("Please exchange line L1 of order O100 for size ")


# ================================================================================================
# C6: readability of the Team view


TEAM_METRICS = """() => {
  const panel = document.getElementById('panel-team');
  const box = (e) => e.getBoundingClientRect();
  const visible = (e) => e.getClientRects().length > 0 && getComputedStyle(e).visibility !== 'hidden';
  const texts = [...panel.querySelectorAll('*')].filter(e => visible(e) &&
    [...e.childNodes].some(n => n.nodeType === Node.TEXT_NODE && n.textContent.trim()));
  const small = texts.map(e => [parseFloat(getComputedStyle(e).fontSize), e.className || e.tagName, e.textContent.trim().slice(0, 30)]);
  const goals = [...panel.querySelectorAll('.node-goal')].filter(visible);
  const graph = document.getElementById('team-graph');
  return {
    layout: graph.dataset.layout, minFont: Math.min(...small.map(s => s[0])), smallest: small.sort((a, b) => a[0] - b[0]).slice(0, 3),
    goalsVisible: goals.length, goalsCut: goals.filter(g => g.scrollHeight > g.clientHeight + 1).map(g => g.textContent),
    goalLines: Math.max(0, ...goals.map(g => Math.round(g.clientHeight / parseFloat(getComputedStyle(g).lineHeight)))),
    namesBrokenMidWord: [...panel.querySelectorAll('.node-name')].filter(n => n.getClientRects().length > 1 &&
      !n.textContent.includes(' ')).map(n => n.textContent),
    cardsClipped: [...panel.querySelectorAll('.node')].filter(n => n.scrollHeight > n.clientHeight + 1).length,
    panelOverflow: panel.scrollHeight - panel.clientHeight, gridBottom: box(graph.querySelector('.team-grid')).bottom,
    graphBottom: box(graph).bottom, panelBottom: box(panel).bottom, viewport: window.innerHeight,
    legendBottom: box(document.getElementById('legend')).bottom, pageOverflowX: document.documentElement.scrollWidth - window.innerWidth,
  };
}"""


def run_timeout_scenario(desk: Any) -> None:
    desk.open("timeout_replay")
    desk.run_script()
    desk.wait_team_caught_up()


@pytest.mark.parametrize("viewport", [(1280, 720), (1440, 900)])
def test_the_team_view_is_readable_at_laptop_sizes(make_desk: Any, viewport: tuple[int, int]) -> None:
    width, height = viewport
    desk = make_desk(viewport={"width": width, "height": height})
    run_timeout_scenario(desk)
    normal = desk.page.evaluate(TEAM_METRICS)
    assert normal["minFont"] >= 12, normal["smallest"]
    assert normal["goalsVisible"] == 12 and normal["goalsCut"] == [] and normal["goalLines"] <= 2
    assert normal["namesBrokenMidWord"] == [] and normal["cardsClipped"] == 0 and normal["pageOverflowX"] <= 0
    if viewport == (1280, 720):
        desk.shot("team_1280x720_normal.png")
    desk.page.click('[data-testid="theater"]')
    desk.page.wait_for_selector('[data-testid="team-graph"][data-layout="wide"]')
    desk.page.wait_for_timeout(300)  # the edge layer follows the new card positions
    theater = desk.page.evaluate(TEAM_METRICS)
    assert theater["minFont"] >= 14, theater["smallest"]
    assert theater["panelOverflow"] <= 0, theater  # theater fits the viewport: nothing scrolls away
    assert theater["gridBottom"] <= theater["graphBottom"] + 1 <= theater["panelBottom"] + 1 <= theater["viewport"] + 1
    assert theater["legendBottom"] <= theater["panelBottom"] + 1 and theater["cardsClipped"] == 0
    assert theater["goalsCut"] == [] and theater["namesBrokenMidWord"] == []
    if viewport == (1440, 900):
        assert theater["goalsVisible"] == 12  # tall enough for the goal lines too
    if viewport == (1280, 720):
        desk.shot("team_1280x720_theater.png")


def test_a_phone_gets_a_card_list_not_a_scaled_graph(make_desk: Any) -> None:
    desk = make_desk(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    run_timeout_scenario(desk)
    metrics = desk.page.evaluate(TEAM_METRICS)
    assert metrics["layout"] == "list" and metrics["minFont"] >= 12 and metrics["pageOverflowX"] <= 0
    assert desk.page.locator(".team-edges").is_hidden()
    tracks = desk.page.evaluate("() => getComputedStyle(document.querySelector('.team-grid')).gridTemplateColumns.split(' ')")
    widths = desk.page.evaluate("() => [...document.querySelectorAll('[data-node]')].map(n => n.offsetWidth)")
    assert len(tracks) == 1 and min(widths) >= 300  # one column of full-width cards
    two_digit = desk.page.evaluate("""() => { const li = [...document.querySelectorAll('#mission-log li')][9];
      return li ? parseFloat(getComputedStyle(li.parentElement).paddingLeft) : null; }""")
    assert two_digit is None or two_digit >= 28  # room for "10." in the mission log numbering


# ================================================================================================
# Audit entries from a newer desk


def test_unknown_entry_kinds_and_agent_grounding_still_render(desk: Any) -> None:
    desk.open("refund_boundary")
    shown = desk.page.evaluate("""() => {
      const R = window.ReturnsDesk;
      const unknown = {seq: 1, turn: 1, kind: 'verdict', agent: 'orchestrator', tool: 'decide', arguments: {}, result: {},
                       error: {code: 'LINE_NOT_FOUND'}};
      const agent = {seq: 2, turn: 1, kind: 'target', agent: 'orchestrator', tool: 'accept_target', grounded_by: 'agent',
                     arguments: {order_id: 'O100', line_id: 'L1', intent: 'propose_refund'},
                     result: {evidence: ['the size M tee', 'from O100']}};
      const typed = {...agent, seq: 3, grounded_by: undefined, arguments: {...agent.arguments, intent: 'refund', grounded_by: 'customer_ids'}};
      const a = R.describeEntry(unknown, {}); const b = R.describeEntry(agent, {}); const c = R.describeEntry(typed, {});
      return {unknown: a.fx.map(f => [f.node, f.status, f.text]), agent: b.fx.map(f => [f.node, f.status, f.text]),
              typed: c.fx.map(f => f.text),
              title: R.entryTitle(unknown), badges: R.entryBadges(agent), shortened: R.short('one two three four', 12)};
    }""")
    meta = httpx.get(f"{desk.base_url}/api/meta").json()["reason_text"]
    assert shown["unknown"] == [["orchestrator", "blocked", f"verdict · decide: {meta['LINE_NOT_FOUND']}"]]
    assert shown["agent"][-1] == ["grounding", "ok", "agent grounded O100/L1 · cites “the size M tee; from O100”"]
    assert shown["typed"] == ["grounded O100/L1 · refund · the IDs the customer typed"]
    assert shown["title"] == f"decide: {meta['LINE_NOT_FOUND']}" and "grounded by agent" in shown["badges"]
    assert shown["shortened"] == "one two…"  # cut at a word, never mid-word


def test_a_line_picked_by_citing_the_customer_shows_the_evidence(desk: Any) -> None:
    """A real turn, no crafted entries: two lines on O100 (tee, jacket) and no ids typed. The fake model
    cites the customer's word and code accepts the citation (grounded_by "agent", SPEC §6.0)."""
    desk.open("clarify")
    desk.page.fill('[data-testid="composer-input"]', "Please refund the jacket.")
    desk.page.click('[data-testid="send"]')
    assert desk.wait_turn(1) == "propose"
    target = next(e for e in state(desk)["action_audit"] if e["kind"] == "target")
    assert (target["arguments"]["grounded_by"], target["arguments"]["evidence"]) == ("agent", "jacket")
    assert desk.page.locator('[data-testid="proposal-card"]').get_attribute("data-proposal") == "P-0001"
    assert state(desk)["cards"]["P-0001"]["line_id"] == "L2"
    desk.wait_team_caught_up()
    assert desk.node_status("grounding").endswith("agent grounded O100/L2 · cites “jacket”")
