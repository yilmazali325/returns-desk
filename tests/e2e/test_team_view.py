"""E2E tests for the Team (live) view, SPEC §10.1: every node, live handoff edges, specialists' verdicts,
replay, theater mode and reduced motion. Screenshots: team_midturn.png, team_theater.png."""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(180)]

BADGES = {
    "customer": "PERSON", "resolution": "LLM", "operations": "LLM", "orchestrator": "CODE", "grounding": "CODE",
    "policy": "CODE", "approval": "CODE", "gateway": "CODE", "recovery": "CODE", "replyguard": "CODE",
    "store": "DATA", "warehouse": "EXTERNAL",
}


def test_every_node_renders_with_its_badge(desk: Any) -> None:
    nodes = desk.page.locator('[data-testid="team-graph"] [data-node]')
    assert nodes.count() == len(BADGES)
    for node_id, badge in BADGES.items():
        node = desk.page.locator(f'[data-node="{node_id}"]')
        assert node.locator('[data-testid="node-badge"]').text_content() == badge
    legend = desk.page.locator("#legend").inner_text()
    assert "Agents decide; code can say no." in legend and "Deterministic specialist" in legend


JOBS = """() => [...document.querySelectorAll('[data-testid="team-graph"] [data-node]')].map(n => {
  const job = n.querySelector('[data-testid="node-job"]');
  return job && { node: n.dataset.node, text: job.textContent, shown: job.getClientRects().length > 0,
    cut: job.scrollHeight > job.clientHeight + 1, font: parseFloat(getComputedStyle(job).fontSize) };
})"""


@pytest.mark.parametrize("viewport", [(1280, 720), (1440, 900)])
def test_every_card_says_its_job_in_full(make_desk: Any, viewport: tuple[int, int]) -> None:
    desk = make_desk(viewport={"width": viewport[0], "height": viewport[1]})
    jobs = {j["node"]: j for j in desk.page.evaluate(JOBS) if j}
    assert set(jobs) == set(BADGES)
    for job in jobs.values():
        assert job["shown"] and not job["cut"] and job["font"] >= 12, job
        assert job["text"].startswith("Job ") and len(job["text"]) > len("Job ") + 20, job
    assert "Front desk" in jobs["resolution"]["text"]
    assert "Never sees the customer's words" in jobs["operations"]["text"]


def test_a_confirm_turn_animates_the_handoff_approves_and_shows_the_refund_chip(desk: Any) -> None:
    desk.open("refund_boundary")
    assert desk.fill_and_send(1) == "propose"
    desk.wait_team_caught_up()
    desk.page.click('[data-testid="confirm"]')
    desk.page.wait_for_selector('[data-edge="operations--resolution"][data-active="true"]', state="attached", timeout=20_000)
    desk.shot("team_midturn.png")
    assert desk.wait_turn(2) == "completed"
    desk.wait_team_caught_up()
    approval = desk.page.locator('[data-node="approval"]')
    assert approval.get_attribute("data-status") == "ok" and "customer confirmed P-0001" in desk.node_status("approval")
    chip = desk.page.locator('[data-testid="store-chips"] [data-chip="money"]')
    assert chip.inner_text() == "refund RF-0001 $30.00"
    assert desk.page.get_attribute('[data-node="customer"]', "data-status") == "ok"
    strip = desk.page.locator('[data-testid="turn-strip"]').inner_text()
    assert strip.startswith("Turn 2") and "2 LLM agents" in strip and strip.endswith("completed")


def test_replay_steps_through_exactly_the_audit_entries_of_the_turn(desk: Any) -> None:
    desk.open("refund_boundary")
    desk.run_script()
    desk.wait_team_caught_up()
    desk.page.select_option('[data-testid="replay-turn"]', "1")
    desk.page.click('[data-testid="tab-audit"]')
    audit_count = desk.page.locator('[data-turn-group="1"] [data-testid="audit-entry"]').count()
    desk.page.click('[data-testid="tab-team"]')
    desk.page.select_option('[data-testid="replay-speed"]', "2")
    desk.page.click('[data-testid="replay-play"]')
    desk.page.wait_for_function("() => document.getElementById('replay').dataset.playing === 'true'")
    desk.wait_team_caught_up()
    position = desk.page.locator('[data-testid="replay-pos"]')
    assert int(position.get_attribute("data-steps")) == audit_count == int(position.get_attribute("data-step"))
    assert desk.page.locator('[data-testid="mission-log"] li').count() == audit_count
    desk.page.focus('[data-testid="panel-team"]')
    desk.page.keyboard.press("ArrowLeft")
    desk.page.keyboard.press("ArrowLeft")
    assert position.get_attribute("data-step") == str(audit_count - 2)
    desk.page.keyboard.press("ArrowRight")
    assert position.get_attribute("data-step") == str(audit_count - 1)


def test_the_slowest_replay_speed_moves_each_handoff_ten_times_slower(desk: Any) -> None:
    desk.open("refund_boundary")
    desk.run_script()
    desk.wait_team_caught_up()
    desk.page.select_option('[data-testid="replay-speed"]', "0.1")
    desk.page.wait_for_function("() => !document.querySelector('.team-packets .packet')")  # the live run's last hop is gone
    desk.page.click('[data-testid="replay-play"]')
    motion = desk.page.wait_for_selector(".team-packets .packet animateMotion", state="attached", timeout=60_000)
    assert motion.get_attribute("dur") == "3500ms"  # DWELL_MS 350 at 0.1x


def test_theater_toggle_widens_the_team_view(desk: Any) -> None:
    desk.open("timeout_replay")
    desk.run_script()
    desk.page.locator("body").press("t")
    desk.page.wait_for_function("() => document.body.classList.contains('theater')")
    desk.page.wait_for_selector('[data-testid="team-graph"][data-layout="wide"]')
    width = desk.page.evaluate("() => document.querySelector('.side').getBoundingClientRect().width / window.innerWidth")
    assert width > 0.6
    assert desk.page.get_attribute('[data-testid="theater"]', "aria-pressed") == "true"
    desk.wait_team_caught_up()
    desk.shot("team_theater.png")
    desk.page.click('[data-testid="theater"]')
    desk.page.wait_for_function("() => !document.body.classList.contains('theater')")


def test_reduced_motion_keeps_the_highlights_without_moving_packets(make_desk: Any) -> None:
    desk = make_desk(reduced_motion="reduce")
    desk.page.evaluate("""() => { window.__packets = 0; new MutationObserver((records) => {
        for (const r of records) for (const n of r.addedNodes) if (n.classList && n.classList.contains('packet')) window.__packets += 1;
      }).observe(document.getElementById('team-graph'), {childList: true, subtree: true}); }""")
    desk.open("refund_boundary")
    assert desk.fill_and_send(1) == "propose"
    desk.page.click('[data-testid="confirm"]')
    desk.page.wait_for_selector('[data-edge="operations--resolution"][data-active="true"]', state="attached", timeout=20_000)
    assert desk.wait_turn(2) == "completed"
    desk.wait_team_caught_up()
    assert desk.page.get_attribute('[data-node="approval"]', "data-status") == "ok"
    assert desk.page.evaluate("() => window.__packets") == 0


LLM_ANIMATED = """() => [...document.querySelectorAll('[data-testid="team-graph"] [data-node]')]
    .filter(n => n.getAnimations().some(a => a.animationName.startsWith('llm-'))).map(n => n.dataset.node).sort()"""


def test_only_the_llm_agents_animate_and_reduced_motion_stops_them(make_desk: Any) -> None:
    assert make_desk().page.evaluate(LLM_ANIMATED) == ["operations", "resolution"]
    assert make_desk(reduced_motion="reduce").page.evaluate(LLM_ANIMATED) == []


TYPING = """() => [...document.querySelectorAll('[data-testid="team-graph"] [data-node]')]
    .filter(n => getComputedStyle(n, '::after').content !== 'none').map(n => n.dataset.node)"""


def test_no_agent_is_still_typing_once_the_turn_is_over(desk: Any) -> None:
    desk.open("refund_boundary")
    assert desk.fill_and_send(1) == "propose"
    desk.wait_team_caught_up()
    assert desk.page.get_attribute('[data-node="resolution"]', "data-status") == "working"  # last to act
    assert desk.page.evaluate(TYPING) == []
