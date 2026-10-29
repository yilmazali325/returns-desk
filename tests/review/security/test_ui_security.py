"""Security review, browser + real server (docs/review/security.md). Playwright chromium against
`python -m server.app --port 8430 --fake` (fake-policy LLM: the real orchestrator, gateway, store and guards).

- HTML/script payloads in the customer message, SKU, product id, size, order/line ids and customer id
  render as text everywhere (chat, proposal card, team view, audit trail, state inspector): holds.
- SSE disconnect mid-turn: the turn still finishes, the lock is released, state is consistent: holds.
- SEC-1 in the browser: two tabs on one session; a stale Confirm used to execute a proposal the tab never
  showed. Fixed: the server refuses it (409) and the tab says "Not sent" (the test guards the fix).
"""

from __future__ import annotations

import copy
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from playwright.sync_api import Page

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[3]
PORT = 8430  # the port this reviewer was assigned
BASE = f"http://127.0.0.1:{PORT}"
CASES = {c["case_id"]: c for c in json.loads((ROOT / "candidate_assets" / "public_cases.json").read_text())["cases"]}
SHOTS = ROOT / "tests" / "review" / "security" / "shots"
XSS = ("<img src=x onerror=\"window.__xss=(window.__xss||[]).concat('{tag}')\">"
       "<script>window.__xss=(window.__xss||[]).concat('{tag}-script')</script>"
       "\"><svg onload=\"window.__xss=(window.__xss||[]).concat('{tag}-svg')\">")


def payload(tag: str) -> str:
    return XSS.format(tag=tag)


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", PORT)) == 0:
            pytest.skip(f"port {PORT} is busy")
    var_dir = tmp_path_factory.mktemp("ui-security")
    env = {**os.environ, "DESK_VAR_DIR": str(var_dir), "DESK_FAKE_STEP_MS": "150"}
    log = (var_dir / "server.log").open("wb")
    process = subprocess.Popen([sys.executable, "-m", "server.app", "--port", str(PORT), "--fake"], cwd=ROOT, env=env,
                               stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{BASE}/api/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.1)
        else:
            raise RuntimeError("UI server did not start")
        yield BASE
    finally:
        process.terminate()
        process.wait(timeout=10)


def wait_idle(page: Page, turns: int) -> None:
    page.wait_for_function("n => { const a = window.ReturnsDesk.app; return a.session && !a.busy && a.session.turns.length === n; }",
                           arg=turns, timeout=30_000)


def send(page: Page, message: str, turns_after: int) -> None:
    page.fill('[data-testid="composer-input"]', message)
    page.click('[data-testid="send"]')
    wait_idle(page, turns_after)


def open_everything(page: Page) -> None:
    """Show every view that renders data: audit tab with all entries expanded, the state drawer."""
    page.click('[data-testid="tab-audit"]')
    page.evaluate("document.querySelectorAll('details').forEach(d => d.open = true)")
    if page.get_attribute('[data-testid="drawer-toggle"]', "aria-expanded") != "true":
        page.click('[data-testid="drawer-toggle"]')
    page.click('[data-testid="tab-team"]')
    page.wait_for_timeout(300)


def test_html_payloads_from_every_data_source_render_as_text(server: str, browser: Any) -> None:
    state = copy.deepcopy(CASES["public_01_refund_boundary"]["initial_state"])
    product = next(p for p in state["products"] if p["sku"] == "TEE-M")
    hostile_sku = "TEE-M" + payload("sku")
    product.update(sku=hostile_sku, product_id="TEE" + payload("pid"), size="M" + payload("size"))
    order = state["orders"][0]
    order["order_id"] = "O100" + payload("order")
    order["lines"][0].update(sku=hostile_sku, line_id="L1" + payload("line"))
    customer = "C100" + payload("cust")
    order["customer_id"] = customer
    created = httpx.post(f"{server}/api/sessions", json={"initial_state": state, "customer_id": customer}, timeout=10)
    assert created.status_code == 200, created.text
    sid = created.json()["session_id"]

    context = browser.new_context()
    context.add_init_script(f"try {{ localStorage.setItem('returns-desk.session', {json.dumps(json.dumps(sid))}); }} catch (e) {{}}")
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.goto(server)
    page.wait_for_function(f"window.ReturnsDesk.app.session && window.ReturnsDesk.app.session.session_id === {json.dumps(sid)}")
    send(page, "Please refund my item. " + payload("msg"), 1)
    open_everything(page)
    assert page.locator('[data-testid="proposal-card"], [data-testid="desk-message"]').count() >= 1
    SHOTS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SHOTS / "xss_payloads_as_text.png"), full_page=True)

    assert page.evaluate("window.__xss === undefined"), page.evaluate("window.__xss")
    injected = page.evaluate("""() => [...document.querySelectorAll('img[src="x"], body script:not([src="/static/app.js"]), svg[onload], [onerror]')]
                                       .map(n => n.outerHTML.slice(0, 80))""")
    assert injected == []
    text = page.inner_text("body")
    assert "onerror=" in text and "<script>" in text  # the payloads are visible, literally
    assert errors == []
    context.close()


def test_a_disconnected_stream_still_finishes_the_turn_and_frees_the_session(server: str) -> None:
    case = CASES["public_01_refund_boundary"]
    sid = httpx.post(f"{server}/api/sessions", json={"case_id": case["case_id"]}).json()["session_id"]
    with httpx.Client(base_url=server, timeout=30) as client:
        with client.stream("POST", f"/api/sessions/{sid}/turns", json={"message": case["turns"][0]["message"]}) as stream:
            for line in stream.iter_lines():
                if line.startswith("event: audit"):
                    break  # the browser tab closes mid-turn
        assert client.post(f"/api/sessions/{sid}/turns", json={"message": "hello"}).status_code == 409
        assert client.post(f"/api/sessions/{sid}/reset", json={}).status_code == 409
        deadline = time.monotonic() + 30
        while (snap := client.get(f"/api/sessions/{sid}/state").json())["busy"] and time.monotonic() < deadline:
            time.sleep(0.1)
    assert not snap["busy"]
    assert [t["output"]["decision"] for t in snap["turns"]] == ["propose"]
    assert [p["status"] for p in snap["proposals"]] == ["open"]


def test_two_tabs_a_stale_confirm_button_cannot_approve_a_different_proposal(server: str, browser: Any) -> None:
    context = browser.new_context()  # one browser profile: both tabs share localStorage, so one session
    tab_b = context.new_page()
    tab_b.goto(server)
    tab_b.wait_for_function("window.ReturnsDesk.app.session !== null")
    tab_b.select_option('[data-testid="scenario-select"]', "public_04_clarify")
    tab_b.wait_for_function("window.ReturnsDesk.app.session && window.ReturnsDesk.app.session.case_id === 'public_04_clarify'")
    send(tab_b, "Refund line L1 of order O100 please.", 1)
    confirm_b = tab_b.locator('[data-testid="confirm"]').last
    assert confirm_b.inner_text() == "Confirm refund $30.00"
    sid = tab_b.evaluate("window.ReturnsDesk.app.session.session_id")

    tab_a = context.new_page()
    tab_a.goto(server)
    tab_a.wait_for_function(f"window.ReturnsDesk.app.session && window.ReturnsDesk.app.session.session_id === {json.dumps(sid)}")
    tab_a.click('[data-testid="reset"]')
    wait_idle(tab_a, 0)
    send(tab_a, "Refund line L2 of order O100 please.", 1)

    tab_b.bring_to_front()
    confirm_b.click()  # tab B still shows "Confirm refund $30.00" for L1
    # The server refuses the stale binding (409) before anything runs; the tab says so and reloads.
    tab_b.wait_for_selector('[data-testid="notice"] h2:text-is("Not sent")', timeout=30_000)
    wait_idle(tab_b, 1)
    SHOTS.mkdir(parents=True, exist_ok=True)
    tab_b.screenshot(path=str(SHOTS / "stale_tab_confirm.png"), full_page=True)
    refunds = httpx.get(f"{server}/api/sessions/{sid}/state").json()["final_state"]["refunds"]
    context.close()
    # Before the fix: refunds == [{'line_id': 'L2', 'amount_cents': 9000, ...}] after "Confirm refund $30.00".
    assert [(r["line_id"], r["amount_cents"]) for r in refunds] in ([], [("L1", 3000)])
