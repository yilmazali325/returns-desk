"""Browser half of the verification (NEW-1, NEW-2). Playwright chromium against
`python -m server.app --port 8437 --fake` (fake-policy LLM; real orchestrator, gateway, store, guards).
No llama-server. The tests assert the SAFE behaviour. They were xfail(strict=True) while NEW-1 and NEW-2
were open; both are fixed in app.js, so the markers are gone and the tests guard the fixes.

    .venv/bin/pytest tests/review/verify_security/test_verify_ui.py -p no:cacheprovider -o addopts="" -m ""
"""

from __future__ import annotations

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
PORT = 8437
BASE = f"http://127.0.0.1:{PORT}"
SHOTS = ROOT / "tests" / "review" / "verify_security" / "shots"


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", PORT)) == 0:
            pytest.skip(f"port {PORT} is busy")
    var_dir = tmp_path_factory.mktemp("ui-verify-security")
    env = {**os.environ, "DESK_VAR_DIR": str(var_dir), "DESK_FAKE_STEP_MS": "50"}
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


@pytest.fixture
def page(server: str, browser: Any) -> Iterator[Page]:
    context = browser.new_context(viewport={"width": 1400, "height": 1000})  # the shared session browser
    page = context.new_page()
    page.goto(server)
    page.wait_for_function("window.ReturnsDesk.app.session !== null")
    yield page
    context.close()


def open_scenario(page: Page, case_id: str) -> None:
    page.select_option('[data-testid="scenario-select"]', case_id)
    page.wait_for_function(f"window.ReturnsDesk.app.session && window.ReturnsDesk.app.session.case_id === '{case_id}'"
                           " && window.ReturnsDesk.app.session.turns.length === 0")


def wait_turns(page: Page, n: int) -> None:
    page.wait_for_function("n => { const a = window.ReturnsDesk.app; return a.session && !a.busy && a.session.turns.length === n; }",
                           arg=n, timeout=30_000)


def send(page: Page, message: str, turns_after: int) -> None:
    page.fill('[data-testid="composer-input"]', message)
    page.click('[data-testid="send"]')
    wait_turns(page, turns_after)


def last_turn(page: Page) -> dict[str, Any]:
    return page.evaluate("(() => { const t = window.ReturnsDesk.app.session.turns; return t[t.length - 1]; })()")


def test_new1_confirm_button_on_an_off_script_exchange_card(page: Page) -> None:
    open_scenario(page, "public_01_refund_boundary")
    send(page, "Actually I'd like to exchange line L1 of order O100 for size L instead.", 1)
    button = page.locator('[data-testid="confirm"]').last
    label = button.inner_text()
    assert label == "Confirm exchange to TEE-L"
    button.click()
    wait_turns(page, 2)
    record = last_turn(page)
    SHOTS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SHOTS / "new1_confirm_exchange_sends_refund_text.png"), full_page=True)
    exchanges = page.evaluate("window.ReturnsDesk.app.session.final_state.exchanges.length")
    # Before the fix: message 'Yes, refund the proposed item and amount.', decision 'propose' (a refund), 0 exchanges.
    assert (record["message"], record["output"]["decision"], exchanges) == (record["message"], "completed", 1), \
        (label, record["message"], record["output"]["decision"])


def test_new2_confirmation_checkbox_does_not_carry_over_to_a_new_proposal(page: Page) -> None:
    open_scenario(page, "public_01_refund_boundary")
    send(page, "Refund line L1 of my order O100. It is unopened.", 1)
    page.check('[data-testid="confirm-attach"]')
    assert page.inner_text("#confirm-attach-id") == "P-0001"
    send(page, "Actually, exchange it for size L instead.", 2)  # sent WITH confirmation of P-0001: withheld
    assert last_turn(page)["output"]["proposal"]["proposal_id"] == "P-0002"
    still_ticked = page.is_checked('[data-testid="confirm-attach"]')
    now_names = page.inner_text("#confirm-attach-id")
    send(page, "When was my order delivered?", 3)  # a question; the customer never approved P-0002
    record = last_turn(page)
    SHOTS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SHOTS / "new2_sticky_confirmation_checkbox.png"), full_page=True)
    # Before the fix: still ticked, naming P-0002; the question carried confirm P-0002 and completed the exchange.
    assert record["confirm_proposal_id"] is None and record["output"]["decision"] != "completed", \
        (still_ticked, now_names, record["confirm_proposal_id"], record["output"]["decision"])
