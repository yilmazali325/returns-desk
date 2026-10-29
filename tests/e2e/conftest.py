"""Playwright fixtures for the UI end-to-end tests (marked e2e; plain pytest-playwright is not installed).

Each test talks to a real `python -m server.app` process on a free port, started by a fixture, with
DESK_LLM=fake-policy: the REAL orchestrator, gateway, store and guards run, only the model is the
deterministic double, so a scripted conversation always produces the same trace. The model-down
test starts its own server in real-model mode pointed at a port where nothing listens.

`DeskPage` wraps a Playwright page with the few actions every test needs (open a scenario, fill and
send a scripted turn, press Confirm, read the state inspector). Screenshots go to var/e2e/shots/, or
to $E2E_SHOTS_DIR (scripts/screenshots.sh points it at var/screenshots/, the presentation shots).

JS coverage: with E2E_JS_COVERAGE=1 every page records V8 block coverage over CDP; at the end of the
session the union over all pages is reported for server/static/app.js (var/e2e/js_coverage.json).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from playwright.sync_api import Browser, Page

ROOT = Path(__file__).resolve().parents[2]
SCREENSHOTS = Path(os.environ.get("E2E_SHOTS_DIR") or ROOT / "var" / "e2e" / "shots")
CASES = json.loads((ROOT / "candidate_assets" / "public_cases.json").read_text())["cases"]
LABELS = {c["case_id"]: c["expected"] for c in json.loads((ROOT / "evals" / "expected" / "public_expected.json").read_text())["cases"]}
FAKE_STEP_MS = "80"  # slow enough that a turn visibly streams (about 1 s), fast enough for a quick suite
TURN_TIMEOUT_MS = 30_000


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class UIServer:
    url: str
    process: subprocess.Popen[bytes]
    log: Path

    def stop(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()


def start_ui(var_dir: Path, **env: str) -> UIServer:
    """Start `python -m server.app` on a free port and wait until /api/health answers."""
    port = free_port()
    var_dir.mkdir(parents=True, exist_ok=True)
    log = var_dir / f"ui-{port}.log"
    environment = {**os.environ, "DESK_VAR_DIR": str(var_dir), **env}
    with log.open("wb") as out:
        process = subprocess.Popen([sys.executable, "-m", "server.app", "--port", str(port)], cwd=ROOT, env=environment,
                                   stdout=out, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 90  # generous: Python start-up takes seconds on a loaded machine
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"UI server exited early:\n{log.read_text()}")
        try:
            if httpx.get(f"{url}/api/health", timeout=1).status_code == 200:
                return UIServer(url, process, log)
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    process.kill()
    raise RuntimeError(f"UI server not ready:\n{log.read_text()}")


@pytest.fixture(scope="session")
def fake_ui(tmp_path_factory: pytest.TempPathFactory) -> Iterator[UIServer]:
    server = start_ui(tmp_path_factory.mktemp("ui-fake"), DESK_LLM="fake-policy", DESK_FAKE_STEP_MS=FAKE_STEP_MS)
    yield server
    server.stop()


@pytest.fixture
def ui_server_factory(tmp_path: Path) -> Iterator[Any]:
    """Start extra servers with other settings (e.g. a dead model URL); all are stopped afterwards."""
    servers: list[UIServer] = []

    def factory(**env: str) -> UIServer:
        server = start_ui(tmp_path / f"ui-{len(servers)}", **env)
        servers.append(server)
        return server

    factory.free_port = free_port  # type: ignore[attr-defined]
    yield factory
    for server in servers:
        server.stop()


class DeskPage:
    """The UI as a test sees it."""

    def __init__(self, page: Page, base_url: str) -> None:
        self.page = page
        self.base_url = base_url
        self.case: dict[str, Any] | None = None

    # --- navigation ---------------------------------------------------------------------------

    def goto(self) -> DeskPage:
        self.page.goto(self.base_url)
        self.page.wait_for_selector('[data-testid="script-list"] li, [data-testid="notice"]')
        return self

    def open(self, suffix: str) -> DeskPage:
        self.case = next(c for c in CASES if c["case_id"].endswith(suffix))
        select = self.page.locator('[data-testid="scenario-select"]')
        if select.input_value() != self.case["case_id"]:
            select.select_option(self.case["case_id"])
        self.page.wait_for_function(
            "id => window.ReturnsDesk.app.session && window.ReturnsDesk.app.session.case_id === id "
            "&& window.ReturnsDesk.app.session.turns.length === 0", arg=self.case["case_id"])
        return self

    @property
    def label(self) -> dict[str, Any]:
        assert self.case is not None
        return LABELS[self.case["case_id"]]

    # --- turns ----------------------------------------------------------------------------------

    def fill_and_send(self, number: int) -> str:
        self.page.click(f'[data-testid="fill-{number}"]')
        self.page.click('[data-testid="send"]')
        return self.wait_turn(number)

    def wait_turn(self, number: int) -> str:
        selector = f'[data-testid="desk-message"][data-turn="{number}"]:not([data-decision="working"])'
        self.page.wait_for_selector(selector, timeout=TURN_TIMEOUT_MS)
        self.page.wait_for_function("() => !window.ReturnsDesk.app.busy")
        return self.page.get_attribute(selector, "data-decision") or ""

    def confirm(self, number: int) -> str:
        self.page.click('[data-testid="confirm"]')
        return self.wait_turn(number)

    def run_script(self) -> list[str]:
        """Every scripted turn; a confirm turn uses the card's Confirm button when one is offered."""
        assert self.case is not None
        decisions = []
        for number, turn in enumerate(self.case["turns"], start=1):
            confirm_button = self.page.locator('[data-testid="confirm"]')
            if turn.get("confirm_proposal") == "last" and confirm_button.count():
                decisions.append(self.confirm(number))
            else:
                decisions.append(self.fill_and_send(number))
        return decisions

    def decision(self, number: int) -> str:
        return self.page.get_attribute(f'[data-testid="desk-message"][data-turn="{number}"]', "data-decision") or ""

    def message(self, number: int):
        return self.page.locator(f'[data-testid="desk-message"][data-turn="{number}"]')

    # --- state inspector and audit --------------------------------------------------------------

    def open_inspector(self) -> None:
        toggle = self.page.locator('[data-testid="drawer-toggle"]')
        if toggle.get_attribute("aria-expanded") != "true":
            toggle.click()

    def rows(self, testid: str) -> int:
        return self.page.locator(f'[data-testid="{testid}"] tbody tr').count()

    def stock(self, sku: str) -> int:
        text = self.page.locator(f'[data-testid="products"] tr[data-sku="{sku}"] [data-testid="stock"]').inner_text()
        return int(text.split("→")[-1].strip())

    def returned(self, order_id: str, line_id: str) -> bool:
        text = self.page.locator(f'[data-testid="lines"] tr[data-line="{order_id}/{line_id}"] [data-testid="returned"]').inner_text()
        return text.split("→")[-1].strip() in ("true", "yes")

    def node_status(self, node_id: str) -> str:
        """A team card's status line, as text."""
        return self.page.locator(f"[data-node='{node_id}'] [data-role='status']").inner_text()

    def committed_entries(self) -> int:
        return self.page.locator('[data-testid="audit-entry"][data-effects="true"]').count()

    def wait_team_caught_up(self) -> None:
        self.page.wait_for_function(
            "() => { const p = window.ReturnsDesk.player; const pos = document.getElementById('replay-pos');"
            " return !p.live && !p.timer && pos.dataset.step === pos.dataset.steps; }", timeout=TURN_TIMEOUT_MS)

    def shot(self, name: str, full_page: bool = False) -> Path:
        SCREENSHOTS.mkdir(parents=True, exist_ok=True)
        path = SCREENSHOTS / name
        self.page.screenshot(path=str(path), full_page=full_page)
        return path


# --- V8 block coverage of app.js (opt-in) ---------------------------------------------------------

APP_JS = ROOT / "server" / "static" / "app.js"
JS_COVERAGE: list[list[dict[str, Any]]] = []  # one list of function coverages per page


class _CoveredContext:
    """A browser context whose close() first collects the page's precise coverage of app.js."""

    def __init__(self, context: Any, cdp: Any) -> None:
        self.context, self.cdp = context, cdp

    def close(self) -> None:
        if self.cdp is not None:
            for script in self.cdp.send("Profiler.takePreciseCoverage")["result"]:
                if script["url"].endswith("/static/app.js"):
                    JS_COVERAGE.append(script["functions"])
        self.context.close()


def js_coverage_report(source: str, runs: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Union over pages of V8 block coverage. Ranges are painted outer-first so the innermost count
    wins; a line counts as covered when any of its code characters ran."""
    covered = bytearray(len(source))
    for functions in runs:
        painted = bytearray(len(source))
        ranges = sorted((r for f in functions for r in f["ranges"]), key=lambda r: (r["startOffset"], -r["endOffset"]))
        for r in ranges:
            painted[r["startOffset"]:r["endOffset"]] = (b"\x01" if r["count"] > 0 else b"\x00") * (r["endOffset"] - r["startOffset"])
        covered = bytearray(a | b for a, b in zip(covered, painted))
    lines, missed, offset = 0, [], 0
    for number, text in enumerate(source.split("\n"), start=1):
        code = [i for i, ch in enumerate(text) if not ch.isspace()]
        if code and not text.strip().startswith(("//", "/*", "*")):
            lines += 1
            if not any(covered[offset + i] for i in code):
                missed.append(number)
        offset += len(text) + 1
    return {"file": "server/static/app.js", "pages": len(runs), "code_lines": lines, "covered_lines": lines - len(missed),
            "line_coverage_pct": round(100 * (lines - len(missed)) / lines, 1) if lines else None, "missed_lines": missed}


def pytest_sessionfinish(session: pytest.Session) -> None:
    if not JS_COVERAGE:
        return
    report = js_coverage_report(APP_JS.read_text(), JS_COVERAGE)
    out = ROOT / "var" / "e2e" / "js_coverage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(f"app.js line coverage over {report['pages']} pages: {report['covered_lines']}/{report['code_lines']} "
                            f"= {report['line_coverage_pct']}% (missed lines: {report['missed_lines'][:40]}) -> {out}")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "http_errors: the test provokes failed HTTP responses on purpose")


def console_collector(errors: list[str], allow_http_errors: bool) -> Any:
    def collect(msg: Any) -> None:
        if msg.type != "error":
            return
        if allow_http_errors and msg.text.startswith("Failed to load resource"):
            return  # the browser logs every 4xx/5xx; these tests provoke them on purpose
        errors.append(f"console.{msg.type}: {msg.text}")

    return collect


def new_desk_page(browser: Browser, url: str, errors: list[str], *, allow_http_errors: bool = False,
                  before_goto: Any = None, **context_options: Any) -> tuple[DeskPage, Any]:
    context = browser.new_context(**{"viewport": {"width": 1440, "height": 900}, **context_options})
    page = context.new_page()
    if before_goto is not None:
        before_goto(page)
    cdp = None
    if os.environ.get("E2E_JS_COVERAGE"):
        cdp = context.new_cdp_session(page)
        cdp.send("Profiler.enable")
        cdp.send("Profiler.startPreciseCoverage", {"callCount": True, "detailed": True})
    page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
    page.on("console", console_collector(errors, allow_http_errors))
    return DeskPage(page, url).goto(), _CoveredContext(context, cdp)


@pytest.fixture
def desk(request: pytest.FixtureRequest, browser: Browser, fake_ui: UIServer) -> Iterator[DeskPage]:
    errors: list[str] = []
    allow = request.node.get_closest_marker("http_errors") is not None
    desk_page, context = new_desk_page(browser, fake_ui.url, errors, allow_http_errors=allow)
    yield desk_page
    context.close()
    assert not errors, f"the page logged errors: {errors}"


@pytest.fixture
def make_desk(request: pytest.FixtureRequest, browser: Browser, fake_ui: UIServer) -> Iterator[Any]:
    """Factory for pages with other browser options (viewport, colour scheme, reduced motion, routes)."""
    contexts: list[Any] = []
    errors: list[str] = []
    allow = request.node.get_closest_marker("http_errors") is not None

    def factory(url: str | None = None, **options: Any) -> DeskPage:
        desk_page, context = new_desk_page(browser, url or fake_ui.url, errors, allow_http_errors=allow, **options)
        contexts.append(context)
        return desk_page

    yield factory
    for context in contexts:
        context.close()
    assert not errors, f"the page logged errors: {errors}"
