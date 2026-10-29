"""Live smoke test: the first public scenario through the browser with the REAL local model.

Marked `llm` (deselected by default) and `e2e`. Run:
    .venv/bin/pytest -m llm tests/e2e/test_live_smoke.py -p no:cacheprovider -s
Uses llama-server on port 8404, or $E2E_LLAMA_PORT (profile F): attaches if one is already healthy there,
otherwise starts one (sha256 and engine version checked) and stops it afterwards. The UI server runs in
real-model mode.
Turn latencies and the model's step telemetry are written to var/e2e/live_smoke.json.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from desk.llm_server import LlamaServer, server_healthy
from desk.profiles import get_profile

pytestmark = [pytest.mark.llm, pytest.mark.e2e, pytest.mark.timeout(600)]

ROOT = Path(__file__).resolve().parents[2]
LLAMA_PORT = int(os.environ.get("E2E_LLAMA_PORT", "8404"))
LIVE_TURN_TIMEOUT_MS = 90_000


@pytest.fixture(scope="module")
def llama_url() -> Iterator[str]:
    url = f"http://127.0.0.1:{LLAMA_PORT}"
    if server_healthy(url):
        yield url
        return
    server = LlamaServer(get_profile("F"), LLAMA_PORT, ROOT / "var" / "e2e" / f"llama-server-{LLAMA_PORT}.log")
    server.start()
    try:
        yield url
    finally:
        server.stop()


def test_public_refund_with_the_real_model(make_desk: Any, ui_server_factory: Any, llama_url: str) -> None:
    server = ui_server_factory(DESK_LLM="llama", DESK_LLAMA_URL=llama_url)
    desk = make_desk(server.url)
    desk.page.wait_for_selector('#llm-dot[data-state="ok"]')
    assert "Qwen3.5-4B" in desk.page.locator('[data-testid="model-chip"]').inner_text()
    desk.open("refund_boundary")
    desk.page.click('[data-testid="fill-1"]')
    desk.page.click('[data-testid="send"]')
    desk.page.wait_for_selector('[data-testid="desk-message"][data-turn="1"]:not([data-decision="working"])', timeout=LIVE_TURN_TIMEOUT_MS)
    assert desk.decision(1) == "propose"
    line = desk.case["initial_state"]["orders"][0]["lines"][0]
    paid = f"${line['paid_cents'] // 100}.{line['paid_cents'] % 100:02d}"
    assert desk.page.locator('[data-testid="proposal-amount"]').inner_text() == paid
    desk.page.click('[data-testid="confirm"]')
    desk.page.wait_for_selector('[data-testid="desk-message"][data-turn="2"]:not([data-decision="working"])', timeout=LIVE_TURN_TIMEOUT_MS)
    assert desk.decision(2) == "completed"
    desk.open_inspector()
    assert desk.rows("records-refunds") == 1 and paid in desk.page.locator('[data-testid="records-refunds"]').inner_text()
    assert desk.rows("records-quarantine") == 1 and desk.returned(desk.case["initial_state"]["orders"][0]["order_id"], line["line_id"])
    desk.wait_team_caught_up()
    desk.shot("live_public_01.png")
    session = desk.page.evaluate("() => window.ReturnsDesk.app.session")
    steps = [e for e in session["action_audit"] if e["kind"] == "llm_step"]
    assert steps and all(e["error"] is None for e in steps)
    report = {
        "case_id": desk.case["case_id"],
        "decisions": [t["output"]["decision"] for t in session["turns"]],
        "turn_latency_ms": [t["latency_ms"] for t in session["turns"]],
        "llm_steps": len(steps),
        "guards": sum(1 for e in session["action_audit"] if e["kind"] == "guard"),
        "fallbacks": sum(1 for e in session["action_audit"] if e["kind"] == "fallback"),
        "steps": [{k: e["result"].get(k) for k in ("role", "tool", "latency_ms", "prompt_tokens", "cached_tokens", "completion_tokens", "predicted_ms")}
                  for e in steps],
    }
    out = ROOT / "var" / "e2e" / "live_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "steps"}))
