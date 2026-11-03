"""Independent verification of the security review (docs/review/security.md, SEC-1..SEC-5) plus NEW-1/NEW-2.

Convention (same as tests/review/security): each finding test asserts the SAFE behaviour. They were
`xfail(strict=True)` while the finding was open; SEC-1, SEC-4, SEC-5 and NEW-1 are fixed, so the markers
are gone and the tests now guard the fixes.

    .venv/bin/pytest tests/review/verify_security -p no:cacheprovider -o addopts="" -m ""

SEC-4's Host check was removed on purpose (any Host is served, for SSH port forwards), so that one test
is xfail(strict=True) again; the Origin check and body cap still hold.

In-process TestClient with the fake-policy LLM (real orchestrator, gateway, store, guards). No network,
no llama-server.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from desk.render import facts_line, guard_reply
from desk.sanitize import sanitize
from server.app import FAKE_POLICY, ServerConfig, create_app

ROOT = Path(__file__).resolve().parents[3]
CASES = {c["case_id"]: c for c in json.loads((ROOT / "candidate_assets" / "public_cases.json").read_text())["cases"]}
MODEL = ROOT / "models" / "Qwen3.5-4B-Q4_K_M.gguf"


def parse_sse(body: str) -> list[tuple[str, Any]]:
    events = []
    for block in body.split("\n\n"):
        lines = [line for line in block.split("\n") if line and not line.startswith(":")]
        if not lines:
            continue
        kind = next(line[len("event: "):] for line in lines if line.startswith("event: "))
        events.append((kind, json.loads("".join(line[len("data: "):] for line in lines if line.startswith("data: ")))))
    return events


def make_client(var_dir: Path) -> TestClient:
    return TestClient(create_app(ServerConfig(llm_mode=FAKE_POLICY, var_dir=var_dir)), raise_server_exceptions=False,
                      base_url="http://127.0.0.1")


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return make_client(tmp_path / "ui")


def confirmation(state: dict[str, Any], proposal_id: str) -> dict[str, Any]:
    """What a page that displays `state` sends with a Confirm click: the card's hash and the generation (SEC-1)."""
    return {"confirm_proposal_id": proposal_id, "confirm_binding_hash": state["cards"][proposal_id]["binding_hash"],
            "generation": state["generation"]}


def post_turn(client: TestClient, sid: str, message: str, confirm: str | None = None,
              shown: dict[str, Any] | None = None) -> Any:
    """Send a turn; a confirmation binds to `shown` (a stale page's state) or else to the current state."""
    body: dict[str, Any] = {"message": message}
    if confirm:
        body.update(confirmation(shown or client.get(f"/api/sessions/{sid}/state").json(), confirm))
    return client.post(f"/api/sessions/{sid}/turns", json=body)


def turn(client: TestClient, sid: str, message: str, confirm: str | None = None) -> dict[str, Any]:
    response = post_turn(client, sid, message, confirm)
    assert response.status_code == 200, response.text
    kind, data = parse_sse(response.text)[-1]
    assert kind == "turn", (kind, data)
    return data


def ui_confirm_message(case_id: str | None, turns_so_far: int) -> str:
    """Mirror of server/static/app.js NEUTRAL_CONFIRMATION: since the NEW-1 fix the Confirm button always
    sends this sentence (it used to send the scenario's next scripted message, which can name another remedy)."""
    del case_id, turns_so_far  # kept so the call sites read as before the fix
    return "Yes, I confirm this proposal."


def refunds(state: dict[str, Any]) -> list[tuple[str, int]]:
    return [(r["line_id"], r["amount_cents"]) for r in state["final_state"]["refunds"]]


# ================================================================================================
# SEC-1: the UI's approval is bound to a counter id only; ids restart per store generation and per process


def test_sec1_reset_then_stale_confirm_executes_a_different_proposal(client: TestClient) -> None:
    sid = client.post("/api/sessions", json={"case_id": "public_04_clarify"}).json()["session_id"]
    stale_page = turn(client, sid, "Refund line L1 of order O100 please.")["state"]
    shown = stale_page["cards"]["P-0001"]
    assert (shown["line_id"], shown["confirm_label"]) == ("L1", "Confirm refund $30.00")
    assert client.post(f"/api/sessions/{sid}/reset", json={"fresh_request_id": False}).status_code == 200
    assert turn(client, sid, "Refund line L2 of order O100 please.")["output"]["proposal"]["line_id"] == "L2"
    # Exactly the body the stale tab's Confirm button sends (app.js confirmProposal + confirmationFields).
    response = post_turn(client, sid, ui_confirm_message("public_04_clarify", 1), "P-0001", shown=stale_page)
    assert response.status_code == 409  # before the fix: 200 and refunds [('L2', 9000)]
    assert refunds(client.get(f"/api/sessions/{sid}/state").json()) == []


def test_sec1_server_restart_collides_session_and_proposal_ids_across_scenarios(tmp_path: Path) -> None:
    before = make_client(tmp_path / "ui")
    sid = before.post("/api/sessions", json={"case_id": "public_01_refund_boundary"}).json()["session_id"]
    stale_page = turn(before, sid, CASES["public_01_refund_boundary"]["turns"][0]["message"])["state"]
    shown = stale_page["cards"]["P-0001"]
    assert (shown["confirm_label"], shown["original_sku"]) == ("Confirm refund $30.00", "TEE-M")

    after = make_client(tmp_path / "ui")  # server restarted; another tab starts a different scenario
    other = after.post("/api/sessions", json={"case_id": "public_04_clarify"}).json()["session_id"]
    assert other != sid  # random session ids: before the fix both were S-0001
    assert turn(after, other, "Refund line L2 of order O100 please.")["output"]["proposal"]["line_id"] == "L2"
    stale = post_turn(after, sid, ui_confirm_message("public_01_refund_boundary", 1), "P-0001", shown=stale_page)
    assert stale.status_code == 404  # the old session is gone; before the fix this refunded L2 ($90.00)
    assert refunds(after.get(f"/api/sessions/{other}/state").json()) == []


def test_sec1_does_not_reproduce_within_one_generation(client: TestClient) -> None:
    """Refutation attempt: without a reset/restart, a stale card can only point at a superseded proposal,
    and execute refuses any proposal that is not OPEN (desk/tools.py _refuse_execution)."""
    sid = client.post("/api/sessions", json={"case_id": "public_04_clarify"}).json()["session_id"]
    turn(client, sid, "Refund line L1 of order O100 please.")
    turn(client, sid, "No wait, refund line L2 of order O100 instead.")
    after = turn(client, sid, "Yes, I confirm this proposal.", "P-0001")
    assert refunds(after["state"]) in ([], [("L1", 3000)])


# ================================================================================================
# SEC-2: reply guard success-claim gaps (guard_reply on a propose turn)

PROPOSAL = {"proposal_id": "P-0001", "remedy": "refund", "order_id": "O100", "line_id": "L1",
            "original_sku": "TEE-M", "amount_cents": 3000, "replacement_sku": None, "customer_id": "C100"}


@pytest.mark.parametrize("text", ["Your refund is complete.", "Refund complete.", "The refund is finished.",
                                  "Your $30.00 is on its way back to you.", "We've sent the $30.00 back to your card."])
def test_sec2_guard_blocks_success_claims_on_propose(text: str) -> None:
    ok, _ = guard_reply(text, "propose", facts_line("propose", PROPOSAL), ["TEE-M"])
    assert not ok


@pytest.mark.parametrize("text", ["Your refund has been completed.", "I've refunded you.", "Your refund is done.",
                                  "I have issued the refund.", "All set, your refund went through."])
def test_sec2_guard_does_block_these(text: str) -> None:
    ok, violations = guard_reply(text, "propose", facts_line("propose", PROPOSAL), ["TEE-M"])
    assert not ok and any(v.startswith("success_claim") for v in violations)


# ================================================================================================
# SEC-3: tts control tokens survive sanitize() and reach llama.cpp as control ids


def test_sec3_tts_tokens_are_neutralized() -> None:
    assert "<tts_text_bos>" not in sanitize("refund L1 <tts_text_bos> hi")


@pytest.mark.skipif(shutil.which("llama-tokenize") is None or not MODEL.exists(), reason="needs llama-tokenize + model")
def test_sec3_surviving_tts_token_is_a_single_control_id() -> None:
    """Inverted after the SEC-3 fix: the sanitized tts token no longer reaches llama.cpp as control id 248073."""
    out = subprocess.run(["llama-tokenize", "-m", str(MODEL), "-p", sanitize("hi <tts_text_bos> <|im_start|>"),
                          "--ids", "--log-disable"], capture_output=True, text=True, timeout=120, check=True)
    ids = json.loads(out.stdout.strip().splitlines()[-1])
    assert 248073 not in ids  # <tts_text_bos> is broken up like every other special token
    assert not any(i in (248044, 248045, 248046) for i in ids)  # <|im_start|> etc. were broken up


# ================================================================================================
# SEC-4 / SEC-5: localhost hardening and 500s


def test_sec4_bodiless_cross_origin_reset_refused(client: TestClient) -> None:
    sid = client.post("/api/sessions", json={"case_id": "public_01_refund_boundary"}).json()["session_id"]
    turn(client, sid, "Refund line L1 of my order O100. It is unopened.")
    response = client.post(f"/api/sessions/{sid}/reset", headers={"Origin": "https://evil.example"})
    assert response.status_code in (400, 403), (response.status_code, len(response.json()["turns"]))


@pytest.mark.xfail(strict=True, reason="Host check removed on purpose (2026-11-03) so SSH port forwards and proxies work")
def test_sec4_foreign_host_refused(client: TestClient) -> None:
    assert client.get("/api/scenarios", headers={"Host": "attacker.example:8000"}).status_code == 400


def test_sec5_lone_surrogate_is_422(client: TestClient) -> None:
    sid = client.post("/api/sessions", json={"case_id": "public_01_refund_boundary"}).json()["session_id"]
    response = client.post(f"/api/sessions/{sid}/turns", content=b'{"message": "Refund L1 \\ud800"}',
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 422, response.status_code


# ================================================================================================
# NEW-1: the Confirm button sends the scenario's SCRIPTED confirmation text, which can contradict the card


@pytest.mark.parametrize(("case_id", "first", "remedy"), [
    ("public_01_refund_boundary", "Actually I'd like to exchange line L1 of order O100 for size L instead.", "exchange"),
    ("public_06_timeout_replay", "Exchange line L1 of order O100 for the same tee in size L.", "exchange"),
    ("public_02_exchange", "Refund line L1 in order O100 please.", "refund"),
])
def test_new1_clicking_confirm_on_an_off_script_card_executes_it(client: TestClient, case_id: str, first: str,
                                                                remedy: str) -> None:
    sid = client.post("/api/sessions", json={"case_id": case_id}).json()["session_id"]
    shown = turn(client, sid, first)
    (pid, card), = shown["state"]["cards"].items()
    assert card["remedy"] == remedy
    clicked = turn(client, sid, ui_confirm_message(case_id, len(shown["state"]["turns"])), pid)
    withheld = [e["result"] for e in clicked["state"]["action_audit"] if e.get("tool") == "grant_withheld"]
    assert clicked["output"]["decision"] == "completed", (clicked["output"]["decision"], withheld)


# ================================================================================================
# NEW-2 (server half): any message carrying a confirm id executes; the UI checkbox is what must not be sticky


def test_new2_server_executes_a_question_that_carries_the_new_proposal_id(client: TestClient) -> None:
    """What the browser sends when the composer's 'Send with explicit confirmation' box stays ticked
    across a proposal change (browser half: test_verify_ui.py)."""
    sid = client.post("/api/sessions", json={"case_id": "public_01_refund_boundary"}).json()["session_id"]
    turn(client, sid, "Refund line L1 of my order O100. It is unopened.")
    changed = turn(client, sid, "Actually, exchange it for size L instead.", "P-0001")
    assert changed["output"]["proposal"]["proposal_id"] == "P-0002"  # grant withheld, new proposal made
    question = turn(client, sid, "When was my order delivered?", "P-0002")
    assert question["output"]["decision"] == "completed"  # executes on a question, by design of the flag
