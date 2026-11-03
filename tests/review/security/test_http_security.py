"""Security review, HTTP layer (docs/review/security.md). In-process TestClient, fake-policy LLM, no network.

Each finding test asserts the SAFE behaviour. They were `xfail(strict=True)` while the finding was open;
SEC-1, SEC-4 and SEC-5 are fixed, so the markers are gone and the tests guard the fixes. The other tests
pin properties that already held (regression guards for the review's "no finding" items).

SEC-4's Host check was removed on purpose (any Host is served, for SSH port forwards), so that one test
is xfail(strict=True) again; the Origin check and body cap still hold.
"""

from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from desk.profiles import get_profile
from desk.session import CaseSession
from desk.testing.fake_policy_llm import FakePolicyLLM
from server.app import FAKE_POLICY, ServerConfig, create_app

ROOT = Path(__file__).resolve().parents[3]
CASES = {c["case_id"]: c for c in json.loads((ROOT / "candidate_assets" / "public_cases.json").read_text())["cases"]}


def parse_sse(body: str) -> list[tuple[str, Any]]:
    events = []
    for block in body.split("\n\n"):
        lines = [line for line in block.split("\n") if line and not line.startswith(":")]
        if not lines:
            continue
        kind = next(line[len("event: "):] for line in lines if line.startswith("event: "))
        events.append((kind, json.loads("".join(line[len("data: "):] for line in lines if line.startswith("data: ")))))
    return events


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(ServerConfig(llm_mode=FAKE_POLICY, var_dir=tmp_path / "ui")), raise_server_exceptions=False,
                      base_url="http://127.0.0.1")


def confirmation(client: TestClient, sid: str, proposal_id: str) -> dict[str, Any]:
    """The fields a page sends with a Confirm click: the card it shows, bound by hash and generation (SEC-1)."""
    state = client.get(f"/api/sessions/{sid}/state").json()
    return {"confirm_proposal_id": proposal_id, "confirm_binding_hash": state["cards"][proposal_id]["binding_hash"],
            "generation": state["generation"]}


def turn(client: TestClient, sid: str, message: str, confirm: str | None = None) -> dict[str, Any]:
    body = {"message": message, **(confirmation(client, sid, confirm) if confirm else {})}
    response = client.post(f"/api/sessions/{sid}/turns", json=body)
    assert response.status_code == 200, response.text
    kind, data = parse_sse(response.text)[-1]
    assert kind == "turn", (kind, data)
    return data


# ================================================================================================
# FINDING SEC-1: a stale tab's Confirm executes a DIFFERENT proposal (approval not bound to what was shown)


def test_stale_tab_confirm_cannot_execute_a_proposal_it_never_displayed(client: TestClient) -> None:
    """Two tabs share one session (the UI restores it from localStorage). Tab B shows
    'Confirm refund $30.00' for L1; tab A resets and asks for L2; tab B's Confirm must not refund L2 ($90)."""
    sid = client.post("/api/sessions", json={"case_id": "public_04_clarify"}).json()["session_id"]
    tab_b = turn(client, sid, "Refund line L1 of order O100 please.")
    (shown_id, shown_card), = tab_b["state"]["cards"].items()
    assert (shown_card["line_id"], shown_card["confirm_label"]) == ("L1", "Confirm refund $30.00")

    assert client.post(f"/api/sessions/{sid}/reset", json={"fresh_request_id": False}).status_code == 200  # tab A
    tab_a = turn(client, sid, "Refund line L2 of order O100 please.")
    assert tab_a["output"]["proposal"]["proposal_id"] == shown_id  # same counter id, different facts
    assert tab_a["output"]["proposal"]["line_id"] == "L2"

    # Tab B clicks its stale button. It sends what it displayed (binding hash, generation), as the UI does.
    response = client.post(f"/api/sessions/{sid}/turns", json={
        "message": "Yes, I confirm this proposal.", "confirm_proposal_id": shown_id,
        "confirm_binding_hash": shown_card.get("binding_hash") or shown_card["binding_hash_short"],
        "generation": tab_b["state"]["generation"]})
    assert response.status_code == 409  # refused outright: nothing ran
    assert client.get(f"/api/sessions/{sid}/state").json()["final_state"]["refunds"] == []


# ================================================================================================
# FINDING SEC-4: hardening of a localhost demo server (CSRF on the bodiless reset, DNS rebinding)


def test_a_cross_site_bodiless_reset_is_refused(client: TestClient) -> None:
    sid = client.post("/api/sessions", json={"case_id": "public_01_refund_boundary"}).json()["session_id"]
    turn(client, sid, "Refund line L1 of my order O100. It is unopened.")
    # What <form method=post> or fetch(..., {mode: "no-cors", method: "POST"}) from any website sends:
    response = client.post(f"/api/sessions/{sid}/reset", headers={"Origin": "https://evil.example"})
    assert response.status_code in (400, 403)


@pytest.mark.xfail(strict=True, reason="Host check removed on purpose (2026-11-03) so SSH port forwards and proxies work")
def test_a_foreign_host_header_is_refused(client: TestClient) -> None:
    response = client.get("/api/scenarios", headers={"Host": "attacker.example:8000"})
    assert response.status_code == 400


def test_json_turns_and_faults_cannot_be_sent_as_simple_cross_site_requests(client: TestClient) -> None:
    """text/plain bodies (the only JSON a cross-site page can send without a CORS preflight) are refused:
    403 from the Origin check (SEC-4) before the 422 of the body check is reached."""
    sid = client.post("/api/sessions", json={"case_id": "public_06_timeout_replay"}).json()["session_id"]
    plain = {"content-type": "text/plain", "Origin": "https://evil.example"}
    assert client.post(f"/api/sessions/{sid}/turns", content=b'{"message": "hi"}', headers=plain).status_code in (403, 422)
    assert client.post(f"/api/sessions/{sid}/faults", content=b'{"type": "timeout_after_commit"}',
                       headers=plain).status_code in (403, 422)
    same_origin = {"content-type": "text/plain"}  # no Origin: the body check alone refuses it
    assert client.post(f"/api/sessions/{sid}/turns", content=b'{"message": "hi"}', headers=same_origin).status_code == 422
    assert "access-control-allow-origin" not in client.get("/api/scenarios", headers={"Origin": "https://evil.example"}).headers


# ================================================================================================
# FINDING SEC-5: a lone surrogate in a message is a 500 instead of an actionable 422


def test_a_lone_surrogate_message_is_a_422_not_a_500(client: TestClient) -> None:
    sid = client.post("/api/sessions", json={"case_id": "public_01_refund_boundary"}).json()["session_id"]
    response = client.post(f"/api/sessions/{sid}/turns", content=b'{"message": "Refund L1 \\ud800"}',
                           headers={"content-type": "application/json"})
    assert response.status_code == 422


# ================================================================================================
# Properties that hold (regression guards)


@pytest.mark.parametrize("raw", ["refund L1\u0000 of O100", "\u0000​", "refund ‮ L1 of O100", "x" * 1999])
def test_control_characters_and_long_messages_never_break_a_turn(client: TestClient, raw: str) -> None:
    sid = client.post("/api/sessions", json={"case_id": "public_01_refund_boundary"}).json()["session_id"]
    data = turn(client, sid, raw)
    assert data["output"]["decision"] in ("propose", "clarify", "deny")
    assert client.get(f"/api/sessions/{sid}/state").status_code == 200
    llm_text = json.dumps([e for e in data["state"]["action_audit"] if e["kind"] == "grounding"])
    assert "\\u0000" not in llm_text and "\\u202e" not in llm_text


@pytest.mark.parametrize("path", ["/static/../server/app.py", "/static/%2e%2e/server/app.py", "/static/..%2fapp.py",
                                  "/static/%2e%2e%2f%2e%2e%2fpyproject.toml", "/api/sessions/..%2f..%2fetc/state",
                                  "/api/sessions/S-0001%00/state"])
def test_no_path_traversal_through_static_files_or_session_ids(client: TestClient, path: str) -> None:
    client.post("/api/sessions", json={"case_id": "public_01_refund_boundary"})
    assert client.get(path).status_code == 404


def test_two_simultaneous_confirms_commit_once(client: TestClient) -> None:
    sid = client.post("/api/sessions", json={"case_id": "public_01_refund_boundary"}).json()["session_id"]
    pid = turn(client, sid, "Refund line L1 of my order O100. It is unopened.")["output"]["proposal"]["proposal_id"]
    binding = confirmation(client, sid, pid)
    codes: list[int] = []
    barrier = threading.Barrier(2)

    def confirm() -> None:
        barrier.wait()
        codes.append(client.post(f"/api/sessions/{sid}/turns",
                                 json={"message": "Yes, refund the proposed item and amount.", **binding}).status_code)

    threads = [threading.Thread(target=confirm) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    state = client.get(f"/api/sessions/{sid}/state").json()["final_state"]
    assert len(state["refunds"]) == 1 and len(state["quarantine"]) == 1
    assert sorted(codes) in ([200, 200], [200, 409])  # 409 when they overlap; a later one is a replay


def test_another_customers_order_is_indistinguishable_from_a_missing_one() -> None:
    """Same decision, reply and audit (modulo the typed id) for a foreign order and a nonexistent one."""
    state = copy.deepcopy(CASES["public_02_exchange"]["initial_state"])
    foreign = copy.deepcopy(state["orders"][0])
    foreign.update(order_id="O777", customer_id="C999")
    foreign["lines"][0].update(line_id="L5", sku="JACKET-M", paid_cents=8123)
    state["orders"].append(foreign)

    def run(message: str) -> tuple[dict[str, Any], str]:
        session = CaseSession(initial_state=state, customer_id="C100", request_id="R1", faults=[], llm=FakePolicyLLM(),
                              profile=get_profile("F"))
        return session.run_turn(message), json.dumps(session.audit.entries())

    (out_foreign, audit_foreign), (out_missing, audit_missing) = run("Refund line L5 of order O777 please."), run(
        "Refund line L5 of order O778 please.")
    norm = lambda text: text.replace("O777", "O###").replace("O778", "O###")  # noqa: E731
    assert out_foreign["decision"] == out_missing["decision"] == "deny"
    assert norm(out_foreign["response"]) == norm(out_missing["response"])
    assert norm(audit_foreign) == norm(audit_missing)
    assert "C999" not in audit_foreign and "8123" not in audit_foreign
