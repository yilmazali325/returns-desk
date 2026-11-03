"""Tests for the UI server (server/app.py) and the fake-policy LLM its E2E tests use (desk/testing).

Server: every endpoint's success and error paths through FastAPI's TestClient, the SSE turn stream
(event order, entries pushed while the turn is still running, crashes), the failure lab, reset,
telemetry and the command line. The strongest check: every public case driven through the HTTP API
ends with the same decisions, final_state and audit trail as `run_case` (the adapter's path), so the
UI and the adapter really run the same desk.

Fake-policy LLM: it must drive the real orchestrator to the expected decisions without any guard,
fallback or gate firing, and each of its branches is exercised with prompts built by the real prompt
code (desk.agents.prompts), not with hand-written strings.
"""

from __future__ import annotations

import copy
import json
import queue
import re
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import server.app as server_app
from desk.agents.prompts import (
    CONFIRMATION_NONE,
    NO_CARD_TEXT,
    RESOLUTION_SYSTEM,
    account_card_text,
    confirmation_granted,
    ops_brief,
    turn_block,
)
from desk.agents.toolspecs import (
    CREATE_PROPOSAL,
    EXECUTE_PROPOSAL,
    FIND_ORDER_LINES,
    GET_REQUEST_STATUS,
    HANDOFF,
    LIST_EXCHANGE_OPTIONS,
    OPERATIONS_TOOLS_FREE_TEXT,
    OPERATIONS_TOOLS_STRUCTURED,
    REPLY,
    REPORT,
    RESOLUTION_TOOLS,
)
from desk.llm import LLMClient, LLMError, tool_message
from desk.orchestrator import ALREADY_COMPLETED_NOTE, CANCEL_NOTE, REPLAY_NOTE
from desk.profiles import get_profile
from desk.sanitize import sanitize
from desk.session import run_case
from desk.telemetry import Footprint
from desk.testing import fake_policy_llm as fp
from desk.testing.fake_policy_llm import FakePolicyLLM
from server.app import (
    DEMO_CASES,
    FAKE_POLICY,
    REAL_LLM,
    CreateSessionRequest,
    LLMHealth,
    ServerConfig,
    SessionError,
    TelemetryBoard,
    _base_request_id,
    _drain,
    build_llm,
    create_app,
    fault_text,
    proposal_card,
    scenario_title,
    spec_from_request,
    start_turn,
)

ROOT = Path(__file__).resolve().parents[2]
CASES = json.loads((ROOT / "candidate_assets" / "public_cases.json").read_text())["cases"]
EXPECTED = {c["case_id"]: c["expected"]["decisions"]
            for c in json.loads((ROOT / "evals" / "expected" / "public_expected.json").read_text())["cases"]}
CASE_IDS = [c["case_id"] for c in CASES]


def case_by_suffix(suffix: str) -> dict[str, Any]:
    return next(c for c in CASES if c["case_id"].endswith(suffix))


LOCAL = "http://127.0.0.1:8000"  # the server refuses any Host that is not this machine (SEC-4)


def local_client(app: Any, **options: Any) -> TestClient:
    return TestClient(app, base_url=LOCAL, **options)


def make_app(tmp_path: Path, llm: Any = None, **config: Any):
    settings = {"llm_mode": FAKE_POLICY, "var_dir": tmp_path / "ui", "demo_cases_path": None, **config}
    return create_app(ServerConfig(**settings), llm=llm)


@pytest.fixture
def app(tmp_path: Path):
    return make_app(tmp_path)


@pytest.fixture
def client(app) -> TestClient:
    return local_client(app)


def parse_sse(body: str) -> list[tuple[str, Any]]:
    events = []
    for block in body.split("\n\n"):
        lines = [line for line in block.split("\n") if line and not line.startswith(":")]
        if not lines:
            continue
        kind = next(line[len("event: "):] for line in lines if line.startswith("event: "))
        data = json.loads("".join(line[len("data: "):] for line in lines if line.startswith("data: ")))
        events.append((kind, data))
    return events


def create(client: TestClient, **body: Any) -> dict[str, Any]:
    response = client.post("/api/sessions", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def displayed_binding(client: TestClient, session_id: str, proposal_id: str) -> dict[str, Any]:
    """What a page showing this session would send with a confirmation of `proposal_id`."""
    shown = client.get(f"/api/sessions/{session_id}/state").json()
    card = shown["cards"].get(proposal_id) or {}
    return {"confirm_binding_hash": card.get("binding_hash", ""), "generation": shown["generation"]}


def post_turn(client: TestClient, session_id: str, message: str, confirm: str | None = None) -> list[tuple[str, Any]]:
    body: dict[str, Any] = {"message": message, "confirm_proposal_id": confirm}
    if confirm:
        body.update(displayed_binding(client, session_id, confirm))
    response = client.post(f"/api/sessions/{session_id}/turns", json=body)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_sse(response.text)


def turn_event(events: list[tuple[str, Any]]) -> dict[str, Any]:
    kind, data = events[-1]
    assert kind == "turn", events[-1]
    return data


def run_over_http(client: TestClient, case: dict[str, Any]) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    """Drive a contract case through the UI API: "last" becomes the explicit id of the last proposal."""
    session_id = create(client, case_id=case["case_id"])["session_id"]
    last_proposal = None
    decisions = []
    data: dict[str, Any] = {}
    for turn in case["turns"]:
        confirm = last_proposal if turn.get("confirm_proposal") == "last" else None
        data = turn_event(post_turn(client, session_id, turn["message"], confirm))
        decisions.append(data["output"]["decision"])
        if data["output"]["proposal"]:
            last_proposal = data["output"]["proposal"]["proposal_id"]
    return decisions, data["state"], data


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ================================================================================================
# The UI path and the adapter path are the same desk


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_every_public_case_over_http_matches_the_adapter_path(client: TestClient, case: dict[str, Any]) -> None:
    decisions, state, _ = run_over_http(client, case)
    reference = run_case(case, FakePolicyLLM(), get_profile("F"))
    assert decisions == EXPECTED[case["case_id"]] == [t["decision"] for t in reference["turns"]]
    assert state["final_state"] == reference["final_state"]
    assert state["proposals"] == reference["proposals"]
    assert state["external_events"] == reference["external_events"]
    trail = [(e["kind"], e["agent"], e["tool"]) for e in state["action_audit"]]
    assert trail == [(e["kind"], e["agent"], e["tool"]) for e in reference["action_audit"]]


def test_turn_stream_sends_start_then_every_audit_entry_then_the_turn(client: TestClient) -> None:
    case = case_by_suffix("refund_boundary")
    session_id = create(client, case_id=case["case_id"])["session_id"]
    events = post_turn(client, session_id, case["turns"][0]["message"])
    kinds = [k for k, _ in events]
    assert kinds[0] == "start" and kinds[-1] == "turn" and set(kinds[1:-1]) == {"audit"}
    assert events[0][1] == {"turn": 1, "message": case["turns"][0]["message"], "confirm_proposal_id": None}
    streamed = [d for k, d in events if k == "audit"]
    final = turn_event(events)
    assert [e["seq"] for e in streamed] == [e["seq"] for e in final["state"]["action_audit"] if e["turn"] == 1]
    assert final["output"]["decision"] == "propose" and final["state"]["busy"] is False
    assert final["latency_ms"] is not None and final["state"]["turns"][0]["output"] == final["output"]


class GatedLLM:
    """The fake policy, but its second step waits until the test releases it."""

    def __init__(self) -> None:
        self.inner = FakePolicyLLM()
        self.release = threading.Event()
        self.calls = 0

    def step(self, *args: Any) -> Any:
        self.calls += 1
        if self.calls == 2:
            assert self.release.wait(5), "test never released the gated step"
        return self.inner.step(*args)


def test_audit_entries_reach_the_stream_while_the_turn_is_still_running(tmp_path: Path) -> None:
    llm = GatedLLM()
    app = make_app(tmp_path, llm=llm)
    client = local_client(app)
    case = case_by_suffix("refund_boundary")
    session_id = create(client, case_id=case["case_id"])["session_id"]
    registry = app.state.registry
    session = registry.get(session_id)
    session.lock.acquire()  # what the route does before streaming
    stream = start_turn(session, registry.board, case["turns"][0]["message"], None)
    assert next(stream).startswith("event: start")
    seen = [next(stream) for _ in range(4)]  # analyze, prefetch, step 1, set_target: all before step 2 ends
    assert all(chunk.startswith("event: audit") for chunk in seen)
    assert session.busy and llm.calls == 2, "the turn is blocked inside step 2 while entries already streamed"
    llm.release.set()
    rest = list(stream)
    assert rest[-1].startswith("event: turn") and not session.busy


class CrashingLLM:
    def step(self, *args: Any) -> Any:
        raise RuntimeError("boom")


def test_a_crashing_turn_sends_an_error_event_and_frees_the_session(tmp_path: Path) -> None:
    client = local_client(make_app(tmp_path, llm=CrashingLLM()))
    session_id = create(client, case_id=CASES[0]["case_id"])["session_id"]
    events = post_turn(client, session_id, "hello")
    assert events[-1][0] == "error" and "RuntimeError: boom" in events[-1][1]["message"]
    again = post_turn(client, session_id, "hello again")  # not 409: the lock was released
    assert again[-1][0] == "error"
    assert client.get(f"/api/sessions/{session_id}/state").json()["turns"] == []


def test_keep_alive_comments_are_sent_while_a_turn_is_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_app, "SSE_KEEPALIVE_S", 0.01)
    events: queue.Queue[Any] = queue.Queue()
    threading.Timer(0.08, lambda: events.put(server_app._DONE)).start()
    chunks = list(_drain(events))
    assert chunks and all(chunk == ": keep-alive\n\n" for chunk in chunks)


# ================================================================================================
# Health, meta, scenarios, static files


def test_health_in_fake_mode_reports_ok_and_the_start_commands(client: TestClient) -> None:
    health = client.get("/api/health").json()
    assert health["llm"] == "ok" and health["mode"] == FAKE_POLICY and health["profile"] == "F"
    assert health["quantization"] is None and "test double" in health["engine"]
    assert health["start"]["run_script"] == "scripts/run.sh --llama-port 8081"
    assert health["start"]["llama_server"].startswith("llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --port 8081 -ngl 99")


def test_health_with_a_dead_model_url_reports_down(tmp_path: Path) -> None:
    port = free_port()
    client = local_client(make_app(tmp_path, llm_mode=REAL_LLM, llama_url=f"http://127.0.0.1:{port}"))
    health = client.get("/api/health").json()
    assert health["llm"] == "down" and health["mode"] == REAL_LLM
    assert health["model"] == "Qwen3.5-4B" and health["quantization"] == "Q4_K_M" and health["protocol"] == "constrained"
    assert f"--port {port}" in health["start"]["llama_server"]


def test_health_checks_are_cached_for_a_second(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(server_app, "server_healthy", lambda url, timeout_s: calls.append(url) or True)
    health = LLMHealth(ServerConfig(llm_mode=REAL_LLM))
    assert health.ok() and health.ok()
    assert len(calls) == 1


def test_meta_exposes_reason_wording_and_code_classes(client: TestClient) -> None:
    meta = client.get("/api/meta").json()
    assert meta["code_class"]["OUT_OF_STOCK"] == "refusal" and meta["code_class"]["SIZE_UNSPECIFIED"] == "gate"
    assert "30-day" in meta["reason_text"]["RETURN_WINDOW_EXPIRED"]
    assert meta["window_days"] == 30  # the card's "within the N-day window" line reads it (review C6)


def test_scenarios_come_from_the_case_file(client: TestClient) -> None:
    scenarios = client.get("/api/scenarios").json()
    assert [s["case_id"] for s in scenarios] == CASE_IDS
    stale = next(s for s in scenarios if s["case_id"].endswith("stale_stock"))
    assert stale["title"] == "05 · stale stock"
    assert stale["fault_texts"] == ["stock of TEE-L becomes 0 before turn 2"]
    assert stale["customers"] == ["C100"] and "TEE-L" in stale["skus"]
    assert stale["initial_state"] == case_by_suffix("stale_stock")["initial_state"]


def test_demo_chats_follow_the_public_cases(tmp_path: Path) -> None:
    scenarios = local_client(make_app(tmp_path, demo_cases_path=DEMO_CASES)).get("/api/scenarios").json()
    ids = [s["case_id"] for s in scenarios]
    assert ids[:len(CASE_IDS)] == CASE_IDS and len(ids) == len(CASE_IDS) + 3
    assert [s["title"] for s in scenarios[len(CASE_IDS):]] == [
        "07 · typos hoodie exchange", "08 · slang size down then refund", "09 · spanglish tee exchange"]
    assert scenarios[-3]["turns"][0]["message"] == "can i exhange my hoodi for a mediun pls its waaay to big"


def test_scenario_titles_and_fault_texts_are_derived_generically() -> None:
    assert scenario_title("batch_12_late_parcel") == "12 · late parcel"
    assert scenario_title("odd_case") == "odd case"
    assert fault_text({"type": "timeout_after_commit"}) == "the first commit times out after it persisted"
    assert fault_text({"type": "power_cut"}) == "unknown fault 'power_cut'"


def test_the_page_and_its_assets_are_served(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200 and "<title>Returns Desk</title>" in page.text
    assert page.headers["cache-control"] == "no-store"
    for asset in ("app.js", "styles.css"):
        assert client.get(f"/static/{asset}").status_code == 200
    assert client.get("/static/missing.js").status_code == 404


# ================================================================================================
# Sessions


def test_a_scenario_session_uses_the_case_identity_and_a_store_file(app, client: TestClient, tmp_path: Path) -> None:
    case = case_by_suffix("timeout_replay")
    created = create(client, case_id=case["case_id"])
    state = created["state"]
    session_id = created["session_id"]
    assert re.fullmatch(r"S-[0-9a-f]{12}", session_id) and session_id == state["session_id"]
    assert (state["customer_id"], state["request_id"], state["case_id"]) == (case["customer_id"], case["request_id"], case["case_id"])
    assert state["final_state"] == case["initial_state"] == state["initial_state"]
    assert state["faults"] == {"scheduled": case["faults"], "timeout_armed": True, "timeout_fired": False}
    assert [e["tool"] for e in state["action_audit"]] == ["timeout_after_commit"]
    assert (tmp_path / "ui" / "sessions" / f"{session_id}-g1.sqlite").exists()
    assert state["busy"] is False and state["turns"] == [] and state["cards"] == {}


def test_a_blank_session_takes_any_state_and_customer(client: TestClient) -> None:
    state = CASES[0]["initial_state"]
    first = create(client, initial_state=state, customer_id="C-OTHER")["state"]
    second = create(client, initial_state=state, customer_id="C-OTHER", request_id="REQ-MINE")["state"]
    third = create(client, initial_state=state, customer_id="C-OTHER")["state"]
    assert (first["request_id"], second["request_id"], third["request_id"]) == ("REQ-UI-0001", "REQ-MINE", "REQ-UI-0002")
    assert first["case_id"] is None and first["faults"]["scheduled"] == []
    # A customer with no orders: the named order is not theirs -> the same not-found deny as a missing one.
    events = post_turn(client, first["session_id"], f"Refund line {CASES[0]['initial_state']['orders'][0]['lines'][0]['line_id']} "
                                                      f"of order {CASES[0]['initial_state']['orders'][0]['order_id']}.")
    assert turn_event(events)["output"]["decision"] == "deny"


def test_a_scenario_session_can_override_customer_and_request(client: TestClient) -> None:
    state = create(client, case_id=CASES[0]["case_id"], customer_id="C-X", request_id="REQ-X")["state"]
    assert (state["customer_id"], state["request_id"]) == ("C-X", "REQ-X")


@pytest.mark.parametrize(("body", "status", "text"), [
    ({"case_id": "nope"}, 404, "unknown scenario"),
    ({"case_id": CASES[0]["case_id"], "initial_state": {}}, 422, "not both"),
    ({}, 422, "send case_id or initial_state"),
    ({"initial_state": CASES[0]["initial_state"]}, 422, "needs customer_id"),
    ({"initial_state": {"as_of": "not a date"}, "customer_id": "C1"}, 422, "as_of: Value error"),
    ({"initial_state": {**CASES[0]["initial_state"], "orders": ["x"]}, "customer_id": "C1"}, 422, "orders.0: Input should be"),
    ({"initial_state": {**CASES[0]["initial_state"], "products": [{**CASES[0]["initial_state"]["products"][0], "stock": "3"}]},
      "customer_id": "C1"}, 422, "products.0.stock: Input should be a valid integer"),
])
def test_session_creation_errors(client: TestClient, body: dict[str, Any], status: int, text: str) -> None:
    response = client.post("/api/sessions", json=body)
    assert response.status_code == status and text in response.json()["detail"]


def test_a_state_the_desk_itself_rejects_is_still_a_422(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Schema-valid but unusable for the desk (its constructor raises): a 422 naming the problem."""
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise ValueError("as_of is in the future")

    monkeypatch.setattr(server_app, "CaseSession", refuse)
    response = client.post("/api/sessions", json={"initial_state": CASES[0]["initial_state"], "customer_id": "C1"})
    assert response.status_code == 422 and "initial_state is not usable: as_of is in the future" in response.json()["detail"]


def test_line_paid_cents_must_be_an_integer(client: TestClient) -> None:
    state = copy.deepcopy(CASES[0]["initial_state"])
    state["orders"][0]["lines"][0]["paid_cents"] = "10"
    state["orders"][0]["delivered_on"] = None  # an undelivered order may have no date
    response = client.post("/api/sessions", json={"initial_state": state, "customer_id": "C1"})
    assert response.status_code == 422 and "orders.0.lines.0.paid_cents" in response.json()["detail"]


def test_spec_from_request_raises_session_errors(app) -> None:
    with pytest.raises(SessionError) as caught:
        spec_from_request(CreateSessionRequest(case_id="missing"), {}, app.state.registry)
    assert caught.value.status == 404


def test_unknown_sessions_are_404_everywhere(client: TestClient) -> None:
    assert client.get("/api/sessions/S-9999/state").status_code == 404
    assert client.post("/api/sessions/S-9999/turns", json={"message": "hi"}).status_code == 404
    assert client.post("/api/sessions/S-9999/faults", json={"type": "timeout_after_commit"}).status_code == 404
    assert client.post("/api/sessions/S-9999/reset").status_code == 404


def test_stale_store_files_from_an_earlier_server_run_are_removed(tmp_path: Path) -> None:
    sessions = tmp_path / "ui" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "S-0001-g1.sqlite").write_text("old")
    (sessions / "notes.txt").write_text("kept")
    client = local_client(make_app(tmp_path))
    create(client, case_id=CASES[0]["case_id"])
    assert (sessions / "notes.txt").exists() and not (sessions / "S-0001-g1.sqlite").exists()


def test_after_a_restart_a_stale_tab_gets_404_not_someone_elses_session(tmp_path: Path) -> None:
    """Random ids: a new server process never hands out the id an old tab still holds."""
    old = create(local_client(make_app(tmp_path)), case_id=CASES[0]["case_id"])["session_id"]
    restarted = local_client(make_app(tmp_path))
    fresh = create(restarted, case_id=CASES[3]["case_id"])["session_id"]
    assert fresh != old and restarted.get(f"/api/sessions/{old}/state").status_code == 404


@pytest.mark.parametrize("body", [{"message": ""}, {"message": "x" * 2001}, {}])
def test_turn_requests_are_validated(client: TestClient, body: dict[str, Any]) -> None:
    session_id = create(client, case_id=CASES[0]["case_id"])["session_id"]
    assert client.post(f"/api/sessions/{session_id}/turns", json=body).status_code == 422


def test_an_empty_confirm_id_means_no_confirmation(client: TestClient) -> None:
    case = case_by_suffix("refund_boundary")
    session_id = create(client, case_id=case["case_id"])["session_id"]
    post_turn(client, session_id, case["turns"][0]["message"])
    events = post_turn(client, session_id, case["turns"][1]["message"], "")
    data = turn_event(events)
    assert data["confirm_proposal_id"] is None and data["output"]["decision"] == "propose"  # re-presented, not executed
    assert data["state"]["final_state"]["refunds"] == []


def test_the_turn_line_serves_waiting_turns_in_arrival_order() -> None:
    line = server_app.TurnLine(limit=1, max_waiting=2)
    first, second, third = line.join(), line.join(), line.join()
    assert first.is_set() and not second.is_set() and not third.is_set()
    assert (line.position(second), line.position(third)) == (1, 2)
    assert line.join() is None  # the line is full
    line.leave()  # the running turn ends: its place goes to the first one waiting
    assert second.is_set() and not third.is_set() and line.position(third) == 1
    line.leave()
    assert third.is_set()
    line.leave()
    assert line.snapshot() == {"max_turns": 1, "running": 0, "waiting": 0, "peak_waiting": 2, "max_waiting": 2,
                               "turns_refused": 1}


def test_a_waiting_turn_streams_its_place_in_line_then_runs(tmp_path: Path) -> None:
    app = make_app(tmp_path, max_turns=1)
    client = local_client(app)
    line = app.state.turn_line
    session_id = create(client, case_id="public_01_refund_boundary")["session_id"]
    line.join()  # another conversation holds the only place
    result: dict[str, Any] = {}
    waiter = threading.Thread(target=lambda: result.update(events=post_turn(client, session_id, "Hi")))
    waiter.start()
    deadline = time.monotonic() + 5
    while line.snapshot()["waiting"] != 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert line.snapshot()["waiting"] == 1
    line.leave()
    waiter.join(10)
    events = result["events"]
    assert events[0] == ("queued", {"turn": 1, "position": 1})
    starts = [data for kind, data in events if kind == "start"]
    assert starts == [{"turn": 1, "message": "Hi", "confirm_proposal_id": None}]
    turn_event(events)
    assert line.snapshot()["running"] == 0  # the finished turn gave its place back


def test_a_full_line_refuses_with_503_and_retry_after_and_nothing_runs(tmp_path: Path) -> None:
    app = make_app(tmp_path, max_turns=1, max_waiting=0)
    client = local_client(app)
    session_id = create(client, case_id="public_01_refund_boundary")["session_id"]
    app.state.turn_line.join()  # the only place is taken and no one may wait
    refused = client.post(f"/api/sessions/{session_id}/turns", json={"message": "Hi"})
    assert refused.status_code == 503 and refused.headers["retry-after"] == str(server_app.RETRY_AFTER_S)
    assert "at capacity" in refused.json()["detail"]
    assert client.get("/api/telemetry").json()["capacity"]["turns_refused"] == 1
    app.state.turn_line.leave()
    events = post_turn(client, session_id, "Hi")
    assert events[0] == ("start", {"turn": 1, "message": "Hi", "confirm_proposal_id": None})  # the refusal ran nothing
    turn_event(events)


def test_a_busy_session_takes_no_place_in_the_line(tmp_path: Path) -> None:
    app = make_app(tmp_path, max_turns=1)
    client = local_client(app)
    session_id = create(client, case_id="public_01_refund_boundary")["session_id"]
    app.state.registry.get(session_id).lock.acquire()  # a turn is still running in this session
    assert client.post(f"/api/sessions/{session_id}/turns", json={"message": "Hi"}).status_code == 409
    assert app.state.turn_line.snapshot()["running"] == 0


def test_capacity_defaults_to_a_few_turns_per_model_slot_and_no_cap_on_the_fake(tmp_path: Path) -> None:
    assert server_app.turn_limit(ServerConfig(slots=2)) == 2 * server_app.TURNS_PER_SLOT
    assert server_app.turn_limit(ServerConfig(llm_mode=FAKE_POLICY)) is None
    assert make_app(tmp_path).state.turn_line is None


def test_a_busy_session_refuses_turns_faults_and_reset_with_409(app, client: TestClient) -> None:
    session_id = create(client, case_id=CASES[0]["case_id"])["session_id"]
    session = app.state.registry.get(session_id)
    session.lock.acquire()
    try:
        assert client.get(f"/api/sessions/{session_id}/state").json()["busy"] is True
        for path, body in (("turns", {"message": "hi"}), ("faults", {"type": "timeout_after_commit"}), ("reset", {})):
            response = client.post(f"/api/sessions/{session_id}/{path}", json=body)
            assert response.status_code == 409 and "still running" in response.json()["detail"]
    finally:
        session.lock.release()


def test_a_repeated_confirmation_replays_without_new_effects(client: TestClient) -> None:
    case = case_by_suffix("refund_boundary")
    session_id = create(client, case_id=case["case_id"])["session_id"]
    proposal_id = turn_event(post_turn(client, session_id, case["turns"][0]["message"]))["output"]["proposal"]["proposal_id"]
    first = turn_event(post_turn(client, session_id, case["turns"][1]["message"], proposal_id))
    second = turn_event(post_turn(client, session_id, case["turns"][1]["message"], proposal_id))
    assert first["output"]["decision"] == second["output"]["decision"] == "completed"
    assert first["output"]["transaction"] == second["output"]["transaction"]
    audit = second["state"]["action_audit"]
    assert sum(1 for e in audit if e["committed_effects"]) == 1
    assert [e["result"]["replay"] for e in audit if e["kind"] == "route"] == [True]
    assert len(second["state"]["final_state"]["refunds"]) == 1


def test_a_confirmation_for_an_unpresented_proposal_is_refused_before_it_runs(client: TestClient) -> None:
    case = case_by_suffix("refund_boundary")
    session_id = create(client, case_id=case["case_id"])["session_id"]
    post_turn(client, session_id, case["turns"][0]["message"])
    response = client.post(f"/api/sessions/{session_id}/turns", json={
        "message": case["turns"][1]["message"], "confirm_proposal_id": "P-9999",
        **displayed_binding(client, session_id, "P-9999")})
    assert response.status_code == 409
    state = client.get(f"/api/sessions/{session_id}/state").json()
    assert len(state["turns"]) == 1 and state["final_state"]["refunds"] == []


# ================================================================================================
# Failure lab and reset


def test_a_stock_change_applies_now_and_is_recorded(client: TestClient) -> None:
    case = case_by_suffix("exchange")
    session_id = create(client, case_id=case["case_id"])["session_id"]
    proposal = turn_event(post_turn(client, session_id, case["turns"][0]["message"]))["output"]["proposal"]
    answer = client.post(f"/api/sessions/{session_id}/faults", json={"type": "stock_change", "sku": proposal["replacement_sku"], "new_stock": 0})
    assert answer.status_code == 200
    event = answer.json()["event"]
    assert (event["applied"], event["new_stock"], event["before_turn"], event["source"]) == (True, 0, None, "external")
    state = answer.json()["state"]
    assert state["cards"][proposal["proposal_id"]]["replacement_stock"] == 0  # the card shows live stock
    data = turn_event(post_turn(client, session_id, case["turns"][1]["message"], proposal["proposal_id"]))
    assert data["output"]["decision"] == "deny"
    assert data["state"]["final_state"]["exchanges"] == [] and data["state"]["proposals"][0]["status"] == "invalidated"


def test_a_stock_change_for_an_unknown_sku_is_recorded_as_not_applied(client: TestClient) -> None:
    session_id = create(client, case_id=CASES[0]["case_id"])["session_id"]
    event = client.post(f"/api/sessions/{session_id}/faults", json={"type": "stock_change", "sku": "NOPE-1", "new_stock": 3}).json()["event"]
    assert event["applied"] is False and event["reason"] == "SKU_NOT_FOUND"


@pytest.mark.parametrize("body", [
    {"type": "stock_change", "new_stock": 1},
    {"type": "stock_change", "sku": "X", "new_stock": -1},
    {"type": "stock_change", "sku": "X", "new_stock": "3"},
    {"type": "stock_change", "sku": "X", "new_stock": 2.5},
    {"type": "stock_change", "sku": "X"},
    {"type": "power_cut"},
])
def test_invalid_fault_requests_are_422(client: TestClient, body: dict[str, Any]) -> None:
    session_id = create(client, case_id=CASES[0]["case_id"])["session_id"]
    assert client.post(f"/api/sessions/{session_id}/faults", json=body).status_code == 422


def test_an_armed_timeout_fires_once_and_the_desk_reconciles(client: TestClient) -> None:
    case = case_by_suffix("refund_boundary")
    session_id = create(client, case_id=case["case_id"])["session_id"]
    armed = client.post(f"/api/sessions/{session_id}/faults", json={"type": "timeout_after_commit"}).json()
    assert armed["armed"] is True and armed["state"]["faults"]["timeout_armed"] is True
    proposal_id = turn_event(post_turn(client, session_id, case["turns"][0]["message"]))["output"]["proposal"]["proposal_id"]
    data = turn_event(post_turn(client, session_id, case["turns"][1]["message"], proposal_id))
    assert data["output"]["decision"] == "completed"
    assert data["state"]["faults"] == {"scheduled": [], "timeout_armed": False, "timeout_fired": True}
    execute = next(e for e in data["state"]["action_audit"] if e["kind"] == "tool" and e["tool"] == "execute_proposal")
    assert execute["error"]["code"] == "TIMEOUT" and execute["committed_effects"] and execute["visible_to_agent"] is False


def test_reset_restores_the_initial_state_and_keeps_the_session(app, client: TestClient, tmp_path: Path) -> None:
    case = case_by_suffix("refund_boundary")
    _, state, _ = run_over_http(client, case)
    session_id = state["session_id"]
    assert state["final_state"]["refunds"]
    reset = client.post(f"/api/sessions/{session_id}/reset", json={})
    assert reset.status_code == 200
    fresh = reset.json()
    assert fresh["final_state"] == case["initial_state"] and fresh["turns"] == [] and fresh["proposals"] == []
    assert fresh["generation"] == 2 and fresh["request_id"] == case["request_id"] and fresh["action_audit"] == []
    sessions = tmp_path / "ui" / "sessions"
    assert not (sessions / f"{session_id}-g1.sqlite").exists() and (sessions / f"{session_id}-g2.sqlite").exists()
    # The same scripted conversation runs again from scratch (the old idempotency record is gone).
    decisions, _, _ = run_over_http_session(client, session_id, case)
    assert decisions == EXPECTED[case["case_id"]]


def run_over_http_session(client: TestClient, session_id: str, case: dict[str, Any]) -> tuple[list[str], dict, dict]:
    last_proposal, decisions, data = None, [], {}
    for turn in case["turns"]:
        confirm = last_proposal if turn.get("confirm_proposal") == "last" else None
        data = turn_event(post_turn(client, session_id, turn["message"], confirm))
        decisions.append(data["output"]["decision"])
        last_proposal = data["output"]["proposal"]["proposal_id"] if data["output"]["proposal"] else last_proposal
    return decisions, data["state"], data


def test_reset_with_a_fresh_request_id_numbers_the_suffix(client: TestClient) -> None:
    case = CASES[0]
    session_id = create(client, case_id=case["case_id"])["session_id"]
    first = client.post(f"/api/sessions/{session_id}/reset", json={"fresh_request_id": True}).json()
    second = client.post(f"/api/sessions/{session_id}/reset", json={"fresh_request_id": True}).json()
    third = client.post(f"/api/sessions/{session_id}/reset").json()  # no body: keep the current id
    assert first["request_id"] == f"{case['request_id']}-R2"
    assert second["request_id"] == f"{case['request_id']}-R3"
    assert third["request_id"] == second["request_id"]


def test_a_new_request_keeps_the_data_and_gets_a_fresh_request_id(client: TestClient) -> None:
    case = case_by_suffix("refund_boundary")
    _, state, _ = run_over_http(client, case)
    session_id = state["session_id"]
    assert state["final_state"]["refunds"]
    answer = client.post(f"/api/sessions/{session_id}/new-request")
    assert answer.status_code == 200
    fresh = answer.json()
    assert fresh["request_id"].startswith("REQ-UI-") and fresh["request_id"] != case["request_id"]
    assert fresh["final_state"] == fresh["initial_state"] == state["final_state"]
    assert fresh["turns"] == [] and fresh["proposals"] == [] and fresh["action_audit"] == []
    assert fresh["case_id"] is None and fresh["faults"]["scheduled"] == []
    assert fresh["customer_id"] == case["customer_id"] and fresh["generation"] == state["generation"] + 1
    # The refunded line stays refunded: asking for it again under the new request is denied and nothing moves.
    data = turn_event(post_turn(client, session_id, case["turns"][0]["message"]))
    assert data["output"]["decision"] == "deny"
    assert data["state"]["final_state"]["refunds"] == state["final_state"]["refunds"]
    # Reset now returns to the start of this request, not of the scenario.
    reset = client.post(f"/api/sessions/{session_id}/reset").json()
    assert reset["final_state"] == state["final_state"] and reset["request_id"] == fresh["request_id"]


def test_a_new_request_is_refused_while_busy_or_cross_site(app, client: TestClient) -> None:
    session_id = create(client, case_id=CASES[0]["case_id"])["session_id"]
    before = client.get(f"/api/sessions/{session_id}/state").json()["request_id"]
    session = app.state.registry.get(session_id)
    session.lock.acquire()
    try:
        response = client.post(f"/api/sessions/{session_id}/new-request")
        assert response.status_code == 409 and "still running" in response.json()["detail"]
    finally:
        session.lock.release()
    response = client.post(f"/api/sessions/{session_id}/new-request", headers={"Origin": "https://evil.example"})
    assert response.status_code == 403 and "cross-site" in response.json()["detail"]
    assert client.post("/api/sessions/S-000000000000/new-request").status_code == 404
    assert client.get(f"/api/sessions/{session_id}/state").json()["request_id"] == before


def test_base_request_id_only_strips_a_numbered_suffix() -> None:
    assert _base_request_id("REQ-7-R12") == "REQ-7"
    assert _base_request_id("REQ-R") == "REQ-R"
    assert _base_request_id("PLAIN") == "PLAIN"


# ================================================================================================
# Proposal cards


def test_refund_card_formats_money_from_integer_cents_and_reads_line_facts(client: TestClient) -> None:
    case = case_by_suffix("refund_boundary")
    session_id = create(client, case_id=case["case_id"])["session_id"]
    data = turn_event(post_turn(client, session_id, case["turns"][0]["message"]))
    card = data["state"]["cards"][data["output"]["proposal"]["proposal_id"]]
    line = case["initial_state"]["orders"][0]["lines"][0]
    assert card["amount_cents"] == line["paid_cents"] and card["amount_text"] == "$30.00" == card["paid_text"]
    assert card["confirm_label"] == "Confirm refund $30.00"
    assert (card["unopened"], card["final_sale"], card["returned"]) == (True, False, False)
    assert (card["days_since_delivery"], card["window_days"]) == (30, 30)
    assert len(card["binding_hash_short"]) == 12 and card["replacement_sku"] is None
    stored = next(p for p in data["state"]["proposals"] if p["proposal_id"] == card["proposal_id"])
    assert card["binding_hash"] == stored["binding_hash"] and card["binding_hash"].startswith(card["binding_hash_short"])


def test_exchange_card_names_the_replacement_and_its_live_stock(client: TestClient) -> None:
    case = case_by_suffix("exchange")
    session_id = create(client, case_id=case["case_id"])["session_id"]
    data = turn_event(post_turn(client, session_id, case["turns"][0]["message"]))
    card = data["state"]["cards"][data["output"]["proposal"]["proposal_id"]]
    assert card["confirm_label"] == "Confirm exchange to TEE-L"
    assert (card["replacement_size"], card["replacement_stock"], card["amount_text"]) == ("L", 3, "$0.00")


def test_a_card_for_a_proposal_whose_line_is_gone_still_renders() -> None:
    proposal = {"proposal_id": "P-1", "customer_id": "C9", "order_id": "ORD-7Q", "line_id": "LN-3", "remedy": "exchange",
                "original_sku": "SCARF-M", "replacement_sku": "SCARF-L", "amount_cents": None, "binding_hash": "ab" * 32}
    card = proposal_card(proposal, {"products": [], "orders": []})
    assert card["product_id"] is None and card["unopened"] is None and card["amount_text"] is None
    assert card["replacement_stock"] is None and card["confirm_label"] == "Confirm exchange to SCARF-L"


# ================================================================================================
# A model that is down, telemetry, configuration


def test_with_the_model_down_a_turn_is_an_error_and_nothing_changes(tmp_path: Path) -> None:
    client = local_client(make_app(tmp_path, llm_mode=REAL_LLM, llama_url=f"http://127.0.0.1:{free_port()}"))
    case = case_by_suffix("refund_boundary")
    session_id = create(client, case_id=case["case_id"])["session_id"]
    data = turn_event(post_turn(client, session_id, case["turns"][0]["message"]))
    assert data["output"]["decision"] == "error" and data["output"]["proposal"] is None
    assert data["state"]["final_state"] == case["initial_state"] and data["state"]["proposals"] == []
    steps = [e for e in data["state"]["action_audit"] if e["kind"] == "llm_step"]
    assert steps and steps[0]["error"]["code"] == "LLM_ERROR"


def test_telemetry_reports_the_last_step_turn_and_memory(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_app.telemetry, "read_footprint", lambda pid: Footprint(pid, 64 * 2**20, 0, 0, 0))
    monkeypatch.setattr(server_app.telemetry, "gpu_in_use_system_memory", lambda: 512 * 2**20)
    assert client.get("/api/telemetry").json()["last_step"] is None
    case = case_by_suffix("refund_boundary")
    session_id = create(client, case_id=case["case_id"])["session_id"]
    post_turn(client, session_id, case["turns"][0]["message"])
    report = client.get("/api/telemetry").json()
    assert report["last_step"]["agent"] == "resolution" and report["last_step"]["tool"] == "reply_to_customer"
    assert report["last_step"]["decode_tok_s"] is None  # the fake does not decode
    assert report["last_turn"]["turn"] == 1 and report["last_turn"]["llm_calls"] >= 3
    assert report["memory"]["server_footprint_mb"] == 64.0 and report["memory"]["llama_footprint_mb"] is None
    assert report["memory"]["gpu_unified_mb"] == 512.0 and report["mode"] == FAKE_POLICY


def test_step_summary_computes_rates_when_timings_exist() -> None:
    entry = {"kind": "llm_step", "agent": "operations", "tool": "create_proposal", "turn": 2, "error": {"code": "INVALID_ARGS"},
             "result": {"latency_ms": 900, "prompt_tokens": 1000, "cached_tokens": 800, "prefill_tokens": 200,
                        "completion_tokens": 40, "predicted_ms": 500.0, "prompt_ms": 100.0}}
    summary = server_app.step_summary(entry)
    assert (summary["decode_tok_s"], summary["prefill_tok_s"], summary["error"]) == (80.0, 2000.0, "INVALID_ARGS")
    board = TelemetryBoard(None)
    board.observe({"kind": "tool"})
    assert board.last_step is None
    board.observe_turn("S-1", 2, 1234.5, [entry, {**entry, "turn": 1}])
    assert board.last_turn["llm_calls"] == 1 and board.last_turn["decode_tok_s"] == 80.0


def test_memory_of_llama_server_is_found_by_port_and_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    lookups = []
    monkeypatch.setattr(server_app, "pid_listening_on", lambda port: lookups.append(port) or 4242)
    alive = {4242}
    monkeypatch.setattr(server_app.telemetry, "read_footprint",
                        lambda pid: Footprint(pid, 2**30, 0, 0, 0) if pid in alive or pid != 4242 else None)
    monkeypatch.setattr(server_app.telemetry, "gpu_in_use_system_memory", lambda: None)
    board = TelemetryBoard(8404)
    first = board.memory()
    assert first["llama_footprint_mb"] == 1024.0 and first["llama_pid"] == 4242 and first["gpu_unified_mb"] is None
    assert board.memory() == first and lookups == [8404]  # cached within a second
    board._memory_read_at = float("-inf")
    assert board.memory()["llama_footprint_mb"] == 1024.0 and lookups == [8404]  # the known pid is read again
    board._memory_read_at = float("-inf")
    alive.clear()  # llama-server restarted: the old pid is gone, so the port is looked up again
    assert board.memory()["llama_footprint_mb"] is None and lookups == [8404, 8404]


def test_memory_readings_degrade_to_null(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_app.telemetry, "read_footprint", lambda pid: None)
    monkeypatch.setattr(server_app, "pid_listening_on", lambda port: None)
    memory = TelemetryBoard(8404)._read_memory()
    assert memory["footprint_mb"] is None and memory["server_footprint_mb"] is None and memory["llama_pid"] is None


def test_config_reads_the_environment() -> None:
    config = ServerConfig.from_env({"DESK_LLM": "fake-policy", "DESK_LLAMA_URL": "http://127.0.0.1:8404", "DESK_PROFILE": "B",
                                    "DESK_CASES": "/x.json", "DESK_VAR_DIR": "/v", "DESK_FAKE_STEP_MS": "150"})
    assert config.fake and config.llama_port == 8404 and config.profile == "B"
    assert (config.cases_path, config.var_dir, config.fake_step_delay_s) == (Path("/x.json"), Path("/v"), 0.15)
    default = ServerConfig.from_env({})
    assert not default.fake and default.llm_mode == REAL_LLM and default.llama_port == 8081
    assert default.allow_engine_mismatch is False and default.port is None
    assert ServerConfig.from_env({"DESK_ALLOW_ENGINE_MISMATCH": "1"}).allow_engine_mismatch is True
    assert ServerConfig(llama_url="http://localhost").llama_port == 80


def test_build_llm_picks_the_client_for_the_mode() -> None:
    real = build_llm(ServerConfig(llm_mode=REAL_LLM, llama_url="http://127.0.0.1:1"), get_profile("B"))
    assert isinstance(real, LLMClient) and real.protocol == "native"
    real.close()
    fake = build_llm(ServerConfig(llm_mode=FAKE_POLICY, fake_step_delay_s=0.2), get_profile("F"))
    assert isinstance(fake, FakePolicyLLM) and fake.step_delay_s == 0.2


# ================================================================================================
# Command line: warm-up, llama-server attach/start, main()


class RecordingClient:
    """Stands in for LLMClient inside warm_up."""

    instances: list[RecordingClient] = []

    def __init__(self, base_url: str, protocol: str, **kwargs: Any) -> None:
        self.base_url, self.protocol, self.kwargs, self.steps = base_url, protocol, kwargs, []
        RecordingClient.instances.append(self)

    def __enter__(self) -> RecordingClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def step(self, role: str, messages: list, tools: Any, allowed: list, max_tokens: int) -> None:
        self.steps.append((role, messages[0]["content"][:40], list(allowed), max_tokens))


def test_warm_up_primes_both_agents_static_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    RecordingClient.instances = []
    monkeypatch.setattr(server_app, "LLMClient", RecordingClient)
    timings = server_app.warm_up("http://127.0.0.1:8404", get_profile("F"))
    assert set(timings) == {"resolution", "operations"}
    client = RecordingClient.instances[0]
    assert client.protocol == "constrained" and [s[0] for s in client.steps] == ["resolution", "operations"]
    assert client.steps[0][1] == RESOLUTION_SYSTEM[:40] and client.steps[0][2] == [REPLY] and client.steps[1][2] == [REPORT]


def test_ensure_llama_attaches_to_a_healthy_server(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(server_app, "server_healthy", lambda url: True)
    monkeypatch.setattr(server_app, "warm_up", lambda url, profile: {"resolution": 1.0})
    assert server_app.ensure_llama(ServerConfig(llm_mode=REAL_LLM), get_profile("F")) is None
    out = capsys.readouterr().out
    assert "attached to the running llama-server" in out and "prompt cache warmed" in out


class FakeLlamaServer:
    engine = "0.7.0 (build 11669, commit 620d2225c)"

    def __init__(self, profile: Any, port: int, log: Path, allow_engine_mismatch: bool = False, slots: int = 1) -> None:
        self.profile, self.port, self.log, self.stopped = profile, port, log, False
        self.allow_engine_mismatch, self.slots = allow_engine_mismatch, slots

    def start(self) -> dict[str, Any]:
        expected = "0.7.0 (build 11669, commit 620d2225c)"
        return {"server_spawn_to_ready_ms": 1500.0, "first_request_ms": 90.0, "model_page_cache": "warm",
                "engine_version": self.engine, "engine_version_expected": expected,
                "engine_version_match": self.engine == expected}

    def stop(self) -> None:
        self.stopped = True


def test_ensure_llama_starts_its_own_server_and_survives_a_failed_warm_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                                          capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(server_app, "server_healthy", lambda url: False)
    monkeypatch.setattr(server_app, "LlamaServer", FakeLlamaServer)

    def broken_warm_up(url: str, profile: Any) -> dict[str, float]:
        raise LLMError("timeout", "slow")

    monkeypatch.setattr(server_app, "warm_up", broken_warm_up)
    config = ServerConfig(llm_mode=REAL_LLM, llama_url="http://127.0.0.1:8404", var_dir=tmp_path)
    started = server_app.ensure_llama(config, get_profile("F"))
    assert isinstance(started, FakeLlamaServer) and started.port == 8404 and started.log == tmp_path / "llama-server-8404.log"
    out = capsys.readouterr().out
    assert "llama-server ready on port 8404" in out and "warm-up skipped: timeout: slow" in out
    assert server_app.ensure_llama(config, get_profile("F"), warm=False) is not None


def test_ensure_llama_passes_the_mismatch_override_and_warns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                             capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(server_app, "server_healthy", lambda url: False)
    monkeypatch.setattr(FakeLlamaServer, "engine", "0.7.1 (build 11702, commit abcdef123)")
    monkeypatch.setattr(server_app, "LlamaServer", FakeLlamaServer)
    config = ServerConfig(llm_mode=REAL_LLM, llama_url="http://127.0.0.1:8404", var_dir=tmp_path, allow_engine_mismatch=True)
    started = server_app.ensure_llama(config, get_profile("F"), warm=False)
    assert started is not None and started.allow_engine_mismatch is True
    assert "WARNING: llama.cpp '0.7.1 (build 11702, commit abcdef123)' is not the pinned" in capsys.readouterr().out


@pytest.fixture(autouse=True)
def keep_signal_handlers(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """main() installs a SIGTERM handler; record it instead of changing the test process."""
    installed: list[Any] = []
    monkeypatch.setattr(server_app.signal, "signal", lambda signum, handler: installed.append((signum, handler)))
    return installed


def test_sigterm_becomes_a_clean_exit() -> None:
    with pytest.raises(SystemExit) as caught:
        server_app._exit_on_signal(15, None)
    assert caught.value.code == 143


def test_main_serves_the_ui_and_stops_the_server_it_started(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                            keep_signal_handlers: list[Any]) -> None:
    served = []
    started = FakeLlamaServer(None, 8404, tmp_path)
    monkeypatch.setenv("DESK_VAR_DIR", str(tmp_path))
    monkeypatch.delenv("DESK_LLM", raising=False)
    configs: list[ServerConfig] = []
    monkeypatch.setattr(server_app.uvicorn, "run", lambda app, host, port, log_level: served.append((host, port, app.title)))
    monkeypatch.setattr(server_app, "ensure_llama", lambda config, profile, warm: configs.append(config) or started)
    assert server_app.main(["--port", "8410", "--llama-url", "http://127.0.0.1:8404", "--start-llama",
                            "--allow-engine-mismatch"]) == 0
    assert served == [("127.0.0.1", 8410, "Returns Desk")] and started.stopped
    assert configs[0].allow_engine_mismatch is True and configs[0].port == 8410
    assert configs[0].allowed_hostnames == ("127.0.0.1", "localhost", "::1")
    assert keep_signal_handlers[:2] == [(server_app.signal.SIGTERM, server_app._exit_on_signal),
                                        (server_app.signal.SIGHUP, server_app._exit_on_signal)]
    assert server_app.main(["--port", "8411", "--fake", "--start-llama"]) == 0  # fake mode never starts a model
    assert served[-1][1] == 8411


@pytest.mark.parametrize(("host", "names"), [("0.0.0.0", ("127.0.0.1", "localhost", "::1")),
                                             ("[::1]", ("127.0.0.1", "localhost", "::1")),
                                             ("Desk.Local", ("127.0.0.1", "localhost", "::1", "desk.local"))])
def test_main_also_accepts_the_host_it_binds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, host: str,
                                             names: tuple[str, ...]) -> None:
    monkeypatch.setenv("DESK_VAR_DIR", str(tmp_path))
    seen: list[ServerConfig] = []
    monkeypatch.setattr(server_app.uvicorn, "run", lambda app, host, port, log_level: seen.append(app.state.config))
    assert server_app.main(["--fake", "--host", host, "--port", "8412"]) == 0
    assert seen[0].allowed_hostnames == names


def test_main_exits_3_when_llama_server_cannot_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DESK_VAR_DIR", str(tmp_path))
    monkeypatch.delenv("DESK_LLM", raising=False)

    def failing(config: Any, profile: Any, warm: bool) -> None:
        raise server_app.LlamaServerError("model file missing")

    monkeypatch.setattr(server_app, "ensure_llama", failing)
    monkeypatch.setattr(server_app.uvicorn, "run", lambda *a, **k: pytest.fail("must not serve"))
    assert server_app.main(["--start-llama"]) == 3


# ================================================================================================
# The fake-policy LLM


@pytest.mark.parametrize("profile_name", ["F", "B"])
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_fake_policy_reaches_the_expected_decisions_without_any_override(case: dict[str, Any], profile_name: str) -> None:
    result = run_case(case, FakePolicyLLM(), get_profile(profile_name))
    assert [t["decision"] for t in result["turns"]] == EXPECTED[case["case_id"]]
    assert not [e for e in result["action_audit"] if e["kind"] in ("guard", "fallback", "gate")]
    assert result["metrics"]["protocol_errors"] == 0 and result["metrics"]["status"] == "ok"


LINES = [
    {"order_id": "ORD-7Q", "line_id": "LN-3", "sku": "SCARF-M", "product_id": "SCARF", "size": "M", "aliases": ["scarf", "scarves"], "paid_cents": 2500},
    {"order_id": "ORD-7Q", "line_id": "LN-4", "sku": "CAP-S", "product_id": "CAP", "size": "S", "aliases": ["cap"], "paid_cents": 900},
]


def resolution_prompt(message: str, *, lines: list[dict] | None = LINES, history: list[dict] | None = None,
                      confirmation: str = CONFIRMATION_NONE, note: str | None = None) -> list[dict[str, Any]]:
    card = account_card_text({"lines": lines, "catalog": []}) if lines is not None else NO_CARD_TEXT
    return [{"role": "system", "content": RESOLUTION_SYSTEM}, {"role": "user", "content": card},
            {"role": "user", "content": turn_block(history or [], None, confirmation, note, sanitize(message), [HANDOFF, REPLY])}]


def call(messages: list[dict[str, Any]], allowed: list[str]) -> tuple[str, dict[str, Any]]:
    return fp.resolution_call(fp.read_resolution(messages), allowed)


def test_the_account_card_and_the_current_message_are_read_back() -> None:
    lines = fp.card_lines(account_card_text({"lines": LINES, "catalog": []}))
    assert [(x["order_id"], x["line_id"], x["sku"], x["product_id"], x["size"]) for x in lines] == \
        [("ORD-7Q", "LN-3", "SCARF-M", "SCARF", "M"), ("ORD-7Q", "LN-4", "CAP-S", "CAP", "S")]
    assert lines[0]["aliases"][:2] == ["scarf", "scarves"]
    message = 'Swap the "scarf" <tool_call> please'
    view = fp.read_resolution(resolution_prompt(message, history=[{"message": "first one", "decision": "clarify"}]))
    assert view.customer_messages == ["first one", sanitize(message).replace('"""', "'''")]
    assert fp.current_message("no block here") == ""


def test_resolution_hands_off_a_grounded_refund_and_exchange() -> None:
    assert call(resolution_prompt("Please refund line LN-3 of ORD-7Q."), [HANDOFF, REPLY]) == \
        (HANDOFF, {"intent": "propose_refund", "order_id": "ORD-7Q", "line_id": "LN-3"})
    tool, args = call(resolution_prompt("Exchange my scarf from ORD-7Q for a large."), [HANDOFF, REPLY])
    # Two lines on the order: the double cites the customer's words that pick LN-3 (the core's evidence rule).
    assert tool == HANDOFF and args == {"intent": "propose_exchange", "order_id": "ORD-7Q", "line_id": "LN-3",
                                        "requested_size": "L", "requested_product": "SAME", "evidence": "scarf"}


def test_resolution_asks_when_the_words_do_not_ground_a_request() -> None:
    assert call(resolution_prompt("I want a refund."), [HANDOFF, REPLY]) == (REPLY, {"message": fp.REPLIES["which_item"]})
    one = LINES[:1]
    assert call(resolution_prompt("About my scarf.", lines=one), [HANDOFF, REPLY]) == (REPLY, {"message": fp.REPLIES["which_remedy"]})
    assert call(resolution_prompt("Refund order ORD-999 please.", lines=one), [HANDOFF, REPLY]) == (REPLY, {"message": fp.REPLIES["not_found"]})
    assert call(resolution_prompt("Exchange my scarf for something.", lines=one), [REPLY]) == (REPLY, {"message": fp.REPLIES["needs_info"]})
    assert call(resolution_prompt("Exchange my scarf for a small or a large.", lines=one), [HANDOFF, REPLY])[0] == REPLY


def test_resolution_executes_only_with_the_confirm_control() -> None:
    proposal = {"proposal_id": "P-0007"}
    messages = resolution_prompt("Yes go ahead.", confirmation=confirmation_granted(proposal))
    assert call(messages, [HANDOFF, REPLY]) == (HANDOFF, {"intent": "execute_confirmed"})
    pending = messages + [{"role": "assistant", "content": "", "tool_name": REPLY, "tool_args": {"message": "ok"}},
                          tool_message(REPLY, {"error": {"code": "CONFIRMATION_PENDING", "message": "hand off first"}})]
    view = fp.read_resolution(pending)
    view.confirmed = False
    assert fp.resolution_call(view, [HANDOFF, REPLY]) == (HANDOFF, {"intent": "execute_confirmed"})


def test_resolution_follows_a_veto_and_answers_a_gate_rejection_with_a_question() -> None:
    base = resolution_prompt("Refund line LN-3 of ORD-7Q.")
    veto = {"intent": "propose_refund", "order_id": "ORD-7Q", "line_id": "LN-3"}
    vetoed = base + [{"role": "assistant", "content": "", "tool_name": REPLY, "tool_args": {"message": "Which one?"}},
                     tool_message(REPLY, {"error": {"code": "HANDOFF_REQUIRED", "message": "hand off", "handoff": veto}})]
    assert call(vetoed, [HANDOFF, REPLY]) == (HANDOFF, veto)
    rejected = base + [{"role": "assistant", "content": "", "tool_name": HANDOFF, "tool_args": veto},
                       tool_message(HANDOFF, {"error": {"code": "TARGET_MISMATCH", "message": "no"}})]
    assert call(rejected, [HANDOFF, REPLY])[0] == REPLY


def test_resolution_looks_lines_up_when_no_card_is_loaded() -> None:
    messages = resolution_prompt("Refund line LN-3 of ORD-7Q.", lines=None)
    assert call(messages, [FIND_ORDER_LINES, HANDOFF, REPLY]) == (FIND_ORDER_LINES, {"order_ref": "ALL"})
    looked_up = messages + [{"role": "assistant", "content": "", "tool_name": FIND_ORDER_LINES, "tool_args": {"order_ref": "ALL"}},
                            tool_message(FIND_ORDER_LINES, {"ok": True, "data": {"matches": LINES, "count": 2}})]
    assert call(looked_up, [FIND_ORDER_LINES, HANDOFF, REPLY])[1]["line_id"] == "LN-3"
    cached = messages + [tool_message(FIND_ORDER_LINES, {"already_done": True, "result": {"data": {"matches": LINES}}})]
    assert fp.read_resolution(cached).lines == LINES


def test_resolution_replies_from_the_report_and_the_desk_notes() -> None:
    base = resolution_prompt("Refund line LN-3 of ORD-7Q.")
    report = {"outcome": "proposed", "proposal": {"proposal_id": "P-1"}}
    via_tool = base + [tool_message(HANDOFF, report)]
    assert call(via_tool, [HANDOFF, REPLY]) == (REPLY, {"message": fp.REPLIES["proposed"]})
    replay = base + [tool_message(HANDOFF, {"outcome": "completed", "already_processed_earlier": True})]
    assert call(replay, [REPLY]) == (REPLY, {"message": fp.REPLIES["replay"]})
    denied = base + [tool_message(HANDOFF, {"outcome": "denied"})]
    assert call(denied, [REPLY]) == (REPLY, {"message": fp.REPLIES["denied"]})
    assert call(base + [tool_message(HANDOFF, {"outcome": "weird"})], [REPLY]) == (REPLY, {"message": fp.REPLIES["failed"]})
    unrelated = base + [{"role": "user", "content": "DESK: the desk called something for you"}]
    assert fp.read_resolution(unrelated).report is None
    for desk_note, key in ((REPLAY_NOTE, "replay"), (CANCEL_NOTE, "cancel"), (ALREADY_COMPLETED_NOTE, "already_completed")):
        assert call(resolution_prompt("ok", note=desk_note), [REPLY]) == (REPLY, {"message": fp.REPLIES[key]})


def test_grounded_handoff_names_another_product_when_asked() -> None:
    lines = LINES[:1]
    catalog_lines = lines + [{"order_id": "ORD-8", "line_id": "LN-9", "sku": "CAP-S", "product_id": "CAP", "size": "S", "aliases": ["cap"]}]
    analysis = fp.analyze(["Exchange my scarf from ORD-7Q for the cap in size S."], catalog_lines, [])
    handoff = fp.grounded_handoff(analysis, catalog_lines)
    assert handoff is not None and handoff["requested_product"] == "CAP"


def ops_prompt(intent: str, *events: tuple[str, dict[str, Any]], desk_called: tuple[str, dict] | None = None) -> list[dict[str, Any]]:
    messages = [{"role": "system", "content": "ops"},
                {"role": "user", "content": ops_brief(intent, LINES[0], "L", None, None, ["x"])}]
    for tool, payload in events:
        messages.append({"role": "assistant", "content": "", "tool_name": tool, "tool_args": {}})
        messages.append(tool_message(tool, payload))
    if desk_called:
        messages.append({"role": "user", "content": f"DESK: the desk called {desk_called[0]} for you: " + json.dumps(desk_called[1])})
    return messages


def ops(messages: list[dict[str, Any]], allowed: list[str], free_text: bool = False) -> tuple[str, dict[str, Any]]:
    return fp.operations_call(fp.read_operations(messages), allowed, free_text)


ALL_OPS = [LIST_EXCHANGE_OPTIONS, CREATE_PROPOSAL, EXECUTE_PROPOSAL, GET_REQUEST_STATUS, REPORT]
PROPOSAL_OK = {"ok": True, "data": {"proposal": {"proposal_id": "P-1"}}, "error": None}


def test_operations_follows_the_documented_tool_order_for_an_exchange() -> None:
    assert ops(ops_prompt("propose_exchange"), ALL_OPS) == (LIST_EXCHANGE_OPTIONS, {})
    options = ("list_exchange_options", {"ok": True, "data": {"requested_sku": "SCARF-L"}, "error": None})
    assert ops(ops_prompt("propose_exchange", options), ALL_OPS) == (CREATE_PROPOSAL, {"remedy": "exchange", "replacement_sku": "SCARF-L"})
    no_size = ("list_exchange_options", {"ok": True, "data": {"requested_sku": None}, "error": None})
    assert ops(ops_prompt("propose_exchange", no_size), ALL_OPS) == (CREATE_PROPOSAL, {"remedy": "exchange"})
    done = ops_prompt("propose_exchange", options, ("create_proposal", PROPOSAL_OK))
    assert ops(done, ALL_OPS) == (REPORT, {"outcome": "proposed"})
    assert ops(done, ALL_OPS, free_text=True) == (REPORT, {"outcome": "proposed", "summary": fp.SUMMARIES["proposed"]})


def test_operations_reconciles_a_timeout_before_reporting() -> None:
    assert ops(ops_prompt("execute_confirmed"), ALL_OPS) == (EXECUTE_PROPOSAL, {})
    timeout = ("execute_proposal", {"ok": False, "data": None, "error": {"code": "TIMEOUT"}})
    assert ops(ops_prompt("execute_confirmed", timeout), ALL_OPS) == (GET_REQUEST_STATUS, {})
    assert fp.observed_outcome(fp.read_operations(ops_prompt("execute_confirmed", timeout))) == "failed"
    status = {"ok": True, "data": {"status": "completed", "record": {}}, "error": None}
    reconciled = ops_prompt("execute_confirmed", timeout, desk_called=("get_request_status", status))
    assert ops(reconciled, ALL_OPS) == (REPORT, {"outcome": "completed"})
    assert ops(ops_prompt("check_status"), ALL_OPS) == (GET_REQUEST_STATUS, {})
    not_found = ("get_request_status", {"ok": True, "data": {"status": "not_found"}, "error": None})
    assert ops(ops_prompt("check_status", not_found), ALL_OPS) == (REPORT, {"outcome": "needs_info"})


def test_operations_reports_what_the_results_show() -> None:
    def outcome(*events: tuple[str, dict[str, Any]]) -> str:
        return fp.observed_outcome(fp.read_operations(ops_prompt("propose_refund", *events)))

    def error(code: str) -> dict[str, Any]:
        return {"ok": False, "data": None, "error": {"code": code}}

    assert outcome(("create_proposal", error("RETURN_WINDOW_EXPIRED"))) == "denied"
    assert outcome(("create_proposal", error("SIZE_UNSPECIFIED"))) == "needs_info"
    assert outcome(("create_proposal", error("INTERNAL"))) == "failed"
    assert outcome(("create_proposal", PROPOSAL_OK), ("create_proposal", error("INVALID_ARGS"))) == "proposed"
    assert outcome(("list_exchange_options", error("REQUEST_ALREADY_COMPLETED"))) == "needs_info"
    assert outcome(("execute_proposal", {"ok": True, "data": {"transaction": {}}, "error": None})) == "completed"
    assert outcome(("create_proposal", {"ok": False, "error": "not a dict"})) == "needs_info"
    repeat = ops_prompt("propose_refund") + [tool_message(CREATE_PROPOSAL, {"already_done": True, "result": PROPOSAL_OK})]
    assert fp.read_operations(repeat).events == []


def test_operations_falls_back_to_an_allowed_tool() -> None:
    assert ops(ops_prompt("propose_refund"), [REPORT]) == (REPORT, {"outcome": "needs_info"})
    assert ops(ops_prompt("propose_refund"), [GET_REQUEST_STATUS]) == (GET_REQUEST_STATUS, {})


def test_step_validates_like_a_real_answer_and_records_size_estimates() -> None:
    llm = FakePolicyLLM(step_delay_s=0.001)
    messages = resolution_prompt("Please refund line LN-3 of ORD-7Q.")
    started = time.perf_counter()
    step = llm.step("resolution", messages, RESOLUTION_TOOLS, [HANDOFF, REPLY], 160)
    assert time.perf_counter() - started >= 0.001
    assert step.ok and step.tool == HANDOFF and step.role == "resolution" and llm.calls == 1
    assert step.prompt_tokens > 100 and step.completion_tokens >= 1 and step.predicted_ms == 0 and step.latency_ms == 1.0
    ops_step = llm.step("operations", ops_prompt("propose_refund"), OPERATIONS_TOOLS_FREE_TEXT, [CREATE_PROPOSAL, REPORT], 160)
    assert ops_step.ok and ops_step.tool == CREATE_PROPOSAL
    report = llm.step("operations", ops_prompt("propose_refund", ("create_proposal", PROPOSAL_OK)), OPERATIONS_TOOLS_FREE_TEXT, [REPORT], 160)
    assert report.args == {"outcome": "proposed", "summary": fp.SUMMARIES["proposed"]}
    structured = llm.step("operations", ops_prompt("propose_refund", ("create_proposal", PROPOSAL_OK)), OPERATIONS_TOOLS_STRUCTURED, [REPORT], 160)
    assert structured.args == {"outcome": "proposed"}


def test_json_helper_never_raises() -> None:
    assert fp._json("not json") == {} and fp._json("[1, 2]") == {} and fp._json('{"a": 1}') == {"a": 1}


def test_scenario_state_is_never_mutated_by_sessions(client: TestClient) -> None:
    before = copy.deepcopy(client.get("/api/scenarios").json())
    run_over_http(client, case_by_suffix("exchange"))
    assert client.get("/api/scenarios").json() == before


# ================================================================================================
# SEC-1: an approval binds to what the page displayed


def confirm(client: TestClient, session_id: str, card: dict[str, Any], generation: Any, message: str = "Yes, I confirm this proposal."):
    return client.post(f"/api/sessions/{session_id}/turns", json={
        "message": message, "confirm_proposal_id": card["proposal_id"], "confirm_binding_hash": card["binding_hash"],
        "generation": generation})


def test_a_confirmation_carrying_what_the_card_showed_executes(client: TestClient) -> None:
    case = case_by_suffix("refund_boundary")
    session_id = create(client, case_id=case["case_id"])["session_id"]
    shown = turn_event(post_turn(client, session_id, case["turns"][0]["message"]))
    card = next(iter(shown["state"]["cards"].values()))
    response = confirm(client, session_id, card, shown["state"]["generation"])
    assert response.status_code == 200
    assert turn_event(parse_sse(response.text))["output"]["decision"] == "completed"


def test_a_stale_tab_after_a_reset_gets_409_and_nothing_moves(client: TestClient) -> None:
    """Tab B shows 'Confirm refund $30.00' for L1; tab A resets and asks for L2 (same counter id P-0001)."""
    session_id = create(client, case_id=case_by_suffix("clarify")["case_id"])["session_id"]
    tab_b = turn_event(post_turn(client, session_id, "Refund line L1 of order O100 please."))
    card = next(iter(tab_b["state"]["cards"].values()))
    assert client.post(f"/api/sessions/{session_id}/reset", json={}).status_code == 200
    tab_a = turn_event(post_turn(client, session_id, "Refund line L2 of order O100 please."))
    assert tab_a["output"]["proposal"]["proposal_id"] == card["proposal_id"]
    response = confirm(client, session_id, card, tab_b["state"]["generation"])  # the hash and generation tab B showed
    assert response.status_code == 409 and "proposal changed" in response.json()["detail"]
    response = confirm(client, session_id, card, tab_a["state"]["generation"])  # a current generation, the old hash
    assert response.status_code == 409 and "proposal changed" in response.json()["detail"]
    state = client.get(f"/api/sessions/{session_id}/state").json()
    assert state["final_state"]["refunds"] == [] and state["busy"] is False and len(state["turns"]) == 1


def test_a_hash_for_an_unknown_proposal_is_409(client: TestClient) -> None:
    session_id = create(client, case_id=CASES[0]["case_id"])["session_id"]
    response = confirm(client, session_id, {"proposal_id": "P-0001", "binding_hash": "f" * 64}, 1)
    assert response.status_code == 409
    assert post_turn(client, session_id, "Refund line L1 of my order O100.")  # the lock was released


@pytest.mark.parametrize("missing", ["confirm_binding_hash", "generation"])
def test_a_confirmation_without_what_the_page_showed_is_a_422(client: TestClient, missing: str) -> None:
    """The proposal id alone proves nothing (counters restart after a reset): both fields are required."""
    session_id = create(client, case_id=CASES[0]["case_id"])["session_id"]
    shown = turn_event(post_turn(client, session_id, "Refund line L1 of my order O100."))
    card = next(iter(shown["state"]["cards"].values()))
    body = {"message": "Yes, I confirm this proposal.", "confirm_proposal_id": card["proposal_id"],
            "confirm_binding_hash": card["binding_hash"], "generation": shown["state"]["generation"]}
    del body[missing]
    response = client.post(f"/api/sessions/{session_id}/turns", json=body)
    assert response.status_code == 422 and "confirm_binding_hash and generation" in response.json()["detail"]
    state = client.get(f"/api/sessions/{session_id}/state").json()
    assert len(state["turns"]) == 1 and state["final_state"]["refunds"] == [] and state["busy"] is False


def test_a_generation_must_be_an_integer(client: TestClient) -> None:
    session_id = create(client, case_id=CASES[0]["case_id"])["session_id"]
    response = client.post(f"/api/sessions/{session_id}/turns", json={"message": "yes", "confirm_proposal_id": "P-0001",
                                                                      "generation": "1"})
    assert response.status_code == 422 and response.json()["detail"][0]["loc"] == ["body", "generation"]


# ================================================================================================
# SEC-4: local-only requests


@pytest.mark.parametrize("host", ["attacker.example", "attacker.example:8000", "127.0.0.1.attacker.example", "[::1"])
def test_a_foreign_host_header_is_refused(client: TestClient, host: str) -> None:
    response = client.get("/api/scenarios", headers={"Host": host})
    assert response.status_code == 400 and "127.0.0.1 or localhost" in response.json()["detail"]


@pytest.mark.parametrize("host", ["127.0.0.1:8000", "localhost:8000", "LOCALHOST", "[::1]:8000", "127.0.0.1"])
def test_local_host_names_are_served(client: TestClient, host: str) -> None:
    assert client.get("/api/scenarios", headers={"Host": host}).status_code == 200


def test_with_a_known_port_the_host_header_must_name_it(tmp_path: Path) -> None:
    client = local_client(make_app(tmp_path, port=8000))
    assert client.get("/api/scenarios").status_code == 200
    assert client.get("/api/scenarios", headers={"Host": "127.0.0.1"}).status_code == 200  # no port: default
    assert client.get("/api/scenarios", headers={"Host": "127.0.0.1:9999"}).status_code == 400


@pytest.mark.parametrize("headers", [{"Origin": "https://evil.example"}, {"Origin": "null"},
                                     {"Origin": "http://localhost:8000"},  # another origin than 127.0.0.1:8000
                                     {"Referer": "https://evil.example/page"}])
def test_a_cross_site_post_is_refused_before_it_changes_anything(client: TestClient, headers: dict[str, str]) -> None:
    session_id = create(client, case_id=CASES[0]["case_id"])["session_id"]
    post_turn(client, session_id, "Refund line L1 of my order O100.")
    response = client.post(f"/api/sessions/{session_id}/reset", headers=headers)
    assert response.status_code == 403 and "cross-site" in response.json()["detail"]
    assert len(client.get(f"/api/sessions/{session_id}/state").json()["turns"]) == 1


def test_same_origin_posts_and_cross_site_reads_are_allowed(client: TestClient) -> None:
    session_id = create(client, case_id=CASES[0]["case_id"])["session_id"]
    same = {"Origin": LOCAL, "Referer": f"{LOCAL}/"}
    assert client.post(f"/api/sessions/{session_id}/reset", json={}, headers=same).status_code == 200
    read = client.get("/api/scenarios", headers={"Origin": "https://evil.example"})
    assert read.status_code == 200 and "access-control-allow-origin" not in read.headers


def test_request_bodies_are_capped(client: TestClient) -> None:
    big = {"initial_state": {"pad": "x" * server_app.MAX_BODY_BYTES}, "customer_id": "C1"}
    response = client.post("/api/sessions", json=big)
    assert response.status_code == 413 and "MiB" in response.json()["detail"]


def test_a_chunked_body_without_length_and_a_bad_length_are_refused() -> None:
    config = ServerConfig()
    chunked = server_app.Headers(raw=[(b"host", b"127.0.0.1"), (b"transfer-encoding", b"chunked")])
    assert server_app.request_refusal("POST", chunked, config) == (411, "send a Content-Length with the request body")
    bad = server_app.Headers(raw=[(b"host", b"127.0.0.1"), (b"content-length", b"12x")])
    assert server_app.request_refusal("POST", bad, config)[0] == 400
    bodiless = server_app.Headers(raw=[(b"host", b"localhost")])
    assert server_app.request_refusal("POST", bodiless, config) is None


def test_netloc_parsing_never_raises() -> None:
    assert server_app._netloc("http://127.0.0.1:8000/x") == ("127.0.0.1", 8000)
    assert server_app._netloc("[::1]:8000") == ("::1", 8000)
    assert server_app._netloc("127.0.0.1:notaport") == (None, None)
    assert server_app._netloc("") == (None, None)


# ================================================================================================
# SEC-5: malformed input is a readable 422, never a 500


def test_a_lone_surrogate_message_is_a_readable_422(client: TestClient) -> None:
    session_id = create(client, case_id=CASES[0]["case_id"])["session_id"]
    response = client.post(f"/api/sessions/{session_id}/turns", content=b'{"message": "Refund L1 \\ud800"}',
                           headers={"content-type": "application/json"})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail[0]["loc"] == ["body", "message"] and "unicode" in detail[0]["msg"] and "input" not in detail[0]


@pytest.mark.parametrize("content", [b"{not json", b'{"message": 5}', b"[]"])
def test_malformed_turn_bodies_are_422(client: TestClient, content: bytes) -> None:
    session_id = create(client, case_id=CASES[0]["case_id"])["session_id"]
    response = client.post(f"/api/sessions/{session_id}/turns", content=content, headers={"content-type": "application/json"})
    assert response.status_code == 422 and isinstance(response.json()["detail"], list)


def test_error_bodies_encode_any_text(client: TestClient) -> None:
    response = client.get("/api/sessions/S-\u00e9\u0301/state")
    assert response.status_code == 404 and response.json()["detail"].startswith("no session")
    assert response.headers["content-type"] == "application/json"


def test_http_error_headers_are_kept(client: TestClient) -> None:
    response = client.put("/api/scenarios")
    assert response.status_code == 405 and response.headers["allow"] == "GET"


def test_readable_validation_errors_tolerate_sparse_entries() -> None:
    assert server_app.readable_validation_errors([{}]) == [{"loc": [], "msg": "invalid", "type": ""}]


def test_lifespan_events_pass_the_guard(app) -> None:
    with local_client(app) as client:  # the context manager runs startup/shutdown (scope type "lifespan")
        assert client.get("/api/meta").status_code == 200
