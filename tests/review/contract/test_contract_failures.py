"""Contract review: "Include failed runs" — the output must stay contract-valid with every case present
when the model server cannot start, dies mid-run, or a case crashes mid-way.

No llama-server is started here. The LlamaServer class is replaced by stubs at the adapter boundary,
and LLM calls go to a closed local port, which is exactly what a crashed server looks like to the
client (connection refused).

Run: .venv/bin/pytest tests/review/contract -q
Tests that documented a review defect were marked xfail; the defects are fixed, so the markers are
gone and those tests now guard the fixes.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import socket
from pathlib import Path
from typing import Any

import pytest

from desk import adapter_main, telemetry
from desk.llm import LLMClient
from desk.llm_server import LlamaServer, LlamaServerError
from desk.profiles import get_profile
from desk.session import run_case
from desk.testing.fake_policy_llm import FakePolicyLLM
from desk.tools import BusinessTools

ROOT = Path(__file__).resolve().parents[3]
CASES_PATH = ROOT / "candidate_assets" / "public_cases.json"
CASES = json.loads(CASES_PATH.read_text())["cases"]


def validator():
    spec = importlib.util.spec_from_file_location("validate_submission", ROOT / "candidate_assets" / "validate_submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dead_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(autouse=True)
def fast_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(telemetry, "measure_gpu_baseline", lambda *a, **k: None)


class NeverStarts(LlamaServer):
    """The real LlamaServer whose pre-launch verification fails (like a sha256 mismatch); nothing spawns."""

    def __init__(self, profile: Any, port: int, log: Path, **_: Any) -> None:
        super().__init__(profile, port, log, binary="/bin/sh")

    def verify(self) -> dict[str, Any]:
        raise LlamaServerError("simulated: sha256 mismatch for the model file")


def expected_state_without_effects(case: dict[str, Any]) -> dict[str, Any]:
    """Initial state with only the contract's external stock_change faults applied (no business effects)."""
    state = copy.deepcopy(case["initial_state"])
    for fault in case.get("faults", []):
        if fault.get("type") == "stock_change" and fault.get("before_turn", 0) <= len(case["turns"]):
            for product in state["products"]:
                if product["sku"] == fault["sku"]:
                    product["stock"] = fault["new_stock"]
    return state


def test_server_cannot_start_output_is_contract_valid_with_every_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter_main, "LlamaServer", NeverStarts)
    out = tmp_path / "results.json"
    code = adapter_main.main(["--input", str(CASES_PATH), "--output", str(out), "--port", str(dead_port())])
    assert code == adapter_main.EXIT_NO_SERVER  # 3: a grader harness that checks the exit code sees a failure
    assert validator().check(CASES_PATH, out) == len(CASES)
    document = json.loads(out.read_text())
    for case, result in zip(CASES, document["results"], strict=True):
        assert [t["decision"] for t in result["turns"]] == ["error"] * len(case["turns"])
        assert all(t["proposal"] is None for t in result["turns"])
        assert result["final_state"] == case["initial_state"]  # not run at all: faults never applied
        assert result["metrics"]["status"] == "failed"
        assert result["metrics"]["peak_ram_mb"] is None and result["metrics"]["peak_vram_mb"] is None


class StartsButDies(LlamaServer):
    """Reports a successful start, but its port is closed: every LLM call is refused and /health
    fails after each case (what a llama-server that crashed mid-run looks like). Nothing spawns."""

    instances = 0

    def __init__(self, profile: Any, port: int, log: Path, **_: Any) -> None:
        super().__init__(profile, dead_port(), log, binary="/bin/sh")
        StartsButDies.instances += 1

    def start(self) -> dict[str, Any]:
        return {"server_spawn_to_ready_ms": 1.0, "first_request_ms": None, "model_page_cache": "unknown"}


def test_server_dies_mid_run_every_case_present_no_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    StartsButDies.instances = 0
    monkeypatch.setattr(adapter_main, "LlamaServer", StartsButDies)
    monkeypatch.setattr(adapter_main, "LLM_STEP_TIMEOUT_S", 1.0)
    monkeypatch.setattr(telemetry, "run_conditions", lambda *a, **k: {"stub": True})
    out = tmp_path / "results.json"
    code = adapter_main.main(["--input", str(CASES_PATH), "--output", str(out), "--port", str(dead_port())])
    assert code == adapter_main.EXIT_OK
    assert validator().check(CASES_PATH, out) == len(CASES)
    document = json.loads(out.read_text())
    assert StartsButDies.instances == 1 + len(CASES)  # restarted after every unhealthy case
    for case, result in zip(CASES, document["results"], strict=True):
        assert len(result["turns"]) == len(case["turns"])
        assert {t["decision"] for t in result["turns"]} == {"error"}
        assert result["final_state"] == expected_state_without_effects(case)  # never an invented or partial effect
        assert result["metrics"]["status"] == "failed"
        assert result["metrics"]["llm_errors"] >= 1


def test_llm_refused_mid_case_never_mutates() -> None:
    profile = get_profile("F")
    with LLMClient(f"http://127.0.0.1:{dead_port()}", profile.protocol, seed=42, timeout_s=1, retries=1) as llm:
        result = run_case(copy.deepcopy(CASES[5]), llm, profile)
    assert [t["decision"] for t in result["turns"]] == ["error"] * len(CASES[5]["turns"])
    assert result["final_state"] == CASES[5]["initial_state"]
    assert result["proposals"] == []


def _crash_after_commit(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    original = BusinessTools.execute_proposal

    def commit_then_crash(self: BusinessTools, *args: Any, **kwargs: Any) -> Any:
        original(self, *args, **kwargs)
        raise RuntimeError("simulated crash after commit")

    monkeypatch.setattr(BusinessTools, "execute_proposal", commit_then_crash)
    return run_case(copy.deepcopy(CASES[0]), FakePolicyLLM(), get_profile("F"))


def test_crash_after_commit_output_is_structurally_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _crash_after_commit(monkeypatch)
    assert len(result["turns"]) == len(CASES[0]["turns"])
    assert result["turns"][1]["decision"] == "error"
    assert len(result["final_state"]["refunds"]) == 1  # the commit really persisted
    assert result["metrics"]["status"] == "failed"


def test_crash_after_commit_response_does_not_claim_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _crash_after_commit(monkeypatch)
    committed = result["final_state"]["idempotency"]
    assert committed, "precondition: the refund committed"
    response = result["turns"][1]["response"]
    transaction_id = next(iter(committed.values()))["transaction_id"]
    assert not response.startswith("not run"), response
    assert transaction_id in response, "an error turn after a persisted commit should name the committed transaction"


def test_busy_default_port_does_not_fail_every_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[int] = []

    class VerifiedNoSpawn(LlamaServer):
        """Real LlamaServer.start (port check included) with verification stubbed; never spawns."""

        def verify(self) -> dict[str, Any]:
            return {"engine_version": "stub", "engine_version_mismatch_allowed": False, "model_sha256": "stub"}

        def _wait_ready(self, spawned_at: float) -> float:  # pragma: no cover - reached only if a port is chosen
            spawned.append(self.port)
            raise LlamaServerError("stub: stop before talking to a model")

    monkeypatch.setattr(adapter_main, "LlamaServer", VerifiedNoSpawn)
    import subprocess
    real_popen = subprocess.Popen

    def guarded_popen(argv: Any, *a: Any, **k: Any) -> Any:  # never let this test spawn a real llama-server
        if isinstance(argv, (list, tuple)) and argv and "llama-server" in str(argv[0]) and "--version" not in argv:
            raise LlamaServerError("stub: a server would have been spawned on a free port")
        return real_popen(argv, *a, **k)

    monkeypatch.setattr(subprocess, "Popen", guarded_popen)
    with socket.socket() as squatter:  # something already listens on the adapter's port
        squatter.bind(("127.0.0.1", 0))
        squatter.listen()
        port = squatter.getsockname()[1]
        out = tmp_path / "results.json"
        code = adapter_main.main(["--input", str(CASES_PATH), "--output", str(out), "--port", str(port)])
    document = json.loads(out.read_text())
    startup_error = document["startup"].get("error", "")
    assert "already in use" not in startup_error, (code, startup_error)
