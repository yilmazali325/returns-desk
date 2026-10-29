"""End-to-end tests of desk.session: every public case through run_case with scripted model steps,
graded by the independent scorer (evals/score.py) against the hand-written public labels, plus the
crash, timeout and metrics paths. No live model: FakeLLM replays the steps through the real
validation code, so the orchestrator, gateway, grounding and render code all run for real.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from desk.domain import ERROR
from desk.llm import FakeLLM, LLMError
from desk.profiles import get_profile
from desk.session import CaseSession, build_metrics, not_run_result, run_case
from desk.agents import Telemetry
from evals.score import score_case

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = json.loads((ROOT / "candidate_assets" / "public_cases.json").read_text())["cases"]
LABELS = {label["case_id"]: label for label in json.loads((ROOT / "evals" / "expected" / "public_expected.json").read_text())["cases"]}
F, B = get_profile("F"), get_profile("B")


def H(intent: str, **args: Any) -> tuple[str, dict[str, Any]]:
    return ("handoff_to_operations", {"intent": intent, **args})


def R(message: str) -> tuple[str, dict[str, Any]]:
    return ("reply_to_customer", {"message": message})


def REPORT(outcome: str, summary: str | None = None) -> tuple[str, dict[str, Any]]:
    args = {"outcome": outcome}
    if summary is not None:
        args["summary"] = summary
    return ("report_to_resolution", args)


CREATE_REFUND = ("create_proposal", {"remedy": "refund"})
EXECUTE = ("execute_proposal", {})
EXECUTE_TURN = [H("execute_confirmed"), R("Thank you, it is all taken care of.")]

# What a well-behaved model does on each public case (IDs come from the case itself, in the test only).
GOOD_SCRIPTS: dict[str, dict[str, list[Any]]] = {
    "public_01_refund_boundary": {
        "resolution": [H("propose_refund", order_id="O100", line_id="L1"), R("Here is the refund I can offer."),
                       *EXECUTE_TURN],
        "operations": [CREATE_REFUND, REPORT("proposed"), EXECUTE, REPORT("completed")],
    },
    "public_02_exchange": {
        "resolution": [H("propose_exchange", order_id="O100", line_id="L1", requested_size="L", requested_product="SAME"),
                       R("Here is the exchange I can offer."), *EXECUTE_TURN],
        "operations": [("list_exchange_options", {}), ("create_proposal", {"remedy": "exchange", "replacement_sku": "TEE-L"}),
                       REPORT("proposed"), EXECUTE, REPORT("completed")],
    },
    "public_03_expired": {
        "resolution": [H("propose_refund", order_id="O100", line_id="L1"), R("I'm sorry, this one cannot be refunded.")],
        "operations": [CREATE_REFUND, REPORT("denied")],
    },
    "public_04_clarify": {
        "resolution": [R("Which item from that order would you like to return?"),
                       H("propose_refund", order_id="O100", line_id="L1"), R("Here is the refund I can offer."),
                       *EXECUTE_TURN],
        "operations": [CREATE_REFUND, REPORT("proposed"), EXECUTE, REPORT("completed")],
    },
    "public_05_stale_stock": {
        "resolution": [H("propose_exchange", order_id="O100", line_id="L1", requested_size="L", requested_product="SAME"),
                       R("Here is the exchange I can offer."), H("execute_confirmed"),
                       R("I'm sorry, that size just sold out, so nothing was changed.")],
        "operations": [("list_exchange_options", {}), ("create_proposal", {"remedy": "exchange", "replacement_sku": "TEE-L"}),
                       REPORT("proposed"), EXECUTE, REPORT("denied")],
    },
    "public_06_timeout_replay": {
        "resolution": [H("propose_refund", order_id="O100", line_id="L1"), R("Here is the refund I can offer."),
                       H("execute_confirmed"), R("Thanks for waiting, your refund went through."),
                       R("Good news, this was already taken care of.")],
        "operations": [CREATE_REFUND, REPORT("proposed"), EXECUTE, ("get_request_status", {}), REPORT("completed")],
    },
}


def case_by_id(case_id: str) -> dict[str, Any]:
    return copy.deepcopy(next(c for c in PUBLIC if c["case_id"] == case_id))


@pytest.mark.parametrize("case_id", sorted(GOOD_SCRIPTS))
def test_public_case_happy_path_scores_strictly(case_id: str) -> None:
    case = case_by_id(case_id)
    llm = FakeLLM(copy.deepcopy(GOOD_SCRIPTS[case_id]))
    result = run_case(case, llm, F)
    assert llm.remaining == 0, "every scripted step was used"
    assert result["metrics"]["status"] == "ok", result["metrics"]
    score = score_case(case, LABELS[case_id], result)
    assert score.strict, score.diff_lines()
    # well-behaved model: no code override was needed
    assert result["metrics"]["guard_overrides"] == 0, [e for e in result["action_audit"] if e["kind"] == "guard"]
    assert result["metrics"]["fallbacks"] == 0


def test_public_06_reconciles_the_timeout_and_replays_without_effects() -> None:
    case = case_by_id("public_06_timeout_replay")
    result = run_case(case, FakeLLM(copy.deepcopy(GOOD_SCRIPTS[case["case_id"]])), F)
    t1, t2, t3 = result["turns"]
    assert t2["transaction"]["transaction_id"] == t3["transaction"]["transaction_id"]
    assert "timed out" in t2["response"] and "already processed" in t3["response"]
    executes = [e for e in result["action_audit"] if e["kind"] == "tool" and e["tool"] == "execute_proposal"]
    assert len(executes) == 1 and executes[0]["visible_to_agent"] is False and executes[0]["committed_effects"]
    assert len(result["final_state"]["refunds"]) == 1


def test_baseline_profile_looks_lines_up_and_relays_the_ops_summary() -> None:
    case = case_by_id("public_01_refund_boundary")
    llm = FakeLLM({
        "resolution": [("find_order_lines", {"order_ref": "O100"}), H("propose_refund", order_id="O100", line_id="L1"),
                       R("Here is the refund I can offer."), *EXECUTE_TURN],
        "operations": [CREATE_REFUND, REPORT("proposed", "Proposal ready."), EXECUTE, REPORT("completed", "Refunded.")],
    })
    result = run_case(case, llm, B)
    assert score_case(case, LABELS[case["case_id"]], result).strict
    resolution_requests = [r for r in llm.requests if r.role == "resolution"]
    assert "find_order_lines" in resolution_requests[0].allowed
    assert "not loaded" in resolution_requests[0].messages[1]["content"]
    handoff_result = resolution_requests[2].messages[-1]["content"]
    assert json.loads(handoff_result) == {"outcome": "proposed", "summary": "Proposal ready.", "next": "reply_to_customer"}


def test_crash_mid_case_still_emits_every_turn() -> None:
    case = case_by_id("public_04_clarify")
    llm = FakeLLM({"resolution": [R("Which item do you mean?")], "operations": []})  # script runs out on turn 2
    result = run_case(case, llm, F)
    assert [t["decision"] for t in result["turns"]] == ["clarify", ERROR, ERROR]
    assert result["turns"][1]["response"].startswith("not run: FakeLLMExhausted")
    assert result["metrics"]["status"] == "failed" and "FakeLLMExhausted" in result["metrics"]["error"]
    assert result["final_state"] == case["initial_state"]


def test_a_crash_after_a_persisted_commit_names_the_transaction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Contract review C7: money moved, so the crashed turn must not claim it did not run."""
    from desk.tools import BusinessTools
    original = BusinessTools.execute_proposal

    def commit_then_crash(self: BusinessTools, *args: Any, **kwargs: Any) -> Any:
        original(self, *args, **kwargs)
        raise RuntimeError("simulated crash after commit")

    monkeypatch.setattr(BusinessTools, "execute_proposal", commit_then_crash)
    case = case_by_id("public_06_timeout_replay")
    case["faults"] = []
    result = run_case(case, FakeLLM(copy.deepcopy(GOOD_SCRIPTS["public_01_refund_boundary"])), F)
    record = result["final_state"]["idempotency"][case["request_id"]]
    crashed, after = result["turns"][1], result["turns"][2]
    assert crashed["decision"] == ERROR and crashed["transaction"] == record
    assert record["transaction_id"] in crashed["response"] and not crashed["response"].startswith("not run")
    assert after["response"].startswith("not run") and result["metrics"]["status"] == "failed"


def test_a_crash_after_an_earlier_reported_commit_is_plain_not_run() -> None:
    case = case_by_id("public_06_timeout_replay")
    case["faults"] = []
    script = copy.deepcopy(GOOD_SCRIPTS["public_01_refund_boundary"])  # two turns of steps for a three-turn case
    result = run_case(case, FakeLLM(script), F)
    assert result["turns"][1]["decision"] == "completed"
    assert result["turns"][2]["response"].startswith("not run: FakeLLMExhausted")


def test_case_time_budget_stops_between_turns() -> None:
    case = case_by_id("public_01_refund_boundary")
    result = run_case(case, FakeLLM(copy.deepcopy(GOOD_SCRIPTS[case["case_id"]])), F, case_timeout_s=-1)
    assert [t["decision"] for t in result["turns"]] == [ERROR, ERROR]
    assert "time budget" in result["metrics"]["error"]


def test_llm_errors_mark_the_case_failed_but_keep_valid_turns() -> None:
    case = case_by_id("public_01_refund_boundary")
    result = run_case(case, FakeLLM([LLMError("connection", "down"), LLMError("connection", "down")]), F)
    assert [t["decision"] for t in result["turns"]] == [ERROR, ERROR]
    assert result["metrics"]["status"] == "failed" and result["metrics"]["llm_errors"] == 2
    assert result["final_state"] == case["initial_state"]


def test_broken_case_input_is_a_failed_result_not_an_exception() -> None:
    case = case_by_id("public_01_refund_boundary")
    case["initial_state"]["as_of"] = "not a date"
    result = run_case(case, FakeLLM([]), F)
    assert result["metrics"]["status"] == "failed" and len(result["turns"]) == 2
    assert result["final_state"]["as_of"] == "not a date" and result["action_audit"] == []


class FakeProbe:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def start(self) -> None:
        self.calls.append("start")

    def mark(self, label: str) -> None:
        self.calls.append(label)

    def stop(self) -> None:
        self.calls.append("stop")

    def report(self) -> dict[str, Any]:
        return {"peak_ram_mb": 12.5, "peak_vram_mb": 3.0, "peak_unified_gpu_mb": 3.0, "rss_mb": 40.0,
                "memory_architecture": "unified", "memory_notes": ["fake"]}


def test_memory_probe_brackets_every_turn() -> None:
    case = case_by_id("public_01_refund_boundary")
    probe = FakeProbe()
    result = run_case(case, FakeLLM(copy.deepcopy(GOOD_SCRIPTS[case["case_id"]])), F, probe=probe)
    assert probe.calls == ["start", "turn 1", "turn 2", "end", "stop"]
    assert result["metrics"]["peak_ram_mb"] == 12.5 and result["metrics"]["peak_vram_mb"] == 3.0


def test_metrics_are_token_weighted_and_count_audit_kinds() -> None:
    t = Telemetry(llm_calls=3, prompt_tokens=1000, cached_tokens=800, completion_tokens=50, prefill_ms=100.0,
                  decode_ms=500.0, tool_ms=1.25)
    audit = [{"kind": "guard"}, {"kind": "guard"}, {"kind": "fallback"}, {"kind": "gate"}, {"kind": "tool"}]
    metrics = build_metrics(t, [10.0], 20.0, audit, {"peak_ram_mb": None}, "ok", None)
    assert metrics["decode_tok_s"] == 100.0 and metrics["prefill_tok_s"] == 2000.0
    assert (metrics["guard_overrides"], metrics["fallbacks"], metrics["gate_rejections"]) == (2, 1, 1)
    assert "error" not in metrics and metrics["tool_ms"] == 1.2
    assert build_metrics(Telemetry(), [], 0.0, [], {}, "failed", "x")["decode_tok_s"] is None


def test_not_run_result_is_contract_shaped() -> None:
    case = case_by_id("public_06_timeout_replay")
    result = not_run_result(case, "llama-server unavailable: boom")
    assert [t["decision"] for t in result["turns"]] == [ERROR] * 3
    assert all(t["proposal"] is None and t["response"].startswith("not run:") for t in result["turns"])
    assert result["metrics"]["status"] == "failed" and result["metrics"]["elapsed_ms"] == 0.0
    assert result["metrics"]["peak_ram_mb"] is None and result["final_state"] == case["initial_state"]


def test_ui_session_accepts_an_explicit_presented_proposal_id() -> None:
    case = case_by_id("public_01_refund_boundary")
    session = CaseSession(initial_state=case["initial_state"], customer_id=case["customer_id"],
                          request_id=case["request_id"], faults=None, llm=FakeLLM(copy.deepcopy(GOOD_SCRIPTS[case["case_id"]])),
                          profile=F)
    first = session.run_turn(case["turns"][0]["message"])
    second = session.run_turn("yes", confirm=first["proposal"]["proposal_id"])
    assert second["decision"] == "completed" and second["confirmed_proposal_id"] == first["proposal"]["proposal_id"]
    snapshot = session.snapshot()
    assert set(snapshot) == {"action_audit", "final_state", "proposals", "external_events"}
    assert len(session.turns) == 2
