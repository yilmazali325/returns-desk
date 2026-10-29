"""Contract review: field-level checks of real-model outputs that already exist on disk (no model run).

Inputs: var/verify/F.json and results/smoke/public/{B,F}/run-1/results.json (real Qwen3.5-4B runs).
Covers CONTRACT.md Output: decision vocabulary, proposal object on propose turns, audit entries
(agent, tool, arguments, result/error, committed effects, no chain of thought), final_state = input
schema plus records, record identifiers, and the memory metrics.

Tests that documented a review defect were marked xfail; the defects are fixed, so the markers are
gone and those tests now guard the fixes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from desk import telemetry

ROOT = Path(__file__).resolve().parents[3]
CASES = json.loads((ROOT / "candidate_assets" / "public_cases.json").read_text())["cases"]
OUTPUTS = [p for p in [ROOT / "var" / "verify" / "F.json",
                       ROOT / "results" / "smoke" / "public" / "F" / "run-1" / "results.json",
                       ROOT / "results" / "smoke" / "public" / "B" / "run-1" / "results.json"] if p.exists()]
DECISIONS = {"propose", "clarify", "deny", "completed", "error"}
BINDING = ("proposal_id", "request_id", "customer_id", "order_id", "line_id", "remedy", "amount_cents", "replacement_sku")
RECORD_IDS = ("customer_id", "order_id", "line_id", "request_id", "proposal_id", "replacement_sku")
AUDIT_KEYS = ("agent", "tool", "arguments", "result", "error", "committed_effects")
COT_MARKERS = ("reasoning_content", "<think", "chain of thought", "thinking")

pytestmark = pytest.mark.skipif(not OUTPUTS, reason="no real-model outputs on disk")


def documents() -> list[tuple[str, dict[str, Any]]]:
    return [(str(p.relative_to(ROOT)), json.loads(p.read_text())) for p in OUTPUTS]


@pytest.mark.parametrize("name,doc", documents(), ids=lambda v: v if isinstance(v, str) else "")
def test_turns_decisions_and_proposals(name: str, doc: dict[str, Any]) -> None:
    for case, result in zip(CASES, doc["results"], strict=True):
        assert len(result["turns"]) == len(case["turns"])
        for turn in result["turns"]:
            assert turn["decision"] in DECISIONS
            assert isinstance(turn["response"], str) and turn["response"]
            if turn["decision"] == "propose":
                assert isinstance(turn["proposal"], dict)
                missing = [k for k in BINDING if k not in turn["proposal"]]
                assert not missing, f"{name} {case['case_id']}: proposal lacks {missing}"
                assert turn["proposal"]["customer_id"] == case["customer_id"]
                assert turn["proposal"]["request_id"] == case["request_id"]
            else:
                assert turn["proposal"] is None


@pytest.mark.parametrize("name,doc", documents(), ids=lambda v: v if isinstance(v, str) else "")
def test_audit_entries_identify_agent_tool_args_result_effects_without_cot(name: str, doc: dict[str, Any]) -> None:
    for result in doc["results"]:
        assert result["action_audit"], f"{result['case_id']}: empty audit"
        for entry in result["action_audit"]:
            missing = [k for k in AUDIT_KEYS if k not in entry]
            assert not missing, f"{name} {result['case_id']} #{entry.get('seq')}: missing {missing}"
            if entry["kind"] == "llm_step":
                # only the chosen tool call and timings; no free model text outside tool arguments
                assert set(entry["result"]) <= {"role", "tool", "protocol_error", "dropped_calls", "prompt_tokens",
                                                "cached_tokens", "prefill_tokens", "completion_tokens", "prompt_ms",
                                                "predicted_ms", "latency_ms", "finish_reason", "http_attempts",
                                                "raw_tool", "error"}, sorted(entry["result"])
        text = json.dumps(result["action_audit"]).lower()
        assert not [m for m in COT_MARKERS if m in text]
        committed = [e for e in result["action_audit"] if e["committed_effects"]]
        if result["final_state"]["idempotency"]:
            assert committed, f"{result['case_id']}: state changed but no audit entry carries committed effects"


@pytest.mark.parametrize("name,doc", documents(), ids=lambda v: v if isinstance(v, str) else "")
def test_final_state_is_input_schema_plus_records(name: str, doc: dict[str, Any]) -> None:
    for case, result in zip(CASES, doc["results"], strict=True):
        state = result["final_state"]
        assert list(state) == list(case["initial_state"])
        for table in ("returns", "refunds", "exchanges", "quarantine"):
            for record in state[table]:
                missing = [k for k in RECORD_IDS if k not in record and not (table == "exchanges" and k == "replacement_sku")]
                assert not missing, f"{case['case_id']} {table}: {missing}"
        for request_id, record in state["idempotency"].items():
            assert request_id == case["request_id"] and record["status"] == "completed"
            assert all(k in record for k in RECORD_IDS)


@pytest.mark.parametrize("name,doc", documents(), ids=lambda v: v if isinstance(v, str) else "")
def test_metrics_memory_and_limitations(name: str, doc: dict[str, Any]) -> None:
    for result in doc["results"]:
        m = result["metrics"]
        assert isinstance(m["elapsed_ms"], (int, float)) and m["elapsed_ms"] > 0
        for key in ("peak_ram_mb", "peak_vram_mb"):
            assert m[key] is None or (isinstance(m[key], (int, float)) and m[key] >= 0)
        assert m["memory_notes"], "limitations of the memory readings must be explained"


@pytest.mark.parametrize("name,doc", documents(), ids=lambda v: v if isinstance(v, str) else "")
def test_root_declarations(name: str, doc: dict[str, Any]) -> None:
    hw, cfg = doc["hardware"], doc["configuration"]
    for key in ("cpu", "gpu", "gpu_count", "ram_gb", "os", "driver", "power"):
        assert hw.get(key) not in (None, ""), key
    for key in ("model_revision", "model_sha256", "quantization", "engine", "engine_version", "seed",
                "concurrency", "caches", "timeouts"):
        assert cfg.get(key) not in (None, ""), key


def test_hardware_power_reflects_the_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    real_run = telemetry._run

    def on_battery(argv: list[str], *a: Any, **k: Any) -> str | None:
        if argv[:3] == ["pmset", "-g", "batt"]:
            return "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t54%; discharging;"
        return real_run(argv, *a, **k)

    monkeypatch.setattr(telemetry, "_run", on_battery)
    telemetry.static_hardware.cache_clear()
    try:
        assert telemetry.run_conditions()["on_ac_power"] is False  # the live reading sees the battery
        assert "AC power" not in str(telemetry.static_hardware()["power"])
    finally:
        telemetry.static_hardware.cache_clear()
