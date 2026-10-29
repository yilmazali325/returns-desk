"""Sessions: one customer conversation over one business state, and `run_case` for contract cases.

The adapter, the bench and the UI server all build the same `CaseSession` (store + gateway + desk),
so every entry point exercises the same agents, model and business tools (CONTRACT: "The adapter
must invoke the same agents, local models and business tools as the UI").

`run_case` always returns a contract-shaped result with exactly one turn output per input turn.
A crash mid-case is reported, not raised: metrics.status is "failed", the crashed turn becomes
decision "error" and later turns "not run: <reason>". If a commit had already persisted, the
crashed turn says so and names the transaction instead of claiming nothing ran (final_state shows it).
"""

from __future__ import annotations

import copy
import time
from typing import Any

from desk.agents import Telemetry
from desk.audit import AuditLog
from desk.domain import ERROR, TurnOutput
from desk.faults import FaultPlan
from desk.grounding import product_aliases
from desk.orchestrator import Desk
from desk.profiles import Profile
from desk.store import Store
from desk.tools import BusinessTools

CASE_TIMEOUT_S = 600.0  # checked between turns; a single turn is bounded by its step caps
NO_PROBE_NOTES = ["peak_ram_mb / peak_vram_mb are null: no memory probe was attached to this run."]


class CaseSession:
    """A conversation: store, fault plan, audit, gateway and orchestrator for one customer."""

    def __init__(self, *, initial_state: dict[str, Any], customer_id: str, request_id: str,
                 faults: list[Any] | None, llm: Any, profile: Profile, turn_count: int | None = None,
                 store_path: str = ":memory:") -> None:
        self.store = Store(initial_state, store_path)
        self.audit = AuditLog()
        self.tools = BusinessTools(self.store, customer_id, request_id, FaultPlan(faults or [], turn_count),
                                   self.audit, alias_fn=product_aliases)
        self.desk = Desk(self.tools, llm, profile)
        self.turns: list[dict[str, Any]] = []

    @classmethod
    def from_case(cls, case: dict[str, Any], llm: Any, profile: Profile, store_path: str = ":memory:") -> CaseSession:
        return cls(initial_state=case["initial_state"], customer_id=case["customer_id"],
                   request_id=case["request_id"], faults=case.get("faults"), llm=llm, profile=profile,
                   turn_count=len(case["turns"]), store_path=store_path)

    def run_turn(self, message: str, confirm: str | None = None) -> dict[str, Any]:
        """One customer message. `confirm` is "last" (contract) or an explicit proposal id (UI)."""
        output = self.desk.handle_turn(message, confirm)
        self.turns.append(output)
        return output

    def snapshot(self) -> dict[str, Any]:
        return {
            "action_audit": self.audit.entries(),
            "final_state": self.tools.final_state(),
            "proposals": self.tools.proposals(),
            "external_events": self.tools.external_events(),
        }


def run_case(case: dict[str, Any], llm: Any, profile: Profile, probe: Any = None,
             case_timeout_s: float = CASE_TIMEOUT_S) -> dict[str, Any]:
    """Run every turn of one contract case from a fresh state. Never raises for a case problem."""
    started = time.perf_counter()  # telemetry only
    if probe is not None:
        probe.start()
    session: CaseSession | None = None
    outputs: list[dict[str, Any]] = []
    status, error = "ok", None
    try:
        session = CaseSession.from_case(case, llm, profile)
        for number, turn in enumerate(case["turns"], start=1):
            if time.perf_counter() - started > case_timeout_s:
                raise TimeoutError(f"case time budget of {case_timeout_s:.0f} s used up before turn {number}")
            if probe is not None:
                probe.mark(f"turn {number}")
            outputs.append(session.run_turn(turn.get("message", ""), turn.get("confirm_proposal")))
    except Exception as exc:  # noqa: BLE001 - a crash becomes a failed, still contract-valid result
        status, error = "failed", f"{type(exc).__name__}: {exc}"[:500]
    finally:
        if probe is not None:
            probe.mark("end")
            probe.stop()
    if error is not None and len(outputs) < len(case["turns"]):
        outputs.append(crashed_turn(error, _unreported_record(session, outputs)))
    outputs += [not_run_turn(f"not run: {error}") for _ in range(len(case["turns"]) - len(outputs))]
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    telemetry = session.desk.telemetry if session else Telemetry()
    if status == "ok" and telemetry.llm_errors:
        status, error = "failed", f"the LLM was unavailable for {telemetry.llm_errors} step(s)"
    snapshot = session.snapshot() if session else _initial_snapshot(case)
    memory = probe.report() if probe is not None else _no_memory()
    latencies = session.desk.turn_latency_ms if session else []
    metrics = build_metrics(telemetry, latencies, elapsed_ms, snapshot["action_audit"], memory, status, error)
    return {"case_id": case.get("case_id"), "turns": outputs, **snapshot, "metrics": metrics}


def not_run_result(case: dict[str, Any], reason: str) -> dict[str, Any]:
    """Result for a case that could not run at all (e.g. the model server would not start)."""
    turns = [not_run_turn(f"not run: {reason}") for _ in case.get("turns", [])]
    metrics = build_metrics(Telemetry(), [], 0.0, [], _no_memory(), "failed", reason)
    return {"case_id": case.get("case_id"), "turns": turns, **_initial_snapshot(case), "metrics": metrics}


def not_run_turn(response: str) -> dict[str, Any]:
    return TurnOutput(decision=ERROR, response=response).to_dict()


def crashed_turn(error: str, record: dict[str, Any] | None) -> dict[str, Any]:
    """The turn during which the case crashed. When a commit persisted first, say so (and name the
    transaction) rather than "not run": money may have moved, and a replay will not repeat it."""
    if record is None:
        return not_run_turn(f"not run: {error}")
    response = (f"error after commit: transaction {record.get('transaction_id')} was recorded before the "
                f"failure ({error}); it will not be repeated.")
    return TurnOutput(decision=ERROR, response=response, transaction=record).to_dict()


def build_metrics(telemetry: Telemetry, latencies: list[float], elapsed_ms: float, audit: list[dict[str, Any]],
                  memory: dict[str, Any], status: str, error: str | None) -> dict[str, Any]:
    """SPEC §8 result.metrics. Throughput is token-weighted (sum of tokens / sum of time)."""
    kinds = [entry.get("kind") for entry in audit]
    metrics: dict[str, Any] = {
        "elapsed_ms": elapsed_ms,
        **memory,
        "llm_calls": telemetry.llm_calls,
        "prompt_tokens": telemetry.prompt_tokens,
        "cached_tokens": telemetry.cached_tokens,
        "completion_tokens": telemetry.completion_tokens,
        "prefill_ms": round(telemetry.prefill_ms, 1),
        "decode_ms": round(telemetry.decode_ms, 1),
        "tool_ms": round(telemetry.tool_ms, 1),
        "decode_tok_s": _rate(telemetry.completion_tokens, telemetry.decode_ms),
        "prefill_tok_s": _rate(telemetry.prefill_tokens, telemetry.prefill_ms),
        "turn_latency_ms": list(latencies),
        "protocol_errors": telemetry.protocol_errors,
        "guard_overrides": kinds.count("guard"),
        "fallbacks": kinds.count("fallback"),
        "gate_rejections": kinds.count("gate"),
        "dropped_calls": telemetry.dropped_calls,
        "llm_errors": telemetry.llm_errors,
        "first_case_after_start": False,
        "status": status,
    }
    if error:
        metrics["error"] = error
    return metrics


def _unreported_record(session: CaseSession | None, outputs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The completed record, if it persisted during the crashed turn (no earlier turn reported it)."""
    record = session.tools.completed_record() if session else None
    reported = {(o.get("transaction") or {}).get("transaction_id") for o in outputs}
    return record if record is not None and record.get("transaction_id") not in reported else None


def _rate(tokens: int, ms: float) -> float | None:
    return round(tokens / ms * 1000, 1) if ms > 0 else None


def _no_memory() -> dict[str, Any]:
    return {"peak_ram_mb": None, "peak_vram_mb": None, "peak_unified_gpu_mb": None, "rss_mb": None,
            "memory_architecture": "unified", "memory_notes": list(NO_PROBE_NOTES)}


def _initial_snapshot(case: dict[str, Any]) -> dict[str, Any]:
    return {"action_audit": [], "final_state": copy.deepcopy(case.get("initial_state", {})),
            "proposals": [], "external_events": []}
