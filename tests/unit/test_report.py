"""bench/report.py: statistics, taxonomy, traces, the gate and the full report from synthetic runs.

The synthetic run directories are written exactly the way bench/run.py writes them (run.json +
results.json + /metrics scrapes), over a tiny contract input whose IDs are unlike the public cases
(C7 / ORD-55 / X9 / HOODIE). Each case expects a single `deny` with no state change, so a result
passes or fails the real oracle (evals.score) just by its decision.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from bench import report
from bench.report import (
    Comparison,
    bootstrap_ci,
    classify_effect,
    decision_trace,
    gate_verdict,
    geometric_mean,
    nearest_rank,
    short_hash,
    text_trace,
)
from evals.state_view import business_view
from evals.score import wilson_interval

STATE: dict[str, Any] = {
    "as_of": "2026-03-15", "currency": "USD",
    "products": [{"sku": "HOODIE-M", "product_id": "HOODIE", "size": "M", "price_cents": 5000, "stock": 2}],
    "orders": [{"order_id": "ORD-55", "customer_id": "C7", "status": "delivered", "delivered_on": "2026-03-01",
                "lines": [{"line_id": "X9", "sku": "HOODIE-M", "paid_cents": 4200, "unopened": True,
                           "final_sale": False, "returned": False}]}],
    "returns": [], "refunds": [], "exchanges": [], "quarantine": [], "idempotency": {},
}
CASE_IDS = ["k_one", "k_two", "k_three"]
CODE = "ab12" * 16  # one decision-code fingerprint shared by runs and floors unless a test says otherwise
FAMILIES = {"k_one": "fam_a", "k_two": "fam_a", "k_three": "fam_b"}

SWITCHES = {
    "B": ("Q8_0", "native", False, False, 8192),
    "B-repeat": ("Q8_0", "native", False, False, 8192),
    "B-q4": ("Q4_K_M", "native", False, False, 8192),
    "F": ("Q4_K_M", "constrained", True, True, 2048),
    "F-q8": ("Q8_0", "constrained", True, True, 2048),
    "F-two": ("Q8_0", "native", True, True, 2048),  # two switches reverted: not a leave-one-out
}


# --------------------------------------------------------------------------------------------------
# Synthetic raw files

def write_inputs(root: Path, suite: str, families: dict[str, str] | None) -> tuple[str, str]:
    cases = [{"case_id": cid, "customer_id": "C7", "request_id": f"REQ-{cid}", "initial_state": STATE,
              "turns": [{"message": "please refund ORD-55 X9"}], "faults": []} for cid in CASE_IDS]
    labels = [{"case_id": cid, "family": (families or {}).get(cid, "public"),
               "expected": {"decisions": ["deny"], "reasons": ["RETURN_WINDOW_EXPIRED"], "proposals": [None],
                            "events": [], "final_business_state": business_view(STATE), "state_unchanged": True,
                            "alternatives": {}}} for cid in CASE_IDS]
    (root / "inputs").mkdir(exist_ok=True)
    input_rel, labels_rel = f"inputs/{suite}_cases.json", f"inputs/{suite}_labels.json"
    (root / input_rel).write_text(json.dumps({"contract_version": 1, "cases": cases}))
    (root / labels_rel).write_text(json.dumps({"cases": labels}))
    return input_rel, labels_rel


def step(turn: int, agent: str, tool: str, latency: float, *, error: dict[str, Any] | None = None,
         message: str | None = None, telemetry: bool = True) -> dict[str, Any]:
    args = {"message": message} if message is not None else {}
    result = ({"latency_ms": latency, "prompt_ms": latency * 0.2, "predicted_ms": latency * 0.7,
               "prompt_tokens": 1000, "cached_tokens": 900, "completion_tokens": 30} if telemetry else None)
    return {"turn": turn, "kind": "llm_step", "agent": agent, "tool": tool, "arguments": args, "result": result,
            "error": error}


def case_result(cid: str, *, decision: str = "deny", elapsed: float = 1000.0, first: bool = False,
                audit: list[dict[str, Any]] | None = None, status: str = "ok", error: str | None = None,
                gpu: float | None = 3000.0, reply: str = "Sorry, that item cannot be returned.") -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "elapsed_ms": elapsed, "peak_ram_mb": 1700.0, "peak_vram_mb": gpu, "peak_unified_gpu_mb": gpu,
        "rss_mb": 4000.0, "per_process": {"adapter": {"peak_phys_footprint_mb": 50.0},
                                          "llama-server": {"peak_phys_footprint_mb": 1650.0}},
        "llm_calls": 3, "prompt_tokens": 3000, "cached_tokens": 2700, "completion_tokens": 90,
        "prefill_ms": elapsed * 0.1, "decode_ms": elapsed * 0.6, "tool_ms": 1.0,
        "turn_latency_ms": [elapsed - 5], "first_case_after_start": first, "status": status,
    }
    if error:
        metrics["error"] = error
    default_audit = [step(1, "resolution", "handoff_to_operations", elapsed * 0.3),
                     step(1, "operations", "create_proposal", elapsed * 0.2),
                     step(1, "resolution", "reply_to_customer", elapsed * 0.3, message=reply),
                     {"turn": 1, "kind": "grounding", "agent": "orchestrator", "tool": "analyze", "arguments": {}}]
    return {"case_id": cid, "turns": [{"decision": decision, "response": reply, "proposal": None,
                                       "transaction": None}],
            "action_audit": default_audit if audit is None else audit, "final_state": copy.deepcopy(STATE),
            "proposals": [], "external_events": [], "metrics": metrics}


HARDWARE = {"cpu": "Test CPU", "cpu_cores": 8, "gpu": "Test GPU (8 cores)", "gpu_count": 1, "ram_gb": 16,
            "unified_memory": True, "os": "macOS 1.0 (T1)", "driver": "Metal (test)", "power": "see run_conditions"}
CONDITIONS = {"before": {"power_source": "AC Power", "powermode": 0, "thermal_warning": False, "sleep_prevented": True,
                         "load_avg_1_5_15": [1.0, 1.1, 1.2]},
              "after": {"power_summary": "AC Power, powermode 0, no thermal warning, sleep prevented"}}


def results_doc(root: Path, input_rel: str, profile: str, results: list[dict[str, Any]],
                page_cache: str = "warm", hardware: dict[str, Any] | None = None) -> dict[str, Any]:
    quant, protocol, prefetch, structured, cache_ram = SWITCHES[profile]
    return {
        "contract_version": 1,
        "input_sha256": hashlib.sha256((root / input_rel).read_bytes()).hexdigest(),
        "hardware": hardware or HARDWARE,
        "configuration": {"profile": profile, "quantization": quant, "protocol": protocol,
                          "app_switches": {"protocol": protocol, "prefetch_account_card": prefetch,
                                           "structured_ops_report": structured},
                          "caches": {"cache_ram_mib": cache_ram}, "model_repo": "test/repo", "model_revision": "rev-1",
                          "model_file": f"model-{quant}.gguf", "model_sha256": "ab" * 32, "model_sha256_verified": True,
                          "model_sha256_source": "cache(path,size,mtime)", "engine": "llama.cpp llama-server",
                          "engine_version": "9.9.9 (build 1)", "engine_version_verified": True, "seed": 42,
                          "temperature": 0, "concurrency": 1, "timeouts": {"llm_step_s": 20.0},
                          "engine_argv": ["llama-server", "-m", "x.gguf", "--port", "8405", "-ngl", "99", "--seed", "42"],
                          "run_conditions": CONDITIONS},
        "startup": {"server_spawn_to_ready_ms": 900.0, "first_request_ms": 60.0, "model_page_cache": page_cache,
                    "model_page_cache_resident_pct": 100.0 if page_cache == "warm" else 0.0},
        "results": results,
    }


PROM_BEFORE = "# moment: before\nllamacpp:prompt_tokens_total 10\nllamacpp:prompt_seconds_total 0.01\n" \
              "llamacpp:tokens_predicted_total 1\nllamacpp:tokens_predicted_seconds_total 0.01\n"
PROM_AFTER = "# moment: after\nllamacpp:prompt_tokens_total 1010\nllamacpp:prompt_seconds_total 1.01\n" \
             "llamacpp:tokens_predicted_total 101\nllamacpp:tokens_predicted_seconds_total 2.01\n"


def write_run(root: Path, results_root: Path, suite: str, profile: str, k: int, results: list[dict[str, Any]],
              *, clean: bool = True, status: str = "complete", prom: bool = True, page_cache: str = "warm",
              saved_strict: int | None = None, dirname: str | None = None,
              families: dict[str, str] | None = None, hardware: dict[str, Any] | None = None,
              argv: bool = True, code: str | None = CODE) -> Path:
    input_rel, labels_rel = write_inputs(root, suite, families)
    run_dir = results_root / suite / profile / (dirname or f"run-{k}")
    run_dir.mkdir(parents=True)
    manifest = {"suite": suite, "profile": profile, "run": k, "position": 1, "status": status, "input": input_rel,
                "labels": labels_rel, "started_at": "2026-10-28T00:00:00+00:00",
                "preflight": {"passed": clean, "overridden": not clean},
                "adapter": {"exit_code": 0, "wall_s": 12.5,
                            "metrics_after": "captured: last case" if prom else "not captured: server gone"},
                "validate": {"structure_ok": True, "baseline_ok": None if profile == "B" else True},
                "score": {"strict_success": 3, "cases": 3}}
    if code is not None:
        manifest["code_sha256"] = code
    (run_dir / "run.json").write_text(json.dumps(manifest))
    (run_dir / "results.json").write_text(json.dumps(results_doc(root, input_rel, profile, results, page_cache, hardware)))
    if argv:
        (run_dir / "argv.txt").write_text(f"python3 adapter.py --profile {profile} --input {input_rel}\n# cwd: x\n")
    if saved_strict is not None:
        (run_dir / "score.json").write_text(json.dumps({"summary": {"strict_success": saved_strict}}))
    if prom:
        (run_dir / "metrics_before.prom").write_text(PROM_BEFORE)
        (run_dir / "metrics_after.prom").write_text(PROM_AFTER)
    return run_dir


def standard_results(scale: float = 1.0, failing: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    return [case_result(cid, decision="clarify" if cid in failing else "deny", elapsed=1000.0 * scale * (i + 1),
                        first=(i == 0)) for i, cid in enumerate(CASE_IDS)]


@pytest.fixture()
def tree(tmp_path: Path) -> tuple[Path, Path]:
    """A results tree with every report section populated."""
    root, results = tmp_path, tmp_path / "results"
    for k in (1, 2):
        write_run(root, results, "public", "B", k, standard_results(2.0), page_cache="cold" if k == 1 else "warm")
        write_run(root, results, "public", "F", k, standard_results(1.0))
        write_run(root, results, "public", "B-repeat", k, standard_results(2.0 + 0.01 * k))
    write_run(root, results, "public", "F-q8", 1, standard_results(1.5), clean=False, argv=False)
    write_run(root, results, "public", "B-q4", 1, standard_results(1.8, failing=("k_three",)))
    write_run(root, results, "public", "F-two", 1, standard_results(1.2))
    write_run(root, results, "variants", "B", 1, standard_results(2.0), families=FAMILIES)
    write_run(root, results, "variants", "F", 1, standard_results(1.0, failing=("k_two",)), families=FAMILIES)
    write_run(root, results, "public", "F", 3, [], status="no_results", dirname="run-3.attempt-1", prom=False)
    return root, results


# --------------------------------------------------------------------------------------------------
# Statistics

def test_nearest_rank_is_an_observed_value_with_the_stated_rank() -> None:
    values = list(range(1, 21))
    assert nearest_rank(values, 50) == 10  # ceil(0.5 * 20) = 10
    assert nearest_rank(values, 95) == 19  # ceil(0.95 * 20) = 19
    assert nearest_rank(list(range(1, 19)), 95) == 18  # n = 18: p95 is the maximum
    assert nearest_rank([4, 1, 3, 2], 50) == 2  # no interpolation
    assert nearest_rank([7], 0) == 7  # rank is at least 1
    assert nearest_rank([], 50) is None


def test_means_and_geometric_mean() -> None:
    assert report.median([3, 1, 2]) == 2 and report.median([]) is None
    assert report.mean([1, 2]) == 1.5 and report.mean([]) is None
    assert geometric_mean([2.0, 0.5]) == pytest.approx(1.0)
    assert geometric_mean([]) is None
    assert geometric_mean([1.0, 0.0]) is None


def test_wilson_interval_used_by_the_report() -> None:
    low, high = wilson_interval(18, 18)
    assert low == pytest.approx(0.8241, abs=1e-4) and high == 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_bootstrap_is_deterministic_and_brackets_the_estimate() -> None:
    groups = [[0.40], [0.45], [0.42], [0.50], [0.38], [0.47], [0.44], [0.41]]
    first, second = bootstrap_ci(groups), bootstrap_ci(groups)
    assert first == second  # seed 42, same resamples every time
    point = geometric_mean([r for g in groups for r in g])
    assert first is not None and point is not None and first[0] < point < first[1]
    assert bootstrap_ci(groups, seed=7) != first
    assert bootstrap_ci([[0.5], [0.5], [0.5]]) == pytest.approx((0.5, 0.5))
    assert bootstrap_ci([[0.5, 0.6]]) is None  # one case: no between-case variation to resample
    assert bootstrap_ci([[0.5], [-1.0]]) is None
    assert bootstrap_ci([[0.5], [], [0.6]]) is not None  # empty groups are ignored


def test_case_level_bootstrap_is_wider_than_treating_repeated_runs_as_independent() -> None:
    values = [0.40, 0.45, 0.42, 0.50, 0.38, 0.47]
    clustered = bootstrap_ci([[v, v, v] for v in values])  # 6 cases x 3 identical runs
    naive = bootstrap_ci([[v] for v in values * 3])  # the same 18 numbers as 18 cases
    assert clustered is not None and naive is not None
    assert clustered[1] - clustered[0] > naive[1] - naive[0]


def test_bootstrap_resample_count_changes_the_interval_only_slightly() -> None:
    small = bootstrap_ci([[0.9], [1.1], [1.0], [0.95], [1.05]], resamples=500)
    assert small is not None and small[0] < 1.0 < small[1]


def test_classify_effect_needs_ci_and_noise_floor() -> None:
    assert classify_effect(None, None, None).startswith("unavailable")
    assert classify_effect(0.9, (0.8, 1.02), None) == "no effect (CI includes 1)"
    assert classify_effect(0.99, (0.985, 0.995), (0.98, 1.03)) == "no effect (inside the A/A interval)"
    assert classify_effect(0.5, (0.45, 0.55), (0.98, 1.03)) == "50.0% faster"
    assert classify_effect(1.2, (1.1, 1.3), (0.98, 1.03)) == "20.0% slower"
    assert classify_effect(0.5, (0.45, 0.55), None) == "50.0% faster (no A/A noise floor measured)"


# --------------------------------------------------------------------------------------------------
# Traces and taxonomy

def test_decision_trace_ignores_wording_and_timings_but_not_decisions() -> None:
    base = case_result("k_one")
    reworded = case_result("k_one", reply="We cannot accept this return, sorry.")
    slower = case_result("k_one", elapsed=9999.0)
    other = case_result("k_one", decision="clarify")
    hashes = {short_hash(decision_trace(r)) for r in (base, reworded, slower)}
    assert len(hashes) == 1
    assert short_hash(text_trace(base)) != short_hash(text_trace(reworded))
    assert short_hash(decision_trace(base)) != short_hash(decision_trace(other))


def test_decision_trace_includes_bindings_and_error_codes() -> None:
    result = case_result("k_one", decision="propose")
    result["turns"][0]["proposal"] = {"proposal_id": "P-0001", "order_id": "ORD-55", "line_id": "X9",
                                      "remedy": "refund", "amount_cents": 4200, "replacement_sku": None}
    changed = copy.deepcopy(result)
    changed["turns"][0]["proposal"]["amount_cents"] = 5000
    assert short_hash(decision_trace(result)) != short_hash(decision_trace(changed))
    errored = copy.deepcopy(result)
    errored["action_audit"][0]["error"] = {"code": "INVALID_ARGS", "message": "x"}
    assert decision_trace(errored)[1][0][-1] == "INVALID_ARGS"
    assert decision_trace({"turns": None, "action_audit": "not a list"}) == [[], []]


def _score(root: Path, result: dict[str, Any]) -> Any:
    input_rel, labels_rel = write_inputs(root, "tax", None)
    doc = results_doc(root, input_rel, "F", [result])
    scores, _ = report.score_run((root / input_rel).read_bytes(), json.loads((root / labels_rel).read_text()), doc)
    return next(s for s in scores if s.case_id == result["case_id"])


def test_failure_causes_cover_the_taxonomy(tmp_path: Path) -> None:
    timeout = case_result("k_one", decision="error", status="failed", audit=[
        step(1, "resolution", "(none)", 0, error={"code": "LLM_ERROR", "message": "timeout: read timed out"},
             telemetry=False)])
    causes = report.failure_causes(_score(tmp_path, timeout), timeout)
    assert "oracle mismatch: decision" in causes and "LLM request timeout" in causes and "crash" not in causes

    broken = case_result("k_one", status="failed", error="KeyError: boom", audit=[
        step(1, "resolution", "(none)", 0, error={"code": "LLM_ERROR", "message": "connection: refused"},
             telemetry=False)])
    assert "LLM error (not a timeout)" in report.failure_causes(_score(tmp_path, broken), broken)

    crash = case_result("k_one", status="failed", error="RuntimeError: boom", audit=[])
    assert report.failure_causes(_score(tmp_path, crash), crash) == ["crash"]
    budget = case_result("k_one", status="failed", audit=[],
                         error="TimeoutError: case time budget of 600 s used up before turn 2")
    assert report.failure_causes(_score(tmp_path, budget), budget) == ["case time budget exceeded"]

    capped = case_result("k_one", audit=[step(1, "operations", "report_to_resolution", 10.0) for _ in range(12)])
    assert "step/call cap reached" in report.failure_causes(_score(tmp_path, capped), capped)
    resolution_capped = case_result("k_one", audit=[step(2, "resolution", "reply_to_customer", 10.0)] * 5)
    assert report.step_cap_reached(resolution_capped["action_audit"])

    mutated = case_result("k_one")
    mutated["final_state"]["products"][0]["stock"] = 1
    causes = report.failure_causes(_score(tmp_path, mutated), mutated)
    assert "oracle mismatch: binding/state/events" in causes and "invariant violation" in causes

    assert report.failure_causes(_score(tmp_path, case_result("k_one")), case_result("k_one")) == []


def test_injected_faults_are_counted_apart_from_failures() -> None:
    audit = [
        {"turn": 0, "kind": "fault", "tool": "timeout_after_commit", "result": {"armed": True}},
        {"turn": 1, "kind": "fault", "tool": "stock_change", "result": {"applied": True}},
        {"turn": 2, "kind": "fault", "tool": "stock_change", "result": {"applied": False}},
        {"turn": 2, "kind": "tool", "tool": "execute_proposal", "error": {"code": "TIMEOUT"}},
        {"turn": 9, "kind": "tool", "tool": "execute_proposal", "error": {"code": "TIMEOUT"}},  # no such turn
    ]
    turns = [{"decision": "propose"}, {"decision": "completed"}]
    found = report.injected_faults(audit, turns)
    assert found == {"timeout_after_commit armed": 1, "stock_change applied": 1,
                     "stock_change not applied (unknown SKU or turn)": 1, "timeout_after_commit fired": 2,
                     "fired timeout reconciled to completed in the same turn": 1}


# --------------------------------------------------------------------------------------------------
# Gate and leave-one-out

def test_gate_verdict_rules() -> None:
    passing = {k: True for k in report.GATE_CRITERIA}
    assert gate_verdict(passing, 3) == "PASS: ships in F"
    assert gate_verdict(passing, 1) == "PASS (provisional: 1 of 3 public runs)"
    assert gate_verdict({**passing, "variants F >= B": None}, 3) == "INCOMPLETE: no data for variants F >= B"
    assert gate_verdict({**passing, "variants F >= B": None, "public F >= B": False}, 3) == "FAIL: F keeps B's choice"


def test_at_least_handles_missing_sides() -> None:
    s = report.Success(3, 3, 1, 0, ["3/3"])
    assert report.at_least(s, report.Success(2, 3, 1, 0, ["2/3"])) is True
    assert report.at_least(report.Success(2, 3, 1, 0, []), s) is False
    assert report.at_least(None, s) is None
    assert report.at_least(s, report.Success(0, 0, 0, 0, [])) is None


def test_leave_one_out_detects_single_reverted_switches(tree: tuple[Path, Path]) -> None:
    root, results = tree
    model = report.build_model(results, root)
    assert report.leave_one_out(model) == {"F-q8": "quantization"}  # F-two reverts two, B-q4 is B-based
    assert report.change_label(model, "quantization") == "weights quantization: Q4_K_M (B: Q8_0)"
    assert report.leave_one_out(report.Model([], [], [], [], [])) == {}


def test_gate_rows_on_the_synthetic_tree(tree: tuple[Path, Path]) -> None:
    root, results = tree
    rows = report.gate_rows(report.build_model(results, root))
    whole, q8 = rows
    assert whole["public F >= B"] is True and whole["variants F >= B"] is False  # F failed k_two on variants
    assert whole["verdict"] == "FAIL: F keeps B's choice"
    assert q8["profile without"] == "F-q8" and q8["public F >= without"] is True


# --------------------------------------------------------------------------------------------------
# Loading

def test_discover_keeps_every_attempt_and_warns(tmp_path: Path) -> None:
    root, results = tmp_path, tmp_path / "results"
    write_run(root, results, "public", "F", 1, standard_results(), saved_strict=2)  # stale score.json
    write_run(root, results, "public", "F", 2, [], status="no_results", prom=False)
    broken = write_run(root, results, "public", "B", 1, standard_results())
    (broken / "results.json").write_text("{not json")
    unscorable = write_run(root, results, "public", "B", 2, standard_results())
    manifest = json.loads((unscorable / "run.json").read_text())
    (unscorable / "run.json").write_text(json.dumps({**manifest, "input": "inputs/missing.json"}))
    (results / "public" / "F" / "notes").mkdir()  # not a run directory
    (results / "public" / "F" / "run-x").mkdir()  # name does not match
    (results / "public" / "F" / "run-9").write_text("a file, not a directory")
    runs, attempts, warnings = report.discover(results, root)
    assert [r.label for r in runs] == ["public/F/run-1"]
    assert len(attempts) == 4
    assert any("score.json says 2 strict, re-scoring gives 3" in w for w in warnings)
    assert any("results.json is unreadable" in w for w in warnings)
    assert any("could not be scored" in w for w in warnings)


def test_label_families_use_case_ids_for_the_public_family(tree: tuple[Path, Path]) -> None:
    root, results = tree
    model = report.build_model(results, root)
    assert {c.scenario for c in model.case_rows("public", "F")} == set(CASE_IDS)
    assert {c.scenario for c in model.case_rows("variants", "F")} == {"fam_a", "fam_b"}


def test_unlabelled_cases_are_skipped(tmp_path: Path) -> None:
    root, results = tmp_path, tmp_path / "results"
    extra = case_result("not_labelled")
    write_run(root, results, "public", "F", 1, [*standard_results(), extra])
    model = report.build_model(results, root)
    assert {c.case_id for c in model.cases} == set(CASE_IDS)


def test_prom_delta_and_missing_scrapes(tmp_path: Path) -> None:
    root, results = tmp_path, tmp_path / "results"
    write_run(root, results, "public", "F", 1, standard_results())
    write_run(root, results, "public", "F", 2, standard_results(), prom=False)
    model = report.build_model(results, root)
    first, second = model.run_list("public", "F")
    assert report.prom_delta(first) == pytest.approx({
        "llamacpp:prompt_tokens_total": 1000, "llamacpp:prompt_seconds_total": 1.0,
        "llamacpp:tokens_predicted_total": 100, "llamacpp:tokens_predicted_seconds_total": 2.0})
    assert report.prom_delta(second) is None
    assert report.server_rates(model, "public", "F") == ["50.0", "1,000.0", "1"]
    assert any("/metrics delta (public/F/run-2)" == m for m, _ in model.unavailable)
    restarted = report.build_model(results, root)
    restarted.runs[0].manifest["adapter"]["server_restarts"] = 1
    assert report.server_rates(restarted, "public", "F") == ["n/a", "n/a", "0"]
    assert any("restarted during the run" in reason for _, reason in restarted.unavailable)
    only_missing = report.build_model(results, root)
    only_missing.runs = [r for r in only_missing.runs if r.run == 2]
    assert report.server_rates(only_missing, "public", "F") == ["n/a", "n/a", "0"]


# --------------------------------------------------------------------------------------------------
# Comparisons

def test_paired_ratios_use_only_pairs_successful_in_both(tree: tuple[Path, Path]) -> None:
    root, results = tree
    model = report.build_model(results, root)
    by_case, excluded = report.paired_ratios(model.case_rows("public", "B-q4"), model.case_rows("public", "B"))
    assert set(by_case) == {"k_one", "k_two"} and excluded == 1  # k_three failed in B-q4; B-q4 has run 1 only
    assert all(r == pytest.approx(0.9) for group in by_case.values() for r in group)


def test_comparisons_include_the_aa_floor_and_leave_one_out(tree: tuple[Path, Path]) -> None:
    root, results = tree
    model = report.build_model(results, root)
    found = {(c.suite, c.profile, c.reference): c for c in report.comparisons(model)}
    aa = found[("public", "B-repeat", "B")]
    assert aa.verdict == "A/A noise floor" and aa.pairs == 6
    f_vs_b = found[("public", "F", "B")]
    assert f_vs_b.ratio == pytest.approx(0.5) and f_vs_b.verdict == "50.0% faster"
    assert ("public", "F", "F-q8") in found and ("public", "F", "B-q4") in found
    variants = found[("variants", "F", "B")]
    assert variants.pairs == 2 and variants.excluded == 1
    assert variants.verdict.endswith("(no A/A noise floor measured)")
    assert any(m == "A/A noise floor (variants)" for m, _ in model.unavailable)


def test_comparison_skips_missing_references(tmp_path: Path) -> None:
    root, results = tmp_path, tmp_path / "results"
    write_run(root, results, "public", "F", 1, standard_results())
    write_run(root, results, "public", "F-q8", 1, standard_results())
    model = report.build_model(results, root)
    assert report.comparisons(model) == []  # no B in the suite: nothing to compare with


# --------------------------------------------------------------------------------------------------
# The full report

def test_report_end_to_end(tree: tuple[Path, Path]) -> None:
    root, results = tree
    out = results / "REPORT.md"
    model = report.build_report(results, out, root)
    text = out.read_text()
    for heading in ("## Quality", "## Pre-registered acceptance gate", "## Latency", "## Paired latency comparisons",
                    "## Per-change leave-one-out", "## Per-LLM-step latency distribution", "## Where the time goes",
                    "## Memory", "## Cold start", "## Failures and timeouts", "## Decision-trace identity",
                    "## Every attempted run", "## Unavailable metrics", "## Files"):
        assert heading in text, heading
    assert "| F | 6/6 | 100.0% |" in text  # pooled over two runs
    assert "| B-q4 | 2/3 |" in text
    assert "**Timings not clean:** 1 run(s)" in text and "public/F-q8/run-1" in text
    assert "run-3.attempt-1" in text and "no_results" in text
    assert "weights quantization: Q4_K_M (B: Q8_0)" in text
    assert "| public | F | 2 | 3/3 | 3/3 |" in text  # identical traces across both runs
    assert "n=6, p95 = max" in text
    assert "fired timeout" not in text  # no injected faults in this tree
    assert "per-process GPU memory" in text and "energy / power draw" in text
    assert "decision-trace identity (public/F-q8)" in text
    csv_dir = results / "csv"
    with (csv_dir / "cases.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(model.cases) == 3 * 11  # 11 complete runs of 3 cases
    with (csv_dir / "comparisons.csv").open() as handle:
        assert any(r["profile"] == "B-repeat" and r["verdict"] == "A/A noise floor" for r in csv.DictReader(handle))
    assert {p.name for p in csv_dir.iterdir()} == {"runs.csv", "cases.csv", "steps.csv", "summary.csv",
                                                   "comparisons.csv", "gate.csv"}


def test_report_sections_for_failures_faults_and_gaps(tmp_path: Path) -> None:
    root, results = tmp_path, tmp_path / "results"
    faulted = case_result("k_two", audit=[
        {"turn": 0, "kind": "fault", "tool": "timeout_after_commit", "result": {"armed": True}},
        step(1, "operations", "execute_proposal", 300.0, error={"code": "TOOL_NOT_ALLOWED", "message": "x"}),
        {"turn": 1, "kind": "fallback", "tool": "create_proposal"},
        {"turn": 1, "kind": "guard", "tool": "reply_guard"},
    ])
    timed_out = case_result("k_three", decision="error", status="failed", gpu=None, audit=[
        step(1, "resolution", "(none)", 0, error={"code": "LLM_ERROR", "message": "timeout: 20 s"}, telemetry=False)])
    for k in (1, 2):
        variant = "different wording" if k == 2 else "Sorry, that item cannot be returned."
        write_run(root, results, "public", "F", k, [case_result("k_one", reply=variant, decision="deny" if k == 1
                                                                else "clarify"), faulted, timed_out])
    write_run(root, results, "public", "B", 1, standard_results(2.0)[:2])  # B never ran k_three
    text = (report.build_report(results, results / "REPORT.md", root), (results / "REPORT.md").read_text())[1]
    assert "| LLM request timeout | 0 | 2 |" in text
    assert "|   protocol errors: TOOL_NOT_ALLOWED | 0 | 2 |" in text
    assert "|   code fallbacks: create_proposal | 0 | 2 |" in text
    assert "|   guard overrides: reply_guard | 0 | 2 |" in text
    assert "Injected faults (expected, not failures)" in text and "| timeout_after_commit armed | 0 | 2 |" in text
    assert "Failing case-runs" in text and "LLM request timeout" in text
    assert "| public | F | 2 | 2/3 | 2/3 | k_one |" in text  # k_one's decision (and text) changed
    assert "peak_vram_mb (public/F)" in text
    assert "true SSD cold start" in text  # every synthetic start was warm
    assert "No leave-one-out runs yet." in text
    assert "| k_three | n/a |" in text  # per-scenario cell for a case B never ran
    assert "| case time budget exceeded | 0 | 0 |" in text  # canonical causes are always listed


def test_single_profile_report_lists_warnings_and_missing_comparisons(tmp_path: Path) -> None:
    root, results = tmp_path, tmp_path / "results"
    write_run(root, results, "public", "F", 1, standard_results(), saved_strict=1)
    report.build_report(results, results / "REPORT.md", root)
    text = (results / "REPORT.md").read_text()
    assert "Warnings:" in text and "score.json says 1 strict, re-scoring gives 3" in text
    assert "| paired comparisons | no suite has two profiles with complete runs |" in text
    assert "| F as a whole (all changes) | B | n/a | n/a | yes | n/a | INCOMPLETE" in text


def test_gate_without_any_final_runs(tmp_path: Path) -> None:
    root, results = tmp_path, tmp_path / "results"
    write_run(root, results, "public", "B", 1, standard_results())
    (whole,) = report.gate_rows(report.build_model(results, root))
    assert whole["F invariant violations = 0"] is None and whole["verdict"].startswith("INCOMPLETE")


def test_empty_results_root_reports_what_is_missing(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    report.build_report(results, results / "REPORT.md", tmp_path)
    text = (results / "REPORT.md").read_text()
    assert "| everything | no complete run under" in text and "## Quality" not in text


def test_formatting_helpers() -> None:
    assert report.fmt(None) == "n/a" and report.fmt(1234.5) == "1,234" and report.fmt(1.25, 1) == "1.2"
    assert report.fmt_pct(None) == "n/a" and report.fmt_pct(0.5) == "50.0%"
    assert report.fmt_ratio(None) == "n/a" and report.fmt_ratio(0.41234) == "0.412"
    assert report.fmt_ci(None) == "n/a" and report.fmt_ci((0.1, 0.2)) == "[0.100, 0.200]"
    assert report.pctl_label(0) == "n=0" and report.pctl_label(25) == "n=25"
    assert report.rate(10, 0) is None and report.rate(10, 1000) == 10
    assert report.profile_order("B") < report.profile_order("F") < report.profile_order("F-q8") \
        < report.profile_order("B-q4") < report.profile_order("B-repeat")
    assert report._rel(Path("/elsewhere/x")) == "/elsewhere/x"
    assert Comparison("s", "p", "r", 0, 0, None, None).verdict == ""


def test_main_cli(tree: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
                  monkeypatch: pytest.MonkeyPatch) -> None:
    root, results = tree
    assert report.main(["--results", str(tmp := results / "missing")]) == 2
    assert not tmp.exists()
    monkeypatch.setattr(report, "ROOT", root)
    original = report.build_model
    monkeypatch.setattr(report, "build_model", lambda r, _root=report.ROOT: original(r, root))
    assert report.main(["--results", str(results), "--out", str(results / "custom.md")]) == 0
    assert "complete run(s)" in capsys.readouterr().out
    assert (results / "custom.md").exists()
    assert report.main(["--results", str(results)]) == 0
    assert (results / "REPORT.md").exists()


def test_unavailable_entries_are_listed_once() -> None:
    model = report.Model([], [], [], [], [])
    model.missing("energy", "needs root")
    model.missing("energy", "needs root")
    assert model.unavailable == [("energy", "needs root")]


# --------------------------------------------------------------------------------------------------
# Setup disclosure and code-only floors

def write_floor(root: Path, results: Path, suite: str, mode: str, failing: tuple[str, ...] = (),
                code: str | None = CODE) -> Path:
    input_rel, labels_rel = write_inputs(root, suite, None)
    out = results / suite / "floors" / mode
    out.mkdir(parents=True)
    document = {"contract_version": 1, "input_sha256": hashlib.sha256((root / input_rel).read_bytes()).hexdigest(),
                "results": [case_result(cid, decision="clarify" if cid in failing else "deny") for cid in CASE_IDS]}
    (out / "results.json").write_text(json.dumps(document))
    record = {"suite": suite, "mode": mode, "input": input_rel, "labels": labels_rel,
              "command": f"python -m evals.holdout.run_stub --mode {mode}"}
    if code is not None:
        record["code_sha256"] = code
    (out / "floor.json").write_text(json.dumps(record))
    return out


def test_report_opens_with_the_setup_disclosure(tree: tuple[Path, Path]) -> None:
    root, results = tree
    report.build_report(results, results / "REPORT.md", root)
    text = (results / "REPORT.md").read_text()
    assert text.index("## Setup") < text.index("## Quality")
    for value in ("Test CPU", "Test GPU (8 cores)", "16 GB unified memory", "macOS 1.0 (T1)", "Metal (test)",
                  "see run_conditions", "identical in every run"):
        assert value in text, value
    for profile in report.PROFILES.values():  # every profile is defined, run or not
        assert f"| {profile.name} | {profile.quantization} | {profile.protocol} |" in text
        assert profile.description in text
    assert "| F | model-Q4_K_M.gguf (Q4_K_M) | `abababababababab…` verified=True, source=cache(path,size,mtime) |" in text
    assert "llama.cpp llama-server 9.9.9 (build 1) (match=True)" in text and "rev-1" in text
    assert "- F: `-ngl 99 --seed 42`" in text
    assert "| public/F/run-1 | AC Power, powermode 0, no thermal warning, sleep prevented | AC Power, powermode 0, " \
           "no thermal warning, sleep prevented | 1.0, 1.1, 1.2 | passed |" in text
    assert "- F: `python3 adapter.py --profile F --input inputs/public_cases.json`" in text
    assert "- F-q8: `n/a`" in text  # argv.txt missing
    assert "scripts/bench.sh" in text and "python -m bench.run --suite <suite> --floors-only" in text


def test_setup_warns_when_runs_declare_different_hardware(tmp_path: Path) -> None:
    root, results = tmp_path, tmp_path / "results"
    write_run(root, results, "public", "B", 1, standard_results())
    write_run(root, results, "public", "F", 1, standard_results(), hardware={**HARDWARE, "os": "macOS 2.0"})
    report.build_report(results, results / "REPORT.md", root)
    text = (results / "REPORT.md").read_text()
    assert "**Warning:** the hardware declaration differs in 1 run(s): public/F/run-1." in text


def test_power_cells_for_missing_and_partial_readings() -> None:
    assert report._power(None) == "n/a" and report._power({}) == "n/a"
    assert report._power({"power_source": "Battery Power"}) == \
        "Battery Power, powermode unknown, thermal unknown, sleep unknown"


def test_floors_add_rows_and_the_agent_attributable_column(tmp_path: Path) -> None:
    root, results = tmp_path, tmp_path / "results"
    write_run(root, results, "public", "B", 1, standard_results())
    write_run(root, results, "public", "F", 1, standard_results())
    write_floor(root, results, "public", "always-clarify", failing=("k_one", "k_two"))
    write_floor(root, results, "public", "fake-policy", failing=("k_one",))
    write_floor(root, results, "holdout_x", "always-clarify")  # a suite with a floor and no model run yet
    model = report.build_report(results, results / "REPORT.md", root)
    text = (results / "REPORT.md").read_text()
    assert [(f.suite, f.mode, f.strict, f.n) for f in model.floors] == [
        ("holdout_x", "always-clarify", 3, 3), ("public", "always-clarify", 1, 3), ("public", "fake-policy", 2, 3)]
    assert "| F | 3/3 | 100.0% | [43.8%, 100.0%] | 1 | 3/3 | 3/3 | 3/3 | 0 | +66.7 pp |" in text
    assert "| floor: always-clarify (no model) | 1/3 | 33.3% |" in text and "| +0.0 pp |" in text
    assert "| floor: fake-policy (no model) | 2/3 | 66.7% |" in text
    assert "### Suite `holdout_x`" in text and model.quality_suites() == ["public", "holdout_x"]
    assert "code-only floor" not in "".join(m for m, _ in model.unavailable)


OLD_RUN = "no code fingerprint (written before the bench recorded one)"


@pytest.mark.parametrize(("run_code", "floor_code", "shown"), [
    ("c0de" * 16, "beef" * 16, "F runs: code c0dec0dec0de; floor: code beefbeefbeef."),
    (None, CODE, f"F runs: {OLD_RUN}; floor: code {CODE[:12]}."),
    (CODE, None, f"F runs: code {CODE[:12]}; floor: {OLD_RUN}."),
])
def test_agent_attributable_needs_the_same_code_in_runs_and_floor(tmp_path: Path, run_code: str | None,
                                                                   floor_code: str | None, shown: str) -> None:
    """A floor of today's desk minus a run of yesterday's desk would credit the model with a code change."""
    root, results = tmp_path, tmp_path / "results"
    write_run(root, results, "public", "F", 1, standard_results(), code=run_code)
    write_floor(root, results, "public", "always-clarify", failing=("k_one", "k_two"), code=floor_code)
    report.build_report(results, results / "REPORT.md", root)
    text = (results / "REPORT.md").read_text()
    assert "| F | 3/3 | 100.0% | [43.8%, 100.0%] | 1 | 3/3 | 3/3 | 3/3 | 0 | n/a: other code |" in text
    assert "**agent-attributable not computed** (runs and floor executed different decision code" in text
    assert shown in text


def test_a_profile_with_runs_of_mixed_code_is_not_compared(tmp_path: Path) -> None:
    root, results = tmp_path, tmp_path / "results"
    write_run(root, results, "public", "F", 1, standard_results())
    write_run(root, results, "public", "F", 2, standard_results(), code=None)
    write_floor(root, results, "public", "always-clarify")
    report.build_report(results, results / "REPORT.md", root)
    assert f"F runs: code {CODE[:12]} and {OLD_RUN}; floor: code {CODE[:12]}." in (results / "REPORT.md").read_text()


def test_a_suite_without_floors_says_how_to_get_them(tree: tuple[Path, Path]) -> None:
    root, results = tree
    model = report.build_report(results, results / "REPORT.md", root)
    assert ("code-only floor (public)", "no results/public/floors/always-clarify; run "
            "`python -m bench.run --suite public --floors-only`") in model.unavailable
    assert "| F | 6/6 | 100.0% | [61.0%, 100.0%] | 2 | 3/3, 3/3 | 6/6 | 6/6 | 0 | n/a |" in \
        (results / "REPORT.md").read_text()


def test_broken_floors_are_warned_about_and_skipped(tmp_path: Path) -> None:
    root, results = tmp_path, tmp_path / "results"
    unreadable = write_floor(root, results, "public", "always-clarify")
    (unreadable / "results.json").write_text("{not json")
    unscorable = write_floor(root, results, "public", "fake-policy")
    (unscorable / "floor.json").write_text(json.dumps({"input": "missing.json", "labels": "missing.json"}))
    empty = write_floor(root, results, "other", "always-clarify")
    (root / "inputs" / "empty_labels.json").write_text(json.dumps({"cases": []}))
    (empty / "floor.json").write_text(json.dumps({**json.loads((empty / "floor.json").read_text()),
                                                  "labels": "inputs/empty_labels.json"}))
    floors, warnings = report.discover_floors(results, root)
    assert floors == [] and len(warnings) == 3
    assert any("floor files unreadable" in w for w in warnings) and any("could not be scored" in w for w in warnings)
    assert any("no labelled case" in w for w in warnings)


def test_floors_alone_still_get_a_quality_section(tmp_path: Path) -> None:
    root, results = tmp_path, tmp_path / "results"
    write_floor(root, results, "public", "always-clarify")
    report.build_report(results, results / "REPORT.md", root)
    text = (results / "REPORT.md").read_text()
    assert "## Quality" in text and "| floor: always-clarify (no model) | 3/3 |" in text and "## Setup" not in text
