"""bench/run.py (schedule, run directories, preflight handling, the per-run pipeline, the CLI) and
the shell scripts it relies on (bench/preflight.sh, scripts/reset.sh).

The per-run pipeline is exercised end to end against a fake adapter: a small Python script that
serves /health and /metrics on the run's port, logs one '[adapter] <case>: [...]' line per case
like desk/adapter_main.py, writes a contract-valid results.json and a server log. The real
validator (candidate_assets/validate_submission.py) and the real scorer (evals.score) then run on
its output, so the files a run directory holds are the ones a real run produces.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from bench import ROOT, SUITES, Suite, code_fingerprint
from bench import run as bench_run
from bench.run import Bench, ModelCheckFailed, PreflightRefused, RunExists, RunSpec, schedule
from desk.profiles import get_profile
from evals.state_view import business_view

STATE: dict[str, Any] = {
    "as_of": "2026-03-15", "currency": "USD",
    "products": [{"sku": "HOODIE-M", "product_id": "HOODIE", "size": "M", "price_cents": 5000, "stock": 2}],
    "orders": [{"order_id": "ORD-55", "customer_id": "C7", "status": "delivered", "delivered_on": "2026-03-01",
                "lines": [{"line_id": "X9", "sku": "HOODIE-M", "paid_cents": 4200, "unopened": True,
                           "final_sale": False, "returned": False}]}],
    "returns": [], "refunds": [], "exchanges": [], "quarantine": [], "idempotency": {},
}
CASE_IDS = ["k_one", "k_two"]

FAKE_ADAPTER = textwrap.dedent('''
    """Stands in for adapter.py: same CLI, same log lines, a fake llama-server on --port."""
    import argparse, hashlib, http.server, json, os, signal, sys, threading, time
    from pathlib import Path

    parser = argparse.ArgumentParser()
    for flag in ("--profile", "--port", "--input", "--output"):
        parser.add_argument(flag)
    parser.add_argument("--strict-engine", action="store_true")
    args = parser.parse_args()
    engine_check = "strict" if args.strict_engine else "warn"
    mode = os.environ.get("FAKE_MODE", "ok")
    if mode == "ignore_term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                body, code = b'{"status": "ok"}', 200
            elif self.path == "/metrics" and mode != "metrics_500":
                body, code = b"llamacpp:tokens_predicted_total 7\\n", 200
            else:
                body, code = b"no", 500
            self.send_response(code)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = None
    if mode != "no_server":
        server = http.server.HTTPServer(("127.0.0.1", int(args.port)), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        log_dir = Path(os.environ["FAKE_LOG_DIR"])
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"server-{args.profile}-{args.port}.log").write_text("fake server log\\n")
        if mode == "restart":
            (log_dir / f"server-{args.profile}-{args.port}-restart1.log").write_text("restarted\\n")
        time.sleep(0.3)  # let the bench see /health before the cases run
    if mode in ("hang", "ignore_term"):
        time.sleep(60)
    raw = Path(args.input).read_bytes()
    cases = json.loads(raw)["cases"]
    for number, case in enumerate(cases, start=1):
        if mode == "metrics_gone" and number == len(cases) and server is not None:
            server.shutdown()
            server.server_close()
        if mode == "restart" and number == 1:
            print("[adapter] llama-server is not healthy after this case; restarting it for the next case",
                  file=sys.stderr, flush=True)
        print(f"[adapter] {case['case_id']}: ['deny'] (1 ms, ok)", file=sys.stderr, flush=True)
    time.sleep(0.2)  # the real adapter collects its after-run conditions here
    if mode != "no_results":
        results = [{"case_id": c["case_id"],
                    "turns": [{"decision": "deny", "response": "no", "proposal": None} for _ in c["turns"]],
                    "action_audit": [], "final_state": c["initial_state"], "proposals": [], "external_events": [],
                    "metrics": {"elapsed_ms": 1.0, "peak_ram_mb": None, "peak_vram_mb": None}} for c in cases]
        document = {"contract_version": 1, "input_sha256": hashlib.sha256(raw).hexdigest(),
                    "hardware": {"cpu": "fake"}, "configuration": {"profile": args.profile, "engine_check": engine_check},
                    "results": results}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(document))
    sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
''')


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def write_script(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


@pytest.fixture()
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Bench:
    """A Bench rooted in tmp_path: synthetic suite, fake adapter, passing fake preflight."""
    cases = [{"case_id": cid, "customer_id": "C7", "request_id": f"REQ-{cid}", "initial_state": STATE,
              "turns": [{"message": "refund ORD-55 X9"}], "faults": []} for cid in CASE_IDS]
    labels = [{"case_id": cid, "family": "public",
               "expected": {"decisions": ["deny"], "reasons": ["X"], "proposals": [None], "events": [],
                            "final_business_state": business_view(STATE), "state_unchanged": True,
                            "alternatives": {}}} for cid in CASE_IDS]
    (tmp_path / "cases.json").write_text(json.dumps({"contract_version": 1, "cases": cases}))
    (tmp_path / "labels.json").write_text(json.dumps({"cases": labels}))
    (tmp_path / "fake_adapter.py").write_text(FAKE_ADAPTER)
    preflight = write_script(tmp_path / "preflight_ok.sh", 'echo "PASS  all"; echo "PREFLIGHT PASSED"\n')
    monkeypatch.setenv("FAKE_LOG_DIR", str(tmp_path / "var" / "adapter"))
    monkeypatch.setenv("PYTHONPATH", str(ROOT))  # so `python -m evals.score` works from tmp_path
    monkeypatch.delenv("FAKE_MODE", raising=False)
    monkeypatch.delenv("FAKE_EXIT", raising=False)
    return Bench(Suite("public", "cases.json", "labels.json"), port=free_port(), results_root=tmp_path / "results",
                 root=tmp_path, python3=sys.executable, adapter="fake_adapter.py",
                 validator=str(ROOT / "candidate_assets" / "validate_submission.py"), tool_python=sys.executable,
                 preflight=(str(preflight),), server_log_dir=tmp_path / "var" / "adapter", run_timeout_s=30.0,
                 ready_timeout_s=10.0)


# --------------------------------------------------------------------------------------------------
# Schedule

def test_schedule_is_seeded_interleaved_and_pinned() -> None:
    specs = schedule(["B", "F"], 1, "public")
    assert [(s.profile, s.run, s.position) for s in specs] == [("F", 1, 1), ("B", 1, 2)]
    names = ["B", "F", "F-q8", "F-native", "F-noprefetch", "F-freetext", "F-cacheram-default", "B-q4", "B-repeat"]
    matrix = schedule(names, 3, "public")
    assert [s.profile for s in matrix[:9]] == ["F-native", "F-cacheram-default", "B-q4", "F-noprefetch", "B-repeat",
                                               "F-q8", "F-freetext", "B", "F"]
    for block in (1, 2, 3):
        assert sorted(s.profile for s in matrix if s.run == block) == sorted(names)  # every profile once per block
    assert schedule(names, 3, "public") == matrix  # deterministic
    assert schedule(names, 3, "public", seed=7) != matrix
    assert RunSpec("public", "F", 2, 1).directory(Path("/r")) == Path("/r/public/F/run-2")


def test_write_schedule_never_overwrites(bench: Bench) -> None:
    specs = schedule(["B", "F"], 2, "public")
    first, second = bench_run.write_schedule(bench, specs), bench_run.write_schedule(bench, specs)
    third = bench_run.write_schedule(bench, specs)
    assert (first.name, second.name, third.name) == ("schedule.json", "schedule-2.json", "schedule-3.json")
    document = json.loads(first.read_text())
    assert document["seed"] == 42 and len(document["runs"]) == 4


# --------------------------------------------------------------------------------------------------
# Run directories

def test_prepare_run_dir_keeps_every_attempt(tmp_path: Path) -> None:
    run_dir = tmp_path / "public" / "F" / "run-1"
    assert bench_run.prepare_run_dir(run_dir, resume=False) and run_dir.is_dir()
    (run_dir / "run.json").write_text(json.dumps({"status": "no_results"}))
    assert bench_run.prepare_run_dir(run_dir, resume=False)
    assert (tmp_path / "public" / "F" / "run-1.attempt-1" / "run.json").exists()
    (run_dir / "run.json").write_text("not json")
    assert bench_run.prepare_run_dir(run_dir, resume=True)
    assert (tmp_path / "public" / "F" / "run-1.attempt-2").is_dir()
    (run_dir / "run.json").write_text(json.dumps({"status": "complete"}))
    assert bench_run.prepare_run_dir(run_dir, resume=True) is False
    with pytest.raises(RunExists):
        bench_run.prepare_run_dir(run_dir, resume=False)
    (run_dir / "run.json").write_text("[1, 2]")  # valid JSON, not a manifest
    assert bench_run.read_manifest(run_dir) is None


# --------------------------------------------------------------------------------------------------
# Preflight

def test_preflight_pass_override_refuse_wait_and_usage(bench: Bench, tmp_path: Path) -> None:
    run_dir = tmp_path / "rd"
    run_dir.mkdir()
    outcome = bench_run.run_preflight(bench, run_dir)
    assert outcome.passed and not outcome.overridden and outcome.verdict == "PREFLIGHT PASSED"
    assert "--port" in (run_dir / "preflight.txt").read_text()

    dirty = write_script(tmp_path / "dirty.sh", 'echo "FAIL  cpu"; echo "PREFLIGHT REFUSED"; exit 1\n')
    bench.preflight = (str(dirty),)
    with pytest.raises(PreflightRefused, match="FAIL  cpu"):
        bench_run.run_preflight(bench, run_dir)
    bench.allow_dirty = True
    outcome = bench_run.run_preflight(bench, run_dir)
    assert outcome.overridden and not outcome.passed

    bench.allow_dirty = False
    counter = tmp_path / "count"
    flaky = write_script(tmp_path / "flaky.sh", f'if [ -e {counter} ]; then echo ok; exit 0; fi\n'
                                                f'touch {counter}; echo "PREFLIGHT REFUSED"; exit 1\n')
    bench.preflight, bench.preflight_wait_s, bench.preflight_poll_s = (str(flaky),), 1.0, 0.01
    outcome = bench_run.run_preflight(bench, run_dir)
    assert outcome.passed and outcome.waited_s == pytest.approx(0.01)

    bench.preflight = (str(write_script(tmp_path / "usage.sh", "exit 2\n")),)
    with pytest.raises(RuntimeError, match="exit 2"):
        bench_run.run_preflight(bench, run_dir)
    assert bench_run._last_line("") == ""


# --------------------------------------------------------------------------------------------------
# The per-run pipeline against the fake adapter

def run_one(bench: Bench, profile: str, k: int = 1, resume: bool = False) -> dict[str, Any] | None:
    return bench_run.execute_run(bench, RunSpec("public", profile, k, 1), CASE_IDS, resume)


def test_execute_run_writes_every_file(bench: Bench) -> None:
    baseline = run_one(bench, "B")
    assert baseline is not None and baseline["status"] == "complete"
    manifest = run_one(bench, "F")
    assert manifest is not None
    run_dir = bench.results_root / "public" / "F" / "run-1"
    expected = {"results.json", "adapter.log", "server.log", "metrics_before.prom", "metrics_after.prom",
                "validate.txt", "score.txt", "score.json", "run_conditions.json", "argv.txt", "preflight.txt",
                "run.json"}  # (preflight/ comes from the real preflight.sh --out, not from this fake)
    assert expected <= {p.name for p in run_dir.iterdir()}
    assert manifest["adapter"]["metrics_before"].startswith("captured")
    assert manifest["adapter"]["metrics_after"] == "captured: the adapter logged its last case"
    assert manifest["adapter"]["cases_logged"] == 2 and manifest["adapter"]["exit_code"] == 0
    assert "# moment: the adapter logged its last case" in (run_dir / "metrics_after.prom").read_text()
    assert manifest["validate"] == {"structure_ok": True, "baseline": "results/public/B/run-1/results.json",
                                    "baseline_ok": True}
    assert "--baseline results/public/B/run-1/results.json" in (run_dir / "validate.txt").read_text()
    assert manifest["score"] == {"exit": 0, "strict_success": 2, "cases": 2, "invariant_violations": 0}
    assert manifest["code_sha256"] == code_fingerprint(bench.root)  # the report pairs runs and floors by it
    argv = (run_dir / "argv.txt").read_text()
    assert f"fake_adapter.py --profile F --port {bench.port} --strict-engine --input cases.json --output " \
           "results/public/F/run-1/results.json" in argv
    assert json.loads((run_dir / "results.json").read_text())["configuration"]["engine_check"] == "strict"
    conditions = json.loads((run_dir / "run_conditions.json").read_text())
    assert {"preflight", "before_run", "after_run"} <= set(conditions)
    attempts = (bench.results_root / "public" / "attempts.jsonl").read_text().splitlines()
    assert [json.loads(a)["profile"] for a in attempts] == ["B", "F"]
    assert run_one(bench, "F", resume=True) is None  # complete: skipped
    assert "complete: strict 2/2" in bench_run.summary_line(manifest)


def test_execute_run_without_results_and_with_server_restart(bench: Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MODE", "no_results")
    monkeypatch.setenv("FAKE_EXIT", "3")
    manifest = run_one(bench, "F")
    assert manifest is not None and manifest["status"] == "no_results" and manifest["adapter"]["exit_code"] == 3
    assert "validate" not in manifest
    assert "no_results: strict None/None" in bench_run.summary_line(manifest)
    monkeypatch.setenv("FAKE_MODE", "restart")
    monkeypatch.setenv("FAKE_EXIT", "0")
    manifest = run_one(bench, "F")  # the incomplete run-1 becomes run-1.attempt-1
    run_dir = bench.results_root / "public" / "F" / "run-1"
    assert manifest is not None and manifest["adapter"]["server_restarts"] == 1
    assert (run_dir / "server-restart1.log").read_text() == "restarted\n"
    assert (bench.results_root / "public" / "F" / "run-1.attempt-1" / "run.json").exists()


def test_metrics_scrape_failures_are_recorded(bench: Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MODE", "metrics_500")
    manifest = run_one(bench, "F", 1)
    assert manifest is not None
    assert manifest["adapter"]["metrics_before"].startswith("not captured: /metrics answered HTTP 500")
    monkeypatch.setenv("FAKE_MODE", "metrics_gone")
    manifest = run_one(bench, "F", 2)
    assert manifest is not None
    assert manifest["adapter"]["metrics_after"].startswith("not captured: /metrics unreachable")


def test_no_server_no_log_and_failed_scoring(bench: Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MODE", "no_server")
    bench.ready_timeout_s = 0.3
    bench.suite = Suite("public", "cases.json", "missing_labels.json")
    manifest = run_one(bench, "F")
    assert manifest is not None and manifest["status"] == "complete"
    assert manifest["adapter"]["metrics_before"] == "not captured: the server never answered /health"
    assert manifest["server_log"].startswith("unavailable: no server-F-")
    assert manifest["score"]["strict_success"] is None and manifest["score"]["exit"] == 2


def test_run_timeout_kills_the_process_group(bench: Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MODE", "hang")
    bench.run_timeout_s = 1.0
    manifest = run_one(bench, "F")
    assert manifest is not None and manifest["status"] == "timed_out" and manifest["adapter"]["timed_out"]
    assert manifest["adapter"]["exit_code"] == -signal.SIGTERM
    assert manifest["adapter"]["orphan_server_killed"] is None  # the fake's server died with it


def test_a_server_left_on_the_port_after_a_killed_run_is_stopped(bench: Bench) -> None:
    listener = subprocess.Popen([sys.executable, "-c", "import socket, sys, time; s = socket.socket(); "
                                 "s.bind(('127.0.0.1', int(sys.argv[1]))); s.listen(); time.sleep(60)", str(bench.port)],
                                start_new_session=True)  # like llama-server: its own session, outside the adapter's group
    try:
        deadline = time.monotonic() + 60  # lsof can take seconds on a loaded machine
        while bench_run.pid_listening_on(bench.port) is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert bench_run.stop_orphan_server(bench.port) == listener.pid
        assert listener.wait(timeout=10) == -signal.SIGKILL
        assert bench_run.stop_orphan_server(bench.port) is None  # nothing left
    finally:
        if listener.poll() is None:
            listener.kill()


def test_stop_orphan_server_never_kills_itself_or_a_vanished_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench_run, "pid_listening_on", lambda port: os.getpid())
    assert bench_run.stop_orphan_server(1) is None

    def gone(pid: int, sig: int) -> None:
        raise ProcessLookupError(pid)

    monkeypatch.setattr(bench_run, "pid_listening_on", lambda port: 999999)
    monkeypatch.setattr(bench_run.os, "kill", gone)
    assert bench_run.stop_orphan_server(1) is None


def test_stop_process_group_escalates_to_sigkill(bench: Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MODE", "ignore_term")
    monkeypatch.setattr(bench_run, "KILL_GRACE_S", 0.5)
    proc = subprocess.Popen([sys.executable, "fake_adapter.py", "--profile", "F", "--port", str(bench.port),
                             "--input", "cases.json", "--output", "x.json"], cwd=bench.root, start_new_session=True)
    # The fake installs SIG_IGN before it serves /health: wait for that, not a fixed sleep (a loaded
    # machine can take seconds to start Python).
    deadline = time.monotonic() + 60
    while not bench_run.server_healthy(f"http://127.0.0.1:{bench.port}") and time.monotonic() < deadline:
        time.sleep(0.05)
    bench_run.stop_process_group(proc)
    assert proc.returncode == -signal.SIGKILL
    bench_run.stop_process_group(proc)  # already gone: no error


def test_preflight_refusal_is_recorded_and_raised(bench: Bench, tmp_path: Path) -> None:
    bench.preflight = (str(write_script(tmp_path / "dirty.sh", 'echo "PREFLIGHT REFUSED"; exit 1\n')),)
    with pytest.raises(PreflightRefused):
        run_one(bench, "F")
    manifest = json.loads((bench.results_root / "public" / "F" / "run-1" / "run.json").read_text())
    assert manifest["status"] == "preflight_refused"


def test_run_matrix_revalidates_runs_that_finished_before_the_baseline(bench: Bench) -> None:
    specs = schedule(["B", "F"], 1, "public")  # F runs first, before B run-1 exists
    bench.run_timeout_s = None
    manifests = bench_run.run_matrix(bench, specs, resume=False)
    assert [m["profile"] for m in manifests] == ["F", "B"]
    assert bench.run_timeout_s == 600.0  # default: max(600 s, 2 min per case)
    f_manifest = bench_run.read_manifest(bench.results_root / "public" / "F" / "run-1")
    assert f_manifest is not None and f_manifest["validate"]["baseline_ok"] is True
    assert bench_run.run_matrix(bench, specs, resume=True) == []  # everything already complete
    assert bench_run.revalidate_with_baseline(bench, specs) == 0


def test_baseline_and_helpers(bench: Bench, tmp_path: Path) -> None:
    assert bench_run.baseline_for(bench, RunSpec("public", "B", 1, 1)) is None
    assert bench_run.baseline_for(bench, RunSpec("public", "F", 1, 1)) is None  # B run-1 not there yet
    assert bench.rel(tmp_path / "x") == "x" and bench.rel(Path("/elsewhere")) == "/elsewhere"
    assert bench_run._logged_case_id("[adapter] k_one: ['deny'] (1 ms, ok)") == "k_one"
    assert bench_run._logged_case_id("[adapter] wrote 2 case results to x") is None
    assert bench_run._logged_case_id("something else: [x]") is None
    assert not bench_run.results_complete(tmp_path)
    (tmp_path / "results.json").write_text("[]")
    assert not bench_run.results_complete(tmp_path)
    os.environ["LLAMA_ARG_CTX_SIZE"] = "1"
    os.environ["DESK_ALLOW_ENGINE_MISMATCH"] = "1"
    try:
        environment = bench_run.adapter_env()
        assert "LLAMA_ARG_CTX_SIZE" not in environment and "DESK_ALLOW_ENGINE_MISMATCH" not in environment
    finally:
        del os.environ["LLAMA_ARG_CTX_SIZE"]
        del os.environ["DESK_ALLOW_ENGINE_MISMATCH"]
    assert bench_run.default_python3()
    proc = bench_run.start_caffeinate()
    if shutil.which("caffeinate"):
        assert proc is not None
        proc.terminate()
        proc.wait(timeout=5)


def test_start_caffeinate_without_the_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench_run.shutil, "which", lambda name: None)
    assert bench_run.start_caffeinate() is None
    assert bench_run.default_python3() == sys.executable


# --------------------------------------------------------------------------------------------------
# CLI

def test_main_dry_run_and_usage_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert bench_run.main(["--profiles", "baseline", "final", "--runs", "1", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "adapter.py --profile F --port 8405 --strict-engine --input candidate_assets/public_cases.json" in out
    assert "--output results/public/B/run-1/results.json" in out
    assert bench_run.main(["--all-profiles", "--runs", "1", "--dry-run", "--suite", "variants"]) == 0
    assert "evals/variants/variants_v1.json" in capsys.readouterr().out
    assert bench_run.main(["--profiles", "nope"]) == 2
    assert bench_run.main(["--runs", "0"]) == 2
    assert bench_run.main(["--profiles", "B", "baseline"]) == 2  # the same profile twice


def test_main_refuses_a_suite_that_is_not_on_disk(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setitem(bench_run.SUITES, "holdout_v3", Suite("holdout_v3", "evals/none.json", "evals/none_labels.json"))
    assert bench_run.main(["--suite", "holdout_v3", "--floors-only"]) == 2
    assert "not on disk yet" in capsys.readouterr().err


def test_the_held_out_suites_are_registered() -> None:
    assert SUITES["holdout_v2"].available() and SUITES["holdout_v2"].labels_path == "evals/holdout/labels_v2.json"
    assert SUITES["holdout_v3"].input_path == "evals/holdout_v3/holdout_v3.json"


class FakeCaffeinate:
    terminated = False

    def terminate(self) -> None:
        FakeCaffeinate.terminated = True


@pytest.mark.parametrize(("outcome", "code"), [
    ("ok", 0), ("failed_run", 1), ("bad_structure", 1), ("refused", 3), ("exists", 2)])
def test_main_exit_codes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcome: str, code: int) -> None:
    seen: dict[str, Any] = {}

    def fake_matrix(bench: Bench, specs: list[RunSpec], resume: bool) -> list[dict[str, Any]]:
        seen.update(bench=bench, specs=specs, resume=resume)
        if outcome == "refused":
            raise PreflightRefused("FAIL  cpu")
        if outcome == "exists":
            raise RunExists("already complete")
        status = "no_results" if outcome == "failed_run" else "complete"
        return [{"status": status, "validate": {"structure_ok": outcome != "bad_structure"}}]

    monkeypatch.setattr(bench_run, "run_matrix", fake_matrix)
    monkeypatch.setattr(bench_run, "run_floors", lambda bench: [])
    monkeypatch.setattr(bench_run, "verify_models_fresh", lambda bench, names: tmp_path / "model_sha256.json")
    monkeypatch.setattr(bench_run, "start_caffeinate", lambda: FakeCaffeinate())
    FakeCaffeinate.terminated = False
    argv = ["--suite", "variants", "--profiles", "B", "F", "--runs", "1", "--results", str(tmp_path),
            "--allow-dirty", "--preflight-wait-min", "2", "--run-timeout-min", "3", "--resume", "--port", "8409"]
    assert bench_run.main(argv) == code
    assert FakeCaffeinate.terminated
    bench = seen["bench"]
    assert bench.suite == SUITES["variants"] and bench.port == 8409 and bench.allow_dirty
    assert bench.preflight_wait_s == 120 and bench.run_timeout_s == 180 and seen["resume"] is True


def test_main_without_caffeinate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bench_run, "run_matrix", lambda bench, specs, resume: [])
    monkeypatch.setattr(bench_run, "run_floors", lambda bench: [])
    monkeypatch.setattr(bench_run, "verify_models_fresh", lambda bench, names: tmp_path / "model_sha256.json")
    monkeypatch.setattr(bench_run, "start_caffeinate", lambda: None)
    assert bench_run.main(["--runs", "1", "--results", str(tmp_path)]) == 0


def test_main_runs_the_floors_first_and_stops_on_a_bad_model_hash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                                 capsys: pytest.CaptureFixture[str]) -> None:
    order: list[str] = []
    monkeypatch.setattr(bench_run, "run_floors", lambda bench: order.append("floors") or [
        {"mode": "always-clarify", "exit": 0, "score": {"strict_success": 4, "cases": 6}}])

    def bad_hash(bench: Bench, names: list[str]) -> Path:
        order.append("hash")
        raise ModelCheckFailed("sha256 of Qwen3.5-4B-Q4_K_M.gguf is not the pinned value")

    monkeypatch.setattr(bench_run, "verify_models_fresh", bad_hash)
    monkeypatch.setattr(bench_run, "run_matrix", lambda *a: pytest.fail("must not run models"))
    assert bench_run.main(["--runs", "1", "--results", str(tmp_path)]) == 3
    captured = capsys.readouterr()
    assert order == ["floors", "hash"] and "floor always-clarify: exit 0, strict 4/6" in captured.out
    assert "not the pinned value" in captured.err
    order.clear()
    assert bench_run.main(["--runs", "1", "--results", str(tmp_path), "--floors-only"]) == 0
    assert order == ["floors"]


# --------------------------------------------------------------------------------------------------
# Once per bench: fresh model hashes and code-only floors

def test_verify_models_fresh_hashes_every_byte_and_records_it(bench: Bench, tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib
    bench.models_dir = tmp_path / "models"
    bench.models_dir.mkdir()
    weights = {get_profile("F").model_file: b"q4 weights", get_profile("B").model_file: b"q8 weights"}
    for name, data in weights.items():
        (bench.models_dir / name).write_bytes(data)
    pins = {name: hashlib.sha256(data).hexdigest() for name, data in weights.items()}
    monkeypatch.setattr("desk.profiles.MODEL_SHA256", pins)
    record = bench_run.verify_models_fresh(bench, ["B", "F", "F-q8"])
    checks = json.loads(record.read_text())["models"]
    assert [(c["model_file"], c["match"], c["source"]) for c in checks] == [
        (get_profile("F").model_file, True, "fresh"), (get_profile("B").model_file, True, "fresh")]
    assert record.name == "model_sha256.json"
    (bench.models_dir / get_profile("F").model_file).write_bytes(b"tampered")
    with pytest.raises(ModelCheckFailed, match="not the pinned value"):
        bench_run.verify_models_fresh(bench, ["F"])
    assert (bench.suite_dir / "model_sha256-2.json").exists()  # every check is kept
    (bench.models_dir / get_profile("F").model_file).unlink()
    with pytest.raises(ModelCheckFailed, match="model file missing"):
        bench_run.verify_models_fresh(bench, ["F"])


def test_run_floors_scores_both_code_only_modes(bench: Bench) -> None:
    records = bench_run.run_floors(bench)
    assert [r["mode"] for r in records] == ["always-clarify", "fake-policy"]
    for record in records:
        out_dir = bench.suite_dir / "floors" / record["mode"]
        assert record["exit"] == 0 and record["score"]["cases"] == 2
        assert {"results.json", "score.json", "score.txt", "floor.json", "stub.log"} <= {p.name for p in out_dir.iterdir()}
        assert f"run_stub --mode {record['mode']} --input cases.json" in record["command"]
        assert bench_run.floor_line(record).startswith(f"floor {record['mode']}: exit 0, strict ")
        assert record["code_sha256"] == code_fingerprint(bench.root)


def test_the_code_fingerprint_follows_the_decision_code_only(tmp_path: Path) -> None:
    (tmp_path / "desk" / "agents").mkdir(parents=True)
    (tmp_path / "desk" / "policy.py").write_text("WINDOW_DAYS = 30\n")
    (tmp_path / "desk" / "agents" / "prompts.py").write_text("SYSTEM = 'x'\n")
    (tmp_path / "adapter.py").write_text("print('shim')\n")
    (tmp_path / "server").mkdir()
    first = code_fingerprint(tmp_path)
    assert first == code_fingerprint(tmp_path) and len(first) == 64
    (tmp_path / "server" / "app.py").write_text("# the UI does not decide turns\n")
    (tmp_path / "desk" / "notes.txt").write_text("not code\n")
    assert code_fingerprint(tmp_path) == first
    for changed in ("desk/agents/prompts.py", "adapter.py"):
        (tmp_path / changed).write_text((tmp_path / changed).read_text() + "# edit\n")
        assert code_fingerprint(tmp_path) != first
        first = code_fingerprint(tmp_path)
    (tmp_path / "desk" / "policy.py").rename(tmp_path / "desk" / "rules.py")  # same bytes, new path
    assert code_fingerprint(tmp_path) != first


def test_a_floor_that_crashes_is_recorded_without_a_score(bench: Bench) -> None:
    bench.tool_python = str(write_script(bench.root / "broken_python", "exit 4\n"))
    records = bench_run.run_floors(bench)
    assert all(r["exit"] == 4 and "score" not in r for r in records)
    assert bench_run.floor_line(records[0]) == "floor always-clarify: exit 4, strict None/None"


# --------------------------------------------------------------------------------------------------
# Shell scripts

PREFLIGHT = str(ROOT / "bench" / "preflight.sh")
PERMISSIVE = {"BENCH_MAX_LOAD": "1000", "BENCH_GPU_BUSY_PCT": "100", "BENCH_GPU_MEM_MB": "10000000",
              "BENCH_CPU_HOG_PCT": "100000"}


def preflight(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", PREFLIGHT, *args], capture_output=True, text=True, timeout=60,
                          env={**os.environ, **(env or {})}, check=False)


def test_preflight_usage_error() -> None:
    done = preflight("--bogus")
    assert done.returncode == 2 and "usage" in done.stderr


def test_preflight_names_cpu_hogs_and_saves_raw_readings(tmp_path: Path) -> None:
    done = preflight("--out", str(tmp_path / "pf"), env={**PERMISSIVE, "BENCH_CPU_HOG_PCT": "0"})
    assert done.returncode == 1 and "FAIL  cpu:" in done.stdout and "PREFLIGHT REFUSED" in done.stdout
    saved = {p.name for p in (tmp_path / "pf").iterdir()}
    assert {"ps.txt", "pmset.txt", "pmset_batt.txt", "pmset_therm.txt", "gpu_samples.txt", "uptime.txt",
            "verdict.txt"} <= saved
    ignored = preflight(env={**PERMISSIVE, "BENCH_CPU_HOG_PCT": "0", "BENCH_IGNORE_REGEX": "."})
    assert "PASS  cpu:" in ignored.stdout


def test_preflight_refuses_a_foreign_llama_server_and_a_busy_port(tmp_path: Path) -> None:
    # /bin/sleep running under the name .../llama-server (argv[0]); ps shows argv, as for a real server.
    fake = tmp_path / "llama-server"
    sleeper = subprocess.Popen([str(fake), "30"], executable="/bin/sleep")
    time.sleep(0.2)
    port = free_port()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", port))
        listener.listen()
        try:
            done = preflight("--port", str(port), env=PERMISSIVE)
        finally:
            sleeper.kill()
            sleeper.wait()
    assert done.returncode == 1
    # Other llama-servers may be running on a shared machine too; ours must be among those named.
    assert "FAIL  llama-server:" in done.stdout and "foreign process(es) running" in done.stdout
    assert f"{sleeper.pid} " in done.stdout and str(tmp_path) not in done.stdout  # the name only, no path or args
    assert f"FAIL  port {port}: already in use" in done.stdout


def test_preflight_execs_the_command_under_caffeinate_when_clean() -> None:
    probe = preflight(env=PERMISSIVE)
    if probe.returncode != 0:
        pytest.skip("this machine is not clean right now (battery, Low Power Mode, thermal or a llama-server)")
    done = preflight("--", sys.executable, "-c", "print('ran')", env=PERMISSIVE)
    assert done.returncode == 0 and done.stdout.strip().endswith("ran")


def copy_reset(tmp_path: Path) -> Path:
    (tmp_path / "scripts").mkdir()
    script = tmp_path / "scripts" / "reset.sh"
    shutil.copyfile(ROOT / "scripts" / "reset.sh", script)
    for rel in ("var/ui/sessions/S-1-g1.sqlite", "var/ui/llama-server-8410.log", "var/adapter/server-F-8405.log",
                "var/tuning/keep.json", "results/public/F/run-1/results.json", "elsewhere/sessions/S-2-g1.sqlite"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    return script


def reset(script: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    environ = {k: v for k, v in os.environ.items() if k != "DESK_VAR_DIR"}
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True, check=False,
                          env={**environ, **(env or {})})


def test_reset_removes_run_state_and_keeps_evidence(tmp_path: Path) -> None:
    script = copy_reset(tmp_path)
    dry = reset(script, "--dry-run", "--force")  # --force: a real UI server may be running right now
    assert dry.returncode == 0 and "would remove var/ui (2 file(s))" in dry.stdout
    assert "dry run: nothing was removed" in dry.stdout and (tmp_path / "var/ui/sessions/S-1-g1.sqlite").exists()
    done = reset(script, "--force")
    assert done.returncode == 0 and "removed  var/adapter (1 file(s))" in done.stdout and "reset done" in done.stdout
    assert not (tmp_path / "var/ui").exists() and not (tmp_path / "var/adapter").exists()
    assert (tmp_path / "var/tuning/keep.json").exists() and (tmp_path / "results/public/F/run-1/results.json").exists()
    assert "clean    var/ui (nothing there)" in reset(script, "--force").stdout
    custom = reset(script, "--force", env={"DESK_VAR_DIR": "elsewhere"})
    assert "removed  elsewhere (1 file(s))" in custom.stdout and not (tmp_path / "elsewhere").exists()
    assert reset(script, "--bogus").returncode == 2


def test_reset_refuses_while_a_benchmark_runs(tmp_path: Path) -> None:
    script = copy_reset(tmp_path)
    busy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "-m", "bench.run"])
    try:
        time.sleep(0.3)
        done = reset(script)
    finally:
        busy.kill()
        busy.wait()
    assert done.returncode == 1 and "reset refused" in done.stdout and str(busy.pid) in done.stdout
    assert (tmp_path / "var/ui/sessions/S-1-g1.sqlite").exists()


def test_stop_process_group_gives_up_after_sigkill(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process that outlives SIGKILL (stuck in the kernel) must not hang the bench."""
    sent: list[int] = []

    class Stuck:
        pid = 424242

        def wait(self, timeout: float) -> int:
            raise subprocess.TimeoutExpired("stuck", timeout)

    monkeypatch.setattr(bench_run.os, "killpg", lambda pid, sig: sent.append(sig))
    monkeypatch.setattr(bench_run, "KILL_GRACE_S", 0.01)
    bench_run.stop_process_group(Stuck())  # type: ignore[arg-type]
    assert sent == [signal.SIGTERM, signal.SIGKILL]
