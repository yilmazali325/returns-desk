"""Benchmark runner (SPEC §9): profiles x runs in randomized interleaved blocks, one fresh adapter
process (which starts its own fresh llama-server) per run, every raw file kept.

    .venv/bin/python -m bench.run --suite public --profiles B F F-q8 --runs 3 [--port 8405]
                                  [--resume] [--allow-dirty] [--preflight-wait-min 30] [--dry-run]
    .venv/bin/python -m bench.run --suite holdout_v2 --floors-only     (code-only floors, no model)

Why it is built this way:
- Same entry point as the grader. Each run is literally `python3 adapter.py --profile P --port N
  --input <cases> --output results/<suite>/<P>/run-<k>/results.json`, so what is measured is the
  contract code path, including cold start, the sha256/engine checks and the memory probes.
- Interleaved blocks. Block k runs every profile once, in an order shuffled by random.Random(42),
  so slow drift (heat, background load) spreads over all profiles instead of hitting whichever ran
  last. Run k of every profile therefore ran in the same block, which is what report.py pairs on.
  The whole schedule is written to schedule.json before the first run starts (pre-registered).
- A clean machine. bench/preflight.sh runs before every run and refuses foreign llama-servers,
  CPU/GPU hogs, battery power, Low Power Mode and thermal limits; --allow-dirty records the
  override instead, and report.py flags those timings. The runner keeps `caffeinate` attached to
  itself so the machine cannot sleep mid-matrix.
- Nothing is overwritten. An incomplete run directory is renamed to run-<k>.attempt-<n> and kept,
  and every attempt is appended to attempts.jsonl, so failed runs stay visible in the report.
- /metrics is scraped twice per run and never polled: right after the server first answers
  /health (metrics_before.prom) and right after the adapter logs its last case
  (metrics_after.prom), while the adapter is still collecting its after-run conditions and the
  server is still up. Polling during cases would compete with decoding for the server's task loop.
- Strict engine, fresh hashes. Every adapter run gets --strict-engine (a different llama.cpp build
  is refused, never just warned about), and each model file is hashed in full once per bench
  invocation (model_sha256*.json); the runs then read that just-written digest from the cache.
- Code-only floors. Before the model runs, the suite also goes through evals/holdout/run_stub.py
  with no model (floors/<mode>/), so the report can show what the agents add on top of code alone.

Files per run directory: results.json, adapter.log, server.log, metrics_before.prom,
metrics_after.prom, validate.txt, score.txt, score.json, run_conditions.json, argv.txt,
preflight.txt, preflight/ (raw readings) and run.json (the manifest report.py starts from).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

import httpx

from bench import BASELINE, FLOOR_MODES, RESULTS_DIR, ROOT, SUITES, Suite, code_fingerprint
from desk import telemetry
from desk.llm_server import MODELS_DIR, file_sha256, pid_listening_on, server_healthy
from desk.profiles import PROFILES, get_profile

SCHEDULE_SEED = 42  # pre-registered (SPEC §9); not a CLI option, so the order cannot be re-rolled
DEFAULT_PORT = 8405
DEFAULT_PROFILES = ("B", "F")
HEALTH_POLL_S = 0.05
SERVER_READY_TIMEOUT_S = 240.0
KILL_GRACE_S = 15.0
RESTART_MARK = "restarting it for the next case"  # adapter_main logs this when it restarts the server
EXIT_OK, EXIT_RUN_FAILED, EXIT_USAGE, EXIT_REFUSED = 0, 1, 2, 3
COMPLETE = "complete"


class PreflightRefused(RuntimeError):
    """The machine was not clean and --allow-dirty was not given."""


class RunExists(RuntimeError):
    """A complete run directory is already there and --resume was not given."""


class ModelCheckFailed(RuntimeError):
    """A model file is missing or its fresh sha256 is not the pinned one."""


@dataclass(frozen=True)
class RunSpec:
    """One scheduled run: profile P, run number k (= its block) and its place inside the block."""

    suite: str
    profile: str
    run: int
    position: int

    def directory(self, results_root: Path) -> Path:
        return results_root / self.suite / self.profile / f"run-{self.run}"


def default_python3() -> str:
    """The grader types `python3 adapter.py`; use that interpreter (the shim re-execs into .venv)."""
    return shutil.which("python3") or sys.executable


@dataclass
class Bench:
    """Where everything is and which commands to call. Tests point these at small fakes."""

    suite: Suite
    port: int = DEFAULT_PORT
    results_root: Path = RESULTS_DIR
    root: Path = ROOT
    python3: str = field(default_factory=default_python3)
    adapter: str = "adapter.py"
    validator: str = "candidate_assets/validate_submission.py"
    tool_python: str = sys.executable  # runs the validator and the scorer
    preflight: tuple[str, ...] = ("bash", "bench/preflight.sh")
    server_log_dir: Path = ROOT / "var" / "adapter"
    allow_dirty: bool = False
    preflight_wait_s: float = 0.0
    preflight_poll_s: float = 30.0
    run_timeout_s: float | None = None
    ready_timeout_s: float = SERVER_READY_TIMEOUT_S
    models_dir: Path = MODELS_DIR

    @property
    def suite_dir(self) -> Path:
        return self.results_root / self.suite.name

    def rel(self, path: Path) -> str:
        """Repo-relative spelling when possible, so argv.txt reads like the grader command."""
        try:
            return str(Path(path).resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(path)


# --------------------------------------------------------------------------------------------------
# Schedule

def schedule(profiles: list[str], runs: int, suite: str, seed: int = SCHEDULE_SEED) -> list[RunSpec]:
    """Randomized interleaved blocks: block k = every profile once, in a seeded shuffled order."""
    rng = random.Random(seed)
    specs: list[RunSpec] = []
    for block in range(1, runs + 1):
        order = list(profiles)
        rng.shuffle(order)
        specs += [RunSpec(suite, name, block, position) for position, name in enumerate(order, start=1)]
    return specs


def write_schedule(bench: Bench, specs: list[RunSpec]) -> Path:
    """Record the order before anything runs (a new file per matrix invocation; never overwritten)."""
    bench.suite_dir.mkdir(parents=True, exist_ok=True)
    path = _free_path(bench.suite_dir / "schedule.json")
    document = {"seed": SCHEDULE_SEED, "method": "random.Random(seed).shuffle per block, one block per run number",
                "written_at": _now(), "suite": bench.suite.name, "port": bench.port,
                "runs": [asdict(spec) for spec in specs]}
    path.write_text(json.dumps(document, indent=1) + "\n")
    return path


def _free_path(path: Path) -> Path:
    if not path.exists():
        return path
    n = 2
    while path.with_name(f"{path.stem}-{n}{path.suffix}").exists():
        n += 1
    return path.with_name(f"{path.stem}-{n}{path.suffix}")


# --------------------------------------------------------------------------------------------------
# Run directories

def read_manifest(run_dir: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((run_dir / "run.json").read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def is_complete(run_dir: Path) -> bool:
    manifest = read_manifest(run_dir)
    return bool(manifest and manifest.get("status") == COMPLETE)


def prepare_run_dir(run_dir: Path, resume: bool) -> bool:
    """True when the run must execute. A complete run is skipped (--resume) or refused; anything
    else already there is an earlier attempt: renamed to run-<k>.attempt-<n>, never deleted."""
    if is_complete(run_dir):
        if resume:
            return False
        raise RunExists(f"{run_dir} is already complete; pass --resume to skip finished runs "
                        "(nothing is ever overwritten)")
    if run_dir.exists():
        n = 1
        while run_dir.with_name(f"{run_dir.name}.attempt-{n}").exists():
            n += 1
        run_dir.rename(run_dir.with_name(f"{run_dir.name}.attempt-{n}"))
    run_dir.mkdir(parents=True)
    return True


# --------------------------------------------------------------------------------------------------
# Preflight

@dataclass
class Preflight:
    passed: bool
    overridden: bool
    waited_s: float
    verdict: str


def run_preflight(bench: Bench, run_dir: Path) -> Preflight:
    """Run bench/preflight.sh until it passes, waiting up to preflight_wait_s for a clean machine."""
    waited = 0.0
    while True:
        argv = [*bench.preflight, "--out", str(run_dir / "preflight"), "--port", str(bench.port)]
        done = subprocess.run(argv, cwd=bench.root, capture_output=True, text=True, check=False)
        text = done.stdout + done.stderr
        (run_dir / "preflight.txt").write_text(f"$ {shlex.join(argv)}\n{text}")
        verdict = _last_line(text)
        if done.returncode == 0:
            return Preflight(True, False, waited, verdict)
        if done.returncode != 1:
            raise RuntimeError(f"preflight could not run (exit {done.returncode}): {verdict}")
        if bench.allow_dirty:
            return Preflight(False, True, waited, verdict)
        if waited >= bench.preflight_wait_s:
            raise PreflightRefused(text)
        print(f"[bench] machine not clean, waiting {bench.preflight_poll_s:.0f} s: {verdict}", flush=True)
        time.sleep(bench.preflight_poll_s)
        waited += bench.preflight_poll_s


def _last_line(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


# --------------------------------------------------------------------------------------------------
# Watching one adapter process

class ServerWatcher:
    """Scrapes the adapter's llama-server /metrics at the two moments described in the docstring."""

    def __init__(self, base_url: str, run_dir: Path, case_ids: list[str], ready_timeout_s: float) -> None:
        self.base_url = base_url
        self.run_dir = run_dir
        self.case_ids = set(case_ids)
        self.expected_cases = len(case_ids)
        self.ready_timeout_s = ready_timeout_s
        self.cases_logged = 0
        self.restarts = 0
        self.before = "not captured: the server never answered /health"
        self.after = "not captured: the adapter did not log every case"
        self.adapter_done = threading.Event()

    def wait_ready_then_scrape(self) -> None:
        """Thread target: poll /health until ready (or the adapter exits), then scrape once."""
        deadline = time.monotonic() + self.ready_timeout_s
        while not self.adapter_done.is_set() and time.monotonic() < deadline:
            if server_healthy(self.base_url, timeout_s=0.5):
                self.before = self.scrape("metrics_before.prom", "first /health ok, before the first case")
                return
            time.sleep(HEALTH_POLL_S)

    def on_log_line(self, line: str) -> None:
        """Called for every adapter log line: count finished cases, notice server restarts."""
        if RESTART_MARK in line:
            self.restarts += 1
        if _logged_case_id(line) in self.case_ids:
            self.cases_logged += 1
            if self.cases_logged == self.expected_cases:
                self.after = self.scrape("metrics_after.prom", "the adapter logged its last case")

    def scrape(self, name: str, moment: str) -> str:
        try:
            response = httpx.get(f"{self.base_url}/metrics", timeout=5)
        except httpx.HTTPError as exc:
            return f"not captured: /metrics unreachable ({type(exc).__name__}) when {moment}"
        if response.status_code != 200:
            return f"not captured: /metrics answered HTTP {response.status_code} when {moment}"
        header = f"# scraped_at_unix: {time.time():.3f}\n# moment: {moment}\n"
        (self.run_dir / name).write_text(header + response.text)
        return f"captured: {moment}"


def _logged_case_id(line: str) -> str | None:
    """adapter_main logs '[adapter] <case_id>: [decisions] (...)' after each case."""
    prefix = "[adapter] "
    if not line.startswith(prefix) or ": [" not in line:
        return None
    return line[len(prefix):].split(": [", 1)[0]


def pump_log(stream: IO[str], log_path: Path, watcher: ServerWatcher) -> None:
    """Thread target: copy the adapter's output to adapter.log and feed it to the watcher."""
    with log_path.open("w") as log:
        for line in stream:
            log.write(line)
            log.flush()
            watcher.on_log_line(line.rstrip("\n"))


def stop_process_group(proc: subprocess.Popen[str]) -> None:
    """SIGTERM the adapter's process group (adapter + its llama-server), then SIGKILL if needed."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=KILL_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            continue


def stop_orphan_server(port: int) -> int | None:
    """After a killed run: llama-server runs in its own session, so killing the adapter's group does
    not reach it. Preflight proved the port free before the run, so a listener now is this run's."""
    pid = pid_listening_on(port)
    if pid is None or pid == os.getpid():
        return None
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return None
    return pid


@dataclass
class AdapterOutcome:
    exit_code: int | None
    timed_out: bool
    wall_s: float
    metrics_before: str
    metrics_after: str
    cases_logged: int
    server_restarts: int
    orphan_server_killed: int | None = None


def run_adapter(bench: Bench, argv: list[str], run_dir: Path, case_ids: list[str]) -> AdapterOutcome:
    """Start the adapter in its own process group, watch it, enforce the run timeout."""
    watcher = ServerWatcher(f"http://127.0.0.1:{bench.port}", run_dir, case_ids, bench.ready_timeout_s)
    started = time.monotonic()
    proc = subprocess.Popen(argv, cwd=bench.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1, start_new_session=True, env=adapter_env())
    readers = [threading.Thread(target=pump_log, args=(proc.stdout, run_dir / "adapter.log", watcher), daemon=True),
               threading.Thread(target=watcher.wait_ready_then_scrape, daemon=True)]
    for thread in readers:
        thread.start()
    timed_out = False
    orphan = None
    try:
        proc.wait(timeout=bench.run_timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        stop_process_group(proc)
        orphan = stop_orphan_server(bench.port)
    watcher.adapter_done.set()
    for thread in readers:
        thread.join(timeout=10)
    return AdapterOutcome(proc.returncode, timed_out, round(time.monotonic() - started, 3), watcher.before,
                          watcher.after, watcher.cases_logged, watcher.restarts, orphan)


def adapter_env() -> dict[str, str]:
    """The inherited environment minus LLAMA_ARG_* (they would silently override profile flags) and
    minus DESK_ALLOW_ENGINE_MISMATCH (the bench never measures an unpinned engine build)."""
    return {k: v for k, v in os.environ.items()
            if not k.startswith("LLAMA_ARG_") and k != "DESK_ALLOW_ENGINE_MISMATCH"}


# --------------------------------------------------------------------------------------------------
# After the adapter: server log, validation, scoring

def copy_server_logs(bench: Bench, spec: RunSpec, run_dir: Path, since: float) -> str:
    """The adapter writes var/adapter/server-<P>-<port>[-restartN].log; copy this run's files."""
    pattern = f"server-{spec.profile}-{bench.port}*.log"
    fresh = sorted(p for p in bench.server_log_dir.glob(pattern) if p.stat().st_mtime >= since - 1)
    if not fresh:
        return f"unavailable: no {pattern} written in {bench.rel(bench.server_log_dir)} during this run"
    for source in fresh:
        restart = source.stem.rsplit("-restart", 1)
        target = f"server-restart{restart[1]}.log" if len(restart) == 2 else "server.log"
        shutil.copyfile(source, run_dir / target)
    return f"copied {len(fresh)} file(s) from {bench.rel(bench.server_log_dir)}"


def baseline_for(bench: Bench, spec: RunSpec) -> Path | None:
    """B run-1 is the reference every other run of the suite is checked against (--baseline)."""
    reference = RunSpec(spec.suite, BASELINE, 1, 0).directory(bench.results_root)
    if spec.profile == BASELINE and spec.run == 1:
        return None
    path = reference / "results.json"
    return path if is_complete(reference) and path.exists() else None


def validate(bench: Bench, run_dir: Path, baseline: Path | None) -> dict[str, Any]:
    """validate_submission.py (structure) and, when B run-1 exists, its --baseline consistency check."""
    base = [bench.tool_python, bench.validator, "--input", bench.suite.input_path,
            "--output", bench.rel(run_dir / "results.json")]
    commands = [base] + ([[*base, "--baseline", bench.rel(baseline)]] if baseline else [])
    lines: list[str] = []
    codes: list[int] = []
    for argv in commands:
        done = subprocess.run(argv, cwd=bench.root, capture_output=True, text=True, check=False)
        codes.append(done.returncode)
        lines += [f"$ {shlex.join(argv)}", (done.stdout + done.stderr).rstrip(), f"exit {done.returncode}", ""]
    (run_dir / "validate.txt").write_text("\n".join(lines))
    return {"structure_ok": codes[0] == 0, "baseline": bench.rel(baseline) if baseline else None,
            "baseline_ok": (codes[1] == 0) if baseline else None}


def score(bench: Bench, run_dir: Path) -> dict[str, Any]:
    """The independent oracle scorer: score.txt (its text) and score.json (its full report)."""
    argv = [bench.tool_python, "-m", "evals.score", "--input", bench.suite.input_path,
            "--labels", bench.suite.labels_path, "--output", bench.rel(run_dir / "results.json"),
            "--json", bench.rel(run_dir / "score.json")]
    done = subprocess.run(argv, cwd=bench.root, capture_output=True, text=True, check=False)
    (run_dir / "score.txt").write_text(f"$ {shlex.join(argv)}\n{done.stdout}{done.stderr}")
    try:
        summary = json.loads((run_dir / "score.json").read_text())["summary"]
    except (OSError, ValueError, KeyError):
        return {"exit": done.returncode, "strict_success": None, "cases": None}
    return {"exit": done.returncode, "strict_success": summary["strict_success"], "cases": summary["cases"],
            "invariant_violations": summary["invariant_violations"]}


# --------------------------------------------------------------------------------------------------
# One run

def adapter_argv(bench: Bench, spec: RunSpec) -> list[str]:
    output = bench.rel(spec.directory(bench.results_root) / "results.json")
    return [bench.python3, bench.adapter, "--profile", spec.profile, "--port", str(bench.port), "--strict-engine",
            "--input", bench.suite.input_path, "--output", output]


def results_complete(run_dir: Path) -> bool:
    try:
        return isinstance(json.loads((run_dir / "results.json").read_text()).get("results"), list)
    except (OSError, ValueError, AttributeError):
        return False


def execute_run(bench: Bench, spec: RunSpec, case_ids: list[str], resume: bool = False) -> dict[str, Any] | None:
    """Preflight -> adapter -> server log -> validate -> score -> manifest. None when skipped."""
    run_dir = spec.directory(bench.results_root)
    if not prepare_run_dir(run_dir, resume):
        return None
    manifest: dict[str, Any] = {**asdict(spec), "status": "preflight", "port": bench.port, "started_at": _now(),
                                "input": bench.suite.input_path, "labels": bench.suite.labels_path,
                                "code_sha256": code_fingerprint(bench.root)}
    _write_json(run_dir / "run.json", manifest)
    try:
        preflight = run_preflight(bench, run_dir)
    except PreflightRefused:
        _finish(bench, run_dir, manifest, status="preflight_refused")
        raise
    manifest["preflight"] = asdict(preflight)
    argv = adapter_argv(bench, spec)
    (run_dir / "argv.txt").write_text(argv_record(bench, argv))
    before = telemetry.run_conditions()
    since = time.time()
    outcome = run_adapter(bench, argv, run_dir, case_ids)
    after = telemetry.run_conditions()
    _write_json(run_dir / "run_conditions.json", {"preflight": asdict(preflight), "before_run": before,
                                                  "after_run": after, "caffeinate": "attached by bench.run"})
    manifest["adapter"] = asdict(outcome)
    manifest["server_log"] = copy_server_logs(bench, spec, run_dir, since)
    if not results_complete(run_dir):
        status = "timed_out" if outcome.timed_out else "no_results"
        return _finish(bench, run_dir, manifest, status=status)
    manifest["validate"] = validate(bench, run_dir, baseline_for(bench, spec))
    manifest["score"] = score(bench, run_dir)
    return _finish(bench, run_dir, manifest, status=COMPLETE)


def argv_record(bench: Bench, argv: list[str]) -> str:
    version = subprocess.run([bench.python3, "--version"], capture_output=True, text=True, check=False)
    return (f"{shlex.join(argv)}\n# cwd: {bench.root}\n# python3: {bench.python3} "
            f"({(version.stdout + version.stderr).strip()}); adapter.py re-execs into .venv/bin/python\n"
            "# environment: inherited, minus LLAMA_ARG_* variables\n")


def _finish(bench: Bench, run_dir: Path, manifest: dict[str, Any], status: str) -> dict[str, Any]:
    manifest.update(status=status, ended_at=_now())
    _write_json(run_dir / "run.json", manifest)
    attempt = {k: manifest.get(k) for k in ("suite", "profile", "run", "position", "status", "started_at", "ended_at")}
    attempt["dir"] = bench.rel(run_dir)
    attempt["wall_s"] = (manifest.get("adapter") or {}).get("wall_s")
    with (bench.suite_dir / "attempts.jsonl").open("a") as handle:
        handle.write(json.dumps(attempt) + "\n")
    return manifest


def _write_json(path: Path, document: Any) -> None:
    path.write_text(json.dumps(document, indent=1) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------------------------------
# Once per bench: fresh model hashes and the code-only floors

def verify_models_fresh(bench: Bench, profiles: list[str]) -> Path:
    """Hash every model file the profiles use in full (never from the cache) and compare with the pin.
    Writes <suite>/model_sha256.json; raises ModelCheckFailed on a missing file or a mismatch."""
    checks = []
    for model_file, expected in sorted({(get_profile(p).model_file, get_profile(p).model_sha256) for p in profiles}):
        path = bench.models_dir / model_file
        if not path.is_file():
            raise ModelCheckFailed(f"model file missing: {bench.rel(path)} (run scripts/setup.sh)")
        reading = file_sha256(path, fresh=True)
        checks.append({"model_file": model_file, "sha256": reading.value, "expected": expected,
                       "match": reading.value == expected, "source": reading.source})
    bench.suite_dir.mkdir(parents=True, exist_ok=True)
    record = _free_path(bench.suite_dir / "model_sha256.json")
    _write_json(record, {"hashed_at": _now(), "models": checks})
    bad = [c["model_file"] for c in checks if not c["match"]]
    if bad:
        raise ModelCheckFailed(f"sha256 of {', '.join(bad)} is not the pinned value (see {bench.rel(record)})")
    return record


def floor_dir(bench: Bench, mode: str) -> Path:
    return bench.suite_dir / "floors" / mode


def run_floors(bench: Bench) -> list[dict[str, Any]]:
    """The suite through the real desk with no model (evals/holdout/run_stub.py), scored by the oracle."""
    records = []
    for mode in FLOOR_MODES:
        out_dir = floor_dir(bench, mode)
        out_dir.mkdir(parents=True, exist_ok=True)
        argv = [bench.tool_python, "-m", "evals.holdout.run_stub", "--mode", mode, "--input", bench.suite.input_path,
                "--output", bench.rel(out_dir / "results.json")]
        done = subprocess.run(argv, cwd=bench.root, capture_output=True, text=True, check=False)
        (out_dir / "stub.log").write_text(f"$ {shlex.join(argv)}\n{done.stdout}{done.stderr}")
        record = {"suite": bench.suite.name, "mode": mode, "input": bench.suite.input_path,
                  "labels": bench.suite.labels_path, "command": shlex.join(argv), "exit": done.returncode,
                  "written_at": _now(), "code_sha256": code_fingerprint(bench.root)}
        if done.returncode == 0:
            record["score"] = score(bench, out_dir)
        _write_json(out_dir / "floor.json", record)
        records.append(record)
    return records


def floor_line(record: dict[str, Any]) -> str:
    result = record.get("score") or {}
    return f"floor {record['mode']}: exit {record['exit']}, strict {result.get('strict_success')}/{result.get('cases')}"


# --------------------------------------------------------------------------------------------------
# The matrix

def revalidate_with_baseline(bench: Bench, specs: list[RunSpec]) -> int:
    """Runs that finished before B run-1 existed get their --baseline check now. Returns how many."""
    redone = 0
    for spec in specs:
        run_dir = spec.directory(bench.results_root)
        manifest = read_manifest(run_dir)
        baseline = baseline_for(bench, spec)
        if manifest is None or manifest.get("status") != COMPLETE or baseline is None:
            continue
        if manifest.get("validate", {}).get("baseline") is None:
            manifest["validate"] = validate(bench, run_dir, baseline)
            _write_json(run_dir / "run.json", manifest)
            redone += 1
    return redone


def run_matrix(bench: Bench, specs: list[RunSpec], resume: bool) -> list[dict[str, Any]]:
    case_ids = [case["case_id"] for case in json.loads((bench.root / bench.suite.input_path).read_text())["cases"]]
    if bench.run_timeout_s is None:
        bench.run_timeout_s = max(600.0, 120.0 * len(case_ids))
    print(f"[bench] schedule written to {bench.rel(write_schedule(bench, specs))}", flush=True)
    manifests = []
    for number, spec in enumerate(specs, start=1):
        print(f"[bench] {number}/{len(specs)}: {spec.suite} {spec.profile} run-{spec.run} "
              f"(block {spec.run}, position {spec.position})", flush=True)
        manifest = execute_run(bench, spec, case_ids, resume)
        if manifest is None:
            print("[bench]   already complete, skipped (--resume)", flush=True)
            continue
        manifests.append(manifest)
        print(f"[bench]   {summary_line(manifest)}", flush=True)
    redone = revalidate_with_baseline(bench, specs)
    if redone:
        print(f"[bench] re-validated {redone} run(s) against B run-1", flush=True)
    return manifests


def summary_line(manifest: dict[str, Any]) -> str:
    result = manifest.get("score") or {}
    adapter = manifest.get("adapter") or {}
    checks = manifest.get("validate") or {}
    return (f"{manifest['status']}: strict {result.get('strict_success')}/{result.get('cases')}, "
            f"adapter exit {adapter.get('exit_code')}, {adapter.get('wall_s')} s, "
            f"structure {'ok' if checks.get('structure_ok') else 'not ok'}, "
            f"preflight {'passed' if (manifest.get('preflight') or {}).get('passed') else 'OVERRIDDEN'}")


def start_caffeinate() -> subprocess.Popen[bytes] | None:
    """Keep the machine awake (display, idle, disk, system, user) for as long as this process lives."""
    binary = shutil.which("caffeinate")
    if binary is None:
        return None
    return subprocess.Popen([binary, "-dimsu", "-w", str(os.getpid())])


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark profiles through adapter.py (SPEC §9).")
    parser.add_argument("--suite", default="public", choices=sorted(SUITES))
    parser.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES), help="profile names or aliases")
    parser.add_argument("--all-profiles", action="store_true", help=f"every profile: {', '.join(PROFILES)}")
    parser.add_argument("--runs", type=int, default=3, help="runs per profile (= interleaved blocks)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="llama-server port for every run")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR, help="results root directory")
    parser.add_argument("--resume", action="store_true", help="skip runs that are already complete")
    parser.add_argument("--allow-dirty", action="store_true", help="run even if preflight refuses (recorded)")
    parser.add_argument("--preflight-wait-min", type=float, default=0.0, help="wait this long for a clean machine")
    parser.add_argument("--run-timeout-min", type=float, help="kill a run after this long (default 2 min per case)")
    parser.add_argument("--dry-run", action="store_true", help="print the schedule and commands, run nothing")
    parser.add_argument("--floors-only", action="store_true", help="only the code-only floors (no model, seconds)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        names = list(PROFILES) if args.all_profiles else [get_profile(name).name for name in args.profiles]
    except KeyError as exc:
        print(f"[bench] {exc.args[0]}", file=sys.stderr)
        return EXIT_USAGE
    if args.runs < 1 or len(set(names)) != len(names):
        print("[bench] --runs must be >= 1 and profiles must be distinct", file=sys.stderr)
        return EXIT_USAGE
    suite = SUITES[args.suite]
    if not suite.available():
        print(f"[bench] suite {suite.name} is not on disk yet ({suite.input_path}, {suite.labels_path})", file=sys.stderr)
        return EXIT_USAGE
    bench = Bench(suite, port=args.port, results_root=args.results.resolve(),
                  allow_dirty=args.allow_dirty, preflight_wait_s=args.preflight_wait_min * 60,
                  run_timeout_s=args.run_timeout_min * 60 if args.run_timeout_min else None)
    specs = schedule(names, args.runs, args.suite)
    if args.dry_run:
        for spec in specs:
            print(f"block {spec.run} #{spec.position}: {shlex.join(adapter_argv(bench, spec))}")
        return EXIT_OK
    for record in run_floors(bench):
        print(f"[bench] {floor_line(record)}", flush=True)
    if args.floors_only:
        return EXIT_OK
    try:
        print(f"[bench] fresh model hashes: {bench.rel(verify_models_fresh(bench, names))}", flush=True)
    except ModelCheckFailed as exc:
        print(f"[bench] {exc}", file=sys.stderr)
        return EXIT_REFUSED
    caffeinate = start_caffeinate()
    try:
        manifests = run_matrix(bench, specs, args.resume)
    except PreflightRefused as exc:
        print(f"[bench] preflight refused; nothing else was run. Fix the machine or pass --allow-dirty:\n{exc}",
              file=sys.stderr)
        return EXIT_REFUSED
    except RunExists as exc:
        print(f"[bench] {exc}", file=sys.stderr)
        return EXIT_USAGE
    finally:
        if caffeinate is not None:
            caffeinate.terminate()
    failed = [m for m in manifests if m["status"] != COMPLETE or not m["validate"]["structure_ok"]]
    return EXIT_RUN_FAILED if failed else EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    raise SystemExit(main())
