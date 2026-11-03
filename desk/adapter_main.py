"""Contract adapter (candidate_assets/CONTRACT.md): run every case through the desk, write results.json.

    python adapter.py --input cases.json --output results.json [--profile final] [--port 8402]
                      [--llama-url http://127.0.0.1:8402] [--strict-engine | --allow-engine-mismatch]

What it does, in order:
1. Reads the input as exact bytes (input_sha256 is computed over those bytes, not a re-dump).
2. Server: with --llama-url pointing at a healthy server it attaches and never stops it; otherwise
   it measures the idle GPU baseline, then starts llama-server for the profile and records the cold
   start under `startup`. The model sha256 must match the pin. The engine build is compared with the
   pin: by default a different build runs with a WARNING (recorded as engine_version_match false);
   --strict-engine refuses it (the bench uses that); --allow-engine-mismatch or
   DESK_ALLOW_ENGINE_MISMATCH=1 accepts it without a warning. A busy --port is not fatal: the next
   free port is used and recorded (startup.port_requested / startup.port).
3. Runs the cases sequentially, each from a fresh store (reset), through the same CaseSession the UI
   uses, with one MemoryProbe per case. If a case leaves the server unhealthy, the server is
   restarted for the next case. A case that crashes still yields one turn output per input turn.
4. Writes the document atomically and stops the server only if this process started it, also when
   the adapter is stopped by SIGTERM, SIGHUP or SIGINT (a harness timeout must not leave a
   llama-server holding the port and 3 GB of unified memory).

Exit codes: 0 output written; 1 unreadable input or bad arguments; 3 output written but the model
server could not be started (every turn is then decision "error", "not run: ..."); 128 + signal
number when stopped by a signal (no output is written then).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from desk import telemetry
from desk.agents import MAX_LLM_CALLS_PER_TURN, REPLY_STEP_TOKENS, TOOL_STEP_TOKENS
from desk.agents.operations import MAX_OPS_STEPS
from desk.agents.resolution import MAX_RESOLUTION_STEPS
from desk.llm import LLMClient
from desk.llm_server import (
    LlamaServer,
    LlamaServerError,
    next_free_port,
    pid_listening_on,
    port_in_use,
    same_engine_build,
    server_healthy,
)
from desk.orchestrator import MAX_HANDOFFS
from desk.profiles import (
    CTX_SIZE,
    ENGINE,
    ENGINE_VERSION,
    MODEL_NAME,
    MODEL_REPO,
    MODEL_REVISION,
    SEED,
    SLOTS,
    Profile,
    get_profile,
)
from desk.session import CASE_TIMEOUT_S, not_run_result, run_case

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORT = 8402
LLM_STEP_TIMEOUT_S = 20.0
LLM_HTTP_RETRIES = 1
LOG_DIR = ROOT / "var" / "adapter"
EXIT_OK, EXIT_BAD_INPUT, EXIT_NO_SERVER = 0, 1, 3
# Windows has no SIGHUP.
STOP_SIGNALS = tuple(getattr(signal, name) for name in ("SIGTERM", "SIGHUP", "SIGINT") if hasattr(signal, name))
ENGINE_WARN, ENGINE_STRICT, ENGINE_ALLOW = "warn", "strict", "allow"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run contract cases through the returns desk.")
    parser.add_argument("--input", required=True, help="contract cases JSON")
    parser.add_argument("--output", required=True, help="where to write results JSON")
    parser.add_argument("--profile", default="final", help="run profile (final=F, baseline=B, or any desk.profiles name)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="port for the llama-server this run starts (the next free one if it is busy)")
    parser.add_argument("--llama-url", help="attach to this running llama-server if it is healthy")
    parser.add_argument("--strict-engine", action="store_true",
                        help="refuse a llama.cpp build other than the pinned one (default: warn and record it)")
    parser.add_argument("--allow-engine-mismatch", action="store_true",
                        help="accept another llama.cpp build without a warning (also DESK_ALLOW_ENGINE_MISMATCH=1)")
    return parser.parse_args(argv)


def engine_policy(args: argparse.Namespace, environ: Mapping[str, str] = os.environ) -> str:
    """An explicit allow (flag or env) wins, then --strict-engine; the default is to warn."""
    if args.allow_engine_mismatch or environ.get("DESK_ALLOW_ENGINE_MISMATCH", "").strip() == "1":
        return ENGINE_ALLOW
    return ENGINE_STRICT if args.strict_engine else ENGINE_WARN


class ServerHandle:
    """The llama-server for this run: started (and owned) by us, or attached to via --llama-url."""

    def __init__(self, profile: Profile, port: int, llama_url: str | None, log_dir: Path = LOG_DIR,
                 engine_check: str = ENGINE_WARN) -> None:
        self.profile = profile
        self.port = port
        self.requested_port = port
        self.llama_url = llama_url
        self.log_dir = log_dir
        self.engine_check = engine_check
        self.server: LlamaServer | None = None
        self.gpu_baseline: int | None = None
        self.restarts = 0
        self.attached_pid: int | None = None

    @property
    def owned(self) -> bool:
        return self.server is not None

    @property
    def base_url(self) -> str:
        return self.server.base_url if self.server else str(self.llama_url).rstrip("/")

    @property
    def pid(self) -> int | None:
        return self.server.pid if self.server else self.attached_pid

    def start(self) -> dict[str, Any]:
        """Attach when --llama-url is healthy, else launch our own server. Returns `startup`."""
        if self.llama_url and server_healthy(self.llama_url):
            port = urlparse(self.llama_url).port
            self.attached_pid = pid_listening_on(port) if port else None
            return {"attached": True, "llama_url": self.base_url, "server_spawn_to_ready_ms": None,
                    "first_request_ms": None, "model_page_cache": "unknown", "attached_model": attached_model(self.base_url),
                    "note": "attached to a running llama-server: cold start, sha256 and engine version not measured "
                            "by this run; peak_vram_mb is null (no idle GPU baseline)."}
        self.gpu_baseline = telemetry.measure_gpu_baseline()  # before the model is loaded
        return self._launch()

    def _launch(self) -> dict[str, Any]:
        port_note = self._pick_port()
        suffix = f"-restart{self.restarts}" if self.restarts else ""
        log = self.log_dir / f"server-{self.profile.name}-{self.port}{suffix}.log"
        self.server = LlamaServer(self.profile, self.port, log, allow_engine_mismatch=self.engine_check != ENGINE_STRICT)
        startup = self.server.start()
        if startup.get("engine_version_match") is False and self.engine_check == ENGINE_WARN:
            _log(f"WARNING: llama.cpp {startup['engine_version']!r} is not the pinned "
                 f"{startup['engine_version_expected']!r}; running anyway, results may not match the benchmark "
                 "(--strict-engine refuses, --allow-engine-mismatch silences this)")
        return {"attached": False, "port": self.port, "port_requested": self.requested_port, "port_note": port_note,
                "engine_check": self.engine_check, "log": str(log), **startup}

    def _pick_port(self) -> str | None:
        """A busy port (an orphaned server, a second run) must not fail every case: move to a free one."""
        if not port_in_use("127.0.0.1", self.requested_port):
            self.port = self.requested_port
            return None
        busy_pid = pid_listening_on(self.requested_port)
        self.port = next_free_port("127.0.0.1", self.requested_port + 1)
        note = f"port {self.requested_port} was in use (pid {busy_pid}); llama-server started on {self.port} instead"
        _log(note)
        return note

    def healthy(self) -> bool:
        return server_healthy(self.base_url)

    def restart(self) -> dict[str, Any]:
        self.stop()
        self.restarts += 1
        return self._launch()

    def stop(self) -> None:
        if self.server is not None:
            self.server.stop()

    def memory_probe(self) -> telemetry.MemoryProbe:
        pids = {"adapter": os.getpid()}
        if self.pid is not None:
            pids["llama-server"] = self.pid
        return telemetry.MemoryProbe(pids, gpu_baseline_bytes=self.gpu_baseline)

    def own_pids(self) -> list[int]:
        return [pid for pid in (os.getpid(), self.pid) if pid is not None]


def attached_model(base_url: str) -> str | None:
    """The model id an attached server reports (its GGUF path), for the record."""
    try:
        data = httpx.get(f"{base_url}/v1/models", timeout=5).json()
        return str(data["data"][0]["id"])
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
        return None


def configuration(profile: Profile, server: ServerHandle, conditions: dict[str, Any]) -> dict[str, Any]:
    """Root `configuration` (SPEC §8): everything needed to reproduce the run."""
    owned = server.server
    return {
        "profile": profile.name,
        "profile_description": profile.description,
        "model": MODEL_NAME,
        "model_repo": MODEL_REPO,
        "model_file": profile.model_file,
        "model_revision": MODEL_REVISION,
        "model_sha256": (owned.model_sha256 if owned else None) or profile.model_sha256,
        # Hashes are compared with the pin before launch; a mismatch never gets this far.
        "model_sha256_verified": bool(owned and owned.model_sha256),
        "model_sha256_source": owned.model_sha256_source if owned else None,
        "quantization": profile.quantization,
        "engine": ENGINE,
        "engine_version": (owned.engine_version if owned else None) or ENGINE_VERSION,
        "engine_version_expected": ENGINE_VERSION,
        "engine_version_verified": bool(owned and owned.engine_version
                                        and same_engine_build(owned.engine_version, ENGINE_VERSION)),
        "engine_check": server.engine_check,
        "engine_argv": owned.argv if owned else None,
        "protocol": profile.protocol,
        "seed": SEED,
        "temperature": 0,
        "max_tokens": {"tool_step": TOOL_STEP_TOKENS, "reply_step": REPLY_STEP_TOKENS},
        "ctx_per_slot": CTX_SIZE,
        "slots": SLOTS,
        "concurrency": 1,
        "caches": profile.caches(),
        "timeouts": {"llm_step_s": LLM_STEP_TIMEOUT_S, "llm_http_retries": LLM_HTTP_RETRIES, "case_s": CASE_TIMEOUT_S},
        "limits": {"resolution_steps_per_turn": MAX_RESOLUTION_STEPS, "operations_steps_per_handoff": MAX_OPS_STEPS,
                   "handoffs_per_turn": MAX_HANDOFFS, "llm_calls_per_turn": MAX_LLM_CALLS_PER_TURN},
        "app_switches": profile.app_switches(),
        "llama_server": {"url": server.base_url, "owned_by_adapter": server.owned, "restarts": server.restarts},
        "run_conditions": conditions,
    }


def run_cases(cases: list[dict[str, Any]], profile: Profile, server: ServerHandle) -> list[dict[str, Any]]:
    """Sequential cases, fresh state each; restart an owned server that a case left unhealthy."""
    results = []
    first = True
    with LLMClient(server.base_url, profile.protocol, seed=SEED, timeout_s=LLM_STEP_TIMEOUT_S,
                   retries=LLM_HTTP_RETRIES) as llm:
        for case in cases:
            result = run_case(case, llm, profile, probe=server.memory_probe())
            result["metrics"]["first_case_after_start"] = first
            first = False
            results.append(result)
            _log(f"{case.get('case_id')}: {[t['decision'] for t in result['turns']]} "
                 f"({result['metrics']['elapsed_ms']:.0f} ms, {result['metrics']['status']})")
            if server.owned and not server.healthy():
                _log("llama-server is not healthy after this case; restarting it for the next case")
                try:
                    server.restart()
                    first = True
                except LlamaServerError as exc:
                    _log(f"restart failed: {exc}")
    return results


def write_json(path: Path, document: dict[str, Any]) -> None:
    """Write via a temporary file so a crash never leaves a half-written results file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def exit_on_signal(signum: int, _frame: Any) -> None:
    """Python's default SIGTERM/SIGHUP end the process without running `finally`; SystemExit does."""
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        raw = Path(args.input).read_bytes()
        cases = json.loads(raw)["cases"]
        profile = get_profile(args.profile)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"adapter: cannot use the input: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT
    for stop_signal in STOP_SIGNALS:
        signal.signal(stop_signal, exit_on_signal)
    server = ServerHandle(profile, args.port, args.llama_url, engine_check=engine_policy(args))
    document: dict[str, Any] = {"contract_version": 1, "input_sha256": hashlib.sha256(raw).hexdigest(),
                                "hardware": telemetry.static_hardware()}
    code = EXIT_OK
    conditions: dict[str, Any] = {}
    try:
        try:
            startup = server.start()
        except LlamaServerError as exc:
            _log(f"llama-server unavailable: {exc}")
            startup = {"attached": False, "error": str(exc)}
            results = [not_run_result(case, f"llama-server unavailable: {exc}") for case in cases]
            code = EXIT_NO_SERVER
        else:
            conditions["before"] = telemetry.run_conditions(server.own_pids())
            results = run_cases(cases, profile, server)
            conditions["after"] = telemetry.run_conditions(server.own_pids())
        document["configuration"] = configuration(profile, server, conditions)
        document["startup"] = startup
        document["results"] = results
        write_json(Path(args.output), document)
    finally:
        server.stop()
    _log(f"wrote {len(document.get('results', []))} case results to {args.output}")
    return code


def _log(message: str) -> None:
    print(f"[adapter] {message}", file=sys.stderr, flush=True)
