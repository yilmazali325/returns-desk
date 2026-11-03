"""Adapter tests: desk/adapter_main.py with a fake llama-server and FakeLLM, and the adapter.py
bootstrap shim (including a real `python3 adapter.py` run from another directory).

The output document is checked with the organisers' structural validator
(candidate_assets/validate_submission.py) and graded with the independent scorer.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

import desk.adapter_main as adapter_main
from desk import telemetry
from desk.llm import FakeLLM
from desk.llm_server import SHA_CACHED, LlamaServerError, pid_listening_on
from desk.profiles import ENGINE_VERSION, get_profile
from evals.score import score_run

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_PATH = ROOT / "candidate_assets" / "public_cases.json"
LABELS = json.loads((ROOT / "evals" / "expected" / "public_expected.json").read_text())


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_module("validate_submission", ROOT / "candidate_assets" / "validate_submission.py")


def H(intent: str, **args: Any) -> tuple[str, dict[str, Any]]:
    return ("handoff_to_operations", {"intent": intent, **args})


def R(message: str) -> tuple[str, dict[str, Any]]:
    return ("reply_to_customer", {"message": message})


# Scripted steps for public_01 then public_03 (the two cases in the small input below).
SCRIPT = {
    "resolution": [H("propose_refund", order_id="O100", line_id="L1"), R("Here is what I can do."),
                   H("execute_confirmed"), R("All done."),
                   H("propose_refund", order_id="O100", line_id="L1"), R("I'm sorry, I can't do that one.")],
    "operations": [("create_proposal", {"remedy": "refund"}), ("report_to_resolution", {"outcome": "proposed"}),
                   ("execute_proposal", {}), ("report_to_resolution", {"outcome": "completed"}),
                   ("create_proposal", {"remedy": "refund"}), ("report_to_resolution", {"outcome": "denied"})],
}


class FakeServer:
    """Stands in for desk.llm_server.LlamaServer (no process, fixed facts)."""

    instances: list[FakeServer] = []
    fail_on_start: list[bool] = []
    engine = ENGINE_VERSION  # what `llama-server --version` says; tests set another build

    def __init__(self, profile: Any, port: int, log_path: Path, allow_engine_mismatch: bool = False) -> None:
        self.profile, self.port, self.log_path = profile, port, log_path
        self.allow_engine_mismatch = allow_engine_mismatch
        self.pid = 4242
        self.model_sha256 = profile.model_sha256
        self.model_sha256_source = SHA_CACHED
        self.engine_version = FakeServer.engine
        self.stopped = False
        FakeServer.instances.append(self)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def argv(self) -> list[str]:
        return ["llama-server", "-m", self.profile.model_file, "--port", str(self.port)]

    def start(self) -> dict[str, Any]:
        if FakeServer.fail_on_start and FakeServer.fail_on_start.pop(0):
            raise LlamaServerError("llama-server not found")
        match = self.engine_version == ENGINE_VERSION
        if not match and not self.allow_engine_mismatch:  # what LlamaServer.verify() does
            raise LlamaServerError(f"engine version {self.engine_version!r} is not the pinned {ENGINE_VERSION!r}: "
                                   "to run it anyway pass --allow-engine-mismatch")
        return {"server_spawn_to_ready_ms": 700.0, "first_request_ms": 60.0, "warmup": True, "model_page_cache": "warm",
                "engine_version": self.engine_version, "engine_version_expected": ENGINE_VERSION,
                "engine_version_match": match}

    def stop(self) -> None:
        self.stopped = True


class FakeProbe:
    def __init__(self, pids: dict[str, int], gpu_baseline_bytes: int | None) -> None:
        self.pids, self.baseline = pids, gpu_baseline_bytes

    def start(self) -> None: ...

    def mark(self, label: str) -> None: ...

    def stop(self) -> None: ...

    def report(self) -> dict[str, Any]:
        return {"peak_ram_mb": 1700.0, "peak_vram_mb": 3200.0 if self.baseline is not None else None,
                "peak_unified_gpu_mb": None, "rss_mb": 4000.0, "memory_architecture": "unified", "memory_notes": []}


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Patch every external dependency of the adapter; return knobs for the test."""
    FakeServer.instances = []
    FakeServer.fail_on_start = []
    monkeypatch.setattr(FakeServer, "engine", ENGINE_VERSION)
    monkeypatch.delenv("DESK_ALLOW_ENGINE_MISMATCH", raising=False)
    knobs: dict[str, Any] = {"healthy": lambda url: True, "script": copy.deepcopy(SCRIPT), "llm_urls": [],
                             "busy_ports": set(), "signals": []}

    class FakeClient:
        def __init__(self, base_url: str, protocol: str, **_: Any) -> None:
            knobs["llm_urls"].append((base_url, protocol))
            self.fake = FakeLLM(knobs["script"])

        def __enter__(self) -> FakeLLM:
            return self.fake

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(adapter_main, "LlamaServer", FakeServer)
    monkeypatch.setattr(adapter_main, "LLMClient", FakeClient)
    monkeypatch.setattr(adapter_main, "server_healthy", lambda url: knobs["healthy"](url))
    monkeypatch.setattr(adapter_main, "pid_listening_on", lambda port: 5151)
    monkeypatch.setattr(adapter_main, "attached_model", lambda url: "models/x.gguf")
    monkeypatch.setattr(adapter_main, "port_in_use", lambda host, port: port in knobs["busy_ports"])
    # next_free_port itself is tested against real sockets in test_llm_server.py
    monkeypatch.setattr(adapter_main, "next_free_port",
                        lambda host, start: next(p for p in range(start, start + 20) if p not in knobs["busy_ports"]))
    monkeypatch.setattr(adapter_main.signal, "signal", lambda signum, handler: knobs["signals"].append((signum, handler)))
    monkeypatch.setattr(adapter_main, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(telemetry, "measure_gpu_baseline", lambda: 800 * 2**20)
    monkeypatch.setattr(telemetry, "run_conditions", lambda pids=(): {"on_ac_power": True, "pids": list(pids)})
    monkeypatch.setattr(telemetry, "MemoryProbe", FakeProbe)
    cases = json.loads(PUBLIC_PATH.read_text())["cases"]
    small = tmp_path / "cases.json"
    # odd formatting on purpose: input_sha256 must hash these exact bytes, not a re-dump
    small.write_bytes(json.dumps({"contract_version": 1, "cases": [cases[0], cases[2]]}, indent=3).encode() + b"\n\n")
    knobs["input"], knobs["output"] = small, tmp_path / "out" / "results.json"
    return knobs


def run(env: dict[str, Any], *extra: str) -> tuple[int, dict[str, Any] | None]:
    code = adapter_main.main(["--input", str(env["input"]), "--output", str(env["output"]), *extra])
    document = json.loads(env["output"].read_text()) if env["output"].exists() else None
    return code, document


def test_full_run_is_contract_valid_and_scores(env) -> None:
    code, document = run(env, "--profile", "F", "--port", "8499")
    assert code == 0
    assert VALIDATOR.check(env["input"], env["output"]) == 2
    assert document["input_sha256"] == hashlib.sha256(env["input"].read_bytes()).hexdigest()
    scores, _warnings = score_run(env["input"].read_bytes(), LABELS, document)  # (4 labels have no input here)
    assert [(s.case_id, s.strict) for s in scores] == [("public_01_refund_boundary", True), ("public_03_expired", True)]
    config = document["configuration"]
    assert config["profile"] == "F" and config["quantization"] == "Q4_K_M" and config["protocol"] == "constrained"
    assert config["engine_argv"][-1] == "8499" and config["model_sha256_verified"] is True
    assert config["model_sha256_source"] == SHA_CACHED and config["engine_version_verified"] is True
    assert config["engine_version_expected"] == ENGINE_VERSION and config["engine_check"] == "warn"
    assert document["startup"]["engine_version_match"] is True and document["startup"]["port_note"] is None
    assert (document["startup"]["port"], document["startup"]["port_requested"]) == (8499, 8499)
    assert env["signals"] == [(s, adapter_main.exit_on_signal) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)]
    assert config["app_switches"] == get_profile("F").app_switches()
    assert config["run_conditions"]["before"]["pids"] == [os.getpid(), 4242]
    assert document["startup"]["attached"] is False and document["startup"]["server_spawn_to_ready_ms"] == 700.0
    metrics = [r["metrics"] for r in document["results"]]
    assert [m["first_case_after_start"] for m in metrics] == [True, False]
    assert metrics[0]["peak_vram_mb"] == 3200.0
    assert FakeServer.instances[0].stopped and env["llm_urls"] == [("http://127.0.0.1:8499", "constrained")]
    assert set(document) == {"contract_version", "input_sha256", "hardware", "configuration", "startup", "results"}


def test_default_profile_is_final_on_the_default_port(env) -> None:
    code, document = run(env)
    assert code == 0 and document["configuration"]["profile"] == "F"
    assert FakeServer.instances[0].port == adapter_main.DEFAULT_PORT


def test_server_that_cannot_start_still_yields_a_valid_failed_document(env) -> None:
    FakeServer.fail_on_start = [True]
    code, document = run(env)
    assert code == adapter_main.EXIT_NO_SERVER
    assert VALIDATOR.check(env["input"], env["output"]) == 2
    assert document["startup"]["error"] == "llama-server not found"
    assert all(t["decision"] == "error" for r in document["results"] for t in r["turns"])
    assert all(r["metrics"]["status"] == "failed" for r in document["results"])


def test_another_engine_build_runs_with_a_warning_by_default(env, capsys) -> None:
    FakeServer.engine = "0.7.1 (build 11702, commit abcdef123)"
    code, document = run(env)
    assert code == 0 and FakeServer.instances[0].allow_engine_mismatch is True
    assert "WARNING: llama.cpp '0.7.1 (build 11702, commit abcdef123)' is not the pinned" in capsys.readouterr().err
    startup, config = document["startup"], document["configuration"]
    assert startup["engine_version_match"] is False and startup["engine_check"] == "warn"
    assert config["engine_version"].startswith("0.7.1") and config["engine_version_verified"] is False
    assert all(t["decision"] != "error" for r in document["results"] for t in r["turns"])


def test_strict_engine_refuses_another_build_and_names_the_way_out(env, capsys) -> None:
    FakeServer.engine = "0.7.1 (build 11702, commit abcdef123)"
    code, document = run(env, "--strict-engine")
    assert code == adapter_main.EXIT_NO_SERVER and FakeServer.instances[0].allow_engine_mismatch is False
    assert "--allow-engine-mismatch" in capsys.readouterr().err
    assert document["configuration"]["engine_check"] == "strict"
    assert all(t["decision"] == "error" for r in document["results"] for t in r["turns"])


@pytest.mark.parametrize("how", ["flag", "env"])
def test_an_explicit_allow_runs_another_build_quietly_even_in_strict_mode(env, capsys, monkeypatch, how: str) -> None:
    FakeServer.engine = "0.7.1 (build 11702, commit abcdef123)"
    if how == "env":
        monkeypatch.setenv("DESK_ALLOW_ENGINE_MISMATCH", "1")
    code, document = run(env, "--strict-engine", *(["--allow-engine-mismatch"] if how == "flag" else []))
    assert code == 0 and document["startup"]["engine_check"] == "allow"
    assert "WARNING" not in capsys.readouterr().err


def test_a_busy_port_moves_the_server_to_the_next_free_port(env, capsys) -> None:
    env["busy_ports"] = {8499, 8500}
    code, document = run(env, "--port", "8499")
    assert code == 0 and FakeServer.instances[0].port == 8501
    startup = document["startup"]
    assert (startup["port_requested"], startup["port"]) == (8499, 8501)
    assert startup["port_note"] == "port 8499 was in use (pid 5151); llama-server started on 8501 instead"
    assert env["llm_urls"] == [("http://127.0.0.1:8501", "constrained")] and "8501 instead" in capsys.readouterr().err


def test_exit_on_signal_runs_finally_blocks() -> None:
    with pytest.raises(SystemExit) as caught:
        adapter_main.exit_on_signal(signal.SIGTERM, None)
    assert caught.value.code == 128 + signal.SIGTERM


def test_attach_to_a_healthy_server_and_never_stop_it(env) -> None:
    code, document = run(env, "--llama-url", "http://127.0.0.1:8777/")
    assert code == 0 and FakeServer.instances == []
    startup, config = document["startup"], document["configuration"]
    assert startup["attached"] is True and startup["attached_model"] == "models/x.gguf"
    assert config["engine_argv"] is None and config["model_sha256_verified"] is False
    assert config["model_sha256_source"] is None and config["engine_version_verified"] is False
    assert config["llama_server"] == {"url": "http://127.0.0.1:8777", "owned_by_adapter": False, "restarts": 0}
    assert document["results"][0]["metrics"]["peak_vram_mb"] is None  # no idle baseline when attached
    assert config["run_conditions"]["before"]["pids"] == [os.getpid(), 5151]


def test_unhealthy_attach_url_falls_back_to_our_own_server(env) -> None:
    env["healthy"] = lambda url: "8777" not in url
    code, document = run(env, "--llama-url", "http://127.0.0.1:8777")
    assert code == 0 and len(FakeServer.instances) == 1 and document["startup"]["attached"] is False


def test_server_left_unhealthy_is_restarted_for_the_next_case(env) -> None:
    calls = {"n": 0}

    def healthy(url: str) -> bool:
        calls["n"] += 1
        return calls["n"] != 1  # unhealthy right after the first case

    env["healthy"] = healthy
    code, document = run(env)
    assert code == 0 and len(FakeServer.instances) == 2
    assert FakeServer.instances[0].stopped and "restart1" in str(FakeServer.instances[1].log_path)
    assert [r["metrics"]["first_case_after_start"] for r in document["results"]] == [True, True]
    assert document["configuration"]["llama_server"]["restarts"] == 1


def test_failed_restart_is_logged_and_the_run_continues(env, capsys) -> None:
    env["healthy"] = lambda url: False
    FakeServer.fail_on_start = [False, True, True]
    code, document = run(env)
    assert code == 0 and len(document["results"]) == 2
    assert "restart failed" in capsys.readouterr().err


@pytest.mark.parametrize("content, extra", [(None, ()), (b"{not json", ()), (b'{"cases": []}', ("--profile", "Z"))])
def test_unusable_input_exits_1_without_output(env, content: bytes | None, extra: tuple[str, ...], capsys) -> None:
    if content is None:
        env["input"].unlink()
    else:
        env["input"].write_bytes(content)
    code, document = run(env, *extra)
    assert code == adapter_main.EXIT_BAD_INPUT and document is None
    assert "cannot use the input" in capsys.readouterr().err


def test_attached_model_reads_the_server_model_list(monkeypatch) -> None:
    class Response:
        def __init__(self, payload: Any) -> None:
            self.payload = payload

        def json(self) -> Any:
            return self.payload

    monkeypatch.setattr(adapter_main.httpx, "get", lambda url, timeout: Response({"data": [{"id": "m.gguf"}]}))
    assert adapter_main.attached_model("http://x") == "m.gguf"
    monkeypatch.setattr(adapter_main.httpx, "get", lambda url, timeout: Response({"data": []}))
    assert adapter_main.attached_model("http://x") is None


def test_write_json_is_atomic(tmp_path) -> None:
    target = tmp_path / "a" / "b.json"
    adapter_main.write_json(target, {"x": "é"})
    assert json.loads(target.read_text()) == {"x": "é"} and not (tmp_path / "a" / "b.json.tmp").exists()


def test_server_handle_without_a_server(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "MemoryProbe", FakeProbe)
    handle = adapter_main.ServerHandle(get_profile("B"), 8403, "http://127.0.0.1:9/")
    assert handle.base_url == "http://127.0.0.1:9" and handle.pid is None and not handle.owned
    handle.stop()  # nothing to stop
    assert handle.own_pids() == [os.getpid()]
    assert handle.memory_probe().pids == {"adapter": os.getpid()}


# --- adapter.py bootstrap shim ----------------------------------------------------------------


def test_shim_runs_the_adapter_when_dependencies_are_present(monkeypatch, capsys) -> None:
    shim = load_module("adapter_shim", ROOT / "adapter.py")
    monkeypatch.setattr(sys, "argv", ["adapter.py", "--input", "/nonexistent.json", "--output", "/tmp/x.json"])
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(ROOT)])  # as when run from another directory
    assert shim.main() == adapter_main.EXIT_BAD_INPUT  # inserts the repo root into sys.path
    assert shim.main() == adapter_main.EXIT_BAD_INPUT  # root already importable
    assert shim.dependencies_available() and shim.running_in_venv()


def test_shim_reexecs_into_the_venv_when_dependencies_are_missing(monkeypatch) -> None:
    shim = load_module("adapter_shim2", ROOT / "adapter.py")
    seen: dict[str, Any] = {}

    def fake_execv(path: str, argv: list[str]) -> None:
        seen["call"] = (path, argv)
        raise SystemExit(0)

    monkeypatch.setattr(shim, "dependencies_available", lambda: False)
    monkeypatch.setattr(shim, "running_in_venv", lambda: False)
    monkeypatch.delenv(shim.REEXEC_MARKER, raising=False)
    monkeypatch.setattr(shim.os, "execv", fake_execv)
    monkeypatch.setattr(sys, "argv", ["adapter.py", "--input", "a.json"])
    with pytest.raises(SystemExit):
        shim.main()
    assert seen["call"] == (shim.VENV_PYTHON, [shim.VENV_PYTHON, str(ROOT / "adapter.py"), "--input", "a.json"])
    assert os.environ[shim.REEXEC_MARKER] == "1"


def test_shim_runs_the_venv_python_as_a_child_on_windows(monkeypatch) -> None:
    shim = load_module("adapter_shim5", ROOT / "adapter.py")
    calls: list[list[str]] = []
    monkeypatch.setattr(shim, "dependencies_available", lambda: False)
    monkeypatch.setattr(shim, "running_in_venv", lambda: False)
    monkeypatch.delenv(shim.REEXEC_MARKER, raising=False)
    monkeypatch.setattr(shim.os, "name", "nt")  # Windows has no exec
    monkeypatch.setattr(subprocess, "call", lambda argv: calls.append(argv) or 3)
    monkeypatch.setattr(sys, "argv", ["adapter.py", "--input", "a.json"])
    assert shim.main() == 3
    assert calls == [[shim.VENV_PYTHON, str(ROOT / "adapter.py"), "--input", "a.json"]]


def test_shim_prints_the_setup_command_when_no_venv_can_help(monkeypatch, capsys) -> None:
    shim = load_module("adapter_shim3", ROOT / "adapter.py")
    monkeypatch.setattr(shim, "dependencies_available", lambda: False)
    monkeypatch.setattr(shim, "VENV_PYTHON", "/nonexistent/python")
    assert shim.main() == 2
    assert "uv sync --frozen" in capsys.readouterr().err


def test_dependency_probe_reports_a_missing_module(monkeypatch) -> None:
    shim = load_module("adapter_shim4", ROOT / "adapter.py")
    monkeypatch.setitem(sys.modules, "psutil", None)  # makes `import psutil` raise ImportError
    assert shim.dependencies_available() is False


def test_plain_python3_from_another_directory_reaches_the_adapter(tmp_path) -> None:
    python3 = shutil.which("python3", path="/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin")
    if python3 is None:
        pytest.skip("no system python3")
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH", "RETURNS_DESK_ADAPTER_REEXEC")}
    done = subprocess.run([python3, str(ROOT / "adapter.py"), "--input", "missing.json", "--output", "out.json"],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == adapter_main.EXIT_BAD_INPUT, done.stderr
    assert "cannot use the input" in done.stderr and "missing.json" in done.stderr


# --- a real adapter process stopped by a signal must not leave its llama-server behind ----------

STUB_LLAMA = r"""#!{python}
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if "--version" in sys.argv:
    print("version: {version}")
    sys.exit(0)

class LoadingForever(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):  # /health says "loading" forever, so the adapter keeps waiting
        self.send_response(503)
        self.send_header("Content-Length", "0")
        self.end_headers()

ThreadingHTTPServer(("127.0.0.1", int(sys.argv[sys.argv.index("--port") + 1])), LoadingForever).serve_forever()
"""

DRIVER = """import functools, sys
import desk.adapter_main as adapter_main
from desk.llm_server import LlamaServer
adapter_main.LlamaServer = functools.partial(LlamaServer, binary=sys.argv[1], models_dir=sys.argv[2],
                                             expected_sha256=sys.argv[3])
sys.exit(adapter_main.main(sys.argv[4:]))
"""


def _listening(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) == 0


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGHUP, signal.SIGINT])
def test_a_signalled_adapter_stops_the_llama_server_it_started(tmp_path, stop_signal: int) -> None:
    stub = tmp_path / "llama-server"
    stub.write_text(STUB_LLAMA.replace("{python}", sys.executable).replace("{version}", ENGINE_VERSION))
    stub.chmod(0o755)
    models = tmp_path / "models"
    models.mkdir()
    (models / get_profile("F").model_file).write_bytes(b"GGUF stub")
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    adapter = subprocess.Popen(
        [sys.executable, str(driver), str(stub), str(models), hashlib.sha256(b"GGUF stub").hexdigest(),
         "--input", str(PUBLIC_PATH), "--output", str(tmp_path / "out.json"), "--port", str(port)],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 60  # generous: Python start-up takes seconds on a loaded machine
        while not _listening(port):
            assert adapter.poll() is None and time.monotonic() < deadline, adapter.stdout.read().decode()
            time.sleep(0.05)
        adapter.send_signal(stop_signal)
        assert adapter.wait(timeout=60) == 128 + stop_signal
        deadline = time.monotonic() + 5
        while _listening(port) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _listening(port), "the llama-server outlived the adapter"
        assert not (tmp_path / "out.json").exists()  # a stopped run writes nothing half-done
    finally:
        if adapter.poll() is None:
            adapter.kill()
        _kill_stub_on(port)  # a failed assertion above must not leave the stub listening


def _kill_stub_on(port: int) -> None:
    """Test hygiene only: the stub runs in its own session, so killing the adapter does not reach it."""
    pid = pid_listening_on(port) if _listening(port) else None
    if pid is not None and pid != os.getpid():
        os.kill(pid, signal.SIGKILL)


# --- the real adapter process against a fake llama-server binary of another build -----------------

STUB_OTHER_BUILD = r"""#!{python}
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if "--version" in sys.argv:
    print("version: 0.7.1 (build 11702, commit abcdef123)")
    sys.exit(0)

class Healthy(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _answer(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._answer({"status": "ok"})

    def do_POST(self):  # every completion is empty: the desk sees a model that says nothing useful
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self._answer({"choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
                      "usage": {"prompt_tokens": 1, "completion_tokens": 0}})

ThreadingHTTPServer(("127.0.0.1", int(sys.argv[sys.argv.index("--port") + 1])), Healthy).serve_forever()
"""


@pytest.mark.parametrize("strict", [True, False], ids=["strict-refuses", "default-warns"])
def test_another_llama_cpp_build_end_to_end(tmp_path, strict: bool) -> None:
    stub = tmp_path / "llama-server"
    stub.write_text(STUB_OTHER_BUILD.replace("{python}", sys.executable))
    stub.chmod(0o755)
    models = tmp_path / "models"
    models.mkdir()
    (models / get_profile("F").model_file).write_bytes(b"GGUF stub")
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    out = tmp_path / "out.json"
    argv = [sys.executable, str(driver), str(stub), str(models), hashlib.sha256(b"GGUF stub").hexdigest(),
            "--input", str(PUBLIC_PATH), "--output", str(out), "--port", str(port), *(["--strict-engine"] if strict else [])]
    env = {k: v for k, v in os.environ.items() if k != "DESK_ALLOW_ENGINE_MISMATCH"}
    try:
        done = subprocess.run(argv, cwd=ROOT, env={**env, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True,
                              timeout=600)
    finally:
        _kill_stub_on(port)
    document = json.loads(out.read_text())
    assert VALIDATOR.check(str(PUBLIC_PATH), str(out)) == len(document["results"]) == 6  # contract-valid either way
    if strict:
        assert done.returncode == adapter_main.EXIT_NO_SERVER
        assert "--allow-engine-mismatch" in done.stderr and "--llama-url" in done.stderr
        assert all(t["decision"] == "error" for r in document["results"] for t in r["turns"])
        assert not _listening(port)  # refused before anything was spawned
    else:
        assert done.returncode == adapter_main.EXIT_OK, done.stderr[-2000:]
        assert "WARNING: llama.cpp '0.7.1 (build 11702, commit abcdef123)' is not the pinned" in done.stderr
        startup = document["startup"]
        assert startup["engine_version_match"] is False and startup["engine_version_actual"].startswith("0.7.1")
        assert startup["engine_version_expected"] == ENGINE_VERSION and startup["engine_check"] == "warn"
        assert document["configuration"]["engine_version_verified"] is False
