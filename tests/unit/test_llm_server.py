"""Unit tests for desk/llm_server.py.

The lifecycle tests launch a small fake `llama-server` (a Python script written to tmp_path) that
speaks the endpoints we use (/health, /metrics, /v1/chat/completions) and prints the startup log
lines we parse. Its behaviour is switched with FAKE_* environment variables, which LlamaServer passes
through (only LLAMA_ARG_* variables are stripped).
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from desk import llm_server as LS
from desk.llm_server import LlamaServer, LlamaServerError
from desk.profiles import ENGINE_VERSION, get_profile

FAKE_SERVER = r'''#!{python}
import json, os, signal, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if "--version" in sys.argv:
    sys.stderr.write(os.environ.get("FAKE_VERSION", "version: 0.7.0 (build 11669, commit 620d2225c)") + "\n")
    sys.exit(0)
mode = os.environ.get("FAKE_MODE", "ok")
print("srv    load_model: initializing, n_slots = 1, n_ctx_slot = 8192, kv_unified = 'false'", flush=True)
print("cmn          init: llama threadpool init, n_threads = 5", flush=True)
if mode == "exit":
    print("error: cannot load model", flush=True)
    sys.exit(3)
if mode == "ignore_term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
port = int(sys.argv[sys.argv.index("--port") + 1])
state = {"health_calls": 0}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            state["health_calls"] += 1
            ready = mode != "never_ready" and state["health_calls"] > int(os.environ.get("FAKE_LOADING", "2"))
            self.reply(200 if ready else 503, {"status": "ok"} if ready else {"error": "Loading model"})
        elif self.path == "/metrics" and mode != "no_metrics":
            self.reply(200, "# HELP llamacpp:prompt_tokens_total x\nllamacpp:prompt_tokens_total 13\n", "text/plain")
        else:
            self.reply(404, {"error": "not found"})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if mode == "chat_500":
            self.reply(500, {"error": "boom"})
        else:
            self.reply(200, {"choices": [{"message": {"role": "assistant", "content": "p"}}]})

ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def fake_binary(tmp_path):
    path = tmp_path / "llama-server"
    path.write_text(FAKE_SERVER.replace("{python}", sys.executable))
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def models_dir(tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / get_profile("F").model_file).write_bytes(b"GGUF fake weights" * 1000)
    return folder


def fake_sha(models_dir):
    return hashlib.sha256((models_dir / get_profile("F").model_file).read_bytes()).hexdigest()


def make_server(fake_binary, models_dir, tmp_path, **kw):
    kw.setdefault("expected_sha256", fake_sha(models_dir))
    return LlamaServer(get_profile("F"), free_port(), tmp_path / "logs" / "server.log",
                       binary=fake_binary, models_dir=models_dir, **kw)


# --------------------------------------------------------------------------------------------------
# Lifecycle against the fake server

def test_start_reports_cold_start_and_checks(fake_binary, models_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "262144")  # must not leak into the child
    server = make_server(fake_binary, models_dir, tmp_path)
    assert server.pid is None and server.base_url == f"http://127.0.0.1:{server.port}"
    assert server.argv == get_profile("F").engine_argv(fake_binary, str(server.model_path), server.port)
    with server:
        startup = server.startup
        assert startup["server_spawn_to_ready_ms"] > 0 and startup["first_request_ms"] > 0
        assert startup["warmup"] is True
        assert startup["engine_version"] == ENGINE_VERSION and startup["engine_version_mismatch_allowed"] is False
        assert startup["engine_version_actual"] == startup["engine_version_expected"] == ENGINE_VERSION
        assert startup["engine_version_match"] is True
        assert startup["model_sha256"] == fake_sha(models_dir) and startup["model_sha256_source"] == LS.SHA_FRESH
        assert startup["model_page_cache"] in ("warm", "partial", "cold")
        assert 0 <= startup["model_page_cache_resident_pct"] <= 100
        assert startup["server_log_facts"] == {"n_slots": 1, "n_ctx_slot": 8192, "kv_unified": False, "n_threads": 5}
        assert isinstance(server.pid, int) and server.health()
        assert server.metrics() == {"llamacpp:prompt_tokens_total": 13.0}
        assert "prompt_tokens_total" in server.metrics_text()
        with pytest.raises(LlamaServerError, match="already started"):
            server.start()
    assert server.pid is None and not server.health()
    assert server.stop() is None  # idempotent
    with pytest.raises(LlamaServerError, match="unreachable"):
        server.metrics_text()


def test_stop_returns_exit_code(fake_binary, models_dir, tmp_path):
    server = make_server(fake_binary, models_dir, tmp_path, warmup=False)
    startup = server.start()
    assert startup["first_request_ms"] is None and startup["warmup"] is False
    assert server.stop() == -15  # SIGTERM


def test_stop_kills_a_server_that_ignores_sigterm(fake_binary, models_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "ignore_term")
    server = make_server(fake_binary, models_dir, tmp_path)
    server.start()
    assert server.stop(timeout_s=0.5) == -9


@pytest.mark.parametrize(("mode", "code"), [("normal", -15), ("ignore_term", -9)])
def test_stop_without_process_groups_ends_the_child_itself(fake_binary, models_dir, tmp_path, monkeypatch, mode, code):
    monkeypatch.setenv("FAKE_MODE", mode)
    server = make_server(fake_binary, models_dir, tmp_path, warmup=False)
    server.start()
    monkeypatch.delattr(LS.os, "killpg")  # as on Windows
    assert server.stop(timeout_s=0.5) == code


def test_metrics_need_the_metrics_flag(fake_binary, models_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "no_metrics")
    with make_server(fake_binary, models_dir, tmp_path) as server:
        with pytest.raises(LlamaServerError, match="HTTP 404"):
            server.metrics()


def test_server_that_exits_during_startup(fake_binary, models_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "exit")
    server = make_server(fake_binary, models_dir, tmp_path)
    with pytest.raises(LlamaServerError, match=r"(?s)exited with code 3 during startup:\n.*cannot load model"):
        server.start()
    assert server.pid is None


def test_server_that_never_becomes_healthy(fake_binary, models_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "never_ready")
    server = make_server(fake_binary, models_dir, tmp_path, ready_timeout_s=0.5)
    with pytest.raises(LlamaServerError, match="not healthy after"):
        server.start()
    assert server.pid is None


def test_warmup_failures(fake_binary, models_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "chat_500")
    server = make_server(fake_binary, models_dir, tmp_path)
    with pytest.raises(LlamaServerError, match="warm-up request answered HTTP 500"):
        server.start()
    server.stop()

    monkeypatch.setenv("FAKE_MODE", "ok")

    def refuse(*args, **kwargs):
        raise httpx.ConnectError("refused")

    server = make_server(fake_binary, models_dir, tmp_path)
    monkeypatch.setattr(LS.httpx, "post", refuse)
    with pytest.raises(LlamaServerError, match="warm-up request failed"):
        server.start()
    server.stop()


def test_port_already_in_use(fake_binary, models_dir, tmp_path):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        server = LlamaServer(get_profile("F"), port, tmp_path / "s.log", binary=fake_binary,
                             models_dir=models_dir, expected_sha256=fake_sha(models_dir))
        with pytest.raises(LlamaServerError, match=f"port {port} is already in use"):
            server.start()


# --------------------------------------------------------------------------------------------------
# verify(): model hash and engine version

def test_verify_rejects_missing_model(fake_binary, tmp_path):
    server = LlamaServer(get_profile("F"), free_port(), tmp_path / "s.log", binary=fake_binary,
                         models_dir=tmp_path / "empty")
    with pytest.raises(LlamaServerError, match="model file missing"):
        server.verify()


def test_verify_rejects_wrong_hash(fake_binary, models_dir, tmp_path):
    server = make_server(fake_binary, models_dir, tmp_path, expected_sha256=None)  # the real pinned hash
    with pytest.raises(LlamaServerError, match="sha256 mismatch"):
        server.verify()


def test_a_source_build_of_the_pinned_commit_is_the_pinned_engine(fake_binary, models_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_VERSION", "version: 0.7.0-dev (build 11669, commit 620d2225c)")
    checks = make_server(fake_binary, models_dir, tmp_path).verify()
    assert checks["engine_version_match"] is True and checks["engine_version"].startswith("0.7.0-dev")
    assert LS.same_engine_build("0.7.0 (build 11669, commit 620d2225c)", ENGINE_VERSION)
    assert not LS.same_engine_build("0.7.0 (build 11670, commit 620d2225c)", ENGINE_VERSION)
    assert not LS.same_engine_build("0.7.0", "0.7.0-dev")  # an expected string without a build part


def test_verify_engine_version_mismatch(fake_binary, models_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_VERSION", "version: 0.8.0 (build 12000, commit abc)")
    with pytest.raises(LlamaServerError, match="engine version '0.8.0") as refused:
        make_server(fake_binary, models_dir, tmp_path).verify()
    assert "--allow-engine-mismatch" in str(refused.value) and "--llama-url" in str(refused.value)
    checks = make_server(fake_binary, models_dir, tmp_path, allow_engine_mismatch=True).verify()
    assert checks["engine_version_mismatch_allowed"] is True and checks["engine_version"].startswith("0.8.0")
    assert checks["engine_version_match"] is False and checks["engine_version_expected"] == ENGINE_VERSION
    assert checks["model_sha256_source"] == LS.SHA_CACHED  # the first verify() hashed it fresh


def test_engine_version_errors(fake_binary, tmp_path, monkeypatch):
    assert LS.engine_version(fake_binary) == ENGINE_VERSION
    monkeypatch.setenv("FAKE_VERSION", "llama.cpp something else")
    with pytest.raises(LlamaServerError, match="unrecognised"):
        LS.engine_version(fake_binary)
    with pytest.raises(LlamaServerError, match="cannot run"):
        LS.engine_version(str(tmp_path / "missing-binary"))


def test_find_binary(fake_binary, tmp_path, monkeypatch):
    assert LS.find_binary(fake_binary) == fake_binary
    monkeypatch.setenv("LLAMA_SERVER_BIN", fake_binary)
    assert LS.find_binary() == fake_binary
    monkeypatch.delenv("LLAMA_SERVER_BIN")
    monkeypatch.setattr(LS.shutil, "which", lambda name: fake_binary)
    assert LS.find_binary() == fake_binary
    monkeypatch.setattr(LS.shutil, "which", lambda name: None)
    with pytest.raises(LlamaServerError, match="llama-server not found"):
        LS.find_binary()
    not_executable = tmp_path / "plain"
    not_executable.write_text("x")
    with pytest.raises(LlamaServerError, match="not found"):
        LS.find_binary(str(not_executable))


# --------------------------------------------------------------------------------------------------
# sha256 cache

def test_file_sha256_caches_by_path_size_and_mtime(tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"abc")
    expected = hashlib.sha256(b"abc").hexdigest()
    assert LS.file_sha256(model) == LS.Sha256Reading(expected, LS.SHA_FRESH)
    cache_file = tmp_path / LS.SHA_CACHE_NAME
    cache = json.loads(cache_file.read_text())
    (key,) = cache
    assert key.startswith(f"{model.resolve()}|3|") and cache[key] == expected
    cache_file.write_text(json.dumps({key: "served-from-cache"}))
    assert LS.file_sha256(model) == LS.Sha256Reading("served-from-cache", LS.SHA_CACHED)  # no re-hash while unchanged
    assert LS.file_sha256(model, fresh=True) == LS.Sha256Reading(expected, LS.SHA_FRESH)  # the bench's full re-hash
    assert json.loads(cache_file.read_text()) == {key: expected}  # ...which also repairs the cache
    model.write_bytes(b"abcd")  # size + mtime change → stale entry replaced
    assert LS.file_sha256(model).value == hashlib.sha256(b"abcd").hexdigest()
    assert len(json.loads(cache_file.read_text())) == 1


def test_file_sha256_through_a_symlink_and_explicit_cache(tmp_path):
    target = tmp_path / "real.gguf"
    target.write_bytes(b"weights")
    link = tmp_path / "link.gguf"
    link.symlink_to(target)
    cache = tmp_path / "elsewhere.json"
    assert LS.file_sha256(link, cache).value == hashlib.sha256(b"weights").hexdigest()
    assert next(iter(json.loads(cache.read_text()))).startswith(str(target.resolve()))


@pytest.mark.parametrize("content", ["{not json", "[1, 2]"])
def test_file_sha256_ignores_a_corrupt_cache(tmp_path, content):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"abc")
    (tmp_path / LS.SHA_CACHE_NAME).write_text(content)
    assert LS.file_sha256(model).value == hashlib.sha256(b"abc").hexdigest()


def test_file_sha256_survives_an_unwritable_cache(tmp_path, monkeypatch):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"abc")

    def deny(*args):
        raise OSError("read-only file system")

    monkeypatch.setattr(LS.os, "replace", deny)
    assert LS.file_sha256(model).value == hashlib.sha256(b"abc").hexdigest()


# --------------------------------------------------------------------------------------------------
# Page cache residency

def test_page_cache_residency_of_a_fresh_file(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(os.urandom(64 * 1024))
    fraction = LS.page_cache_residency(path)
    assert fraction is not None and 0.0 <= fraction <= 1.0
    assert fraction == 1.0  # just written, so it is in the page cache


def test_page_cache_residency_unknown_cases(tmp_path, monkeypatch):
    assert LS.page_cache_residency(tmp_path / "missing") is None
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    assert LS.page_cache_residency(empty) is None  # mmap of length 0 fails

    class BrokenLibc:
        def mmap(self, *args):
            return 4096

        def mincore(self, *args):
            return -1

        def munmap(self, *args):
            return 0

    data = tmp_path / "d"
    data.write_bytes(b"x" * 10)
    monkeypatch.setattr(LS, "_libc", BrokenLibc)
    assert LS.page_cache_residency(data) is None
    monkeypatch.delattr(LS.mmap, "MAP_SHARED")  # Windows: no mincore to ask
    assert LS.page_cache_residency(data) is None


def test_evict_page_cache_makes_a_file_cold(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(os.urandom(256 * 1024))
    path.read_bytes()
    assert LS.page_cache_residency(path) == 1.0
    assert LS.evict_page_cache(path) is True
    assert LS.page_cache_residency(path) == 0.0


def test_evict_page_cache_failures(tmp_path, monkeypatch):
    assert LS.evict_page_cache(tmp_path / "missing") is False
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    assert LS.evict_page_cache(empty) is False
    monkeypatch.delattr(LS.os, "posix_fadvise", raising=False)  # the macOS (msync) path, on any system

    class NoMsync:
        def mmap(self, *args):
            return 4096

        def msync(self, *args):
            return -1

        def munmap(self, *args):
            return 0

    data = tmp_path / "d"
    data.write_bytes(b"x" * 10)
    monkeypatch.setattr(LS, "_libc", NoMsync)
    assert LS.evict_page_cache(data) is False
    monkeypatch.delattr(LS.mmap, "MAP_SHARED")  # Windows: neither msync nor posix_fadvise
    assert LS.evict_page_cache(data) is False


def test_evict_page_cache_uses_posix_fadvise_where_it_exists(tmp_path, monkeypatch):
    calls = []
    data = tmp_path / "d"
    data.write_bytes(b"x" * 10)
    monkeypatch.setattr(LS.os, "POSIX_FADV_DONTNEED", 4, raising=False)
    monkeypatch.setattr(LS.os, "posix_fadvise", lambda fd, offset, length, advice: calls.append(advice), raising=False)
    assert LS.evict_page_cache(data) is True and calls == [4]

    def refused(fd, offset, length, advice):
        raise OSError("not supported on this file system")

    monkeypatch.setattr(LS.os, "posix_fadvise", refused)
    assert LS.evict_page_cache(data) is False


@pytest.mark.parametrize("fraction, label", [(None, "unknown"), (1.0, "warm"), (0.995, "warm"),
                                             (0.0, "cold"), (0.5, "partial")])
def test_page_cache_label(fraction, label):
    assert LS.page_cache_label(fraction) == label


# --------------------------------------------------------------------------------------------------
# Health, metrics, log parsing, ports

@pytest.mark.parametrize("response, healthy", [
    (httpx.Response(200, json={"status": "ok"}), True),
    (httpx.Response(503, json={"error": {"message": "Loading model"}}), False),
    (httpx.Response(200, text="not json"), False),
    (httpx.Response(200, json=["ok"]), False),
])
def test_server_healthy(response, healthy):
    transport = httpx.MockTransport(lambda request: response)
    assert LS.server_healthy("http://127.0.0.1:8401/", transport=transport) is healthy


def test_server_healthy_when_unreachable():
    def refuse(request):
        raise httpx.ConnectError("refused")

    assert LS.server_healthy("http://127.0.0.1:1", transport=httpx.MockTransport(refuse)) is False


def test_parse_prometheus():
    text = ("# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.\n"
            "# TYPE llamacpp:prompt_tokens_total counter\n"
            "llamacpp:prompt_tokens_total 3801\n\n"
            "llamacpp:prompt_seconds_total 2.5e-01\n"
            'http_requests{method="POST", path="/v1 chat"} 7 1700000000\n'
            "no_value_line\n"
            "bad_value abc\n"
            "nan_value NaN\n")
    values = LS.parse_prometheus(text)
    assert values["llamacpp:prompt_tokens_total"] == 3801.0
    assert values["llamacpp:prompt_seconds_total"] == 0.25
    assert values['http_requests{method="POST", path="/v1 chat"}'] == 7.0
    assert "no_value_line" not in values and "bad_value" not in values
    assert values["nan_value"] != values["nan_value"]  # NaN
    assert LS.parse_prometheus("") == {}


def test_parse_startup_log_and_missing_log(tmp_path):
    text = ("srv    load_model: initializing, n_slots = 1, n_ctx_slot = 8192, kv_unified = 'false'\n"
            "cmn          init: llama threadpool init, n_threads = 5\n")
    assert LS.parse_startup_log(text) == {"n_slots": 1, "n_ctx_slot": 8192, "kv_unified": False, "n_threads": 5}
    assert LS.parse_startup_log("kv_unified = 'true'") == {"kv_unified": True}
    assert LS.parse_startup_log("") == {}
    server = LlamaServer(get_profile("F"), free_port(), tmp_path / "never-written.log", binary=sys.executable)
    assert server.startup_facts() == {}
    assert server._log_tail() == "(no log)"


def test_ports_and_listening_pid(monkeypatch):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert LS.port_in_use("127.0.0.1", port) is True
        assert LS.pid_listening_on(port) == os.getpid()
    assert LS.port_in_use("127.0.0.1", port) is False
    assert LS.pid_listening_on(port) is None

    def no_lsof(*args, **kwargs):
        raise FileNotFoundError("lsof")

    monkeypatch.setattr(LS.subprocess, "run", no_lsof)
    assert LS.pid_listening_on(port) is None

    def slow_lsof(*args, **kwargs):
        raise LS.subprocess.TimeoutExpired("lsof", 30)

    monkeypatch.setattr(LS.subprocess, "run", slow_lsof)
    assert LS.pid_listening_on(port) is None


def test_without_lsof_the_listening_pid_comes_from_psutil(monkeypatch):
    def no_lsof(*args, **kwargs):
        raise FileNotFoundError("lsof")

    def conn(port, status, pid):
        return SimpleNamespace(laddr=SimpleNamespace(port=port), status=status, pid=pid)

    connections = [conn(8081, "ESTABLISHED", 11), conn(8081, LS.psutil.CONN_LISTEN, None),
                   conn(9000, LS.psutil.CONN_LISTEN, 12), conn(8081, LS.psutil.CONN_LISTEN, 13)]
    monkeypatch.setattr(LS.subprocess, "run", no_lsof)
    monkeypatch.setattr(LS.psutil, "net_connections", lambda kind: connections)
    assert LS.pid_listening_on(8081) == 13
    assert LS.pid_listening_on(7000) is None

    def denied(kind):
        raise LS.psutil.AccessDenied()

    monkeypatch.setattr(LS.psutil, "net_connections", denied)  # macOS without root
    assert LS.pid_listening_on(8081) is None


def test_next_free_port_skips_busy_ports_then_asks_the_os(monkeypatch):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        busy = listener.getsockname()[1]
        assert LS.next_free_port("127.0.0.1", busy) != busy
        assert LS.next_free_port("127.0.0.1", busy, attempts=1) not in (busy, 0)  # the OS picked one
    monkeypatch.setattr(LS, "port_in_use", lambda host, port: True)
    assert LS.next_free_port("127.0.0.1", 65535) > 0  # nothing left above 65535: the OS picks


def test_the_server_runs_in_its_own_process_group_and_stop_takes_the_group(fake_binary, models_dir, tmp_path):
    server = make_server(fake_binary, models_dir, tmp_path, warmup=False)
    server.start()
    pid = server.pid
    assert os.getpgid(pid) == pid != os.getpgid(0)  # a Ctrl+C to our group never reaches it directly
    assert server.stop() == -15
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, 0)


def test_signal_group_ignores_a_process_that_is_already_gone(monkeypatch):
    def gone(pid, sig):
        raise ProcessLookupError(pid)

    monkeypatch.setattr(LS.os, "killpg", gone)
    LS._signal_group(12345, 15)  # no exception


def test_child_env_strips_llama_arg_overrides(monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_CACHE_RAM", "-1")
    monkeypatch.setenv("FAKE_KEEP_ME", "1")
    env = LS._child_env()
    assert "LLAMA_ARG_CACHE_RAM" not in env and env["FAKE_KEEP_ME"] == "1"


def test_models_dir_default_points_at_repo_models():
    assert LS.MODELS_DIR == Path(LS.__file__).resolve().parents[1] / "models"
