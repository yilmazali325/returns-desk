"""Start, verify, observe and stop the local llama-server for one profile.

Responsibilities, and why they live here rather than in the adapter:
- Integrity before launch: the model file's sha256 must equal the pinned hash, and `llama-server
  --version` is compared with the pinned engine build (a Homebrew upgrade silently changes template
  handling and caching, so results are not comparable). A different build is refused unless the
  caller allows it; the startup record then says so. Hashing 2.5-4.5 GB takes seconds, so the digest
  is cached in models/.sha256-cache.json keyed by real path + size + mtime_ns, and every reading
  records whether it was hashed fresh or served from that cache.
- Cold start reported separately from warm work: start() returns spawn→/health-ready time and the
  latency of a first tiny request (Metal pipeline warm-up), plus how much of the model file was
  already in the OS page cache (mincore), because "cold" means little if the weights are cached.
  evict_page_cache() (msync MS_INVALIDATE, no sudo) lets the bench measure a true SSD cold start.
- A clean child environment: inherited LLAMA_ARG_* variables would override profile flags without
  showing up in the argv we record, so they are removed.
- Its own process group: the server is started in a new session and stopped with killpg, so a
  terminal Ctrl+C reaches only the parent (which then stops it) and no helper process survives.
- Observation: health(), /metrics scraping (Prometheus text) and facts from the startup log.
"""

from __future__ import annotations

import atexit
import ctypes
import functools
import hashlib
import json
import mmap
import os
import re
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import httpx

from desk.profiles import ENGINE_VERSION, SLOTS, Profile

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
SHA_CACHE_NAME = ".sha256-cache.json"
HEALTH_POLL_S = 0.05
FREE_PORT_ATTEMPTS = 20
MISMATCH_HELP = ("to run it anyway pass --allow-engine-mismatch (or set DESK_ALLOW_ENGINE_MISMATCH=1), "
                 "or start llama-server yourself and pass --llama-url")


class LlamaServerError(RuntimeError):
    """The server could not be verified, started or reached."""


# --------------------------------------------------------------------------------------------------
# Pre-launch checks

def find_binary(explicit: str | None = None) -> str:
    """llama-server path: explicit argument, then $LLAMA_SERVER_BIN, then PATH."""
    candidate = explicit or os.environ.get("LLAMA_SERVER_BIN") or shutil.which("llama-server")
    if not candidate or not os.access(candidate, os.X_OK):
        raise LlamaServerError("llama-server not found: install llama.cpp 0.7.0 (brew install llama.cpp) "
                               "or set LLAMA_SERVER_BIN")
    return candidate


_VERSION_RE = re.compile(r"version:\s*(.+)")


def engine_version(binary: str) -> str:
    """The engine's own version string, e.g. '0.7.0 (build 11669, commit 620d2225c)'."""
    try:
        done = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise LlamaServerError(f"cannot run {binary} --version: {exc}") from exc
    match = _VERSION_RE.search(done.stdout + done.stderr)
    if match is None:
        raise LlamaServerError(f"unrecognised --version output from {binary}")
    return match.group(1).strip()


SHA_FRESH = "fresh"
SHA_CACHED = "cache(path,size,mtime)"


@dataclass(frozen=True)
class Sha256Reading:
    """A file digest and how it was obtained: SHA_FRESH (every byte hashed now) or SHA_CACHED."""

    value: str
    source: str


def file_sha256(path: Path, cache_path: Path | None = None, fresh: bool = False) -> Sha256Reading:
    """sha256 of `path`, served from the on-disk cache while real path, size and mtime are unchanged.
    `fresh` hashes the whole file regardless (the bench does this once per invocation)."""
    real = Path(path).resolve()
    stat = real.stat()
    key = f"{real}|{stat.st_size}|{stat.st_mtime_ns}"
    cache_path = cache_path or Path(path).parent / SHA_CACHE_NAME
    cache = _read_json_dict(cache_path)
    if not fresh and isinstance(cache.get(key), str):
        return Sha256Reading(cache[key], SHA_CACHED)
    digest = hashlib.sha256()
    with real.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    # Keep one entry per file: entries for an older size/mtime of the same path are stale.
    cache = {k: v for k, v in cache.items() if not k.startswith(f"{real}|")}
    cache[key] = value
    tmp = cache_path.with_name(cache_path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(cache, indent=1, sort_keys=True))
        os.replace(tmp, cache_path)
    except OSError:
        pass  # a read-only models dir only costs a re-hash next time
    return Sha256Reading(value, SHA_FRESH)


def _read_json_dict(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


@functools.lru_cache(maxsize=1)
def _libc() -> ctypes.CDLL:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                          ctypes.c_longlong]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    return libc


def page_cache_residency(path: Path) -> float | None:
    """Fraction of the file's pages resident in the OS page cache (mmap + mincore); None if unknown.

    Mapping a file does not read it, so the probe itself does not warm the cache.
    """
    try:
        size = os.path.getsize(path)
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    libc = _libc()
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr is None or addr == ctypes.c_void_p(-1).value:  # MAP_FAILED (e.g. empty file)
            return None
        try:
            pages = -(-size // mmap.PAGESIZE)
            vec = ctypes.create_string_buffer(pages)
            if libc.mincore(addr, size, vec) != 0:
                return None
            return sum(byte & 1 for byte in vec.raw) / pages
        finally:
            libc.munmap(addr, size)
    finally:
        os.close(fd)


MS_INVALIDATE = 0x2  # <sys/mman.h> on macOS


def evict_page_cache(path: Path) -> bool:
    """Drop the file's clean pages from the OS page cache (msync MS_INVALIDATE; no sudo needed).

    Used only by the bench to measure a true cold start (weights read from SSD); `sudo purge` would
    evict everything on the machine instead.
    """
    try:
        size = os.path.getsize(path)
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    libc = _libc()
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr is None or addr == ctypes.c_void_p(-1).value:
            return False
        try:
            return libc.msync(addr, size, MS_INVALIDATE) == 0
        finally:
            libc.munmap(addr, size)
    finally:
        os.close(fd)


def page_cache_label(fraction: float | None) -> str:
    if fraction is None:
        return "unknown"
    return "warm" if fraction >= 0.99 else "cold" if fraction <= 0.01 else "partial"


# --------------------------------------------------------------------------------------------------
# Observation helpers

def server_healthy(base_url: str, timeout_s: float = 2.0,
                   transport: httpx.BaseTransport | None = None) -> bool:
    """True iff GET /health answers 200 with status ok (503 while the model is loading)."""
    try:
        with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s, transport=transport) as http:
            response = http.get("/health")
            return response.status_code == 200 and response.json().get("status") == "ok"
    except (httpx.HTTPError, ValueError, AttributeError):
        return False


def parse_prometheus(text: str) -> dict[str, float]:
    """Prometheus text exposition → {metric name (with labels): value}; comments are skipped."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Label values may contain spaces, so split after the closing brace when there is one.
        cut = line.rindex("}") + 1 if "}" in line else line.find(" ")
        if cut <= 0:
            continue
        name, rest = line[:cut], line[cut:].split()
        try:
            values[name] = float(rest[0])
        except (ValueError, IndexError):
            continue
    return values


_LOG_FACTS = {
    "n_slots": re.compile(r"n_slots = (\d+)"),
    "n_ctx_slot": re.compile(r"n_ctx_slot = (\d+)"),
    "kv_unified": re.compile(r"kv_unified = '(\w+)'"),
    "n_threads": re.compile(r"threadpool init, n_threads = (\d+)"),
}


def parse_startup_log(text: str) -> dict[str, Any]:
    """Facts the server prints at startup (the effective values, not what we asked for)."""
    facts: dict[str, Any] = {}
    for name, pattern in _LOG_FACTS.items():
        match = pattern.search(text)
        if match is not None:
            value = match.group(1)
            facts[name] = int(value) if value.isdigit() else value == "true"
    return facts


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def next_free_port(host: str, start: int, attempts: int = FREE_PORT_ATTEMPTS) -> int:
    """`start` if nothing listens there, else the next free port above it, else one the OS picks."""
    for port in range(start, min(start + attempts, 65536)):
        if not port_in_use(host, port):
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


LSOF_TIMEOUT_S = 30.0  # lsof walks every process's files: seconds, not milliseconds, on a busy machine


def pid_listening_on(port: int) -> int | None:
    """PID of the process listening on a local TCP port (via lsof), e.g. an attached server."""
    try:
        done = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                              capture_output=True, text=True, timeout=LSOF_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    first = done.stdout.split()
    return int(first[0]) if first and first[0].isdigit() else None


def _child_env() -> dict[str, str]:
    """Inherited environment without LLAMA_ARG_* overrides (they would bypass the recorded argv)."""
    return {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}


# --------------------------------------------------------------------------------------------------
# The server

class LlamaServer:
    """One llama-server process for one profile on one port.

    Typical use: `with LlamaServer(profile, 8401, log_path) as server: startup = server.startup`.
    """

    def __init__(self, profile: Profile, port: int, log_path: Path, *, binary: str | None = None,
                 models_dir: Path = MODELS_DIR, host: str = "127.0.0.1", ready_timeout_s: float = 180.0,
                 expected_sha256: str | None = None, expected_engine_version: str = ENGINE_VERSION,
                 allow_engine_mismatch: bool = False, warmup: bool = True, slots: int = SLOTS) -> None:
        self.profile = profile
        self.slots = slots
        self.port = port
        self.host = host
        self.log_path = Path(log_path)
        self.model_path = Path(models_dir) / profile.model_file
        self.binary = find_binary(binary)
        self.ready_timeout_s = ready_timeout_s
        self.expected_sha256 = expected_sha256 or profile.model_sha256
        self.expected_engine_version = expected_engine_version
        self.allow_engine_mismatch = allow_engine_mismatch
        self.warmup = warmup
        self.engine_version: str | None = None
        self.model_sha256: str | None = None
        self.model_sha256_source: str | None = None
        self.startup: dict[str, Any] = {}
        self._proc: subprocess.Popen[bytes] | None = None
        self._log: IO[bytes] | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    @property
    def argv(self) -> list[str]:
        return self.profile.engine_argv(self.binary, str(self.model_path), self.port, self.slots)

    def verify(self) -> dict[str, Any]:
        """Model hash + engine version checks (run by start(); callable alone for preflight)."""
        if not self.model_path.exists():
            raise LlamaServerError(f"model file missing: {self.model_path} (run scripts/setup.sh)")
        reading = file_sha256(self.model_path)
        self.model_sha256, self.model_sha256_source = reading.value, reading.source
        if self.model_sha256 != self.expected_sha256:
            raise LlamaServerError(f"sha256 mismatch for {self.model_path.name}: got {self.model_sha256}, "
                                   f"expected {self.expected_sha256}")
        self.engine_version = engine_version(self.binary)
        match = self.engine_version == self.expected_engine_version
        if not match and not self.allow_engine_mismatch:
            raise LlamaServerError(f"engine version {self.engine_version!r} is not the pinned "
                                   f"{self.expected_engine_version!r}: {MISMATCH_HELP}")
        return {
            "engine_version": self.engine_version,
            "engine_version_actual": self.engine_version,
            "engine_version_expected": self.expected_engine_version,
            "engine_version_match": match,
            "engine_version_mismatch_allowed": not match,
            "model_sha256": self.model_sha256,
            "model_sha256_source": self.model_sha256_source,
        }

    def start(self) -> dict[str, Any]:
        """Verify, spawn, wait for /health, send one warm-up request; returns the startup record."""
        if self._proc is not None:
            raise LlamaServerError("server already started")
        checks = self.verify()
        if port_in_use(self.host, self.port):
            raise LlamaServerError(f"port {self.port} is already in use (pid {pid_listening_on(self.port)})")
        # Measured at spawn time, after the hash check: a first-time hash reads (and so warms) the file.
        residency = page_cache_residency(self.model_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("wb")
        spawned = time.perf_counter()
        self._proc = subprocess.Popen(self.argv, stdout=self._log, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL, env=_child_env(), start_new_session=True)
        atexit.register(self.stop)
        ready_ms = self._wait_ready(spawned)
        first_ms = self._first_request_ms() if self.warmup else None
        self.startup = {
            "server_spawn_to_ready_ms": ready_ms,
            "first_request_ms": first_ms,
            "warmup": self.warmup,
            "model_page_cache": page_cache_label(residency),
            "model_page_cache_resident_pct": None if residency is None else round(100 * residency, 1),
            **checks,
            "server_log_facts": self.startup_facts(),
        }
        return self.startup

    def _wait_ready(self, spawned: float) -> float:
        deadline = spawned + self.ready_timeout_s
        while time.perf_counter() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                code = self._proc.returncode
                tail = self._log_tail()
                self.stop()
                raise LlamaServerError(f"llama-server exited with code {code} during startup:\n{tail}")
            if self.health():
                return round((time.perf_counter() - spawned) * 1000, 1)
            time.sleep(HEALTH_POLL_S)
        self.stop()
        raise LlamaServerError(f"llama-server not healthy after {self.ready_timeout_s:.0f} s")

    def _first_request_ms(self) -> float:
        """Latency of a 1-token request: first Metal pipeline compile and graph allocation."""
        body = {"messages": [{"role": "user", "content": "ping"}], "max_tokens": 1, "temperature": 0,
                "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
        started = time.perf_counter()
        try:
            response = httpx.post(f"{self.base_url}/v1/chat/completions", json=body, timeout=120)
        except httpx.HTTPError as exc:
            raise LlamaServerError(f"warm-up request failed: {exc!r}") from exc
        if response.status_code != 200:
            raise LlamaServerError(f"warm-up request answered HTTP {response.status_code}")
        return round((time.perf_counter() - started) * 1000, 1)

    def stop(self, timeout_s: float = 10.0) -> int | None:
        """SIGTERM (then SIGKILL) our process group; idempotent. Returns the exit code if we stopped one."""
        atexit.unregister(self.stop)
        proc, self._proc = self._proc, None
        code = None
        if proc is not None:
            if proc.poll() is None:
                _signal_group(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    _signal_group(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=timeout_s)
            code = proc.returncode
        if self._log is not None:
            self._log.close()
            self._log = None
        return code

    def __enter__(self) -> LlamaServer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def health(self) -> bool:
        return server_healthy(self.base_url, timeout_s=1.0)

    def metrics_text(self) -> str:
        """Raw /metrics (saved as metrics_before/after.prom by the bench)."""
        try:
            response = httpx.get(f"{self.base_url}/metrics", timeout=5)
        except httpx.HTTPError as exc:
            raise LlamaServerError(f"/metrics unreachable: {exc!r}") from exc
        if response.status_code != 200:
            raise LlamaServerError(f"/metrics answered HTTP {response.status_code} (needs --metrics)")
        return response.text

    def metrics(self) -> dict[str, float]:
        return parse_prometheus(self.metrics_text())

    def startup_facts(self) -> dict[str, Any]:
        try:
            return parse_startup_log(self.log_path.read_text(errors="replace"))
        except OSError:
            return {}

    def _log_tail(self, lines: int = 20) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return "(no log)"


def _signal_group(pid: int, sig: int) -> None:
    """Signal the process group a start_new_session child leads (its pgid is its pid)."""
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass  # already gone between poll() and here
