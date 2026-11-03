"""Staying up under load (SPEC §10): the first-come, first-served turn line, and load runs started
from the page (bench.load in its own process, calling the server over HTTP like outside customers)."""

from __future__ import annotations

import json
import secrets
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from server.errors import SessionError

ROOT = Path(__file__).resolve().parents[1]


# ================================================================================================
# The turn line: first come, first served


class TurnLine:
    """At most `limit` turns run at once; the next ones wait in arrival order (at most `max_waiting`,
    then join() refuses). A finished turn hands its place straight to the first one waiting."""

    def __init__(self, limit: int, max_waiting: int) -> None:
        self.limit = limit
        self.max_waiting = max_waiting
        self._lock = threading.Lock()
        self._free = limit
        self._waiting: deque[threading.Event] = deque()
        self.refused = 0
        self.peak_waiting = 0

    def join(self) -> threading.Event | None:
        """A ticket that is set once this turn may run; None when the line is full."""
        ticket = threading.Event()
        with self._lock:
            if self._free and not self._waiting:
                self._free -= 1
                ticket.set()
            elif len(self._waiting) >= self.max_waiting:
                self.refused += 1
                return None
            else:
                self._waiting.append(ticket)
                self.peak_waiting = max(self.peak_waiting, len(self._waiting))
        return ticket

    def position(self, ticket: threading.Event) -> int:
        """1 for the first one waiting; 0 once the turn may run."""
        with self._lock:
            return 0 if ticket.is_set() else self._waiting.index(ticket) + 1

    def leave(self) -> None:
        with self._lock:
            if self._waiting:
                self._waiting.popleft().set()
            else:
                self._free += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"max_turns": self.limit, "running": self.limit - self._free, "waiting": len(self._waiting),
                    "peak_waiting": self.peak_waiting, "max_waiting": self.max_waiting, "turns_refused": self.refused}


# ================================================================================================
# Load runs started from the page


MAX_LOAD_CUSTOMERS = 20000
MAX_LOAD_CONCURRENCY = 2000


class LoadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: StrictInt = Field(ge=1, le=MAX_LOAD_CUSTOMERS)
    concurrency: StrictInt = Field(ge=1, le=MAX_LOAD_CONCURRENCY)
    scenarios: list[str] | None = None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class LoadRuns:
    """Load runs started from the page. bench.load runs in its own process and calls this server over
    HTTP, as outside customers would, rewriting a live snapshot that GET /api/load reads. One run at a
    time; the newest is kept so a reloaded page still shows it."""

    def __init__(self, out_dir: Path, launch: Callable[..., Any] = subprocess.Popen) -> None:
        self.out_dir = out_dir
        self.launch = launch
        self._lock = threading.Lock()
        self._run: dict[str, Any] | None = None

    def start(self, url: str, body: LoadRequest) -> dict[str, Any]:
        with self._lock:
            if self._run is not None and self._run["proc"].poll() is None:
                raise SessionError(409, "a load run is still going; stop it or wait for it to finish")
            run_id = f"L-{secrets.token_hex(4)}"
            self.out_dir.mkdir(parents=True, exist_ok=True)
            live, report, log = (self.out_dir / f"{run_id}{suffix}" for suffix in (".live.json", ".json", ".log"))
            argv = [sys.executable, "-m", "bench.load", "--url", url, "--customers", str(body.customers),
                    "--concurrency", str(body.concurrency), "--out", str(report), "--live", str(live)]
            if body.scenarios:
                argv += ["--scenarios", *body.scenarios]
            with log.open("wb") as log_file:  # the child keeps its own copy of the handle
                proc = self.launch(argv, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT)
            self._run = {"run_id": run_id, "customers": body.customers, "concurrency": body.concurrency,
                         "scenarios": body.scenarios, "started": time.time(), "proc": proc, "stopped": False,
                         "live": live, "report": report, "log": log}
        return self.view() or {}

    def view(self) -> dict[str, Any] | None:
        with self._lock:
            run = self._run
        if run is None:
            return None
        running = run["proc"].poll() is None
        report = None if running else _read_json(run["report"])
        if running:
            state = "running"
        elif run["stopped"]:
            state = "stopped"
        else:
            state = "finished" if report is not None else "failed"
        out: dict[str, Any] = {key: run[key] for key in ("run_id", "customers", "concurrency", "scenarios")}
        out.update(state=state, live=_read_json(run["live"]),
                   report={k: v for k, v in report.items() if k != "samples"} if report else None,
                   report_path=str(run["report"].relative_to(ROOT)) if run["report"].is_relative_to(ROOT) else None)
        if state == "failed":
            lines = run["log"].read_text(errors="replace").strip().splitlines() if run["log"].exists() else []
            out["error"] = lines[-1][:300] if lines else "the load tool exited without a report"
        return out

    def stop(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._run
            if run is None or run["run_id"] != run_id:
                raise SessionError(404, f"no load run {run_id}")
            if run["proc"].poll() is None:
                run["stopped"] = True
                run["proc"].terminate()
                try:
                    run["proc"].wait(timeout=5)
                except subprocess.TimeoutExpired:
                    run["proc"].kill()
        return self.view() or {}

    def stop_all(self) -> None:
        with self._lock:
            run = self._run
        if run is not None and run["proc"].poll() is None:
            self.stop(run["run_id"])
