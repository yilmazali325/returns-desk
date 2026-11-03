"""Load runs started from the page: POST/GET /api/load and stop, with a stand-in for the bench.load process."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from server.app import FAKE_POLICY, ServerConfig, create_app

PORT = 8123


class FakeRun:
    """Stands in for the bench.load process: writes what the real one would, exits when told to."""

    def __init__(self, argv: list[str], **options: Any) -> None:
        self.argv, self.options = argv, options
        self.returncode: int | None = None
        self.terminated = False
        self.live = Path(argv[argv.index("--live") + 1])
        self.report = Path(argv[argv.index("--out") + 1])
        self.live.write_text(json.dumps({"done": 3, "in_flight": 2, "turns_finished_per_second": [1, 4]}))

    def finish(self, code: int = 0, report: dict[str, Any] | None = None) -> None:
        if report is not None:
            self.report.write_text(json.dumps(report))
        self.returncode = code

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


def make(tmp_path: Path, port: int | None = PORT) -> tuple[TestClient, list[FakeRun]]:
    app = create_app(ServerConfig(llm_mode=FAKE_POLICY, var_dir=tmp_path / "ui", demo_cases_path=None, port=port))
    runs: list[FakeRun] = []

    def launch(argv: list[str], **options: Any) -> FakeRun:
        runs.append(FakeRun(argv, **options))
        return runs[-1]

    app.state.load_runs.launch = launch
    return TestClient(app, base_url=f"http://127.0.0.1:{port or 8000}"), runs


def test_a_run_starts_bench_load_against_this_server_and_shows_live_numbers(tmp_path: Path) -> None:
    client, runs = make(tmp_path)
    started = client.post("/api/load", json={"customers": 50, "concurrency": 10,
                                             "scenarios": ["public_01_refund_boundary"]})
    assert started.status_code == 200, started.text
    argv = runs[0].argv
    assert argv[:3] == [sys.executable, "-m", "bench.load"]
    assert argv[argv.index("--url") + 1] == f"http://127.0.0.1:{PORT}"
    assert (argv[argv.index("--customers") + 1], argv[argv.index("--concurrency") + 1]) == ("50", "10")
    assert argv[argv.index("--scenarios") + 1:] == ["public_01_refund_boundary"]
    run = client.get("/api/load").json()["run"]
    assert (run["state"], run["customers"], run["live"]["done"]) == ("running", 50, 3)
    again = client.post("/api/load", json={"customers": 5, "concurrency": 1})
    assert again.status_code == 409 and len(runs) == 1  # one run at a time
    runs[0].finish(0, {"completed": 50, "failed": 0, "samples": [{"t": 0.5}]})
    run = client.get("/api/load").json()["run"]
    assert run["state"] == "finished" and run["report"] == {"completed": 50, "failed": 0}  # samples left out
    assert client.post("/api/load", json={"customers": 5, "concurrency": 1}).status_code == 200


def test_stop_ends_the_run_process(tmp_path: Path) -> None:
    client, runs = make(tmp_path)
    run_id = client.post("/api/load", json={"customers": 50, "concurrency": 10}).json()["run_id"]
    stopped = client.post(f"/api/load/{run_id}/stop")
    assert stopped.status_code == 200 and stopped.json()["state"] == "stopped" and runs[0].terminated
    assert client.post("/api/load/L-unknown/stop").status_code == 404


def test_a_run_that_exits_without_a_report_is_failed_with_its_last_log_line(tmp_path: Path) -> None:
    client, runs = make(tmp_path)
    client.post("/api/load", json={"customers": 5, "concurrency": 1})
    Path(runs[0].options["stdout"].name).write_text("Traceback\nboom: no scenarios\n")  # what the process printed
    runs[0].finish(2)
    run = client.get("/api/load").json()["run"]
    assert run["state"] == "failed" and run["error"] == "boom: no scenarios"


def test_bad_requests_are_refused_before_anything_starts(tmp_path: Path) -> None:
    client, runs = make(tmp_path)
    assert client.post("/api/load", json={"customers": 0, "concurrency": 1}).status_code == 422
    assert client.post("/api/load", json={"customers": 5, "concurrency": 5000}).status_code == 422
    assert client.post("/api/load", json={"customers": 5, "concurrency": 1, "scenarios": ["nope"]}).status_code == 422
    assert client.get("/api/load").json() == {"run": None} and runs == []
    unknown_port, _ = make(tmp_path / "other", port=None)
    assert unknown_port.post("/api/load", json={"customers": 5, "concurrency": 1}).status_code == 503
