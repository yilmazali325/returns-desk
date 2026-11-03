"""bench/load.py against the in-process server on the test double: concurrent customers, the report."""

from __future__ import annotations

from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from bench.load import (format_summary, live_snapshot, machine_peaks, percentile, read_turn, run_customer,
                        run_load, timeline)
from desk.llm import CircuitBreaker, LLMClient
from server.app import FAKE_POLICY, ServerConfig, create_app


def test_concurrent_customers_all_finish_with_consistent_outcomes(tmp_path: Path) -> None:
    app = create_app(ServerConfig(llm_mode=FAKE_POLICY, var_dir=tmp_path / "ui", demo_cases_path=None))
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        report = run_load(client, customers=12, concurrency=4)
    assert (report["completed"], report["failed"], report["errors"]) == (12, 0, {})
    assert report["duplicate_refunds_or_exchanges"] == 0
    assert report["turns"] == 2 * 13  # each of the 6 public cases twice; they have 13 turns together
    assert all(v["distinct_outcomes"] == 1 for v in report["scenarios"].values())
    assert report["scenarios"]["public_01_refund_boundary"]["outcomes"] == {"propose → completed": 2}
    assert report["latency_client_ms"]["p50"] is not None and report["turns_per_s"] > 0
    assert "12 ok, 0 failed" in format_summary(report)


def test_a_turn_stream_without_a_result_is_a_failure() -> None:
    lines = ["event: start", 'data: {"turn": 1}', "", "event: error", 'data: {"message": "boom"}']
    try:
        read_turn(iter(lines))
    except Exception as exc:  # noqa: BLE001 - the type is the point
        assert type(exc).__name__ == "TurnFailed" and "boom" in str(exc)
    else:
        raise AssertionError("an error event must fail the turn")


def test_percentile_is_nearest_rank() -> None:
    assert percentile([], 50) is None
    assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], 50) == 3.0
    assert percentile(list(map(float, range(1, 101))), 95) == 95.0


def test_a_turn_refused_at_capacity_is_retried_after_retry_after(tmp_path: Path, monkeypatch) -> None:
    app = create_app(ServerConfig(llm_mode=FAKE_POLICY, var_dir=tmp_path / "ui", demo_cases_path=None, max_turns=1,
                                  max_waiting=0))
    waits: list[float] = []

    def wait(seconds: float) -> None:  # the other conversation finishes while this customer waits
        waits.append(seconds)
        app.state.turn_line.leave()

    monkeypatch.setattr("bench.load.time.sleep", wait)
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        scenario = next(s for s in client.get("/api/scenarios").json() if s["case_id"] == "public_01_refund_boundary")
        app.state.turn_line.join()
        result = run_customer(client, 0, scenario, 0.0)
    assert result["error"] is None and result["refused"] == 1 and waits == [2.0]
    assert [t["decision"] for t in result["turns"]] == ["propose", "completed"]


def test_with_the_model_down_every_turn_ends_in_a_clean_error_and_the_breaker_stops_the_calls(tmp_path: Path) -> None:
    attempts: list[str] = []

    def down(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.path)
        raise httpx.ConnectError("connection refused")

    breaker = CircuitBreaker(failures=3, open_s=600)  # stays open for the whole test
    model = LLMClient("http://model", "constrained", transport=httpx.MockTransport(down), retry_backoff_s=0,
                      max_in_flight=1, breaker=breaker)
    app = create_app(ServerConfig(var_dir=tmp_path / "ui", demo_cases_path=None), llm=model)
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        report = run_load(client, customers=12, concurrency=1)
    assert (report["completed"], report["failed"], report["duplicate_refunds_or_exchanges"]) == (12, 0, 0)
    assert set(report["decisions"]) == {"error"} and report["decisions"]["error"] == report["turns"]
    assert len(attempts) == 3 * 2  # 3 failed calls (each tried twice), then the breaker answers alone
    assert breaker.snapshot()["state"] == "open"


def test_live_numbers_count_the_turns_of_customers_still_talking() -> None:
    talking = {"turns": [{"end": 1.2, "client_ms": 900.0}], "error": None, "duplicates": 0, "refused": 0}
    snapshot = live_snapshot([talking], [], [], 2.0, customers=5, concurrency=5, in_flight=1)
    assert (snapshot["done"], snapshot["turns"], snapshot["turns_per_s"]) == (0, 1, 0.5)
    assert snapshot["latency_client_ms"]["p50"] == 900.0 and snapshot["turns_finished_per_second"] == [0, 1, 0]


def test_reply_time_line_is_the_median_of_the_last_ten_seconds() -> None:
    from bench.load import reply_p50_by_second
    turns = [{"end": 0.5, "client_ms": 100.0}, {"end": 3.2, "client_ms": 300.0}, {"end": 12.0, "client_ms": 900.0}]
    line = reply_p50_by_second(turns, 14)
    assert line[0] == 100.0 and line[3] == 100.0 and line[9] == 100.0  # nearest-rank median of [100, 300]
    assert line[10] == 300.0 and line[12] == 300.0 and line[13] == 900.0


def test_machine_numbers_follow_the_run_and_peak_in_the_report() -> None:
    reads = [{"cpu_pct": 20.0, "server_cpu_pct": 1.0, "llama_cpu_pct": 3.0, "gpu_busy_pct": 90, "memory_used_mb": 12000.0,
              "memory_total_mb": 24576.0, "footprint_mb": 3000.0},
             {"cpu_pct": 35.0, "server_cpu_pct": None, "llama_cpu_pct": 2.0, "gpu_busy_pct": 70, "memory_used_mb": 12500.0,
              "memory_total_mb": 24576.0, "footprint_mb": 3100.0}]
    samples = [{"t": 0.5, "in_flight": 2}, {"t": 1.0, "in_flight": 2, "memory": reads[0]},
               {"t": 2.0, "in_flight": 1, "memory": reads[1]}, {"t": 3.0, "in_flight": 0, "memory": None}]
    points = timeline(samples)
    assert [(p["cpu"], p["gpu"], p["mem_used_mb"], p["footprint_mb"]) for p in points] == [
        (20.0, 90, 12000.0, 3000.0), (35.0, 70, 12500.0, 3100.0), (None, None, None, None)]
    assert machine_peaks(samples) == {"cpu_pct": 35.0, "server_cpu_pct": 1.0, "llama_cpu_pct": 3.0, "gpu_busy_pct": 90,
                                      "memory_used_mb": 12500.0, "memory_total_mb": 24576.0}
    snapshot = live_snapshot([], [], samples[:3], 2.0, customers=2, concurrency=2, in_flight=1)
    assert snapshot["machine"]["cpu_pct"] == 35.0 and snapshot["machine"]["memory_used_mb"] == 12500.0
