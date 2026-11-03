"""Load test: many customers at once against a running desk, through the UI server's HTTP API.

    scripts/run.sh --fake --port 8010            # the desk on the instant test double (or the real model)
    .venv/bin/python -m bench.load --url http://127.0.0.1:8010 --customers 1000 --concurrency 100

Each simulated customer opens its own session for one scenario (the server's public cases and demo
chats, taken in turn) and sends its scripted messages in order, confirming the way the page does
(proposal id + binding hash + generation). Nothing is shared between customers but the server.

Telemetry in the report (var/load/load-<UTC time>.json) and the printed summary:
- speed: wall time, customers and turns per second, turns finished in each second of the run;
- latency per turn: client side (request to final event) and server side, p50/p95/p99/max;
- load: customers in flight (sampled every 0.5 s); the server's memory, model calls waiting for a
  slot and at the model (sampled every 2 s); turns refused at capacity (503) and retried;
- outcome: decision mix, errors by kind;
- correctness: no order line refunded or exchanged twice, and every copy of a scenario reached the
  same decisions (expected on the test double; the real model may legitimately vary).

With --live PATH, a snapshot of the numbers so far is rewritten every 0.5 s (the page's Load test
panel reads it while the run goes on).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "var" / "load"
SAMPLE_S = 0.5  # customers-in-flight sampling period
MEMORY_EVERY = 2  # the server's telemetry is read every 2nd sample (1 s): cheap but not free
MAX_POINTS = 300  # a timeline longer than this keeps every n-th point
REPLY_WINDOW_S = 10  # the reply-time line is the median of the turns that ended in the last 10 s
MAX_REFUSALS = 300  # a turn refused at capacity (503) is retried after Retry-After, up to this many times


class TurnFailed(Exception):
    """A turn that did not end with a result: an HTTP refusal, an `error` event or a cut stream."""


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (as in docs/RESULTS.md); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def read_turn(lines: Any) -> dict[str, Any]:
    """The `turn` event of a turn's event stream; an `error` event or a stream without one raises."""
    event = None
    for line in lines:
        if line.startswith("event: "):
            event = line[len("event: "):]
        elif line.startswith("data: ") and event in ("turn", "error"):
            data = json.loads(line[len("data: "):])
            if event == "error":
                raise TurnFailed(f"server error: {data.get('message', '')[:120]}")
            return data
    raise TurnFailed("stream ended without a result")


def run_customer(client: httpx.Client, index: int, scenario: dict[str, Any], t0: float,
                 out: dict[str, Any] | None = None) -> dict[str, Any]:
    """One customer's whole conversation. Times are seconds since the run started. Pass `out` to
    watch its turns arrive while the conversation goes on."""
    out = {} if out is None else out
    out.update({"customer": index, "scenario": scenario["case_id"], "turns": [], "error": None,
                "duplicates": 0, "refused": 0})
    try:
        created = client.post("/api/sessions", json={"case_id": scenario["case_id"]})
        if created.status_code != 200:
            raise TurnFailed(f"create session: HTTP {created.status_code}")
        session_id, state = created.json()["session_id"], created.json()["state"]
        proposal_id = None
        for turn in scenario["turns"]:
            body: dict[str, Any] = {"message": turn["message"]}
            if turn.get("confirm_proposal") == "last" and proposal_id in state.get("cards", {}):
                body.update(confirm_proposal_id=proposal_id, generation=state["generation"],
                            confirm_binding_hash=state["cards"][proposal_id]["binding_hash"])
            started = time.perf_counter() - t0  # the customer's wait includes any refusals
            record = None
            while record is None:
                with client.stream("POST", f"/api/sessions/{session_id}/turns", json=body) as response:
                    if response.status_code == 503 and out["refused"] < MAX_REFUSALS:
                        response.read()
                        out["refused"] += 1
                        time.sleep(float(response.headers.get("Retry-After", "1")))
                        continue
                    if response.status_code != 200:
                        response.read()
                        raise TurnFailed(f"turn: HTTP {response.status_code}")
                    record = read_turn(response.iter_lines())
            ended = time.perf_counter() - t0
            output = record.get("output") or {}
            state = record["state"]
            if output.get("proposal"):
                proposal_id = output["proposal"]["proposal_id"]
            out["turns"].append({"start": round(started, 4), "end": round(ended, 4),
                                 "client_ms": round((ended - started) * 1000, 1),
                                 "server_ms": record.get("latency_ms"), "decision": output.get("decision")})
        final = state["final_state"]
        lines = [(r["order_id"], r["line_id"]) for key in ("refunds", "exchanges") for r in final.get(key, [])]
        out["duplicates"] = len(lines) - len(set(lines))
    except (TurnFailed, httpx.HTTPError, KeyError, ValueError) as exc:
        out["error"] = str(exc) if isinstance(exc, TurnFailed) else f"{type(exc).__name__}: {str(exc)[:120]}"
    return out


def timeline(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Where the customers were over time: in flight (load tool), running and waiting in the server's
    turn line, and the model's calls (at the model, waiting for a slot). One point per telemetry read."""
    points = [{"t": s["t"], "in_flight": s["in_flight"], "running": (s.get("line") or {}).get("running"),
               "waiting": (s.get("line") or {}).get("waiting"), "model_busy": (s.get("model") or {}).get("in_flight"),
               "model_waiting": (s.get("model") or {}).get("waiting")} for s in samples if "memory" in s]
    return points[::max(1, math.ceil(len(points) / MAX_POINTS))]


def reply_p50_by_second(turns: list[dict[str, Any]], seconds: int) -> list[float | None]:
    """For each second: the median reply time of the turns that ended in the REPLY_WINDOW_S before it."""
    by_second: dict[int, list[float]] = defaultdict(list)
    for t in turns:
        by_second[int(t["end"])].append(t["client_ms"])
    return [percentile([v for back in range(REPLY_WINDOW_S) for v in by_second.get(s - back, [])], 50)
            for s in range(seconds)]


def live_snapshot(started: list[dict[str, Any]], finished: list[dict[str, Any]], samples: list[dict[str, Any]],
                  elapsed: float, customers: int, concurrency: int, in_flight: int) -> dict[str, Any]:
    """The numbers so far, for a page to show while the run goes on. Turns count as soon as they end,
    also those of customers still talking."""
    turns = [t for r in started for t in list(r.get("turns", []))]
    client_ms = [t["client_ms"] for t in turns]
    per_second = Counter(int(t["end"]) for t in turns)
    model = next((s["model"] for s in reversed(samples) if s.get("model")), None)
    memory = next((s["memory"] for s in reversed(samples) if s.get("memory")), None)
    return {
        "elapsed_s": round(elapsed, 1), "customers": customers, "concurrency": concurrency,
        "done": len(finished), "failed": sum(1 for r in finished if r["error"]), "in_flight": in_flight,
        "turns": len(turns), "turns_per_s": round(len(turns) / elapsed, 2) if elapsed else None,
        "latency_client_ms": {"p50": percentile(client_ms, 50), "p95": percentile(client_ms, 95)},
        "refused": max(sum(r["refused"] for r in finished),
                       next((s["refused"] for s in reversed(samples) if "refused" in s), 0)),
        "duplicates": sum(r["duplicates"] for r in finished),
        "turns_finished_per_second": [per_second.get(s, 0) for s in range(int(elapsed) + 1)],
        "reply_p50_ms_by_second": reply_p50_by_second(turns, int(elapsed) + 1),
        "timeline": timeline(samples),
        "line": next((s["line"] for s in reversed(samples) if s.get("line")), None),
        "model": model, "server_footprint_mb": (memory or {}).get("server_footprint_mb"),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write whole or not at all: a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def run_load(client: httpx.Client, customers: int, concurrency: int, case_ids: list[str] | None = None,
             progress: bool = False, live: Path | None = None) -> dict[str, Any]:
    """Run `customers` conversations with at most `concurrency` in flight; return the report."""
    scenarios = [s for s in client.get("/api/scenarios").json() if not case_ids or s["case_id"] in case_ids]
    if not scenarios:
        raise SystemExit("no scenarios to run (check --scenarios)")
    telemetry = client.get("/api/telemetry").json()
    mode = telemetry.get("mode")
    refused_before = (telemetry.get("capacity") or {}).get("turns_refused", 0)  # the server counts refusals live
    in_flight = 0
    lock = threading.Lock()
    started: list[dict[str, Any]] = []  # every customer's result, filled in as its turns end
    finished: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    stop = threading.Event()
    t0 = time.perf_counter()

    def one(index: int) -> None:
        nonlocal in_flight
        result: dict[str, Any] = {}
        with lock:
            in_flight += 1
            started.append(result)
        run_customer(client, index, scenarios[index % len(scenarios)], t0, result)
        with lock:
            in_flight -= 1
            finished.append(result)

    def sampler() -> None:
        tick = 0
        while not stop.wait(SAMPLE_S):
            sample: dict[str, Any] = {"t": round(time.perf_counter() - t0, 2), "in_flight": in_flight,
                                      "done": len(finished)}
            if tick % MEMORY_EVERY == 0:
                try:
                    telemetry = client.get("/api/telemetry").json()
                    sample["memory"] = telemetry.get("memory")
                    sample["model"] = (telemetry.get("capacity") or {}).get("model")
                    sample["refused"] = (telemetry.get("capacity") or {}).get("turns_refused", 0) - refused_before
                    if (telemetry.get("capacity") or {}).get("max_turns") is not None:
                        capacity = telemetry["capacity"]
                        sample["line"] = {k: capacity.get(k) for k in ("max_turns", "running", "waiting", "peak_waiting")}
                except httpx.HTTPError:
                    sample["memory"] = None
            samples.append(sample)
            if live is not None:
                with lock:
                    snapshot = live_snapshot(list(started), list(finished), samples, sample["t"], customers,
                                             concurrency, in_flight)
                write_json(live, snapshot)
            if progress and tick % 4 == 0:
                errors = sum(1 for r in finished if r["error"])
                print(f"  {sample['t']:7.1f}s  done {sample['done']}/{customers}  in flight {in_flight}"
                      f"  errors {errors}", file=sys.stderr)
            tick += 1

    watcher = threading.Thread(target=sampler, daemon=True)
    watcher.start()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(one, range(customers)))
    wall = time.perf_counter() - t0
    stop.set()
    watcher.join()
    return build_report(finished, samples, wall, customers, concurrency, mode)


def build_report(results: list[dict[str, Any]], samples: list[dict[str, Any]], wall: float, customers: int,
                 concurrency: int, mode: str | None) -> dict[str, Any]:
    turns = [t for r in results for t in r["turns"]]
    client_ms = [t["client_ms"] for t in turns]
    server_ms = [t["server_ms"] for t in turns if isinstance(t["server_ms"], (int, float))]
    per_second = Counter(int(t["end"]) for t in turns)
    by_scenario: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    for r in results:
        if not r["error"]:
            by_scenario[r["scenario"]][tuple(t["decision"] for t in r["turns"])] += 1
    footprints = [s["memory"].get("server_footprint_mb") for s in samples if s.get("memory")]
    footprints = [f for f in footprints if isinstance(f, (int, float))]
    models = [s["model"] for s in samples if s.get("model")]

    def latency(values: list[float]) -> dict[str, float | None]:
        return {"p50": percentile(values, 50), "p95": percentile(values, 95), "p99": percentile(values, 99),
                "max": max(values) if values else None}

    return {
        "when": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": mode,
        "customers": customers,
        "concurrency": concurrency,
        "completed": sum(1 for r in results if not r["error"]),
        "failed": sum(1 for r in results if r["error"]),
        "turns": len(turns),
        "wall_s": round(wall, 2),
        "customers_per_s": round(customers / wall, 2) if wall else None,
        "turns_per_s": round(len(turns) / wall, 2) if wall else None,
        "latency_client_ms": latency(client_ms),
        "latency_server_ms": latency(server_ms),
        "peak_in_flight": max((s["in_flight"] for s in samples), default=None),
        "server_footprint_mb_peak": max(footprints) if footprints else None,
        "model_slots": models[-1]["slots"] if models else None,
        "peak_model_calls_waiting": max((m["waiting"] for m in models), default=None),
        "max_model_queue_wait_ms": models[-1]["max_queue_wait_ms"] if models else None,
        "breaker": models[-1].get("breaker") if models else None,
        "turns_refused_at_capacity": sum(r["refused"] for r in results),
        "decisions": dict(Counter(t["decision"] for t in turns)),
        "errors": dict(Counter(r["error"] for r in results if r["error"])),
        "duplicate_refunds_or_exchanges": sum(r["duplicates"] for r in results),
        "scenarios": {case: {"customers": sum(seqs.values()), "distinct_outcomes": len(seqs),
                             "outcomes": {" → ".join(seq): n for seq, n in seqs.most_common()}}
                      for case, seqs in sorted(by_scenario.items())},
        "turns_finished_per_second": [per_second.get(s, 0) for s in range(int(wall) + 1)],
        "reply_p50_ms_by_second": reply_p50_by_second(turns, int(wall) + 1),
        "timeline": timeline(samples),
        "peak_waiting_in_line": max(((s.get("line") or {}).get("waiting") or 0 for s in samples), default=None),
        "turn_line_limit": next((s["line"]["max_turns"] for s in reversed(samples) if s.get("line")), None),
        "samples": samples,
    }


def format_summary(report: dict[str, Any]) -> str:
    def ms(value: float | None) -> str:
        return "–" if value is None else f"{value:,.0f} ms"

    c, s = report["latency_client_ms"], report["latency_server_ms"]
    rows = [
        f"Load test ({report['mode']}): {report['customers']} customers, {report['concurrency']} at a time",
        f"  finished      {report['completed']} ok, {report['failed']} failed, {report['turns']} turns "
        f"in {report['wall_s']} s",
        f"  throughput    {report['customers_per_s']} customers/s, {report['turns_per_s']} turns/s",
        f"  turn latency  client p50 {ms(c['p50'])}, p95 {ms(c['p95'])}, p99 {ms(c['p99'])}, max {ms(c['max'])}",
        f"                server p50 {ms(s['p50'])}, p95 {ms(s['p95'])}, p99 {ms(s['p99'])}",
        f"  load          peak {report['peak_in_flight']} customers in flight, server memory peak "
        f"{report['server_footprint_mb_peak'] or '–'} MB, {report['turns_refused_at_capacity']} turns refused "
        "at capacity (retried)",
        f"  decisions     {', '.join(f'{k} {v}' for k, v in sorted(report['decisions'].items(), key=str))}",
        f"  correctness   {report['duplicate_refunds_or_exchanges']} duplicate refunds/exchanges; "
        + ", ".join(f"{case} {v['distinct_outcomes']} outcome(s)" for case, v in report["scenarios"].items()),
    ]
    if report["model_slots"] is not None:
        breaker = report["breaker"] or {}
        rows.append(f"  model         {report['model_slots']} slot(s), peak {report['peak_model_calls_waiting']} calls "
                    f"waiting, longest slot wait since start {ms(report['max_model_queue_wait_ms'])}, breaker "
                    f"{breaker.get('state', '–')} (opened {breaker.get('times_opened', 0)}×)")
    if report["errors"]:
        rows.append("  errors        " + "; ".join(f"{k} ×{v}" for k, v in report["errors"].items()))
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Many customers at once against a running desk.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="the desk's UI server")
    parser.add_argument("--customers", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=50, help="customers in flight at most")
    parser.add_argument("--scenarios", nargs="*", help="case ids to use (default: every scenario listed)")
    parser.add_argument("--out", type=Path, help="report path (default var/load/load-<UTC time>.json)")
    parser.add_argument("--live", type=Path, help="rewrite a snapshot of the numbers so far here every 0.5 s")
    args = parser.parse_args(argv)
    limits = httpx.Limits(max_connections=args.concurrency + 4, max_keepalive_connections=args.concurrency + 4)
    with httpx.Client(base_url=args.url, timeout=httpx.Timeout(900, connect=10), limits=limits) as client:
        report = run_load(client, args.customers, args.concurrency, args.scenarios, progress=True, live=args.live)
    out = args.out or OUT_DIR / f"load-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.json"
    write_json(out, report)
    print(format_summary(report))
    print(f"report: {out}")
    return 0 if not report["failed"] and not report["duplicate_refunds_or_exchanges"] else 1


if __name__ == "__main__":
    sys.exit(main())
