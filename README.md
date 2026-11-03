# Returns Desk: a local, agentic retail returns and exchanges desk

Two cooperating LLM agents run a returns desk entirely on one laptop:

- **Resolution** is customer-facing. It reads the conversation, picks the item, remedy and size,
  decides whether to ask or act, and writes the reply.
- **Operations** is the back office. It checks exchange options, creates the proposal and, after
  the customer's explicit confirmation, executes it.

Deterministic code owns identity, policy, money, stock, commits, idempotency and recovery. Its
design rule is **agents say yes; code can say no**: code can block, read or replay, but it never
starts a business action for an agent. The web UI shows the whole team working live:
cards for each agent and code check, handoffs, tool calls, blocks and the audit trail.

Everything runs locally: Qwen3.5-4B on llama.cpp (Metal), FastAPI, vanilla JS and SQLite. After
setup the demo, the adapter and the tests run with the network off
([offline proof](results/offline/OFFLINE_PROOF.md)).

| Read next | What it covers |
|---|---|
| [RUN_ON_HP.md](RUN_ON_HP.md) | step-by-step setup on an HP Z workstation: Linux or Windows (WSL2), NVIDIA GPU |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | agent boundaries, data and tool flow, failure handling |
| [docs/RESULTS.md](docs/RESULTS.md) | measured baseline vs final, kept apart from proposed improvements |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | known limitations |
| [docs/AI_DISCLOSURE.md](docs/AI_DISCLOSURE.md) | AI tools used to build this |
| [docs/CLARIFICATIONS.md](docs/CLARIFICATIONS.md) | questions for the interviewers and the defaults implemented |
| [docs/SPEC.md](docs/SPEC.md) | the layer-by-layer build spec (source of truth for every module) |
| [results/REPORT.md](results/REPORT.md) | the full raw benchmark report, regenerated from per-run records |

## Requirements

- An Apple Silicon Mac (measured on an M5 Pro with 24 GB unified memory, macOS 26.4.1), or Linux,
  e.g. an HP Z workstation with an NVIDIA GPU (see [RUN_ON_HP.md](RUN_ON_HP.md)).
  Only the timings differ.
- Python 3.12 via [uv](https://docs.astral.sh/uv/) (0.11.14 used).
- llama.cpp `llama-server`, pinned to 0.7.0 (build 11669, commit `620d2225c`). On macOS install it with
  `brew install llama.cpp`. Another build runs with a warning (`scripts/check_engine.sh` explains how
  to get the pinned one), and `--strict-engine` refuses it.
- About 7 GB of disk for the two model files.

### Linux, NVIDIA and Windows

> **Tested on a Mac only.** The desk was built, benchmarked and fully tested on an Apple Silicon Mac.
> Elsewhere it was only checked in a CPU-only Ubuntu container. It has not run on a real NVIDIA GPU,
> WSL2 or Windows, so expect some hiccups there; [RUN_ON_HP.md](RUN_ON_HP.md) has a troubleshooting table.

On an HP Z workstation or any PC with an NVIDIA GPU, follow [RUN_ON_HP.md](RUN_ON_HP.md): Ubuntu, or
Windows through WSL2, step by step from the driver check to the UI, the adapter, the tests and the
load test. In short: build the pinned llama.cpp with `-DGGML_CUDA=ON`, set `LLAMA_SERVER_BIN`, then run
the same scripts as below.

## Setup (once, needs internet)

```bash
scripts/setup.sh
```

This checks the llama.cpp build, creates `.venv` from `uv.lock` (`uv sync --frozen`), installs the
Playwright browser for the UI tests, and downloads both model files at the pinned revision. Each
model file is verified against its sha256. Model binaries are not in the repository.

## Run the demo UI

```bash
scripts/run.sh
```

Open http://127.0.0.1:8000. The script starts llama-server for profile F and waits until the model
is warm. Ctrl+C stops both.

- Load a public scenario with **Fill** or type your own request.
- Watch the **Team (live)** tab: press **T** for full-screen theater mode, or replay a past turn.
- `scripts/run.sh --fake` runs the UI on the deterministic test double, with no model needed.

## Run the contract adapter

```bash
python3 adapter.py --input candidate_assets/public_cases.json --output results.json
```

```bash
python3 candidate_assets/validate_submission.py --input candidate_assets/public_cases.json --output results.json
```

`adapter.py` is a standard-library shim that re-executes itself under `.venv`. It starts its own
llama-server on a free port, runs every case from a fresh in-memory state, writes the results with
hardware, configuration, startup and per-case metrics, and stops the server.

| Flag | Effect |
|---|---|
| `--profile baseline` | runs B instead of F |
| `--llama-url URL` | attaches to a running server |
| `--strict-engine` | refuses any llama.cpp build other than the pinned one |

## Reset

```bash
scripts/reset.sh
```

Adapter runs need no reset: every case starts from its own `initial_state`, and every run starts a
fresh llama-server. `reset.sh` clears what persists between runs: UI session stores and server logs
under `var/`. `--dry-run` lists what would go. It never touches `results/` or `models/`.

## Tests

```bash
scripts/test.sh
```

This runs everything that needs no real model:

- unit and review tests, at 100% line and branch coverage over `desk/`, `server/`, `bench/` and
  `evals/`;
- then the Playwright suite against a fake-model UI server.

It takes about 3 minutes.

| Command | What it runs |
|---|---|
| `.venv/bin/pytest` | the fast loop, unit and review tests only (about 50 s) |
| `.venv/bin/pytest -m llm` | the tests that need the real model |

The independent oracle (`evals/oracle.py`) is written separately from the desk and scores business
correctness:

```bash
.venv/bin/python -m evals.score --input candidate_assets/public_cases.json --labels evals/expected/public_expected.json --output results.json
```

## Load test

Many customers at once against a running desk, through the same HTTP API the page uses. In the page:
open the **Load test** tab, enter how many customers and how many at the same time, and press Start. It
shows the flow (customers → waiting in line → turns running → model slots busy → done), live numbers (turns
per second, reply time p50/p95, turns refused and retried, double refunds, breaker, CPU, GPU and memory) and
six charts: turns finished each second, reply time, turns running vs waiting in line, model calls at the model
vs waiting for a slot, CPU vs GPU busy, and memory in use (whole machine vs desk + model). Hover a chart for
exact values; a table view is under the charts. The page header also shows CPU, GPU busy and memory in use.
From a terminal:

```bash
scripts/run.sh --fake --port 8010
.venv/bin/python -m bench.load --url http://127.0.0.1:8010 --customers 1000 --concurrency 100
```

Each customer gets its own session for one scenario (public cases and demo chats in turn) and sends its
scripted messages, confirming like the page does. The report (`var/load/load-<UTC time>.json`, plus a printed
summary) has throughput, client and server latency p50/p95/p99, customers in flight, server memory, the decision
mix, errors, and two checks: no line refunded or exchanged twice, and every copy of a scenario reached the same
decisions. Point `--url` at `scripts/run.sh` (real model) for real speeds.

How the desk stays up under load (real model; `server/app.py`, `desk/llm.py`):
- `scripts/run.sh --slots 4` lets the model serve 4 requests at once (8192 tokens each). Benchmarked
  results use the default, 1 slot.
- At most that many calls go to the model at once; the rest wait in the desk, so the 20 s step timeout
  measures only the model's work. A call that gets no slot within 60 s ends its turn in a decision
  error: nothing changes, and the customer gets the safe reply.
- At most 2 turns per slot run at once. Later turns wait in a first-come, first-served line and stream
  `queued` events with their place every second. Only when 1000 turns already wait does a turn get `503`
  with `Retry-After: 2`, which the load tool honours.
- After 3 model failures in a row (down, hung, 5xx), a circuit breaker fails calls at once for 10 s, then
  lets one trial call through.
- `GET /api/telemetry` → `capacity`: turns running and waiting in line, slots, calls waiting and at the
  model, turns refused, breaker state; `memory`: CPU (whole machine, desk server, model server, as a share
  of all cores), GPU busy %, machine memory in use, and the two servers' footprints.

Measured on the M5 Pro, 40 customers at once (`var/load/`):

| Setup | Turns failed | Server turn p50 / p99 | Turns/s |
|---|---|---|---|
| Before: 1 slot, no limits | 44 of 86 (timeouts) | 40 s / 128 s | 0.53 |
| 4 slots, 8 turns per slot | 2 of 86 (slot wait > 60 s) | 41 s / 107 s | 0.62 |
| 4 slots, 2 turns per slot | 0 of 86 | 12 s / 25 s | 0.61 |
| 4 slots, 2 turns per slot, waiting line | 0 of 88, 0 refused | 13 s / 31 s | 0.55 |

On the test double, 1000 customers at 100 at a time finish in about 11 s (about 200 turns/s, 0 failed). The
model is the ceiling: one local GPU gives about 0.6 turns/s whatever the slot count, so more throughput
means more model servers. With the line, the customer's reply time includes the wait: p50 60 s at 40 at once,
of which the server turn is 13 s.

## Benchmark

```bash
scripts/bench.sh --port 8405
```

The full matrix takes about 2 hours, unattended:

- **public:** 6 public cases × 3 runs × 9 profiles (B, F, 5 leave-one-out variants of F, B-q4, and an
  A/A copy of B), in randomized interleaved blocks;
- **variants:** 241 variant cases;
- **held-out:** two held-out suites, each with code-only floors.

Each run gets a fresh adapter and llama-server. A preflight refuses a busy machine (another
llama-server, CPU hogs, load, GPU use, battery, Low Power Mode, thermal warnings) and records its
readings. Per-run records go to `results/<suite>/<profile>/run-<k>/`. `results/REPORT.md` and
`results/csv/` are recomputed from them by `python -m bench.report --results results`.

To prove the run works offline:

```bash
scripts/offline_check.sh
```

It reruns the public suite with every non-loopback network path closed, after negative controls.

## Pinned versions and hardware

| Item | Value |
|---|---|
| Model | `unsloth/Qwen3.5-4B-GGUF`, revision `e87f176479d0855a907a41277aca2f8ee7a09523` |
| F (final) weights | `Qwen3.5-4B-Q4_K_M.gguf`, sha256 `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| B (baseline) weights | `Qwen3.5-4B-Q8_0.gguf`, sha256 `10cc391b403021dd11c614679d2fd92f611c3681d29e29651b717316965d61e1` |
| Inference engine | llama.cpp `llama-server` 0.7.0, build 11669, commit `620d2225c`, Metal backend |
| Decoding | greedy (temperature 0), seed 42, one request at a time (1 slot), context 8192 |
| Python deps | `pyproject.toml` + `uv.lock` (FastAPI 0.115.12, uvicorn 0.34.2, httpx 0.28.1, pydantic 2.11.4, psutil 7.0.0) |
| CPU / GPU | Apple M5 Pro, 15 CPU cores, integrated 16-core GPU (1 GPU) |
| Memory | 24 GB unified memory, so RAM and VRAM are the same pool (reported separately, never added) |
| OS / driver | macOS 26.4.1 (25E253); Metal ships with macOS, there is no separate GPU driver |
| Power | AC power, `pmset powermode 0`, sleep prevented; measured at the start and end of every run |

## Repository map

```
adapter.py            contract entry point (shim → desk/adapter_main.py)
desk/                 the application
  agents/             Resolution and Operations loops, prompts, tool schemas, budgets
  orchestrator.py     the per-turn pipeline: grants, gates, decision, response
  grounding.py        deterministic reading of IDs, products, sizes and cues (fact checks)
  policy.py           eligibility, exchange options, refund amount (pure)
  tools.py, store.py  business tools gateway and SQLite store (atomic commits, idempotency)
  llm.py, llm_server.py, profiles.py, telemetry.py   inference client, server lifecycle, profiles, metrics
server/               FastAPI app and the UI (index.html, app.js, styles.css); SSE event stream
evals/                independent oracle, labels, variants, held-out suites, scorer
bench/                benchmark runner, report generator, preflight
tests/                unit, review (adversarial findings), e2e (Playwright)
results/              benchmark records, report, offline proof, archived evidence
docs/                 architecture, results, limitations, clarifications, spec, notes
```
