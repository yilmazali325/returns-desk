# Running Returns Desk on an HP Z workstation (Linux or Windows)

This guide takes a fresh HP Z workstation (or any PC with an NVIDIA GPU) to a working desk: the demo UI,
the contract adapter, the tests and the load test. The project was built and measured on a Mac
([README](README.md)); everything below is the Linux and Windows path.

> **Tested on a Mac only.** Off the Mac the project was only checked in a CPU-only Ubuntu container
> ([what was checked](#what-has-been-checked)). It has not run on a real NVIDIA GPU, WSL2 or Windows, so
> expect some hiccups; the [troubleshooting](#troubleshooting) table covers the likely ones.

## What this project is

A retail returns and exchanges desk run by two cooperating LLM agents, entirely on one machine:

- **Resolution** talks to the customer. It reads the conversation, picks the item, remedy and size,
  decides whether to ask or act, and writes the reply.
- **Operations** is the back office. It checks exchange options, creates the proposal and, after the
  customer's explicit confirmation, executes it.
- **Deterministic code** owns identity, policy, money, stock, commits, idempotency and recovery. The rule
  is *agents say yes; code can say no*: code can block, read or replay, but never starts a business action.

| Part | What it is |
|---|---|
| Model | Qwen3.5-4B (GGUF, Q4_K_M for the final profile F, Q8_0 for the baseline B), about 7 GB on disk |
| Engine | llama.cpp `llama-server`, pinned to build 11669, commit `620d2225c` |
| App | Python 3.12 (FastAPI, SQLite), a vanilla-JS web page, all local; no cloud calls after setup |
| Entry points | `scripts/run.sh` (web UI), `python3 adapter.py` (contract adapter), `scripts/test.sh` (tests) |
| More detail | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/RESULTS.md](docs/RESULTS.md), [docs/LIMITATIONS.md](docs/LIMITATIONS.md) |

## Pick your path

| Machine | Path |
|---|---|
| Ubuntu 22.04 or 24.04 with an NVIDIA GPU | the steps below, as written |
| Windows 10 or 11 with an NVIDIA GPU | install WSL2 with Ubuntu first ([step 0](#step-0-windows-only-install-wsl2)), then the same steps inside Ubuntu |
| HP ZGX (DGX OS, Ubuntu-based) | the steps below; the NVIDIA driver and CUDA are normally preinstalled |
| AMD Radeon Pro or Intel Arc GPU | the steps below, but build llama.cpp with `-DGGML_VULKAN=ON` instead of `-DGGML_CUDA=ON` (needs `sudo apt install libvulkan-dev glslc`; not tested here) |
| No GPU at all | the steps below without `-DGGML_CUDA=ON`; it works, but see [CPU only](#cpu-only) |

Every command runs in a normal user terminal (bash). Only the `sudo` lines need an administrator password.

## Step 0 (Windows only): install WSL2

In PowerShell as administrator:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart, open **Ubuntu** from the Start menu, and create a user. Keep the NVIDIA **Windows** driver up to
date; it makes the GPU visible inside WSL2. Do not install a Linux NVIDIA driver inside WSL2. Run every
later step inside the Ubuntu window, and keep the project inside the Linux home folder (`~/`), not under
`/mnt/c`, which is much slower.

## Step 1: check the GPU

```bash
nvidia-smi
```

It should list the GPU and a driver version. If the command is missing, install the NVIDIA driver
(Ubuntu: **Software & Updates → Additional Drivers**, or `sudo ubuntu-drivers install`), then reboot.

## Step 2: system packages and the CUDA toolkit

```bash
sudo apt update && sudo apt install -y git curl cmake build-essential python3
```

```bash
nvcc --version
```

If `nvcc` is missing, install NVIDIA's CUDA Toolkit 12 or newer for your Ubuntu version from
https://developer.nvidia.com/cuda-downloads (on WSL2 pick **Linux → x86_64 → WSL-Ubuntu**). NVIDIA's
installer puts it under `/usr/local/cuda`, so add it to the path:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
```

## Step 3: install uv (the Python manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
source $HOME/.local/bin/env
```

uv downloads Python 3.12 itself; the system Python does not matter.

## Step 4: build the pinned llama.cpp with CUDA (about 10 minutes)

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp && cd ~/llama.cpp && git checkout 620d2225c
```

```bash
cmake -B build -DGGML_CUDA=ON && cmake --build build --target llama-server -j
```

Tell the project where it is, now and in every new terminal:

```bash
echo "export LLAMA_SERVER_BIN=$HOME/llama.cpp/build/bin/llama-server" >> ~/.bashrc && source ~/.bashrc
```

Check: `$LLAMA_SERVER_BIN --version` prints `version: 0.7.0-dev (build 11669, commit 620d2225c)`. The
project accepts this as the pinned build (Homebrew's build of the same commit says `0.7.0`).

## Step 5: set up the project (needs internet once)

Get the repository into your home folder (clone or unzip it), then:

```bash
cd ~/retail-returns-desk && scripts/setup.sh
```

This checks the llama.cpp build, creates `.venv` from the lock file, installs the Playwright browser for
the UI tests, and downloads both model files (about 7 GB), each verified by its sha256. Then, once per
machine, add the browser's system libraries so the UI tests can start Chromium:

```bash
sudo .venv/bin/python -m playwright install-deps chromium
```

After this, everything runs with the network off.

## Step 6: run the demo UI

```bash
scripts/run.sh
```

It starts llama-server, waits until the model is warm, and prints `Returns Desk is ready:
http://127.0.0.1:8000`. Open that address in a browser (on WSL2, in the Windows browser: WSL2 forwards
`127.0.0.1`). Ctrl+C stops the UI and the model server.

- **Fill** loads a public scenario; or type your own request.
- The **Team (live)** tab shows the agents working; press **T** for full-screen theater mode.
- The page header shows CPU, GPU busy, GPU memory and machine memory, read from `nvidia-smi` and the OS.
- To check that the model really runs on the GPU: while the desk runs, `nvidia-smi` lists `llama-server`
  and its GPU memory.

Options: `--port 8010` for another port, `--fake` for the deterministic test double (no model needed).

From another computer, forward the port over SSH and open the forwarded address there; any local port works:

```bash
ssh -N -L 9000:127.0.0.1:8000 user@workstation
```

Then open `http://localhost:9000` in that computer's browser.

## Step 7: run the contract adapter

```bash
python3 adapter.py --input candidate_assets/public_cases.json --output results.json
```

```bash
python3 candidate_assets/validate_submission.py --input candidate_assets/public_cases.json --output results.json
```

The adapter starts its own llama-server on a free port, runs every case from a fresh state, writes
`results.json` (with this machine's CPU, NVIDIA GPU and driver under `hardware`), and stops the server.
`--profile baseline` runs B instead of F; `--strict-engine` refuses any other llama.cpp build.

## Step 8: run the tests

```bash
scripts/test.sh
```

Unit, review and browser tests against a fake model; no GPU needed. `.venv/bin/pytest` runs the fast loop
only; `.venv/bin/pytest -m llm` runs the tests that need the real model.

## Step 9: the load test

A discrete NVIDIA GPU has its own memory, so the model can serve more customers at once than on the Mac:

```bash
scripts/run.sh --slots 8
```

Then open the **Load test** tab, set customers and how many at the same time, and press Start. From a
terminal, against the fake model:

```bash
scripts/run.sh --fake --port 8010
```

```bash
.venv/bin/python -m bench.load --url http://127.0.0.1:8010 --customers 1000 --concurrency 100
```

Each slot keeps 8192 tokens of context in GPU memory; if llama-server fails to start with more slots,
use fewer. Benchmarked results use 1 slot.

## Step 10: reset

```bash
scripts/reset.sh
```

Clears UI sessions and logs under `var/`. It never touches `results/` or `models/`.

## Troubleshooting

| What you see | What to do |
|---|---|
| `llama-server not found` | Step 4, and check `echo $LLAMA_SERVER_BIN` in this terminal |
| `WARNING: llama.cpp is '...'` | another llama.cpp build; it runs, but rebuild at commit `620d2225c` for the measured engine |
| The UI says `engine version ... is not the pinned` | the same; or start it with `scripts/run.sh --allow-engine-mismatch` |
| Replies take minutes, `nvidia-smi` shows no `llama-server` | llama.cpp was built without CUDA: redo step 4 with `-DGGML_CUDA=ON` |
| `cmake` cannot find CUDA | `nvcc --version` must work: step 2, then remove `~/llama.cpp/build` and build again |
| Browser tests fail to start Chromium | `sudo .venv/bin/python -m playwright install-deps chromium` |
| `Port 8000 is already in use` | `scripts/run.sh --port 8010` |
| `No supported WebSocket library detected` | harmless: the desk uses no WebSockets; something sent an upgrade request and got a normal reply |
| `No .venv here` | run `scripts/setup.sh` first, from the project folder |
| `python3: command not found` | `sudo apt install -y python3` (any Python 3 starts the adapter; it switches to `.venv` itself) |
| The model server log | `var/llama-server-8081.log` (UI) |

## What differs from the Mac

- Memory per process is the resident set size; macOS reports `phys_footprint`. GPU busy and GPU memory
  come from `nvidia-smi`. The macOS power and thermal readings (`pmset`) are null.
- macOS only: `scripts/offline_check.sh` (it uses the macOS sandbox) and the clean-machine checks of
  `scripts/bench.sh` (`pmset`, `caffeinate`). `.venv/bin/python -m bench.run --allow-dirty` runs the
  benchmark elsewhere and marks its timings as not clean.
- The CPU side of llama.cpp uses 5 threads (tuned for the Mac). With the model on the GPU this does not
  matter; a CPU-only run is slower than it could be.

## CPU only

Without a GPU the desk still runs and never changes anything it should not, but it is slow. Each model
step has a fixed 20-second limit (recorded in every result under `configuration.timeouts`). On a slow CPU a
long step can pass it. The turn then ends with the decision `error`, nothing changes, the customer gets
the safe reply, and the adapter restarts the model server for the next case. In the 4-core test container,
4 of the 6 public cases came out right and 2 hit this limit on their second turn, against 6 of 6 on the Mac
GPU. On an NVIDIA GPU a step takes a few seconds.

## What has been checked

- Checked in an Ubuntu 24.04 container (ARM64, 4 CPU cores, 6 GB memory, no GPU, llama.cpp built from
  commit `620d2225c` for the CPU, without `lsof`):
  - `scripts/setup.sh`: the source build counts as the pinned engine, and both models pass their sha256 check;
  - `scripts/test.sh`: all unit, review and browser tests pass;
  - `scripts/run.sh` with the real model: the model loads, warms up and answers, and the header telemetry works;
  - `python3 adapter.py` on a public case: the right decision, and `validate_submission.py` passes;
  - `python -m bench.run --allow-dirty`: the preflight refuses with its reasons, the run completes (CPU only, above).
- Not checked here: a real NVIDIA GPU with CUDA, WSL2, and native Windows (without WSL2 `adapter.py`
  starts and finds `.venv\Scripts\python.exe`, but the scripts need bash).
