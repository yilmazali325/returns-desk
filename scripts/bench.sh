#!/bin/bash
# Full benchmark matrix (SPEC §9), unattended. Takes about 1.5 hours on the M5 Pro (see
# docs/notes/bench_harness.md for the estimate).
#
#   scripts/bench.sh [--port 8405] [--resume] [--skip-public] [--skip-variants] [--skip-holdouts]
#
# 1. public suite: every profile (B, F, the 5 leave-one-out F-*, B-q4, B-repeat) x 3 runs, in
#    randomized interleaved blocks (bench/run.py, seed 42)
# 2. variant suite (241 cases): one quality run each for B and F (the gate's variant criterion)
# 3. held-out suites (evals/holdout/holdout_v2.json, evals/holdout_v3/holdout_v3.json when present):
#    one run each for B and F; never used for tuning, so these are the generalisation numbers
# 4. results/REPORT.md + results/csv/*.csv from the raw run directories (bench/report.py)
# Every suite also gets its code-only floors first (no model, seconds), and each bench invocation
# hashes the model files in full once; adapter runs use --strict-engine.
#
# Each suite starts behind bench/preflight.sh, which refuses a contaminated machine and then
# execs the runner under `caffeinate -dimsu`; bench/run.py also re-runs the preflight before every
# single run and waits up to 30 minutes for the machine to become clean again. --resume skips runs
# that are already complete (nothing is ever overwritten).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT=8405
RESUME=()
PUBLIC=1
VARIANTS=1
HOLDOUTS=1
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --resume) RESUME=(--resume); shift ;;
    --skip-public) PUBLIC=0; shift ;;
    --skip-variants) VARIANTS=0; shift ;;
    --skip-holdouts) HOLDOUTS=0; shift ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "usage: scripts/bench.sh [--port N] [--resume] [--skip-public] [--skip-variants] [--skip-holdouts]" >&2; exit 2 ;;
  esac
done

PY=.venv/bin/python
COMMON=(--port "$PORT" --preflight-wait-min 30 ${RESUME[@]+"${RESUME[@]}"})
mkdir -p results/preflight

if [ "$PUBLIC" -eq 1 ]; then
  bench/preflight.sh --out results/preflight/public-start --port "$PORT" -- \
    "$PY" -m bench.run --suite public --all-profiles --runs 3 "${COMMON[@]}"
fi
if [ "$VARIANTS" -eq 1 ]; then
  bench/preflight.sh --out results/preflight/variants-start --port "$PORT" -- \
    "$PY" -m bench.run --suite variants --profiles B F --runs 1 "${COMMON[@]}"
fi
if [ "$HOLDOUTS" -eq 1 ]; then
  for SUITE in holdout_v2 holdout_v3; do
    if ! "$PY" -c "import sys; from bench import SUITES; sys.exit(0 if SUITES['$SUITE'].available() else 1)"; then
      echo "skip $SUITE: its input or labels file is not on disk yet"
      continue
    fi
    bench/preflight.sh --out "results/preflight/$SUITE-start" --port "$PORT" -- \
      "$PY" -m bench.run --suite "$SUITE" --profiles B F --runs 1 "${COMMON[@]}"
  done
fi
"$PY" -m bench.report --results results
echo "report: results/REPORT.md"
