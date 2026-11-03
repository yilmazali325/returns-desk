#!/bin/bash
# Start the Returns Desk demo: the local model (llama-server) and the web UI, then print the URL.
#   scripts/run.sh [--port 8000] [--llama-port 8081] [--profile F] [--slots 1] [--fake] [--no-warmup] [--allow-engine-mismatch]
# The UI process attaches to a healthy llama-server on --llama-port, or starts one for the profile
# (model sha256 and engine version are checked, one warm-up request, both agents' static prompts are
# primed in the prompt cache). The UI answers only once the model is ready. Ctrl+C stops the UI and
# any llama-server it started. --fake runs the deterministic test double instead of the model.
# --allow-engine-mismatch starts a llama.cpp build other than the pinned one (scripts/check_engine.sh).
# --slots N lets the model serve N requests at once (for load tests; benchmarked results use 1).
set -euo pipefail
cd "$(dirname "$0")/.."

UI_PORT=8000
LLAMA_PORT=8081
PROFILE=F
EXTRA=()
while (( $# )); do
  case $1 in
    --port) UI_PORT=$2; shift 2 ;;
    --llama-port) LLAMA_PORT=$2; shift 2 ;;
    --profile) PROFILE=$2; shift 2 ;;
    --slots) EXTRA+=(--slots "$2"); shift 2 ;;
    --fake) EXTRA+=(--fake); shift ;;
    --no-warmup) EXTRA+=(--no-warmup); shift ;;
    --allow-engine-mismatch) EXTRA+=(--allow-engine-mismatch); shift ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

[[ -x .venv/bin/python ]] || { echo "No .venv here: run scripts/setup.sh first." >&2; exit 1; }
if lsof -nP -iTCP:"$UI_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $UI_PORT is already in use; pick another with --port." >&2
  exit 1
fi

.venv/bin/python -m server.app --port "$UI_PORT" --llama-url "http://127.0.0.1:$LLAMA_PORT" \
  --profile "$PROFILE" --start-llama ${EXTRA[@]+"${EXTRA[@]}"} &
UI_PID=$!
trap 'kill -TERM $UI_PID 2>/dev/null || true; wait $UI_PID 2>/dev/null || true' INT TERM EXIT

echo "Starting (a cold model load can take a minute)..."
until curl -fs "http://127.0.0.1:$UI_PORT/api/health" >/dev/null 2>&1; do
  if ! kill -0 $UI_PID 2>/dev/null; then
    echo "The UI server exited before it was ready (see the messages above)." >&2
    trap - EXIT
    exit 1
  fi
  sleep 0.5
done

HEALTH=$(curl -fs "http://127.0.0.1:$UI_PORT/api/health")
echo "Model: $(echo "$HEALTH" | .venv/bin/python -c 'import json,sys; h=json.load(sys.stdin); print(h["llm"], "-", h["model"], h.get("quantization") or "", "-", h["engine"])')"
echo
echo "  Returns Desk is ready:  http://127.0.0.1:$UI_PORT"
echo "  (all inference local; Ctrl+C stops the UI and the llama-server it started)"
echo
wait $UI_PID
