#!/bin/bash
# Offline proof: the full public adapter run works with every non-loopback network path closed.
#
#   scripts/offline_check.sh [--port 8406] [--profile F]
#
# Steps (each command and its output go verbatim into results/offline/OFFLINE_PROOF.md):
#   0. positive control OUTSIDE the sandbox (informational): can this machine reach the internet?
#      If it cannot, the negative controls prove less, and the proof says so.
#   1. negative controls INSIDE the sandbox, before anything else; all must FAIL or the script aborts
#      (a sandbox that is silently not applied would make the proof vacuous):
#        TCP connect to 1.1.1.1:443 (nc), the same from the project's own Python,
#        https://huggingface.co by name (DNS), and by IP with DNS pre-resolved outside (curl --resolve)
#   2. the full public run inside the sandbox: `python3 adapter.py --profile F ...` (it starts its own
#      llama-server on 127.0.0.1, which inherits the sandbox), with lsof snapshots taken from outside
#      showing that every socket of the adapter and the server is loopback
#   3. validate_submission.py and evals.score on the output, also inside the sandbox
#
# The sandbox profile is scripts/offline.sb. HF_HUB_OFFLINE=1 and UV_OFFLINE=1 are set as a second
# line of defence; llama-server needs no flag because it is given a local model path (and
# desk.llm_server strips LLAMA_ARG_* variables, so LLAMA_ARG_OFFLINE could not reach it anyway).
# Exit codes: 0 proof written and the run passed; 1 a negative control connected (sandbox not
# applied) or the run/validation failed; 2 usage error.

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT=8406
PROFILE=F
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "usage: scripts/offline_check.sh [--port N] [--profile NAME]" >&2; exit 2 ;;
  esac
done

SB=scripts/offline.sb
OUT=results/offline
MD="$OUT/OFFLINE_PROOF.md"
INPUT=candidate_assets/public_cases.json
LABELS=evals/expected/public_expected.json
RESULTS="$OUT/results.json"
PY=.venv/bin/python
export HF_HUB_OFFLINE=1 UV_OFFLINE=1
mkdir -p "$OUT"
rm -f "$RESULTS" "$OUT/score.json"
FAILED=0

# quote_cmd ARGS...: the command as a copy-pasteable shell line (arguments with spaces or shell
# characters quoted by Python's shlex.join), so the proof shows exactly what ran.
quote_cmd() {
  "$PY" -c 'import shlex, sys; print(shlex.join(sys.argv[1:]), end="")' "$@"
}

# record TITLE EXPECT CMD...: run CMD, append the command, output and exit code to the proof.
# EXPECT is "fail" (must exit non-zero), "pass" (must exit 0) or "info" (either is fine).
record() {
  local title="$1" expect="$2"; shift 2
  local output code verdict
  output="$("$@" 2>&1)"; code=$?
  verdict="exit $code"
  if [ "$expect" = "fail" ]; then
    if [ "$code" -ne 0 ]; then verdict="$verdict: blocked as required"; else verdict="$verdict: CONNECTED (sandbox not applied)"; FAILED=1; fi
  elif [ "$expect" = "pass" ]; then
    if [ "$code" -eq 0 ]; then verdict="$verdict: ok"; else verdict="$verdict: FAILED"; FAILED=1; fi
  fi
  {
    echo "### $title"
    echo
    echo '```'
    echo "\$ $(quote_cmd "$@")"
    printf '%s\n' "$output" | tail -n 40
    echo '```'
    echo
    echo "Result: $verdict"
    echo
  } >> "$MD"
  echo "[offline] $title: $verdict"
  return "$code"
}

{
  echo "# Offline proof"
  echo
  echo "Written by \`scripts/offline_check.sh --port $PORT --profile $PROFILE\` on $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "($(sw_vers -productName) $(sw_vers -productVersion) $(sw_vers -buildVersion), $(sysctl -n machdep.cpu.brand_string))."
  echo "Every command below ran exactly as shown; outputs are verbatim (long outputs trimmed to their last 40 lines)."
  echo
  echo "## Sandbox profile (\`$SB\`)"
  echo
  echo '```scheme'
  cat "$SB"
  echo '```'
  echo
  echo "Environment for every sandboxed command: HF_HUB_OFFLINE=1 UV_OFFLINE=1."
  echo
  echo "## 0. Positive control outside the sandbox (informational)"
  echo
} > "$MD"

command -v sandbox-exec >/dev/null || { echo "sandbox-exec not found (macOS only)" >&2; exit 2; }
ONLINE=1
record "TCP connect to 1.1.1.1:443 without the sandbox" info /usr/bin/nc -z -G 3 1.1.1.1 443 || ONLINE=0
record "HTTPS to huggingface.co without the sandbox" info \
  /usr/bin/curl -sS -o /dev/null -w 'HTTP %{http_code}\n' --max-time 10 https://huggingface.co || ONLINE=0
HF_IP="$(/usr/bin/dig +short huggingface.co A 2>/dev/null | grep -E '^[0-9.]+$' | head -1)"
if [ "$ONLINE" -eq 1 ]; then
  echo "The machine is online, so the failures below are caused by the sandbox." >> "$MD"
else
  echo "**The machine could not reach the internet even without the sandbox**, so the negative controls below only show that nothing connected; they do not isolate the sandbox's effect." >> "$MD"
fi
printf '\n## 1. Negative controls inside the sandbox (must all fail)\n\n' >> "$MD"

record "TCP connect to 1.1.1.1:443 inside the sandbox" fail \
  sandbox-exec -f "$SB" /usr/bin/nc -z -G 3 1.1.1.1 443
record "TCP connect to 1.1.1.1:443 from the project's Python inside the sandbox" fail \
  sandbox-exec -f "$SB" "$PY" -c 'import socket; socket.create_connection(("1.1.1.1", 443), 3); print("connected")'
record "HTTPS to huggingface.co by name inside the sandbox" fail \
  sandbox-exec -f "$SB" /usr/bin/curl -sS -o /dev/null --max-time 10 https://huggingface.co
if [ -n "$HF_IP" ]; then
  record "HTTPS to huggingface.co with DNS pre-resolved outside ($HF_IP) inside the sandbox" fail \
    sandbox-exec -f "$SB" /usr/bin/curl -sS -o /dev/null --max-time 10 --resolve "huggingface.co:443:$HF_IP" https://huggingface.co
fi
if [ "$FAILED" -ne 0 ]; then
  printf '\n**ABORTED: a negative control connected, so the sandbox is not applied. No run was made.**\n' >> "$MD"
  echo "[offline] ABORTED: the sandbox did not block the network; see $MD" >&2
  exit 1
fi

printf '\n## 2. Full public run inside the sandbox\n\n' >> "$MD"
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port $PORT is in use; pick another with --port" >&2
  exit 2
fi
ADAPTER_CMD=(sandbox-exec -f "$SB" python3 adapter.py --profile "$PROFILE" --port "$PORT" --input "$INPUT" --output "$RESULTS")
echo "Command (adapter.py re-execs into .venv/bin/python; llama-server is its child, so both run sandboxed):" >> "$MD"
printf '\n```\n$ %s\n```\n\n' "$(quote_cmd "${ADAPTER_CMD[@]}")" >> "$MD"
STARTED=$(date +%s)
"${ADAPTER_CMD[@]}" > "$OUT/adapter.log" 2>&1 &
ADAPTER_PID=$!
# lsof from OUTSIDE the sandbox while the run is in progress: every socket must be loopback.
LSOF="$OUT/lsof.txt"
: > "$LSOF"
SNAPSHOTS=0
while kill -0 "$ADAPTER_PID" 2>/dev/null; do
  SERVER_PID="$(pgrep -f "llama-server.*--port $PORT" | head -1)"
  if [ -n "$SERVER_PID" ] && [ "$SNAPSHOTS" -lt 6 ]; then
    { echo "# snapshot $((SNAPSHOTS + 1)) at +$(( $(date +%s) - STARTED )) s (adapter pid $ADAPTER_PID, llama-server pid $SERVER_PID)"
      lsof -nP -i -a -p "$ADAPTER_PID,$SERVER_PID"; echo; } >> "$LSOF" 2>&1
    SNAPSHOTS=$((SNAPSHOTS + 1))
  fi
  sleep 2
done
wait "$ADAPTER_PID"; ADAPTER_CODE=$?
ELAPSED=$(( $(date +%s) - STARTED ))
{
  echo "Adapter exit code: $ADAPTER_CODE, wall time ${ELAPSED} s. Adapter log:"
  echo
  echo '```'
  cat "$OUT/adapter.log"
  echo '```'
  echo
  echo "### Sockets of the adapter and llama-server during the run (lsof, taken outside the sandbox)"
  echo
  echo '```'
  cat "$LSOF"
  echo '```'
  echo
} >> "$MD"
NON_LOOPBACK="$(grep -E 'TCP|UDP' "$LSOF" | grep -vE '127\.0\.0\.1|localhost|\[::1\]' || true)"
if [ "$SNAPSHOTS" -eq 0 ]; then
  echo "Result: no lsof snapshot was taken (the server was never seen running)." >> "$MD"; FAILED=1
elif [ -n "$NON_LOOPBACK" ]; then
  printf 'Result: NON-LOOPBACK SOCKETS FOUND:\n\n```\n%s\n```\n\n' "$NON_LOOPBACK" >> "$MD"; FAILED=1
else
  echo "Result: $SNAPSHOTS snapshot(s); every socket is on 127.0.0.1." >> "$MD"
fi
[ "$ADAPTER_CODE" -eq 0 ] || FAILED=1

printf '\n## 3. Validation and scoring (inside the sandbox)\n\n' >> "$MD"
record "Structural validator" pass \
  sandbox-exec -f "$SB" "$PY" candidate_assets/validate_submission.py --input "$INPUT" --output "$RESULTS"
record "Independent oracle score" pass \
  sandbox-exec -f "$SB" "$PY" -m evals.score --input "$INPUT" --labels "$LABELS" --output "$RESULTS" \
  --json "$OUT/score.json"
STRICT="$("$PY" -c "import json; s = json.load(open('$OUT/score.json'))['summary']; print(f\"{s['strict_success']}/{s['cases']}\")" 2>/dev/null || echo "unknown")"

if [ "$FAILED" -eq 0 ]; then
  printf '\n## Verdict\n\nPASSED: the network was provably closed (negative controls failed inside the sandbox), the full public run completed inside it using only loopback sockets, and its output passes the structural validator. Strict oracle score of this run: %s.\n' "$STRICT" >> "$MD"
  echo "[offline] PASSED; proof in $MD"
  exit 0
fi
printf '\n## Verdict\n\nFAILED: see the results above.\n' >> "$MD"
echo "[offline] FAILED; see $MD" >&2
exit 1
