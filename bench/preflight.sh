#!/bin/bash
# Bench preflight (SPEC §9): refuse to benchmark on a machine that would contaminate the numbers,
# and keep the evidence. Why each check exists (measured on this machine, docs/notes/):
#   - a foreign llama-server sharing the GPU halved decode speed (70 -> 37 tok/s) during review;
#   - an ffmpeg job from another project (load average 10-13) inflated a whole B run;
#   - battery power, Low Power Mode and thermal limits all lower clocks, and the results.json
#     `hardware.power` field declares "AC power, pmset powermode 0 (automatic), sleep prevented".
#
#   bench/preflight.sh [--out DIR] [--port N]... [-- command args...]
#
#   --out DIR   save the raw readings (pmset, therm, load, ps, GPU statistics, port owners) in DIR;
#               processes are saved as pid, %cpu, %mem and executable NAME only, never full command
#               lines (arguments can carry tokens and private paths, and these files are committed)
#   --port N    also require TCP port N to be free (repeatable)
#   -- cmd ...  when every check passes, exec `caffeinate -dimsu cmd ...` so the machine cannot
#               sleep (or dim and throttle) during the run
#
# Exit codes: 0 clean (or, with `-- cmd`, the command's own exit code), 1 refused, 2 usage error.
# Thresholds (environment, recorded in the output):
#   BENCH_CPU_HOG_PCT  50    any other process at or above this %CPU (ps decaying average) is a hog
#   BENCH_MAX_LOAD     5.0   maximum 1-minute load average (15 cores; idle here is about 1-2)
#   BENCH_GPU_BUSY_PCT 10    maximum median IOAccelerator "Device Utilization %" over 5 samples
#   BENCH_GPU_MEM_MB   2048  maximum IOAccelerator "In use system memory" with no llama-server
#                            running (idle here is 600-900 MiB; a loaded 4B model adds ~3 GiB)
#   BENCH_IGNORE_REGEX       processes whose command matches this extended regex are not hogs

set -u
export LC_ALL=C

CPU_HOG_PCT="${BENCH_CPU_HOG_PCT:-50}"
MAX_LOAD="${BENCH_MAX_LOAD:-5.0}"
GPU_BUSY_PCT="${BENCH_GPU_BUSY_PCT:-10}"
GPU_MEM_MB="${BENCH_GPU_MEM_MB:-2048}"
IGNORE_REGEX="${BENCH_IGNORE_REGEX:-}"

OUT=""
PORTS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) [ $# -ge 2 ] || { echo "usage: --out needs a directory" >&2; exit 2; }; OUT="$2"; shift 2 ;;
    --port) [ $# -ge 2 ] || { echo "usage: --port needs a number" >&2; exit 2; }; PORTS+=("$2"); shift 2 ;;
    --) shift; break ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "usage: bench/preflight.sh [--out DIR] [--port N]... [-- command args...] (got: $1)" >&2; exit 2 ;;
  esac
done

if [ -n "$OUT" ]; then mkdir -p "$OUT" || exit 2; fi
PROBLEMS=0

save() {  # save NAME: stdin -> $OUT/NAME (when --out was given), always echoed to stdout by callers
  if [ -n "$OUT" ]; then cat > "$OUT/$1"; else cat > /dev/null; fi
}
pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1"; PROBLEMS=$((PROBLEMS + 1)); }

echo "preflight $(date -u +%Y-%m-%dT%H:%M:%SZ) thresholds: cpu_hog>=${CPU_HOG_PCT}% load1<=${MAX_LOAD}" \
     "gpu_busy<=${GPU_BUSY_PCT}% gpu_mem<=${GPU_MEM_MB}MiB ignore='${IGNORE_REGEX}'"

# 1. No llama-server at all: ours is started by the adapter after this check passes.
#    `comm` is the executable path without arguments; keep only its basename.
PS_ALL="$(ps -Ao pid=,%cpu=,%mem=,comm= | awk '{pid = $1; cpu = $2; mem = $3; $1 = $2 = $3 = ""; sub(/^ +/, "");
  n = split($0, parts, "/"); print pid, cpu, mem, parts[n]}')"
{ echo "# pid %cpu %mem name"; printf '%s\n' "$PS_ALL"; } | save ps.txt
LLAMAS="$(printf '%s\n' "$PS_ALL" | awk '$4 == "llama-server" {print}')"
if [ -z "$LLAMAS" ]; then
  pass "llama-server: none running"
else
  fail "llama-server: $(printf '%s\n' "$LLAMAS" | wc -l | tr -d ' ') foreign process(es) running"
  printf '%s\n' "$LLAMAS" | cut -c1-200 | sed 's/^/      /'
fi

# 2. CPU: no hog besides this script and its parent (the bench runner), and a low load average.
HOGS="$(printf '%s\n' "$PS_ALL" | awk -v me="$$" -v parent="$PPID" -v limit="$CPU_HOG_PCT" \
  '$1 != me && $1 != parent && $2 + 0 >= limit + 0 {print}')"
if [ -n "$IGNORE_REGEX" ] && [ -n "$HOGS" ]; then
  HOGS="$(printf '%s\n' "$HOGS" | grep -Ev -- "$IGNORE_REGEX" || true)"
fi
if [ -z "$HOGS" ]; then
  pass "cpu: no other process at or above ${CPU_HOG_PCT}% CPU"
else
  fail "cpu: $(printf '%s\n' "$HOGS" | wc -l | tr -d ' ') process(es) at or above ${CPU_HOG_PCT}% CPU (pid %cpu %mem name)"
  printf '%s\n' "$HOGS" | cut -c1-200 | sed 's/^/      /'
fi
LOADAVG="$(sysctl -n vm.loadavg 2>/dev/null | tr -d '{}' | awk '{print $1, $2, $3}')"
uptime | save uptime.txt
LOAD1="$(printf '%s\n' "$LOADAVG" | awk '{print $1}')"
if [ -z "$LOAD1" ]; then
  fail "load: vm.loadavg unreadable"
elif awk -v l="$LOAD1" -v m="$MAX_LOAD" 'BEGIN {exit !(l + 0 <= m + 0)}'; then
  pass "load: 1/5/15-minute load average ${LOADAVG} (limit ${MAX_LOAD})"
else
  fail "load: 1-minute load average ${LOAD1} > ${MAX_LOAD} (1/5/15: ${LOADAVG})"
fi

# 3. GPU: Apple Silicon has no per-process GPU accounting without sudo, so check the system-wide
#    AGX IOAccelerator statistics: median utilization of 5 samples and in-use unified memory.
GPU_SAMPLES=""
for i in 1 2 3 4 5; do
  STATS="$(ioreg -r -d 1 -w 0 -c IOAccelerator 2>/dev/null)"
  UTIL="$(printf '%s\n' "$STATS" | grep -oE '"Device Utilization %"=[0-9]+' | head -1 | grep -oE '[0-9]+$')"
  MEM="$(printf '%s\n' "$STATS" | grep -oE '"In use system memory"=[0-9]+' | head -1 | grep -oE '[0-9]+$')"
  GPU_SAMPLES="${GPU_SAMPLES}${UTIL:-NA} ${MEM:-NA}"$'\n'
  [ "$i" -lt 5 ] && sleep 0.4
done
printf '# device_utilization_pct in_use_system_memory_bytes\n%s' "$GPU_SAMPLES" | save gpu_samples.txt
GPU_VERDICT="$(printf '%s' "$GPU_SAMPLES" | awk -v busy="$GPU_BUSY_PCT" -v memlimit="$GPU_MEM_MB" '
  $1 == "NA" || $2 == "NA" {unreadable = 1; next}
  {u[n] = $1; m[n] = $2 / 1048576; n++}
  END {
    if (unreadable || n == 0) {print "unreadable"; exit}
    for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) {
      if (u[j] < u[i]) {t = u[i]; u[i] = u[j]; u[j] = t}
      if (m[j] < m[i]) {t = m[i]; m[i] = m[j]; m[j] = t}
    }
    mu = u[int(n / 2)]; mm = m[int(n / 2)]
    ok = (mu <= busy + 0 && mm <= memlimit + 0) ? "ok" : "busy"
    printf "%s median utilization %d%%, median in-use system memory %.0f MiB\n", ok, mu, mm
  }')"
case "$GPU_VERDICT" in
  ok*) pass "gpu: ${GPU_VERDICT#ok }" ;;
  busy*) fail "gpu: ${GPU_VERDICT#busy } (limits ${GPU_BUSY_PCT}% / ${GPU_MEM_MB} MiB)" ;;
  *) fail "gpu: IOAccelerator statistics unreadable (cannot verify an idle GPU)" ;;
esac

# 4-6. Power source, Low Power Mode, thermal state (the declared bench condition).
BATT="$(pmset -g batt 2>&1)"; printf '%s\n' "$BATT" | save pmset_batt.txt
SETTINGS="$(pmset -g 2>&1)"; printf '%s\n' "$SETTINGS" | save pmset.txt
THERM="$(pmset -g therm 2>&1)"; printf '%s\n' "$THERM" | save pmset_therm.txt
if printf '%s\n' "$BATT" | grep -q "'AC Power'"; then
  pass "power: AC Power ($(printf '%s\n' "$BATT" | grep -oE '[0-9]+%;[^;]*' | head -1))"
else
  fail "power: not on AC power: $(printf '%s\n' "$BATT" | head -2 | tr '\n' ' ')"
fi
MODE="$(printf '%s\n' "$SETTINGS" | awk '$1 == "powermode" || $1 == "lowpowermode" {print $2; exit}')"
if [ "$MODE" = "0" ]; then
  pass "low power mode: off (pmset powermode 0)"
else
  fail "low power mode: pmset powermode is '${MODE:-missing}', expected 0 (automatic)"
fi
SPEED="$(printf '%s\n' "$THERM" | grep -oE 'CPU_Speed_Limit[[:space:]]*=[[:space:]]*[0-9]+' | grep -oE '[0-9]+$')"
WARN="$(printf '%s\n' "$THERM" | grep -i 'warning level' | grep -v 'No ' || true)"
if [ -n "$WARN" ] || { [ -n "$SPEED" ] && [ "$SPEED" -lt 100 ]; }; then
  fail "thermal: throttling recorded: ${WARN} ${SPEED:+CPU_Speed_Limit=$SPEED}"
else
  pass "thermal: no thermal or performance warning recorded${SPEED:+, CPU_Speed_Limit=$SPEED}"
fi
SLEEP_LINE="$(printf '%s\n' "$SETTINGS" | awk '$1 == "sleep" {$1 = $1; print; exit}')"
echo "INFO  sleep setting: ${SLEEP_LINE:-unknown} (the run itself is wrapped in caffeinate -dimsu)"

# 7. Ports the run will listen on.
for PORT in ${PORTS[@]+"${PORTS[@]}"}; do
  OWNER="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | tail -n +2)"
  if [ -z "$OWNER" ]; then
    pass "port $PORT: free"
  else
    fail "port $PORT: already in use"
    printf '%s\n' "$OWNER" | sed 's/^/      /'
  fi
  printf '%s\n' "$OWNER" | save "port_${PORT}.txt"
done

if [ "$PROBLEMS" -gt 0 ]; then
  echo "PREFLIGHT REFUSED: $PROBLEMS problem(s); timings taken now would not be clean."
  [ -n "$OUT" ] && echo "refused $PROBLEMS" | save verdict.txt
  exit 1
fi
echo "PREFLIGHT PASSED"
[ -n "$OUT" ] && echo "passed" | save verdict.txt
if [ $# -gt 0 ]; then
  exec caffeinate -dimsu "$@"
fi
exit 0
