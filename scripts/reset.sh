#!/bin/bash
# Reset local run state to a clean start.
#
#   scripts/reset.sh [--dry-run] [--force]
#
# Adapter runs need no reset: every case starts from a fresh in-memory SQLite store built from the
# input's initial_state (desk/session.py), and every adapter run starts a fresh llama-server, so its
# prompt cache starts empty. What persists between runs, and is removed here:
#   var/ui/        UI state (server/app.py): session stores in var/ui/sessions/*.sqlite and the UI's
#                  llama-server log; $DESK_VAR_DIR replaces var/ui when the UI was started with it
#   var/adapter/   llama-server logs the adapter writes per profile and port (bench.run copies the
#                  ones it needs into each run directory, so nothing measured is lost)
# Kept on purpose: results/ (benchmark and offline evidence; results/evidence/ holds the tracked
# copies of var/tuning/ and var/llm_infra/ that the docs cite), models/ (weights and the sha256
# cache, keyed by path + size + mtime; the bench re-hashes every byte once per invocation),
# var/tuning/ and var/llm_infra/ (the originals). One UI session is reset from the UI instead
# (POST /api/sessions/{id}/reset rebuilds it from its initial state).
#
# Refuses while the UI server or a benchmark is running (their in-memory state would outlive the
# files and they would write new ones half-way); --force skips that check, e.g. when the running
# server belongs to another checkout. Exit codes: 0 done, 1 refused, 2 usage error.

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DRY_RUN=0
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "usage: scripts/reset.sh [--dry-run] [--force]" >&2; exit 2 ;;
  esac
done

# Match on the executable, not the whole command line: a shell whose -c string merely mentions
# bench.run (or this script) is not a running benchmark.
BUSY="$(ps -Ao pid=,command= | awk '{n = split($2, parts, "/"); exe = parts[n]}
  (exe ~ /^(python[0-9.]*|uvicorn)$/ && ($0 ~ /server\.app|server\/app\.py/ || $0 ~ / -m bench\.run( |$)/)) ||
  (exe ~ /^(bash|zsh|sh)$/ && $3 ~ /scripts\/bench\.sh$/) {print}')"
if [ -n "$BUSY" ] && [ "$FORCE" -eq 0 ]; then
  echo "reset refused: stop these first (UI server or benchmark in progress), or pass --force:"
  printf '%s\n' "$BUSY" | cut -c1-160 | sed 's/^/  /'
  exit 1
fi

for path in "${DESK_VAR_DIR:-var/ui}" var/adapter; do
  if [ ! -e "$path" ]; then
    echo "clean    $path (nothing there)"
    continue
  fi
  count="$(find "$path" -type f | wc -l | tr -d ' ')"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "would remove $path ($count file(s))"
  else
    rm -rf "$path" && echo "removed  $path ($count file(s))"
  fi
done
if [ "$DRY_RUN" -eq 1 ]; then echo "dry run: nothing was removed"; else echo "reset done"; fi
exit 0
