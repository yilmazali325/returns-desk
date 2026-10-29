#!/bin/bash
# Is the installed llama.cpp the build the benchmark was measured with? Called by scripts/setup.sh.
#   scripts/check_engine.sh            uses $LLAMA_SERVER_BIN, else llama-server on PATH
# Prints the found and pinned versions and records them in var/setup/engine.txt.
# Exit codes: 0 pinned build found, or another build (WARNING: the adapter runs it with a warning,
# `--strict-engine` refuses it); 1 no llama-server at all; 2 its --version output is unreadable.
set -uo pipefail
cd "$(dirname "$0")/.."
PINNED_VERSION="0.7.0 (build 11669, commit 620d2225c)"  # = desk/profiles.py ENGINE_VERSION
PINNED_COMMIT="620d2225c"
BIN="${LLAMA_SERVER_BIN:-$(command -v llama-server || true)}"
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
  echo "llama-server not found. Install llama.cpp (brew install llama.cpp), or build the pinned commit:"
  echo "  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout $PINNED_COMMIT"
  echo "  cmake -B build && cmake --build build --target llama-server -j && export LLAMA_SERVER_BIN=\$PWD/build/bin/llama-server"
  exit 1
fi
FOUND="$("$BIN" --version 2>&1 | sed -n 's/^version: *//p' | head -1)"
mkdir -p var/setup
printf 'binary: %s\nfound: %s\npinned: %s\n' "$BIN" "${FOUND:-unreadable}" "$PINNED_VERSION" > var/setup/engine.txt
if [ -z "$FOUND" ]; then
  echo "cannot read the version of $BIN (expected a 'version: ...' line from --version)"
  exit 2
fi
if [ "$FOUND" = "$PINNED_VERSION" ]; then
  echo "ok  llama.cpp $FOUND ($BIN)"
  exit 0
fi
echo "WARNING: llama.cpp is '$FOUND', the benchmark was measured with '$PINNED_VERSION'."
echo "  python adapter.py runs it anyway and records engine_version_match=false (--strict-engine refuses);"
echo "  the UI needs --allow-engine-mismatch (scripts/run.sh passes it through). For the exact build:"
echo "  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout $PINNED_COMMIT"
echo "  cmake -B build && cmake --build build --target llama-server -j && export LLAMA_SERVER_BIN=\$PWD/build/bin/llama-server"
exit 0
