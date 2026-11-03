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
PINNED_BUILD="(build 11669, commit $PINNED_COMMIT)"  # a source build says 0.7.0-dev: same build and commit
BIN="${LLAMA_SERVER_BIN:-$(command -v llama-server || true)}"
# The GPU backend: Metal is the default on macOS; on Linux/WSL2 with an NVIDIA GPU it must be asked for.
if [ "$(uname -s)" = Darwin ]; then GPU_FLAG=""; else GPU_FLAG=" -DGGML_CUDA=ON"; fi
build_hint() {
  echo "  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout $PINNED_COMMIT"
  echo "  cmake -B build$GPU_FLAG && cmake --build build --target llama-server -j && export LLAMA_SERVER_BIN=\$PWD/build/bin/llama-server"
  [ -z "$GPU_FLAG" ] || echo "  (-DGGML_CUDA=ON needs the CUDA toolkit; leave it out on a machine without an NVIDIA GPU)"
}
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
  if [ "$(uname -s)" = Darwin ]; then
    echo "llama-server not found. Install llama.cpp (brew install llama.cpp), or build the pinned commit:"
  else
    echo "llama-server not found. Build the pinned llama.cpp commit (needs git, cmake and a C++ compiler):"
  fi
  build_hint
  exit 1
fi
FOUND="$("$BIN" --version 2>&1 | sed -n 's/^version: *//p' | head -1)"
mkdir -p var/setup
printf 'binary: %s\nfound: %s\npinned: %s\n' "$BIN" "${FOUND:-unreadable}" "$PINNED_VERSION" > var/setup/engine.txt
if [ -z "$FOUND" ]; then
  echo "cannot read the version of $BIN (expected a 'version: ...' line from --version)"
  exit 2
fi
if [ "$FOUND" = "$PINNED_VERSION" ] || [[ "$FOUND" == *"$PINNED_BUILD"* ]]; then
  echo "ok  llama.cpp $FOUND ($BIN)"
  exit 0
fi
echo "WARNING: llama.cpp is '$FOUND', the benchmark was measured with '$PINNED_VERSION'."
echo "  python adapter.py runs it anyway and records engine_version_match=false (--strict-engine refuses);"
echo "  the UI needs --allow-engine-mismatch (scripts/run.sh passes it through). For the exact build:"
build_hint
exit 0
