#!/bin/bash
# Every test that needs no real model: unit + review tests, then the Playwright E2E suite against
# a fake-model UI server (about 2 minutes). Extra arguments go to pytest, e.g. `scripts/test.sh -x`.
#   fast loop only:    .venv/bin/pytest                 (unit + review, ~40 s)
#   browser only:      .venv/bin/pytest -m e2e
#   real model:        .venv/bin/pytest -m llm          (needs llama-server and the model files)
# E2E screenshots go to var/e2e/shots/; the presentation shots are written only by scripts/screenshots.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/pytest ] || { echo "No .venv here: run scripts/setup.sh first." >&2; exit 1; }
exec .venv/bin/pytest -m "not llm" "$@"
