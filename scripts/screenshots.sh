#!/bin/bash
# Regenerate the presentation screenshots (var/screenshots/, cited by docs/SPEC.md §10.1) from the
# current UI: runs the E2E suite against a fake-model server with E2E_SHOTS_DIR pointed there.
# Plain test runs never touch var/screenshots/ (they write var/e2e/shots/).
#   scripts/screenshots.sh [pytest args]     e.g. scripts/screenshots.sh -k team
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/pytest ] || { echo "No .venv here: run scripts/setup.sh first." >&2; exit 1; }
E2E_SHOTS_DIR="$PWD/var/screenshots" .venv/bin/pytest -m "e2e and not llm" tests/e2e "$@"
echo "screenshots: var/screenshots/"
