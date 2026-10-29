#!/bin/zsh
# One-time setup (needs internet). Afterwards everything runs offline.
# 0) llama.cpp build check (pinned: 0.7.0, build 11669, commit 620d2225c; recorded in var/setup/engine.txt)
# 1) Python env with pinned deps  2) Playwright browser for UI tests  3) model files (pinned revision + sha256)
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/"; exit 1; }
# The engine build is pinned (desk/profiles.py ENGINE_VERSION); another build works but is flagged.
scripts/check_engine.sh || exit 1
uv sync --frozen
.venv/bin/python -m playwright install chromium
REV=e87f176479d0855a907a41277aca2f8ee7a09523
mkdir -p models
fetch() {  # repo file sha256
  local repo=$1 file=$2 sha=$3
  if [[ -f models/$file ]] && [[ "$(shasum -a 256 models/$file | cut -d' ' -f1)" == "$sha" ]]; then echo "ok  $file"; return; fi
  echo "get $file"; curl -L --fail --retry 5 -C - -o "models/$file" "https://huggingface.co/$repo/resolve/$REV_FOR/$file"
  [[ "$(shasum -a 256 models/$file | cut -d' ' -f1)" == "$sha" ]] || { echo "sha256 mismatch for $file"; exit 1; }
}
REV_FOR=$REV fetch unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf 00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4
REV_FOR=$REV fetch unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q8_0.gguf 10cc391b403021dd11c614679d2fd92f611c3681d29e29651b717316965d61e1
echo "Setup complete. Next: scripts/run.sh (UI) or python adapter.py --input candidate_assets/public_cases.json --output results.json"
