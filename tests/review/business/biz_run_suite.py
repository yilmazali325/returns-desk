"""Run a whole contract input through run_case + FakePolicyLLM and write an adapter-shaped results
file that evals.score can grade (no llama-server needed).

    .venv/bin/python tests/review/business/biz_run_suite.py <cases.json> <results.json>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desk.profiles import FINAL  # noqa: E402
from desk.session import run_case  # noqa: E402
from desk.testing.fake_policy_llm import FakePolicyLLM  # noqa: E402


def main(inp: str, out: str) -> int:
    raw = Path(inp).read_bytes()
    doc = json.loads(raw)
    results = [run_case(case, FakePolicyLLM(), FINAL) for case in doc["cases"]]
    Path(out).write_text(json.dumps({"contract_version": 1, "input_sha256": hashlib.sha256(raw).hexdigest(),
                                     "hardware": {}, "configuration": {"engine": "fake-policy"},
                                     "results": results}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
