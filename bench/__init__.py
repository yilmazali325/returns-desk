"""Benchmark harness (SPEC §9): run profiles through the real adapter, then report from raw files.

    bench/preflight.sh      refuses a machine that would contaminate timings (other GPU/CPU work,
                            battery, Low Power Mode, thermal limits) and records its state
    bench/run.py            profiles x runs in randomized interleaved blocks; each run is a fresh
                            `python3 adapter.py` process with its own llama-server; keeps every file
    bench/report.py         results/REPORT.md + CSVs, computed only from the files run.py saved

Besides the model runs, bench.run records two code-only floors per suite (evals/holdout/run_stub.py:
an agent that only ever asks a generic question, and the repo's deterministic double). The report
subtracts the always-clarify floor from F, so what the model adds is visible on its own.

The constants below are shared by run.py and report.py so both agree on suite names, file layout
and which profiles play the baseline, final and A/A roles. code_fingerprint() names the decision
code a run or floor executed, so the report never subtracts a floor from a run of other code.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

BASELINE = "B"  # the competent first version every change is compared with
FINAL = "F"  # the shipped candidate
AA_REPEAT = "B-repeat"  # identical to B under another name: the A/A noise floor


@dataclass(frozen=True)
class Suite:
    """A named input file plus the oracle labels the scorer compares results with."""

    name: str
    input_path: str  # repo-relative, so argv.txt shows the literal grader-style command
    labels_path: str


    def available(self, root: Path = ROOT) -> bool:
        """Both files on disk (a held-out suite may still be being written)."""
        return (root / self.input_path).is_file() and (root / self.labels_path).is_file()


SUITES: dict[str, Suite] = {
    "public": Suite("public", "candidate_assets/public_cases.json", "evals/expected/public_expected.json"),
    "variants": Suite("variants", "evals/variants/variants_v1.json", "evals/variants/labels_v1.json"),
    # Held-out suites, written blind (never used for tuning): the honest generalisation numbers.
    "holdout_v2": Suite("holdout_v2", "evals/holdout/holdout_v2.json", "evals/holdout/labels_v2.json"),
    "holdout_v3": Suite("holdout_v3", "evals/holdout_v3/holdout_v3.json", "evals/holdout_v3/labels_v3.json"),
}
PUBLIC_SUITE = "public"
VARIANT_SUITE = "variants"
FLOOR_MODES = ("always-clarify", "fake-policy")  # evals/holdout/run_stub.py: code-only floors, no model


DECISION_CODE = ("desk", "adapter.py")  # what decides a turn; bench/, server/ and evals/ do not


def code_fingerprint(root: Path = ROOT) -> str:
    """sha256 over the decision code (every desk/**/*.py and adapter.py, path and bytes, sorted).

    Recorded in every run.json and floor.json; two results are comparable only when it is equal."""
    digest = hashlib.sha256()
    files = [root / name for name in DECISION_CODE if (root / name).is_file()]
    files += [path for name in DECISION_CODE if (root / name).is_dir() for path in (root / name).rglob("*.py")]
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
