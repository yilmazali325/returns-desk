"""Build a contract-shaped results file straight from the oracle (no model involved).

Two uses: (1) prove the scorer gives 100% to a perfect run and that the variant inputs satisfy
candidate_assets/validate_submission.py; (2) give the desk team a concrete example of the exact
output the grader expects for every case. It is labelled as an oracle reference, never a benchmark.

CLI: python -m evals.reference --input cases.json --labels labels.json --output reference_results.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evals.oracle import expected_outcome

REFERENCE_NOTE = "oracle reference output: no model, no adapter; for scorer/validator self-checks only"


def reference_result(case: dict, intents: list[dict]) -> dict:
    """One contract result whose decisions, proposals and final_state come from the oracle."""
    outcome = expected_outcome(case, intents)
    turns = []
    for turn in outcome.turns:
        proposal = None
        if turn.proposal is not None:
            proposal = {"proposal_id": f"ORACLE-P-{turn.proposal['seq']}",
                        **{k: v for k, v in turn.proposal.items() if k != "seq"}}
        turns.append({"decision": turn.decision, "response": f"[oracle] {turn.reason}", "proposal": proposal})
    events = [{"event_id": f"ORACLE-EV-{n}", "source": "external", **event}
              for n, event in enumerate(outcome.events, start=1)]
    return {
        "case_id": case["case_id"],
        "turns": turns,
        "action_audit": [],
        "final_state": outcome.final_state,
        "proposals": outcome.proposals,
        "external_events": events,
        "metrics": {"elapsed_ms": 0, "peak_ram_mb": None, "peak_vram_mb": None, "notes": REFERENCE_NOTE},
    }


def reference_results(input_bytes: bytes, labels_doc: dict) -> dict:
    cases = json.loads(input_bytes)["cases"]
    intents_by_case = {label["case_id"]: label["intents"] for label in labels_doc["cases"]}
    return {
        "contract_version": 1,
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "hardware": {"note": REFERENCE_NOTE},
        "configuration": {"note": REFERENCE_NOTE},
        "results": [reference_result(case, intents_by_case[case["case_id"]]) for case in cases],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    labels = json.loads(Path(args.labels).read_text())
    document = reference_results(Path(args.input).read_bytes(), labels)
    Path(args.output).write_text(json.dumps(document, indent=1) + "\n")
    print(f"wrote {len(document['results'])} oracle reference results to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    raise SystemExit(main())
