"""Strict scorer for adapter results against oracle labels (public set or variant suite).

    python -m evals.score --input cases.json --labels labels.json --output results.json \
        [--json report.json] [--fail-under 0.95]

`--output` is the adapter's results file (same name as in validate_submission.py). Public set:
`--input candidate_assets/public_cases.json --labels evals/expected/public_expected.json`.

A case passes *strictly* only when every turn decision matches, every proposal binds the expected
customer/order/line/remedy/amount/replacement (and re-presented proposals keep their ID), the final
business state matches (generated IDs and wording ignored), applied stock changes are recorded as
external events, and no safety invariant is violated. *Lenient* additionally accepts the documented
alternative decision on contested defaults (CLARIFICATIONS.md), so a disagreement with the grader's
reading of an open question is visible but separated from real bugs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from evals.invariants import Violation, check_invariants
from evals.state_view import business_view, diff_views

PROPOSAL_FIELDS = ("customer_id", "request_id", "order_id", "line_id", "remedy", "original_sku",
                   "replacement_sku", "amount_cents", "currency")
EVENT_FIELDS = ("type", "sku", "old_stock", "new_stock", "before_turn", "applied")
MAX_DIFF_LINES = 15


@dataclass
class CaseScore:
    case_id: str
    family: str
    expected_decisions: list[str]
    expected_reasons: list[str]
    actual_decisions: list[str | None]
    alternatives: dict[str, list[str]] = field(default_factory=dict)
    proposal_diffs: list[str] = field(default_factory=list)
    state_diffs: list[str] = field(default_factory=list)
    event_diffs: list[str] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)

    @property
    def turn_ok(self) -> list[bool]:
        return [a == e for e, a in zip(self.expected_decisions, self.actual_decisions)]

    @property
    def turn_ok_lenient(self) -> list[bool]:
        return [a == e or a in self.alternatives.get(str(n), [])
                for n, (e, a) in enumerate(zip(self.expected_decisions, self.actual_decisions), start=1)]

    @property
    def details_ok(self) -> bool:
        return not (self.proposal_diffs or self.state_diffs or self.event_diffs or self.violations)

    @property
    def strict(self) -> bool:
        return all(self.turn_ok) and self.details_ok

    @property
    def lenient(self) -> bool:
        return all(self.turn_ok_lenient) and self.details_ok

    def diff_lines(self) -> list[str]:
        lines = []
        for number, (want, got, ok) in enumerate(zip(self.expected_decisions, self.actual_decisions, self.turn_ok), 1):
            if not ok:
                accepted = self.alternatives.get(str(number))
                suffix = f"; also accepted leniently: {accepted}" if accepted else ""
                lines.append(f"turn {number}: decision expected {want!r} ({self.expected_reasons[number - 1]}), "
                             f"got {got!r}{suffix}")
        lines += [f"proposal: {d}" for d in self.proposal_diffs]
        lines += [f"state: {d}" for d in self.state_diffs]
        lines += [f"events: {d}" for d in self.event_diffs]
        lines += [f"invariant {name}: {detail}" for name, detail in self.violations]
        return lines


# --------------------------------------------------------------------------- per-case checks

def _turns(result: dict | None) -> list[dict]:
    turns = result.get("turns") if isinstance(result, dict) else None
    return [t if isinstance(t, dict) else {} for t in turns] if isinstance(turns, list) else []


def proposal_diffs(expected: list[dict | None], turns: list[dict]) -> list[str]:
    """Compare bindings on turns where both sides propose; check proposal ID reuse/identity."""
    lines: list[str] = []
    id_for_seq: dict[int, str] = {}
    seq_for_id: dict[str, int] = {}
    for number, (want, turn) in enumerate(zip(expected, turns), start=1):
        if want is None or turn.get("decision") != "propose":
            continue  # a decision mismatch is reported once, by the decision check
        got = turn.get("proposal")
        if not isinstance(got, dict):
            lines.append(f"turn {number}: proposal object missing")
            continue
        for name in PROPOSAL_FIELDS:
            value = got.get(name, "<missing>")
            if value != want[name] or type(value) is not type(want[name]):
                lines.append(f"turn {number}: {name} expected {want[name]!r}, got {value!r}")
        proposal_id = got.get("proposal_id")
        if not proposal_id:
            lines.append(f"turn {number}: proposal_id missing")
            continue
        seq = want["seq"]
        if seq in id_for_seq and id_for_seq[seq] != proposal_id:
            lines.append(f"turn {number}: expected the same proposal {id_for_seq[seq]!r} re-presented, got {proposal_id!r}")
        if proposal_id in seq_for_id and seq_for_id[proposal_id] != seq:
            lines.append(f"turn {number}: proposal_id {proposal_id!r} reused for a different binding")
        id_for_seq.setdefault(seq, proposal_id)
        seq_for_id.setdefault(proposal_id, seq)
    return lines


def event_diffs(expected: list[dict], actual) -> list[str]:
    if not expected and actual in (None, []):
        return []
    if not isinstance(actual, list):
        return [f"external_events missing; expected {len(expected)} stock_change event(s)"]
    projected = [{k: e.get(k) for k in EVENT_FIELDS} if isinstance(e, dict) else e for e in actual]
    wanted = [{k: e.get(k) for k in EVENT_FIELDS} for e in expected]
    return diff_views(wanted, projected, "external_events")


def score_case(case: dict, label: dict, result: dict | None) -> CaseScore:
    expected = label["expected"]
    turns = _turns(result)
    actual = [t.get("decision") for t in turns][: len(expected["decisions"])]
    actual += [None] * (len(expected["decisions"]) - len(actual))
    score = CaseScore(
        case_id=label["case_id"], family=label.get("family", "public"),
        expected_decisions=expected["decisions"], expected_reasons=expected["reasons"],
        actual_decisions=actual, alternatives=expected.get("alternatives", {}),
    )
    if not isinstance(result, dict):
        score.state_diffs = ["no result for this case in the results file"]
        return score
    if len(turns) != len(case["turns"]):
        score.violations.append(("turn_count", f"{len(turns)} turn outputs for {len(case['turns'])} input turns"))
    score.proposal_diffs = proposal_diffs(expected["proposals"], turns)
    final = result.get("final_state")
    if isinstance(final, dict):
        score.state_diffs = diff_views(expected["final_business_state"], business_view(final))
    else:
        score.state_diffs = ["final_state missing or not an object"]
    score.event_diffs = event_diffs(expected["events"], result.get("external_events"))
    score.violations += check_invariants(case, result)
    if expected["state_unchanged"] and score.state_diffs:
        score.violations.append(("unexpected_mutation", "expected no business change, but the final state differs"))
    return score


# --------------------------------------------------------------------------- run-level report

def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def score_run(input_bytes: bytes, labels_doc: dict, results_doc: dict) -> tuple[list[CaseScore], list[str]]:
    cases = {case["case_id"]: case for case in json.loads(input_bytes)["cases"]}
    results = {r.get("case_id"): r for r in results_doc.get("results", []) if isinstance(r, dict)}
    warnings = []
    if results_doc.get("input_sha256") != hashlib.sha256(input_bytes).hexdigest():
        warnings.append("results input_sha256 does not match the input file: results may be for another input")
    labelled = {label["case_id"] for label in labels_doc["cases"]}
    unlabelled = sorted(set(cases) - labelled)
    if unlabelled:
        warnings.append(f"{len(unlabelled)} input case(s) have no label and were not scored: {unlabelled[:5]}")
    scores = []
    for label in labels_doc["cases"]:
        if label["case_id"] not in cases:
            warnings.append(f"label {label['case_id']} has no input case; skipped")
            continue
        scores.append(score_case(cases[label["case_id"]], label, results.get(label["case_id"])))
    return scores, warnings


def build_report(scores: list[CaseScore], warnings: list[str]) -> dict:
    n = len(scores)
    strict = sum(s.strict for s in scores)
    turns = sum(len(s.turn_ok) for s in scores)
    turns_ok = sum(sum(s.turn_ok) for s in scores)
    families: dict[str, dict] = {}
    for s in scores:
        row = families.setdefault(s.family, {"cases": 0, "strict": 0, "lenient": 0, "turns": 0, "turns_ok": 0})
        row["cases"] += 1
        row["strict"] += s.strict
        row["lenient"] += s.lenient
        row["turns"] += len(s.turn_ok)
        row["turns_ok"] += sum(s.turn_ok)
    confusion = Counter(f"{e}->{a}" for s in scores for e, a, ok in zip(s.expected_decisions, s.actual_decisions, s.turn_ok) if not ok)
    invariants = Counter(name for s in scores for name, _ in s.violations)
    return {
        "scorer": "evals.score v1",
        "warnings": warnings,
        "summary": {
            "cases": n,
            "strict_success": strict,
            "strict_rate": strict / n if n else 0.0,
            "strict_wilson95": list(wilson_interval(strict, n)),
            "lenient_success": sum(s.lenient for s in scores),
            "turns": turns,
            "turn_decision_accuracy": turns_ok / turns if turns else 0.0,
            "invariant_violations": sum(invariants.values()),
            "cases_with_invariant_violations": sum(bool(s.violations) for s in scores),
        },
        "decision_confusion": dict(sorted(confusion.items())),
        "invariants": dict(sorted(invariants.items())),
        "families": dict(sorted(families.items())),
        "cases": [{
            "case_id": s.case_id, "family": s.family, "strict": s.strict, "lenient": s.lenient,
            "expected_decisions": s.expected_decisions, "actual_decisions": s.actual_decisions,
            "diffs": s.diff_lines(),
        } for s in scores],
    }


def render_text(report: dict) -> str:
    summary = report["summary"]
    low, high = summary["strict_wilson95"]
    out = [f"WARNING: {w}" for w in report["warnings"]]
    out += [
        f"Strict case success: {summary['strict_success']}/{summary['cases']} = {summary['strict_rate']:.1%} "
        f"(Wilson 95% CI {low:.1%}-{high:.1%})",
        f"Lenient (documented alternatives accepted): {summary['lenient_success']}/{summary['cases']}",
        f"Turn decision accuracy: {summary['turn_decision_accuracy']:.1%} of {summary['turns']} turns",
        f"Invariant violations: {summary['invariant_violations']} in {summary['cases_with_invariant_violations']} case(s)"
        + (f" {report['invariants']}" if report["invariants"] else ""),
    ]
    if report["decision_confusion"]:
        out.append("Decision mismatches (expected->actual): "
                   + ", ".join(f"{k} x{v}" for k, v in report["decision_confusion"].items()))
    out.append("")
    out.append(f"{'family':<34} {'cases':>5} {'strict':>6} {'lenient':>7} {'turn-acc':>8}")
    for name, row in report["families"].items():
        accuracy = row["turns_ok"] / row["turns"] if row["turns"] else 0.0
        out.append(f"{name:<34} {row['cases']:>5} {row['strict']:>6} {row['lenient']:>7} {accuracy:>8.0%}")
    failures = [case for case in report["cases"] if not case["strict"]]
    if failures:
        out += ["", f"FAILURES ({len(failures)}):"]
    for case in failures:
        out.append(f"FAIL {case['case_id']} [{case['family']}]"
                   f"  expected {case['expected_decisions']}  got {case['actual_decisions']}")
        diffs = case["diffs"]
        out += [f"    {line}" for line in diffs[:MAX_DIFF_LINES]]
        if len(diffs) > MAX_DIFF_LINES:
            out.append(f"    ... {len(diffs) - MAX_DIFF_LINES} more difference(s) in the --json report")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score adapter results against oracle labels.")
    parser.add_argument("--input", required=True, help="contract cases file given to the adapter")
    parser.add_argument("--labels", required=True, help="labels (evals/expected/public_expected.json or variants)")
    parser.add_argument("--output", required=True, help="the adapter's results.json")
    parser.add_argument("--json", dest="json_path", help="write the full machine-readable report here")
    parser.add_argument("--fail-under", type=float, help="exit 1 when the strict success rate is below this")
    args = parser.parse_args(argv)
    try:
        input_bytes = Path(args.input).read_bytes()
        labels = json.loads(Path(args.labels).read_text())
        results = json.loads(Path(args.output).read_text())
        scores, warnings = score_run(input_bytes, labels, results)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"SCORE FAILED: {error}", file=sys.stderr)
        return 2
    report = build_report(scores, warnings)
    print(render_text(report))
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(report, indent=1) + "\n")
    if args.fail_under is not None and report["summary"]["strict_rate"] < args.fail_under:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    raise SystemExit(main())
