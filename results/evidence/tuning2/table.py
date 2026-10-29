"""One markdown row per results file: strict n/N, turn accuracy, invariant violations, median case
ms, LLM calls/case, guard+gate blocks, fallbacks, refund/exchange records.

    .venv/bin/python var/tuning2/table.py <suite> <label>=<results.json> [...]
suite: pub var hold adv probe live2 live3
"""
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUITES = {
    "pub": ("candidate_assets/public_cases.json", "evals/expected/public_expected.json"),
    "var": ("evals/variants/variants_v1.json", "evals/variants/labels_v1.json"),
    "hold": ("evals/holdout/holdout_v2.json", "evals/holdout/labels_v2.json"),
    "adv": ("tests/review/business/cases/adversarial_v1.json", "tests/review/business/cases/adversarial_labels.json"),
    "probe": ("evals/holdout/probes/probes.json", "evals/holdout/probes/probe_labels.json"),
    "live2": ("tests/review/verify_redteam/live2.json", "tests/review/verify_redteam/live2_expected.json"),
    "live3": ("tests/review/verify_redteam/live3.json", "tests/review/verify_redteam/live3_expected.json"),
}


def score(suite: str, out: str) -> tuple[int, int, float, int]:
    inp, labels = SUITES[suite]
    doc = json.load(open(out))
    if suite in ("live2", "live3"):  # decision lists (+ replacement SKU) only; no oracle labels exist
        want = json.load(open(ROOT / labels))
        ok = turns = turns_ok = 0
        for r in doc["results"]:
            exp = want[r["case_id"]]
            decisions = exp["decisions"] if isinstance(exp, dict) else exp
            got = [t["decision"] for t in r["turns"]]
            sku_ok = True
            if isinstance(exp, dict) and exp.get("replacement_sku"):
                sku_ok = [x.get("replacement_sku") for x in r["final_state"]["exchanges"]] == [exp["replacement_sku"]]
            ok += got == decisions and sku_ok
            turns += len(decisions)
            turns_ok += sum(a == b for a, b in zip(got, decisions))
        return ok, len(doc["results"]), turns_ok / turns, 0
    score_json = out.replace(".json", ".score.json")
    subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "evals.score", "--input", inp, "--labels", labels,
                    "--output", out, "--json", score_json], cwd=ROOT, capture_output=True, check=False)
    m = json.load(open(score_json))["summary"]
    return m["strict_success"], m["cases"], m["turn_decision_accuracy"], m["invariant_violations"]


def row(suite: str, label: str, out: str) -> str:
    ok, n, acc, inv = score(suite, out)
    res = json.load(open(out))["results"]
    ms = statistics.median([x["metrics"].get("elapsed_ms") or 0 for x in res])
    calls = sum(x["metrics"].get("llm_calls") or 0 for x in res) / len(res)
    kinds = [a["kind"] for x in res for a in x["action_audit"]]
    blocks, falls = sum(k in ("guard", "gate") for k in kinds), kinds.count("fallback")
    money = sum(len(x["final_state"]["refunds"]) + len(x["final_state"]["exchanges"]) for x in res)
    return f"| {label} | {ok}/{n} | {acc:.3f} | {inv} | {ms:.0f} | {calls:.2f} | {blocks} | {falls} | {money} |"


if __name__ == "__main__":
    suite = sys.argv[1]
    print("| run | strict | turn acc | inv. viol. | median case ms | LLM calls/case | guard+gate blocks | fallbacks | refund+exchange records |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arg in sys.argv[2:]:
        label, path = arg.split("=", 1)
        print(row(suite, label, path))
