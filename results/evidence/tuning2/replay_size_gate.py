"""Re-check every recorded propose_exchange handoff with check_requested_size with and without the
agent's requested_product, to find the handoffs whose gate verdict the other-product rule changes.

    .venv/bin/python var/tuning2/replay_size_gate.py <cases.json> <results.json> [...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from desk.grounding import analyze, check_requested_size  # noqa: E402
from desk.profiles import get_profile  # noqa: E402
from desk.sanitize import sanitize  # noqa: E402
from desk.session import CaseSession  # noqa: E402

cases = {c["case_id"]: c for c in json.load(open(sys.argv[1]))["cases"]}
total = changed = 0
for path in sys.argv[2:]:
    for result in json.load(open(path))["results"]:
        case = cases[result["case_id"]]
        session = CaseSession.from_case(case, None, get_profile("F"))
        lines, catalog = session.tools.customer_lines(), case["initial_state"]["products"]
        messages = [sanitize(t["message"]) for t in case["turns"]]
        for entry in result["action_audit"]:
            args = entry.get("arguments") or {}
            if entry["kind"] != "llm_step" or entry["tool"] != "handoff_to_operations" or args.get("intent") != "propose_exchange":
                continue
            line = next((x for x in lines if x["order_id"] == args.get("order_id") and x["line_id"] == args.get("line_id")), None)
            if line is None:
                continue
            total += 1
            a = analyze(messages[:entry["turn"]], lines, catalog, case["initial_state"]["as_of"])
            old = check_requested_size(a, args.get("requested_size"), line["sku"], catalog)
            new = check_requested_size(a, args.get("requested_size"), line["sku"], catalog, args.get("requested_product"))
            if old != new:
                changed += 1
                print(Path(path).name, result["case_id"], "turn", entry["turn"], args.get("requested_size"),
                      args.get("requested_product"), old, "->", new)
print(f"{total} exchange handoffs re-checked, {changed} verdicts change")
