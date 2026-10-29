"""Structural and declared-run consistency checks only; not a business oracle."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(input_path, output_path, baseline_path=None):
    raw = Path(input_path).read_bytes()
    inputs = json.loads(raw)
    result = json.loads(Path(output_path).read_text())
    require(result.get("contract_version") == 1, "contract_version must be 1")
    require(result.get("input_sha256") == hashlib.sha256(raw).hexdigest(), "input_sha256 mismatch")
    for field in ("hardware", "configuration"):
        require(isinstance(result.get(field), dict) and result[field], field + " must be a nonempty object")
    require(isinstance(result.get("results"), list), "results must be an array")
    expected_ids = [case["case_id"] for case in inputs["cases"]]
    actual_ids = [item.get("case_id") for item in result["results"]]
    require(actual_ids == expected_ids, "case IDs or order differ")
    for case, item in zip(inputs["cases"], result["results"]):
        require(isinstance(item.get("turns"), list), "turns must be an array")
        require(len(item["turns"]) == len(case["turns"]), "turn count differs for " + case["case_id"])
        for turn in item["turns"]:
            require(turn.get("decision") in {"propose", "clarify", "deny", "completed", "error"}, "invalid decision")
            require(isinstance(turn.get("response"), str), "response must be text")
            require("proposal" in turn and (turn["proposal"] is None or isinstance(turn["proposal"], dict)), "invalid proposal")
        require(isinstance(item.get("action_audit"), list), "action_audit must be an array")
        require(isinstance(item.get("final_state"), dict), "final_state must be an object")
        require(set(case["initial_state"]).issubset(item["final_state"]), "missing final state fields")
        require(isinstance(item.get("metrics"), dict), "metrics must be an object")
        for key in ("elapsed_ms", "peak_ram_mb", "peak_vram_mb"):
            require(key in item["metrics"], "missing metric " + key)
            value = item["metrics"][key]
            if value is None:
                require(key != "elapsed_ms", "elapsed_ms cannot be null")
            else:
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "invalid metric " + key)
    if baseline_path:
        baseline = json.loads(Path(baseline_path).read_text())
        require(baseline.get("input_sha256") == result["input_sha256"], "baseline input differs")
        require(baseline.get("hardware") == result["hardware"], "baseline hardware declaration differs")
        require([x.get("case_id") for x in baseline.get("results", [])] == actual_ids, "baseline cases differ")
    return len(expected_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline")
    args = parser.parse_args()
    try:
        count = check(args.input, args.output, args.baseline)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, "CHECK FAILED: " + str(error) + "\n")
    print(f"STRUCTURE CHECK PASSED: {count} cases. Business correctness was not scored.")
