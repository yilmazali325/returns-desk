"""The frozen evaluation files are reproducible from their builders, and the code-only floor stub.

- evals/holdout/build_holdout.py regenerates holdout_v2.json and labels_v2.json byte for byte. The build
  fails if the oracle disagrees with any hand label, so this also checks that oracle fixes (review B4,
  B5) did not move a held-out label.
- evals/holdout/probes/build_probes.py regenerates the post-hoc probes the same way.
- evals/holdout/run_stub.py's always-clarify agent (the floor the report subtracts) never acts.
- evals/variants/metamorphic.shift_dates leaves an order without a delivery date alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from desk.agents.toolspecs import HANDOFF, REPLY, REPORT, RESOLUTION_TOOLS, operations_tools
from evals.holdout import build_holdout
from evals.holdout.probes import build_probes
from evals.holdout.run_stub import AlwaysClarifyLLM
from evals.variants.metamorphic import shift_dates

ROOT = Path(__file__).resolve().parents[2]
HOLDOUT = ROOT / "evals" / "holdout"


def test_build_holdout_regenerates_the_frozen_v2_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_holdout, "HERE", tmp_path)
    build_holdout.main()
    for name in ("holdout_v2.json", "labels_v2.json"):
        assert (tmp_path / name).read_text() == (HOLDOUT / name).read_text(), name


def test_build_holdout_refuses_a_hand_label_the_oracle_disagrees_with(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_holdout, "HERE", tmp_path)
    first = build_holdout.CASES[0]
    monkeypatch.setitem(first, "expect", ["completed"] * len(first["expect"]))
    with pytest.raises(AssertionError, match="oracle"):
        build_holdout.main()
    assert not (tmp_path / "labels_v2.json").exists()


def test_build_probes_regenerates_the_frozen_probe_files(tmp_path: Path) -> None:
    build_probes.main(tmp_path)
    for name in ("probes.json", "probe_labels.json"):
        assert (tmp_path / name).read_text() == (HOLDOUT / "probes" / name).read_text(), name


def test_the_always_clarify_floor_only_asks_or_reports_needs_info() -> None:
    stub = AlwaysClarifyLLM()
    resolution, operations = RESOLUTION_TOOLS, operations_tools(structured_report=False)
    asked = stub.step("resolution", [], resolution, [HANDOFF, REPLY], 64)
    assert (asked.tool, asked.args["message"]) == (REPLY, "Could you tell me a bit more about what you need?")
    reported = stub.step("operations", [], operations, [REPORT], 64)
    assert (reported.tool, reported.args) == (REPORT, {"outcome": "needs_info", "summary": "Need more information."})
    structured = stub.step("operations", [], operations_tools(structured_report=True), [REPORT], 64)
    assert structured.args == {"outcome": "needs_info"}


def test_the_always_clarify_floor_takes_the_first_allowed_tool_when_its_usual_one_is_not_offered() -> None:
    stub = AlwaysClarifyLLM()
    resolution, operations = RESOLUTION_TOOLS, operations_tools(structured_report=True)
    assert stub.step("resolution", [], resolution, [HANDOFF], 64).tool == HANDOFF
    first_ops_tool = next(t.name for t in operations if t.name != REPORT)
    assert stub.step("operations", [], operations, [first_ops_tool], 64).tool == first_ops_tool


def test_shift_dates_moves_delivery_dates_and_leaves_a_missing_one_missing() -> None:
    case = json.loads((ROOT / "candidate_assets" / "public_cases.json").read_text())["cases"][0]
    case["initial_state"]["orders"].append({"order_id": "O-NODATE", "customer_id": case["customer_id"],
                                            "status": "shipped", "delivered_on": None, "lines": []})
    shifted = shift_dates(case, 3, "shifted")
    orders = {o["order_id"]: o for o in shifted["initial_state"]["orders"]}
    assert orders["O-NODATE"]["delivered_on"] is None
    assert orders[case["initial_state"]["orders"][0]["order_id"]]["delivered_on"] != \
        case["initial_state"]["orders"][0]["delivered_on"]
