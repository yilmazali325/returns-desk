"""Evidence review: the brief asks the benchmark write-up to disclose CPU, GPU and count, RAM / unified
memory, OS, driver, model revision, quantization, inference engine and power settings, and to declare
seeds, concurrency, caches and timeouts with reproducible commands. The values exist in every
results.json; this checks whether bench/report.py surfaces them in REPORT.md, and whether the report
says what the compared profiles (B, F) actually are.

The report is rebuilt from results/smoke into tmp_path (nothing under results/ is written).
Tests that documented a review defect were marked xfail; the defects are fixed, so the markers are
gone and those tests now guard the fixes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.report import build_report

ROOT = Path(__file__).resolve().parents[3]
SMOKE = ROOT / "results" / "smoke"
pytestmark = pytest.mark.skipif(not (SMOKE / "public" / "F" / "run-1" / "results.json").exists(),
                                reason="no smoke bench results on disk")


@pytest.fixture(scope="module")
def report_text(tmp_path_factory: pytest.TempPathFactory) -> str:
    out = tmp_path_factory.mktemp("report") / "REPORT.md"
    build_report(SMOKE, out)
    return out.read_text()


@pytest.fixture(scope="module")
def raw() -> dict:
    return json.loads((SMOKE / "public" / "F" / "run-1" / "results.json").read_text())


def test_report_discloses_hardware(report_text: str, raw: dict) -> None:
    hw = raw["hardware"]
    missing = [f"{k}={hw[k]}" for k in ("cpu", "gpu", "os", "driver", "power") if str(hw[k]) not in report_text]
    if f"{hw['ram_gb']} GB" not in report_text and f"ram_gb {hw['ram_gb']}" not in report_text:
        missing.append(f"ram_gb={hw['ram_gb']}")
    assert not missing, f"REPORT.md does not disclose: {missing}"


def test_report_declares_configuration(report_text: str, raw: dict) -> None:
    cfg = raw["configuration"]
    wanted = {"model_revision": cfg["model_revision"], "quantization F": cfg["quantization"], "quantization B": "Q8_0",
              "engine_version": cfg["engine_version"], "protocol F": cfg["protocol"], "protocol B": "native",
              "cache_ram_mib": str(cfg["caches"]["cache_ram_mib"]), "llm_step_s": str(cfg["timeouts"]["llm_step_s"]).rstrip("0").rstrip("."),
              "concurrency": "concurrency"}
    missing = [k for k, v in wanted.items() if v not in report_text]
    assert not missing, f"REPORT.md does not declare: {missing}"


def test_report_gives_reproduce_command(report_text: str) -> None:
    assert "python -m bench.run" in report_text or "scripts/bench.sh" in report_text
