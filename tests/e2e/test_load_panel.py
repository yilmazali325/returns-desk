"""The Load test panel: start a load from the page, then read its live and final numbers."""

from __future__ import annotations

import re
from typing import Any

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(180)]

STATUS = '[data-testid="load-status"]'


def test_a_load_test_runs_from_the_page_and_shows_its_numbers(desk: Any) -> None:
    page = desk.page
    page.click('[data-testid="tab-load"]')
    assert page.get_attribute('[data-testid="panel-load"]', "hidden") is None
    page.fill('[data-testid="load-customers"]', "0")
    page.click('[data-testid="load-start"]')
    assert "whole number from 1" in page.inner_text(STATUS)  # refused in the page; nothing started
    page.fill('[data-testid="load-customers"]', "12")
    page.fill('[data-testid="load-concurrency"]', "4")
    page.click('[data-testid="load-start"]')
    page.wait_for_function(f"() => /^Finished/.test(document.querySelector('{STATUS}').textContent)", timeout=120_000)
    assert "12 of 12 customers ok, 0 failed" in page.inner_text(STATUS)
    kpis = page.inner_text('[data-testid="load-kpis"]')
    assert "12 / 12" in kpis and "Reply time p50" in kpis and "Turns per second" in kpis
    charts = page.locator('[data-testid="load-charts"] figure')
    assert charts.count() == 3  # throughput, reply time, customers in flight (the test double has no line or model)
    assert page.locator('[data-testid="load-charts"] svg path.bar').count() > 0
    assert page.locator('[data-testid="load-charts"] svg path.line').count() >= 2
    flow = page.inner_text('[data-testid="load-flow"]')
    assert "Customers" in flow and "Done" in flow and "12" in flow
    box = page.locator('[data-testid="load-charts"] svg').first.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)  # hover reads out that moment
    assert re.match(r"\d+ s · Turns/s ", page.inner_text('[data-testid="load-charts"] .chart-tip >> nth=0'))
    result = page.inner_text('[data-testid="load-result"]')
    assert "No order line was refunded or exchanged twice." in result
    assert "Every copy of each scenario ended the same way." in result
    assert page.is_enabled('[data-testid="load-start"]') and page.is_hidden('[data-testid="load-stop"]')
    page.focus('[data-testid="tab-load"]')
    page.keyboard.press("ArrowRight")  # the tabs wrap around: load -> team
    assert page.get_attribute('[data-testid="tab-team"]', "aria-selected") == "true"
