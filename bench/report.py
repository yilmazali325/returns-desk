"""Benchmark report (SPEC §9), computed only from the raw files bench/run.py saved.

    .venv/bin/python -m bench.report [--results results] [--out results/REPORT.md]

Input: results/<suite>/<profile>/run-<k>/{run.json, results.json, metrics_before.prom,
metrics_after.prom, run_conditions.json} plus every run-<k>.attempt-<n> directory (earlier attempts:
listed, never measured), and the code-only floors in results/<suite>/floors/<mode>/. Each
results.json is re-scored with the independent oracle (evals.score), so the report never trusts a
number it did not recompute. Output: REPORT.md, CSVs in csv/ and summary.json (public p50/p95).

The report opens with the setup disclosure the brief asks for (hardware, every profile's
configuration and engine flags, power readings, seeds, concurrency, caches, timeouts and the exact
commands), taken from the same raw files.

Statistical choices (stated again at the top of the report):
- Success = strict case success (every turn decision, every proposal binding, the final business
  state and zero invariant violations), as n/N with a Wilson 95% interval.
- Percentiles are nearest-rank: the value at rank ceil(p/100 * n) of the sorted sample, so they
  are always an observed value. With n < 20, p95 is the maximum, and the report says so.
- Latency comparisons are paired on (case_id, run k): run k of every profile ran in the same
  interleaved block. The effect is the geometric mean of the per-pair ratios with a percentile
  bootstrap 95% CI (10,000 resamples, random.Random(42)) that resamples whole cases: the runs of
  one case are near-copies (greedy decoding), so resampling them as independent pairs would
  make the interval too narrow. Pairs are successful in both profiles, so a failure cannot make
  a profile look fast.
- An effect counts only when its CI excludes 1 and its point estimate lies outside the A/A interval
  (B-repeat vs B); otherwise it is "no effect".
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench import AA_REPEAT, BASELINE, FINAL, FLOOR_MODES, PUBLIC_SUITE, RESULTS_DIR, ROOT, VARIANT_SUITE
from desk.agents import MAX_LLM_CALLS_PER_TURN
from desk.agents.resolution import MAX_RESOLUTION_STEPS
from desk.llm_server import parse_prometheus
from desk.profiles import PROFILES
from desk.telemetry import power_summary
from evals.score import CaseScore, score_run, wilson_interval

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
GATE_PUBLIC_RUNS = 3  # pre-registered: "public success (18 runs)" = 6 cases x 3 runs
RUN_DIR_RE = re.compile(r"^run-(\d+)$")
ATTEMPT_DIR_RE = re.compile(r"^run-(\d+)\.attempt-(\d+)$")
# Audit entry kinds that make up the decision trace (what the system did, not how it worded it).
TRACE_KINDS = {"llm_step", "tool", "fallback", "gate", "guard", "route", "grant", "confirmation", "fault",
               "target", "ops_report"}
# The five switches a profile is made of (configuration in results.json), for leave-one-out.
SWITCH_NAMES = {
    "quantization": "weights quantization",
    "protocol": "tool-call protocol",
    "prefetch_account_card": "prefetched account card",
    "structured_ops_report": "code-built ops report",
    "cache_ram_mib": "host prompt cache (--cache-ram MiB)",
}
NA = "n/a"
AGENT_FLOOR = "always-clarify"  # the floor F is measured against: an agent that decides nothing
# Every failure cause the report knows, always listed (a zero row is evidence too).
CANONICAL_CAUSES = ("oracle mismatch: decision", "oracle mismatch: binding/state/events", "invariant violation",
                    "LLM request timeout", "LLM error (not a timeout)", "step/call cap reached",
                    "case time budget exceeded", "crash")


# ==================================================================================================
# Statistics (pure)

def nearest_rank(values: Iterable[float], p: float) -> float | None:
    """Nearest-rank percentile: sorted(values)[ceil(p/100 * n) - 1]; None for an empty sample."""
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def median(values: Iterable[float]) -> float | None:
    data = list(values)
    return statistics.median(data) if data else None


def mean(values: Iterable[float]) -> float | None:
    data = list(values)
    return statistics.fmean(data) if data else None


def geometric_mean(ratios: list[float]) -> float | None:
    if not ratios or any(r <= 0 for r in ratios):
        return None
    return math.exp(statistics.fmean(math.log(r) for r in ratios))


def bootstrap_ci(groups: list[list[float]], resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED) -> tuple[float, float] | None:
    """Percentile bootstrap 95% CI of the geometric mean of ratios grouped by case.

    Each resample draws as many cases as there are, with replacement, and keeps all ratios
    (runs) of a drawn case together, so repeated runs of one case count as one piece of evidence.
    Needs at least 2 cases; None otherwise or when a ratio is not positive."""
    logs = [[math.log(r) for r in group] for group in groups if group and all(r > 0 for r in group)]
    if len(logs) < 2 or len(logs) != len([g for g in groups if g]):
        return None
    rng = random.Random(seed)
    k = len(logs)
    means = []
    for _ in range(resamples):
        drawn = [logs[rng.randrange(k)] for _ in range(k)]
        means.append(sum(sum(g) for g in drawn) / sum(len(g) for g in drawn))
    low, high = nearest_rank(means, 2.5), nearest_rank(means, 97.5)
    return math.exp(low), math.exp(high)  # type: ignore[arg-type]


def classify_effect(ratio: float | None, ci: tuple[float, float] | None,
                    noise: tuple[float, float] | None) -> str:
    """Name an effect only when the CI excludes 1 and the estimate is outside the A/A interval."""
    if ratio is None or ci is None:
        return "unavailable (fewer than 2 pairs)"
    if ci[0] <= 1.0 <= ci[1]:
        return "no effect (CI includes 1)"
    if noise is not None and noise[0] <= ratio <= noise[1]:
        return "no effect (inside the A/A interval)"
    direction = "faster" if ratio < 1 else "slower"
    return f"{abs(1 - ratio):.1%} {direction}" + ("" if noise else " (no A/A noise floor measured)")


# ==================================================================================================
# Loading raw files

@dataclass
class Run:
    """One complete run directory, re-scored."""

    suite: str
    profile: str
    run: int
    directory: Path
    manifest: dict[str, Any]
    document: dict[str, Any]
    scores: dict[str, CaseScore]
    score_warnings: list[str]
    conditions: dict[str, Any]

    @property
    def clean(self) -> bool:
        return bool((self.manifest.get("preflight") or {}).get("passed"))

    @property
    def label(self) -> str:
        return f"{self.suite}/{self.profile}/run-{self.run}"


@dataclass
class Attempt:
    """Any run directory (complete or not), for the inventory: every attempted run is listed."""

    suite: str
    profile: str
    run: int
    directory: Path
    manifest: dict[str, Any]
    renamed: bool


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class Inputs:
    """Input and label files, read once each (runs of a suite share them)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._bytes: dict[str, bytes] = {}
        self._labels: dict[str, Any] = {}

    def input_bytes(self, rel: str) -> bytes:
        if rel not in self._bytes:
            self._bytes[rel] = (self.root / rel).read_bytes()
        return self._bytes[rel]

    def labels(self, rel: str) -> Any:
        if rel not in self._labels:
            self._labels[rel] = json.loads((self.root / rel).read_text())
        return self._labels[rel]


def discover(results_root: Path, root: Path = ROOT) -> tuple[list[Run], list[Attempt], list[str]]:
    """Walk results/<suite>/<profile>/run-*; returns complete runs, every attempt and warnings."""
    runs: list[Run] = []
    attempts: list[Attempt] = []
    warnings: list[str] = []
    inputs = Inputs(root)
    for run_dir in sorted(results_root.glob("*/*/run-*")):
        match = RUN_DIR_RE.match(run_dir.name) or ATTEMPT_DIR_RE.match(run_dir.name)
        if not run_dir.is_dir() or match is None:
            continue
        suite, profile = run_dir.parent.parent.name, run_dir.parent.name
        manifest = _read_json(run_dir / "run.json") or {}
        attempts.append(Attempt(suite, profile, int(match.group(1)), run_dir, manifest,
                                renamed=bool(ATTEMPT_DIR_RE.match(run_dir.name))))
        if attempts[-1].renamed or manifest.get("status") != "complete":
            continue
        run = load_run(run_dir, suite, profile, int(match.group(1)), manifest, inputs, warnings)
        if run is not None:
            runs.append(run)
    return runs, attempts, warnings


def load_run(run_dir: Path, suite: str, profile: str, k: int, manifest: dict[str, Any], inputs: Inputs,
             warnings: list[str]) -> Run | None:
    document = _read_json(run_dir / "results.json")
    if not isinstance(document, dict) or not isinstance(document.get("results"), list):
        warnings.append(f"{run_dir}: run.json says complete but results.json is unreadable; not measured")
        return None
    try:
        case_scores, score_warnings = score_run(inputs.input_bytes(manifest["input"]),
                                                inputs.labels(manifest["labels"]), document)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        warnings.append(f"{run_dir}: could not be scored ({type(exc).__name__}: {exc}); not measured")
        return None
    run = Run(suite, profile, k, run_dir, manifest, document, {s.case_id: s for s in case_scores},
              score_warnings, _read_json(run_dir / "run_conditions.json") or {})
    saved = _read_json(run_dir / "score.json")
    recomputed = sum(s.strict for s in case_scores)
    if isinstance(saved, dict) and saved.get("summary", {}).get("strict_success") != recomputed:
        warnings.append(f"{run.label}: score.json says {saved['summary'].get('strict_success')} strict, "
                        f"re-scoring gives {recomputed} (labels or scorer changed since the run)")
    return run


@dataclass
class Floor:
    """A code-only floor (evals/holdout/run_stub.py, no model) for one suite, re-scored."""

    suite: str
    mode: str
    directory: Path
    record: dict[str, Any]
    scores: dict[str, CaseScore]

    @property
    def strict(self) -> int:
        return sum(s.strict for s in self.scores.values())

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def code_sha256(self) -> str | None:
        return self.record.get("code_sha256")


def discover_floors(results_root: Path, root: Path = ROOT) -> tuple[list[Floor], list[str]]:
    """results/<suite>/floors/<mode>/{floor.json, results.json}, written by bench.run."""
    floors: list[Floor] = []
    warnings: list[str] = []
    inputs = Inputs(root)
    for record_path in sorted(results_root.glob("*/floors/*/floor.json")):
        record = _read_json(record_path)
        document = _read_json(record_path.parent / "results.json")
        if not isinstance(record, dict) or not isinstance(document, dict) or not isinstance(document.get("results"), list):
            warnings.append(f"{record_path.parent}: floor files unreadable; not measured")
            continue
        try:
            case_scores, _ = score_run(inputs.input_bytes(record["input"]), inputs.labels(record["labels"]), document)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.append(f"{record_path.parent}: floor could not be scored ({type(exc).__name__}: {exc})")
            continue
        if not case_scores:
            warnings.append(f"{record_path.parent}: floor has no labelled case; not measured")
            continue
        floors.append(Floor(record_path.parent.parent.parent.name, record_path.parent.name, record_path.parent,
                            record, {s.case_id: s for s in case_scores}))
    return floors, warnings


# ==================================================================================================
# Per-case and per-step rows

@dataclass
class CaseRow:
    suite: str
    profile: str
    run: int
    case_id: str
    scenario: str
    clean: bool
    strict: bool
    lenient: bool
    decision_ok: bool
    details_ok: bool
    violations: list[str]
    elapsed_ms: float
    turn_latency_ms: list[float]
    first_case: bool
    status: str
    error: str | None
    metrics: dict[str, Any]
    llm_latency_ms: float
    causes: list[str]
    injected: Counter[str]
    trace_hash: str
    text_hash: str
    protocol_codes: Counter[str]
    fallbacks: Counter[str]
    guards: Counter[str]


@dataclass
class StepRow:
    suite: str
    profile: str
    run: int
    case_id: str
    turn: int
    agent: str
    tool: str
    latency_ms: float
    prompt_ms: float
    predicted_ms: float
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    protocol_error: str | None


def _audit(result: dict[str, Any]) -> list[dict[str, Any]]:
    audit = result.get("action_audit")
    return [e for e in audit if isinstance(e, dict)] if isinstance(audit, list) else []


def _error_code(entry: dict[str, Any]) -> str | None:
    error = entry.get("error")
    return str(error.get("code")) if isinstance(error, dict) and error.get("code") else None


def decision_trace(result: dict[str, Any]) -> list[Any]:
    """What the system decided and did: turn decisions, proposal bindings, transaction ids, and the
    ordered audit of tool choices (arguments without free text, error codes). No timings, no wording."""
    turns = []
    for turn in result.get("turns") or []:
        proposal = turn.get("proposal") if isinstance(turn.get("proposal"), dict) else {}
        binding = [proposal.get(k) for k in ("proposal_id", "order_id", "line_id", "remedy", "amount_cents",
                                            "replacement_sku")]
        transaction = turn.get("transaction") if isinstance(turn.get("transaction"), dict) else {}
        turns.append([turn.get("decision"), binding, transaction.get("transaction_id")])
    steps = []
    for entry in _audit(result):
        if entry.get("kind") not in TRACE_KINDS:
            continue
        args = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
        args = {k: v for k, v in args.items() if k != "message"}  # reply wording is the text trace
        steps.append([entry.get("turn"), entry.get("kind"), entry.get("agent"), entry.get("tool"), args,
                      _error_code(entry)])
    return [turns, steps]


def text_trace(result: dict[str, Any]) -> list[Any]:
    """The words: every response and every reply the model drafted."""
    replies = [(e.get("arguments") or {}).get("message") for e in _audit(result)
               if e.get("kind") == "llm_step" and e.get("tool") == "reply_to_customer"]
    return [[t.get("response") for t in result.get("turns") or []], replies]


def short_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def failure_causes(score: CaseScore, result: dict[str, Any]) -> list[str]:
    """Why a case failed or struggled, from the oracle diff and the audit (injected faults excluded)."""
    causes: list[str] = []
    if not all(score.turn_ok):
        causes.append("oracle mismatch: decision")
    if score.proposal_diffs or score.state_diffs or score.event_diffs:
        causes.append("oracle mismatch: binding/state/events")
    if score.violations:
        causes.append("invariant violation")
    audit = _audit(result)
    llm_errors = [str((e.get("error") or {}).get("message", "")) for e in audit
                  if e.get("kind") == "llm_step" and _error_code(e) == "LLM_ERROR"]
    if any(message.startswith("timeout") for message in llm_errors):
        causes.append("LLM request timeout")
    if any(not message.startswith("timeout") for message in llm_errors):
        causes.append("LLM error (not a timeout)")
    if step_cap_reached(audit):
        causes.append("step/call cap reached")
    metrics = result.get("metrics") or {}
    if "time budget" in str(metrics.get("error", "")):
        causes.append("case time budget exceeded")
    elif metrics.get("status") == "failed" and not llm_errors:
        causes.append("crash")
    return causes


def step_cap_reached(audit: list[dict[str, Any]]) -> bool:
    """Inferred: a turn used the whole LLM call budget or every Resolution step (not audited directly)."""
    calls: Counter[Any] = Counter()
    resolution: Counter[Any] = Counter()
    for entry in audit:
        if entry.get("kind") == "llm_step":
            calls[entry.get("turn")] += 1
            resolution[entry.get("turn")] += entry.get("agent") == "resolution"
    return any(n >= MAX_LLM_CALLS_PER_TURN for n in calls.values()) or any(
        n >= MAX_RESOLUTION_STEPS for n in resolution.values())


def injected_faults(audit: list[dict[str, Any]], turns: list[dict[str, Any]]) -> Counter[str]:
    """Faults the case injected on purpose (expected behaviour, reported apart from failures)."""
    found: Counter[str] = Counter()
    fired_turns = set()
    for entry in audit:
        if entry.get("kind") == "fault" and entry.get("tool") == "stock_change":
            applied = (entry.get("result") or {}).get("applied")
            found["stock_change applied" if applied else "stock_change not applied (unknown SKU or turn)"] += 1
        elif entry.get("kind") == "fault" and entry.get("tool") == "timeout_after_commit":
            found["timeout_after_commit armed"] += 1
        elif entry.get("kind") == "tool" and _error_code(entry) == "TIMEOUT":
            found["timeout_after_commit fired"] += 1
            fired_turns.add(entry.get("turn"))
    for number in fired_turns:
        if isinstance(number, int) and 0 < number <= len(turns) and turns[number - 1].get("decision") == "completed":
            found["fired timeout reconciled to completed in the same turn"] += 1
    return found


def case_rows(run: Run, families: dict[str, str]) -> list[CaseRow]:
    rows = []
    for result in run.document["results"]:
        case_id = str(result.get("case_id"))
        score = run.scores.get(case_id)
        if score is None:
            continue  # unlabelled case: score_run already warned
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        audit = _audit(result)
        steps = [e for e in audit if e.get("kind") == "llm_step"]
        rows.append(CaseRow(
            suite=run.suite, profile=run.profile, run=run.run, case_id=case_id,
            scenario=families.get(case_id, case_id), clean=run.clean, strict=score.strict,
            lenient=score.lenient, decision_ok=all(score.turn_ok),
            details_ok=not (score.proposal_diffs or score.state_diffs or score.event_diffs),
            violations=[name for name, _ in score.violations],
            elapsed_ms=float(metrics.get("elapsed_ms") or 0.0),
            turn_latency_ms=[float(x) for x in metrics.get("turn_latency_ms") or []],
            first_case=bool(metrics.get("first_case_after_start")), status=str(metrics.get("status")),
            error=metrics.get("error"), metrics=metrics,
            llm_latency_ms=sum(float((e.get("result") or {}).get("latency_ms") or 0.0) for e in steps),
            causes=failure_causes(score, result),
            injected=injected_faults(audit, result.get("turns") or []),
            trace_hash=short_hash(decision_trace(result)), text_hash=short_hash(text_trace(result)),
            protocol_codes=Counter(_error_code(e) or "?" for e in steps
                                   if _error_code(e) and _error_code(e) != "LLM_ERROR"),
            fallbacks=Counter(str(e.get("tool")) for e in audit if e.get("kind") == "fallback"),
            guards=Counter(str(e.get("tool")) for e in audit if e.get("kind") == "guard"),
        ))
    return rows


def step_rows(run: Run) -> list[StepRow]:
    rows = []
    for result in run.document["results"]:
        for entry in _audit(result):
            telemetry = entry.get("result")
            if entry.get("kind") != "llm_step" or not isinstance(telemetry, dict):
                continue  # LLM errors have no timings
            rows.append(StepRow(
                suite=run.suite, profile=run.profile, run=run.run, case_id=str(result.get("case_id")),
                turn=int(entry.get("turn") or 0), agent=str(entry.get("agent")), tool=str(entry.get("tool")),
                latency_ms=float(telemetry.get("latency_ms") or 0.0),
                prompt_ms=float(telemetry.get("prompt_ms") or 0.0),
                predicted_ms=float(telemetry.get("predicted_ms") or 0.0),
                prompt_tokens=int(telemetry.get("prompt_tokens") or 0),
                cached_tokens=int(telemetry.get("cached_tokens") or 0),
                completion_tokens=int(telemetry.get("completion_tokens") or 0),
                protocol_error=_error_code(entry)))
    return rows


def label_families(inputs_root: Path, runs: list[Run]) -> dict[str, dict[str, str]]:
    """suite -> case_id -> scenario: the label family, or the case id when every family is 'public'."""
    families: dict[str, dict[str, str]] = {}
    for run in runs:
        if run.suite in families:
            continue
        labels = Inputs(inputs_root).labels(run.manifest["labels"])["cases"]
        named = {c["case_id"]: c.get("family", "public") for c in labels}
        generic = set(named.values()) <= {"public"}
        families[run.suite] = {cid: (cid if generic else fam) for cid, fam in named.items()}
    return families


# ==================================================================================================
# Report model: everything the sections need, plus the "unavailable" list

@dataclass
class Model:
    runs: list[Run]
    attempts: list[Attempt]
    cases: list[CaseRow]
    steps: list[StepRow]
    warnings: list[str]
    unavailable: list[tuple[str, str]] = field(default_factory=list)
    floors: list[Floor] = field(default_factory=list)

    def suites(self) -> list[str]:
        return _suite_order({r.suite for r in self.runs})

    def quality_suites(self) -> list[str]:
        """Suites with model runs or code-only floors (a floor alone is still worth showing)."""
        return _suite_order({r.suite for r in self.runs} | {f.suite for f in self.floors})

    def floor(self, suite: str, mode: str) -> Floor | None:
        return next((f for f in self.floors if f.suite == suite and f.mode == mode), None)

    def profiles(self, suite: str) -> list[str]:
        names = {r.profile for r in self.runs if r.suite == suite}
        return sorted(names, key=profile_order)

    def case_rows(self, suite: str, profile: str) -> list[CaseRow]:
        return [c for c in self.cases if c.suite == suite and c.profile == profile]

    def run_list(self, suite: str, profile: str) -> list[Run]:
        return sorted((r for r in self.runs if r.suite == suite and r.profile == profile), key=lambda r: r.run)

    def missing(self, metric: str, reason: str) -> None:
        if (metric, reason) not in self.unavailable:
            self.unavailable.append((metric, reason))


def _suite_order(names: set[str]) -> list[str]:
    order = {PUBLIC_SUITE: 0, VARIANT_SUITE: 1}
    return sorted(names, key=lambda s: (order.get(s, 2), s))


def profile_order(name: str) -> tuple[int, str]:
    rank = {BASELINE: 0, FINAL: 1, AA_REPEAT: 9}
    return (rank.get(name, 5 if name.startswith("F-") else 7), name)


def build_model(results_root: Path, root: Path = ROOT) -> Model:
    runs, attempts, warnings = discover(results_root, root)
    families = label_families(root, runs)
    cases = [row for run in runs for row in case_rows(run, families[run.suite])]
    steps = [row for run in runs for row in step_rows(run)]
    for run in runs:
        warnings += [f"{run.label}: {w}" for w in run.score_warnings]
    floors, floor_warnings = discover_floors(results_root, root)
    return Model(runs, attempts, cases, steps, warnings + floor_warnings, floors=floors)


# ==================================================================================================
# Formatting helpers

def fmt(value: float | int | None, digits: int = 0) -> str:
    if value is None:
        return NA
    return f"{value:,.{digits}f}"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    return NA if value is None else f"{value:.{digits}%}"


def fmt_ratio(value: float | None) -> str:
    return NA if value is None else f"{value:.3f}"


def fmt_ci(ci: tuple[float, float] | None) -> str:
    return NA if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines + [""]


def pctl_label(n: int) -> str:
    return f"n={n}" + (", p95 = max" if 0 < n < 20 else "")


# ==================================================================================================
# Setup disclosure: hardware, profiles, configuration, power, commands

HARDWARE_KEYS = ("cpu", "cpu_cores", "gpu", "gpu_count", "ram_gb", "unified_memory", "os", "driver", "power")


def first_run_per_profile(model: Model) -> list[Run]:
    seen: dict[str, Run] = {}
    for run in sorted(model.runs, key=lambda r: (profile_order(r.profile), r.suite, r.run)):
        seen.setdefault(run.profile, run)
    return list(seen.values())


def section_hardware(model: Model) -> list[str]:
    first = model.runs[0].document.get("hardware") or {}
    differ = [r.label for r in model.runs if (r.document.get("hardware") or {}) != first]
    rows = [[key, f"{first.get(key)} GB unified memory" if key == "ram_gb" else first.get(key, NA)] for key in HARDWARE_KEYS]
    out = ["### Hardware (results.json `hardware`, identical in every run)" if not differ
           else "### Hardware (results.json `hardware` of the first run)", ""]
    if differ:
        out += [f"**Warning:** the hardware declaration differs in {len(differ)} run(s): {', '.join(differ)}.", ""]
    return out + table(["field", "value"], rows)


def section_profiles() -> list[str]:
    """Every profile the bench knows (desk/profiles.py), so B, F and each ablation are defined here."""
    rows = [[p.name, p.quantization, p.protocol, "yes" if p.prefetch else "no", "yes" if p.structured_report else "no",
             p.cache_ram_mib, p.description] for p in PROFILES.values()]
    return ["### Profiles", "",
            "B is the first competent version, F the shipped candidate; each F-* reverts exactly one switch to "
            "B's value (leave-one-out), B-q4 isolates quantization, B-repeat is an A/A copy of B.", ""] + \
        table(["profile", "quantization", "tool-call protocol", "prefetched account card", "code-built ops report",
               "--cache-ram MiB", "definition"], rows)


def section_configuration(model: Model) -> list[str]:
    rows = []
    for run in first_run_per_profile(model):
        c = run.document.get("configuration") or {}
        caches, timeouts, tokens = c.get("caches") or {}, c.get("timeouts") or {}, c.get("max_tokens") or {}
        sha = str(c.get("model_sha256") or NA)
        rows.append([run.profile, f"{c.get('model_file', NA)} ({c.get('quantization', NA)})",
                     f"`{sha[:16]}…` verified={c.get('model_sha256_verified')}, source={c.get('model_sha256_source') or 'not recorded'}",
                     f"{c.get('engine', NA)} {c.get('engine_version', NA)} (match={c.get('engine_version_verified')})",
                     c.get("protocol", NA), c.get("seed", NA), c.get("temperature", NA),
                     f"tool {tokens.get('tool_step', NA)} / reply {tokens.get('reply_step', NA)}",
                     f"{c.get('ctx_per_slot', NA)} x {c.get('slots', NA)} slot(s)", c.get("concurrency", NA),
                     f"cache_prompt={caches.get('cache_prompt')}, cache_ram_mib={caches.get('cache_ram_mib')}, "
                     f"ctx_checkpoints={caches.get('ctx_checkpoints')}; {caches.get('persistence', NA)}",
                     f"llm_step_s {timeouts.get('llm_step_s', NA)}, retries {timeouts.get('llm_http_retries', NA)}, "
                     f"case_s {timeouts.get('case_s', NA)}"])
    revision = {(r.document.get("configuration") or {}).get("model_revision") for r in model.runs}
    out = ["### Configuration per profile (results.json `configuration`, first run of each profile)", "",
           f"Model: {', '.join(sorted({str((r.document.get('configuration') or {}).get('model_repo')) for r in model.runs}))} "
           f"at model_revision {', '.join(sorted(str(v) for v in revision))}. Greedy decoding (temperature 0) and a "
           "fixed seed; one request at a time (concurrency 1, one slot); the prompt cache persists within a run "
           "and is cleared by the fresh llama-server every run starts.", ""]
    out += table(["profile", "model file (quantization)", "sha256", "engine", "protocol", "seed", "temperature",
                  "max tokens", "context", "concurrency", "caches", "timeouts"], rows)
    out += ["Engine flags (llama-server argv after the model path and port):", ""]
    for run in first_run_per_profile(model):
        argv = (run.document.get("configuration") or {}).get("engine_argv") or []
        flags = argv[5:] if len(argv) > 5 else argv
        out.append(f"- {run.profile}: `{' '.join(str(a) for a in flags) or NA}`")
    return out + [""]


def _power(reading: Any) -> str:
    if not isinstance(reading, dict) or not reading:
        return NA
    return str(reading.get("power_summary") or power_summary(reading))


def section_power(model: Model) -> list[str]:
    rows = []
    for run in sorted(model.runs, key=lambda r: (r.suite, profile_order(r.profile), r.run)):
        measured = (run.document.get("configuration") or {}).get("run_conditions") or {}
        load = (measured.get("before") or {}).get("load_avg_1_5_15")
        rows.append([run.label, _power(measured.get("before")), _power(measured.get("after")),
                     ", ".join(str(x) for x in load) if load else NA,
                     "passed" if run.clean else "OVERRIDDEN", run.conditions.get("caffeinate", NA)])
    return ["### Power and machine state (measured with pmset at the start and end of every run)", ""] + \
        table(["run", "before", "after", "load 1/5/15 before", "preflight", "sleep"], rows)


def section_reproduce(model: Model, results_root: Path) -> list[str]:
    out = ["### Reproduce", "",
           "Whole matrix: `scripts/bench.sh` (every suite: model runs + code-only floors, then this report). "
           "One suite: `.venv/bin/python -m bench.run --suite <suite> --profiles B F --runs 3`; floors only: "
           f"`.venv/bin/python -m bench.run --suite <suite> --floors-only`; report: "
           f"`.venv/bin/python -m bench.report --results {_rel(results_root)}`. "
           "Exact adapter command of each profile's first run (argv.txt):", ""]
    for run in first_run_per_profile(model):
        argv = (run.directory / "argv.txt")
        command = argv.read_text().splitlines()[0] if argv.is_file() else NA
        out.append(f"- {run.profile}: `{command}`")
    return out + [""]


def section_setup(model: Model, results_root: Path) -> list[str]:
    out = ["## Setup: hardware, configuration and how to reproduce", ""]
    out += section_hardware(model) + section_profiles() + section_configuration(model)
    out += section_power(model) + section_reproduce(model, results_root)
    return out


# ==================================================================================================
# Quality and the gate

@dataclass
class Success:
    strict: int
    n: int
    runs: int
    violations: int
    per_run: list[str]

    @property
    def rate(self) -> float | None:
        return self.strict / self.n if self.n else None


def success(model: Model, suite: str, profile: str) -> Success | None:
    rows = model.case_rows(suite, profile)
    if not rows:
        return None
    per_run = [f"{sum(c.strict for c in rows if c.run == r.run)}/{sum(1 for c in rows if c.run == r.run)}"
               for r in model.run_list(suite, profile)]
    return Success(sum(c.strict for c in rows), len(rows), len(model.run_list(suite, profile)),
                   sum(len(c.violations) for c in rows), per_run)


DIFFERENT_CODE = "n/a: other code"


def agent_attributable(rate: float | None, floor: Floor | None, run_code: set[str | None]) -> str:
    """Strict rate above the always-clarify floor, in percentage points: what code alone does not reach.
    Only when the runs and the floor executed the same decision code (bench.code_fingerprint)."""
    if rate is None or floor is None:
        return NA
    if floor.code_sha256 is None or run_code != {floor.code_sha256}:
        return DIFFERENT_CODE
    return f"{100 * (rate - floor.strict / floor.n):+.1f} pp"


def code_of_runs(model: Model, suite: str, profile: str) -> set[str | None]:
    """The decision-code fingerprints of a profile's runs (None: recorded before fingerprints existed)."""
    return {r.manifest.get("code_sha256") for r in model.run_list(suite, profile)}


def _code_label(values: set[str | None]) -> str:
    """'code e5265954dee8' and/or 'no code fingerprint (...)' for a set of fingerprints."""
    known = sorted(v[:12] for v in values if v is not None)
    parts = [f"code {', '.join(known)}"] if known else []
    if None in values:
        parts.append("no code fingerprint (written before the bench recorded one)")
    return " and ".join(parts)


def floor_row(floor: Floor) -> list[Any]:
    low, high = wilson_interval(floor.strict, floor.n)
    scores = list(floor.scores.values())
    return [f"floor: {floor.mode} (no model)", f"{floor.strict}/{floor.n}", fmt_pct(floor.strict / floor.n),
            f"[{low:.1%}, {high:.1%}]", 1, f"{floor.strict}/{floor.n}", f"{sum(s.lenient for s in scores)}/{floor.n}",
            f"{sum(all(s.turn_ok) for s in scores)}/{floor.n}", sum(len(s.violations) for s in scores),
            "+0.0 pp" if floor.mode == AGENT_FLOOR else NA]


def section_quality(model: Model) -> list[str]:
    out = ["## Quality: strict case success vs the independent oracle", "",
           "Strict = every turn decision, proposal binding and the final business state match "
           "`evals/` labels and no safety invariant is violated. Pooled over runs; Wilson 95% interval. "
           "Pooled case-runs are not independent (the same cases repeat and decoding is greedy), so this "
           "interval is optimistic; 'per run' and the trace-identity section show how much runs differ.", "",
           "Code-only floors run the same desk (orchestrator, gates, gateway, store) with no model "
           "(`evals/holdout/run_stub.py`): `always-clarify` is an agent that only ever asks a generic question "
           "and reports needs_info, so whatever it passes is solved by code; `fake-policy` is the repo's "
           "deterministic test double. 'agent-attributable' = the profile's strict rate minus the "
           "always-clarify floor: the share of the suite only the model's decisions reach. It is computed "
           "only when the runs and the floor executed the same decision code (sha256 of desk/ and "
           "adapter.py, `code_sha256` in run.json and floor.json).", ""]
    for suite in model.quality_suites():
        rows = []
        notes: list[str] = []
        clarify_floor = model.floor(suite, AGENT_FLOOR)
        for profile in model.profiles(suite):
            s = success(model, suite, profile)
            assert s is not None
            low, high = wilson_interval(s.strict, s.n)
            cases = model.case_rows(suite, profile)
            attributable = agent_attributable(s.rate, clarify_floor, code_of_runs(model, suite, profile))
            rows.append([profile, f"{s.strict}/{s.n}", fmt_pct(s.rate), f"[{low:.1%}, {high:.1%}]", s.runs,
                         ", ".join(s.per_run), f"{sum(c.lenient for c in cases)}/{s.n}",
                         f"{sum(c.decision_ok for c in cases)}/{s.n}", s.violations, attributable])
            if attributable == DIFFERENT_CODE and clarify_floor is not None:
                notes.append(f"{profile} runs: {_code_label(code_of_runs(model, suite, profile))}; floor: "
                             f"{_code_label({clarify_floor.code_sha256})}.")
        for mode in FLOOR_MODES:
            floor = model.floor(suite, mode)
            if floor is not None:
                rows.append(floor_row(floor))
        if clarify_floor is None:
            model.missing(f"code-only floor ({suite})", f"no results/{suite}/floors/{AGENT_FLOOR}; run "
                          f"`python -m bench.run --suite {suite} --floors-only`")
        out += [f"### Suite `{suite}`", ""]
        out += table(["profile", "strict", "rate", "Wilson 95%", "runs", "per run", "lenient", "decisions ok",
                      "invariant violations", "agent-attributable"], rows)
        if notes:
            out += ["**agent-attributable not computed** (runs and floor executed different decision code; re-run "
                    "both from the same code with `scripts/bench.sh`): " + " ".join(notes), ""]
    return out


def switches(document: dict[str, Any]) -> dict[str, Any]:
    """A profile's five switches, read from results.json configuration (raw file, not desk.profiles)."""
    config = document.get("configuration") or {}
    app = config.get("app_switches") or {}
    return {"quantization": config.get("quantization"), "protocol": config.get("protocol"),
            "prefetch_account_card": app.get("prefetch_account_card"),
            "structured_ops_report": app.get("structured_ops_report"),
            "cache_ram_mib": (config.get("caches") or {}).get("cache_ram_mib")}


def leave_one_out(model: Model) -> dict[str, str]:
    """profile -> the switch it reverts: differs from F in exactly one switch, set to B's value."""
    by_profile = {r.profile: switches(r.document) for r in model.runs}
    final, base = by_profile.get(FINAL), by_profile.get(BASELINE)
    if final is None or base is None:
        return {}
    found = {}
    for name, values in sorted(by_profile.items(), key=lambda kv: profile_order(kv[0])):
        diff = [k for k in values if values[k] != final[k]]
        if name != FINAL and len(diff) == 1 and values[diff[0]] == base[diff[0]]:
            found[name] = diff[0]
    return found


def change_label(model: Model, switch: str) -> str:
    by_profile = {r.profile: switches(r.document) for r in model.runs}
    return f"{SWITCH_NAMES[switch]}: {by_profile[FINAL][switch]} (B: {by_profile[BASELINE][switch]})"


def at_least(a: Success | None, b: Success | None) -> bool | None:
    """a's strict rate >= b's; None when either side is missing."""
    if a is None or b is None or a.rate is None or b.rate is None:
        return None
    return a.rate >= b.rate


def gate_rows(model: Model) -> list[dict[str, Any]]:
    """Pre-registered gate (SPEC §9) per change, plus F as a whole.

    A change ships when F (which has it) has public AND variant-suite success >= B, F has zero
    invariant violations, and F's public success is >= that of F without the change. Missing data
    makes the verdict INCOMPLETE; fewer than 3 public runs makes a pass PROVISIONAL."""
    f_public, b_public = success(model, PUBLIC_SUITE, FINAL), success(model, PUBLIC_SUITE, BASELINE)
    f_var, b_var = success(model, VARIANT_SUITE, FINAL), success(model, VARIANT_SUITE, BASELINE)
    f_violations = sum((s.violations for s in (f_public, f_var) if s is not None))
    common = {"public F >= B": at_least(f_public, b_public), "variants F >= B": at_least(f_var, b_var),
              "F invariant violations = 0": (f_violations == 0) if (f_public or f_var) else None}
    rows = [{"change": "F as a whole (all changes)", "profile without": BASELINE, **common,
             "public F >= without": common["public F >= B"]}]
    for profile, switch in leave_one_out(model).items():
        without = success(model, PUBLIC_SUITE, profile)
        rows.append({"change": change_label(model, switch), "profile without": profile, **common,
                     "public F >= without": at_least(f_public, without)})
    public_runs = f_public.runs if f_public else 0
    for row in rows:
        row["verdict"] = gate_verdict(row, public_runs)
    return rows


GATE_CRITERIA = ("public F >= B", "variants F >= B", "F invariant violations = 0", "public F >= without")


def gate_verdict(row: dict[str, Any], public_runs: int) -> str:
    checks = [row[k] for k in GATE_CRITERIA]
    if any(c is False for c in checks):
        return "FAIL: F keeps B's choice"
    if any(c is None for c in checks):
        missing = [k for k in GATE_CRITERIA if row[k] is None]
        return "INCOMPLETE: no data for " + "; ".join(missing)
    if public_runs < GATE_PUBLIC_RUNS:
        return f"PASS (provisional: {public_runs} of {GATE_PUBLIC_RUNS} public runs)"
    return "PASS: ships in F"


def section_gate(model: Model) -> list[str]:
    out = ["## Pre-registered acceptance gate", "",
           "SPEC §9, written before any measurement: a change ships only if public success (18 case-runs) "
           "AND variant-suite success are >= B, with zero invariant violations (duplicate money/stock, "
           "invented success, mutation without a grant). Per change we also require F >= F-without-that-"
           "change on the public suite, so a change cannot ride on the others. Otherwise F keeps B's choice.", ""]
    rows = gate_rows(model)
    if len(rows) == 1:
        model.missing("per-change gate", "no leave-one-out profile (F with one switch reverted to B) has "
                      "complete runs, so only F as a whole is gated")
    yes_no = {True: "yes", False: "NO", None: NA}
    out += table(["change", "without it", *GATE_CRITERIA, "verdict"],
                 [[r["change"], r["profile without"], *[yes_no[r[k]] for k in GATE_CRITERIA], r["verdict"]]
                  for r in rows])
    return out


# ==================================================================================================
# Latency

def successful(rows: list[CaseRow]) -> list[CaseRow]:
    return [c for c in rows if c.strict]


def section_latency(model: Model) -> list[str]:
    out = ["## Latency", "",
           "Case latency = `metrics.elapsed_ms` (state reset + every turn; server start excluded). "
           "Percentiles are nearest-rank; `p95 = max` marks samples smaller than 20. "
           "All-attempts includes failed cases (an LLM timeout counts at the time it actually took).", ""]
    for suite in model.suites():
        rows = []
        for profile in model.profiles(suite):
            cases = model.case_rows(suite, profile)
            ok = successful(cases)
            ok_ms = [c.elapsed_ms for c in ok]
            turns = [t for c in ok for t in c.turn_latency_ms]
            all_ms = [c.elapsed_ms for c in cases]
            first = [c.elapsed_ms for c in ok if c.first_case]
            warm = [c.elapsed_ms for c in ok if not c.first_case]
            rows.append([profile, f"{len(ok)}/{len(cases)}", fmt(nearest_rank(ok_ms, 50)),
                         fmt(nearest_rank(ok_ms, 95)), pctl_label(len(ok_ms)), fmt(nearest_rank(turns, 50)),
                         fmt(nearest_rank(turns, 95)), pctl_label(len(turns)), fmt(nearest_rank(all_ms, 50)),
                         fmt(nearest_rank(all_ms, 95)), fmt(median(first)), fmt(median(warm)),
                         "" if all(c.clean for c in cases) else "DIRTY"])
        out += [f"### Suite `{suite}`: successful cases, ms", ""]
        out += table(["profile", "successful", "case p50", "case p95", "case sample", "turn p50", "turn p95",
                      "turn sample", "all-attempts p50", "all-attempts p95", "first case after start (median)",
                      "warm cases (median)", "conditions"], rows)
        out += section_scenarios(model, suite) + section_intersection(model, suite)
    return out


def section_scenarios(model: Model, suite: str) -> list[str]:
    profiles = model.profiles(suite)
    scenarios = sorted({c.scenario for c in model.cases if c.suite == suite})
    rows = []
    for scenario in scenarios:
        row: list[Any] = [scenario]
        for profile in profiles:
            cases = [c for c in model.case_rows(suite, profile) if c.scenario == scenario]
            ok = successful(cases)
            row.append(f"{fmt(median(c.elapsed_ms for c in ok))} ({len(ok)}/{len(cases)})" if cases else NA)
        rows.append(row)
    return ([f"Per-scenario median case latency, ms (successful/attempted) — suite `{suite}`", ""]
            + table(["scenario", *profiles], rows))


def section_intersection(model: Model, suite: str) -> list[str]:
    """Latency over the cases every profile solved in every run, so survivorship cannot help anyone."""
    profiles = model.profiles(suite)
    by_case: dict[str, list[bool]] = defaultdict(list)
    for c in model.cases:
        if c.suite == suite:
            by_case[c.case_id].append(c.strict)
    common = {cid for cid, oks in by_case.items() if all(oks)
              and all(any(c.case_id == cid for c in model.case_rows(suite, p)) for p in profiles)}
    rows = []
    for profile in profiles:
        ms = [c.elapsed_ms for c in model.case_rows(suite, profile) if c.case_id in common]
        rows.append([profile, len(ms), fmt(nearest_rank(ms, 50)), fmt(nearest_rank(ms, 95)), fmt(mean(ms))])
    return ([f"Intersection of successes — suite `{suite}`: {len(common)} of {len(by_case)} cases were strict "
             f"successes in every run of every profile listed; latency over exactly those case-runs, ms", ""]
            + table(["profile", "case-runs", "p50", "p95", "mean"], rows))


# ==================================================================================================
# Paired comparisons, A/A noise floor, leave-one-out

@dataclass
class Comparison:
    suite: str
    profile: str
    reference: str
    pairs: int
    excluded: int
    ratio: float | None
    ci: tuple[float, float] | None
    verdict: str = ""


def paired_ratios(a: list[CaseRow], b: list[CaseRow]) -> tuple[dict[str, list[float]], int]:
    """Ratios a/b per (case_id, run k) pair where both succeeded, grouped by case_id; also how
    many pairs were excluded because either side failed."""
    index = {(c.case_id, c.run): c for c in b}
    by_case: dict[str, list[float]] = defaultdict(list)
    excluded = 0
    for row in a:
        other = index.get((row.case_id, row.run))
        if other is None:
            continue
        if row.strict and other.strict and row.elapsed_ms > 0 and other.elapsed_ms > 0:
            by_case[row.case_id].append(row.elapsed_ms / other.elapsed_ms)
        else:
            excluded += 1
    return dict(by_case), excluded


def compare(model: Model, suite: str, profile: str, reference: str) -> Comparison:
    by_case, excluded = paired_ratios(model.case_rows(suite, profile), model.case_rows(suite, reference))
    ratios = [r for group in by_case.values() for r in group]
    return Comparison(suite, profile, reference, len(ratios), excluded, geometric_mean(ratios),
                      bootstrap_ci(list(by_case.values())))


def noise_floor(model: Model, suite: str) -> Comparison | None:
    if not model.case_rows(suite, AA_REPEAT) or not model.case_rows(suite, BASELINE):
        model.missing(f"A/A noise floor ({suite})", f"{AA_REPEAT} or {BASELINE} has no complete runs in this "
                      "suite; effects are judged by their CI alone")
        return None
    return compare(model, suite, AA_REPEAT, BASELINE)


def comparisons(model: Model) -> list[Comparison]:
    found = []
    loo = leave_one_out(model)
    for suite in model.suites():
        profiles = model.profiles(suite)
        aa = noise_floor(model, suite)
        noise = aa.ci if aa is not None else None
        pairs = [(p, BASELINE) for p in profiles if p != BASELINE]
        pairs += [(FINAL, p) for p in profiles if p in loo]
        if "B-q4" in profiles and FINAL in profiles:
            pairs.append((FINAL, "B-q4"))
        for profile, reference in pairs:
            if reference not in profiles:
                continue
            comparison = compare(model, suite, profile, reference)
            is_aa = profile == AA_REPEAT and reference == BASELINE
            comparison.verdict = ("A/A noise floor" if is_aa
                                  else classify_effect(comparison.ratio, comparison.ci, noise))
            found.append(comparison)
    return found


def section_comparisons(model: Model, found: list[Comparison]) -> list[str]:
    out = ["## Paired latency comparisons", "",
           "Ratio = geometric mean over (case, run k) pairs of profile/reference case latency; < 1 means the "
           "profile is faster. Only pairs that succeeded in both are used (excluded = pairs where either "
           f"failed). 95% CI: percentile bootstrap, {BOOTSTRAP_RESAMPLES:,} resamples, seed {BOOTSTRAP_SEED}, "
           "resampling whole cases (the runs of one case stay together; they are near-copies). "
           f"The `{AA_REPEAT}` vs `{BASELINE}` row is the A/A noise floor: an effect whose estimate falls "
           "inside that interval is reported as no effect.", ""]
    if not found:
        model.missing("paired comparisons", "no suite has two profiles with complete runs")
    out += table(["suite", "profile", "vs", "pairs", "excluded", "geo-mean ratio", "95% CI", "verdict"],
                 [[c.suite, c.profile, c.reference, c.pairs, c.excluded, fmt_ratio(c.ratio), fmt_ci(c.ci),
                   c.verdict] for c in found])
    return out


def per_case_mean(rows: list[CaseRow], pick: Callable[[CaseRow], float | None]) -> float | None:
    values = [v for v in (pick(r) for r in rows) if v is not None]
    return mean(values)


def section_leave_one_out(model: Model, found: list[Comparison]) -> list[str]:
    loo = leave_one_out(model)
    out = ["## Per-change leave-one-out (public suite)", "",
           "Each row is F with exactly one switch reverted to B's value. Latency ratio = F / F-without "
           "(< 1: the change makes F faster). Deltas are F minus F-without, per case.", ""]
    if not loo:
        model.missing("leave-one-out table", "no profile that differs from F in exactly one switch has "
                      "complete runs (run the F-* profiles)")
        return out + ["No leave-one-out runs yet.", ""]
    gates = {r["profile without"]: r["verdict"] for r in gate_rows(model)}
    f_rows = model.case_rows(PUBLIC_SUITE, FINAL)
    rows = []
    for profile, switch in loo.items():
        other = model.case_rows(PUBLIC_SUITE, profile)
        match = next((c for c in found if c.suite == PUBLIC_SUITE and c.profile == FINAL and c.reference == profile),
                     None)
        f_s, o_s = success(model, PUBLIC_SUITE, FINAL), success(model, PUBLIC_SUITE, profile)

        def delta(pick: Callable[[CaseRow], float | None]) -> str:
            a, b = per_case_mean(f_rows, pick), per_case_mean(other, pick)
            return NA if a is None or b is None else f"{a - b:+,.1f}"

        rows.append([change_label(model, switch), profile,
                     f"{f_s.strict}/{f_s.n}" if f_s else NA, f"{o_s.strict}/{o_s.n}" if o_s else NA,
                     fmt_ratio(match.ratio) if match else NA, fmt_ci(match.ci) if match else NA,
                     match.verdict if match else NA,
                     delta(lambda c: c.metrics.get("llm_calls")), delta(lambda c: c.metrics.get("completion_tokens")),
                     delta(lambda c: c.metrics.get("peak_ram_mb")),
                     delta(lambda c: c.metrics.get("peak_unified_gpu_mb")), gates.get(profile, NA)])
    return out + table(["change (F value)", "without it", "F strict", "without strict", "latency F/without",
                        "95% CI", "latency verdict", "Δ LLM calls", "Δ completion tokens", "Δ peak RAM MiB",
                        "Δ peak unified GPU MiB", "gate"], rows)


# ==================================================================================================
# Steps, time breakdown, throughput, memory, cold start

def section_steps(model: Model) -> list[str]:
    out = ["## Per-LLM-step latency distribution", "",
           "One sample per LLM call (`llm_step` audit entries; failed HTTP calls have no timings). "
           "latency = client wall time of the call; prefill/decode = the server's own timers.", ""]
    for suite in model.suites():
        rows = []
        for profile in model.profiles(suite):
            for role in ("all", "resolution", "operations"):
                steps = [s for s in model.steps if s.suite == suite and s.profile == profile
                         and role in ("all", s.agent)]
                ms = [s.latency_ms for s in steps]
                rows.append([profile, role, len(ms), fmt(nearest_rank(ms, 50)), fmt(nearest_rank(ms, 95)),
                             fmt(max(ms) if ms else None), fmt(mean(s.prompt_ms for s in steps)),
                             fmt(mean(s.predicted_ms for s in steps)),
                             fmt(mean(s.completion_tokens for s in steps), 1)])
        out += [f"### Suite `{suite}`", ""]
        out += table(["profile", "agent", "steps", "p50 ms", "p95 ms", "max ms", "mean prefill ms",
                      "mean decode ms", "mean completion tokens"], rows)
        for profile in [p for p in (BASELINE, FINAL) if p in model.profiles(suite)]:
            groups: dict[str, list[StepRow]] = defaultdict(list)
            for s in model.steps:
                if s.suite == suite and s.profile == profile:
                    groups[f"{s.agent}.{s.tool}"].append(s)
            rows = [[name, len(g), fmt(nearest_rank([s.latency_ms for s in g], 50)),
                     fmt(nearest_rank([s.latency_ms for s in g], 95)), fmt(mean(s.completion_tokens for s in g), 1),
                     sum(1 for s in g if s.protocol_error)] for name, g in sorted(groups.items())]
            out += [f"By tool, profile `{profile}`, suite `{suite}`", ""]
            out += table(["agent.tool", "steps", "p50 ms", "p95 ms", "mean completion tokens", "protocol errors"], rows)
    return out


def _sum_metric(rows: list[CaseRow], key: str) -> float:
    return float(sum(float(c.metrics.get(key) or 0) for c in rows))


def rate(tokens: float, ms: float) -> float | None:
    return tokens / ms * 1000 if ms > 0 else None


def prom_delta(run: Run) -> dict[str, float] | None:
    """Server-side counters over the run: metrics_after - metrics_before (None if either is missing)."""
    try:
        before = parse_prometheus((run.directory / "metrics_before.prom").read_text())
        after = parse_prometheus((run.directory / "metrics_after.prom").read_text())
    except OSError:
        return None
    return {k: after[k] - before.get(k, 0.0) for k in after}


def section_breakdown(model: Model) -> list[str]:
    out = ["## Where the time goes, throughput, cache, calls", "",
           "Means per case-run (all attempts). http/template = LLM call wall time minus the server's "
           "prefill and decode timers; orchestration = case time minus LLM wall time minus gateway (tool) "
           "time. Throughput is token-weighted: sum of tokens / sum of time (not a mean of per-call rates). "
           "Cache hit = cached prompt tokens / all prompt tokens.", ""]
    for suite in model.suites():
        rows, rates = [], []
        for profile in model.profiles(suite):
            cases = model.case_rows(suite, profile)
            n = len(cases)
            prompt, cached = _sum_metric(cases, "prompt_tokens"), _sum_metric(cases, "cached_tokens")
            prefill, decode = _sum_metric(cases, "prefill_ms"), _sum_metric(cases, "decode_ms")
            tool, elapsed = _sum_metric(cases, "tool_ms"), sum(c.elapsed_ms for c in cases)
            llm = sum(c.llm_latency_ms for c in cases)
            calls = [float(c.metrics.get("llm_calls") or 0) for c in cases]
            rows.append([profile, n, fmt(elapsed / n), fmt(statistics.fmean(calls), 2), fmt(median(calls)),
                         fmt(nearest_rank(calls, 95)), fmt((prompt - cached) / n), fmt(cached / n),
                         fmt(_sum_metric(cases, "completion_tokens") / n), fmt(prefill / n), fmt(decode / n),
                         fmt((llm - prefill - decode) / n), fmt(tool / n, 1), fmt((elapsed - llm - tool) / n),
                         fmt_pct(decode / elapsed if elapsed else None)])
            server = server_rates(model, suite, profile)
            rates.append([profile, fmt(rate(_sum_metric(cases, "completion_tokens"), decode), 1),
                          fmt(rate(prompt - cached, prefill), 1), fmt_pct(cached / prompt if prompt else None),
                          *server])
        out += [f"### Suite `{suite}`", ""]
        out += table(["profile", "case-runs", "case ms", "LLM calls/case", "calls p50", "calls p95",
                      "uncached prompt tok", "cached prompt tok", "completion tok", "prefill ms", "decode ms",
                      "http/template ms", "tool ms", "orchestration ms", "decode share"], rows)
        out += table(["profile", "decode tok/s", "prefill tok/s (uncached)", "cache hit",
                      "server decode tok/s (/metrics)", "server prefill tok/s (/metrics)", "/metrics runs"], rates)
    return out


def server_rates(model: Model, suite: str, profile: str) -> list[str]:
    """Cross-check from the server's own counters (the 'before' scrape races the adapter's 1-token
    warm-up request, so the delta may include it). A server restart resets the counters, so such
    runs are left out rather than subtracted across two processes."""
    deltas = []
    for run in model.run_list(suite, profile):
        delta = prom_delta(run)
        if (run.manifest.get("adapter") or {}).get("server_restarts"):
            model.missing(f"/metrics delta ({run.label})", "llama-server restarted during the run; its counters "
                          "restart from zero, so before/after do not describe one process")
        elif delta is None:
            model.missing(f"/metrics delta ({run.label})",
                          (run.manifest.get("adapter") or {}).get("metrics_after", "metrics file missing"))
        else:
            deltas.append(delta)
    if not deltas:
        return [NA, NA, "0"]

    def total(key: str) -> float:
        return sum(d.get(key, 0.0) for d in deltas)

    decode_s, prefill_s = total("llamacpp:tokens_predicted_seconds_total"), total("llamacpp:prompt_seconds_total")
    return [fmt(total("llamacpp:tokens_predicted_total") / decode_s if decode_s else None, 1),
            fmt(total("llamacpp:prompt_tokens_total") / prefill_s if prefill_s else None, 1), str(len(deltas))]


def section_memory(model: Model) -> list[str]:
    out = ["## Memory (unified memory: never add RAM and GPU figures)", "",
           "peak RAM = sum of kernel phys_footprint high-water marks (adapter + llama-server; interval reset "
           "at case start). peak unified GPU = IOAccelerator in-use system memory minus the idle baseline "
           "taken before the server started (system-wide; overlaps RAM). RSS includes mmapped weights and "
           "overcounts; shown for comparison only.", ""]
    for suite in model.suites():
        rows = []
        for profile in model.profiles(suite):
            cases = model.case_rows(suite, profile)
            ram = [float(c.metrics["peak_ram_mb"]) for c in cases if c.metrics.get("peak_ram_mb") is not None]
            gpu = [float(c.metrics["peak_unified_gpu_mb"]) for c in cases
                   if c.metrics.get("peak_unified_gpu_mb") is not None]
            server = [float(p["peak_phys_footprint_mb"]) for c in cases
                      for name, p in (c.metrics.get("per_process") or {}).items()
                      if name == "llama-server" and p.get("peak_phys_footprint_mb") is not None]
            rss = [float(c.metrics["rss_mb"]) for c in cases if c.metrics.get("rss_mb") is not None]
            if len(gpu) < len(cases):
                model.missing(f"peak_vram_mb ({suite}/{profile})", f"null in {len(cases) - len(gpu)} of {len(cases)} "
                              "case-runs (see metrics.memory_notes in results.json)")
            rows.append([profile, fmt(max(ram, default=None)), fmt(median(ram)), fmt(max(server, default=None)),
                         fmt(max(gpu, default=None)), fmt(median(gpu)), fmt(median(rss))])
        out += [f"### Suite `{suite}`, MiB", ""]
        out += table(["profile", "peak RAM max", "peak RAM median", "llama-server footprint max",
                      "peak unified GPU max", "peak unified GPU median", "RSS median"], rows)
    return out


def section_cold_start(model: Model) -> list[str]:
    out = ["## Cold start (separate from warm work)", "",
           "spawn -> ready = llama-server process start to first /health 200; first request = a 1-token "
           "warm-up call (Metal pipeline compile). Page cache = share of the model file already in the OS "
           "page cache at spawn (cold = read from SSD). The first case after a start also pays a cold prompt "
           "cache; warm cases are all the others.", ""]
    rows = []
    for run in sorted(model.runs, key=lambda r: (r.suite, profile_order(r.profile), r.run)):
        startup = run.document.get("startup") or {}
        cases = [c for c in model.cases if c.suite == run.suite and c.profile == run.profile and c.run == run.run]
        first = [c.elapsed_ms for c in cases if c.first_case]
        warm = [c.elapsed_ms for c in cases if not c.first_case]
        rows.append([run.label, fmt(startup.get("server_spawn_to_ready_ms")), fmt(startup.get("first_request_ms")),
                     f"{startup.get('model_page_cache', NA)} ({fmt(startup.get('model_page_cache_resident_pct'), 1)}%)",
                     fmt(first[0] if first else None), fmt(median(warm))])
    labels = {str((r.document.get("startup") or {}).get("model_page_cache")) for r in model.runs}
    if model.runs and "cold" not in labels:
        model.missing("true SSD cold start", "no run started with the model file out of the page cache; "
                      "desk.llm_server.evict_page_cache() can force one (not used by the bench)")
    return out + table(["run", "spawn -> ready ms", "first request ms", "model page cache",
                        "first case ms", "warm case median ms"], rows)


# ==================================================================================================
# Failures, determinism, inventory

def section_failures(model: Model) -> list[str]:
    out = ["## Failures and timeouts", "",
           "Case-level causes (a case can have several). 'LLM request timeout' is a real 20 s step timeout; "
           "the contract's injected `timeout_after_commit` is listed separately below because it is expected. "
           "'step/call cap reached' is inferred from the audit (a turn used every Resolution step or the "
           "whole call budget).", ""]
    for suite in model.suites():
        profiles = model.profiles(suite)
        causes = [*CANONICAL_CAUSES, *sorted({cause for c in model.cases if c.suite == suite
                                              for cause in c.causes} - set(CANONICAL_CAUSES))]
        failed = {p: sum(not c.strict for c in model.case_rows(suite, p)) for p in profiles}
        rows = [["strict failures (cases)"] + [f"{failed[p]}/{len(model.case_rows(suite, p))}" for p in profiles]]
        rows += [[cause] + [sum(cause in c.causes for c in model.case_rows(suite, p)) for p in profiles]
                 for cause in causes]
        for title, pick in (("protocol errors", lambda c: c.protocol_codes), ("code fallbacks", lambda c: c.fallbacks),
                            ("guard overrides", lambda c: c.guards)):
            totals = {p: sum((pick(c) for c in model.case_rows(suite, p)), Counter()) for p in profiles}
            rows.append([f"{title} (count)"] + [sum(totals[p].values()) for p in profiles])
            names = sorted({k for t in totals.values() for k in t})
            rows += [[f"  {title}: {name}"] + [totals[p][name] for p in profiles] for name in names]
        out += [f"### Suite `{suite}`", ""] + table(["cause", *profiles], rows)
        injected = {p: sum((c.injected for c in model.case_rows(suite, p)), Counter()) for p in profiles}
        names = sorted({k for t in injected.values() for k in t})
        if names:
            out += [f"Injected faults (expected, not failures) — suite `{suite}`", ""]
            out += table(["fault", *profiles], [[name] + [injected[p][name] for p in profiles] for name in names])
        failing = [c for c in model.cases if c.suite == suite and not c.strict]
        if failing:
            out += [f"Failing case-runs — suite `{suite}` (first 30)", ""]
            out += table(["profile", "run", "case", "causes"],
                         [[c.profile, c.run, c.case_id, "; ".join(c.causes)] for c in failing[:30]])
    return out


def section_traces(model: Model) -> list[str]:
    out = ["## Decision-trace identity across runs", "",
           "Decision trace = per-turn decisions, proposal bindings, transaction ids and the ordered audit of "
           "tool choices with their non-text arguments and error codes (no timings, no wording). Text trace = "
           "every response and drafted reply. Identical = the same hash in every run of that profile.", ""]
    rows = []
    for suite in model.suites():
        for profile in model.profiles(suite):
            cases = model.case_rows(suite, profile)
            by_case: dict[str, list[CaseRow]] = defaultdict(list)
            for c in cases:
                by_case[c.case_id].append(c)
            multi = {cid: rs for cid, rs in by_case.items() if len(rs) >= 2}
            if not multi:
                model.missing(f"decision-trace identity ({suite}/{profile})", "needs at least 2 runs of the profile")
                rows.append([suite, profile, len(model.run_list(suite, profile)), NA, NA, ""])
                continue
            same_decisions = [cid for cid, rs in multi.items() if len({r.trace_hash for r in rs}) == 1]
            same_text = [cid for cid, rs in multi.items() if len({r.text_hash for r in rs}) == 1]
            differ = sorted(set(multi) - set(same_decisions))
            rows.append([suite, profile, len(model.run_list(suite, profile)), f"{len(same_decisions)}/{len(multi)}",
                         f"{len(same_text)}/{len(multi)}", ", ".join(differ[:10])])
    return out + table(["suite", "profile", "runs", "identical decision trace", "identical text",
                        "cases whose decision trace differs (first 10)"], rows)


def section_runs(model: Model) -> list[str]:
    out = ["## Every attempted run", "",
           "All run directories, including earlier attempts kept as run-<k>.attempt-<n>. Only complete runs "
           "are measured. 'dirty' = bench preflight refused the machine and --allow-dirty overrode it: "
           "its timings are not clean.", ""]
    rows = []
    for attempt in sorted(model.attempts, key=lambda a: (a.suite, profile_order(a.profile), a.run, str(a.directory))):
        m = attempt.manifest
        adapter, checks, scored = m.get("adapter") or {}, m.get("validate") or {}, m.get("score") or {}
        pre = m.get("preflight") or {}
        structure = {True: "ok", False: "FAILED", None: NA}[checks.get("structure_ok")]
        baseline = {True: "ok", False: "FAILED", None: NA}[checks.get("baseline_ok")]
        rows.append([attempt.directory.name if attempt.renamed else f"run-{attempt.run}", attempt.suite,
                     attempt.profile, m.get("status", "no run.json"), "passed" if pre.get("passed") else
                     ("dirty" if pre.get("overridden") else NA), adapter.get("exit_code", NA),
                     fmt(adapter.get("wall_s"), 1), structure, baseline,
                     f"{scored.get('strict_success')}/{scored.get('cases')}" if scored else NA,
                     "yes" if str(adapter.get("metrics_after", "")).startswith("captured") else "no",
                     m.get("started_at", NA)])
    return out + table(["dir", "suite", "profile", "status", "preflight", "adapter exit", "wall s", "structure",
                        "--baseline", "strict (run time)", "/metrics after", "started (UTC)"], rows)


def section_unavailable(model: Model) -> list[str]:
    model.missing("per-process GPU memory", "IOAccelerator statistics are system-wide; per-process attribution "
                  "needs root (powermetrics/Instruments), which the bench does not use")
    model.missing("energy / power draw", "needs `sudo powermetrics`; the bench runs without root")
    model.missing("discrete VRAM", "Apple Silicon has unified memory; peak_vram_mb reports the GPU's unified "
                  "allocation (idle baseline subtracted) and overlaps peak RAM")
    out = ["## Unavailable metrics (and why)", ""]
    return out + table(["metric", "reason"], [[metric, reason] for metric, reason in model.unavailable])


# ==================================================================================================
# CSV and the document

def write_csvs(model: Model, found: list[Comparison], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def write(name: str, headers: list[str], rows: Iterable[list[Any]]) -> None:
        path = out_dir / name
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(rows)
        written.append(path)

    write("runs.csv", ["suite", "profile", "run", "position", "status", "clean", "adapter_exit", "wall_s",
                       "spawn_to_ready_ms", "first_request_ms", "model_page_cache", "strict", "cases", "dir"],
          [[r.suite, r.profile, r.run, r.manifest.get("position"), r.manifest.get("status"), r.clean,
            (r.manifest.get("adapter") or {}).get("exit_code"), (r.manifest.get("adapter") or {}).get("wall_s"),
            (r.document.get("startup") or {}).get("server_spawn_to_ready_ms"),
            (r.document.get("startup") or {}).get("first_request_ms"),
            (r.document.get("startup") or {}).get("model_page_cache"), sum(s.strict for s in r.scores.values()),
            len(r.scores), str(r.directory)] for r in model.runs])
    metric_keys = ["llm_calls", "prompt_tokens", "cached_tokens", "completion_tokens", "prefill_ms", "decode_ms",
                   "tool_ms", "peak_ram_mb", "peak_unified_gpu_mb", "rss_mb", "protocol_errors", "guard_overrides",
                   "fallbacks"]
    write("cases.csv", ["suite", "profile", "run", "case_id", "scenario", "strict", "lenient", "violations",
                        "elapsed_ms", "llm_latency_ms", "turn_latency_ms", "first_case", "status", "clean",
                        "causes", "trace_hash", "text_hash", *metric_keys],
          [[c.suite, c.profile, c.run, c.case_id, c.scenario, c.strict, c.lenient, ";".join(c.violations),
            c.elapsed_ms, round(c.llm_latency_ms, 1), ";".join(str(t) for t in c.turn_latency_ms), c.first_case,
            c.status, c.clean, ";".join(c.causes), c.trace_hash, c.text_hash,
            *[c.metrics.get(k) for k in metric_keys]] for c in model.cases])
    write("steps.csv", ["suite", "profile", "run", "case_id", "turn", "agent", "tool", "latency_ms", "prompt_ms",
                        "predicted_ms", "prompt_tokens", "cached_tokens", "completion_tokens", "protocol_error"],
          [[s.suite, s.profile, s.run, s.case_id, s.turn, s.agent, s.tool, s.latency_ms, s.prompt_ms,
            s.predicted_ms, s.prompt_tokens, s.cached_tokens, s.completion_tokens, s.protocol_error or ""]
           for s in model.steps])
    summary_rows = []
    for suite in model.suites():
        for profile in model.profiles(suite):
            cases = model.case_rows(suite, profile)
            s = success(model, suite, profile)
            assert s is not None
            low, high = wilson_interval(s.strict, s.n)
            ok = [c.elapsed_ms for c in successful(cases)]
            summary_rows.append([suite, profile, s.runs, s.strict, s.n, round(low, 4), round(high, 4), s.violations,
                                 nearest_rank(ok, 50), nearest_rank(ok, 95), len(ok)])
    write("summary.csv", ["suite", "profile", "runs", "strict", "case_runs", "wilson_low", "wilson_high",
                          "invariant_violations", "case_p50_ms", "case_p95_ms", "latency_sample"], summary_rows)
    write("comparisons.csv", ["suite", "profile", "reference", "pairs", "excluded", "geomean_ratio", "ci_low",
                              "ci_high", "verdict"],
          [[c.suite, c.profile, c.reference, c.pairs, c.excluded, c.ratio, c.ci[0] if c.ci else None,
            c.ci[1] if c.ci else None, c.verdict] for c in found])
    write("gate.csv", ["change", "profile_without", *GATE_CRITERIA, "verdict"],
          [[r["change"], r["profile without"], *[r[k] for k in GATE_CRITERIA], r["verdict"]] for r in gate_rows(model)])
    return written


def render(model: Model, results_root: Path, csv_paths: list[Path], found: list[Comparison]) -> str:
    dirty = [r.label for r in model.runs if not r.clean]
    head = ["# Benchmark report", "",
            f"Generated by `.venv/bin/python -m bench.report --results {_rel(results_root)}` from "
            f"{len(model.runs)} complete run(s) ({len(model.attempts)} run directories in total). Every number "
            "below is recomputed from the raw files in those directories; results.json is re-scored with "
            "`evals.score` (the independent oracle).", "",
            "Method: strict success with Wilson 95% intervals; nearest-rank percentiles; latency comparisons "
            f"paired on (case, run k) with a {BOOTSTRAP_RESAMPLES:,}-resample case-level bootstrap "
            f"(seed {BOOTSTRAP_SEED}); "
            "effects inside the A/A interval are no effect. Runs follow bench/run.py's interleaved schedule "
            "(schedule.json, seed 42).", ""]
    if dirty:
        head += [f"**Timings not clean:** {len(dirty)} run(s) ran after preflight refused the machine "
                 f"(--allow-dirty): {', '.join(dirty)}. Their latency, throughput and memory numbers are evidence "
                 "of the pipeline, not benchmark results.", ""]
    if model.warnings:
        head += ["Warnings:", ""] + [f"- {w}" for w in model.warnings] + [""]
    body: list[str] = []
    if not model.runs:
        model.missing("everything", f"no complete run under {_rel(results_root)}")
        if model.floors:
            body += section_quality(model)
    else:
        body += section_setup(model, results_root)
        body += section_quality(model) + section_gate(model) + section_latency(model)
        body += section_comparisons(model, found) + section_leave_one_out(model, found)
        body += section_steps(model) + section_breakdown(model) + section_memory(model)
        body += section_cold_start(model) + section_failures(model) + section_traces(model)
    body += section_runs(model) + section_unavailable(model)
    files = ["## Files", "", *[f"- `{_rel(p)}`" for p in csv_paths],
             "- raw per run: `<suite>/<profile>/run-<k>/{results.json, adapter.log, server.log, metrics_before.prom, "
             "metrics_after.prom, validate.txt, score.txt, score.json, run_conditions.json, argv.txt, "
             "preflight.txt, run.json}`; floors: `<suite>/floors/<mode>/{results.json, score.json, floor.json, "
             "stub.log}`; fresh model hashes: `<suite>/model_sha256*.json`", ""]
    return "\n".join(head + body + files)


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def write_summary(model: Model, out_path: Path) -> Path:
    """Public-suite case latency of B and F as JSON, recomputed from the raw runs like section_latency."""
    def side(profile: str) -> dict[str, Any]:
        runs = model.run_list(PUBLIC_SUITE, profile)
        cases = model.case_rows(PUBLIC_SUITE, profile)
        ok_ms = [c.elapsed_ms for c in successful(cases)]
        return {"profile": profile, "runs": len(runs), "successful_cases": f"{len(ok_ms)}/{len(cases)}",
                "p50_ms": nearest_rank(ok_ms, 50), "p95_ms": nearest_rank(ok_ms, 95),
                "results": [_rel(r.directory / "results.json") for r in runs]}
    summary = {"suite": PUBLIC_SUITE,
               "method": "metrics.elapsed_ms of strict-success cases pooled over all complete runs; "
                         "nearest-rank percentiles (p95 = max below 20 samples)",
               "baseline": side(BASELINE), "final": side(FINAL)}
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    return out_path


def build_report(results_root: Path, out_path: Path, root: Path = ROOT) -> Model:
    model = build_model(results_root, root)
    found = comparisons(model) if model.runs else []
    csv_paths = write_csvs(model, found, out_path.parent / "csv")
    summary_path = write_summary(model, out_path.parent / "summary.json")
    out_path.write_text(render(model, results_root, [*csv_paths, summary_path], found))
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write REPORT.md + CSVs from bench run directories (SPEC §9).")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR, help="results root written by bench.run")
    parser.add_argument("--out", type=Path, help="report path (default: <results>/REPORT.md)")
    args = parser.parse_args(argv)
    if not args.results.is_dir():
        print(f"[report] no results directory at {args.results}", file=sys.stderr)
        return 2
    out = args.out or args.results / "REPORT.md"
    model = build_report(args.results, out)
    print(f"[report] {len(model.runs)} complete run(s), {len(model.attempts)} attempt dir(s) -> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    raise SystemExit(main())
