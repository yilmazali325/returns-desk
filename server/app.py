"""HTTP server for the Returns Desk UI (SPEC §10): sessions, live turn streaming, failure lab, telemetry.

The server owns no business rules. Every UI session is a `desk.session.CaseSession`, the same
object the adapter builds (CONTRACT: "The adapter must invoke the same agents, local models and
business tools as the UI"), over a SQLite file under var/ui/sessions/.

Live streaming: a turn runs in a worker thread. The session's AuditLog calls our listener on every
entry it records; the listener pushes the entry into a thread-safe queue, and the HTTP response
drains that queue as Server-Sent Events (`audit` events while the turn runs, then one `turn`
event). Nothing is batched at the end of the turn, so the browser can animate each step as it
happens. One turn per session at a time: a second request gets 409 (this is also what makes a
double-clicked Confirm harmless at the HTTP level; the gateway's idempotency covers the rest).

Which LLM: `DESK_LLM=fake-policy` selects the deterministic test double (E2E tests, machines
without the model); anything else uses the real local llama-server at DESK_LLAMA_URL.

Approvals bind to what the browser displayed: a confirmation must carry the proposal id, the full
binding hash shown on its card and the session generation; a mismatch is a 409 and nothing runs.

Local-only hardening (a demo server on 127.0.0.1, but any web page the operator visits can send it
requests): the Host header must name this machine (DNS rebinding), a state-changing POST must come
from this origin when the browser says where it came from (cross-site requests), and request bodies
are capped. Malformed input is a 422 with a readable message, never a 500.

Run: `python -m server.app [--port 8000] [--llama-url URL] [--start-llama] [--fake] [--allow-engine-mismatch]`.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import queue
import secrets
import signal
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlparse, urlsplit

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StrictBool, StrictInt, ValidationError
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from desk import policy, telemetry
from desk.adapter_main import LLM_HTTP_RETRIES, LLM_STEP_TIMEOUT_S
from desk.agents.prompts import RESOLUTION_SYSTEM, operations_system
from desk.agents.toolspecs import REPLY, REPORT, RESOLUTION_TOOLS, operations_tools
from desk.domain import CODE_CLASS, WINDOW_DAYS
from desk.llm import CircuitBreaker, LLMClient, LLMError
from desk.llm_server import LlamaServer, LlamaServerError, pid_listening_on, server_healthy
from desk.profiles import ENGINE, ENGINE_VERSION, MODEL_NAME, SEED, SLOTS, Profile, get_profile
from desk.render import REASON_TEXT, money
from desk.session import CaseSession
from desk.testing.fake_policy_llm import ENGINE_LABEL, FakePolicyLLM
from server.capacity import LoadRequest, LoadRuns, TurnLine
from server.errors import SessionError

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_CASES = ROOT / "candidate_assets" / "public_cases.json"
DEMO_CASES = ROOT / "server" / "demo_cases.json"  # natural-language chats for the demo, listed after the public ones
DEFAULT_VAR_DIR = ROOT / "var" / "ui"
DEFAULT_LLAMA_URL = "http://127.0.0.1:8081"
DEFAULT_UI_PORT = 8000
FAKE_POLICY = "fake-policy"
REAL_LLM = "llama"
HEALTH_CACHE_S = 1.0  # the header polls; one /health probe per second is plenty
# Under load (real model): at most TURNS_PER_SLOT turns per model slot run at once; the next ones wait
# in a first-come, first-served line (at most MAX_WAITING; one more is refused with 503 + Retry-After).
# A model call waits at most LLM_QUEUE_TIMEOUT_S for a free slot; then that turn ends in a decision error
# and nothing changes.
TURNS_PER_SLOT = 2  # measured: 8 per slot left 28 calls waiting on 4 slots and 2 turns hit the queue cap
MAX_WAITING = 1000
QUEUED_EVERY_S = 1.0  # a waiting turn's stream repeats its place in line this often
RETRY_AFTER_S = 2
LLM_QUEUE_TIMEOUT_S = 60.0
MEMORY_CACHE_S = 1.0  # ioreg and lsof take tens of milliseconds each
SSE_KEEPALIVE_S = 10.0
MAX_MESSAGE_CHARS = 2000  # sanitize() caps prompts at 1,000; this only refuses absurd payloads
HASH_PREFIX = 12
MIB = 1024 * 1024
MAX_BODY_BYTES = 1 * MIB  # a scenario state is a few KB; nothing legitimate comes close
LOCAL_HOSTNAMES = ("127.0.0.1", "localhost", "::1")
STATE_CHANGING_METHODS = ("POST", "PUT", "PATCH", "DELETE")
STALE_PROPOSAL = "this proposal changed since this page showed it; reload the page and check it again"
MISSING_BINDING = ("a confirmation must carry the confirm_binding_hash and generation of the proposal card it "
                   "approves (GET /api/sessions/{id}/state: cards[...].binding_hash, generation)")


# ================================================================================================
# Configuration


@dataclass(frozen=True)
class ServerConfig:
    """Everything the server needs to know, read from the environment by default."""

    llm_mode: str = REAL_LLM  # REAL_LLM or FAKE_POLICY
    llama_url: str = DEFAULT_LLAMA_URL
    profile: str = "F"
    cases_path: Path = DEFAULT_CASES
    demo_cases_path: Path | None = DEMO_CASES
    var_dir: Path = DEFAULT_VAR_DIR
    fake_step_delay_s: float = 0.0
    allow_engine_mismatch: bool = False  # --start-llama with another llama.cpp build: warn instead of refusing
    allowed_hostnames: tuple[str, ...] = LOCAL_HOSTNAMES
    port: int | None = None  # the UI port, when known: load runs started from the page call it
    slots: int = SLOTS  # parallel requests the model server serves (scripts/run.sh --slots)
    max_turns: int | None = None  # turns running at once; None: TURNS_PER_SLOT x slots (no cap on the fake)
    max_waiting: int = MAX_WAITING  # turns waiting in line for one of those places

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> ServerConfig:
        return cls(
            llm_mode=FAKE_POLICY if environ.get("DESK_LLM", "").strip() == FAKE_POLICY else REAL_LLM,
            llama_url=environ.get("DESK_LLAMA_URL", DEFAULT_LLAMA_URL),
            profile=environ.get("DESK_PROFILE", "F"),
            cases_path=Path(environ.get("DESK_CASES", str(DEFAULT_CASES))),
            var_dir=Path(environ.get("DESK_VAR_DIR", str(DEFAULT_VAR_DIR))),
            fake_step_delay_s=float(environ.get("DESK_FAKE_STEP_MS", "0")) / 1000,
            allow_engine_mismatch=environ.get("DESK_ALLOW_ENGINE_MISMATCH", "").strip() == "1",
        )

    @property
    def fake(self) -> bool:
        return self.llm_mode == FAKE_POLICY

    @property
    def llama_port(self) -> int:
        return urlparse(self.llama_url).port or 80


def build_llm(config: ServerConfig, profile: Profile) -> Any:
    """One LLM client shared by every session (llama-server queues concurrent requests)."""
    if config.fake:
        return FakePolicyLLM(step_delay_s=config.fake_step_delay_s)
    return LLMClient(config.llama_url, profile.protocol, seed=SEED, timeout_s=LLM_STEP_TIMEOUT_S,
                     retries=LLM_HTTP_RETRIES, max_in_flight=config.slots, queue_timeout_s=LLM_QUEUE_TIMEOUT_S,
                     breaker=CircuitBreaker())


def turn_limit(config: ServerConfig) -> int | None:
    if config.max_turns is not None:
        return config.max_turns
    return None if config.fake else TURNS_PER_SLOT * config.slots


def model_info(config: ServerConfig, profile: Profile) -> dict[str, Any]:
    """What the header's model chip shows."""
    if config.fake:
        return {"model": "fake-policy", "quantization": None, "engine": "deterministic test double, no model loaded",
                "engine_short": "test double, no model", "engine_detail": ENGINE_LABEL, "profile": profile.name,
                "protocol": FAKE_POLICY, "local": True}
    return {"model": MODEL_NAME, "model_file": profile.model_file, "quantization": profile.quantization,
            "engine": f"{ENGINE} {ENGINE_VERSION}", "engine_short": f"llama.cpp {ENGINE_VERSION.split()[0]}",
            "profile": profile.name, "protocol": profile.protocol, "local": True}


def start_commands(config: ServerConfig, profile: Profile) -> dict[str, str]:
    """The exact commands the model-down banner offers (run from the project folder)."""
    argv = profile.engine_argv("llama-server", f"models/{profile.model_file}", config.llama_port)
    return {"run_script": f"scripts/run.sh --llama-port {config.llama_port}", "llama_server": " ".join(argv)}


# ================================================================================================
# Scenarios (the case file, read at startup; nothing about any case is written in this code)


def load_cases(path: Path) -> list[dict[str, Any]]:
    return list(json.loads(path.read_text())["cases"])


def scenario_title(case_id: str) -> str:
    """'name_07_late_parcel' -> '07 · late parcel' (a readable label from the id alone)."""
    parts = case_id.split("_")
    if len(parts) > 2 and parts[0].isalpha() and parts[1].isdigit():
        return f"{parts[1]} · {' '.join(parts[2:])}"
    return case_id.replace("_", " ")


def fault_text(fault: Mapping[str, Any]) -> str:
    if fault.get("type") == "stock_change":
        return f"stock of {fault.get('sku')} becomes {fault.get('new_stock')} before turn {fault.get('before_turn')}"
    if fault.get("type") == "timeout_after_commit":
        return "the first commit times out after it persisted"
    return f"unknown fault {fault.get('type')!r}"


def customers_in(state: Mapping[str, Any]) -> list[str]:
    return sorted({str(o.get("customer_id")) for o in state.get("orders", []) if o.get("customer_id")})


def scenario_view(case: Mapping[str, Any]) -> dict[str, Any]:
    state = case["initial_state"]
    return {
        "case_id": case["case_id"],
        "customer_id": case["customer_id"],
        "request_id": case["request_id"],
        "title": scenario_title(str(case["case_id"])),
        "turns": copy.deepcopy(case.get("turns", [])),
        "faults": copy.deepcopy(case.get("faults", [])),
        "fault_texts": [fault_text(f) for f in case.get("faults", []) if isinstance(f, dict)],
        "customers": sorted(set(customers_in(state)) | {str(case["customer_id"])}),
        "skus": [p.get("sku") for p in state.get("products", [])],
        "initial_state": copy.deepcopy(state),
    }


# ================================================================================================
# Sessions


@dataclass(frozen=True)
class SessionSpec:
    """What a session was created from; reset rebuilds the CaseSession from exactly this."""

    initial_state: dict[str, Any]
    customer_id: str
    request_id: str
    faults: list[Any] = field(default_factory=list)
    case_id: str | None = None
    turn_count: int | None = None


@dataclass
class UISession:
    session_id: str
    spec: SessionSpec
    case: CaseSession
    store_path: Path
    generation: int = 1
    lock: threading.Lock = field(default_factory=threading.Lock)
    turn_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def busy(self) -> bool:
        return self.lock.locked()


class SessionRegistry:
    """All UI sessions of this server process.

    Session ids are random (S-<12 hex>): after a server restart a stale tab must get a 404, never a
    new session that happens to reuse its id. Request ids of blank sessions are counters
    (REQ-UI-0001); each session has its own store, so they cannot collide."""

    def __init__(self, llm: Any, profile: Profile, sessions_dir: Path, board: TelemetryBoard) -> None:
        self.llm = llm
        self.profile = profile
        self.sessions_dir = sessions_dir
        self.board = board
        self._sessions: dict[str, UISession] = {}
        self._request_counter = 0
        self._lock = threading.Lock()
        sessions_dir.mkdir(parents=True, exist_ok=True)
        for stale in sessions_dir.glob("S-*.sqlite"):  # scratch from an earlier server run
            stale.unlink()

    def next_request_id(self) -> str:
        with self._lock:
            self._request_counter += 1
            return f"REQ-UI-{self._request_counter:04d}"

    def create(self, spec: SessionSpec) -> UISession:
        session_id = f"S-{secrets.token_hex(6)}"
        path = self.sessions_dir / f"{session_id}-g1.sqlite"
        session = UISession(session_id, spec, self._case_session(spec, path), path)
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> UISession:
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionError(404, f"no session {session_id!r}")
        return session

    def reset(self, session: UISession, fresh_request_id: bool) -> None:
        """Back to the initial state: a new store, audit and desk (the caller holds session.lock)."""
        spec = session.spec
        if fresh_request_id:
            spec = replace(spec, request_id=f"{_base_request_id(spec.request_id)}-R{session.generation + 1}")
        self._rebuild(session, spec)

    def new_request(self, session: UISession) -> None:
        """The same customer's next request: today's data carried over, a fresh request id, no script and no
        scripted faults. A later reset returns to this point (the caller holds session.lock)."""
        spec = replace(session.spec, initial_state=session.case.tools.final_state(),
                       request_id=self.next_request_id(), faults=[], case_id=None, turn_count=None)
        self._rebuild(session, spec)

    def _rebuild(self, session: UISession, spec: SessionSpec) -> None:
        old_path = session.store_path
        session.case.store.close()
        session.generation += 1
        session.store_path = self.sessions_dir / f"{session.session_id}-g{session.generation}.sqlite"
        session.spec = spec
        session.case = self._case_session(spec, session.store_path)
        session.turn_log = []
        old_path.unlink(missing_ok=True)

    def _case_session(self, spec: SessionSpec, path: Path) -> CaseSession:
        path.unlink(missing_ok=True)
        case = CaseSession(initial_state=copy.deepcopy(spec.initial_state), customer_id=spec.customer_id,
                           request_id=spec.request_id, faults=copy.deepcopy(spec.faults), llm=self.llm,
                           profile=self.profile, turn_count=spec.turn_count, store_path=str(path))
        case.audit.subscribe(self.board.observe)  # live header telemetry, mid-turn included
        return case


def _base_request_id(request_id: str) -> str:
    head, sep, tail = request_id.rpartition("-R")
    return head if sep and tail.isdigit() else request_id


# --- the contract's state schema, checked for blank sessions (scenario states come from the case file)


def _iso_date(value: str) -> str:
    datetime.date.fromisoformat(value)  # raises ValueError -> a 422 naming the field
    return value


IsoDate = Annotated[str, AfterValidator(_iso_date)]


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")  # extra fields are kept and ignored, like the adapter does


class ProductShape(_Open):
    sku: str
    product_id: str
    size: str
    price_cents: StrictInt
    stock: StrictInt


class LineShape(_Open):
    line_id: str
    sku: str
    paid_cents: StrictInt
    unopened: StrictBool
    final_sale: StrictBool
    returned: StrictBool


class OrderShape(_Open):
    order_id: str
    customer_id: str
    status: str
    delivered_on: IsoDate | None = None
    lines: list[LineShape]


class StateShape(_Open):
    as_of: IsoDate
    currency: str
    products: list[ProductShape]
    orders: list[OrderShape]


def check_state_shape(state: Mapping[str, Any]) -> None:
    """Refuse a blank session's initial_state that does not follow the contract schema (422, not a
    crash in the middle of a turn: money and stock must be integers, dates ISO dates)."""
    try:
        StateShape.model_validate(state)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5])
        raise SessionError(422, f"initial_state does not follow the contract schema: {problems}") from exc


def spec_from_request(body: CreateSessionRequest, cases: Mapping[str, dict[str, Any]],
                      registry: SessionRegistry) -> SessionSpec:
    """A scenario (its state, customer, request id and faults) or a blank state for a chosen customer."""
    if body.case_id is not None and body.initial_state is not None:
        raise SessionError(422, "send either case_id or initial_state, not both")
    if body.case_id is not None:
        case = cases.get(body.case_id)
        if case is None:
            raise SessionError(404, f"unknown scenario {body.case_id!r}")
        return SessionSpec(initial_state=copy.deepcopy(case["initial_state"]),
                           customer_id=body.customer_id or case["customer_id"],
                           request_id=body.request_id or case["request_id"],
                           faults=copy.deepcopy(case.get("faults", [])), case_id=body.case_id,
                           turn_count=len(case.get("turns", [])))
    if body.initial_state is None:
        raise SessionError(422, "send case_id or initial_state")
    if not body.customer_id:
        raise SessionError(422, "a blank session needs customer_id")
    check_state_shape(body.initial_state)
    return SessionSpec(initial_state=copy.deepcopy(body.initial_state), customer_id=body.customer_id,
                       request_id=body.request_id or registry.next_request_id())


# ================================================================================================
# What the browser reads: proposal cards and the session snapshot


def proposal_card(proposal: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    """Everything the proposal card shows, from the stored proposal and the CURRENT state (live stock).

    Money is formatted from integer cents by desk.render.money; nothing here is a float."""
    original = policy.find_product(state, proposal.get("original_sku")) or {}
    replacement_sku = proposal.get("replacement_sku")
    replacement = policy.find_product(state, replacement_sku) if replacement_sku else None
    order = policy.find_customer_order(dict(state), str(proposal.get("customer_id")), str(proposal.get("order_id")))
    line = policy.find_line(order, str(proposal.get("line_id"))) if order else None
    eligibility = proposal.get("eligibility") or {}
    amount = proposal.get("amount_cents")
    refund = proposal.get("remedy") == "refund"
    return {
        "proposal_id": proposal.get("proposal_id"),
        "status": proposal.get("status"),
        "remedy": proposal.get("remedy"),
        "order_id": proposal.get("order_id"),
        "line_id": proposal.get("line_id"),
        "original_sku": proposal.get("original_sku"),
        "product_id": original.get("product_id"),
        "size": original.get("size"),
        "replacement_sku": replacement_sku,
        "replacement_size": replacement.get("size") if replacement else None,
        "replacement_stock": replacement.get("stock") if replacement else None,
        "amount_cents": amount,
        "amount_text": money(amount) if isinstance(amount, int) else None,
        "paid_text": money(line["paid_cents"]) if line and isinstance(line.get("paid_cents"), int) else None,
        "currency": proposal.get("currency"),
        "delivered_on": eligibility.get("delivered_on"),
        "as_of": eligibility.get("as_of"),
        "days_since_delivery": eligibility.get("days_since_delivery"),
        "window_days": eligibility.get("window_days"),
        "unopened": line.get("unopened") if line else None,
        "final_sale": line.get("final_sale") if line else None,
        "returned": line.get("returned") if line else None,
        "binding_hash": proposal.get("binding_hash"),  # the browser sends it back with a confirmation
        "binding_hash_short": str(proposal.get("binding_hash", ""))[:HASH_PREFIX],
        "created_turn": proposal.get("created_turn"),
        "confirm_label": (f"Confirm refund {money(amount)}" if refund and isinstance(amount, int)
                          else f"Confirm exchange to {replacement_sku}"),
    }


def snapshot(session: UISession, *, busy: bool | None = None) -> dict[str, Any]:
    """GET /state: final_state shape + proposals + external events + audit, plus what the UI derives views from."""
    snap = session.case.snapshot()
    plan = session.case.tools.faults
    return {
        "session_id": session.session_id,
        "case_id": session.spec.case_id,
        "customer_id": session.spec.customer_id,
        "request_id": session.spec.request_id,
        "generation": session.generation,
        "busy": session.busy if busy is None else busy,
        "initial_state": copy.deepcopy(session.spec.initial_state),
        **snap,
        "turns": copy.deepcopy(session.turn_log),
        "cards": {p["proposal_id"]: proposal_card(p, snap["final_state"]) for p in snap["proposals"]},
        "faults": {"scheduled": copy.deepcopy(session.spec.faults),
                   "timeout_armed": plan.timeout_armed and not plan.timeout_fired,
                   "timeout_fired": plan.timeout_fired},
    }


# ================================================================================================
# Live telemetry for the header


def step_summary(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Numbers of one LLM step (from its audit entry; never model text)."""
    numbers = entry.get("result") or {}
    completion = int(numbers.get("completion_tokens") or 0)
    predicted_ms = float(numbers.get("predicted_ms") or 0.0)
    prefill = int(numbers.get("prefill_tokens") or 0)
    prompt_ms = float(numbers.get("prompt_ms") or 0.0)
    return {
        "agent": entry.get("agent"), "tool": entry.get("tool"), "turn": entry.get("turn"),
        "latency_ms": numbers.get("latency_ms"), "prompt_tokens": numbers.get("prompt_tokens"),
        "cached_tokens": numbers.get("cached_tokens"), "completion_tokens": completion,
        "decode_tok_s": round(completion / predicted_ms * 1000, 1) if predicted_ms > 0 else None,
        "prefill_tok_s": round(prefill / prompt_ms * 1000, 1) if prompt_ms > 0 else None,
        "error": (entry.get("error") or {}).get("code"),
    }


class TelemetryBoard:
    """The last LLM step (pushed by every session's audit), the last turn, and memory readings."""

    def __init__(self, llama_port: int | None) -> None:
        self.llama_port = llama_port
        self.last_step: dict[str, Any] | None = None
        self.last_turn: dict[str, Any] | None = None
        self._llama_pid: int | None = None
        self._memory: dict[str, Any] | None = None
        self._memory_read_at = float("-inf")
        self._cpu = telemetry.CpuMeter()
        self._lock = threading.Lock()

    def observe(self, entry: dict[str, Any]) -> None:
        if entry.get("kind") == "llm_step":
            self.last_step = step_summary(entry)

    def observe_turn(self, session_id: str, turn: int, latency_ms: float | None, entries: list[dict[str, Any]]) -> None:
        steps = [e.get("result") or {} for e in entries if e.get("kind") == "llm_step" and e.get("turn") == turn]
        completion = sum(int(s.get("completion_tokens") or 0) for s in steps)
        decode_ms = sum(float(s.get("predicted_ms") or 0.0) for s in steps)
        self.last_turn = {
            "session_id": session_id, "turn": turn, "latency_ms": latency_ms, "llm_calls": len(steps),
            "prompt_tokens": sum(int(s.get("prompt_tokens") or 0) for s in steps),
            "cached_tokens": sum(int(s.get("cached_tokens") or 0) for s in steps),
            "completion_tokens": completion,
            "decode_tok_s": round(completion / decode_ms * 1000, 1) if decode_ms > 0 else None,
        }

    def memory(self) -> dict[str, Any]:
        with self._lock:
            if time.monotonic() - self._memory_read_at >= MEMORY_CACHE_S or self._memory is None:
                self._memory = self._read_memory()
                self._memory_read_at = time.monotonic()
            return dict(self._memory)

    def _read_memory(self) -> dict[str, Any]:
        """phys_footprint of this server and of llama-server, the GPU's unified-memory use, and how busy the
        machine is: CPU (whole machine and each server, as a share of all cores), GPU, memory used."""
        server = _footprint_bytes(os.getpid())
        llama = self._llama_footprint()
        gpu = telemetry.gpu_in_use_system_memory()
        vram = None if gpu is not None else telemetry.nvidia_gpu()["memory_used"]
        parts = [b for b in (server, llama) if b is not None]
        machine = telemetry.system_memory()
        return {
            "server_footprint_mb": _mib(server) if server is not None else None,
            "llama_footprint_mb": _mib(llama) if llama is not None else None,
            "footprint_mb": _mib(sum(parts)) if parts else None,
            "gpu_unified_mb": _mib(gpu) if gpu is not None else None,
            "gpu_vram_mb": _mib(vram) if vram is not None else None,
            "llama_pid": self._llama_pid,
            "cpu_pct": self._cpu.machine(),
            "server_cpu_pct": self._cpu.process(os.getpid()),
            "llama_cpu_pct": self._cpu.process(self._llama_pid),
            "gpu_busy_pct": telemetry.gpu_busy_percent(),
            "memory_used_mb": _mib(machine[0]) if machine else None,
            "memory_total_mb": _mib(machine[1]) if machine else None,
            "notes": "footprint = kernel phys_footprint (server + llama-server; resident set size where there is "
                     "no libproc, e.g. Linux); GPU = IOAccelerator 'In use system memory', system-wide unified "
                     "memory that overlaps the footprint (NVIDIA: gpu_vram_mb, dedicated memory in use from "
                     "nvidia-smi). CPU = share of all cores since the last read; GPU busy = 'Device Utilization "
                     "%' (NVIDIA: utilization.gpu); memory used = total - available, whole machine.",
        }

    def _llama_footprint(self) -> int | None:
        if self.llama_port is None:
            return None
        reading = _footprint_bytes(self._llama_pid) if self._llama_pid else None
        if reading is None:  # first call, or llama-server was restarted: look the pid up again
            self._llama_pid = pid_listening_on(self.llama_port)
            reading = _footprint_bytes(self._llama_pid) if self._llama_pid else None
        return reading


def _footprint_bytes(pid: int) -> int | None:
    """phys_footprint where libproc exists (macOS), else resident set size; None if the process is gone."""
    reading = telemetry.read_footprint(pid)
    return reading.phys_footprint if reading else telemetry.read_rss(pid)


def _mib(value: int) -> float:
    return round(value / MIB, 1)


class LLMHealth:
    """Is the local model answering? Cached briefly because the header polls it."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self._ok = False
        self._checked_at = float("-inf")
        self._lock = threading.Lock()

    def ok(self) -> bool:
        if self.config.fake:
            return True
        with self._lock:
            if time.monotonic() - self._checked_at >= HEALTH_CACHE_S:
                self._ok = server_healthy(self.config.llama_url, timeout_s=1.0)
                self._checked_at = time.monotonic()
            return self._ok


# ================================================================================================
# The turn stream


_DONE = object()


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def start_turn(session: UISession, board: TelemetryBoard, message: str, confirm_id: str | None,
               line: TurnLine | None = None, ticket: threading.Event | None = None) -> Iterator[str]:
    """Run one turn in a worker thread and return the SSE stream of its audit entries.

    The caller already holds `session.lock` (and a `ticket` in `line`, if given). While the ticket
    waits, the stream sends `queued` events with the place in line; then the turn runs. The worker
    releases both when the turn ends, even if the browser went away mid-stream (a turn is never
    abandoned half way)."""
    events: queue.Queue[Any] = queue.Queue()
    case = session.case
    turn_no = len(case.desk.history) + 1
    unsubscribe: Callable[[], None] = lambda: None  # noqa: E731 - replaced once the turn starts

    def work() -> None:
        nonlocal unsubscribe
        released = False
        try:
            while ticket is not None and line is not None and not ticket.wait(QUEUED_EVERY_S):
                events.put(("queued", {"turn": turn_no, "position": line.position(ticket)}))
            unsubscribe = case.audit.subscribe(lambda entry: events.put(("audit", entry)))
            events.put(("start", {"turn": turn_no, "message": message, "confirm_proposal_id": confirm_id}))
            output = case.run_turn(message, confirm_id)
            latency = case.desk.turn_latency_ms[-1] if case.desk.turn_latency_ms else None
            record = {"turn": turn_no, "message": message, "confirm_proposal_id": confirm_id, "output": output,
                      "latency_ms": latency}
            session.turn_log.append(record)
            board.observe_turn(session.session_id, turn_no, latency, case.audit.entries())
            unsubscribe()
            session.lock.release()
            released = True
            events.put(("turn", {**record, "state": snapshot(session, busy=False)}))
        except Exception as exc:  # noqa: BLE001 - reported to the browser, never swallowed
            events.put(("error", {"turn": turn_no, "message": f"{type(exc).__name__}: {exc}"[:500]}))
        finally:
            if not released:
                unsubscribe()
                session.lock.release()
            if line is not None:
                line.leave()
            events.put(_DONE)

    if ticket is not None and line is not None and not ticket.is_set():
        events.put(("queued", {"turn": turn_no, "position": line.position(ticket)}))
    threading.Thread(target=work, name=f"turn-{session.session_id}-{turn_no}", daemon=True).start()
    return _drain(events)


def _drain(events: queue.Queue[Any]) -> Iterator[str]:
    while True:
        try:
            item = events.get(timeout=SSE_KEEPALIVE_S)
        except queue.Empty:
            yield ": keep-alive\n\n"  # an SSE comment; keeps proxies from closing a slow turn
            continue
        if item is _DONE:
            return
        kind, data = item
        yield sse(kind, data)


# ================================================================================================
# HTTP API


class CreateSessionRequest(BaseModel):
    case_id: str | None = None
    initial_state: dict[str, Any] | None = None
    customer_id: str | None = None
    request_id: str | None = None


class TurnRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    confirm_proposal_id: str | None = None  # explicit approval bound to this proposal
    # What the page displayed; required with a confirmation (check_confirmation).
    confirm_binding_hash: str | None = Field(default=None, max_length=128)
    generation: StrictInt | None = None


class FaultRequest(BaseModel):
    type: Literal["stock_change", "timeout_after_commit"]
    sku: str | None = None
    new_stock: int | None = Field(default=None, ge=0, strict=True)


class ResetRequest(BaseModel):
    fresh_request_id: bool = False


def check_confirmation(session: UISession, body: TurnRequest) -> None:
    """A confirmation approves what the page showed, so it must say what that was: the session
    generation (no reset since) and the card's binding hash (customer, line, remedy, amount,
    replacement). Missing -> 422; different from the server's -> 409 and nothing runs. Proposal ids
    are per-store counters that restart after a reset, so the id alone proves nothing."""
    if not body.confirm_proposal_id:
        return
    if body.confirm_binding_hash is None or body.generation is None:
        raise HTTPException(422, MISSING_BINDING)
    stored = {p.get("proposal_id"): p.get("binding_hash") for p in session.case.snapshot()["proposals"]}
    if body.generation != session.generation or stored.get(body.confirm_proposal_id) != body.confirm_binding_hash:
        raise HTTPException(409, STALE_PROPOSAL)


# --- local-only request guard ---------------------------------------------------------------------


def _netloc(url_or_host: str) -> tuple[str | None, int | None]:
    """(hostname, port) of 'http://127.0.0.1:8000/x' or '127.0.0.1:8000'; (None, None) if unparsable."""
    text = url_or_host if "//" in url_or_host else f"//{url_or_host}"
    try:
        parts = urlsplit(text)
        return parts.hostname, parts.port
    except ValueError:
        return None, None


def request_refusal(method: str, headers: Headers, config: ServerConfig) -> tuple[int, str] | None:
    """Why this request must not reach the app, as (status, message); None when it may."""
    hostname, port = _netloc(headers.get("host", ""))
    # Any port: an SSH port forward (ssh -L 9000:127.0.0.1:8000) arrives as localhost:9000.
    if hostname not in config.allowed_hostnames:
        return 400, "this server only answers requests addressed to 127.0.0.1 or localhost"
    if method not in STATE_CHANGING_METHODS:
        return None
    source = headers.get("origin") or headers.get("referer")
    if source is not None and _netloc(source) != (hostname, port):
        return 403, "cross-site requests are refused: open the desk at its own address and try again"
    if "content-length" not in headers and "chunked" in headers.get("transfer-encoding", ""):
        return 411, "send a Content-Length with the request body"
    length = headers.get("content-length", "0")
    if not length.isdigit():
        return 400, "Content-Length is not a number"
    if int(length) > MAX_BODY_BYTES:
        return 413, f"request body over {MAX_BODY_BYTES // MIB} MiB"
    return None


class LocalRequestGuard:
    """ASGI middleware applying request_refusal() before any route runs (SSE streams pass through)."""

    def __init__(self, app: ASGIApp, config: ServerConfig) -> None:
        self.app = app
        self.config = config

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            refusal = request_refusal(scope["method"], Headers(scope=scope), self.config)
            if refusal is not None:
                await json_error(*refusal)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def json_error(status: int, detail: Any, headers: Mapping[str, str] | None = None) -> Response:
    """An error body that always encodes: json.dumps escapes non-ASCII (a lone surrogate included)."""
    return Response(json.dumps({"detail": detail}), status_code=status, media_type="application/json",
                    headers=dict(headers or {}))


def readable_validation_errors(errors: list[Any]) -> list[dict[str, Any]]:
    """Field, message and type of each problem; never the rejected input (it may not even encode)."""
    return [{"loc": [str(part) for part in e.get("loc", ())], "msg": str(e.get("msg", "invalid")),
             "type": str(e.get("type", ""))} for e in errors]


def create_app(config: ServerConfig | None = None, llm: Any = None) -> FastAPI:
    """The FastAPI app. `uvicorn server.app:create_app --factory` reads the config from the environment.
    `llm` replaces the configured client (tests pass scripted or deliberately broken ones)."""
    config = config or ServerConfig.from_env()
    profile = get_profile(config.profile)
    demo = load_cases(config.demo_cases_path) if config.demo_cases_path else []
    cases = {c["case_id"]: c for c in load_cases(config.cases_path) + demo}
    board = TelemetryBoard(None if config.fake else config.llama_port)
    model = llm or build_llm(config, profile)
    registry = SessionRegistry(model, profile, config.var_dir / "sessions", board)
    health = LLMHealth(config)
    limit = turn_limit(config)
    line = TurnLine(limit, config.max_waiting) if limit is not None else None
    app = FastAPI(title="Returns Desk", docs_url=None, redoc_url=None)
    app.state.registry = registry
    app.state.config = config
    app.state.turn_line = line
    app.state.load_runs = load_runs = LoadRuns(config.var_dir / "load")
    app.add_middleware(LocalRequestGuard, config=config)

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(_request: Request, exc: RequestValidationError) -> Response:
        return json_error(422, readable_validation_errors(list(exc.errors())))

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(_request: Request, exc: StarletteHTTPException) -> Response:
        return json_error(exc.status_code, exc.detail, exc.headers)

    def session_or_error(session_id: str) -> UISession:
        try:
            return registry.get(session_id)
        except SessionError as exc:
            raise HTTPException(exc.status, exc.message) from exc

    def claim(session: UISession) -> None:
        if not session.lock.acquire(blocking=False):
            raise HTTPException(409, "a turn is still running for this session; wait for it to finish")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/health")
    def get_health() -> dict[str, Any]:
        return {"llm": "ok" if health.ok() else "down", "mode": config.llm_mode, "llama_url": config.llama_url,
                **model_info(config, profile), "start": start_commands(config, profile)}

    @app.get("/api/meta")
    def get_meta() -> dict[str, Any]:
        """Plain-language reason codes and their decision class, so the UI words codes like the desk does."""
        return {"reason_text": REASON_TEXT, "code_class": CODE_CLASS, "window_days": WINDOW_DAYS}

    @app.get("/api/scenarios")
    def get_scenarios() -> list[dict[str, Any]]:
        return [scenario_view(case) for case in cases.values()]

    @app.post("/api/sessions")
    def create_session(body: CreateSessionRequest) -> dict[str, Any]:
        try:
            session = registry.create(spec_from_request(body, cases, registry))
        except SessionError as exc:
            raise HTTPException(exc.status, exc.message) from exc
        except (ValueError, TypeError, KeyError) as exc:  # e.g. an initial_state without a valid as_of
            raise HTTPException(422, f"initial_state is not usable: {exc}") from exc
        return {"session_id": session.session_id, "state": snapshot(session)}

    @app.get("/api/sessions/{session_id}/state")
    def get_state(session_id: str) -> dict[str, Any]:
        return snapshot(session_or_error(session_id))

    @app.post("/api/sessions/{session_id}/turns")
    def post_turn(session_id: str, body: TurnRequest) -> StreamingResponse:
        session = session_or_error(session_id)
        claim(session)
        try:  # checked under the session lock, so no reset can slip in between the check and the turn
            check_confirmation(session, body)
        except HTTPException:
            session.lock.release()
            raise
        ticket = line.join() if line is not None else None
        if line is not None and ticket is None:
            session.lock.release()
            raise HTTPException(503, f"the desk is at capacity ({line.max_waiting} turns already waiting); "
                                     f"retry in {RETRY_AFTER_S} s", headers={"Retry-After": str(RETRY_AFTER_S)})
        stream = start_turn(session, board, body.message, body.confirm_proposal_id or None, line, ticket)
        return StreamingResponse(stream, media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.post("/api/sessions/{session_id}/faults")
    def post_fault(session_id: str, body: FaultRequest) -> dict[str, Any]:
        session = session_or_error(session_id)
        if body.type == "stock_change" and (not body.sku or body.new_stock is None):
            raise HTTPException(422, "stock_change needs sku and a non-negative integer new_stock")
        claim(session)
        try:
            tools = session.case.tools
            if body.type == "stock_change":
                result: dict[str, Any] = {"event": tools.inject_stock_change(str(body.sku), int(body.new_stock or 0))}
            else:
                tools.arm_timeout()
                result = {"armed": True}
        finally:
            session.lock.release()
        return {**result, "state": snapshot(session)}

    @app.post("/api/sessions/{session_id}/reset")
    def post_reset(session_id: str, body: ResetRequest | None = None) -> dict[str, Any]:
        session = session_or_error(session_id)
        claim(session)
        try:
            registry.reset(session, fresh_request_id=bool(body and body.fresh_request_id))
        finally:
            session.lock.release()
        return snapshot(session)

    @app.post("/api/sessions/{session_id}/new-request")
    def post_new_request(session_id: str) -> dict[str, Any]:
        session = session_or_error(session_id)
        claim(session)
        try:
            registry.new_request(session)
        finally:
            session.lock.release()
        return snapshot(session)

    @app.post("/api/load")
    def post_load(body: LoadRequest) -> dict[str, Any]:
        """Start a load run: `customers` simulated customers, `concurrency` at a time, through this API."""
        if config.port is None:
            raise HTTPException(503, "the load tool needs this server's port: start the desk with scripts/run.sh")
        unknown = [case_id for case_id in body.scenarios or [] if case_id not in cases]
        if unknown:
            raise HTTPException(422, f"unknown scenario(s): {', '.join(unknown)}")
        try:
            return load_runs.start(f"http://127.0.0.1:{config.port}", body)
        except SessionError as exc:
            raise HTTPException(exc.status, exc.message) from exc

    @app.get("/api/load")
    def get_load() -> dict[str, Any]:
        return {"run": load_runs.view()}

    @app.post("/api/load/{run_id}/stop")
    def stop_load(run_id: str) -> dict[str, Any]:
        try:
            return load_runs.stop(run_id)
        except SessionError as exc:
            raise HTTPException(exc.status, exc.message) from exc

    @app.get("/api/telemetry")
    def get_telemetry() -> dict[str, Any]:
        capacity: dict[str, Any] = line.snapshot() if line is not None else {"max_turns": None, "turns_refused": 0}
        if hasattr(model, "load"):
            capacity["model"] = model.load()
        return {"last_step": board.last_step, "last_turn": board.last_turn, "memory": board.memory(),
                "mode": config.llm_mode, "capacity": capacity}

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


# ================================================================================================
# Command line: optionally start (or attach to) llama-server, warm it up, then serve the UI


def warm_up(base_url: str, profile: Profile) -> dict[str, float]:
    """Prime the prompt cache with each agent's static prefix (system prompt + tool docs), so the
    first demo turn does not pay a cold prefill. Answers are discarded; returns ms per role."""
    timings: dict[str, float] = {}
    roles = (("resolution", RESOLUTION_SYSTEM, RESOLUTION_TOOLS, [REPLY]),
             ("operations", operations_system(profile.structured_report), operations_tools(profile.structured_report), [REPORT]))
    with LLMClient(base_url, profile.protocol, seed=SEED, timeout_s=120, retries=0) as client:
        for role, system, tools, allowed in roles:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": "Warm-up. Reply briefly."}]
            started = time.perf_counter()
            client.step(role, messages, tools, allowed, 16)
            timings[role] = round((time.perf_counter() - started) * 1000, 1)
    return timings


def ensure_llama(config: ServerConfig, profile: Profile, warm: bool = True) -> LlamaServer | None:
    """Attach to a healthy llama-server at the URL, or start our own (sha256 + version checked)."""
    if server_healthy(config.llama_url):
        _log(f"attached to the running llama-server at {config.llama_url}")
        server = None
    else:
        log = config.var_dir / f"llama-server-{config.llama_port}.log"
        server = LlamaServer(profile, config.llama_port, log, allow_engine_mismatch=config.allow_engine_mismatch,
                             slots=config.slots)
        startup = server.start()
        if startup.get("engine_version_match") is False:
            _log(f"WARNING: llama.cpp {startup['engine_version']!r} is not the pinned {startup['engine_version_expected']!r}; "
                 "running anyway because of --allow-engine-mismatch (results may differ from the benchmark)")
        _log(f"llama-server ready on port {config.llama_port}: spawn to ready {startup['server_spawn_to_ready_ms']} ms, "
             f"first request {startup['first_request_ms']} ms, model page cache {startup['model_page_cache']}")
    if warm:
        try:
            _log(f"prompt cache warmed: {warm_up(config.llama_url, profile)} ms")
        except LLMError as exc:
            _log(f"warm-up skipped: {exc}")
    return server


def served_slots(llama_url: str) -> int | None:
    """The running llama-server's slot count (GET /props), or None when it does not say."""
    try:
        return int(httpx.get(f"{llama_url.rstrip('/')}/props", timeout=2.0).json()["total_slots"])
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return None


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Returns Desk UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_UI_PORT)
    parser.add_argument("--llama-url", help=f"llama-server URL (default $DESK_LLAMA_URL or {DEFAULT_LLAMA_URL})")
    parser.add_argument("--profile", help="desk profile (default $DESK_PROFILE or F)")
    parser.add_argument("--fake", action="store_true", help="use the deterministic fake-policy LLM (no model)")
    parser.add_argument("--start-llama", action="store_true", help="start llama-server if none is healthy at the URL")
    parser.add_argument("--no-warmup", action="store_true", help="skip the prompt-cache warm-up")
    parser.add_argument("--slots", type=int, default=SLOTS,
                        help=f"parallel model requests, with --start-llama (default {SLOTS}, as benchmarked)")
    parser.add_argument("--max-turns", type=int, help=f"turns running at once (default {TURNS_PER_SLOT} per slot)")
    parser.add_argument("--allow-engine-mismatch", action="store_true",
                        help="with --start-llama, run a llama.cpp build other than the pinned one (or DESK_ALLOW_ENGINE_MISMATCH=1)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ServerConfig.from_env()
    # A non-wildcard --host (e.g. a LAN name) must also be accepted in the Host header.
    hostnames = config.allowed_hostnames + (() if args.host in ("0.0.0.0", "::") else (args.host.strip("[]").lower(),))
    config = replace(config, llama_url=args.llama_url or config.llama_url, profile=args.profile or config.profile,
                     llm_mode=FAKE_POLICY if args.fake else config.llm_mode, port=args.port,
                     allow_engine_mismatch=args.allow_engine_mismatch or config.allow_engine_mismatch,
                     allowed_hostnames=tuple(dict.fromkeys(hostnames)), slots=args.slots, max_turns=args.max_turns)
    profile = get_profile(config.profile)
    # Python's default SIGTERM/SIGHUP end the process without running `finally`; turn them into SystemExit
    # so a llama-server we started is always stopped (scripts/run.sh sends SIGTERM on exit).
    signal.signal(signal.SIGTERM, _exit_on_signal)
    if hasattr(signal, "SIGHUP"):  # not on Windows
        signal.signal(signal.SIGHUP, _exit_on_signal)
    llama: LlamaServer | None = None
    try:
        if args.start_llama and not config.fake:
            try:
                llama = ensure_llama(config, profile, warm=not args.no_warmup)
            except LlamaServerError as exc:
                _log(f"could not start llama-server: {exc}")
                return 3
            actual = served_slots(config.llama_url)
            if actual is not None and actual != config.slots:  # attached to a server started with other flags
                _log(f"llama-server serves {actual} slot(s), not {config.slots}: the desk sends {actual} call(s) at a time")
                config = replace(config, slots=actual)
        _log(f"Returns Desk UI on http://{args.host}:{args.port} (profile {profile.name}, llm {config.llm_mode})")
        app = create_app(config)
        try:
            uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
        finally:
            app.state.load_runs.stop_all()  # a load run started from the page ends with the server
    finally:
        if llama is not None:
            llama.stop()
    return 0


def _exit_on_signal(signum: int, _frame: Any) -> None:
    raise SystemExit(128 + signum)


def _log(message: str) -> None:
    print(f"[ui] {message}", flush=True)


if __name__ == "__main__":  # pragma: no cover - the entry point itself; main() is tested
    raise SystemExit(main())
