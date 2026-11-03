"""LLM client for the agents: exactly one validated tool call per step, over llama-server's OpenAI API.

The agents never see the wire format. They build role-agnostic messages and get back a `Step`
(chosen tool, validated arguments, token/timing telemetry, or a protocol error to feed back).

Message format (what the orchestrator appends; plain dicts):
    {"role": "system",    "content": str}                    first message only (static role prompt)
    {"role": "user",      "content": str}
    {"role": "assistant", "content": "", "tool_name": str, "tool_args": dict}   a call the model made
    {"role": "assistant", "content": str}                    plain text (only after NO_TOOL_CALL)
    {"role": "tool",      "content": str, "tool_name": str}  result of the preceding call (JSON text)
`assistant_message(step)` and `tool_message(name, result)` build the last two kinds.

Protocols (SPEC §7; measurements in docs/notes/llm_infra.md):
  native       OpenAI `tools` (the full role list, byte-identical every step) + tool_choice "required"
               + parallel_tool_calls false. The Qwen3.5 template renders tool calls as XML with free-text
               values, so the server does NOT enforce enums or required fields; everything is checked
               here. Only the first call is used; extra calls are returned in `dropped_calls`.
               A tagged-union (`oneOf`) parameters schema is sent flattened (flatten_one_of): the
               native grammar is built from top-level properties only. The strict union is validated.
  constrained  no `tools` parameter (llama-server 0.7.0 answers HTTP 400 when `tools` and
               `response_format` are combined). Tool docs are rendered once into the system prompt,
               byte-stable, and each step sends `response_format` = json_schema `oneOf` over the tools
               allowed at this step, `{"tool": const, "args": schema}` with `tool` first. The decoder can
               then only produce an allowed tool with schema-valid arguments; we still validate.
Per-step restrictions live in `allowed` (code + sampling side), never in the prompt prefix, so the
prompt cache keeps hitting across steps (changing the tools block at the top of the prompt costs a
full re-prefill). Keep a turn's messages append-only and replay calls with assistant_message(step):
Qwen3.5 is a hybrid recurrent model, so any byte change inside the history makes llama-server fall
back to its checkpoint at the start of the last user message instead of the point of divergence.

Protocol errors (string codes, fed back to the model as a tool error and counted in metrics):
  NO_TOOL_CALL, INVALID_JSON, TRUNCATED, UNKNOWN_TOOL, TOOL_NOT_ALLOWED, INVALID_ARGS.
Transport failures raise `LLMError` after one retry of the HTTP request only (a tool never re-runs).
`FakeLLM` replays scripted model outputs through the same interpretation code, including adversarial
outputs (two calls, unknown tool, bad arguments), so tests exercise the real validation path.
"""

from __future__ import annotations

import copy
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

PROTOCOLS = ("native", "constrained")
CHAT_PATH = "/v1/chat/completions"

# --------------------------------------------------------------------------------------------------
# Tool specs and a small JSON-schema validator (only the subset we use, so nothing is silently ignored)

SCHEMA_KEYWORDS = frozenset({
    "type", "properties", "required", "additionalProperties", "enum", "const", "oneOf", "items",
    "description", "title",
})
JSON_TYPES = frozenset({"object", "string", "integer", "number", "boolean", "array", "null"})


def check_schema(schema: Any, path: str = "parameters") -> None:
    """Reject schemas that use keywords our validator does not enforce, or open-ended objects.

    Failing at import time is the point: a toolspec that relies on e.g. `pattern` would look
    constrained but never be checked in native mode.
    """
    if not isinstance(schema, Mapping):
        raise ValueError(f"{path}: schema must be an object")
    unknown = set(schema) - SCHEMA_KEYWORDS
    if unknown:
        raise ValueError(f"{path}: unsupported schema keywords {sorted(unknown)}")
    types = schema.get("type")
    for t in types if isinstance(types, list) else ([types] if types else []):
        if t not in JSON_TYPES:
            raise ValueError(f"{path}: unknown type {t!r}")
    if "properties" in schema:
        if schema.get("additionalProperties") is not False:
            raise ValueError(f"{path}: objects must set additionalProperties: false")
        missing = set(schema.get("required", [])) - set(schema["properties"])
        if missing:
            raise ValueError(f"{path}: required names not in properties: {sorted(missing)}")
        for name, sub in schema["properties"].items():
            check_schema(sub, f"{path}.{name}")
    elif types == "object" and "oneOf" not in schema:
        raise ValueError(f"{path}: object schemas need properties (use {{}} for no arguments) or oneOf")
    for i, branch in enumerate(schema.get("oneOf", [])):
        check_schema(branch, f"{path}.oneOf[{i}]")
    if "items" in schema:
        check_schema(schema["items"], f"{path}[]")


def _type_ok(expected: str, value: Any) -> bool:
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, {"object": dict, "string": str, "boolean": bool, "array": list,
                              "null": type(None)}[expected])


def _discriminator(branches: Sequence[Mapping[str, Any]]) -> str | None:
    """A property that every oneOf branch pins with `const` (e.g. "intent"), if there is one."""
    common: set[str] | None = None
    for branch in branches:
        consts = {k for k, v in branch.get("properties", {}).items() if "const" in v}
        common = consts if common is None else common & consts
    return sorted(common)[0] if common else None


def validate(schema: Mapping[str, Any], value: Any, path: str = "args") -> list[str]:
    """Return human-readable violations of `schema` by `value` (empty list = valid)."""
    if "const" in schema and value != schema["const"]:
        return [f"{path}: must be {schema['const']!r}"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"{path}: {value!r} is not one of {list(schema['enum'])}"]
    expected = schema.get("type")
    if expected is not None:
        options = expected if isinstance(expected, list) else [expected]
        if not any(_type_ok(t, value) for t in options):
            return [f"{path}: expected {' or '.join(options)}, got {type(value).__name__}"]
    errors: list[str] = []
    if isinstance(value, dict) and "properties" in schema:
        props = schema["properties"]
        errors += [f"{path}: missing required property '{name}'"
                   for name in schema.get("required", []) if name not in value]
        errors += [f"{path}: unexpected property '{name}'" for name in value if name not in props]
        for name, sub_value in value.items():
            if name in props:
                errors += validate(props[name], sub_value, f"{path}.{name}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors += validate(schema["items"], item, f"{path}[{i}]")
    if "oneOf" in schema:
        errors += _validate_one_of(schema["oneOf"], value, path)
    return errors


def _validate_one_of(branches: Sequence[Mapping[str, Any]], value: Any, path: str) -> list[str]:
    key = _discriminator(branches)
    if key is not None and isinstance(value, dict):
        # Tagged union: pick the branch by its tag so the error names the real problem.
        tags = [b["properties"][key]["const"] for b in branches]
        for branch, tag in zip(branches, tags):
            if value.get(key) == tag:
                return validate(branch, value, path)
        return [f"{path}.{key}: {value.get(key)!r} is not one of {tags}"]
    matching = sum(1 for b in branches if not validate(b, value, path))
    return [] if matching == 1 else [f"{path}: must match exactly one oneOf branch (matched {matching})"]


def flatten_one_of(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    """A flat object schema equivalent in spirit to a top-level tagged union, for the native protocol.

    Measured on llama-server 0.7.0 + Qwen3.5: the native tool-call grammar is built from the top-level
    `properties`, so a top-level `oneOf` leaves the model no parameters at all and every call arrives
    as `{}`. Native mode therefore sends the union of the branch properties, the tag as an enum
    (with each branch's required fields spelled out in its description) and only the fields every
    branch requires. The strict oneOf is still what `validate()` checks, so nothing is loosened.
    """
    branches = schema.get("oneOf")
    key = _discriminator(branches) if branches else None
    if key is None:
        return schema
    properties: dict[str, Any] = {}
    for branch in branches:
        for name, sub in branch["properties"].items():
            properties.setdefault(name, sub)
    tags = [b["properties"][key]["const"] for b in branches]
    needs = "; ".join(f"{tag}: {', '.join(r for r in b.get('required', []) if r != key) or 'nothing else'}"
                      for tag, b in zip(tags, branches))
    properties[key] = {"type": "string", "enum": tags, "description": f"Required fields per {key}: {needs}"}
    required = [name for name in properties if all(name in b.get("required", []) for b in branches)]
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


@dataclass(frozen=True)
class ToolSpec:
    """One tool an agent may call. Define these as module-level constants (they must be byte-stable).

    `parameters` is a JSON schema of the arguments object; `terminal` marks tools that end an agent
    loop (e.g. reply_to_customer, report_to_resolution).
    """

    name: str
    description: str
    parameters: Mapping[str, Any]
    terminal: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"tool name must be a plain identifier: {self.name!r}")
        if self.parameters.get("type") != "object":
            raise ValueError(f"{self.name}: parameters must be a JSON schema of type object")
        check_schema(self.parameters, f"{self.name}.parameters")

    def openai_tool(self) -> dict[str, Any]:
        """Native-protocol tool entry; a tagged-union schema is sent flattened (see flatten_one_of)."""
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": flatten_one_of(self.parameters)}}

    def doc_line(self) -> str:
        return json.dumps({"name": self.name, "description": self.description,
                           "parameters": self.parameters}, ensure_ascii=False)


# --------------------------------------------------------------------------------------------------
# Step: what one LLM call produced

@dataclass
class Step:
    """Result of one LLM step. `ok` steps carry a tool from `allowed` with schema-valid `args`."""

    tool: str | None
    args: dict[str, Any] = field(default_factory=dict)
    protocol_error: str | None = None
    dropped_calls: list[dict[str, Any]] = field(default_factory=list)
    role: str = ""
    prompt_tokens: int = 0  # whole prompt, cached + newly processed
    completion_tokens: int = 0
    cached_tokens: int = 0
    prompt_ms: float = 0.0  # prefill time of the uncached part
    predicted_ms: float = 0.0  # decode time
    latency_ms: float = 0.0  # wall time of step(), including a retry
    finish_reason: str | None = None
    text: str = ""  # visible model output (constrained: the raw JSON); never contains reasoning
    http_attempts: int = 0

    @property
    def ok(self) -> bool:
        return self.protocol_error is None and self.tool is not None

    @property
    def prefill_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.cached_tokens)

    def telemetry(self) -> dict[str, Any]:
        """Flat numbers for metrics/audit (no model text)."""
        return {
            "role": self.role, "tool": self.tool, "protocol_error": self.protocol_error,
            "dropped_calls": len(self.dropped_calls), "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens, "prefill_tokens": self.prefill_tokens,
            "completion_tokens": self.completion_tokens, "prompt_ms": self.prompt_ms,
            "predicted_ms": self.predicted_ms, "latency_ms": self.latency_ms,
            "finish_reason": self.finish_reason, "http_attempts": self.http_attempts,
        }


class LLMError(Exception):
    """The LLM could not be reached or answered unusably (after the retry). Turn → decision error."""

    def __init__(self, kind: str, message: str, *, status_code: int | None = None,
                 attempts: int = 0) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind  # connection | timeout | http_status | bad_response | reasoning | busy | unavailable
        self.status_code = status_code
        self.attempts = attempts


def assistant_message(step: Step) -> dict[str, Any]:
    """The role-agnostic message to append after `step` (before its tool result)."""
    if step.tool is not None:
        return {"role": "assistant", "content": "", "tool_name": step.tool, "tool_args": step.args}
    return {"role": "assistant", "content": step.text}


def tool_message(tool_name: str, result: Any) -> dict[str, Any]:
    """A tool result message; non-string results are serialized as compact, key-sorted JSON."""
    content = result if isinstance(result, str) else json.dumps(
        result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"role": "tool", "content": content, "tool_name": tool_name}


# --------------------------------------------------------------------------------------------------
# Interpreting raw model output (shared by LLMClient and FakeLLM)

@dataclass(frozen=True)
class RawCall:
    """A call as the model emitted it, before validation. `arguments` is JSON text or a dict."""

    name: Any
    arguments: Any


def _check_request(tools: Sequence[ToolSpec], allowed: Sequence[str]) -> None:
    names = [t.name for t in tools]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate tool names: {names}")
    if not allowed or set(allowed) - set(names):
        raise ValueError(f"allowed must be a non-empty subset of {names}, got {list(allowed)}")


def _parse_arguments(arguments: Any) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return None, f"INVALID_JSON: arguments are not JSON ({exc.msg})"
    if not isinstance(arguments, dict):
        return None, f"INVALID_ARGS: args: expected object, got {type(arguments).__name__}"
    return arguments, None


def interpret_calls(calls: Sequence[RawCall], tools: Sequence[ToolSpec],
                    allowed: Sequence[str]) -> Step:
    """Turn raw calls into a Step: first call only, known + allowed tool, schema-valid arguments."""
    if not calls:
        return Step(tool=None, protocol_error="NO_TOOL_CALL: answer with exactly one tool call")
    first, extra = calls[0], calls[1:]
    dropped = [{"tool": c.name, "args": _parse_arguments(c.arguments)[0]} for c in extra]
    name = first.name if isinstance(first.name, str) else None
    args, error = _parse_arguments(first.arguments)
    by_name = {t.name: t for t in tools}
    if name not in by_name:
        error = f"UNKNOWN_TOOL: {first.name!r} is not a tool (tools: {', '.join(by_name)})"
        name = None
    elif name not in allowed:
        error = f"TOOL_NOT_ALLOWED: {name} is not allowed now (allowed now: {', '.join(allowed)})"
    elif error is None:
        violations = validate(by_name[name].parameters, args)
        if violations:
            error = "INVALID_ARGS: " + "; ".join(violations)
    return Step(tool=name, args=args or {}, protocol_error=error, dropped_calls=dropped)


def parse_constrained_content(content: str, finish_reason: str | None,
                              max_tokens: int) -> list[RawCall] | str:
    """Raw calls from a constrained-mode answer, or a protocol error string."""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as exc:
        if finish_reason == "length":
            return f"TRUNCATED: output hit max_tokens={max_tokens} before the JSON closed"
        return f"INVALID_JSON: answer is not JSON ({exc.msg})"
    if not isinstance(obj, dict) or "tool" not in obj:
        return 'INVALID_JSON: expected {"tool": <name>, "args": {...}}'
    return [RawCall(name=obj["tool"], arguments=obj.get("args", {}))]


# --------------------------------------------------------------------------------------------------
# Wire formats

def render_tool_docs(tools: Sequence[ToolSpec]) -> str:
    """Tool documentation for the constrained protocol's system prompt (same bytes every step)."""
    return "\n".join([
        "# Tools",
        "",
        "You have access to the following tools:",
        "",
        "<tools>",
        *(t.doc_line() for t in tools),
        "</tools>",
        "",
        'Answer every step with exactly one JSON object and nothing else: '
        '{"tool": "<tool name>", "args": {<arguments of that tool>}}. '
        "Only the tools allowed at the current step are accepted.",
    ])


def encode_constrained_call(tool_name: str, tool_args: Mapping[str, Any]) -> str:
    """How an earlier constrained call is replayed as assistant content.

    Same bytes the decoder produces for `{"tool": ..., "args": ...}` with the property order the
    schema fixes (`json.dumps` default separators), so the replayed history matches the cached tokens.
    """
    return json.dumps({"tool": tool_name, "args": dict(tool_args)}, ensure_ascii=False)


def response_format(tools: Sequence[ToolSpec], allowed: Sequence[str]) -> dict[str, Any]:
    """json_schema response_format: one branch per allowed tool, in the role's tool order."""
    branches = [
        {"type": "object",
         "properties": {"tool": {"const": t.name}, "args": t.parameters},
         "required": ["tool", "args"],
         "additionalProperties": False}
        for t in tools if t.name in allowed
    ]
    schema = branches[0] if len(branches) == 1 else {"oneOf": branches}
    return {"type": "json_schema", "json_schema": {"name": "agent_step", "strict": True, "schema": schema}}


def _check_messages(messages: Sequence[Mapping[str, Any]]) -> None:
    for i, m in enumerate(messages):
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"message {i}: unknown role {role!r}")
        if role == "system" and i != 0:
            raise ValueError(f"message {i}: system message must come first")
        is_call = role == "assistant" and m.get("tool_name")
        if not isinstance(m.get("content", "" if is_call else None), str):
            raise ValueError(f"message {i}: content must be a string")


def native_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Role-agnostic messages → OpenAI chat messages with tool_calls (deterministic call ids)."""
    out: list[dict[str, Any]] = []
    call_id = "call_0"
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_name"):
            call_id = f"call_{sum(1 for o in out if o.get('tool_calls')) + 1}"
            out.append({"role": "assistant", "content": m.get("content", ""), "tool_calls": [{
                "id": call_id, "type": "function", "function": {
                    "name": m["tool_name"],
                    "arguments": json.dumps(m.get("tool_args") or {}, ensure_ascii=False)}}]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": call_id, "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def constrained_messages(messages: Sequence[Mapping[str, Any]],
                         tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    """Role-agnostic messages → plain chat messages; tool docs appended to the system prompt."""
    docs = render_tool_docs(tools)
    out: list[dict[str, Any]] = []
    if not messages or messages[0]["role"] != "system":
        out.append({"role": "system", "content": docs})
    call_no = 0
    for m in messages:
        if m["role"] == "system":
            out.append({"role": "system", "content": f"{m['content']}\n\n{docs}"})
        elif m["role"] == "assistant" and m.get("tool_name"):
            call_no += 1
            out.append({"role": "assistant",
                        "content": encode_constrained_call(m["tool_name"], m.get("tool_args") or {})})
        elif m["role"] == "tool":
            # The template renders role "tool" as <tool_response> inside a user turn.
            out.append({"role": "tool", "tool_call_id": f"call_{call_no}", "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


# --------------------------------------------------------------------------------------------------
# The real client

def _model_failure(exc: LLMError) -> bool:
    """The model is down or hung (what the breaker counts), not a bad answer from a working model."""
    return exc.kind in ("connection", "timeout") or (exc.kind == "http_status" and (exc.status_code or 0) >= 500)


class CircuitBreaker:
    """Stop calling a model that is down. After `failures` failed calls in a row, every call fails at
    once (LLMError "unavailable") for `open_s` seconds, so turns end in a clean error instead of each
    waiting out its own timeouts; then one trial call goes through, and its success closes the breaker."""

    def __init__(self, failures: int = 3, open_s: float = 10.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.failures = failures
        self.open_s = open_s
        self._clock = clock
        self._lock = threading.Lock()
        self._streak = 0
        self._open_until: float | None = None
        self._trial = False
        self.opened = 0

    def before_call(self) -> None:
        with self._lock:
            if self._open_until is None:
                return
            if self._trial or self._clock() < self._open_until:
                raise LLMError("unavailable", "the model failed several calls in a row; retrying it shortly")
            self._trial = True  # half-open: this call decides

    def record(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self._streak, self._open_until, self._trial = 0, None, False
                return
            self._streak += 1
            if self._trial or self._streak >= self.failures:
                self._open_until = self._clock() + self.open_s
                self._trial = False
                self.opened += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = "closed" if self._open_until is None else ("half-open" if self._trial else "open")
            return {"state": state, "failures_in_a_row": self._streak, "times_opened": self.opened}


class LLMClient:
    """One tool-call step at a time against a llama-server; see the module docstring for protocols."""

    def __init__(self, base_url: str, protocol: str, *, seed: int = 42, timeout_s: float = 20.0,
                 retries: int = 1, retry_backoff_s: float = 0.25,
                 transport: httpx.BaseTransport | None = None, max_in_flight: int | None = None,
                 queue_timeout_s: float = 60.0, breaker: CircuitBreaker | None = None) -> None:
        if protocol not in PROTOCOLS:
            raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
        self.base_url = base_url.rstrip("/")
        self.protocol = protocol
        self.seed = seed
        self.retries = retries
        self.retry_backoff_s = retry_backoff_s
        # At most `max_in_flight` calls at the model at once (its slot count); the rest wait here, so the
        # step timeout measures the model's work, never time queued behind other sessions' calls.
        self.max_in_flight = max_in_flight
        self.queue_timeout_s = queue_timeout_s
        self.breaker = breaker
        self._gate = threading.BoundedSemaphore(max_in_flight) if max_in_flight else None
        self._load_lock = threading.Lock()
        self._waiting = 0
        self._in_flight = 0
        self._busy = 0
        self._max_wait_ms = 0.0
        self._http = httpx.Client(base_url=self.base_url, transport=transport,
                                  timeout=httpx.Timeout(timeout_s, connect=min(5.0, timeout_s)))

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def build_payload(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[ToolSpec],
                      allowed: Sequence[str], max_tokens: int) -> dict[str, Any]:
        """The exact request body for one step (public so tests and probes can inspect it)."""
        _check_request(tools, allowed)
        _check_messages(messages)
        payload: dict[str, Any] = {
            "temperature": 0,
            "seed": self.seed,
            "max_tokens": max_tokens,
            "stream": False,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.protocol == "native":
            payload["messages"] = native_messages(messages)
            payload["tools"] = [t.openai_tool() for t in tools]
            payload["tool_choice"] = "required"
            payload["parallel_tool_calls"] = False
        else:
            payload["messages"] = constrained_messages(messages, tools)
            payload["response_format"] = response_format(tools, allowed)
        return payload

    def step(self, role: str, messages: Sequence[Mapping[str, Any]], tools: Sequence[ToolSpec],
             allowed: Sequence[str], max_tokens: int) -> Step:
        """Run one LLM call and return the interpreted Step. Raises LLMError on transport failure."""
        payload = self.build_payload(messages, tools, allowed, max_tokens)
        started = time.perf_counter()  # telemetry only, never business logic
        body, attempts = self._call(payload)
        step = self._interpret(body, tools, allowed, max_tokens)
        step.role = role
        step.http_attempts = attempts
        step.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return step

    def load(self) -> dict[str, Any]:
        """Telemetry: calls waiting for a model slot and at the model now, refusals, the breaker."""
        with self._load_lock:
            out: dict[str, Any] = {"slots": self.max_in_flight, "waiting": self._waiting, "in_flight": self._in_flight,
                                   "busy_refusals": self._busy, "max_queue_wait_ms": round(self._max_wait_ms, 1)}
        if self.breaker is not None:
            out["breaker"] = self.breaker.snapshot()
        return out

    def _call(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """`_post` behind the slot gate and the circuit breaker (both optional)."""
        if self._gate is not None:
            with self._load_lock:
                self._waiting += 1
            queued = time.perf_counter()
            got = self._gate.acquire(timeout=self.queue_timeout_s)
            with self._load_lock:
                self._waiting -= 1
                self._max_wait_ms = max(self._max_wait_ms, (time.perf_counter() - queued) * 1000)
                self._busy += not got
            if not got:
                raise LLMError("busy", f"no model slot free within {self.queue_timeout_s:g} s")
        with self._load_lock:
            self._in_flight += 1
        try:
            if self.breaker is not None:
                self.breaker.before_call()
            try:
                result = self._post(payload)
            except LLMError as exc:
                if self.breaker is not None:
                    self.breaker.record(ok=not _model_failure(exc))
                raise
            if self.breaker is not None:
                self.breaker.record(ok=True)
            return result
        finally:
            with self._load_lock:
                self._in_flight -= 1
            if self._gate is not None:
                self._gate.release()

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """POST with one retry for connection errors, timeouts and 5xx; 4xx fail immediately."""
        last: LLMError | None = None
        for attempt in range(1, self.retries + 2):
            if attempt > 1 and self.retry_backoff_s:
                time.sleep(self.retry_backoff_s)
            try:
                response = self._http.post(CHAT_PATH, json=payload)
            except httpx.TimeoutException as exc:
                last = LLMError("timeout", f"no answer within the step timeout ({exc!r})", attempts=attempt)
                continue
            except httpx.TransportError as exc:
                last = LLMError("connection", f"cannot reach {self.base_url} ({exc!r})", attempts=attempt)
                continue
            if response.status_code >= 500:
                last = LLMError("http_status", response.text[:300], status_code=response.status_code,
                                attempts=attempt)
                continue
            if response.status_code != 200:
                raise LLMError("http_status", response.text[:300], status_code=response.status_code,
                               attempts=attempt)
            try:
                body = response.json()
            except ValueError as exc:
                raise LLMError("bad_response", f"response is not JSON ({exc})", attempts=attempt) from exc
            return body, attempt
        assert last is not None
        raise last

    def _interpret(self, body: Mapping[str, Any], tools: Sequence[ToolSpec], allowed: Sequence[str],
                   max_tokens: int) -> Step:
        try:
            choice = body["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("bad_response", f"no choices[0].message in response ({exc!r})") from exc
        if message.get("reasoning_content"):
            raise LLMError("reasoning", "reasoning content returned; run the server with --reasoning off")
        finish = choice.get("finish_reason")
        text = message.get("content") or ""
        if self.protocol == "native":
            calls = [RawCall(c.get("function", {}).get("name"), c.get("function", {}).get("arguments", ""))
                     for c in message.get("tool_calls") or []]
            step = interpret_calls(calls, tools, allowed)
        else:
            parsed = parse_constrained_content(text, finish, max_tokens)
            step = (Step(tool=None, protocol_error=parsed) if isinstance(parsed, str)
                    else interpret_calls(parsed, tools, allowed))
        step.text = text
        step.finish_reason = finish
        _apply_usage(step, body)
        return step


def _apply_usage(step: Step, body: Mapping[str, Any]) -> None:
    """Token counts and timings: llama-server `timings` first, OpenAI `usage` as fallback."""
    usage = body.get("usage") or {}
    timings = body.get("timings") or {}
    cached = timings.get("cache_n", (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
    if "prompt_n" in timings:
        prompt_total = int(timings["prompt_n"]) + int(cached)
    else:
        prompt_total = int(usage.get("prompt_tokens", 0))
    step.prompt_tokens = prompt_total
    step.cached_tokens = int(cached)
    step.completion_tokens = int(timings.get("predicted_n", usage.get("completion_tokens", 0)))
    step.prompt_ms = round(float(timings.get("prompt_ms", 0.0)), 2)
    step.predicted_ms = round(float(timings.get("predicted_ms", 0.0)), 2)


# --------------------------------------------------------------------------------------------------
# FakeLLM for tests

@dataclass
class StepRequest:
    """What an agent asked the (fake) LLM for; recorded for assertions."""

    role: str
    messages: list[dict[str, Any]]
    tools: list[str]
    allowed: list[str]
    max_tokens: int


class FakeLLMExhausted(AssertionError):
    """The test script ran out of steps (a test bug, deliberately not an LLMError)."""


ScriptItem = Any  # Step | (name, args) | [(name, args), ...] | str | BaseException | callable


class FakeLLM:
    """Scripted stand-in for LLMClient with the same `step()` signature.

    `script` is a list of items (consumed in order across roles), a {role: [items]} mapping, or a
    callable `fn(StepRequest) -> item` used for every step. Items:
      ("tool", {args})                one call → validated exactly like a real model answer
      [("a", {...}), ("b", {...})]    several calls in one answer → first used, rest dropped_calls
      "free text"                     no tool call → NO_TOOL_CALL
      Step(...)                       returned as-is (full control, e.g. custom token counts)
      LLMError(...) / any exception   raised
      callable(StepRequest)           returns one of the above
    Unknown tools, disallowed tools and bad arguments produce the same protocol errors as LLMClient.
    `usage` (optional) sets token/timing fields on every interpreted step.
    """

    protocol = "fake"

    def __init__(self, script: Sequence[ScriptItem] | Mapping[str, Sequence[ScriptItem]] | Callable,
                 *, usage: Mapping[str, float] | None = None) -> None:
        self._fn: Callable | None = None
        self._by_role: dict[str, list[ScriptItem]] | None = None
        self._queue: list[ScriptItem] = []
        if callable(script):
            self._fn = script
        elif isinstance(script, Mapping):
            self._by_role = {role: list(items) for role, items in script.items()}
        else:
            self._queue = list(script)
        self._usage = dict(usage or {})
        self.requests: list[StepRequest] = []

    @property
    def remaining(self) -> int:
        if self._by_role is not None:
            return sum(len(items) for items in self._by_role.values())
        return len(self._queue)

    def step(self, role: str, messages: Sequence[Mapping[str, Any]], tools: Sequence[ToolSpec],
             allowed: Sequence[str], max_tokens: int) -> Step:
        _check_request(tools, allowed)
        _check_messages(messages)
        request = StepRequest(role=role, messages=copy.deepcopy(list(messages)),
                              tools=[t.name for t in tools], allowed=list(allowed), max_tokens=max_tokens)
        self.requests.append(request)
        item = self._next(request)
        if callable(item) and not isinstance(item, (Step, BaseException)):
            item = item(request)
        step = self._materialize(item, tools, allowed)
        step.role = role
        return step

    def _next(self, request: StepRequest) -> ScriptItem:
        if self._fn is not None:
            return self._fn(request)
        queue = self._queue if self._by_role is None else self._by_role.get(request.role, [])
        if not queue:
            raise FakeLLMExhausted(f"FakeLLM script exhausted at request {len(self.requests)} "
                                   f"(role {request.role!r}, allowed {request.allowed})")
        return queue.pop(0)

    def _materialize(self, item: ScriptItem, tools: Sequence[ToolSpec], allowed: Sequence[str]) -> Step:
        if isinstance(item, Step):
            return item
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            step = interpret_calls([], tools, allowed)
            step.text = item
        elif isinstance(item, tuple):
            step = interpret_calls([RawCall(*item)], tools, allowed)
        elif isinstance(item, list):
            step = interpret_calls([RawCall(*call) for call in item], tools, allowed)
        else:
            raise TypeError(f"unsupported FakeLLM script item: {item!r}")
        for name, value in self._usage.items():
            setattr(step, name, value)
        step.http_attempts = 1
        return step
