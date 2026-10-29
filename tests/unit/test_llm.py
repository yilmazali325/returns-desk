"""Unit tests for desk/llm.py: schema checks, validator, both wire protocols, HTTP errors, FakeLLM.

HTTP is mocked with httpx.MockTransport; no llama-server is needed.
"""

from __future__ import annotations

import json

import httpx
import pytest

from desk.llm import (
    FakeLLM,
    FakeLLMExhausted,
    LLMClient,
    LLMError,
    RawCall,
    Step,
    ToolSpec,
    assistant_message,
    check_schema,
    constrained_messages,
    encode_constrained_call,
    flatten_one_of,
    interpret_calls,
    native_messages,
    parse_constrained_content,
    render_tool_docs,
    response_format,
    tool_message,
    validate,
)

S = {"type": "string"}


def obj(props, required=()):
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


HANDOFF = ToolSpec("handoff_to_operations", "Hand off.", {"type": "object", "oneOf": [
    obj({"intent": {"const": "propose_refund"}, "order_id": S, "line_id": S}, ["intent", "order_id", "line_id"]),
    obj({"intent": {"const": "propose_exchange"}, "order_id": S, "line_id": S, "requested_size": S},
        ["intent", "order_id", "line_id", "requested_size"]),
    obj({"intent": {"const": "execute_confirmed"}}, ["intent"]),
]})
REPLY = ToolSpec("reply_to_customer", "Reply and end the turn.", obj({"message": S}, ["message"]), terminal=True)
PROPOSE = ToolSpec("create_proposal", "Propose.", obj(
    {"remedy": {"type": "string", "enum": ["refund", "exchange"]}, "replacement_sku": {"type": ["string", "null"]}},
    ["remedy"]))
TOOLS = [HANDOFF, PROPOSE, REPLY]
ALL = [t.name for t in TOOLS]


# --------------------------------------------------------------------------------------------------
# ToolSpec / check_schema

@pytest.mark.parametrize("name", ["", "has space", "dash-name"])
def test_toolspec_rejects_bad_names(name):
    with pytest.raises(ValueError, match="plain identifier"):
        ToolSpec(name, "d", obj({}))


def test_toolspec_requires_object_parameters():
    with pytest.raises(ValueError, match="type object"):
        ToolSpec("t", "d", {"type": "string"})


@pytest.mark.parametrize("schema, message", [
    (obj({"a": {"type": "string", "pattern": "^O"}}), "unsupported schema keywords"),
    ({"type": "object", "properties": {"a": S}}, "additionalProperties: false"),
    ({"type": "object", "properties": {"a": S}, "additionalProperties": True}, "additionalProperties: false"),
    (obj({"a": S}, ["b"]), "required names not in properties"),
    (obj({"a": {"type": "text"}}), "unknown type"),
    (obj({"a": {"type": ["string", "date"]}}), "unknown type"),
    (obj({"a": {"type": "object"}}), "need properties"),
    ({"type": "object", "oneOf": [{"type": "object", "properties": {"a": S}}]}, "additionalProperties"),
    (obj({"a": {"type": "array", "items": {"type": "string", "format": "x"}}}), "unsupported"),
    (obj({"a": "string"}), "schema must be an object"),
])
def test_check_schema_rejects_unenforced_or_open_schemas(schema, message):
    with pytest.raises(ValueError, match=message):
        check_schema(schema)


def test_toolspec_openai_tool_and_doc_line():
    tool = REPLY.openai_tool()
    assert tool == {"type": "function", "function": {
        "name": "reply_to_customer", "description": "Reply and end the turn.", "parameters": REPLY.parameters}}
    assert json.loads(REPLY.doc_line()) == {"name": "reply_to_customer", "description": "Reply and end the turn.",
                                            "parameters": REPLY.parameters}
    assert REPLY.terminal and not PROPOSE.terminal


# --------------------------------------------------------------------------------------------------
# validate()

@pytest.mark.parametrize("schema, value, ok", [
    ({"type": "string"}, "x", True),
    ({"type": "string"}, 3, False),
    ({"type": "integer"}, 3, True),
    ({"type": "integer"}, True, False),  # bool is not an integer in JSON schema
    ({"type": "integer"}, 3.5, False),
    ({"type": "number"}, 3.5, True),
    ({"type": "number"}, False, False),
    ({"type": "boolean"}, False, True),
    ({"type": "null"}, None, True),
    ({"type": ["string", "null"]}, None, True),
    ({"type": ["string", "null"]}, 1, False),
    ({"type": "array", "items": {"type": "integer"}}, [1, 2], True),
    ({"type": "array", "items": {"type": "integer"}}, [1, "2"], False),
    ({"enum": ["a", "b"]}, "c", False),
    ({"const": "x"}, "x", True),
    ({"const": "x"}, "y", False),
])
def test_validate_types_enums_consts(schema, value, ok):
    assert (validate(schema, value) == []) is ok


def test_validate_object_reports_every_problem():
    errors = validate(PROPOSE.parameters, {"replacement_sku": 7, "customer_id": "C1"})
    assert errors == [
        "args: missing required property 'remedy'",
        "args: unexpected property 'customer_id'",
        "args.replacement_sku: expected string or null, got int",
    ]


def test_validate_enum_message():
    assert validate(PROPOSE.parameters, {"remedy": "gift"}) == [
        "args.remedy: 'gift' is not one of ['refund', 'exchange']"]


def test_validate_one_of_uses_the_tag_to_pick_the_branch():
    schema = HANDOFF.parameters
    assert validate(schema, {"intent": "propose_refund", "order_id": "ORD-7Q", "line_id": "LN-3"}) == []
    assert validate(schema, {"intent": "propose_refund", "order_id": "ORD-7Q"}) == [
        "args: missing required property 'line_id'"]
    assert validate(schema, {"intent": "execute_confirmed", "order_id": "ORD-7Q"}) == [
        "args: unexpected property 'order_id'"]
    assert validate(schema, {"intent": "cancel"}) == [
        "args.intent: 'cancel' is not one of ['propose_refund', 'propose_exchange', 'execute_confirmed']"]
    assert validate(schema, "not an object") == ["args: expected object, got str"]


def test_validate_one_of_without_tag_needs_exactly_one_match():
    schema = {"oneOf": [{"type": "string"}, {"type": "integer"}, {"type": ["integer", "null"]}]}
    assert validate(schema, "x") == []
    assert validate(schema, 3) == ["args: must match exactly one oneOf branch (matched 2)"]
    assert validate(schema, 2.5) == ["args: must match exactly one oneOf branch (matched 0)"]


def test_flatten_one_of_for_native_protocol():
    flat = flatten_one_of(HANDOFF.parameters)
    assert list(flat["properties"]) == ["intent", "order_id", "line_id", "requested_size"]
    assert flat["properties"]["intent"]["enum"] == ["propose_refund", "propose_exchange", "execute_confirmed"]
    assert "propose_exchange: order_id, line_id, requested_size" in flat["properties"]["intent"]["description"]
    assert "execute_confirmed: nothing else" in flat["properties"]["intent"]["description"]
    assert flat["required"] == ["intent"] and flat["additionalProperties"] is False
    check_schema(flat)
    assert HANDOFF.openai_tool()["function"]["parameters"] == flat
    assert flatten_one_of(REPLY.parameters) is REPLY.parameters  # flat schemas pass through
    untagged = {"type": "object", "oneOf": [obj({"a": S}), obj({"b": S})]}
    assert flatten_one_of(untagged) is untagged


# --------------------------------------------------------------------------------------------------
# interpret_calls / constrained parsing

def test_interpret_valid_call():
    step = interpret_calls([RawCall("create_proposal", '{"remedy": "refund"}')], TOOLS, ALL)
    assert step.ok and step.tool == "create_proposal" and step.args == {"remedy": "refund"}
    assert step.dropped_calls == []


def test_interpret_no_call():
    step = interpret_calls([], TOOLS, ALL)
    assert not step.ok and step.tool is None and step.protocol_error.startswith("NO_TOOL_CALL")


def test_interpret_two_calls_keeps_first_and_records_dropped():
    step = interpret_calls([RawCall("reply_to_customer", {"message": "hi"}),
                            RawCall("create_proposal", '{"remedy": "refund"}'),
                            RawCall("create_proposal", "{not json")], TOOLS, ALL)
    assert step.ok and step.tool == "reply_to_customer"
    assert step.dropped_calls == [{"tool": "create_proposal", "args": {"remedy": "refund"}},
                                  {"tool": "create_proposal", "args": None}]


def test_interpret_unknown_tool():
    step = interpret_calls([RawCall("issue_refund_now", "{}")], TOOLS, ALL)
    assert step.tool is None and step.protocol_error.startswith("UNKNOWN_TOOL: 'issue_refund_now'")
    step = interpret_calls([RawCall(None, "{}")], TOOLS, ALL)
    assert step.protocol_error.startswith("UNKNOWN_TOOL: None")


def test_interpret_tool_not_allowed_now():
    step = interpret_calls([RawCall("create_proposal", '{"remedy": "refund"}')], TOOLS, ["reply_to_customer"])
    assert step.tool == "create_proposal"
    assert step.protocol_error == ("TOOL_NOT_ALLOWED: create_proposal is not allowed now "
                                   "(allowed now: reply_to_customer)")


def test_interpret_bad_json_and_bad_args():
    bad_json = interpret_calls([RawCall("create_proposal", '{"remedy": ')], TOOLS, ALL)
    assert bad_json.protocol_error.startswith("INVALID_JSON")
    not_object = interpret_calls([RawCall("create_proposal", "[1, 2]")], TOOLS, ALL)
    assert not_object.protocol_error == "INVALID_ARGS: args: expected object, got list"
    bad_args = interpret_calls([RawCall("create_proposal", {"remedy": "gift", "x": 1})], TOOLS, ALL)
    assert bad_args.protocol_error == ("INVALID_ARGS: args: unexpected property 'x'; "
                                       "args.remedy: 'gift' is not one of ['refund', 'exchange']")
    assert bad_args.args == {"remedy": "gift", "x": 1}  # kept for the audit / feedback


def test_interpret_empty_argument_string_means_no_arguments():
    step = interpret_calls([RawCall("handoff_to_operations", "  ")], TOOLS, ALL)
    assert step.protocol_error.startswith("INVALID_ARGS: args.intent")


def test_parse_constrained_content():
    calls = parse_constrained_content('{"tool": "reply_to_customer", "args": {"message": "hi"}}', "stop", 160)
    assert calls == [RawCall("reply_to_customer", {"message": "hi"})]
    assert parse_constrained_content('{"tool": "x"}', "stop", 160) == [RawCall("x", {})]
    assert parse_constrained_content('{"tool": "reply_to_customer", "args": {"mess', "length", 160) == (
        "TRUNCATED: output hit max_tokens=160 before the JSON closed")
    assert parse_constrained_content("hello", "stop", 160).startswith("INVALID_JSON: answer is not JSON")
    assert parse_constrained_content('{"args": {}}', "stop", 160).startswith("INVALID_JSON: expected")
    assert parse_constrained_content("[1]", "stop", 160).startswith("INVALID_JSON: expected")


# --------------------------------------------------------------------------------------------------
# Wire formats

def test_tool_docs_are_byte_stable_and_complete():
    first, second = render_tool_docs(TOOLS), render_tool_docs(list(TOOLS))
    assert first == second
    for tool in TOOLS:
        assert tool.doc_line() in first
    assert first.startswith("# Tools\n") and '{"tool": "<tool name>"' in first


def test_response_format_one_branch_per_allowed_tool_in_tool_order():
    rf = response_format(TOOLS, ["reply_to_customer", "handoff_to_operations"])
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    branches = rf["json_schema"]["schema"]["oneOf"]
    assert [b["properties"]["tool"]["const"] for b in branches] == ["handoff_to_operations", "reply_to_customer"]
    assert all(list(b["properties"]) == ["tool", "args"] and b["required"] == ["tool", "args"] for b in branches)
    assert branches[0]["properties"]["args"] == HANDOFF.parameters  # the strict union, not flattened
    single = response_format(TOOLS, ["reply_to_customer"])["json_schema"]["schema"]
    assert single["properties"]["tool"] == {"const": "reply_to_customer"} and "oneOf" not in single


CONVERSATION = [
    {"role": "system", "content": "You are the Resolution agent."},
    {"role": "user", "content": "Refund LN-3 of ORD-7Q."},
    {"role": "assistant", "content": "", "tool_name": "create_proposal", "tool_args": {"remedy": "refund"}},
    {"role": "tool", "content": '{"ok":true}', "tool_name": "create_proposal"},
    {"role": "assistant", "tool_name": "reply_to_customer", "tool_args": {"message": "Done?"}},
    {"role": "tool", "content": '{"ok":true}', "tool_name": "reply_to_customer"},
]


def test_native_messages_use_tool_calls_with_deterministic_ids():
    out = native_messages(CONVERSATION)
    assert out[0] == {"role": "system", "content": "You are the Resolution agent."}
    assert out[2]["tool_calls"] == [{"id": "call_1", "type": "function", "function": {
        "name": "create_proposal", "arguments": '{"remedy": "refund"}'}}]
    assert out[3] == {"role": "tool", "tool_call_id": "call_1", "content": '{"ok":true}'}
    assert out[4]["tool_calls"][0]["id"] == "call_2" and out[4]["content"] == ""
    assert out[5]["tool_call_id"] == "call_2"
    assert native_messages(CONVERSATION) == out


def test_constrained_messages_put_docs_in_system_and_replay_calls_as_json():
    out = constrained_messages(CONVERSATION, TOOLS)
    assert out[0]["content"] == "You are the Resolution agent.\n\n" + render_tool_docs(TOOLS)
    assert out[2] == {"role": "assistant", "content": '{"tool": "create_proposal", "args": {"remedy": "refund"}}'}
    assert out[3] == {"role": "tool", "tool_call_id": "call_1", "content": '{"ok":true}'}
    assert all("tool_calls" not in m for m in out)
    no_system = constrained_messages(CONVERSATION[1:2], TOOLS)
    assert no_system[0] == {"role": "system", "content": render_tool_docs(TOOLS)}


def test_encode_constrained_call_matches_decoder_bytes():
    # Exactly what the decoder emitted live (docs/notes/llm_infra.md), so replay keeps the cache prefix.
    assert encode_constrained_call("handoff_to_operations", {"intent": "propose_refund", "order_id": "ORD-7Q"}) == (
        '{"tool": "handoff_to_operations", "args": {"intent": "propose_refund", "order_id": "ORD-7Q"}}')


def test_assistant_and_tool_message_helpers():
    assert assistant_message(Step(tool="reply_to_customer", args={"message": "hi"})) == {
        "role": "assistant", "content": "", "tool_name": "reply_to_customer", "tool_args": {"message": "hi"}}
    assert assistant_message(Step(tool=None, text="free text")) == {"role": "assistant", "content": "free text"}
    assert tool_message("t", {"b": 1, "a": "é"}) == {"role": "tool", "content": '{"a":"é","b":1}', "tool_name": "t"}
    assert tool_message("t", "raw")["content"] == "raw"


@pytest.mark.parametrize("messages, message", [
    ([{"role": "robot", "content": "x"}], "unknown role"),
    ([{"role": "user", "content": "x"}, {"role": "system", "content": "y"}], "system message must come first"),
    ([{"role": "user", "content": 3}], "content must be a string"),
    ([{"role": "user"}], "content must be a string"),
])
def test_message_validation(messages, message):
    with pytest.raises(ValueError, match=message):
        LLMClient("http://x", "native").build_payload(messages, TOOLS, ALL, 10)


def test_request_validation():
    client = LLMClient("http://x", "constrained")
    with pytest.raises(ValueError, match="non-empty subset"):
        client.build_payload(CONVERSATION[:2], TOOLS, [], 10)
    with pytest.raises(ValueError, match="non-empty subset"):
        client.build_payload(CONVERSATION[:2], TOOLS, ["delete_everything"], 10)
    with pytest.raises(ValueError, match="duplicate tool names"):
        client.build_payload(CONVERSATION[:2], [REPLY, REPLY], ["reply_to_customer"], 10)
    with pytest.raises(ValueError, match="unknown protocol"):
        LLMClient("http://x", "xml")


# --------------------------------------------------------------------------------------------------
# LLMClient over a mocked HTTP transport

TIMINGS = {"cache_n": 671, "prompt_n": 147, "prompt_ms": 169.6, "predicted_n": 40, "predicted_ms": 541.8}


def chat_response(message, finish="stop", timings=TIMINGS, usage=None):
    body = {"choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", **message}}]}
    if timings is not None:
        body["timings"] = timings
    if usage is not None:
        body["usage"] = usage
    return body


class Server:
    """Scripted MockTransport handler: each entry is a response body, an int status, or an exception."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        self.requests.append(json.loads(request.content))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, int):
            return httpx.Response(reply, text="grammar error" if reply < 500 else "loading")
        if isinstance(reply, str):
            return httpx.Response(200, text=reply)
        return httpx.Response(200, json=reply)


def client_for(server, protocol, **kw):
    return LLMClient("http://127.0.0.1:8401/", protocol, transport=httpx.MockTransport(server),
                     retry_backoff_s=0, **kw)


def test_native_step_payload_and_parsing():
    server = Server(chat_response({"content": "", "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "create_proposal", "arguments": '{"remedy":"refund"}'}},
        {"id": "y", "type": "function", "function": {"name": "reply_to_customer", "arguments": '{"message":"ok"}'}},
    ]}, finish="tool_calls"))
    with client_for(server, "native") as client:
        step = client.step("operations", CONVERSATION[:2], TOOLS, ["create_proposal"], 160)
    sent = server.requests[0]
    assert sent["tools"] == [t.openai_tool() for t in TOOLS]
    assert sent["tool_choice"] == "required" and sent["parallel_tool_calls"] is False
    assert "response_format" not in sent
    assert sent["temperature"] == 0 and sent["seed"] == 42 and sent["cache_prompt"] is True
    assert sent["max_tokens"] == 160 and sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert step.ok and step.tool == "create_proposal" and step.args == {"remedy": "refund"}
    assert step.dropped_calls == [{"tool": "reply_to_customer", "args": {"message": "ok"}}]
    assert step.role == "operations" and step.finish_reason == "tool_calls" and step.http_attempts == 1
    assert (step.prompt_tokens, step.cached_tokens, step.prefill_tokens) == (818, 671, 147)
    assert (step.completion_tokens, step.prompt_ms, step.predicted_ms) == (40, 169.6, 541.8)
    assert step.latency_ms >= 0


def test_native_step_without_tool_call():
    server = Server(chat_response({"content": "Sure, I can help!"}))
    step = client_for(server, "native").step("resolution", CONVERSATION[:2], TOOLS, ALL, 160)
    assert step.protocol_error.startswith("NO_TOOL_CALL") and step.text == "Sure, I can help!"
    assert assistant_message(step) == {"role": "assistant", "content": "Sure, I can help!"}


def test_native_invalid_args_from_unenforced_enum():
    server = Server(chat_response({"tool_calls": [
        {"function": {"name": "create_proposal", "arguments": '{"remedy": "store_credit"}'}}]}))
    step = client_for(server, "native").step("operations", CONVERSATION[:2], TOOLS, ALL, 160)
    assert step.protocol_error.startswith("INVALID_ARGS: args.remedy")


def test_constrained_step_payload_and_parsing():
    content = '{"tool": "handoff_to_operations", "args": {"intent": "execute_confirmed"}}'
    server = Server(chat_response({"content": content}))
    step = client_for(server, "constrained", seed=7).step(
        "resolution", CONVERSATION[:2], TOOLS, ["handoff_to_operations", "reply_to_customer"], 160)
    sent = server.requests[0]
    assert "tools" not in sent and "tool_choice" not in sent
    assert sent["seed"] == 7
    assert sent["messages"][0]["content"].endswith(render_tool_docs(TOOLS))
    assert sent["response_format"] == response_format(TOOLS, ["handoff_to_operations", "reply_to_customer"])
    assert step.ok and step.args == {"intent": "execute_confirmed"} and step.text == content


def test_constrained_tools_block_is_identical_across_steps_with_different_allowed():
    server = Server(*[chat_response({"content": '{"tool": "reply_to_customer", "args": {"message": "x"}}'})] * 2)
    client = client_for(server, "constrained")
    client.step("resolution", CONVERSATION[:2], TOOLS, ALL, 160)
    client.step("resolution", CONVERSATION[:2], TOOLS, ["reply_to_customer"], 160)
    first, second = server.requests
    assert first["messages"] == second["messages"]  # cache-friendly prefix
    assert first["response_format"] != second["response_format"]  # restriction lives in sampling only


def test_constrained_truncated_and_garbage_answers():
    server = Server(chat_response({"content": '{"tool": "reply_to_customer", "args": {"message": "Your re'},
                                  finish="length"),
                    chat_response({"content": "I will refund you."}))
    client = client_for(server, "constrained")
    truncated = client.step("resolution", CONVERSATION[:2], TOOLS, ALL, 32)
    assert truncated.protocol_error == "TRUNCATED: output hit max_tokens=32 before the JSON closed"
    assert truncated.tool is None and truncated.finish_reason == "length"
    garbage = client.step("resolution", CONVERSATION[:2], TOOLS, ALL, 32)
    assert garbage.protocol_error.startswith("INVALID_JSON")


def test_usage_fallback_when_timings_are_missing():
    usage = {"prompt_tokens": 500, "completion_tokens": 12, "prompt_tokens_details": {"cached_tokens": 400}}
    server = Server(chat_response({"content": '{"tool": "reply_to_customer", "args": {"message": "x"}}'},
                                  timings=None, usage=usage),
                    chat_response({"content": '{"tool": "reply_to_customer", "args": {"message": "x"}}'},
                                  timings=None))
    client = client_for(server, "constrained")
    step = client.step("resolution", CONVERSATION[:2], TOOLS, ALL, 32)
    assert (step.prompt_tokens, step.cached_tokens, step.completion_tokens) == (500, 400, 12)
    assert step.prompt_ms == 0.0 and step.predicted_ms == 0.0
    bare = client.step("resolution", CONVERSATION[:2], TOOLS, ALL, 32)
    assert (bare.prompt_tokens, bare.cached_tokens, bare.completion_tokens) == (0, 0, 0)


def test_reasoning_content_is_refused():
    server = Server(chat_response({"content": "", "reasoning_content": "Let me think..."}))
    with pytest.raises(LLMError) as err:
        client_for(server, "constrained").step("resolution", CONVERSATION[:2], TOOLS, ALL, 32)
    assert err.value.kind == "reasoning"


def test_retry_once_after_connection_error_then_succeed():
    ok = chat_response({"content": '{"tool": "reply_to_customer", "args": {"message": "x"}}'})
    server = Server(httpx.ConnectError("refused"), ok)
    step = client_for(server, "constrained").step("resolution", CONVERSATION[:2], TOOLS, ALL, 32)
    assert step.ok and step.http_attempts == 2 and len(server.requests) == 2


def test_retry_once_after_5xx_then_succeed_with_backoff(monkeypatch):
    slept = []
    monkeypatch.setattr("desk.llm.time.sleep", slept.append)
    ok = chat_response({"content": '{"tool": "reply_to_customer", "args": {"message": "x"}}'})
    server = Server(503, ok)
    client = LLMClient("http://h", "constrained", transport=httpx.MockTransport(server), retry_backoff_s=0.25)
    assert client.step("resolution", CONVERSATION[:2], TOOLS, ALL, 32).http_attempts == 2
    assert slept == [0.25]


@pytest.mark.parametrize("failures, kind, status", [
    ([httpx.ConnectError("refused"), httpx.ConnectError("refused")], "connection", None),
    ([httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow")], "timeout", None),
    ([500, 502], "http_status", 502),
])
def test_gives_up_after_one_retry(failures, kind, status):
    server = Server(*failures)
    with pytest.raises(LLMError) as err:
        client_for(server, "native").step("resolution", CONVERSATION[:2], TOOLS, ALL, 32)
    assert err.value.kind == kind and err.value.status_code == status and err.value.attempts == 2
    assert len(server.requests) == 2


def test_no_retries_configured():
    server = Server(httpx.ConnectError("refused"))
    with pytest.raises(LLMError) as err:
        client_for(server, "native", retries=0).step("resolution", CONVERSATION[:2], TOOLS, ALL, 32)
    assert err.value.attempts == 1


def test_client_errors_are_not_retried():
    server = Server(400)
    with pytest.raises(LLMError) as err:
        client_for(server, "constrained").step("resolution", CONVERSATION[:2], TOOLS, ALL, 32)
    assert err.value.kind == "http_status" and err.value.status_code == 400 and "grammar" in str(err.value)
    assert len(server.requests) == 1


@pytest.mark.parametrize("reply", ["<html>not json</html>", {"choices": []}, {"error": "x"}, {"choices": [{}]}])
def test_malformed_responses(reply):
    with pytest.raises(LLMError) as err:
        client_for(Server(reply), "native").step("resolution", CONVERSATION[:2], TOOLS, ALL, 32)
    assert err.value.kind == "bad_response"


def test_step_telemetry_has_numbers_only():
    step = Step(tool="t", role="resolution", prompt_tokens=10, cached_tokens=4, completion_tokens=3,
                prompt_ms=1.5, predicted_ms=2.5, latency_ms=5.0, finish_reason="stop", http_attempts=1,
                text="secret model text", dropped_calls=[{"tool": "x", "args": {}}])
    assert step.telemetry() == {
        "role": "resolution", "tool": "t", "protocol_error": None, "dropped_calls": 1, "prompt_tokens": 10,
        "cached_tokens": 4, "prefill_tokens": 6, "completion_tokens": 3, "prompt_ms": 1.5,
        "predicted_ms": 2.5, "latency_ms": 5.0, "finish_reason": "stop", "http_attempts": 1}
    assert Step(tool=None, prompt_tokens=1, cached_tokens=5).prefill_tokens == 0


# --------------------------------------------------------------------------------------------------
# FakeLLM

def test_fake_llm_list_script_runs_real_validation():
    fake = FakeLLM([
        ("create_proposal", {"remedy": "refund"}),
        [("reply_to_customer", {"message": "a"}), ("create_proposal", {"remedy": "refund"})],  # two calls
        ("issue_refund_now", {}),  # unknown tool
        ("create_proposal", {"remedy": "gift"}),  # bad args
        ("create_proposal", {"remedy": "refund"}),  # not allowed at this step
        "I already refunded you!",  # free text, no call
    ])
    steps = [fake.step("operations", CONVERSATION[:2], TOOLS, ALL, 160) for _ in range(4)]
    steps.append(fake.step("operations", CONVERSATION[:2], TOOLS, ["reply_to_customer"], 160))
    steps.append(fake.step("operations", CONVERSATION[:2], TOOLS, ALL, 160))
    assert steps[0].ok and steps[0].args == {"remedy": "refund"} and steps[0].role == "operations"
    assert steps[1].tool == "reply_to_customer" and steps[1].dropped_calls[0]["tool"] == "create_proposal"
    assert steps[2].protocol_error.startswith("UNKNOWN_TOOL")
    assert steps[3].protocol_error.startswith("INVALID_ARGS")
    assert steps[4].protocol_error.startswith("TOOL_NOT_ALLOWED")
    assert steps[5].protocol_error.startswith("NO_TOOL_CALL") and steps[5].text == "I already refunded you!"
    assert fake.remaining == 0
    with pytest.raises(FakeLLMExhausted, match="exhausted at request 7"):
        fake.step("operations", CONVERSATION[:2], TOOLS, ALL, 160)


def test_fake_llm_per_role_script_and_request_log():
    fake = FakeLLM({"resolution": [("reply_to_customer", {"message": "hi"})],
                    "operations": [("create_proposal", {"remedy": "exchange", "replacement_sku": None})]})
    assert fake.remaining == 2
    messages = [dict(m) for m in CONVERSATION[:2]]
    ops = fake.step("operations", messages, TOOLS, ALL, 96)
    messages[1]["content"] = "mutated after the call"
    res = fake.step("resolution", messages, TOOLS, ["reply_to_customer"], 220)
    assert ops.ok and res.ok and fake.remaining == 0
    assert [r.role for r in fake.requests] == ["operations", "resolution"]
    assert fake.requests[0].messages[1]["content"] == "Refund LN-3 of ORD-7Q."  # deep copy
    assert fake.requests[1].allowed == ["reply_to_customer"] and fake.requests[1].max_tokens == 220
    assert fake.requests[0].tools == ALL
    with pytest.raises(FakeLLMExhausted):
        fake.step("resolution", messages, TOOLS, ALL, 10)


def test_fake_llm_callable_script_items_steps_and_errors():
    seen = []

    def brain(request):
        seen.append(request.allowed)
        if request.allowed == ["reply_to_customer"]:
            return ("reply_to_customer", {"message": "only option"})
        return lambda req: ("create_proposal", {"remedy": "refund"})  # nested callable item

    fake = FakeLLM(brain, usage={"prompt_tokens": 100, "cached_tokens": 80, "completion_tokens": 9})
    a = fake.step("operations", CONVERSATION[:2], TOOLS, ALL, 10)
    b = fake.step("resolution", CONVERSATION[:2], TOOLS, ["reply_to_customer"], 10)
    assert a.tool == "create_proposal" and b.tool == "reply_to_customer"
    assert (a.prompt_tokens, a.cached_tokens, a.completion_tokens, a.http_attempts) == (100, 80, 9, 1)
    assert seen == [ALL, ["reply_to_customer"]] and fake.remaining == 0

    custom = Step(tool="reply_to_customer", args={"message": "verbatim"}, protocol_error=None, latency_ms=12.0)
    scripted = FakeLLM([custom, LLMError("timeout", "slow"), object()])
    assert scripted.step("resolution", CONVERSATION[:2], TOOLS, ALL, 10) is custom
    with pytest.raises(LLMError, match="timeout"):
        scripted.step("resolution", CONVERSATION[:2], TOOLS, ALL, 10)
    with pytest.raises(TypeError, match="unsupported FakeLLM script item"):
        scripted.step("resolution", CONVERSATION[:2], TOOLS, ALL, 10)


def test_fake_llm_validates_requests_like_the_real_client():
    fake = FakeLLM([("reply_to_customer", {"message": "x"})])
    with pytest.raises(ValueError):
        fake.step("resolution", CONVERSATION[:2], TOOLS, ["not_a_tool"], 10)
    with pytest.raises(ValueError):
        fake.step("resolution", [{"role": "boss", "content": "x"}], TOOLS, ALL, 10)
    assert fake.protocol == "fake" and fake.remaining == 1
