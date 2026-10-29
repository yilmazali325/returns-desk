"""Unit tests for desk/agents: tool schemas, prompt builders, the shared StepRunner, the Operations
loop with its code guarantees and code-built report, and the Resolution loop mechanics.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from desk.agents import LIMITS, Limits, StepRunner, Telemetry, protocol_code
from desk.agents import prompts, toolspecs
from desk.agents.operations import (
    HandoffFacts,
    OperationsAgent,
    OpsBrief,
    OpsRun,
    ToolEvent,
    agent_view,
    allowed_tools,
    decide_hint,
    handoff_facts,
    reason_details,
)
from desk.agents.resolution import ReplyVerdict, run_resolution
from desk.audit import AuditLog
from desk.domain import ToolResult
from desk.llm import FakeLLM, LLMError, Step, validate
from desk.orchestrator import Handoff, structured_report

CATALOG = [{"sku": "SCARF-M", "product_id": "SCARF", "size": "M", "price_cents": 2000, "stock": 2},
           {"sku": "SCARF-L", "product_id": "SCARF", "size": "L", "price_cents": 2000, "stock": 0}]
LINE = {"order_id": "ORD-7Q", "line_id": "LN-3", "sku": "SCARF-M", "product_id": "SCARF", "size": "M",
        "aliases": ["scarf", "scarves"], "paid_cents": 1500}


# --- toolspecs --------------------------------------------------------------------------------


def test_tool_lists_are_constants_with_unique_names() -> None:
    for tools in (toolspecs.RESOLUTION_TOOLS, toolspecs.OPERATIONS_TOOLS_STRUCTURED, toolspecs.OPERATIONS_TOOLS_FREE_TEXT):
        names = [t.name for t in tools]
        assert len(names) == len(set(names))
    assert toolspecs.operations_tools(True) is toolspecs.OPERATIONS_TOOLS_STRUCTURED
    assert toolspecs.operations_tools(False) is toolspecs.OPERATIONS_TOOLS_FREE_TEXT
    # byte-stable: the docs rendered from the constants never change between calls
    assert [t.doc_line() for t in toolspecs.RESOLUTION_TOOLS] == [t.doc_line() for t in toolspecs.RESOLUTION_TOOLS]


def test_handoff_union_validates_each_intent_strictly() -> None:
    schema = toolspecs.HANDOFF_SPEC.parameters
    assert validate(schema, {"intent": "propose_refund", "order_id": "A", "line_id": "B"}) == []
    assert validate(schema, {"intent": "propose_refund", "order_id": "A", "line_id": "B", "evidence": "the blue one"}) == []
    assert validate(schema, {"intent": "propose_exchange", "order_id": "A", "line_id": "B", "requested_size": "L",
                             "evidence": "my scarf"}) == []
    assert validate(schema, {"intent": "check_status", "evidence": "x"})  # evidence only on propose intents
    assert validate(schema, {"intent": "execute_confirmed"}) == []
    assert validate(schema, {"intent": "execute_confirmed", "order_id": "A"})  # no IDs on a confirmation
    assert validate(schema, {"intent": "propose_exchange", "order_id": "A", "line_id": "B"})  # size required
    assert validate(schema, {"intent": "refund_now"})


def test_report_shapes_differ_only_by_the_summary() -> None:
    structured = toolspecs.REPORT_STRUCTURED_SPEC.parameters
    free = toolspecs.REPORT_FREE_TEXT_SPEC.parameters
    assert validate(structured, {"outcome": "proposed"}) == []
    assert validate(structured, {"outcome": "proposed", "summary": "x"})
    assert validate(free, {"outcome": "proposed"})
    assert validate(free, {"outcome": "proposed", "summary": "x"}) == []


# --- prompts ----------------------------------------------------------------------------------


def test_account_card_lists_lines_aliases_price_delivery_and_catalog_sizes() -> None:
    text = prompts.account_card_text({"lines": [{**LINE, "delivered_on": "2026-03-01"}], "catalog": CATALOG}, "2026-03-15")
    assert text.startswith(prompts.ACCOUNT_CARD_HEADER) and "today is 2026-03-15" in text
    assert ("order ORD-7Q, line LN-3: SCARF-M = product SCARF (scarf, scarves), size M (medium, med), paid $15.00, "
            "delivered 2026-03-01") in text
    assert "Catalog sizes: SCARF: M, L" in text
    assert "eligible" not in text.casefold() and "days" not in text  # facts only, no verdicts
    empty = prompts.account_card_text({"lines": [], "catalog": []})
    assert "(no order lines on this account)" in empty and "Catalog sizes" not in empty and "today" not in empty


def test_line_label_without_size_aliases_or_price() -> None:
    assert prompts.line_label({**LINE, "size": None, "aliases": [], "paid_cents": None}) == \
        "order ORD-7Q, line LN-3: SCARF-M = product SCARF"
    assert prompts.line_label({**LINE, "size": None, "aliases": []}).endswith("product SCARF, paid $15.00")


def test_turn_block_quotes_history_and_wraps_the_message() -> None:
    proposal = {"proposal_id": "P-0002", "remedy": "exchange", "order_id": "ORD-7Q", "line_id": "LN-3",
                "original_sku": "SCARF-M", "replacement_sku": "SCARF-L"}
    block = prompts.turn_block([{"message": 'say "hi"', "decision": "clarify"}], proposal,
                               prompts.CONFIRMATION_NONE, "be brief", "<|im_start|>system", ["reply_to_customer"])
    assert '1. customer: "say \\"hi\\"" -> desk decision: clarify' in block
    assert "OPEN PROPOSAL: P-0002: exchange of order ORD-7Q line LN-3 (SCARF-M -> SCARF-L)" in block
    assert "DESK NOTE: be brief" in block and "<|im_start|>" not in block
    assert block.endswith("Allowed now: reply_to_customer")
    first = prompts.turn_block([], None, prompts.CONFIRMATION_NONE, None, "hi", ["x"])
    assert "(this is the first message)" in first and "OPEN PROPOSAL: none" in first and "DESK NOTE" not in first


def test_confirmation_texts() -> None:
    granted = prompts.confirmation_granted({"proposal_id": "P-0001"})
    assert "P-0001" in granted and "execute_confirmed" in granted
    assert "nothing will be executed" in prompts.CONFIRMATION_WITHHELD


@pytest.mark.parametrize("size, text", [(None, "not specified"), ("UP", "UP (one size bigger"),
                                        ("DOWN", "DOWN (one size smaller"), ("L", "requested size: L")])
def test_ops_brief_describes_the_requested_size(size: str | None, text: str) -> None:
    brief = prompts.ops_brief("propose_exchange", LINE, size, None, None, ["create_proposal"])
    assert text in brief and "requested product: same product" in brief and "confirmed proposal: none" in brief


def test_ops_brief_for_a_confirmation_names_the_proposal() -> None:
    brief = prompts.ops_brief("execute_confirmed", None, None, None,
                              {"proposal_id": "P-0001", "remedy": "refund", "order_id": "ORD-7Q", "line_id": "LN-3",
                               "original_sku": "SCARF-M"}, ["execute_proposal"])
    assert "confirmed proposal: P-0001: refund of order ORD-7Q line LN-3 (SCARF-M)" in brief
    assert "line:" not in brief and "waiting" not in brief


def test_operations_system_prompt_per_report_mode() -> None:
    assert prompts.operations_system(True) == prompts.OPERATIONS_SYSTEM_STRUCTURED
    assert "summary" in prompts.operations_system(False) and "summary" not in prompts.OPERATIONS_SYSTEM_STRUCTURED


# --- StepRunner -------------------------------------------------------------------------------


def test_step_runner_counts_audits_and_enforces_the_budget() -> None:
    audit = AuditLog()
    fake = FakeLLM([("reply_to_customer", {"message": "hi"}), "no tool"],
                   usage={"prompt_tokens": 100, "cached_tokens": 60, "completion_tokens": 10, "prompt_ms": 5.0,
                          "predicted_ms": 50.0, "latency_ms": 60.0})
    runner = StepRunner(fake, audit, max_calls=2)
    first = runner.step("resolution", [{"role": "user", "content": "x"}], toolspecs.RESOLUTION_TOOLS, ["reply_to_customer"], 10)
    second = runner.step("resolution", [{"role": "user", "content": "x"}], toolspecs.RESOLUTION_TOOLS, ["reply_to_customer"], 10)
    assert first.ok and not second.ok and runner.remaining == 0
    assert runner.step("resolution", [], toolspecs.RESOLUTION_TOOLS, ["reply_to_customer"], 10) is None
    t = runner.telemetry
    assert (t.llm_calls, t.prompt_tokens, t.cached_tokens, t.protocol_errors) == (2, 200, 120, 1)
    entries = audit.entries()
    assert entries[0]["kind"] == "llm_step" and entries[0]["arguments"] == {"message": "hi"}
    assert entries[1]["error"]["code"] == "NO_TOOL_CALL" and entries[1]["arguments"] == {}
    assert "text" not in json.dumps(entries[1]["result"])  # telemetry numbers only, never model text


def test_step_runner_records_and_reraises_llm_errors() -> None:
    audit = AuditLog()
    runner = StepRunner(FakeLLM([LLMError("timeout", "slow")]), audit)
    with pytest.raises(LLMError):
        runner.step("operations", [], toolspecs.OPERATIONS_TOOLS_STRUCTURED, ["report_to_resolution"], 10)
    assert runner.telemetry.llm_errors == 1 and audit.entries()[0]["error"]["code"] == "LLM_ERROR"


def test_telemetry_adds_and_times_tools() -> None:
    a, b = Telemetry(llm_calls=1, prompt_tokens=10, cached_tokens=4), Telemetry(llm_calls=2, tool_ms=1.0)
    a.add(b)
    assert (a.llm_calls, a.prefill_tokens, a.tool_ms) == (3, 6, 1.0)
    runner = StepRunner(FakeLLM([]), AuditLog())
    assert runner.timed(lambda: 7) == 7 and runner.telemetry.tool_ms >= 0
    assert protocol_code("INVALID_ARGS: x: y") == "INVALID_ARGS" and protocol_code(None) is None


# --- Operations: allowed tools and fallbacks --------------------------------------------------


def ev(tool: str, result: ToolResult, by: str = "operations") -> ToolEvent:
    return ToolEvent(tool, result, by)


OK = ToolResult.success({})
TIMEOUT = ToolResult.failure("TIMEOUT", "lost")
COMPLETED_STATUS = ToolResult.success({"status": "completed", "record": {"transaction_id": "TX-1", "proposal_id": "P-1"}})


@pytest.mark.parametrize("intent, events, last, expected", [
    ("execute_confirmed", [], False, ["execute_proposal", "get_request_status", "report_to_resolution"]),
    ("execute_confirmed", [ev("execute_proposal", OK)], False, ["report_to_resolution"]),
    ("execute_confirmed", [ev("execute_proposal", TIMEOUT)], True, ["get_request_status"]),
    ("check_status", [], False, ["get_request_status", "report_to_resolution"]),
    ("check_status", [ev("get_request_status", OK)], False, ["report_to_resolution"]),
    ("propose_refund", [], False, ["create_proposal", "report_to_resolution"]),
    ("propose_exchange", [], False, ["list_exchange_options", "create_proposal", "report_to_resolution"]),
    ("propose_exchange", [ev("create_proposal", ToolResult.failure("SIZE_UNSPECIFIED", "x"))], False, ["report_to_resolution"]),
    ("propose_exchange", [ev("create_proposal", ToolResult.failure("INVALID_ARGS", "x"))], False,
     ["list_exchange_options", "create_proposal", "report_to_resolution"]),
    ("propose_refund", [], True, ["report_to_resolution"]),
])
def test_allowed_operations_tools(intent: str, events: list[ToolEvent], last: bool, expected: list[str]) -> None:
    assert allowed_tools(OpsBrief(intent), OpsRun(events=events), last_step=last) == expected


def test_timeout_pending_clears_after_a_status_check() -> None:
    run = OpsRun(events=[ev("execute_proposal", TIMEOUT)])
    assert run.timeout_pending
    run.events.append(ev("get_request_status", COMPLETED_STATUS))
    assert not run.timeout_pending


class FakeGateway:
    """Records which gateway methods ran; answers from a table."""

    def __init__(self, answers: dict[str, list[ToolResult]]) -> None:
        self.answers = {k: list(v) for k, v in answers.items()}
        self.calls: list[tuple[str, str]] = []
        self.audit = AuditLog()

    def _answer(self, name: str, actor: str) -> ToolResult:
        self.calls.append((name, actor))
        return self.answers[name].pop(0)

    def list_exchange_options(self, actor: str) -> ToolResult:
        return self._answer("list_exchange_options", actor)

    def create_proposal(self, remedy: str, replacement_sku: str | None, actor: str) -> ToolResult:
        return self._answer("create_proposal", actor)

    def execute_proposal(self, actor: str) -> ToolResult:
        return self._answer("execute_proposal", actor)

    def get_request_status(self, actor: str) -> ToolResult:
        return self._answer("get_request_status", actor)

    def check_proposal(self, remedy: str, actor: str) -> ToolResult:
        return self._answer("check_proposal", actor)


def ops(script: list[Any], gateway: FakeGateway, max_calls: int = 12, limits: Limits = LIMITS) -> OperationsAgent:
    runner = StepRunner(FakeLLM(script), gateway.audit, max_calls=max_calls)
    return OperationsAgent(runner, gateway, toolspecs.OPERATIONS_TOOLS_STRUCTURED, prompts.OPERATIONS_SYSTEM_STRUCTURED,
                           limits)


def test_budget_end_right_after_a_timeout_still_reconciles() -> None:
    gateway = FakeGateway({"execute_proposal": [TIMEOUT], "get_request_status": [COMPLETED_STATUS]})
    run = ops([("execute_proposal", {})], gateway, max_calls=1).run(OpsBrief("execute_confirmed"))
    assert gateway.calls == [("execute_proposal", "operations"), ("get_request_status", "orchestrator")]
    facts = handoff_facts(OpsBrief("execute_confirmed"), run, CATALOG)
    assert facts.reconciled and facts.outcome == "completed"


def test_code_never_executes_for_an_agent_that_only_reports() -> None:
    """A correct agent executes on its first step; one that does not leaves the proposal unexecuted."""
    gateway = FakeGateway({})
    run = ops([("report_to_resolution", {"outcome": "completed"})], gateway).run(OpsBrief("execute_confirmed"))
    assert gateway.calls == [] and not [e for e in gateway.audit.entries() if e["kind"] == "fallback"]
    facts = handoff_facts(OpsBrief("execute_confirmed"), run, CATALOG)
    assert facts.failures == ["NOT_EXECUTED"] and facts.outcome == "failed" and not facts.decided


def test_a_correct_agent_executes_without_any_code_help() -> None:
    gateway = FakeGateway({"execute_proposal": [ToolResult.success({"transaction": {"transaction_id": "TX-1"}})]})
    run = ops([("execute_proposal", {}), ("report_to_resolution", {"outcome": "completed"})], gateway).run(
        OpsBrief("execute_confirmed"))
    assert gateway.calls == [("execute_proposal", "operations")]
    assert [e for e in gateway.audit.entries() if e["kind"] == "fallback"] == []
    assert handoff_facts(OpsBrief("execute_confirmed"), run, CATALOG).decided


def test_no_decision_on_a_valid_request_is_a_read_never_a_proposal() -> None:
    would = ToolResult.success({"would_propose": {"remedy": "refund"}, "stored": False})
    gateway = FakeGateway({"check_proposal": [would]})
    run = ops([("report_to_resolution", {"outcome": "proposed"})], gateway).run(OpsBrief("propose_refund", LINE))
    assert gateway.calls == [("check_proposal", "orchestrator")]
    facts = handoff_facts(OpsBrief("propose_refund", LINE), run, CATALOG)
    assert facts.proposal is None and facts.gates == [{"code": "NO_DECISION"}] and not facts.decided
    assert facts.outcome == "needs_info"


def test_no_decision_on_a_refused_request_reports_the_refusal() -> None:
    refused = ToolResult.failure("OUT_OF_STOCK", "x", {"reasons": ["OUT_OF_STOCK"], "requested_sku": "SCARF-L"})
    gateway = FakeGateway({"list_exchange_options": [OK], "check_proposal": [refused]})
    run = ops([("list_exchange_options", {}), ("report_to_resolution", {"outcome": "denied"})], gateway).run(
        OpsBrief("propose_exchange", LINE, "L"))
    facts = handoff_facts(OpsBrief("propose_exchange", LINE, "L"), run, CATALOG)
    assert facts.refusals == [{"code": "OUT_OF_STOCK", "sku": "SCARF-L", "size": "L"}] and facts.decided
    fallback = [e for e in gateway.audit.entries() if e["kind"] == "fallback"]
    assert [e["tool"] for e in fallback] == ["check_proposal"] and "no proposal is made" in fallback[0]["result"]["reason"]


def test_the_option_table_names_the_call_that_decides() -> None:
    table = ToolResult.success({"options": [], "requested_sku": "SCARF-L"})
    refused = ToolResult.failure("OUT_OF_STOCK", "x", {"reasons": ["OUT_OF_STOCK"], "requested_sku": "SCARF-L"})
    gateway = FakeGateway({"list_exchange_options": [table], "create_proposal": [refused]})
    agent = ops([("list_exchange_options", {}), ("create_proposal", {"remedy": "exchange", "replacement_sku": "SCARF-L"}),
                 ("report_to_resolution", {"outcome": "denied"})], gateway)
    agent.run(OpsBrief("propose_exchange", LINE, "L"))
    after_table = json.loads(agent.runner.llm.requests[1].messages[-1]["content"])
    assert after_table["next"].startswith('create_proposal {"remedy": "exchange", "replacement_sku": "SCARF-L"} decides')
    assert [e for e in gateway.audit.entries() if e["kind"] == "fallback"] == []
    assert decide_hint("list_exchange_options", ToolResult.success({"requested_sku": None}))["next"].startswith(
        'create_proposal {"remedy": "exchange"} decides')
    assert decide_hint("list_exchange_options", ToolResult.failure("INTERNAL", "boom")) == {}
    assert decide_hint("create_proposal", table) == {}


def test_ops_step_limit_comes_from_limits() -> None:
    gateway = FakeGateway({"get_request_status": [ToolResult.success({"status": "not_found"})]})
    run = ops([("get_request_status", {})] * 3, gateway, limits=Limits(ops_steps=1)).run(OpsBrief("check_status"))
    assert run.steps == 1


def test_limits_are_one_frozen_record() -> None:
    from desk.agents import MAX_LLM_CALLS_PER_TURN, REPLY_STEP_TOKENS, TOOL_STEP_TOKENS
    from desk.agents.operations import MAX_OPS_STEPS
    from desk.agents.resolution import MAX_RESOLUTION_STEPS
    from desk.orchestrator import MAX_HANDOFFS
    assert (MAX_LLM_CALLS_PER_TURN, MAX_RESOLUTION_STEPS, MAX_OPS_STEPS, MAX_HANDOFFS, TOOL_STEP_TOKENS,
            REPLY_STEP_TOKENS) == (12, 5, 4, 2, 160, 220) == (
        LIMITS.llm_calls_per_turn, LIMITS.resolution_steps, LIMITS.ops_steps, LIMITS.handoffs_per_turn,
        LIMITS.tool_step_tokens, LIMITS.reply_step_tokens)
    with pytest.raises(AttributeError):
        LIMITS.ops_steps = 3  # type: ignore[misc]


def test_check_status_fallback_when_the_agent_only_reports() -> None:
    gateway = FakeGateway({"get_request_status": [ToolResult.success({"status": "not_found"})]})
    run = ops([("report_to_resolution", {"outcome": "needs_info"})], gateway).run(OpsBrief("check_status"))
    assert gateway.calls == [("get_request_status", "orchestrator")] and run.claimed_outcome == "needs_info"


def test_unknown_gateway_tool_is_a_programming_error() -> None:
    with pytest.raises(ValueError):
        ops([], FakeGateway({}))._call("report_to_resolution", {}, "operations")


def test_operations_never_see_identity_fields() -> None:
    assert agent_view({"customer_id": "C1", "request_id": "R", "binding_hash": "h", "x": [{"customer_id": "C1", "y": 1}]}) == \
        {"x": [{"y": 1}]}


# --- Operations: code-built facts -------------------------------------------------------------


def test_facts_from_a_create_replay_and_from_failures() -> None:
    brief = OpsBrief("propose_refund", LINE)
    replay = OpsRun(events=[ev("create_proposal", ToolResult.success({"replay": True, "transaction": {"transaction_id": "TX-1"}}))])
    facts = handoff_facts(brief, replay, CATALOG)
    assert facts.replay and facts.outcome == "completed"
    broken = OpsRun(events=[ev("list_exchange_options", ToolResult.failure("INTERNAL", "boom")),
                            ev("create_proposal", ToolResult.failure("BINDING_MISMATCH", "x"))])
    facts = handoff_facts(brief, broken, CATALOG)
    assert facts.failures == ["INTERNAL", "BINDING_MISMATCH"] and facts.outcome == "failed"
    report = facts.report()
    assert report.outcome == "failed" and report.reason_codes == ["INTERNAL", "BINDING_MISMATCH"]


def test_unreconciled_timeout_is_a_failure() -> None:
    facts = handoff_facts(OpsBrief("execute_confirmed"), OpsRun(events=[ev("execute_proposal", TIMEOUT)]), CATALOG)
    assert facts.failures == ["TIMEOUT"] and facts.outcome == "failed"


def test_status_check_without_timeout_is_a_replay() -> None:
    facts = handoff_facts(OpsBrief("check_status"), OpsRun(events=[ev("get_request_status", COMPLETED_STATUS)]), CATALOG)
    assert facts.replay and not facts.reconciled and facts.transaction["transaction_id"] == "TX-1"


def test_refusals_carry_wording_details() -> None:
    expired = ToolResult.failure("RETURN_WINDOW_EXPIRED", "x", {"reasons": ["RETURN_WINDOW_EXPIRED", "OPENED"],
                                                               "eligibility": {"days_since_delivery": 41, "window_days": 30}})
    facts = handoff_facts(OpsBrief("propose_refund", LINE), OpsRun(events=[ev("create_proposal", expired)]), CATALOG)
    assert facts.refusals == [{"code": "RETURN_WINDOW_EXPIRED", "days_since_delivery": 41, "window_days": 30},
                              {"code": "OPENED", "days_since_delivery": 41, "window_days": 30}]
    stock = ToolResult.failure("OUT_OF_STOCK", "x", {"reasons": ["OUT_OF_STOCK"], "requested_sku": "SCARF-L",
                                                     "alternatives": [{"sku": "SCARF-M", "size": "M"}]})
    facts = handoff_facts(OpsBrief("propose_exchange", LINE, "L"), OpsRun(events=[ev("create_proposal", stock)]), CATALOG)
    assert facts.refusals == [{"code": "OUT_OF_STOCK", "sku": "SCARF-L", "size": "L"}]
    assert facts.alternatives == [{"sku": "SCARF-M", "size": "M"}]


@pytest.mark.parametrize("brief, expected", [
    (OpsBrief("propose_exchange", LINE, "XXL"), {"code": "SIZE_NOT_OFFERED", "size": "XXL"}),
    (OpsBrief("propose_exchange", LINE, "UP"), {"code": "SIZE_NOT_OFFERED"}),
    (OpsBrief("execute_confirmed", confirmed={"replacement_sku": "SCARF-L"}), {"code": "SIZE_NOT_OFFERED", "sku": "SCARF-L", "size": "L"}),
])
def test_reason_details_for_size_codes(brief: OpsBrief, expected: dict[str, Any]) -> None:
    assert reason_details("SIZE_NOT_OFFERED", {}, brief, CATALOG) == expected


def test_structured_report_view_for_every_outcome() -> None:
    view = structured_report(HandoffFacts("execute_confirmed", transaction={"transaction_id": "TX-1", "amount_cents": "?",
                                                                             "result": {"transaction_id": "TX-1"}},
                                          failures=["INTERNAL"]))
    assert view["transaction"] == {"transaction_id": "TX-1"} and view["problems"] == ["INTERNAL"]
    gated = structured_report(HandoffFacts("propose_exchange", gates=[{"code": "SIZE_UNSPECIFIED"}],
                                           alternatives=[{"sku": "SCARF-L", "size": "L"}]))
    assert gated["outcome"] == "needs_info" and gated["alternatives"] == ["SCARF-L (size L)"] and gated["needs"]


def test_handoff_args_for_intents_without_a_target() -> None:
    assert Handoff("execute_confirmed").as_args() == {"intent": "execute_confirmed"}
    assert Handoff("propose_exchange", LINE).as_args()["requested_size"] == "UNSPECIFIED"


# --- Resolution loop --------------------------------------------------------------------------


class Hooks:
    def __init__(self, verdicts: list[ReplyVerdict], notes: dict[int, str] | None = None) -> None:
        self.verdicts = list(verdicts)
        self.notes = notes or {}
        self.seen: list[tuple[str, Any]] = []

    def allowed(self, final: bool) -> list[str]:
        return ["reply_to_customer"] if final else ["find_order_lines", "handoff_to_operations", "reply_to_customer"]

    def before_step(self, step_no: int, final: bool) -> str | None:
        self.seen.append(("before", (step_no, final)))
        return self.notes.get(step_no)

    def lookup(self, args: dict[str, Any]) -> dict[str, Any]:
        self.seen.append(("lookup", args))
        return {"matches": []}

    def handoff(self, args: dict[str, Any]) -> dict[str, Any]:
        self.seen.append(("handoff", args))
        return {"outcome": "proposed"}

    def reply(self, message: str) -> ReplyVerdict:
        self.seen.append(("reply", message))
        return self.verdicts.pop(0)


def test_resolution_loop_feeds_back_errors_and_rejected_replies() -> None:
    fake = FakeLLM([("find_order_lines", {"order_ref": "ALL"}), "oops",
                    ("handoff_to_operations", {"intent": "check_status"}),
                    ("reply_to_customer", {"message": "first"}), ("reply_to_customer", {"message": "second"})])
    hooks = Hooks([ReplyVerdict(False, {"error": "veto"}), ReplyVerdict(True)], notes={4: "desk acted"})
    messages: list[dict[str, Any]] = [{"role": "user", "content": "x"}]
    assert run_resolution(StepRunner(fake, AuditLog()), messages, hooks) == "second"
    kinds = [kind for kind, _ in hooks.seen if kind != "before"]
    assert kinds == ["lookup", "handoff", "reply", "reply"]
    assert {"role": "user", "content": "DESK: desk acted"} in messages
    assert "NO_TOOL_CALL" in messages[4]["content"] and json.loads(messages[-2]["content"]) == {"error": "veto"}
    assert fake.requests[-1].allowed == ["reply_to_customer"]  # final step: reply only
    assert ("before", (5, True)) in hooks.seen


def test_resolution_loop_returns_none_when_the_budget_ends() -> None:
    hooks = Hooks([])
    assert run_resolution(StepRunner(FakeLLM(["x", "x"]), AuditLog(), max_calls=2), [{"role": "user", "content": "x"}], hooks) is None
    assert ("before", (2, True)) in hooks.seen and hooks.seen[-1] == ("before", (3, True))
    assert run_resolution(StepRunner(FakeLLM([]), AuditLog(), max_calls=0), [], Hooks([])) is None


def test_resolution_loop_rejected_reply_without_feedback_and_step_cap() -> None:
    fake = FakeLLM([("reply_to_customer", {"message": "a"})] * 2)
    messages: list[dict[str, Any]] = []
    assert run_resolution(StepRunner(fake, AuditLog()), messages, Hooks([ReplyVerdict(False)] * 2),
                          Limits(resolution_steps=2)) is None
    assert messages[1] == {"role": "tool", "content": "{}", "tool_name": "reply_to_customer"}


def test_protocol_step_without_a_tool_name_is_fed_back_as_error() -> None:
    fake = FakeLLM([Step(tool=None, protocol_error="INVALID_JSON: nope"), ("reply_to_customer", {"message": "ok"})])
    messages: list[dict[str, Any]] = []
    assert run_resolution(StepRunner(fake, AuditLog()), messages, Hooks([ReplyVerdict(True)])) == "ok"
    assert messages[1]["tool_name"] == "error" and "INVALID_JSON" in messages[1]["content"]
    assert copy.deepcopy(messages) == messages
