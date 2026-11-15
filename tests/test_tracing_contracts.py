"""Unit and negative-control tests for Chapter 1 (llmlab.tracing)."""
from __future__ import annotations

import datetime
import math
import uuid

import pytest

from llmlab.durable.envelope import AgentDirectory, AgentRegistration
from llmlab.durable.idempotency import EffectIdentity, ProviderCapability
from llmlab.tracing import (
    AuthorityContract,
    AuthorityContractViolation,
    DeadlineContract,
    DeadlineContractViolation,
    EffectContract,
    EffectContractViolation,
    FinancialBudget,
    FinancialBudgetExceeded,
    FourBudgetLedger,
    LatencyBudget,
    LatencyBudgetExceeded,
    MemoryBudget,
    MemoryBudgetExceeded,
    ResourceContract,
    ResourceContractViolation,
    SpanKind,
    StateContract,
    StateContractViolation,
    StreamingLatencyBreakdown,
    TemporalContainmentError,
    TokenUsage,
    TraceGraph,
    TraceSpan,
    TrustBudget,
    TrustBudgetExceeded,
)


def _build_healthy_payment_assistant_trace() -> TraceGraph:
    """Construct the nominal 4-stage payment-enabled enterprise assistant trace."""
    trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    ms = 1_000_000  # 1 ms in ns

    root = TraceSpan(
        span_id="span_root",
        trace_id=trace_id,
        parent_id=None,
        name="PaymentAssistantWorkflow",
        kind=SpanKind.AGENT,
        start_ns=0,
        end_ns=1_950 * ms,
    )
    retriever_dense = TraceSpan(
        span_id="span_ret_dense",
        trace_id=trace_id,
        parent_id="span_root",
        name="HybridHNSWRetriever",
        kind=SpanKind.RETRIEVER,
        start_ns=20 * ms,
        end_ns=180 * ms,
    )
    retriever_bm25 = TraceSpan(
        span_id="span_ret_bm25",
        trace_id=trace_id,
        parent_id="span_root",
        name="SparseBM25Retriever",
        kind=SpanKind.RETRIEVER,
        start_ns=20 * ms,
        end_ns=95 * ms,
    )
    # 100 output tokens: 1 first token at 140 ms + 99 decode steps at 12.0 ms each = 1,328 ms
    llm_plan = TraceSpan(
        span_id="span_llm_plan",
        trace_id=trace_id,
        parent_id="span_root",
        name="vLLM-Llama-3-70B-Planner",
        kind=SpanKind.LLM,
        start_ns=190 * ms,
        end_ns=1_518 * ms,
        token_usage=TokenUsage(
            prompt_uncached=512,
            prompt_cached=3_584,
            completion=100,
        ),
        latency_breakdown=StreamingLatencyBreakdown(
            ttft_ms=140.0,
            itl_ms=tuple(12.0 for _ in range(99)),
        ),
    )
    payment_tool = TraceSpan(
        span_id="span_tool_pay",
        trace_id=trace_id,
        parent_id="span_root",
        name="StripeChargeExecution",
        kind=SpanKind.TOOL,
        start_ns=1_530 * ms,
        end_ns=1_910 * ms,
    )
    checkpoint_commit = TraceSpan(
        span_id="span_chk_commit",
        trace_id=trace_id,
        parent_id="span_root",
        name="PostgresCASCommit",
        kind=SpanKind.CHECKPOINT,
        start_ns=1_915 * ms,
        end_ns=1_940 * ms,
    )
    return TraceGraph(
        [root, retriever_dense, retriever_bm25, llm_plan, payment_tool, checkpoint_commit]
    )


def test_healthy_trace_critical_path_and_four_budget_ledger() -> None:
    """Verify exact critical path, concurrency efficiency, and four-budget passage."""
    trace = _build_healthy_payment_assistant_trace()
    report = trace.analyze_critical_path()

    assert report.root_span_id == "span_root"
    assert math.isclose(report.critical_path_ms, 1950.0, rel_tol=1e-9)
    # Leaf spans: 160 + 75 + 1328 + 380 + 25 = 1968 ms
    assert math.isclose(report.total_leaf_work_ms, 1968.0, rel_tol=1e-9)
    assert math.isclose(report.concurrency_efficiency, 1968.0 / 1950.0, rel_tol=1e-9)
    assert report.aggregated_tokens.prompt_total == 4096
    assert math.isclose(report.aggregated_tokens.cache_hit_ratio, 3584 / 4096, rel_tol=1e-9)

    ledger = FourBudgetLedger(
        latency=LatencyBudget(
            max_ttft_ms=250.0,
            max_tpot_ms=20.0,
            max_p99_itl_ms=35.0,
            workflow_deadline_ms=3000.0,
        ),
        memory=MemoryBudget(
            hbm_capacity_bytes=80 * (1024**3),
            weight_bytes=70 * (1024**3),
            kv_bytes_per_token=163_840,  # FP8 GQA 80-layer 8-KV-head 128-dim
            activation_bytes_per_token=65_536,
            system_reserve_bytes=2 * (1024**3),
        ),
        financial=FinancialBudget(
            price_uncached_per_1m=3.00,
            price_cached_per_1m=0.30,
            price_completion_per_1m=15.00,
            max_cost_usd=0.05,
        ),
        trust=TrustBudget(
            max_trajectory_failure_prob=0.10,
            max_unverified_steps=3,
            require_idempotent_external_writes=True,
        ),
    )

    assessment = ledger.evaluate_trace(
        trace,
        resident_kv_tokens=32_768,
        step_tokens=512,
        unverified_step_error_probs=(0.02, 0.02),
        has_unreconciled_external_write=False,
    )
    assert math.isclose(assessment.remaining_deadline_slack_ms, 1050.0, rel_tol=1e-9)
    # Cost: 512 * 3e-6 + 3584 * 0.3e-6 + 100 * 15e-6 = 0.001536 + 0.0010752 + 0.0015 = 0.0041112
    assert math.isclose(assessment.total_cost_usd, 0.0041112, rel_tol=1e-9)
    assert math.isclose(assessment.compounded_failure_prob, 1.0 - (0.98 * 0.98), rel_tol=1e-9)


def test_negative_control_orphaned_child_violates_temporal_containment() -> None:
    """Negative control: an uncancelled payment RPC outliving parent timeout fails validation."""
    ms = 1_000_000
    root = TraceSpan(
        span_id="root",
        trace_id="trace_01",
        parent_id=None,
        name="GatewayRequest",
        kind=SpanKind.AGENT,
        start_ns=0,
        end_ns=3_000 * ms,
        status="TIMEOUT",
    )
    orphaned_payment = TraceSpan(
        span_id="child_payment",
        trace_id="trace_01",
        parent_id="root",
        name="OrphanedStripeCharge",
        kind=SpanKind.TOOL,
        start_ns=2_850 * ms,
        end_ns=3_320 * ms,  # Completes 320 ms AFTER parent timed out at 3,000 ms!
        status="OK",
    )
    graph = TraceGraph([root, orphaned_payment])
    with pytest.raises(TemporalContainmentError, match="orphaned child continued executing"):
        graph.validate_temporal_containment()


def test_negative_control_unchunked_prefill_stalls_decode_and_cascades_across_contracts() -> None:
    """Negative control: unchunked 32K prefill breaches Resource, Latency, Deadline, and Effect contracts."""
    mem_budget = MemoryBudget(
        hbm_capacity_bytes=80 * (1024**3),
        weight_bytes=70 * (1024**3),
        kv_bytes_per_token=163_840,
        activation_bytes_per_token=65_536,
        system_reserve_bytes=2 * (1024**3),
        peak_compute_tflops=989.4,
        peak_hbm_bandwidth_gbps=3350.0,
    )
    resource_contract = ResourceContract(
        memory_budget=mem_budget,
        max_step_tokens=2048,
        max_decode_step_ms=50.0,
        active_parameters=70_000_000_000,
    )

    # 1. Admitting an unchunked 32,768-token prefill alongside 32 active decode streams
    # violates ResourceContract.
    with pytest.raises(ResourceContractViolation, match="exceeds B_tokens=2048"):
        resource_contract.validate_step_admission(
            active_decode_tokens=32,
            prefill_chunk_tokens=32_768,
            resident_kv_tokens=16_384,
        )

    # 2. If the scheduler ignores ResourceContract and runs the unchunked prefill, it injects a
    # mid-stream ITL stall (e.g., 1,450 ms) into active decode streams, violating P99 ITL SLO
    # even when mean TPOT looks acceptable across a long generation.
    stalled_itl = tuple([12.0] * 98 + [1450.0])
    breakdown = StreamingLatencyBreakdown(ttft_ms=140.0, itl_ms=stalled_itl)
    lat_budget = LatencyBudget(
        max_ttft_ms=250.0,
        max_tpot_ms=30.0,  # Mean TPOT is (98*12 + 1450)/99 = 26.53 ms <= 30.0 ms!
        max_p99_itl_ms=50.0,
        workflow_deadline_ms=3000.0,
    )
    assert breakdown.tpot_ms < lat_budget.max_tpot_ms
    with pytest.raises(LatencyBudgetExceeded, match="P99 ITL"):
        lat_budget.check_stream(breakdown)

    # 3. Because the decode stream stalled by 1,450 ms, the workflow reaches the payment tool
    # at t = 2,850 ms against a 3,000 ms deadline (only 150 ms slack < 400 ms required).
    deadline_contract = DeadlineContract(
        workflow_deadline_ns=3_000 * 1_000_000,
        min_external_rpc_budget_ms=400.0,
    )
    with pytest.raises(DeadlineContractViolation, match="Insufficient deadline slack"):
        deadline_contract.verify_dispatch_slack(2_850 * 1_000_000, "StripeChargeExecution")


def test_negative_control_state_effect_and_authority_contract_violations() -> None:
    """Negative control: verify State, Effect, and Authority contracts fail closed on violation."""
    # State Contract: stale fencing token or version mismatch must raise StateContractViolation
    state_contract = StateContract(expected_version=4, fencing_token=11)
    with pytest.raises(StateContractViolation, match="Stale worker fencing token"):
        state_contract.verify_transition(
            head_version=4,
            head_fencing_token=12,
            proposed_version=5,
        )

    # Effect Contract: dispatching external write without Phase 1 INTENT commit must fail closed
    effect_contract = EffectContract(
        secret=b"0123456789abcdef0123456789abcdef",
        capability=ProviderCapability.NON_IDEMPOTENT,
    )
    ident = EffectIdentity(
        workflow_id=str(uuid.uuid4()),
        graph_digest="a" * 64,
        node_id="charge_card",
        logical_attempt=0,
        state_hash=b"\x01" * 32,
    )
    with pytest.raises(EffectContractViolation, match="Phase 1 'INTENT' row must be committed"):
        effect_contract.mint_intent(ident, intent_committed_in_db=False)

    kappa = effect_contract.mint_intent(ident, intent_committed_in_db=True)
    with pytest.raises(EffectContractViolation, match="NON_IDEMPOTENT provider"):
        effect_contract.classify_post_dispatch_timeout(kappa)

    # Authority Contract: untrusted reader agent attempting to invoke payment tool must be denied
    directory = AgentDirectory(
        agents={
            "doc_reader": AgentRegistration(
                agent_id="doc_reader",
                allowed_recipients=frozenset({"payment_executor"}),
                granted_capabilities=frozenset({"rag:read"}),
            ),
            "payment_executor": AgentRegistration(
                agent_id="payment_executor",
                allowed_recipients=frozenset(),
                granted_capabilities=frozenset({"payments:charge"}),
            ),
        },
        tool_required_capability={"issue_refund": "payments:charge"},
    )
    auth_contract = AuthorityContract(directory=directory)
    with pytest.raises(AuthorityContractViolation, match="lacks capability 'payments:charge'"):
        auth_contract.verify_tool_invocation(
            workflow_id=uuid.uuid4(),
            branch_id="main",
            step=3,
            version=4,
            authenticated_sender="doc_reader",
            claimed_sender="doc_reader",
            recipient="payment_executor",
            tool_name="issue_refund",
            arguments_json='{"amount_usd": 250}',
            issued_at=datetime.datetime.now(datetime.timezone.utc),
        )


def test_financial_and_trust_budget_coupling_equations() -> None:
    """Verify closed-form restart vs. durable cost equations and compounded failure bounds."""
    c_restart = FinancialBudget.expected_uncheckpointed_cost_usd(
        step_cost_usd=0.10,
        steps=50,
        per_step_failure_prob=0.05,
    )
    c_durable = FinancialBudget.expected_durable_cost_usd(
        step_cost_usd=0.10,
        steps=50,
        per_step_failure_prob=0.05,
    )
    # Uncheckpointed: (0.10 / 0.05) * (0.95^(-50) - 1) = 2.0 * (12.996300 - 1) = $23.9926
    # Durable: 50 * 0.10 / 0.95 = $5.2632
    assert math.isclose(c_restart, 23.99260046296078, rel_tol=1e-9)
    assert math.isclose(c_durable, 5.2631578947368425, rel_tol=1e-9)

    fin_budget = FinancialBudget(
        price_uncached_per_1m=3.0,
        price_cached_per_1m=0.3,
        price_completion_per_1m=15.0,
        max_cost_usd=10.0,
    )
    fin_budget.check_spend(c_durable)
    with pytest.raises(FinancialBudgetExceeded):
        fin_budget.check_spend(c_restart)

    trust_budget = TrustBudget(max_trajectory_failure_prob=0.15, max_unverified_steps=5)
    # 8 unverified steps at eps=0.03 -> 1 - 0.97^8 = 21.63% > 15% and depth 8 > 5
    with pytest.raises(TrustBudgetExceeded):
        trust_budget.verify_trajectory_trust([0.03] * 8)

    with pytest.raises(MemoryBudgetExceeded):
        mem = MemoryBudget(
            hbm_capacity_bytes=80 * (1024**3),
            weight_bytes=70 * (1024**3),
            kv_bytes_per_token=327_680,
            activation_bytes_per_token=65_536,
            system_reserve_bytes=2 * (1024**3),
        )
        mem.verify_hbm_admission(resident_kv_tokens=32_768, step_tokens=2048)
