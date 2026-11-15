"""Positive and negative-control test suite for Chapter 3 loadgen, queueing, and capacity math."""
from __future__ import annotations

import pytest

from llmlab.loadgen import (
    ReplicaSimConfig,
    ServiceDistributionStats,
    ThroughputLatencyPoint,
    WorkloadProfile,
    compare_open_vs_closed_omission,
    extract_pareto_frontier,
    generate_burst_replay_workload,
    generate_poisson_workload,
    inject_hol_stall_breakdown,
    kingman_estimate,
    plan_cluster_capacity,
    simulate_open_loop,
    solve_coupled_kv_saturation,
    summarize_goodput,
)
from llmlab.loadgen.queueing import QueueInstabilityError
from llmlab.memory import (
    H100_SXM5,
    H200_SXM,
    LAYOUT_BF16,
    LAYOUT_FP8,
    LLAMA_70B_GQA,
    LLAMA_70B_HYPOTHETICAL_MHA,
    PrecisionFormat,
    RooflineAnalyzer,
)
from llmlab.tracing.budgets import LatencyBudget, LatencyBudgetExceeded


def test_kingman_vut_heavy_tailed_explosion_over_mm1() -> None:
    """Heavy-tailed output lengths (C_s^2 > 1) multiply G/G/1 wait time far past M/M/1 predictions."""
    lognorm_stats = ServiceDistributionStats.from_lognormal(mu_ln=7.1, sigma_ln=1.15)
    assert lognorm_stats.c_s_sq > 2.5

    est_poisson = kingman_estimate(arrival_rate_rps=0.32, service_stats=lognorm_stats, c_a_sq=1.0)
    est_bursty = kingman_estimate(arrival_rate_rps=0.32, service_stats=lognorm_stats, c_a_sq=2.5)
    assert est_poisson.tail_inflation_vs_mm1 > 1.8
    assert est_bursty.mean_wait_ms > est_poisson.mean_wait_ms
    assert est_bursty.littles_concurrency > est_poisson.littles_concurrency

    with pytest.raises(QueueInstabilityError):
        kingman_estimate(arrival_rate_rps=0.50, service_stats=lognorm_stats)


def test_negative_control_closed_loop_hides_queueing_collapse_via_coordinated_omission() -> None:
    """Closed-loop (concurrency=N) benchmarks mask order-of-magnitude P99 TTFT collapse at burst/saturation."""
    analyzer = RooflineAnalyzer(LLAMA_70B_GQA, H100_SXM5, LAYOUT_FP8, LAYOUT_FP8, PrecisionFormat.FP8)
    budget_mem = LLAMA_70B_GQA.to_memory_budget(
        H100_SXM5, LAYOUT_FP8, LAYOUT_FP8, compute_format=PrecisionFormat.FP8, tp_size=2
    )
    sim_cfg = ReplicaSimConfig(
        analyzer=analyzer,
        memory_budget=budget_mem,
        tp_size=2,
        chunk_size_tokens=512,
        max_step_tokens=1024,
    )
    # Burst to 22 req/s exceeds single-replica TP=2 FP8 capacity (~10-12 req/s at 1024-token prompts)
    trace = generate_burst_replay_workload(
        duration_s=20.0,
        baseline_rps=5.0,
        burst_rps=22.0,
        burst_start_s=5.0,
        burst_duration_s=6.0,
        prompt_tokens_base=1024,
        output_tokens_mean=96,
        seed=42,
    )
    slo = LatencyBudget(
        max_ttft_ms=500.0,
        max_tpot_ms=30.0,
        max_p99_itl_ms=50.0,
        workflow_deadline_ms=20_000.0,
    )
    open_sum, closed_sum = compare_open_vs_closed_omission(
        schedule=trace,
        closed_concurrency=4,
        config=sim_cfg,
        budget=slo,
    )

    assert open_sum.ttft_p99_ms > 5.0 * closed_sum.ttft_p99_ms
    assert closed_sum.mean_queue_wait_ms < 50.0
    assert open_sum.mean_queue_wait_ms > 500.0
    assert open_sum.joint_attainment_pct < closed_sum.joint_attainment_pct


def test_negative_control_healthy_mean_tpot_masks_hol_p99_itl_stall() -> None:
    """Unchunked long prefill keeps mean TPOT within SLO while triggering fatal P99 ITL stalls."""
    slo = LatencyBudget(
        max_ttft_ms=1500.0,
        max_tpot_ms=30.0,
        max_p99_itl_ms=45.0,
        workflow_deadline_ms=10_000.0,
    )
    stalled = inject_hol_stall_breakdown(
        normal_tpot_ms=22.0,
        ttft_ms=120.0,
        output_tokens=80,
        hol_stall_ms=500.0,
        stall_token_index=40,
    )
    # Mean TPOT stays <= 30 ms SLO (~28.33 ms), hiding a 522.0 ms interactive stream freeze at P99 ITL!
    assert stalled.tpot_ms <= slo.max_tpot_ms
    assert stalled.max_itl_gap_ms > 500.0
    with pytest.raises(LatencyBudgetExceeded, match="P99 ITL"):
        slo.check_stream(stalled)

    # Also prove end-to-end in mixed open-loop simulation: unchunked vs chunked=512 prefill
    analyzer = RooflineAnalyzer(LLAMA_70B_GQA, H100_SXM5, LAYOUT_FP8, LAYOUT_FP8, PrecisionFormat.FP8)
    budget_mem = LLAMA_70B_GQA.to_memory_budget(
        H100_SXM5, LAYOUT_FP8, LAYOUT_FP8, compute_format=PrecisionFormat.FP8, tp_size=2
    )
    mixed = generate_poisson_workload(
        num_requests=40,
        arrival_rate_rps=1.2,
        prompt_tokens_base=256,
        long_prompt_tokens=4096,
        p_long_prompt=0.15,
        output_mu_ln=4.8,
        output_sigma_ln=0.25,
        seed=11,
    )
    unchunked = summarize_goodput(
        simulate_open_loop(
            mixed,
            ReplicaSimConfig(analyzer=analyzer, memory_budget=budget_mem, tp_size=2, chunk_size_tokens=None),
        ),
        slo,
        mode="unchunked",
        offered_rps=1.2,
    )
    chunked = summarize_goodput(
        simulate_open_loop(
            mixed,
            ReplicaSimConfig(analyzer=analyzer, memory_budget=budget_mem, tp_size=2, chunk_size_tokens=512),
        ),
        slo,
        mode="chunked_512",
        offered_rps=1.2,
    )
    assert unchunked.tpot_mean_ms < 35.0
    assert unchunked.itl_max_ms > 250.0
    assert chunked.itl_max_ms < unchunked.itl_max_ms * 0.5


def test_negative_control_kv_capacity_constraint_overrides_compute_sizing() -> None:
    """Ignoring KV residency under long contexts underprovisions replicas and triggers KV saturation."""
    slo = LatencyBudget(
        max_ttft_ms=1500.0,
        max_tpot_ms=45.0,
        max_p99_itl_ms=60.0,
        workflow_deadline_ms=40_000.0,
    )
    long_ctx_workload = WorkloadProfile(
        target_rps=25.0,
        short_prompt_tokens=8192,
        long_prompt_tokens=16384,
        p_long_prompt=0.20,
        mean_output_tokens=512,
        prefix_hit_ratio=0.85,
        shared_system_prefix_tokens=1024,
        target_compute_util=0.75,
        target_kv_util=0.80,
    )
    # 70B GQA in BF16 on H100 TP=2 hits hard 160 GB HBM walls (only ~20 GB left after weights), shrinking B_KV << B_SLO
    plan_kv = plan_cluster_capacity(
        workload=long_ctx_workload,
        arch=LLAMA_70B_GQA,
        accel=H100_SXM5,
        weight_layout=LAYOUT_BF16,
        kv_layout=LAYOUT_BF16,
        compute_format=PrecisionFormat.BF16,
        latency_budget=slo,
        tp_size=2,
        redundancy_k=2,
        spot_max_fraction=0.25,
    )
    assert plan_kv.binding_constraint == "kv_capacity"
    assert plan_kv.replicas_kv_bound > plan_kv.replicas_compute_bound
    assert plan_kv.total_ha_replicas == plan_kv.active_replicas + 2
    assert plan_kv.on_demand_floor_replicas + plan_kv.spot_surge_replicas == plan_kv.total_ha_replicas

    # Coupled Little's Law solver raises QueueInstabilityError once arrival rate breaches single-replica KV slots
    analyzer_bf16 = RooflineAnalyzer(
        LLAMA_70B_GQA, H100_SXM5, LAYOUT_BF16, LAYOUT_BF16, PrecisionFormat.BF16
    )
    mem_bf16 = LLAMA_70B_GQA.to_memory_budget(
        H100_SXM5, LAYOUT_BF16, LAYOUT_BF16, compute_format=PrecisionFormat.BF16, tp_size=2
    )
    with pytest.raises(QueueInstabilityError, match="KV admission saturation"):
        solve_coupled_kv_saturation(
            arrival_rate_rps=plan_kv.kv_rps_per_replica * 2.0,
            analyzer=analyzer_bf16,
            budget=mem_bf16,
            mean_prompt_tokens=9830,
            mean_output_tokens=512,
            tp_size=2,
        )


def test_pareto_frontier_extracts_non_dominated_goodput_points() -> None:
    """Pareto frontier drops throughput-saturated points where tail TTFT spikes without goodput gain."""
    pts = [
        ThroughputLatencyPoint("r1", 1.0, 1.0, 1.0, 128.0, 128.0, 140.0, 20.0, 100.0),
        ThroughputLatencyPoint("r2", 2.0, 2.0, 1.9, 256.0, 243.0, 220.0, 25.0, 95.0),
        ThroughputLatencyPoint("r3_collapse", 3.5, 2.3, 0.4, 294.0, 51.0, 3200.0, 40.0, 17.4),
    ]
    frontier = extract_pareto_frontier(pts)
    assert [p.label for p in frontier] == ["r1", "r2"]
