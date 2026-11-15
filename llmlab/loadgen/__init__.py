"""Queueing models, open/closed-loop load generators, goodput metrics, and capacity planning."""
from __future__ import annotations

from llmlab.loadgen.capacity import CapacityPlan, WorkloadProfile, plan_cluster_capacity
from llmlab.loadgen.generator import (
    ReplicaSimConfig,
    RequestSpec,
    compare_open_vs_closed_omission,
    generate_burst_replay_workload,
    generate_poisson_workload,
    simulate_closed_loop,
    simulate_open_loop,
)
from llmlab.loadgen.metrics import (
    GoodputSummary,
    RequestTelemetry,
    ThroughputLatencyPoint,
    empirical_quantile,
    extract_pareto_frontier,
    summarize_goodput,
)
from llmlab.loadgen.queueing import (
    CoupledSaturationPoint,
    KingmanEstimate,
    QueueInstabilityError,
    ServiceDistributionStats,
    inject_hol_stall_breakdown,
    kingman_estimate,
    solve_coupled_kv_saturation,
)

__all__ = [
    "CapacityPlan",
    "CoupledSaturationPoint",
    "GoodputSummary",
    "KingmanEstimate",
    "QueueInstabilityError",
    "ReplicaSimConfig",
    "RequestSpec",
    "RequestTelemetry",
    "ServiceDistributionStats",
    "ThroughputLatencyPoint",
    "WorkloadProfile",
    "compare_open_vs_closed_omission",
    "empirical_quantile",
    "extract_pareto_frontier",
    "generate_burst_replay_workload",
    "generate_poisson_workload",
    "inject_hol_stall_breakdown",
    "kingman_estimate",
    "plan_cluster_capacity",
    "simulate_closed_loop",
    "simulate_open_loop",
    "solve_coupled_kv_saturation",
    "summarize_goodput",
]
