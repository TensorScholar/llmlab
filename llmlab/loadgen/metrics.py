"""Streaming latency telemetry, joint-SLO goodput accounting, and Pareto extraction."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from llmlab.tracing.budgets import LatencyBudget, LatencyBudgetExceeded
from llmlab.tracing.spans import StreamingLatencyBreakdown


def empirical_quantile(values: Sequence[float], q: float) -> float:
    """Compute discrete empirical quantile q in [0.0, 1.0] without interpolation bias."""
    if not (0.0 <= q <= 1.0):
        raise ValueError("Quantile q must lie in [0.0, 1.0].")
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


@dataclass(frozen=True, slots=True)
class RequestTelemetry:
    """Complete timing and token telemetry for a single completed inference request."""

    request_id: str
    scheduled_arrival_ms: float
    execution_start_ms: float
    completion_ms: float
    prompt_tokens: int
    cached_prefix_tokens: int
    breakdown: StreamingLatencyBreakdown

    def __post_init__(self) -> None:
        if self.scheduled_arrival_ms < 0.0:
            raise ValueError("scheduled_arrival_ms must be non-negative.")
        if self.execution_start_ms < self.scheduled_arrival_ms:
            raise ValueError("execution_start_ms cannot precede scheduled_arrival_ms.")
        if self.completion_ms < self.execution_start_ms:
            raise ValueError("completion_ms cannot precede execution_start_ms.")
        if self.prompt_tokens < 1 or self.cached_prefix_tokens < 0:
            raise ValueError("prompt_tokens >= 1 and cached_prefix_tokens >= 0 required.")
        if self.cached_prefix_tokens > self.prompt_tokens:
            raise ValueError("cached_prefix_tokens cannot exceed prompt_tokens.")

    @property
    def queue_wait_ms(self) -> float:
        """Pure queueing delay W_q = tau_start - tau_sched prior to first prefill step."""
        return self.execution_start_ms - self.scheduled_arrival_ms

    @property
    def sojourn_ms(self) -> float:
        """Total end-to-end sojourn time W = completion_ms - scheduled_arrival_ms."""
        return self.completion_ms - self.scheduled_arrival_ms

    @property
    def output_tokens(self) -> int:
        """Generated completion tokens N_out."""
        return self.breakdown.completion_tokens

    @property
    def uncached_prompt_tokens(self) -> int:
        """Prompt tokens requiring active prefill FLOP execution."""
        return self.prompt_tokens - self.cached_prefix_tokens

    def satisfies_slo(self, budget: LatencyBudget) -> bool:
        """Return True iff TTFT, mean TPOT, P99 ITL, and total deadline all pass."""
        if self.sojourn_ms > budget.workflow_deadline_ms:
            return False
        try:
            budget.check_stream(self.breakdown)
        except LatencyBudgetExceeded:
            return False
        return True


@dataclass(frozen=True, slots=True)
class GoodputSummary:
    """Aggregate latency percentiles, raw throughput, and joint-SLO goodput metrics."""

    mode: str
    offered_rps: float
    window_duration_s: float
    total_requests: int
    compliant_requests: int
    raw_rps: float
    goodput_rps: float
    raw_output_tps: float
    goodput_output_tps: float
    joint_attainment_pct: float
    ttft_p50_ms: float
    ttft_p99_ms: float
    tpot_mean_ms: float
    tpot_p99_ms: float
    itl_p99_ms: float
    itl_max_ms: float
    mean_queue_wait_ms: float
    mean_concurrency: float


def summarize_goodput(
    records: Sequence[RequestTelemetry],
    budget: LatencyBudget,
    *,
    mode: str,
    offered_rps: float,
    window_duration_s: float | None = None,
) -> GoodputSummary:
    """Evaluate joint-SLO goodput, tail ITL jitter, and Little's Law concurrency L = lambda * W."""
    if not records:
        raise ValueError("Cannot summarize empty telemetry trace.")

    min_t = min(r.scheduled_arrival_ms for r in records)
    max_t = max(r.completion_ms for r in records)
    eff_window_s = window_duration_s if window_duration_s is not None else max(1e-6, (max_t - min_t) / 1000.0)

    ttfts = [r.breakdown.ttft_ms for r in records]
    tpots = [r.breakdown.tpot_ms for r in records]
    all_itls = [gap for r in records for gap in r.breakdown.itl_ms]
    waits = [r.queue_wait_ms for r in records]
    sojourns_s = [r.sojourn_ms / 1000.0 for r in records]

    compliant = [r for r in records if r.satisfies_slo(budget)]
    raw_tokens = sum(r.output_tokens for r in records)
    goodput_tokens = sum(r.output_tokens for r in compliant)

    total = len(records)
    raw_rps = total / eff_window_s
    goodput_rps = len(compliant) / eff_window_s
    mean_sojourn_s = math.fsum(sojourns_s) / total
    mean_concurrency = raw_rps * mean_sojourn_s

    return GoodputSummary(
        mode=mode,
        offered_rps=offered_rps,
        window_duration_s=eff_window_s,
        total_requests=total,
        compliant_requests=len(compliant),
        raw_rps=raw_rps,
        goodput_rps=goodput_rps,
        raw_output_tps=raw_tokens / eff_window_s,
        goodput_output_tps=goodput_tokens / eff_window_s,
        joint_attainment_pct=100.0 * (len(compliant) / total),
        ttft_p50_ms=empirical_quantile(ttfts, 0.50),
        ttft_p99_ms=empirical_quantile(ttfts, 0.99),
        tpot_mean_ms=math.fsum(tpots) / total,
        tpot_p99_ms=empirical_quantile(tpots, 0.99),
        itl_p99_ms=empirical_quantile(all_itls, 0.99),
        itl_max_ms=max(all_itls) if all_itls else 0.0,
        mean_queue_wait_ms=math.fsum(waits) / total,
        mean_concurrency=mean_concurrency,
    )


@dataclass(frozen=True, slots=True)
class ThroughputLatencyPoint:
    """Single operating point along a throughput-vs-tail-latency curve."""

    label: str
    offered_rps: float
    raw_rps: float
    goodput_rps: float
    raw_output_tps: float
    goodput_output_tps: float
    ttft_p99_ms: float
    itl_p99_ms: float
    joint_attainment_pct: float


def extract_pareto_frontier(
    points: Sequence[ThroughputLatencyPoint],
) -> tuple[ThroughputLatencyPoint, ...]:
    """Return non-dominated operating points maximizing goodput_rps while minimizing ttft_p99_ms."""
    if not points:
        return ()
    ordered = sorted(points, key=lambda p: (p.ttft_p99_ms, -p.goodput_rps))
    frontier: list[ThroughputLatencyPoint] = []
    best_goodput = -1.0
    for pt in ordered:
        if pt.goodput_rps > best_goodput:
            frontier.append(pt)
            best_goodput = pt.goodput_rps
    return tuple(frontier)
