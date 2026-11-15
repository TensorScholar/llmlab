"""Autoregressive queueing models: Kingman's VUT equation, Little's Law, and HoL spikes."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from llmlab.memory.roofline import RooflineAnalyzer
from llmlab.tracing.budgets import MemoryBudget
from llmlab.tracing.spans import StreamingLatencyBreakdown


class QueueInstabilityError(RuntimeError):
    """Raised when offered load reaches or exceeds service/KV capacity (rho >= 1.0)."""


@dataclass(frozen=True, slots=True)
class ServiceDistributionStats:
    """First two moments and squared coefficient of variation C_s^2 for service durations."""

    mean_ms: float
    variance_ms2: float

    def __post_init__(self) -> None:
        if self.mean_ms <= 0.0:
            raise ValueError("mean_ms must be strictly positive.")
        if self.variance_ms2 < 0.0:
            raise ValueError("variance_ms2 must be non-negative.")

    @property
    def std_ms(self) -> float:
        """Standard deviation sigma_s in milliseconds."""
        return math.sqrt(self.variance_ms2)

    @property
    def c_s_sq(self) -> float:
        """Squared coefficient of variation C_s^2 = Var(S) / E[S]^2."""
        return self.variance_ms2 / (self.mean_ms * self.mean_ms)

    @classmethod
    def from_samples(cls, samples_ms: Sequence[float]) -> ServiceDistributionStats:
        """Compute exact empirical mean and population variance from service duration samples."""
        if not samples_ms:
            raise ValueError("samples_ms must be non-empty.")
        n = len(samples_ms)
        mean_val = math.fsum(samples_ms) / n
        var_val = math.fsum((x - mean_val) ** 2 for x in samples_ms) / n
        return cls(mean_ms=mean_val, variance_ms2=var_val)

    @classmethod
    def from_lognormal(cls, mu_ln: float, sigma_ln: float) -> ServiceDistributionStats:
        """Exact analytical moments for S ~ LogNormal(mu_ln, sigma_ln) in milliseconds."""
        if sigma_ln < 0.0:
            raise ValueError("sigma_ln must be non-negative.")
        mean_val = math.exp(mu_ln + 0.5 * sigma_ln * sigma_ln)
        var_val = (math.exp(sigma_ln * sigma_ln) - 1.0) * math.exp(
            2.0 * mu_ln + sigma_ln * sigma_ln
        )
        return cls(mean_ms=mean_val, variance_ms2=var_val)

    @classmethod
    def from_bimodal_mixture(
        cls,
        *,
        p_heavy: float,
        light_mean_ms: float,
        heavy_mean_ms: float,
        light_c_sq: float = 0.25,
        heavy_c_sq: float = 1.0,
    ) -> ServiceDistributionStats:
        """Mix two workloads (e.g., short chat + heavy reasoning/long-prefill traces)."""
        if not (0.0 <= p_heavy <= 1.0):
            raise ValueError("p_heavy must be in [0.0, 1.0].")
        p_light = 1.0 - p_heavy
        mean_total = p_light * light_mean_ms + p_heavy * heavy_mean_ms
        e_sq_light = (1.0 + light_c_sq) * (light_mean_ms ** 2)
        e_sq_heavy = (1.0 + heavy_c_sq) * (heavy_mean_ms ** 2)
        e_sq_total = p_light * e_sq_light + p_heavy * e_sq_heavy
        var_total = max(0.0, e_sq_total - mean_total * mean_total)
        return cls(mean_ms=mean_total, variance_ms2=var_total)


@dataclass(frozen=True, slots=True)
class KingmanEstimate:
    """Decomposed G/G/1 queue wait and Little's Law concurrency prediction."""

    arrival_rate_rps: float
    rho: float
    c_a_sq: float
    c_s_sq: float
    mean_service_ms: float
    mean_wait_ms: float
    mean_sojourn_ms: float
    littles_concurrency: float
    mm1_wait_ms: float

    @property
    def tail_inflation_vs_mm1(self) -> float:
        """Ratio of G/G/1 wait time to memoryless M/M/1 expectation."""
        if self.mm1_wait_ms <= 0.0:
            return 1.0
        return self.mean_wait_ms / self.mm1_wait_ms


def kingman_estimate(
    *,
    arrival_rate_rps: float,
    service_stats: ServiceDistributionStats,
    c_a_sq: float = 1.0,
    num_servers: int = 1,
) -> KingmanEstimate:
    """Compute Kingman's G/G/1 VUT waiting time and Little's Law active concurrency L = lambda * W.

    For num_servers > 1 sharing one centralized FIFO queue, uses Sakasegawa's (1977)
    sqrt(2(c+1))-1 utilization exponent over G/G/c while scaling rho = lambda * E[S] / c.
    """
    if arrival_rate_rps < 0.0:
        raise ValueError("arrival_rate_rps must be non-negative.")
    if c_a_sq < 0.0 or num_servers < 1:
        raise ValueError("c_a_sq >= 0 and num_servers >= 1 required.")

    mean_service_s = service_stats.mean_ms / 1000.0
    rho = (arrival_rate_rps * mean_service_s) / num_servers
    if rho >= 1.0:
        raise QueueInstabilityError(
            f"Offered load lambda={arrival_rate_rps:.2f} req/s yields rho={rho:.4f} >= 1.0: "
            "queue grows without bound."
        )

    variability_factor = 0.5 * (c_a_sq + service_stats.c_s_sq)
    if num_servers == 1:
        util_factor = rho / (1.0 - rho)
    else:
        exp_val = math.sqrt(2.0 * (num_servers + 1.0)) - 1.0
        util_factor = (math.pow(rho, exp_val) / (num_servers * (1.0 - rho)))

    wait_ms = util_factor * variability_factor * service_stats.mean_ms
    mm1_wait_ms = (rho / (1.0 - rho)) * service_stats.mean_ms
    sojourn_ms = wait_ms + service_stats.mean_ms
    concurrency = arrival_rate_rps * (sojourn_ms / 1000.0)

    return KingmanEstimate(
        arrival_rate_rps=arrival_rate_rps,
        rho=rho,
        c_a_sq=c_a_sq,
        c_s_sq=service_stats.c_s_sq,
        mean_service_ms=service_stats.mean_ms,
        mean_wait_ms=wait_ms,
        mean_sojourn_ms=sojourn_ms,
        littles_concurrency=concurrency,
        mm1_wait_ms=mm1_wait_ms,
    )


@dataclass(frozen=True, slots=True)
class CoupledSaturationPoint:
    """Fixed-point solution coupling Roofline decode TPOT(B), Little's Law L, and KV capacity."""

    arrival_rate_rps: float
    resident_batch_size: float
    step_tpot_ms: float
    mean_sojourn_ms: float
    resident_kv_tokens: float
    kv_utilization: float
    rho_kv_admission: float
    kingman_wait_ms: float
    converged: bool


def solve_coupled_kv_saturation(
    *,
    arrival_rate_rps: float,
    analyzer: RooflineAnalyzer,
    budget: MemoryBudget,
    mean_prompt_tokens: int,
    mean_output_tokens: int,
    tp_size: int = 1,
    ep_size: int = 1,
    c_a_sq: float = 1.0,
    c_s_sq: float = 2.5,
    max_iters: int = 80,
) -> CoupledSaturationPoint:
    """Solve coupled Little's Law fixed point B = lambda * W(B) bounded by HBM KV capacity.

    As concurrency B grows, historical KV reads per decode step inflate TPOT(B),
    increasing sojourn W(B), which feeds back into higher L = lambda * W(B) until
    resident KV tokens hit max_resident_kv_tokens (rho_kv -> 1) and waiting time explodes.
    """
    if mean_prompt_tokens < 1 or mean_output_tokens < 2:
        raise ValueError("mean_prompt_tokens >= 1 and mean_output_tokens >= 2 required.")

    mean_ctx = mean_prompt_tokens + (mean_output_tokens // 2)
    prefill_res = analyzer.analyze_prefill_step(
        mean_prompt_tokens, tp_size=tp_size, ep_size=ep_size
    )
    max_kv_tokens = budget.max_resident_kv_tokens(step_tokens=256)
    kv_slot_servers = max(1, max_kv_tokens // mean_ctx)
    kv_slots_cap = float(kv_slot_servers)

    # Fixed-point iteration over active running batch B <= kv_slots_cap
    b_est = max(1.0, min(kv_slots_cap * 0.99, arrival_rate_rps * 1.0))
    step_tpot_ms = 0.0
    service_ms = 0.0
    converged = False
    for _ in range(max_iters):
        b_int = max(1, min(int(math.ceil(b_est)), int(kv_slots_cap)))
        dec_res = analyzer.analyze_decode_step(
            b_int, mean_ctx, tp_size=tp_size, ep_size=ep_size
        )
        step_tpot_ms = dec_res.step_time_ms
        # Concurrent batch amortizes step cost: service sojourn = prefill + decode_steps * tpot
        service_ms = prefill_res.step_time_ms + (mean_output_tokens - 1) * step_tpot_ms
        next_b = arrival_rate_rps * (service_ms / 1000.0)
        if abs(next_b - b_est) < 1e-4:
            b_est = next_b
            converged = True
            break
        b_est = 0.5 * b_est + 0.5 * min(next_b, kv_slots_cap * 0.9999)

    resident_kv_tokens = min(float(max_kv_tokens), b_est * mean_ctx)
    kv_util = resident_kv_tokens / float(max_kv_tokens) if max_kv_tokens > 0 else 1.0
    # Compute arrival vs KV-slot service throughput utilization rho_kv = lambda / mu_kv
    mu_kv_rps = kv_slots_cap / (service_ms / 1000.0) if service_ms > 0.0 else 0.0
    rho_kv = arrival_rate_rps / mu_kv_rps if mu_kv_rps > 0.0 else float("inf")

    if rho_kv >= 1.0:
        raise QueueInstabilityError(
            f"KV admission saturation breached: lambda={arrival_rate_rps:.2f} req/s >= "
            f"KV-slot capacity mu_kv={mu_kv_rps:.2f} req/s (kv_slots_cap={kv_slots_cap:.1f})."
        )

    stats = ServiceDistributionStats(
        mean_ms=service_ms,
        variance_ms2=c_s_sq * service_ms * service_ms,
    )
    kv_slot_servers = max(1, int(round(kv_slots_cap)))
    kg = kingman_estimate(
        arrival_rate_rps=arrival_rate_rps,
        service_stats=stats,
        c_a_sq=c_a_sq,
        num_servers=kv_slot_servers,
    )
    total_sojourn = kg.mean_sojourn_ms
    return CoupledSaturationPoint(
        arrival_rate_rps=arrival_rate_rps,
        resident_batch_size=min(kv_slots_cap, arrival_rate_rps * (service_ms / 1000.0)),
        step_tpot_ms=step_tpot_ms,
        mean_sojourn_ms=total_sojourn,
        resident_kv_tokens=resident_kv_tokens,
        kv_utilization=kv_util,
        rho_kv_admission=rho_kv,
        kingman_wait_ms=kg.mean_wait_ms,
        converged=converged,
    )


def inject_hol_stall_breakdown(
    *,
    normal_tpot_ms: float,
    ttft_ms: float,
    output_tokens: int,
    hol_stall_ms: float,
    stall_token_index: int = 32,
) -> StreamingLatencyBreakdown:
    """Construct a StreamingLatencyBreakdown with a single unchunked prefill HoL stall."""
    if output_tokens < 2:
        raise ValueError("output_tokens must be >= 2 to expose ITL steps.")
    gaps = [normal_tpot_ms] * (output_tokens - 1)
    idx = min(len(gaps) - 1, max(0, stall_token_index))
    gaps[idx] = gaps[idx] + hol_stall_ms
    return StreamingLatencyBreakdown(ttft_ms=ttft_ms, itl_ms=tuple(gaps))
