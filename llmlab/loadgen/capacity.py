"""Dual-constraint GPU cluster capacity planning coupled to llmlab.memory and latency SLOs."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from llmlab.memory.hardware import AcceleratorSpec, PrecisionFormat
from llmlab.memory.kv_cache import QuantizationLayout, TransformerArchitecture
from llmlab.memory.roofline import RooflineAnalyzer
from llmlab.tracing.budgets import LatencyBudget, MemoryBudget


@dataclass(frozen=True, slots=True)
class WorkloadProfile:
    """Statistical workload profile driving compute FLOPs, KV footprint, and prefix reuse."""

    target_rps: float
    short_prompt_tokens: int
    long_prompt_tokens: int
    p_long_prompt: float
    mean_output_tokens: int
    prefix_hit_ratio: float = 0.0
    shared_system_prefix_tokens: int = 0
    target_compute_util: float = 0.70
    target_kv_util: float = 0.80

    def __post_init__(self) -> None:
        if self.target_rps <= 0.0:
            raise ValueError("target_rps must be positive.")
        if self.short_prompt_tokens < 1 or self.long_prompt_tokens < self.short_prompt_tokens:
            raise ValueError("1 <= short_prompt_tokens <= long_prompt_tokens required.")
        if not (0.0 <= self.p_long_prompt <= 1.0):
            raise ValueError("p_long_prompt must lie in [0.0, 1.0].")
        if self.mean_output_tokens < 2:
            raise ValueError("mean_output_tokens must be >= 2.")
        if not (0.0 <= self.prefix_hit_ratio < 1.0):
            raise ValueError("prefix_hit_ratio must lie in [0.0, 1.0).")
        if self.shared_system_prefix_tokens < 0:
            raise ValueError("shared_system_prefix_tokens must be non-negative.")
        if not (0.0 < self.target_compute_util < 1.0) or not (0.0 < self.target_kv_util < 1.0):
            raise ValueError("Utilization headrooms must lie in (0.0, 1.0).")

    @property
    def mean_prompt_tokens(self) -> float:
        """Expected total prompt tokens S_in before prefix cache deduplication."""
        return (1.0 - self.p_long_prompt) * self.short_prompt_tokens + (
            self.p_long_prompt * self.long_prompt_tokens
        )

    @property
    def mean_uncached_prompt_tokens(self) -> float:
        """Expected prompt tokens requiring prefill computation after prefix cache hits."""
        return self.mean_prompt_tokens * (1.0 - self.prefix_hit_ratio)

    @property
    def mean_session_unique_kv_tokens(self) -> float:
        """Per-request non-shared resident KV tokens averaged across active generation."""
        unique_prompt = max(1.0, self.mean_prompt_tokens - float(self.shared_system_prefix_tokens))
        return unique_prompt + (self.mean_output_tokens / 2.0)


@dataclass(frozen=True, slots=True)
class CapacityPlan:
    """Dual-constraint sizing output linking Roofline physics, Little's Law, and N+K HA."""

    gpus_per_replica: int
    max_tpot_batch_size: int
    max_kv_concurrent_seqs: int
    effective_operating_batch: int
    mean_prefill_ms: float
    step_tpot_ms: float
    mean_exec_sojourn_s: float
    compute_rps_per_replica: float
    kv_rps_per_replica: float
    replicas_compute_bound: int
    replicas_kv_bound: int
    active_replicas: int
    binding_constraint: Literal["kv_capacity", "compute_latency"]
    nk_redundancy_replicas: int
    total_ha_replicas: int
    total_ha_gpus: int
    on_demand_floor_replicas: int
    spot_surge_replicas: int


def plan_cluster_capacity(
    *,
    workload: WorkloadProfile,
    arch: TransformerArchitecture,
    accel: AcceleratorSpec,
    weight_layout: QuantizationLayout,
    kv_layout: QuantizationLayout,
    compute_format: PrecisionFormat,
    latency_budget: LatencyBudget,
    tp_size: int = 1,
    ep_size: int = 1,
    pp_size: int = 1,
    mfu: float = 0.50,
    mbu: float = 0.80,
    absorb_mla: bool = True,
    redundancy_k: int = 1,
    spot_max_fraction: float = 0.0,
    step_tokens_budget: int = 512,
) -> CapacityPlan:
    """Size GPU cluster replicas bounded by max(N_compute, N_KV) under N+K & spot policies."""
    if redundancy_k < 0:
        raise ValueError("redundancy_k must be non-negative.")
    if not (0.0 <= spot_max_fraction < 1.0):
        raise ValueError("spot_max_fraction must lie in [0.0, 1.0).")

    gpus_per_replica = tp_size * pp_size if ep_size == 1 else max(tp_size * pp_size, ep_size)
    analyzer = RooflineAnalyzer(arch, accel, weight_layout, kv_layout, compute_format)
    budget: MemoryBudget = arch.to_memory_budget(
        accel,
        weight_layout,
        kv_layout,
        compute_format=compute_format,
        tp_size=tp_size,
        pp_size=pp_size,
        ep_size=ep_size,
    )

    # Compute uncached prefill work accounting for bimodal short/long prompt mixture
    s_short_unc = max(1, int(round(workload.short_prompt_tokens * (1.0 - workload.prefix_hit_ratio))))
    s_long_unc = max(1, int(round(workload.long_prompt_tokens * (1.0 - workload.prefix_hit_ratio))))
    pre_short = analyzer.analyze_prefill_step(
        s_short_unc, tp_size=tp_size, ep_size=ep_size, mfu=mfu, mbu=mbu
    )
    pre_long = analyzer.analyze_prefill_step(
        s_long_unc, tp_size=tp_size, ep_size=ep_size, mfu=mfu, mbu=mbu
    )
    mean_prefill_ms = (
        (1.0 - workload.p_long_prompt) * pre_short.step_time_ms
        + workload.p_long_prompt * pre_long.step_time_ms
    )

    # Find largest decode batch size preserving per-step TPOT <= max_tpot_ms
    mean_ctx_int = max(1, int(round(workload.mean_prompt_tokens + workload.mean_output_tokens / 2.0)))
    max_kv_tokens = budget.max_resident_kv_tokens(step_tokens=step_tokens_budget)
    usable_kv_tokens = max(0, max_kv_tokens - workload.shared_system_prefix_tokens)
    max_kv_seqs = max(0, int(usable_kv_tokens // max(1.0, workload.mean_session_unique_kv_tokens)))

    max_slo_batch = 1
    for candidate_b in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512):
        dec_check = analyzer.analyze_decode_step(
            candidate_b, mean_ctx_int, tp_size=tp_size, ep_size=ep_size, mfu=mfu, mbu=mbu, absorb_mla=absorb_mla
        )
        if dec_check.step_time_ms <= latency_budget.max_tpot_ms:
            max_slo_batch = candidate_b
        else:
            break

    dec_slo = analyzer.analyze_decode_step(
        max_slo_batch, mean_ctx_int, tp_size=tp_size, ep_size=ep_size, mfu=mfu, mbu=mbu, absorb_mla=absorb_mla
    )
    slo_tpot_ms = dec_slo.step_time_ms

    eff_batch = max(1, min(max_slo_batch, max(1, max_kv_seqs)))
    dec_op = (
        dec_slo
        if eff_batch == max_slo_batch
        else analyzer.analyze_decode_step(
            eff_batch, mean_ctx_int, tp_size=tp_size, ep_size=ep_size, mfu=mfu, mbu=mbu, absorb_mla=absorb_mla
        )
    )
    step_tpot_ms = dec_op.step_time_ms
    mean_exec_sojourn_s = (mean_prefill_ms + (workload.mean_output_tokens - 1) * step_tpot_ms) / 1000.0

    # Constraint 1: Unconstrained-KV compute/bandwidth ceiling evaluated at B_SLO (what naive sizing assumes)
    compute_duty_s_per_req = (
        mean_prefill_ms + ((workload.mean_output_tokens - 1) * slo_tpot_ms) / max_slo_batch
    ) / 1000.0
    compute_rps = workload.target_compute_util / compute_duty_s_per_req

    # Constraint 2: Little's Law KV-pool admission ceiling lambda_KV = rho_KV * B_KV / W
    if max_kv_seqs == 0:
        kv_rps = 1e-6
    else:
        kv_rps = workload.target_kv_util * (float(max_kv_seqs) / mean_exec_sojourn_s)

    n_compute = max(1, math.ceil(workload.target_rps / compute_rps))
    n_kv = max(1, math.ceil(workload.target_rps / kv_rps))
    active_replicas = max(n_compute, n_kv)
    binding: Literal["kv_capacity", "compute_latency"] = (
        "kv_capacity" if n_kv >= n_compute else "compute_latency"
    )

    total_ha_replicas = active_replicas + redundancy_k
    total_ha_gpus = total_ha_replicas * gpus_per_replica

    # Keep guaranteed baseline floor on on-demand nodes so simultaneous spot eviction cannot violate HA
    spot_surge = int(math.floor(active_replicas * spot_max_fraction))
    on_demand_floor = total_ha_replicas - spot_surge

    return CapacityPlan(
        gpus_per_replica=gpus_per_replica,
        max_tpot_batch_size=max_slo_batch,
        max_kv_concurrent_seqs=max_kv_seqs,
        effective_operating_batch=eff_batch,
        mean_prefill_ms=mean_prefill_ms,
        step_tpot_ms=step_tpot_ms,
        mean_exec_sojourn_s=mean_exec_sojourn_s,
        compute_rps_per_replica=compute_rps,
        kv_rps_per_replica=kv_rps,
        replicas_compute_bound=n_compute,
        replicas_kv_bound=n_kv,
        active_replicas=active_replicas,
        binding_constraint=binding,
        nk_redundancy_replicas=redundancy_k,
        total_ha_replicas=total_ha_replicas,
        total_ha_gpus=total_ha_gpus,
        on_demand_floor_replicas=on_demand_floor,
        spot_surge_replicas=spot_surge,
    )
