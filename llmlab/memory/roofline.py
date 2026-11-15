"""Roofline operational intensity models, critical batch size solvers, and MoE All-to-All bounds."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from llmlab.memory.hardware import AcceleratorSpec, PrecisionFormat
from llmlab.memory.kv_cache import AttentionVariant, QuantizationLayout, TransformerArchitecture


class StepPhase(str, Enum):
    """Transformer execution regime within an iteration."""

    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True, slots=True)
class StepRooflineResult:
    """Decomposed FLOPs, HBM byte traffic, arithmetic intensity, and Roofline latency."""

    phase: StepPhase
    batch_size: int
    step_tokens: int
    mean_context_len: int
    linear_flops: float
    attention_flops: float
    weight_bytes_read: int
    kv_bytes_transferred: int
    total_hbm_bytes: int
    arithmetic_intensity: float
    ridge_point: float
    compute_time_ms: float
    memory_time_ms: float
    step_time_ms: float

    @property
    def is_memory_bound(self) -> bool:
        """True when operational arithmetic intensity falls below the hardware ridge point."""
        return self.arithmetic_intensity < self.ridge_point

    @property
    def total_flops(self) -> float:
        """Total step floating-point operations (linear projections + core attention)."""
        return self.linear_flops + self.attention_flops

    @property
    def achieved_mfu(self) -> float:
        """Fraction of step time spent compute-saturated vs theoretically achievable."""
        if self.step_time_ms <= 0.0:
            return 0.0
        return min(1.0, self.compute_time_ms / self.step_time_ms)


@dataclass(frozen=True, slots=True)
class MoEAllToAllResult:
    """Expert Parallelism (EP) weight residency, All-to-All traffic, and skew straggler bounds."""

    ep_size: int
    local_batch_tokens: int
    global_batch_tokens: int
    weight_bytes_per_rank: int
    expected_unique_experts_per_rank: float
    uniform_bytes_per_rank_per_layer: int
    skewed_bottleneck_bytes_per_layer: int
    uniform_comm_ms_all_layers: float
    skewed_comm_ms_all_layers: float
    skew_factor: float
    straggler_ffn_flops_amplification: float
    straggler_hbm_activation_bytes_amplification: float

    @property
    def communication_slowdown(self) -> float:
        """Ratio of skewed barrier All-to-All communication time to uniform routing time."""
        if self.uniform_comm_ms_all_layers <= 0.0:
            return 1.0
        return self.skewed_comm_ms_all_layers / self.uniform_comm_ms_all_layers


class RooflineAnalyzer:
    """Unified Roofline and Mixture-of-Experts interconnect model for llmlab v0.1.2."""

    def __init__(
        self,
        arch: TransformerArchitecture,
        accel: AcceleratorSpec,
        weight_layout: QuantizationLayout,
        kv_layout: QuantizationLayout,
        compute_format: PrecisionFormat = PrecisionFormat.FP8,
    ) -> None:
        self.arch = arch
        self.accel = accel
        self.weight_layout = weight_layout
        self.kv_layout = kv_layout
        self.compute_format = compute_format

    @staticmethod
    def expected_touched_experts(
        global_batch_tokens: int,
        num_routed_experts: int,
        num_active_experts: int,
    ) -> float:
        """Expected unique routed experts activated across a batch: E * (1 - (1 - k/E)^B)."""
        if global_batch_tokens <= 0:
            return 0.0
        p_miss_single = 1.0 - (num_active_experts / num_routed_experts)
        return num_routed_experts * (1.0 - math.pow(p_miss_single, global_batch_tokens))

    def _active_weight_bytes_read_per_gpu(
        self,
        global_batch_tokens: int,
        *,
        tp_size: int = 1,
        ep_size: int = 1,
    ) -> int:
        if self.arch.num_moe_layers == 0:
            return self.arch.weight_bytes_per_gpu(
                self.weight_layout, tp_size=tp_size, pp_size=1, ep_size=1
            )
        dense_params_per_gpu = math.ceil(self.arch.non_routed_parameters / tp_size)
        touched_experts_global = self.expected_touched_experts(
            global_batch_tokens,
            self.arch.num_routed_experts,
            self.arch.num_active_experts,
        )
        touched_experts_per_ep_rank = touched_experts_global / ep_size
        routed_params_touched = round(
            self.arch.num_moe_layers
            * touched_experts_per_ep_rank
            * self.arch.routed_expert_params_per_expert_layer
        )
        return self.weight_layout.tensor_bytes(dense_params_per_gpu + routed_params_touched)

    def _decode_attention_flops_per_gpu(
        self,
        batch_size: int,
        mean_context_len: int,
        *,
        tp_size: int = 1,
        absorb_mla: bool = True,
    ) -> float:
        if mean_context_len <= 0 or batch_size <= 0:
            return 0.0
        heads_per_gpu = self.arch.num_q_heads / tp_size
        if self.arch.attention_variant in (AttentionVariant.MHA, AttentionVariant.GQA):
            # Per layer & head: Q@K^T (2*d_h*S) + P@V (2*d_h*S) = 4*d_h*S FLOPs per token in batch
            return float(
                self.arch.num_layers * batch_size * heads_per_gpu * 4 * self.arch.head_dim * mean_context_len
            )
        if absorb_mla:
            # Absorbed MLA attends against shared latent c^{KV} (d_c) for QK^T and PV, plus k^R (d_R) for QK^T
            flops_per_head_tok = 2 * (2 * self.arch.mla_kv_lora_rank + self.arch.mla_rope_head_dim)
        else:
            flops_per_head_tok = 2 * (
                self.arch.head_dim + self.arch.mla_rope_head_dim + self.arch.mla_v_head_dim
            )
        return float(
            self.arch.num_layers * batch_size * heads_per_gpu * flops_per_head_tok * mean_context_len
        )

    def analyze_prefill_step(
        self,
        prompt_tokens: int,
        *,
        batch_size: int = 1,
        tp_size: int = 1,
        ep_size: int = 1,
        mfu: float = 0.50,
        mbu: float = 0.80,
        include_attention: bool = True,
    ) -> StepRooflineResult:
        """Analyze compute-saturated prefill over ``S_in = prompt_tokens`` tokens per sequence."""
        if prompt_tokens <= 0 or batch_size <= 0 or tp_size <= 0 or ep_size <= 0:
            raise ValueError("prompt_tokens, batch_size, tp_size, and ep_size must be positive.")
        total_step_tokens = batch_size * prompt_tokens
        linear_flops = float(2.0 * (self.arch.active_parameters / tp_size) * total_step_tokens)
        attn_flops = 0.0
        if include_attention:
            heads_per_gpu = self.arch.num_q_heads / tp_size
            # Causal self-attention performs S^2 / 2 query-key and S^2 / 2 P-V interactions
            if self.arch.attention_variant == AttentionVariant.MLA:
                per_head_dim_work = self.arch.head_dim + self.arch.mla_rope_head_dim + self.arch.mla_v_head_dim
            else:
                per_head_dim_work = 2 * self.arch.head_dim
            attn_flops = float(
                self.arch.num_layers
                * batch_size
                * heads_per_gpu
                * per_head_dim_work
                * (prompt_tokens * prompt_tokens)
            )
        w_bytes = self._active_weight_bytes_read_per_gpu(
            total_step_tokens, tp_size=tp_size, ep_size=ep_size
        )
        kv_write_bytes = self.arch.kv_bytes_per_token(
            self.kv_layout, tp_size=tp_size, absorb_mla=True
        ) * total_step_tokens
        total_bytes = w_bytes + kv_write_bytes
        total_flops = linear_flops + attn_flops
        ai = total_flops / total_bytes
        ridge = self.accel.ridge_point(self.compute_format, mfu=mfu, mbu=mbu)
        p_eff = self.accel.peak_compute_tflops(self.compute_format) * 1e12 * mfu
        b_eff = self.accel.peak_hbm_bandwidth_gbps * 1e9 * mbu
        t_comp_ms = (total_flops / p_eff) * 1000.0
        t_mem_ms = (total_bytes / b_eff) * 1000.0
        return StepRooflineResult(
            phase=StepPhase.PREFILL,
            batch_size=batch_size,
            step_tokens=total_step_tokens,
            mean_context_len=prompt_tokens,
            linear_flops=linear_flops,
            attention_flops=attn_flops,
            weight_bytes_read=w_bytes,
            kv_bytes_transferred=kv_write_bytes,
            total_hbm_bytes=total_bytes,
            arithmetic_intensity=ai,
            ridge_point=self.accel.ridge_point(self.compute_format),
            compute_time_ms=t_comp_ms,
            memory_time_ms=t_mem_ms,
            step_time_ms=max(t_comp_ms, t_mem_ms),
        )

    def analyze_decode_step(
        self,
        batch_size: int,
        mean_context_len: int,
        *,
        tp_size: int = 1,
        ep_size: int = 1,
        mfu: float = 0.50,
        mbu: float = 0.80,
        absorb_mla: bool = True,
        include_attention: bool = True,
    ) -> StepRooflineResult:
        """Analyze one autoregressive decode step over ``B = batch_size`` active sequences."""
        if batch_size <= 0 or mean_context_len < 0 or tp_size <= 0 or ep_size <= 0:
            raise ValueError("batch_size, tp_size, ep_size > 0 and mean_context_len >= 0 required.")
        linear_flops = float(2.0 * (self.arch.active_parameters / tp_size) * batch_size)
        attn_flops = (
            self._decode_attention_flops_per_gpu(
                batch_size, mean_context_len, tp_size=tp_size, absorb_mla=absorb_mla
            )
            if include_attention
            else 0.0
        )
        w_bytes = self._active_weight_bytes_read_per_gpu(
            batch_size, tp_size=tp_size, ep_size=ep_size
        )
        kv_per_tok = self.arch.kv_bytes_per_token(
            self.kv_layout, tp_size=tp_size, absorb_mla=absorb_mla
        )
        # Read S historical KV tokens + write 1 newly generated token per sequence in batch
        kv_bytes = batch_size * (mean_context_len + 1) * kv_per_tok
        total_bytes = w_bytes + kv_bytes
        total_flops = linear_flops + attn_flops
        ai = total_flops / total_bytes
        p_eff = self.accel.peak_compute_tflops(self.compute_format) * 1e12 * mfu
        b_eff = self.accel.peak_hbm_bandwidth_gbps * 1e9 * mbu
        t_comp_ms = (total_flops / p_eff) * 1000.0
        t_mem_ms = (total_bytes / b_eff) * 1000.0
        return StepRooflineResult(
            phase=StepPhase.DECODE,
            batch_size=batch_size,
            step_tokens=batch_size,
            mean_context_len=mean_context_len,
            linear_flops=linear_flops,
            attention_flops=attn_flops,
            weight_bytes_read=w_bytes,
            kv_bytes_transferred=kv_bytes,
            total_hbm_bytes=total_bytes,
            arithmetic_intensity=ai,
            ridge_point=self.accel.ridge_point(self.compute_format),
            compute_time_ms=t_comp_ms,
            memory_time_ms=t_mem_ms,
            step_time_ms=max(t_comp_ms, t_mem_ms),
        )

    def critical_batch_size(
        self,
        mean_context_len: int = 0,
        *,
        tp_size: int = 1,
        mfu: float = 1.0,
        mbu: float = 1.0,
        absorb_mla: bool = True,
        max_search_batch: int = 65_536,
    ) -> int | None:
        """Smallest batch size B* reaching I(B*) >= I*_eff, or None if KV bandwidth caps I < I*."""
        target_ridge = self.accel.ridge_point(self.compute_format, mfu=mfu, mbu=mbu)
        # Asymptotic arithmetic intensity at B -> inf (numerator and denominator both ~ O(B))
        probe_large = self.analyze_decode_step(
            batch_size=1_000_000,
            mean_context_len=mean_context_len,
            tp_size=tp_size,
            mfu=mfu,
            mbu=mbu,
            absorb_mla=absorb_mla,
        )
        if probe_large.arithmetic_intensity < target_ridge:
            return None
        lo, hi = 1, max_search_batch
        while lo < hi:
            mid = (lo + hi) // 2
            res = self.analyze_decode_step(
                batch_size=mid,
                mean_context_len=mean_context_len,
                tp_size=tp_size,
                mfu=mfu,
                mbu=mbu,
                absorb_mla=absorb_mla,
            )
            if res.arithmetic_intensity >= target_ridge:
                hi = mid
            else:
                lo = mid + 1
        final = self.analyze_decode_step(
            batch_size=lo,
            mean_context_len=mean_context_len,
            tp_size=tp_size,
            mfu=mfu,
            mbu=mbu,
            absorb_mla=absorb_mla,
        )
        return lo if final.arithmetic_intensity >= target_ridge else None

    def analyze_moe_all_to_all(
        self,
        *,
        ep_size: int,
        local_batch_tokens: int,
        dispatch_format: PrecisionFormat = PrecisionFormat.FP8,
        combine_format: PrecisionFormat = PrecisionFormat.BF16,
        link_unidir_gbps: float | None = None,
        per_collective_latency_us: float = 15.0,
        skew_factor: float = 1.0,
    ) -> MoEAllToAllResult:
        """Quantify Expert Parallelism (EP) All-to-All communication and hot-expert straggler effects.

        Args:
            ep_size: Number of GPUs participating in Expert Parallelism.
            local_batch_tokens: Tokens arriving at each EP rank before dispatch.
            dispatch_format: Activation precision sent on Dispatch All-to-All (e.g., FP8 = 1 B).
            combine_format: Activation precision returned on Combine All-to-All (e.g., BF16 = 2 B).
            link_unidir_gbps: Bottleneck unidirectional fabric rate (NVLink within a node or NIC across nodes).
            per_collective_latency_us: Fixed kernel + RDMA/NVLink barrier launch overhead per collective.
            skew_factor: Ratio L_max / L_avg (>= 1.0) of tokens hitting the hottest EP rank.
        """
        if self.arch.num_moe_layers == 0:
            raise ValueError("MoE All-to-All analysis requires an architecture with num_moe_layers > 0.")
        if ep_size <= 1 or self.arch.num_routed_experts % ep_size != 0:
            raise ValueError("ep_size must be > 1 and divide num_routed_experts evenly.")
        if local_batch_tokens <= 0 or skew_factor < 1.0:
            raise ValueError("local_batch_tokens > 0 and skew_factor >= 1.0 required.")
        if skew_factor > ep_size:
            raise ValueError("skew_factor cannot exceed ep_size (all global tokens on one rank).")

        effective_link_gbps = (
            link_unidir_gbps
            if link_unidir_gbps is not None
            else (self.accel.nvlink_unidir_gbps if ep_size <= 8 else self.accel.nic_unidir_gbps)
        )
        global_tokens = local_batch_tokens * ep_size
        k = self.arch.num_active_experts
        remote_fraction = 1.0 - (1.0 / ep_size)
        msg_bytes_per_token_route = round(
            self.arch.hidden_size
            * (dispatch_format.raw_bytes_per_element + combine_format.raw_bytes_per_element)
        )
        uniform_bytes_layer = round(
            local_batch_tokens * k * remote_fraction * msg_bytes_per_token_route
        )
        # Under incast skew alpha = L_max / L_avg, the hottest rank's NIC/NVLink port receives/returns
        # alpha * remote_fraction * L_avg routed token payloads across all peers.
        skewed_bottleneck_bytes_layer = round(uniform_bytes_layer * skew_factor)

        link_bps = effective_link_gbps * 1e9
        collective_overhead_ms = (2.0 * per_collective_latency_us) / 1000.0
        uniform_ms_layer = (uniform_bytes_layer / link_bps) * 1000.0 + collective_overhead_ms
        skewed_ms_layer = (skewed_bottleneck_bytes_layer / link_bps) * 1000.0 + collective_overhead_ms

        w_bytes_per_rank = self.arch.weight_bytes_per_gpu(
            self.weight_layout, tp_size=1, pp_size=1, ep_size=ep_size
        )
        touched_global = self.expected_touched_experts(
            global_tokens, self.arch.num_routed_experts, k
        )
        return MoEAllToAllResult(
            ep_size=ep_size,
            local_batch_tokens=local_batch_tokens,
            global_batch_tokens=global_tokens,
            weight_bytes_per_rank=w_bytes_per_rank,
            expected_unique_experts_per_rank=touched_global / ep_size,
            uniform_bytes_per_rank_per_layer=uniform_bytes_layer,
            skewed_bottleneck_bytes_per_layer=skewed_bottleneck_bytes_layer,
            uniform_comm_ms_all_layers=uniform_ms_layer * self.arch.num_moe_layers,
            skewed_comm_ms_all_layers=skewed_ms_layer * self.arch.num_moe_layers,
            skew_factor=skew_factor,
            straggler_ffn_flops_amplification=skew_factor,
            straggler_hbm_activation_bytes_amplification=skew_factor,
        )
