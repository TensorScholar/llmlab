"""Exact KV-cache, MLA up-projection absorption, and static weight byte accounting."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Final

from llmlab.memory.hardware import AcceleratorSpec, PrecisionFormat
from llmlab.tracing.budgets import MemoryBudget


class AttentionVariant(str, Enum):
    """Transformer KV-cache sharing and latent compression architectures."""

    MHA = "mha"
    GQA = "gqa"
    MLA = "mla"


@dataclass(frozen=True, slots=True)
class QuantizationLayout:
    """Bit layout including payload precision and block-quantization scale/zero-point overhead."""

    format: PrecisionFormat
    group_size: int | None = None
    scale_bytes_per_group: int = 0

    def __post_init__(self) -> None:
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError("group_size must be strictly positive when provided.")
        if self.scale_bytes_per_group < 0:
            raise ValueError("scale_bytes_per_group must be non-negative.")
        if self.group_size is None and self.scale_bytes_per_group > 0:
            raise ValueError("scale_bytes_per_group > 0 requires an explicit group_size.")

    @property
    def effective_bytes_per_element(self) -> float:
        """Amortized bytes per tensor element including group metadata."""
        raw = self.format.raw_bytes_per_element
        if self.group_size is None or self.scale_bytes_per_group == 0:
            return raw
        return raw + (self.scale_bytes_per_group / self.group_size)

    def tensor_bytes(self, num_elements: int) -> int:
        """Exact byte footprint of ``num_elements`` packed under this quantization layout."""
        if num_elements < 0:
            raise ValueError("num_elements must be non-negative.")
        if num_elements == 0:
            return 0
        payload_bytes = math.ceil((num_elements * self.format.raw_bits_per_element) / 8)
        if self.group_size is None or self.scale_bytes_per_group == 0:
            return payload_bytes
        num_groups = math.ceil(num_elements / self.group_size)
        return payload_bytes + (num_groups * self.scale_bytes_per_group)


LAYOUT_BF16: Final[QuantizationLayout] = QuantizationLayout(PrecisionFormat.BF16)
LAYOUT_FP16: Final[QuantizationLayout] = QuantizationLayout(PrecisionFormat.FP16)
LAYOUT_FP8: Final[QuantizationLayout] = QuantizationLayout(PrecisionFormat.FP8)
LAYOUT_INT8: Final[QuantizationLayout] = QuantizationLayout(PrecisionFormat.INT8)
# Asymmetric INT4 with 1 FP16 scale (2 B) + 1 FP16 zero-point (2 B) per 128 elements (0.53125 B/elem)
LAYOUT_INT4_G128_ASYM: Final[QuantizationLayout] = QuantizationLayout(
    PrecisionFormat.INT4,
    group_size=128,
    scale_bytes_per_group=4,
)
# Symmetric INT4 (e.g., AWQ) with 1 FP16 scale (2 B) per 128 elements (0.515625 B/elem)
LAYOUT_INT4_G128_SYM: Final[QuantizationLayout] = QuantizationLayout(
    PrecisionFormat.INT4,
    group_size=128,
    scale_bytes_per_group=2,
)
# Blackwell NVFP4 microscaling: 4-bit E2M1 payload + 1-byte FP8/E8M0 scale per 16 elements
LAYOUT_NVFP4: Final[QuantizationLayout] = QuantizationLayout(
    PrecisionFormat.NVFP4,
    group_size=16,
    scale_bytes_per_group=1,
)


@dataclass(frozen=True, slots=True)
class TransformerArchitecture:
    """Complete attention, dense/MoE parameter, and latent projection dimensions."""

    name: str
    num_layers: int
    hidden_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    attention_variant: AttentionVariant
    total_parameters: int
    active_parameters: int
    mla_kv_lora_rank: int = 0
    mla_rope_head_dim: int = 0
    mla_v_head_dim: int = 0
    mla_q_lora_rank: int = 0
    num_moe_layers: int = 0
    num_routed_experts: int = 1
    num_active_experts: int = 1
    num_shared_experts: int = 0
    moe_intermediate_size: int = 0

    def __post_init__(self) -> None:
        if self.num_layers <= 0 or self.hidden_size <= 0 or self.num_q_heads <= 0 or self.head_dim <= 0:
            raise ValueError("Layer count, hidden size, head count, and head_dim must be positive.")
        if self.total_parameters <= 0 or self.active_parameters <= 0:
            raise ValueError("Model parameter counts must be positive.")
        if self.active_parameters > self.total_parameters:
            raise ValueError("active_parameters cannot exceed total_parameters.")
        if self.attention_variant == AttentionVariant.MLA:
            if self.mla_kv_lora_rank <= 0 or self.mla_rope_head_dim < 0 or self.mla_v_head_dim <= 0:
                raise ValueError("MLA architectures require positive mla_kv_lora_rank and mla_v_head_dim.")
        elif self.num_kv_heads <= 0 or self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("MHA/GQA requires num_kv_heads > 0 dividing num_q_heads evenly.")
        if self.num_moe_layers > 0:
            if self.num_active_experts <= 0 or self.num_active_experts > self.num_routed_experts:
                raise ValueError("MoE top-k num_active_experts must lie in [1, num_routed_experts].")
            if self.moe_intermediate_size <= 0:
                raise ValueError("MoE layers require positive moe_intermediate_size.")

    @property
    def routed_expert_params_per_expert_layer(self) -> int:
        """Parameter count of one SwiGLU routed expert subnetwork (gate, up, down projections)."""
        if self.num_moe_layers == 0:
            return 0
        return 3 * self.hidden_size * self.moe_intermediate_size

    @property
    def total_routed_expert_parameters(self) -> int:
        """Total parameters across all routed experts across all MoE layers."""
        return (
            self.num_moe_layers
            * self.num_routed_experts
            * self.routed_expert_params_per_expert_layer
        )

    @property
    def non_routed_parameters(self) -> int:
        """Dense backbone, embeddings, attention, and shared-expert parameters replicated or TP-sharded."""
        return max(0, self.total_parameters - self.total_routed_expert_parameters)

    def kv_elements_per_token_per_layer(self, *, absorb_mla: bool = True) -> int:
        """Resident scalar elements stored per token per layer in the persistent KV pool.

        For MHA/GQA, both K and V have ``num_kv_heads * head_dim`` elements per layer.
        For MLA with matrix absorption (``absorb_mla=True``), only the compressed latent
        ``c_t^{KV}`` (rank ``d_c``) plus decoupled RoPE key ``k_t^R`` (dim ``d_R``) reside
        in KV cache. Without absorption, materializing expanded K and V requires storing or
        recomputing ``n_h * (head_dim_nope + mla_v_head_dim)`` elements per token per layer.
        """
        if self.attention_variant in (AttentionVariant.MHA, AttentionVariant.GQA):
            return 2 * self.num_kv_heads * self.head_dim
        if absorb_mla:
            return self.mla_kv_lora_rank + self.mla_rope_head_dim
        return self.num_q_heads * (self.head_dim + self.mla_v_head_dim) + self.mla_rope_head_dim

    def kv_bytes_per_token(
        self,
        kv_layout: QuantizationLayout,
        *,
        tp_size: int = 1,
        absorb_mla: bool = True,
        replicate_mla_across_tp: bool = True,
    ) -> int:
        """Exact per-GPU persistent KV-cache bytes per token across all layers."""
        if tp_size <= 0:
            raise ValueError("tp_size must be positive.")
        if self.attention_variant in (AttentionVariant.MHA, AttentionVariant.GQA):
            if self.num_kv_heads % tp_size == 0:
                kv_heads_per_rank = self.num_kv_heads // tp_size
            elif tp_size > self.num_kv_heads and tp_size % self.num_kv_heads == 0:
                kv_heads_per_rank = 1
            else:
                raise ValueError(
                    f"tp_size={tp_size} must divide num_kv_heads={self.num_kv_heads} (or be a multiple)."
                )
            per_layer_bytes = kv_layout.tensor_bytes(2 * kv_heads_per_rank * self.head_dim)
            return self.num_layers * per_layer_bytes

        # MLA stores a single compressed latent vector per token per layer (n_kv = 1).
        if not absorb_mla:
            if self.num_q_heads % tp_size != 0:
                raise ValueError(f"tp_size={tp_size} must divide num_q_heads={self.num_q_heads}.")
            q_heads_per_rank = self.num_q_heads // tp_size
            per_layer_bytes = kv_layout.tensor_bytes(
                q_heads_per_rank * (self.head_dim + self.mla_v_head_dim) + self.mla_rope_head_dim
            )
            return self.num_layers * per_layer_bytes

        latent_elems = self.mla_kv_lora_rank + self.mla_rope_head_dim
        per_layer_bytes = kv_layout.tensor_bytes(latent_elems)
        if replicate_mla_across_tp or tp_size == 1:
            return self.num_layers * per_layer_bytes
        # Context/data-parallel MLA attention partitions sequence tokens rather than heads
        return math.ceil((self.num_layers * per_layer_bytes) / tp_size)

    def mla_decompression_workspace_bytes(
        self,
        workspace_tokens: int,
        act_layout: QuantizationLayout,
        *,
        concurrent_layers: int = 1,
        tp_size: int = 1,
        include_rope_broadcast: bool = True,
    ) -> int:
        """Transient HBM scratchpad bytes needed when MLA does NOT absorb W^{UK}, W^{UV}.

        During unabsorbed prefill or naive decode without matrix absorption, the serving kernel
        up-projects latent vectors ``c_t^{KV}`` back into per-head ``[B*S_buf, n_h/TP, d_nope + d_v]``
        (plus optional broadcast ``k_t^R`` of dim ``d_R`` per head for contiguous FlashAttention).
        """
        if workspace_tokens < 0 or concurrent_layers <= 0 or tp_size <= 0:
            raise ValueError("workspace_tokens >= 0, concurrent_layers > 0, tp_size > 0 required.")
        if self.attention_variant != AttentionVariant.MLA or workspace_tokens == 0:
            return 0
        if self.num_q_heads % tp_size != 0:
            raise ValueError(f"tp_size={tp_size} must divide num_q_heads={self.num_q_heads}.")
        q_heads_per_rank = self.num_q_heads // tp_size
        per_head_elems = self.head_dim + self.mla_v_head_dim + (
            self.mla_rope_head_dim if include_rope_broadcast else 0
        )
        elems_per_layer = workspace_tokens * q_heads_per_rank * per_head_elems
        return concurrent_layers * act_layout.tensor_bytes(elems_per_layer)

    def weight_bytes_per_gpu(
        self,
        weight_layout: QuantizationLayout,
        *,
        tp_size: int = 1,
        pp_size: int = 1,
        ep_size: int = 1,
        active_only_misconception: bool = False,
    ) -> int:
        """Static weight HBM bytes resident per GPU under TP/PP/EP sharding.

        Setting ``active_only_misconception=True`` intentionally reproduces the anti-pattern
        of sizing HBM using only ``active_parameters`` for MoE checkpoints (negative control).
        """
        if tp_size <= 0 or pp_size <= 0 or ep_size <= 0:
            raise ValueError("Parallelism degrees tp_size, pp_size, and ep_size must be positive.")
        if active_only_misconception:
            per_gpu_params = math.ceil(self.active_parameters / (tp_size * pp_size))
            return weight_layout.tensor_bytes(per_gpu_params)
        if self.num_moe_layers == 0:
            per_gpu_params = math.ceil(self.total_parameters / (tp_size * pp_size))
            return weight_layout.tensor_bytes(per_gpu_params)
        if self.num_routed_experts % ep_size != 0:
            raise ValueError(
                f"ep_size={ep_size} must divide num_routed_experts={self.num_routed_experts}."
            )
        # Non-routed backbone parameters shard across TP * PP; routed MoE experts shard across EP * PP
        # (assuming standard EP over the world size tp_size * dp_size, or explicit ep_size per PP stage).
        dense_params_per_gpu = math.ceil(self.non_routed_parameters / (tp_size * pp_size))
        routed_params_per_gpu = math.ceil(self.total_routed_expert_parameters / (ep_size * pp_size))
        return weight_layout.tensor_bytes(dense_params_per_gpu + routed_params_per_gpu)

    def to_memory_budget(
        self,
        accel: AcceleratorSpec,
        weight_layout: QuantizationLayout,
        kv_layout: QuantizationLayout,
        *,
        compute_format: PrecisionFormat = PrecisionFormat.BF16,
        tp_size: int = 1,
        pp_size: int = 1,
        ep_size: int = 1,
        activation_bytes_per_token: int = 65_536,
        system_reserve_bytes: int = 2 * (1024**3),
        mla_workspace_bytes: int = 0,
        active_only_misconception: bool = False,
    ) -> MemoryBudget:
        """Bridge Chapter 2 byte accounting directly into Chapter 1's MemoryBudget contract."""
        w_bytes = self.weight_bytes_per_gpu(
            weight_layout,
            tp_size=tp_size,
            pp_size=pp_size,
            ep_size=ep_size,
            active_only_misconception=active_only_misconception,
        )
        kv_per_tok = self.kv_bytes_per_token(kv_layout, tp_size=tp_size, absorb_mla=True)
        return MemoryBudget(
            hbm_capacity_bytes=accel.usable_hbm_bytes,
            weight_bytes=w_bytes,
            kv_bytes_per_token=kv_per_tok,
            activation_bytes_per_token=activation_bytes_per_token,
            system_reserve_bytes=system_reserve_bytes + mla_workspace_bytes,
            peak_compute_tflops=accel.peak_compute_tflops(compute_format),
            peak_hbm_bandwidth_gbps=accel.peak_hbm_bandwidth_gbps,
        )


# Standard reference architectures verified against published configurations
LLAMA_70B_HYPOTHETICAL_MHA: Final[TransformerArchitecture] = TransformerArchitecture(
    name="Dense-70B-MHA (Baseline)",
    num_layers=80,
    hidden_size=8192,
    num_q_heads=64,
    num_kv_heads=64,
    head_dim=128,
    attention_variant=AttentionVariant.MHA,
    total_parameters=70_000_000_000,
    active_parameters=70_000_000_000,
)

LLAMA_70B_GQA: Final[TransformerArchitecture] = TransformerArchitecture(
    name="Dense-70B-GQA (8 KV Heads)",
    num_layers=80,
    hidden_size=8192,
    num_q_heads=64,
    num_kv_heads=8,
    head_dim=128,
    attention_variant=AttentionVariant.GQA,
    total_parameters=70_000_000_000,
    active_parameters=70_000_000_000,
)

DEEPSEEK_V3_MLA_MOE: Final[TransformerArchitecture] = TransformerArchitecture(
    name="DeepSeek-V3 (671B-A37B MLA+MoE)",
    num_layers=61,
    hidden_size=7168,
    num_q_heads=128,
    num_kv_heads=1,
    head_dim=128,
    attention_variant=AttentionVariant.MLA,
    total_parameters=671_000_000_000,
    active_parameters=37_000_000_000,
    mla_kv_lora_rank=512,
    mla_rope_head_dim=64,
    mla_v_head_dim=128,
    mla_q_lora_rank=1536,
    num_moe_layers=58,
    num_routed_experts=256,
    num_active_experts=8,
    num_shared_experts=1,
    moe_intermediate_size=2048,
)
