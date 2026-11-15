"""Prefill-Decode (P/D) disaggregation, KV migration fabrics, and heterogeneous pool sizing."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from llmlab.loadgen.capacity import CapacityPlan, WorkloadProfile, plan_cluster_capacity
from llmlab.memory.hardware import AcceleratorSpec, PrecisionFormat
from llmlab.memory.kv_cache import QuantizationLayout, TransformerArchitecture
from llmlab.memory.roofline import RooflineAnalyzer
from llmlab.tracing.budgets import LatencyBudget, MemoryBudget


class KVMigrationBandwidthError(RuntimeError):
    """Raised when P->D KV tensor migration exceeds remaining TTFT SLO headroom."""


@dataclass(frozen=True, slots=True)
class DisaggFabricSpec:
    """Point-to-point interconnect specification governing P->D KV cache transfer."""

    name: str
    protocol: Literal["NVLink", "InfiniBand_NDR", "RDMA_RoCEv2"]
    unidir_gbps: float
    bidir_gbps: float
    wire_efficiency: float = 0.85
    base_setup_ms: float = 0.05

    def __post_init__(self) -> None:
        if self.unidir_gbps <= 0.0 or self.bidir_gbps < self.unidir_gbps:
            raise ValueError("0 < unidir_gbps <= bidir_gbps required.")
        if not (0.1 <= self.wire_efficiency <= 1.0):
            raise ValueError("wire_efficiency must lie in [0.1, 1.0].")
        if self.base_setup_ms < 0.0:
            raise ValueError("base_setup_ms must be non-negative.")

    @property
    def effective_unidir_gbps(self) -> float:
        """Usable single-direction payload throughput (GB/s) after RDMA/NVSwitch framing."""
        return self.unidir_gbps * self.wire_efficiency

    @classmethod
    def from_accelerator_nvlink(cls, accel: AcceleratorSpec) -> DisaggFabricSpec:
        """Construct intra-domain NVLink P/D transfer fabric directly from AcceleratorSpec."""
        return cls(
            name=f"{accel.name}_NVLink",
            protocol="NVLink",
            unidir_gbps=accel.nvlink_unidir_gbps,
            bidir_gbps=accel.nvlink_bidir_gbps,
            wire_efficiency=0.88,
            base_setup_ms=0.02,
        )

    @classmethod
    def from_accelerator_nic(cls, accel: AcceleratorSpec) -> DisaggFabricSpec:
        """Construct single-NIC InfiniBand/RoCEv2 RDMA fabric from AcceleratorSpec."""
        return cls(
            name=f"{accel.name}_NIC",
            protocol="RDMA_RoCEv2",
            unidir_gbps=accel.nic_unidir_gbps,
            bidir_gbps=accel.nic_unidir_gbps * 2.0,
            wire_efficiency=0.82,
            base_setup_ms=0.08,
        )


FABRIC_NVLINK4_H100 = DisaggFabricSpec(
    name="NVLink4_900GBs_Bidir",
    protocol="NVLink",
    unidir_gbps=450.0,
    bidir_gbps=900.0,
    wire_efficiency=0.88,
    base_setup_ms=0.02,
)

FABRIC_NVLINK5_B200 = DisaggFabricSpec(
    name="NVLink5_1800GBs_Bidir",
    protocol="NVLink",
    unidir_gbps=900.0,
    bidir_gbps=1800.0,
    wire_efficiency=0.88,
    base_setup_ms=0.02,
)

FABRIC_RDMA_800GBE = DisaggFabricSpec(
    name="InfiniBand_NDR_800GbE",
    protocol="InfiniBand_NDR",
    unidir_gbps=100.0,
    bidir_gbps=200.0,
    wire_efficiency=0.82,
    base_setup_ms=0.08,
)

FABRIC_RDMA_400GBE = DisaggFabricSpec(
    name="RoCEv2_400GbE",
    protocol="RDMA_RoCEv2",
    unidir_gbps=50.0,
    bidir_gbps=100.0,
    wire_efficiency=0.82,
    base_setup_ms=0.08,
)

FABRIC_RDMA_100GBE = DisaggFabricSpec(
    name="Ethernet_100GbE",
    protocol="RDMA_RoCEv2",
    unidir_gbps=12.5,
    bidir_gbps=25.0,
    wire_efficiency=0.80,
    base_setup_ms=0.15,
)


@dataclass(frozen=True, slots=True)
class PDTransferEstimate:
    """Exact TTFT budget breakdown across prefill compute, P->D KV migration, and first decode."""

    prompt_tokens: int
    uncached_tokens: int
    kv_transfer_bytes: int
    kv_transfer_mib: float
    fabric_name: str
    effective_bw_gbps: float
    pipelined_by_layer: bool
    prefill_compute_ms: float
    serial_transfer_ms: float
    exposed_transfer_ms: float
    first_decode_ms: float
    total_ttft_ms: float
    ttft_slo_ms: float
    within_slo: bool
    bottleneck: Literal["compute_hidden", "link_bandwidth"]


@dataclass(frozen=True, slots=True)
class DisaggTopologyPlan:
    """Heterogeneous P/D cluster sizing compared directly against colocated serving."""

    prefill_accel_name: str
    decode_accel_name: str
    fabric_name: str
    prefill_tp: int
    decode_tp: int
    prefill_instances: int
    decode_instances: int
    prefill_gpus: int
    decode_gpus: int
    total_disagg_gpus: int
    prefill_to_decode_ratio: float
    prefill_rps_per_instance: float
    decode_rps_per_instance: float
    decode_binding_constraint: Literal["kv_capacity", "compute_latency"]
    max_decode_kv_seqs_per_instance: int
    decode_tpot_ms: float
    long_prompt_transfer: PDTransferEstimate
    colocated_ha_gpus: int
    disagg_total_ha_gpus: int


def evaluate_kv_migration(
    *,
    prompt_tokens: int,
    uncached_tokens: int | None = None,
    arch: TransformerArchitecture,
    prefill_accel: AcceleratorSpec,
    decode_accel: AcceleratorSpec,
    weight_layout: QuantizationLayout,
    kv_layout: QuantizationLayout,
    compute_format: PrecisionFormat,
    fabric: DisaggFabricSpec,
    latency_budget: LatencyBudget,
    prefill_tp: int = 1,
    decode_tp: int = 1,
    ep_size: int = 1,
    nics_per_instance: int = 1,
    pipelined_by_layer: bool = True,
    decode_batch_context: int = 32,
    mfu: float = 0.50,
    mbu: float = 0.80,
    absorb_mla: bool = True,
    enforce_slo: bool = False,
) -> PDTransferEstimate:
    """Compute blocking vs. layer-pipelined KV migration time across an explicit interconnect."""
    if prompt_tokens < 1 or nics_per_instance < 1:
        raise ValueError("prompt_tokens and nics_per_instance must be >= 1.")
    transfer_tokens = prompt_tokens if uncached_tokens is None else max(1, uncached_tokens)

    pre_analyzer = RooflineAnalyzer(arch, prefill_accel, weight_layout, kv_layout, compute_format)
    dec_analyzer = RooflineAnalyzer(arch, decode_accel, weight_layout, kv_layout, compute_format)

    pre_step = pre_analyzer.analyze_prefill_step(
        transfer_tokens, batch_size=1, tp_size=prefill_tp, ep_size=ep_size, mfu=mfu, mbu=mbu
    )
    first_dec = dec_analyzer.analyze_decode_step(
        max(1, decode_batch_context),
        prompt_tokens,
        tp_size=decode_tp,
        ep_size=ep_size,
        mfu=mfu,
        mbu=mbu,
        absorb_mla=absorb_mla,
    )

    kv_b_per_tok = arch.kv_bytes_per_token(kv_layout, tp_size=1, absorb_mla=absorb_mla)
    kv_bytes = transfer_tokens * kv_b_per_tok
    kv_mib = kv_bytes / float(1024 * 1024)

    eff_bw_gbps = fabric.effective_unidir_gbps * float(nics_in_flight(fabric, nics_per_instance, prefill_tp))
    # Convert GB/s (= 1e9 bytes/s) to ms: (bytes / (eff_bw_gbps * 1e9)) * 1000
    payload_xfer_ms = (kv_bytes / (eff_bw_gbps * 1.0e9)) * 1000.0
    serial_transfer_ms = fabric.base_setup_ms + payload_xfer_ms

    num_layers = arch.num_layers
    if pipelined_by_layer and num_layers > 1:
        t_layer_comp = pre_step.step_time_ms / float(num_layers)
        t_layer_xfer = payload_xfer_ms / float(num_layers)
        # Closed-form two-stage linear pipeline: final layer + cumulative lag when xfer > comp
        lag_ms = max(0.0, (num_layers - 1) * (t_layer_xfer - t_layer_comp))
        exposed_ms = fabric.base_setup_ms + t_layer_xfer + lag_ms
        bottleneck: Literal["compute_hidden", "link_bandwidth"] = (
            "link_bandwidth" if t_layer_xfer > t_layer_comp else "compute_hidden"
        )
    else:
        exposed_ms = serial_transfer_ms
        bottleneck = "link_bandwidth"

    total_ttft_ms = pre_step.step_time_ms + exposed_ms + first_dec.step_time_ms
    within_slo = total_ttft_ms <= latency_budget.max_ttft_ms

    estimate = PDTransferEstimate(
        prompt_tokens=prompt_tokens,
        uncached_tokens=transfer_tokens,
        kv_transfer_bytes=kv_bytes,
        kv_transfer_mib=kv_mib,
        fabric_name=fabric.name,
        effective_bw_gbps=eff_bw_gbps,
        pipelined_by_layer=pipelined_by_layer,
        prefill_compute_ms=pre_step.step_time_ms,
        serial_transfer_ms=serial_transfer_ms,
        exposed_transfer_ms=exposed_ms,
        first_decode_ms=first_dec.step_time_ms,
        total_ttft_ms=total_ttft_ms,
        ttft_slo_ms=latency_budget.max_ttft_ms,
        within_slo=within_slo,
        bottleneck=bottleneck,
    )
    if enforce_slo and not within_slo:
        raise KVMigrationBandwidthError(
            f"P->D KV migration ({kv_mib:.1f} MiB over {fabric.name} at {eff_bw_gbps:.1f} GB/s) "
            f"yields TTFT {total_ttft_ms:.1f} ms > SLO {latency_budget.max_ttft_ms:.1f} ms "
            f"(exposed_transfer={exposed_ms:.1f} ms)."
        )
    return estimate


def nics_in_flight(fabric: DisaggFabricSpec, nics_per_instance: int, prefill_tp: int) -> int:
    """Bound concurrent NICs by TP rank rail symmetry on Ethernet/IB, or single P2P NVLink path."""
    if fabric.protocol == "NVLink":
        return min(nics_per_instance, prefill_tp)
    return min(nics_per_instance, prefill_tp)


def plan_disaggregated_topology(
    *,
    workload: WorkloadProfile,
    arch: TransformerArchitecture,
    prefill_accel: AcceleratorSpec,
    decode_accel: AcceleratorSpec,
    weight_layout: QuantizationLayout,
    kv_layout: QuantizationLayout,
    compute_format: PrecisionFormat,
    latency_budget: LatencyBudget,
    fabric: DisaggFabricSpec,
    prefill_tp: int = 1,
    decode_tp: int = 1,
    ep_size: int = 1,
    nics_per_instance: int = 1,
    pipelined_by_layer: bool = True,
    mfu: float = 0.50,
    mbu: float = 0.80,
    absorb_mla: bool = True,
    redundancy_k_prefill: int = 1,
    redundancy_k_decode: int = 1,
    enforce_slo: bool = False,
) -> DisaggTopologyPlan:
    """Size independent P and D pools for bimodal traffic and compare against colocated capacity."""
    pre_analyzer = RooflineAnalyzer(arch, prefill_accel, weight_layout, kv_layout, compute_format)
    dec_analyzer = RooflineAnalyzer(arch, decode_accel, weight_layout, kv_layout, compute_format)

    s_short_unc = max(1, int(round(workload.short_prompt_tokens * (1.0 - workload.prefix_hit_ratio))))
    s_long_unc = max(1, int(round(workload.long_prompt_tokens * (1.0 - workload.prefix_hit_ratio))))

    pre_short = pre_analyzer.analyze_prefill_step(
        s_short_unc, batch_size=1, tp_size=prefill_tp, ep_size=ep_size, mfu=mfu, mbu=mbu
    )
    pre_long = pre_analyzer.analyze_prefill_step(
        s_long_unc, batch_size=1, tp_size=prefill_tp, ep_size=ep_size, mfu=mfu, mbu=mbu
    )
    mean_prefill_ms = (
        (1.0 - workload.p_long_prompt) * pre_short.step_time_ms
        + workload.p_long_prompt * pre_long.step_time_ms
    )
    prefill_rps_per_inst = workload.target_compute_util / max(1e-6, mean_prefill_ms / 1000.0)
    prefill_instances = max(1, math.ceil(workload.target_rps / prefill_rps_per_inst))

    # Decode pool: dual-bound by TPOT latency ceiling AND Little's Law resident KV pool
    dec_budget: MemoryBudget = arch.to_memory_budget(
        decode_accel,
        weight_layout,
        kv_layout,
        compute_format=compute_format,
        tp_size=decode_tp,
        ep_size=ep_size,
    )
    max_kv_tokens = dec_budget.max_resident_kv_tokens(step_tokens=256)
    usable_kv_tokens = max(0, max_kv_tokens - workload.shared_system_prefix_tokens)
    # On D pool, active slots must hold full prompt + generated suffix for mean decode sojourn
    mean_total_session_kv = max(
        1.0,
        (workload.mean_prompt_tokens - workload.shared_system_prefix_tokens)
        + (workload.mean_output_tokens / 2.0),
    )
    max_kv_seqs = max(0, int(usable_kv_tokens // mean_total_session_kv))

    mean_ctx_int = max(1, int(round(workload.mean_prompt_tokens + workload.mean_output_tokens / 2.0)))
    max_tpot_batch = 1
    for b_cand in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512):
        dec_step = dec_analyzer.analyze_decode_step(
            b_cand, mean_ctx_int, tp_size=decode_tp, ep_size=ep_size, mfu=mfu, mbu=mbu, absorb_mla=absorb_mla
        )
        if dec_step.step_time_ms <= latency_budget.max_tpot_ms:
            max_tpot_batch = b_cand
        else:
            break

    eff_dec_batch = max(1, min(max_tpot_batch, max(1, max_kv_seqs)))
    dec_op = dec_analyzer.analyze_decode_step(
        eff_dec_batch, mean_ctx_int, tp_size=decode_tp, ep_size=ep_size, mfu=mfu, mbu=mbu, absorb_mla=absorb_mla
    )
    decode_tpot_ms = dec_op.step_time_ms
    decode_sojourn_s = ((workload.mean_output_tokens - 1) * decode_tpot_ms) / 1000.0

    # Dedicated decode compute RPS when zero prefill interference exists
    dec_slo_step = dec_analyzer.analyze_decode_step(
        max_tpot_batch, mean_ctx_int, tp_size=decode_tp, ep_size=ep_size, mfu=mfu, mbu=mbu, absorb_mla=absorb_mla
    )
    dec_compute_duty_s = ((workload.mean_output_tokens - 1) * dec_slo_step.step_time_ms) / (
        1000.0 * float(max_tpot_batch)
    )
    dec_compute_rps = workload.target_compute_util / max(1e-6, dec_compute_duty_s)
    dec_kv_rps = (
        1e-6
        if max_kv_seqs == 0
        else workload.target_kv_util * (float(max_kv_seqs) / max(1e-6, decode_sojourn_s))
    )

    decode_rps_per_inst = min(dec_compute_rps, dec_kv_rps)
    dec_binding: Literal["kv_capacity", "compute_latency"] = (
        "kv_capacity" if dec_kv_rps <= dec_compute_rps else "compute_latency"
    )
    decode_instances = max(1, math.ceil(workload.target_rps / decode_rps_per_inst))

    p_gpus_per_inst = max(prefill_tp, ep_size if ep_size > 1 else prefill_tp)
    d_gpus_per_inst = max(decode_tp, ep_size if ep_size > 1 else decode_tp)
    prefill_gpus = prefill_instances * p_gpus_per_inst
    decode_gpus = decode_instances * d_gpus_per_inst

    long_transfer = evaluate_kv_migration(
        prompt_tokens=workload.long_prompt_tokens,
        uncached_tokens=s_long_unc,
        arch=arch,
        prefill_accel=prefill_accel,
        decode_accel=decode_accel,
        weight_layout=weight_layout,
        kv_layout=kv_layout,
        compute_format=compute_format,
        fabric=fabric,
        latency_budget=latency_budget,
        prefill_tp=prefill_tp,
        decode_tp=decode_tp,
        ep_size=ep_size,
        nics_per_instance=nics_per_instance,
        pipelined_by_layer=pipelined_by_layer,
        decode_batch_context=eff_dec_batch,
        mfu=mfu,
        mbu=mbu,
        absorb_mla=absorb_mla,
        enforce_slo=enforce_slo,
    )

    colocated: CapacityPlan = plan_cluster_capacity(
        workload=workload,
        arch=arch,
        accel=decode_accel,
        weight_layout=weight_layout,
        kv_layout=kv_layout,
        compute_format=compute_format,
        latency_budget=latency_budget,
        tp_size=decode_tp,
        ep_size=ep_size,
        mfu=mfu,
        mbu=mbu,
        absorb_mla=absorb_mla,
        redundancy_k=max(redundancy_k_prefill, redundancy_k_decode),
    )

    disagg_ha_gpus = (
        (prefill_instances + redundancy_k_prefill) * p_gpus_per_inst
        + (decode_instances + redundancy_k_decode) * d_gpus_per_inst
    )

    return DisaggTopologyPlan(
        prefill_accel_name=prefill_accel.name,
        decode_accel_name=decode_accel.name,
        fabric_name=fabric.name,
        prefill_tp=prefill_tp,
        decode_tp=decode_tp,
        prefill_instances=prefill_instances,
        decode_instances=decode_instances,
        prefill_gpus=prefill_gpus,
        decode_gpus=decode_gpus,
        total_disagg_gpus=prefill_gpus + decode_gpus,
        prefill_to_decode_ratio=prefill_instances / float(decode_instances),
        prefill_rps_per_instance=prefill_rps_per_inst,
        decode_rps_per_instance=decode_rps_per_inst,
        decode_binding_constraint=dec_binding,
        max_decode_kv_seqs_per_instance=max_kv_seqs,
        decode_tpot_ms=decode_tpot_ms,
        long_prompt_transfer=long_transfer,
        colocated_ha_gpus=colocated.total_ha_gpus,
        disagg_total_ha_gpus=disagg_ha_gpus,
    )
