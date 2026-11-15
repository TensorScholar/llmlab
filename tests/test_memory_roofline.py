"""Unit and negative-control tests for Chapter 2: llmlab.memory hardware, KV cache, and Roofline."""
from __future__ import annotations

import math

import pytest

from llmlab.memory import (
    B200_SXM,
    DEEPSEEK_V3_MLA_MOE,
    H100_SXM5,
    H200_SXM,
    LAYOUT_BF16,
    LAYOUT_FP8,
    LAYOUT_FP16,
    LAYOUT_INT4_G128_ASYM,
    LAYOUT_INT4_G128_SYM,
    LAYOUT_NVFP4,
    LLAMA_70B_GQA,
    LLAMA_70B_HYPOTHETICAL_MHA,
    PrecisionFormat,
    RooflineAnalyzer,
    UnsupportedPrecisionError,
)
from llmlab.tracing.budgets import MemoryBudgetExceeded


def test_hardware_ridge_points_and_hierarchy_specs() -> None:
    """Verify exact Hopper (H100/H200) and Blackwell (B200) ridge points and native precision checks."""
    assert math.isclose(H100_SXM5.ridge_point(PrecisionFormat.FP16), 989.4 / 3.35, rel_tol=1e-6)
    assert math.isclose(H100_SXM5.ridge_point(PrecisionFormat.FP8), 1978.8 / 3.35, rel_tol=1e-6)
    assert math.isclose(H200_SXM.ridge_point(PrecisionFormat.FP16), 989.4 / 4.80, rel_tol=1e-6)
    assert math.isclose(H200_SXM.ridge_point(PrecisionFormat.FP8), 1978.8 / 4.80, rel_tol=1e-6)
    assert math.isclose(B200_SXM.ridge_point(PrecisionFormat.FP16), 2250.0 / 8.00, rel_tol=1e-6)
    assert math.isclose(B200_SXM.ridge_point(PrecisionFormat.FP8), 4500.0 / 8.00, rel_tol=1e-6)
    assert math.isclose(B200_SXM.ridge_point(PrecisionFormat.NVFP4), 9000.0 / 8.00, rel_tol=1e-6)

    with pytest.raises(UnsupportedPrecisionError, match="lacks native NVFP4"):
        H100_SXM5.ridge_point(PrecisionFormat.NVFP4)

    with pytest.raises(UnsupportedPrecisionError, match="lacks native NVFP4"):
        H200_SXM.ridge_point(PrecisionFormat.NVFP4)


def test_negative_control_single_stream_decode_is_strictly_memory_bound_across_gpus() -> None:
    """Prove single-stream decode (B=1) sits orders of magnitude below ridge I* on H100, H200, B200."""
    for accel, fmt, layout in (
        (H100_SXM5, PrecisionFormat.FP8, LAYOUT_FP8),
        (H200_SXM, PrecisionFormat.FP8, LAYOUT_FP8),
        (B200_SXM, PrecisionFormat.FP8, LAYOUT_FP8),
    ):
        analyzer = RooflineAnalyzer(LLAMA_70B_GQA, accel, layout, layout, fmt)
        prefill = analyzer.analyze_prefill_step(prompt_tokens=4096)
        decode_b1 = analyzer.analyze_decode_step(batch_size=1, mean_context_len=4096)

        assert not prefill.is_memory_bound
        assert decode_b1.is_memory_bound
        assert decode_b1.arithmetic_intensity < 3.0
        assert decode_b1.memory_time_ms > 20.0 * decode_b1.compute_time_ms
        # TPOT is pinned by memory_time_ms, never compute_time_ms
        assert math.isclose(decode_b1.step_time_ms, decode_b1.memory_time_ms, rel_tol=1e-9)


def test_kv_cache_exact_byte_accounting_mha_gqa_mla() -> None:
    """Verify exact per-token KV cache footprints across MHA, GQA, MLA and FP16/FP8/INT4 layouts."""
    # MHA (L=80, n_kv=64, d_h=128) -> 327,680 B/tok/layer? No: 2*64*128*2 = 32,768 B/layer -> 2,621,440 B/tok
    assert LLAMA_70B_HYPOTHETICAL_MHA.kv_bytes_per_token(LAYOUT_FP16) == 2_621_440
    # GQA 8:1 reduction (L=80, n_kv=8, d_h=128) -> 327,680 B/tok FP16, 163,840 B/tok FP8
    assert LLAMA_70B_GQA.kv_bytes_per_token(LAYOUT_FP16) == 327_680
    assert LLAMA_70B_GQA.kv_bytes_per_token(LAYOUT_FP8) == 163_840
    # INT4 g=128 asymmetric (4B scale+zp per 128 elems -> 0.53125 B/elem)
    assert LLAMA_70B_GQA.kv_bytes_per_token(LAYOUT_INT4_G128_ASYM) == 87_040
    # INT4 g=128 symmetric (2B scale per 128 elems -> 0.515625 B/elem)
    assert LLAMA_70B_GQA.kv_bytes_per_token(LAYOUT_INT4_G128_SYM) == 84_480
    # DeepSeek-V3 MLA absorbed (L=61, d_c=512, d_R=64) -> 61 * 576 * 2 = 70,272 B/tok BF16
    assert DEEPSEEK_V3_MLA_MOE.kv_bytes_per_token(LAYOUT_BF16, absorb_mla=True) == 70_272
    assert DEEPSEEK_V3_MLA_MOE.kv_bytes_per_token(LAYOUT_FP8, absorb_mla=True) == 35_136
    # Blackwell NVFP4 (group=16, 1B scale -> 4.5 bits/elem): 576 elems -> 288 + 36 = 324 B/layer -> 19,764 B/tok
    assert DEEPSEEK_V3_MLA_MOE.kv_bytes_per_token(LAYOUT_NVFP4, absorb_mla=True) == 19_764

    # Long-context asymptotic arithmetic intensity cap prevents reaching ridge I* at S=4096
    an_fp8 = RooflineAnalyzer(LLAMA_70B_GQA, H100_SXM5, LAYOUT_FP8, LAYOUT_FP8, PrecisionFormat.FP8)
    assert an_fp8.critical_batch_size(mean_context_len=0) == 296
    assert an_fp8.critical_batch_size(mean_context_len=4096) is None


def test_negative_control_mla_without_up_projection_absorption_causes_burst_oom() -> None:
    """Prove sizing HBM without MLA absorption / transient decompression buffer triggers OOM at burst."""
    # Deploy DeepSeek-V3 across 8x H200 (TP=8, EP=8, FP8 weights, BF16 MLA KV cache)
    # Absorptive persistent budget allows 524,288 resident KV tokens with 26+ GiB headroom
    budget_absorbed = DEEPSEEK_V3_MLA_MOE.to_memory_budget(
        H200_SXM,
        LAYOUT_FP8,
        LAYOUT_BF16,
        compute_format=PrecisionFormat.FP8,
        tp_size=8,
        ep_size=8,
    )
    budget_absorbed.verify_hbm_admission(resident_kv_tokens=524_288, step_tokens=64)

    # Without decode up-projection absorption (materializing K_c, K_r, V_c across 6 pipelined layers
    # for the burst context), transient decompression workspace spikes by 30.00 GiB and breaches HBM.
    transient_ws = DEEPSEEK_V3_MLA_MOE.mla_decompression_workspace_bytes(
        workspace_tokens=524_288,
        act_layout=LAYOUT_BF16,
        concurrent_layers=6,
        tp_size=8,
    )
    assert transient_ws == 32_212_254_720  # 30.00 GiB exact

    budget_unabsorbed = DEEPSEEK_V3_MLA_MOE.to_memory_budget(
        H200_SXM,
        LAYOUT_FP8,
        LAYOUT_BF16,
        compute_format=PrecisionFormat.FP8,
        tp_size=8,
        ep_size=8,
        mla_workspace_bytes=transient_ws,
    )
    with pytest.raises(MemoryBudgetExceeded, match="HBM admission rejected"):
        budget_unabsorbed.verify_hbm_admission(resident_kv_tokens=524_288, step_tokens=64)


def test_negative_control_moe_active_parameters_misconception_causes_oom() -> None:
    """Prove MoE HBM capacity is governed by total parameters, not sparse active_parameters."""
    # On 4x H200 (TP=4, EP=4, FP8 weights), naive capacity planning using 37B active params predicts
    # < 9 GiB/GPU weights and admits large batches; real residency demands 156.23 GiB/GPU > 140.40 GiB!
    naive_budget = DEEPSEEK_V3_MLA_MOE.to_memory_budget(
        H200_SXM,
        LAYOUT_FP8,
        LAYOUT_BF16,
        compute_format=PrecisionFormat.FP8,
        tp_size=4,
        ep_size=4,
        active_only_misconception=True,
    )
    naive_budget.verify_hbm_admission(resident_kv_tokens=65_536, step_tokens=32)

    true_budget = DEEPSEEK_V3_MLA_MOE.to_memory_budget(
        H200_SXM,
        LAYOUT_FP8,
        LAYOUT_BF16,
        compute_format=PrecisionFormat.FP8,
        tp_size=4,
        ep_size=4,
        active_only_misconception=False,
    )
    with pytest.raises(MemoryBudgetExceeded, match="HBM admission rejected"):
        true_budget.verify_hbm_admission(resident_kv_tokens=65_536, step_tokens=32)


def test_moe_all_to_all_interconnect_and_hot_expert_skew() -> None:
    """Verify NVLink vs InfiniBand All-to-All scaling and alpha_skew bottleneck amplification."""
    analyzer = RooflineAnalyzer(
        DEEPSEEK_V3_MLA_MOE, H200_SXM, LAYOUT_FP8, LAYOUT_BF16, PrecisionFormat.FP8
    )
    intra_nvlink = analyzer.analyze_moe_all_to_all(
        ep_size=8, local_batch_tokens=64, skew_factor=1.0
    )
    inter_ib_unif = analyzer.analyze_moe_all_to_all(
        ep_size=32, local_batch_tokens=64, skew_factor=1.0
    )
    inter_ib_skew = analyzer.analyze_moe_all_to_all(
        ep_size=32, local_batch_tokens=64, skew_factor=4.0
    )

    assert inter_ib_unif.uniform_comm_ms_all_layers > 4.0 * intra_nvlink.uniform_comm_ms_all_layers
    assert inter_ib_skew.skewed_bottleneck_bytes_per_layer == (
        4 * inter_ib_unif.uniform_bytes_per_rank_per_layer
    )
    assert inter_ib_skew.communication_slowdown > 3.5
    assert inter_ib_skew.straggler_ffn_flops_amplification == 4.0
