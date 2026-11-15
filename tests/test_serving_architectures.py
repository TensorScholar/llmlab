"""Verification and negative-control tests for llmlab.serving (Chapter 4)."""
from __future__ import annotations

import pytest

from llmlab.loadgen.capacity import WorkloadProfile
from llmlab.memory.hardware import H100_SXM5, H200_SXM, PrecisionFormat
from llmlab.memory.kv_cache import (
    DEEPSEEK_V3_MLA_MOE,
    LAYOUT_BF16,
    LAYOUT_FP8,
    LAYOUT_INT4_G128_SYM,
    LLAMA_70B_GQA,
)
from llmlab.memory.roofline import RooflineAnalyzer
from llmlab.serving import (
    FABRIC_NVLINK4_H100,
    FABRIC_RDMA_100GBE,
    FABRIC_RDMA_400GBE,
    FABRIC_RDMA_800GBE,
    KVBlockPoolExhaustedError,
    KVMigrationBandwidthError,
    PagedKVBlockManager,
    PinnedBlockEvictionError,
    RadixPrefixTree,
    ScheduledSequence,
    UnifiedTokenScheduler,
    analyze_block_granularity_tradeoffs,
    compare_contiguous_vs_paged,
    compare_preemption_policies,
    dynamic_slack_chunk_size,
    evaluate_chunk_sizes,
    evaluate_kv_migration,
    plan_disaggregated_topology,
    quadratic_kv_reread_bytes,
    simulate_tool_result_cache_sensitivity,
)
from llmlab.tracing.budgets import LatencyBudget


def test_paged_attention_eliminates_reservation_and_supports_cow_fork() -> None:
    """Negative control 1: contiguous max_seq_len reservation exhausts HBM while PagedAttention admits all."""
    budget = LLAMA_70B_GQA.to_memory_budget(
        H100_SXM5,
        weight_layout=LAYOUT_FP8,
        kv_layout=LAYOUT_FP8,
        compute_format=PrecisionFormat.FP8,
        tp_size=2,
    )
    # 120 heterogeneous requests with max_seq_len=32768 would need ~3.93M contiguous slots -> OOM
    lengths = [256 + (idx % 12) * 192 for idx in range(120)]
    audits = compare_contiguous_vs_paged(
        request_lengths=lengths,
        memory_budget=budget,
        max_seq_len=32_768,
        block_sizes=(16, 32, 64),
    )

    contig = audits["contiguous_max_seq_len"]
    paged16 = audits["paged_B16"]
    assert contig.exhausted_hbm is True
    assert contig.rejected_sequences > 0
    assert contig.memory_waste_pct > 85.0

    assert paged16.exhausted_hbm is False
    assert paged16.rejected_sequences == 0
    assert paged16.external_waste_tokens == 0
    assert paged16.reservation_waste_tokens == 0
    assert paged16.memory_waste_pct < 2.0

    # Copy-on-Write branching for parallel beams must share prompt prefix blocks
    mgr = PagedKVBlockManager(memory_budget=budget, block_size=16)
    prompt_blocks = mgr.allocate_sequence("parent", 500)  # 31 full blocks + tail block (used_slots=4)
    before_used = mgr.num_used_blocks
    mgr.fork_sequence("parent", "child_a")
    assert mgr.num_used_blocks == before_used  # Zero physical duplication at fork time
    assert mgr.blocks[prompt_blocks[0]].ref_count == 2

    # Appending 1 token on child_a must copy ONLY the partial tail block, keeping shared ancestors intact
    new_tail_bid = mgr.append_token("child_a")
    assert mgr.cow_copies_triggered == 1
    assert new_tail_bid != prompt_blocks[-1]
    assert mgr.blocks[prompt_blocks[0]].ref_count == 2
    assert mgr.num_used_blocks == before_used + 1

    mgr.free_sequence("parent")
    assert mgr.blocks[prompt_blocks[0]].ref_count == 1
    mgr.free_sequence("child_a")
    assert mgr.num_used_blocks == 0


def test_paged_pool_exhaustion_raises_explicit_error() -> None:
    """Negative control 1b: exhausting physical KV blocks raises KVBlockPoolExhaustedError."""
    budget = LLAMA_70B_GQA.to_memory_budget(
        H100_SXM5,
        weight_layout=LAYOUT_FP8,
        kv_layout=LAYOUT_FP8,
        compute_format=PrecisionFormat.FP8,
        tp_size=1,  # Leaves tight KV pool on single 80GB H100 (~70B FP8 ~70GB weights)
    )
    mgr = PagedKVBlockManager(memory_budget=budget, block_size=16, step_tokens=512)
    max_tokens = budget.max_resident_kv_tokens(step_tokens=512)
    with pytest.raises(KVBlockPoolExhaustedError):
        mgr.allocate_sequence("overflow_seq", max_tokens + 2048)

    tradeoffs = analyze_block_granularity_tradeoffs(memory_budget=budget, mean_seq_len=1024)
    assert len(tradeoffs) == 3
    assert tradeoffs[0].expected_internal_waste_tokens_per_seq < tradeoffs[2].expected_internal_waste_tokens_per_seq


def test_chunked_prefill_tpot_and_quadratic_reread_tradeoff() -> None:
    """Negative control 2: tiny chunks amplify KV reads quadratically; oversized chunks breach TPOT."""
    analyzer = RooflineAnalyzer(
        LLAMA_70B_GQA, H100_SXM5, LAYOUT_FP8, LAYOUT_FP8, PrecisionFormat.FP8
    )
    kv_b = LLAMA_70B_GQA.kv_bytes_per_token(LAYOUT_FP8, tp_size=2)
    exact_128, approx_128, k_128 = quadratic_kv_reread_bytes(
        prompt_tokens=32_768, chunk_size=128, kv_bytes_per_token=kv_b
    )
    exact_512, _, k_512 = quadratic_kv_reread_bytes(
        prompt_tokens=32_768, chunk_size=512, kv_bytes_per_token=kv_b
    )
    exact_16k_512, _, _ = quadratic_kv_reread_bytes(
        prompt_tokens=16_384, chunk_size=512, kv_bytes_per_token=kv_b
    )
    assert k_128 == 256 and k_512 == 64
    assert exact_128 > 4.0 * exact_512  # O(S_in^2 / C_chunk): 4x smaller C_chunk yields >4x KV reads
    assert exact_512 > 3.9 * exact_16k_512  # 2x longer S_in quadruples quadratic KV re-reads
    assert approx_128 > exact_128

    pareto = evaluate_chunk_sizes(
        analyzer=analyzer,
        prompt_tokens=32_768,
        active_decode_seqs=32,
        mean_decode_ctx=4096,
        tpot_slo_ms=35.0,
        candidate_chunks=(64, 128, 256, 512, 1024, 2048, 4096, 8192),
        tp_size=2,
    )
    by_chunk = {c.chunk_size: c for c in pareto.all_candidates}
    assert by_chunk[64].tpot_slo_compliant is True
    assert by_chunk[8192].tpot_slo_compliant is False  # Oversized chunk violates 35 ms TPOT SLO
    assert pareto.optimal_eval.tpot_slo_compliant is True
    assert pareto.optimal_chunk_size <= 2048

    # Dynamic slack adapts upward when waiting queue grows but stays TPOT-safe
    slack_chunk = dynamic_slack_chunk_size(
        analyzer=analyzer,
        active_decode_seqs=8,
        mean_decode_ctx=2048,
        tpot_slo_ms=35.0,
        waiting_prefill_tokens=4096,
        base_chunk_size=256,
        max_token_budget=2048,
        tp_size=2,
    )
    assert slack_chunk >= 256

    # vLLM V1 unified scheduler must allocate M_dec*(1+gamma) decode tokens BEFORE prefill slack
    scheduler = UnifiedTokenScheduler(
        analyzer=analyzer,
        token_budget=512,
        max_prefill_chunk=512,
        speculation_gamma=3,  # Requires 4 verify tokens per decoding sequence
        tp_size=2,
    )
    dec_seqs = [ScheduledSequence(f"dec_{i}", 1024, 256, prefilled_tokens=1024) for i in range(16)]
    pre_seqs = [ScheduledSequence("pre_0", 4096, 128, prefilled_tokens=0)]
    plan = scheduler.schedule_iteration(decoding=dec_seqs, prefilling=pre_seqs)
    assert plan.decode_tokens_scheduled == 64  # 16 * (1 + 3)
    assert plan.prefill_tokens_scheduled == 512 - 64  # Exact remaining token budget slack


def test_preemption_recompute_vs_swap_crossover() -> None:
    """Short contexts favor recomputation over PCIe Gen5 DMA; deep contexts cross over."""
    analyzer = RooflineAnalyzer(
        LLAMA_70B_GQA, H100_SXM5, LAYOUT_FP8, LAYOUT_FP8, PrecisionFormat.FP8
    )
    short = compare_preemption_policies(analyzer=analyzer, context_tokens=2048, tp_size=2)
    deep = compare_preemption_policies(analyzer=analyzer, context_tokens=65_536, tp_size=2)

    assert short.nvlink_swap_roundtrip_ms < short.pcie_swap_roundtrip_ms
    # Compare recompute quadratic compute vs linear DMA growth
    assert deep.recompute_time_ms / short.recompute_time_ms > (
        deep.pcie_swap_roundtrip_ms / short.pcie_swap_roundtrip_ms
    )


def test_radix_tree_pinning_and_parallel_tool_jitter() -> None:
    """Negative control 3: evicting pinned/internal RadixNodes raises PinnedBlockEvictionError."""
    tree = RadixPrefixTree(max_cached_tokens=100, kv_bytes_per_token=32_768, block_size=16)

    sys_prefix = list(range(1000, 1040))
    req1_tokens = sys_prefix + list(range(2000, 2020))
    req2_tokens = sys_prefix + list(range(3000, 3020))

    h1 = tree.insert_and_pin("req_1", req1_tokens)
    h2 = tree.insert_and_pin("req_2", req2_tokens)
    assert h1.matched_tokens == 0
    assert h2.matched_tokens == 40  # Entire sys_prefix shared!

    shared_root_child_id = h2.path_node_ids[0]
    assert tree.ancestor_path("req_1")[0] == shared_root_child_id
    leaf_1_id = h1.path_node_ids[-1]
    assert tree._nodes[shared_root_child_id].ref_count == 2
    assert tree._nodes[leaf_1_id].ref_count == 1

    # Negative control: cannot evict pinned leaf OR parent node with active children
    with pytest.raises(PinnedBlockEvictionError):
        tree.force_evict_node(leaf_1_id)
    with pytest.raises(PinnedBlockEvictionError):
        tree.force_evict_node(shared_root_child_id)

    # Release req_1 -> its suffix leaf becomes unpinned & evictable, while shared prefix remains pinned by req_2
    tree.release_handle("req_1")
    assert tree._nodes[shared_root_child_id].ref_count == 1
    with pytest.raises(PinnedBlockEvictionError):
        tree.force_evict_node(shared_root_child_id)
    reclaimed = tree.force_evict_node(leaf_1_id)
    assert reclaimed == 20

    # All capacity pinned -> exhausting max_cached_tokens raises PinnedBlockEvictionError
    with pytest.raises(PinnedBlockEvictionError):
        tree.insert_and_pin("req_3_overflow", list(range(5000, 5090)))

    tree.release_handle("req_2")

    # Parallel tool arrival-order jitter vs canonical emission-index sort
    audit = simulate_tool_result_cache_sensitivity(
        num_turns=8,
        tools_per_turn=4,
        tokens_per_tool_result=512,
        system_prompt_tokens=2048,
        assistant_tokens_per_turn=128,
    )
    assert audit.canonical_hit_ratio > 0.90
    assert audit.jittered_hit_ratio < 0.70
    assert audit.prefill_compute_reduction_pct > 75.0


def test_disaggregation_fabric_slo_and_topology_sizing() -> None:
    """Negative control 4: 100GbE breaches TTFT_SLO while pipelined 800GbE/NVLink stay within SLO."""
    slo = LatencyBudget(
        max_ttft_ms=1500.0,
        max_tpot_ms=25.0,
        max_p99_itl_ms=50.0,
        workflow_deadline_ms=15_000.0,
    )

    # 8k uncached long prompt KV transfer on Llama-3 70B FP8 across single NIC vs NVLink4
    slow_est = evaluate_kv_migration(
        prompt_tokens=16_384,
        uncached_tokens=16_384,
        arch=LLAMA_70B_GQA,
        prefill_accel=H100_SXM5,
        decode_accel=H200_SXM,
        weight_layout=LAYOUT_FP8,
        kv_layout=LAYOUT_FP8,
        compute_format=PrecisionFormat.FP8,
        fabric=FABRIC_RDMA_100GBE,
        latency_budget=slo,
        prefill_tp=2,
        decode_tp=2,
        nics_per_instance=1,
        pipelined_by_layer=False,
    )
    assert slow_est.within_slo is False

    with pytest.raises(KVMigrationBandwidthError):
        evaluate_kv_migration(
            prompt_tokens=16_384,
            uncached_tokens=16_384,
            arch=LLAMA_70B_GQA,
            prefill_accel=H100_SXM5,
            decode_accel=H200_SXM,
            weight_layout=LAYOUT_FP8,
            kv_layout=LAYOUT_FP8,
            compute_format=PrecisionFormat.FP8,
            fabric=FABRIC_RDMA_100GBE,
            latency_budget=slo,
            prefill_tp=2,
            decode_tp=2,
            nics_per_instance=1,
            pipelined_by_layer=False,
            enforce_slo=True,
        )

    fast_pipelined = evaluate_kv_migration(
        prompt_tokens=16_384,
        uncached_tokens=16_384,
        arch=LLAMA_70B_GQA,
        prefill_accel=H100_SXM5,
        decode_accel=H200_SXM,
        weight_layout=LAYOUT_FP8,
        kv_layout=LAYOUT_FP8,
        compute_format=PrecisionFormat.FP8,
        fabric=FABRIC_NVLINK4_H100,
        latency_budget=slo,
        prefill_tp=2,
        decode_tp=2,
        nics_per_instance=2,
        pipelined_by_layer=True,
        enforce_slo=True,
    )
    assert fast_pipelined.within_slo is True
    assert fast_pipelined.exposed_transfer_ms < fast_pipelined.serial_transfer_ms

    # Compare DeepSeek-V3 MLA compression vs GQA under P/D sizing
    gqa_work = WorkloadProfile(
        target_rps=25.0,
        short_prompt_tokens=1024,
        long_prompt_tokens=8192,
        p_long_prompt=0.25,
        mean_output_tokens=256,
        prefix_hit_ratio=0.40,
        shared_system_prefix_tokens=512,
    )
    plan = plan_disaggregated_topology(
        workload=gqa_work,
        arch=LLAMA_70B_GQA,
        prefill_accel=H100_SXM5,
        decode_accel=H200_SXM,
        weight_layout=LAYOUT_FP8,
        kv_layout=LAYOUT_FP8,
        compute_format=PrecisionFormat.FP8,
        latency_budget=slo,
        fabric=FABRIC_RDMA_800GBE,
        prefill_tp=2,
        decode_tp=2,
        nics_per_instance=2,
        pipelined_by_layer=True,
        enforce_slo=True,
    )
    assert plan.prefill_instances >= 1
    assert plan.decode_instances >= 1
    assert plan.total_disagg_gpus == (plan.prefill_gpus + plan.decode_gpus)

    # Verify MLA dramatically slashes KV transfer bytes on DeepSeek-V3 across 400GbE (FP8 vs FP8)
    ds_est = evaluate_kv_migration(
        prompt_tokens=16_384,
        uncached_tokens=16_384,
        arch=DEEPSEEK_V3_MLA_MOE,
        prefill_accel=H100_SXM5,
        decode_accel=H200_SXM,
        weight_layout=LAYOUT_INT4_G128_SYM,
        kv_layout=LAYOUT_FP8,
        compute_format=PrecisionFormat.FP8,
        fabric=FABRIC_RDMA_400GBE,
        latency_budget=slo,
        prefill_tp=4,
        decode_tp=4,
        ep_size=8,
        nics_per_instance=1,
        pipelined_by_layer=True,
        enforce_slo=True,
    )
    assert ds_est.kv_transfer_bytes < slow_est.kv_transfer_bytes * 0.25
