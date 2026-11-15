"""vLLM V1 unified token-budget scheduler, chunked prefill optimizer, and preemption tradeoffs."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

from llmlab.memory.kv_cache import AttentionVariant
from llmlab.memory.roofline import RooflineAnalyzer


@dataclass
class ScheduledSequence:
    """Active request tracked inside an iteration-level unified scheduler queue."""

    request_id: str
    prompt_tokens: int
    target_output_tokens: int
    prefilled_tokens: int = 0
    decoded_tokens: int = 0
    is_preempted: bool = False

    @property
    def is_prefill_complete(self) -> bool:
        """True iff all prompt tokens have populated KV blocks."""
        return self.prefilled_tokens >= self.prompt_tokens

    @property
    def rem_prefill_tokens(self) -> int:
        """Remaining prompt tokens awaiting chunked prefill execution."""
        return max(0, self.prompt_tokens - self.prefilled_tokens)

    @property
    def resident_kv_tokens(self) -> int:
        """Current KV tokens stored in HBM for this sequence."""
        return self.prefilled_tokens + self.decoded_tokens


@dataclass(frozen=True, slots=True)
class StepSchedulePlan:
    """Single-iteration unified token allocation returned by vLLM V1 scheduler."""

    step_index: int
    token_budget: int
    decode_tokens_scheduled: int
    prefill_tokens_scheduled: int
    active_decode_seqs: int
    prefill_allocations: tuple[tuple[str, int], ...]
    total_step_tokens: int
    estimated_step_ms: float


@dataclass(frozen=True, slots=True)
class ChunkedPrefillEvaluation:
    """Exact Roofline step duration and quadratic KV-read cost for candidate chunk size C_chunk."""

    chunk_size: int
    num_chunks: int
    piggybacked_decode_seqs: int
    total_step_tokens: int
    first_chunk_step_ms: float
    worst_chunk_step_ms: float
    total_prefill_wall_ms: float
    kv_reread_bytes_exact: int
    kv_reread_bytes_approx: float
    kv_reread_amplification_ratio: float
    tpot_slo_compliant: bool


@dataclass(frozen=True, slots=True)
class OptimalChunkResult:
    """Closed-form & Roofline-verified Pareto operating point maximizing prefill without TPOT breach."""

    optimal_chunk_size: int
    tpot_slo_ms: float
    all_candidates: tuple[ChunkedPrefillEvaluation, ...]
    optimal_eval: ChunkedPrefillEvaluation


@dataclass(frozen=True, slots=True)
class PreemptionComparison:
    """HBM exhaustion recovery comparison: KV recomputation vs. PCIe/NVLink block swapping."""

    context_tokens: int
    kv_bytes: int
    recompute_time_ms: float
    pcie_swap_roundtrip_ms: float
    nvlink_swap_roundtrip_ms: float
    cheaper_against_pcie: Literal["recompute", "swap_pcie"]
    cheaper_against_nvlink: Literal["recompute", "swap_nvlink"]


def quadratic_kv_reread_bytes(
    *,
    prompt_tokens: int,
    chunk_size: int,
    kv_bytes_per_token: int,
) -> tuple[int, float, int]:
    """Compute exact vs. asymptotic quadratic KV-read bytes across K = ceil(S_in / C_chunk) chunks.

    Exact formula: sum_{j=1}^K min((j-1)*C_chunk, S_in) * b_KV (or exact full-prefix scan
    ``(K*(K-1)/2) * C_chunk * b_KV`` when chunks divide evenly), vs. smooth approximation
    ``(S_in^2 / (2 * C_chunk)) * b_KV``. Returns ``(exact_bytes, approx_bytes, K)``.
    """
    if prompt_tokens < 1 or chunk_size < 1 or kv_bytes_per_token < 1:
        raise ValueError("prompt_tokens, chunk_size, and kv_bytes_per_token must be positive.")

    num_chunks = math.ceil(prompt_tokens / chunk_size)
    # Every chunk j = 0..K-1 attends over prefix_len = j * chunk_size historical KV tokens
    exact_prefix_tokens = sum(j * chunk_size for j in range(num_chunks))
    exact_bytes = exact_prefix_tokens * kv_bytes_per_token
    approx_bytes = ((prompt_tokens * prompt_tokens) / (2.0 * chunk_size)) * kv_bytes_per_token
    return exact_bytes, approx_bytes, num_chunks


def estimate_unified_step_ms(
    *,
    analyzer: RooflineAnalyzer,
    prefill_chunk_tokens: int,
    prefill_prefix_tokens: int,
    decode_seqs: int,
    mean_decode_ctx: int,
    tokens_per_decode: int = 1,
    tp_size: int = 1,
    ep_size: int = 1,
    mfu: float = 0.50,
    mbu: float = 0.80,
    absorb_mla: bool = True,
) -> float:
    """Compute unified single-weight-scan iteration latency packing decodes + prefill chunk."""
    dec_tokens = decode_seqs * tokens_per_decode
    total_tokens = dec_tokens + prefill_chunk_tokens
    if total_tokens <= 0:
        return 0.0

    # Linear projections share one active parameter sweep per token across decodes + prefill chunk
    linear_flops = float(2.0 * (analyzer.arch.active_parameters / tp_size) * total_tokens)

    # Attention FLOPs: decode KV scan + prefill intra-chunk causal + cross-chunk prefix scan
    heads_per_gpu = analyzer.arch.num_q_heads / tp_size
    if analyzer.arch.attention_variant == AttentionVariant.MLA:
        dec_flop_per_tok = 2 * (2 * analyzer.arch.mla_kv_lora_rank + analyzer.arch.mla_rope_head_dim)
        pre_flop_per_pair = (
            analyzer.arch.head_dim + analyzer.arch.mla_rope_head_dim + analyzer.arch.mla_v_head_dim
        )
    else:
        dec_flop_per_tok = 4 * analyzer.arch.head_dim
        pre_flop_per_pair = 2 * analyzer.arch.head_dim

    dec_attn_flops = float(
        analyzer.arch.num_layers * dec_tokens * heads_per_gpu * dec_flop_per_tok * max(0, mean_decode_ctx)
    )
    # Prefill chunk covers intra-chunk causal (c^2/2 -> c^2 full-coeff*2) + cross-prefix (c * prefix)
    pre_pairs = (prefill_chunk_tokens * prefill_chunk_tokens) + (
        2 * prefill_chunk_tokens * prefill_prefix_tokens
    )
    pre_attn_flops = float(analyzer.arch.num_layers * heads_per_gpu * pre_flop_per_pair * pre_pairs)

    # HBM traffic: weights loaded ONCE per step; KV reads for decodes + chunk prefix re-reads + new KV writes
    w_bytes = analyzer._active_weight_bytes_read_per_gpu(
        total_tokens, tp_size=tp_size, ep_size=ep_size
    )
    kv_per_tok_dec = analyzer.arch.kv_bytes_per_token(
        analyzer.kv_layout, tp_size=tp_size, absorb_mla=absorb_mla
    )
    kv_per_tok_pre = analyzer.arch.kv_bytes_per_token(
        analyzer.kv_layout, tp_size=tp_size, absorb_mla=True
    )
    dec_kv_bytes = dec_tokens * (max(0, mean_decode_ctx) + 1) * kv_per_tok_dec
    pre_kv_bytes = (prefill_prefix_tokens + prefill_chunk_tokens) * kv_per_tok_pre
    total_hbm_bytes = w_bytes + dec_kv_bytes + pre_kv_bytes

    p_eff = analyzer.accel.peak_compute_tflops(analyzer.compute_format) * 1e12 * mfu
    b_eff = analyzer.accel.peak_hbm_bandwidth_gbps * 1e9 * mbu
    t_comp_ms = ((linear_flops + dec_attn_flops + pre_attn_flops) / p_eff) * 1000.0
    t_mem_ms = (total_hbm_bytes / b_eff) * 1000.0
    return max(t_comp_ms, t_mem_ms)


def evaluate_chunk_sizes(
    *,
    analyzer: RooflineAnalyzer,
    prompt_tokens: int,
    active_decode_seqs: int,
    mean_decode_ctx: int,
    tpot_slo_ms: float,
    candidate_chunks: Sequence[int] = (32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 8192),
    tp_size: int = 1,
    ep_size: int = 1,
    mfu: float = 0.50,
    mbu: float = 0.80,
) -> OptimalChunkResult:
    """Solve for optimal prefill chunk C_chunk* maximizing prefill speed while honoring TPOT_SLO."""
    if prompt_tokens < 1 or tpot_slo_ms <= 0.0:
        raise ValueError("prompt_tokens >= 1 and tpot_slo_ms > 0 required.")

    kv_per_tok = analyzer.arch.kv_bytes_per_token(
        analyzer.kv_layout, tp_size=tp_size, absorb_mla=True
    )
    base_kv_bytes = prompt_tokens * kv_per_tok
    evaluations: list[ChunkedPrefillEvaluation] = []

    for c_sz in candidate_chunks:
        if c_sz < 1:
            continue
        eff_c = min(c_sz, prompt_tokens)
        exact_bytes, approx_bytes, k_chunks = quadratic_kv_reread_bytes(
            prompt_tokens=prompt_tokens, chunk_size=eff_c, kv_bytes_per_token=kv_per_tok
        )
        step_times: list[float] = []
        done = 0
        for _ in range(k_chunks):
            take = min(eff_c, prompt_tokens - done)
            st = estimate_unified_step_ms(
                analyzer=analyzer,
                prefill_chunk_tokens=take,
                prefill_prefix_tokens=done,
                decode_seqs=active_decode_seqs,
                mean_decode_ctx=mean_decode_ctx,
                tp_size=tp_size,
                ep_size=ep_size,
                mfu=mfu,
                mbu=mbu,
            )
            step_times.append(st)
            done += take

        worst_ms = max(step_times)
        eval_item = ChunkedPrefillEvaluation(
            chunk_size=eff_c,
            num_chunks=k_chunks,
            piggybacked_decode_seqs=active_decode_seqs,
            total_step_tokens=active_decode_seqs + eff_c,
            first_chunk_step_ms=step_times[0],
            worst_chunk_step_ms=worst_ms,
            total_prefill_wall_ms=math.fsum(step_times),
            kv_reread_bytes_exact=exact_bytes,
            kv_reread_bytes_approx=approx_bytes,
            kv_reread_amplification_ratio=(exact_bytes / base_kv_bytes) if base_kv_bytes > 0 else 0.0,
            tpot_slo_compliant=worst_ms <= tpot_slo_ms,
        )
        evaluations.append(eval_item)

    # Deduplicate effective chunk size == prompt_tokens if multiple candidates exceed prompt_tokens
    seen_sizes: set[int] = set()
    unique_evals: list[ChunkedPrefillEvaluation] = []
    for ev in sorted(evaluations, key=lambda x: x.chunk_size):
        if ev.chunk_size not in seen_sizes:
            seen_sizes.add(ev.chunk_size)
            unique_evals.append(ev)

    compliant = [e for e in unique_evals if e.tpot_slo_compliant]
    if compliant:
        # Optimal C_chunk* is the compliant chunk minimizing total prefill wall time
        optimal = min(compliant, key=lambda e: (e.total_prefill_wall_ms, -e.chunk_size))
    else:
        optimal = min(unique_evals, key=lambda e: e.worst_chunk_step_ms)

    return OptimalChunkResult(
        optimal_chunk_size=optimal.chunk_size,
        tpot_slo_ms=tpot_slo_ms,
        all_candidates=tuple(unique_evals),
        optimal_eval=optimal,
    )


def dynamic_slack_chunk_size(
    *,
    analyzer: RooflineAnalyzer,
    active_decode_seqs: int,
    mean_decode_ctx: int,
    tpot_slo_ms: float,
    waiting_prefill_tokens: int,
    base_chunk_size: int,
    max_token_budget: int = 2048,
    tp_size: int = 1,
    mfu: float = 0.50,
    mbu: float = 0.80,
) -> int:
    """Adapt C_chunk upward into unused Tensor Core slack during queue spikes without breaching TPOT."""
    if waiting_prefill_tokens <= 0:
        return 0
    free_budget = max(0, max_token_budget - active_decode_seqs)
    if free_budget == 0:
        return 0

    best_c = min(base_chunk_size, free_budget)
    for cand in (128, 256, 384, 512, 768, 1024, 1536, 2048):
        if cand > free_budget:
            break
        step_ms = estimate_unified_step_ms(
            analyzer=analyzer,
            prefill_chunk_tokens=cand,
            prefill_prefix_tokens=mean_decode_ctx // 2,
            decode_seqs=active_decode_seqs,
            mean_decode_ctx=mean_decode_ctx,
            tp_size=tp_size,
            mfu=mfu,
            mbu=mbu,
        )
        if step_ms <= tpot_slo_ms:
            best_c = cand
    return min(waiting_prefill_tokens, max(min(64, free_budget), best_c))


class UnifiedTokenScheduler:
    """vLLM V1-style iteration-level unified token-budget scheduler prioritizing decodes."""

    def __init__(
        self,
        *,
        analyzer: RooflineAnalyzer,
        token_budget: int = 1024,
        max_prefill_chunk: int = 512,
        speculation_gamma: int = 0,
        tp_size: int = 1,
        ep_size: int = 1,
        mfu: float = 0.50,
        mbu: float = 0.80,
    ) -> None:
        if token_budget < 1 or max_prefill_chunk < 1 or speculation_gamma < 0:
            raise ValueError("token_budget >= 1, max_prefill_chunk >= 1, and gamma >= 0 required.")
        self.analyzer = analyzer
        self.token_budget = token_budget
        self.max_prefill_chunk = max_prefill_chunk
        self.tokens_per_decode = 1 + speculation_gamma
        self.tp_size = tp_size
        self.ep_size = ep_size
        self.mfu = mfu
        self.mbu = mbu
        self.step_counter = 0

    def schedule_iteration(
        self,
        *,
        decoding: Sequence[ScheduledSequence],
        prefilling: Sequence[ScheduledSequence],
    ) -> StepSchedulePlan:
        """Pack M_dec * (1 + gamma) decode tokens first, filling slack with bounded prefill chunks."""
        self.step_counter += 1
        dec_count = len(decoding)
        dec_tokens = dec_count * self.tokens_per_decode
        remaining_budget = max(0, self.token_budget - dec_tokens)

        allocations: list[tuple[str, int]] = []
        pre_scheduled = 0
        for seq in prefilling:
            if remaining_budget <= 0:
                break
            take = min(seq.rem_prefill_tokens, self.max_prefill_chunk, remaining_budget)
            if take > 0:
                allocations.append((seq.request_id, take))
                pre_scheduled += take
                remaining_budget -= take

        mean_dec_ctx = (
            int(round(sum(s.resident_kv_tokens for s in decoding) / dec_count)) if dec_count > 0 else 0
        )
        lead_prefix = prefilling[0].prefilled_tokens if (prefilling and pre_scheduled > 0) else 0
        step_ms = estimate_unified_step_ms(
            analyzer=self.analyzer,
            prefill_chunk_tokens=pre_scheduled,
            prefill_prefix_tokens=lead_prefix,
            decode_seqs=dec_count,
            mean_decode_ctx=mean_dec_ctx,
            tokens_per_decode=self.tokens_per_decode,
            tp_size=self.tp_size,
            ep_size=self.ep_size,
            mfu=self.mfu,
            mbu=self.mbu,
        )

        # Advance state cleanly
        for seq in decoding:
            seq.decoded_tokens += self.tokens_per_decode
        alloc_map = dict(allocations)
        for seq in prefilling:
            if seq.request_id in alloc_map:
                seq.prefilled_tokens += alloc_map[seq.request_id]

        return StepSchedulePlan(
            step_index=self.step_counter,
            token_budget=self.token_budget,
            decode_tokens_scheduled=dec_tokens,
            prefill_tokens_scheduled=pre_scheduled,
            active_decode_seqs=dec_count,
            prefill_allocations=tuple(allocations),
            total_step_tokens=dec_tokens + pre_scheduled,
            estimated_step_ms=step_ms,
        )


def compare_preemption_policies(
    *,
    analyzer: RooflineAnalyzer,
    context_tokens: int,
    tp_size: int = 1,
    pcie_effective_gbps: float = 50.0,
    nvlink_effective_gbps: float | None = None,
    dma_setup_ms: float = 0.25,
    mfu: float = 0.50,
    mbu: float = 0.80,
) -> PreemptionComparison:
    """Compare EVICT+RECOMPUTE vs. PCIe Gen5 / NVLink SWAP round-trip time under HBM OOM."""
    if context_tokens < 1 or pcie_effective_gbps <= 0.0:
        raise ValueError("context_tokens >= 1 and pcie_effective_gbps > 0 required.")

    kv_per_tok = analyzer.arch.kv_bytes_per_token(
        analyzer.kv_layout, tp_size=tp_size, absorb_mla=True
    )
    kv_bytes = context_tokens * kv_per_tok

    recompute_res = analyzer.analyze_prefill_step(
        context_tokens, tp_size=tp_size, mfu=mfu, mbu=mbu
    )
    recompute_ms = recompute_res.step_time_ms

    # Swapping incurs TWO crossings: evict HBM -> CPU/peer, then stage back -> HBM before resume
    pcie_ms = (2.0 * kv_bytes / (pcie_effective_gbps * 1e9)) * 1000.0 + (2.0 * dma_setup_ms)
    nvl_bw = nvlink_effective_gbps if nvlink_effective_gbps is not None else analyzer.accel.nvlink_unidir_gbps * 0.85
    nvlink_ms = (2.0 * kv_bytes / (nvl_bw * 1e9)) * 1000.0 + (2.0 * dma_setup_ms)

    return PreemptionComparison(
        context_tokens=context_tokens,
        kv_bytes=kv_bytes,
        recompute_time_ms=recompute_ms,
        pcie_swap_roundtrip_ms=pcie_ms,
        nvlink_swap_roundtrip_ms=nvlink_ms,
        cheaper_against_pcie="recompute" if recompute_ms <= pcie_ms else "swap_pcie",
        cheaper_against_nvlink="recompute" if recompute_ms <= nvlink_ms else "swap_nvlink",
    )
