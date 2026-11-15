"""Deterministic open-loop (Poisson/burst) and closed-loop continuous-batching simulators."""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

from llmlab.loadgen.metrics import GoodputSummary, RequestTelemetry, summarize_goodput
from llmlab.memory.roofline import RooflineAnalyzer
from llmlab.tracing.budgets import LatencyBudget, MemoryBudget
from llmlab.tracing.spans import StreamingLatencyBreakdown


@dataclass(frozen=True, slots=True)
class RequestSpec:
    """Workload request specification decoupled from serving execution timestamps."""

    request_id: str
    scheduled_arrival_ms: float
    prompt_tokens: int
    cached_prefix_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if self.scheduled_arrival_ms < 0.0:
            raise ValueError("scheduled_arrival_ms must be non-negative.")
        if self.prompt_tokens < 1 or self.output_tokens < 2:
            raise ValueError("prompt_tokens >= 1 and output_tokens >= 2 required.")
        if not (0 <= self.cached_prefix_tokens <= self.prompt_tokens):
            raise ValueError("cached_prefix_tokens must be in [0, prompt_tokens].")

    @property
    def uncached_prompt_tokens(self) -> int:
        """Uncached prompt tokens requiring active prefill computation."""
        return self.prompt_tokens - self.cached_prefix_tokens


def generate_poisson_workload(
    *,
    num_requests: int,
    arrival_rate_rps: float,
    prompt_tokens_base: int = 512,
    long_prompt_tokens: int = 4096,
    p_long_prompt: float = 0.10,
    output_mu_ln: float = 4.6,
    output_sigma_ln: float = 0.85,
    max_output_tokens: int = 1024,
    prefix_hit_ratio: float = 0.0,
    seed: int = 42,
) -> tuple[RequestSpec, ...]:
    """Generate an open-loop Poisson arrival workload with bimodal prompts and heavy-tailed outputs."""
    if num_requests < 1 or arrival_rate_rps <= 0.0:
        raise ValueError("num_requests >= 1 and arrival_rate_rps > 0 required.")
    if not (0.0 <= p_long_prompt <= 1.0) or not (0.0 <= prefix_hit_ratio < 1.0):
        raise ValueError("Probabilities must lie in [0, 1] (prefix_hit_ratio < 1.0).")

    rng = random.Random(seed)
    t_ms = 0.0
    specs: list[RequestSpec] = []
    for idx in range(num_requests):
        t_ms += rng.expovariate(arrival_rate_rps) * 1000.0
        prompt_len = long_prompt_tokens if rng.random() < p_long_prompt else prompt_tokens_base
        cached_len = int(round(prompt_len * prefix_hit_ratio))
        raw_out = int(round(rng.lognormvariate(output_mu_ln, output_sigma_ln)))
        out_len = max(8, min(max_output_tokens, raw_out))
        specs.append(
            RequestSpec(
                request_id=f"req-{idx:05d}",
                scheduled_arrival_ms=t_ms,
                prompt_tokens=prompt_len,
                cached_prefix_tokens=cached_len,
                output_tokens=out_len,
            )
        )
    return tuple(specs)


def generate_burst_replay_workload(
    *,
    duration_s: float,
    baseline_rps: float,
    burst_rps: float,
    burst_start_s: float,
    burst_duration_s: float,
    prompt_tokens_base: int = 512,
    output_tokens_mean: int = 128,
    prefix_hit_ratio: float = 0.0,
    seed: int = 7,
) -> tuple[RequestSpec, ...]:
    """Generate a deterministic time-varying arrival trace with a transient burst wave."""
    if duration_s <= 0.0 or baseline_rps <= 0.0 or burst_rps <= 0.0:
        raise ValueError("Rates and duration must be positive.")

    rng = random.Random(seed)
    t_s = 0.0
    idx = 0
    specs: list[RequestSpec] = []
    burst_end_s = burst_start_s + burst_duration_s
    while t_s < duration_s:
        rate = burst_rps if (burst_start_s <= t_s < burst_end_s) else baseline_rps
        t_s += rng.expovariate(rate)
        if t_s >= duration_s:
            break
        cached_len = int(round(prompt_tokens_base * prefix_hit_ratio))
        out_len = max(8, min(1024, int(round(rng.lognormvariate(math.log(output_tokens_mean) - 0.35, 0.80)))))
        specs.append(
            RequestSpec(
                request_id=f"burst-{idx:05d}",
                scheduled_arrival_ms=t_s * 1000.0,
                prompt_tokens=prompt_tokens_base,
                cached_prefix_tokens=cached_len,
                output_tokens=out_len,
            )
        )
        idx += 1
    return tuple(specs)


@dataclass(frozen=True, slots=True)
class ReplicaSimConfig:
    """Hardware-coupled configuration for a single continuous-batching GPU replica."""

    analyzer: RooflineAnalyzer
    memory_budget: MemoryBudget
    tp_size: int = 1
    ep_size: int = 1
    mfu: float = 0.50
    mbu: float = 0.80
    chunk_size_tokens: int | None = 512
    max_step_tokens: int = 1024
    absorb_mla: bool = True


@dataclass
class _ActiveSeq:
    spec: RequestSpec
    execution_start_ms: float
    rem_prefill_tokens: int
    execution_started: bool = False
    generated_tokens: int = 0
    first_token_ms: float | None = None
    last_token_ms: float | None = None
    pending_hol_stall_ms: float = 0.0
    itl_gaps: list[float] = field(default_factory=list)

    @property
    def current_resident_kv(self) -> int:
        if self.generated_tokens == 0:
            prefilled_done = self.spec.prompt_tokens - self.rem_prefill_tokens
            return max(self.spec.cached_prefix_tokens, prefilled_done)
        return self.spec.prompt_tokens + self.generated_tokens - 1


def _estimate_mixed_step_ms(
    *,
    config: ReplicaSimConfig,
    prefill_slice: int,
    num_decoding: int,
    mean_decode_ctx: int,
) -> float:
    """Compute unified single-weight-scan iteration duration via Chapter 2 Roofline physics."""
    if prefill_slice > 0 and num_decoding == 0:
        return max(
            0.05,
            config.analyzer.analyze_prefill_step(
                prefill_slice,
                tp_size=config.tp_size,
                ep_size=config.ep_size,
                mfu=config.mfu,
                mbu=config.mbu,
            ).step_time_ms,
        )
    if num_decoding > 0 and prefill_slice == 0:
        return max(
            0.05,
            config.analyzer.analyze_decode_step(
                num_decoding,
                max(1, mean_decode_ctx),
                tp_size=config.tp_size,
                ep_size=config.ep_size,
                mfu=config.mfu,
                mbu=config.mbu,
                absorb_mla=config.absorb_mla,
            ).step_time_ms,
        )

    # Fused chunked prefill + decode shares one HBM weight read across step_tokens = dec + chunk
    dec_res = config.analyzer.analyze_decode_step(
        max(1, num_decoding),
        max(1, mean_decode_ctx),
        tp_size=config.tp_size,
        ep_size=config.ep_size,
        mfu=config.mfu,
        mbu=config.mbu,
        absorb_mla=config.absorb_mla,
    )
    pre_res = config.analyzer.analyze_prefill_step(
        prefill_slice,
        tp_size=config.tp_size,
        ep_size=config.ep_size,
        mfu=config.mfu,
        mbu=config.mbu,
    )
    # Weights loaded once per iteration; FLOPs add; decode reads resident KV + prefill writes chunk KV
    total_flops = dec_res.total_flops + pre_res.total_flops
    total_hbm_bytes = (
        max(dec_res.weight_bytes_read, pre_res.weight_bytes_read)
        + dec_res.kv_bytes_transferred
        + pre_res.kv_bytes_transferred
    )
    eff_tflops = (
        config.analyzer.accel.peak_compute_tflops(config.analyzer.compute_format) * 1e12 * config.mfu
    )
    eff_bw_bps = config.analyzer.accel.peak_hbm_bandwidth_gbps * 1e9 * config.mbu
    t_comp_ms = (total_flops / eff_tflops) * 1000.0
    t_mem_ms = (total_hbm_bytes / eff_bw_bps) * 1000.0
    return max(0.05, t_comp_ms, t_mem_ms)


def _step_continuous_batch(
    *,
    clock_ms: float,
    active: list[_ActiveSeq],
    config: ReplicaSimConfig,
) -> tuple[float, list[RequestTelemetry]]:
    """Advance active continuous batch by one scheduler iteration and return completed traces."""
    decoding = [s for s in active if s.rem_prefill_tokens == 0]
    prefilling = [s for s in active if s.rem_prefill_tokens > 0]

    mean_ctx = (
        int(round(sum(s.current_resident_kv for s in decoding) / len(decoding)))
        if decoding
        else 256
    )

    # Unchunked prefill mode locks GPU to complete the head prompt, stalling active decodes
    if config.chunk_size_tokens is None and prefilling:
        head = prefilling[0]
        if not head.execution_started:
            head.execution_start_ms = clock_ms
            head.execution_started = True
        prefill_toks = head.rem_prefill_tokens
        dt_ms = _estimate_mixed_step_ms(
            config=config,
            prefill_slice=prefill_toks,
            num_decoding=0,
            mean_decode_ctx=mean_ctx,
        )
        new_clock = clock_ms + dt_ms
        # Active decoding sequences receive zero tokens during HoL block; charge stall to next ITL
        for seq in decoding:
            seq.pending_hol_stall_ms += dt_ms
        head.rem_prefill_tokens = 0
        head.generated_tokens = 1
        head.first_token_ms = new_clock
        head.last_token_ms = new_clock
        return new_clock, _finalize_completed(active)

    # Chunked prefill / unified token-budget step: run active decodes + bounded prefill chunk
    dec_count = len(decoding)
    chunk_budget = max(0, min(config.chunk_size_tokens or config.max_step_tokens, config.max_step_tokens - dec_count))
    prefill_slice = 0
    prefilled_finished_heads: list[_ActiveSeq] = []

    if prefilling and chunk_budget > 0:
        target = prefilling[0]
        if not target.execution_started:
            target.execution_start_ms = clock_ms
            target.execution_started = True
        take = min(target.rem_prefill_tokens, chunk_budget)
        prefill_slice = take
        target.rem_prefill_tokens -= take
        if target.rem_prefill_tokens == 0:
            prefilled_finished_heads.append(target)

    dt_ms = _estimate_mixed_step_ms(
        config=config,
        prefill_slice=prefill_slice,
        num_decoding=dec_count,
        mean_decode_ctx=mean_ctx,
    )
    new_clock = clock_ms + dt_ms

    for seq in decoding:
        assert seq.last_token_ms is not None
        gap = (new_clock - seq.last_token_ms) + seq.pending_hol_stall_ms
        seq.pending_hol_stall_ms = 0.0
        seq.itl_gaps.append(gap)
        seq.generated_tokens += 1
        seq.last_token_ms = new_clock

    for head in prefilled_finished_heads:
        head.generated_tokens = 1
        head.first_token_ms = new_clock
        head.last_token_ms = new_clock

    return new_clock, _finalize_completed(active)


def _finalize_completed(active: list[_ActiveSeq]) -> list[RequestTelemetry]:
    finished: list[RequestTelemetry] = []
    remaining: list[_ActiveSeq] = []
    for seq in active:
        if seq.generated_tokens >= seq.spec.output_tokens:
            assert seq.first_token_ms is not None and seq.last_token_ms is not None
            ttft_ms = seq.first_token_ms - seq.spec.scheduled_arrival_ms
            breakdown = StreamingLatencyBreakdown(
                ttft_ms=ttft_ms,
                itl_ms=tuple(seq.itl_gaps),
            )
            finished.append(
                RequestTelemetry(
                    request_id=seq.spec.request_id,
                    scheduled_arrival_ms=seq.spec.scheduled_arrival_ms,
                    execution_start_ms=seq.execution_start_ms,
                    completion_ms=seq.last_token_ms,
                    prompt_tokens=seq.spec.prompt_tokens,
                    cached_prefix_tokens=seq.spec.cached_prefix_tokens,
                    breakdown=breakdown,
                )
            )
        else:
            remaining.append(seq)
    active[:] = remaining
    return finished


def simulate_open_loop(
    schedule: Sequence[RequestSpec],
    config: ReplicaSimConfig,
) -> tuple[RequestTelemetry, ...]:
    """Execute a deterministic open-loop simulation honoring scheduled arrival times tau_sched."""
    if not schedule:
        return ()

    max_kv_tokens = config.memory_budget.max_resident_kv_tokens(config.max_step_tokens)
    ordered = deque(sorted(schedule, key=lambda r: r.scheduled_arrival_ms))
    waiting: deque[RequestSpec] = deque()
    active: list[_ActiveSeq] = []
    completed: list[RequestTelemetry] = []
    clock_ms = ordered[0].scheduled_arrival_ms

    while ordered or waiting or active:
        while ordered and ordered[0].scheduled_arrival_ms <= clock_ms:
            waiting.append(ordered.popleft())

        if not waiting and not active and ordered:
            clock_ms = ordered[0].scheduled_arrival_ms
            while ordered and ordered[0].scheduled_arrival_ms <= clock_ms:
                waiting.append(ordered.popleft())

        used_kv = sum(s.current_resident_kv for s in active)
        while waiting:
            nxt = waiting[0]
            needed_kv = nxt.prompt_tokens + 1
            if used_kv + needed_kv > max_kv_tokens and active:
                break
            waiting.popleft()
            used_kv += needed_kv
            active.append(
                _ActiveSeq(
                    spec=nxt,
                    execution_start_ms=clock_ms,
                    rem_prefill_tokens=max(1, nxt.uncached_prompt_tokens),
                )
            )

        if active:
            clock_ms, done = _step_continuous_batch(clock_ms=clock_ms, active=active, config=config)
            completed.extend(done)

    return tuple(completed)


def simulate_closed_loop(
    template_workloads: Sequence[RequestSpec],
    *,
    concurrency: int,
    config: ReplicaSimConfig,
) -> tuple[RequestTelemetry, ...]:
    """Execute a naive closed-loop (concurrency=N) benchmark exhibiting coordinated omission."""
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1.")
    if not template_workloads:
        return ()

    max_kv_tokens = config.memory_budget.max_resident_kv_tokens(config.max_step_tokens)
    templates = deque(template_workloads)
    active: list[_ActiveSeq] = []
    completed: list[RequestTelemetry] = []
    clock_ms = 0.0

    while templates or active:
        used_kv = sum(s.current_resident_kv for s in active)
        while templates and len(active) < concurrency:
            tmpl = templates[0]
            needed_kv = tmpl.prompt_tokens + 1
            if used_kv + needed_kv > max_kv_tokens and active:
                break
            templates.popleft()
            used_kv += needed_kv
            # Closed-loop resets scheduled arrival to execution time, omitting queue wait!
            closed_spec = RequestSpec(
                request_id=tmpl.request_id,
                scheduled_arrival_ms=clock_ms,
                prompt_tokens=tmpl.prompt_tokens,
                cached_prefix_tokens=tmpl.cached_prefix_tokens,
                output_tokens=tmpl.output_tokens,
            )
            active.append(
                _ActiveSeq(
                    spec=closed_spec,
                    execution_start_ms=clock_ms,
                    rem_prefill_tokens=max(1, closed_spec.uncached_prompt_tokens),
                )
            )

        if active:
            clock_ms, done = _step_continuous_batch(clock_ms=clock_ms, active=active, config=config)
            completed.extend(done)

    return tuple(completed)


def compare_open_vs_closed_omission(
    *,
    schedule: Sequence[RequestSpec],
    closed_concurrency: int,
    config: ReplicaSimConfig,
    budget: LatencyBudget,
) -> tuple[GoodputSummary, GoodputSummary]:
    """Run identical workload payloads through open-loop vs. closed-loop benchmarking."""
    open_records = simulate_open_loop(schedule, config)
    closed_records = simulate_closed_loop(schedule, concurrency=closed_concurrency, config=config)

    if len(schedule) > 1:
        offered_span_s = max(1e-6, (schedule[-1].scheduled_arrival_ms - schedule[0].scheduled_arrival_ms) / 1000.0)
        offered_rps = len(schedule) / offered_span_s
    else:
        offered_rps = 1.0

    open_summary = summarize_goodput(
        open_records, budget, mode="open_loop", offered_rps=offered_rps
    )
    closed_summary = summarize_goodput(
        closed_records, budget, mode=f"closed_loop(N={closed_concurrency})", offered_rps=offered_rps
    )
    return open_summary, closed_summary
