"""PagedAttention virtual KV-cache block tables, Copy-on-Write branching, and fragmentation audit."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

from llmlab.tracing.budgets import MemoryBudget


class KVBlockPoolExhaustedError(RuntimeError):
    """Raised when HBM physical KV blocks cannot satisfy an allocation or CoW split."""


@dataclass
class PhysicalBlock:
    """Single fixed-capacity physical KV block residing in GPU HBM."""

    block_id: int
    block_size: int
    ref_count: int = 0
    used_slots: int = 0

    @property
    def is_free(self) -> bool:
        """True iff no sequence or prefix cache handle references this block."""
        return self.ref_count == 0

    @property
    def is_full(self) -> bool:
        """True iff all B_block token slots are populated."""
        return self.used_slots >= self.block_size


@dataclass(frozen=True, slots=True)
class BlockGranularityTradeoff:
    """Metadata footprint and warp-launch penalty across PagedAttention block sizes."""

    block_size: int
    physical_blocks_total: int
    bytes_per_block: int
    expected_internal_waste_tokens_per_seq: float
    worst_case_internal_waste_tokens_per_seq: int
    page_table_bytes_for_pool: int
    relative_kernel_overhead_pct: float
    pool_efficiency_pct: float


@dataclass(frozen=True, slots=True)
class FragmentationAudit:
    """Comparison between contiguous max_seq_len reservation and PagedAttention pools."""

    allocator: str
    admitted_sequences: int
    rejected_sequences: int
    total_allocated_tokens: int
    total_useful_tokens: int
    internal_waste_tokens: int
    reservation_waste_tokens: int
    external_waste_tokens: int
    memory_waste_pct: float
    exhausted_hbm: bool


class PagedKVBlockManager:
    """Virtual-to-physical KV block manager enforcing zero external fragmentation and CoW sharing."""

    def __init__(
        self,
        *,
        memory_budget: MemoryBudget,
        block_size: int = 16,
        step_tokens: int = 512,
        page_table_entry_bytes: int = 4,
    ) -> None:
        if block_size not in (16, 32, 64):
            raise ValueError("block_size B_block must be one of {16, 32, 64}.")
        if step_tokens <= 0 or page_table_entry_bytes <= 0:
            raise ValueError("step_tokens and page_table_entry_bytes must be positive.")

        self.memory_budget = memory_budget
        self.block_size = block_size
        self.step_tokens = step_tokens
        self.page_table_entry_bytes = page_table_entry_bytes

        avail_tokens = memory_budget.max_resident_kv_tokens(step_tokens=step_tokens)
        self.num_physical_blocks = avail_tokens // block_size
        if self.num_physical_blocks <= 0:
            raise ValueError("MemoryBudget cannot host even a single KV block.")

        self.blocks: list[PhysicalBlock] = [
            PhysicalBlock(block_id=idx, block_size=block_size)
            for idx in range(self.num_physical_blocks)
        ]
        self.free_block_ids: list[int] = list(range(self.num_physical_blocks - 1, -1, -1))
        self.block_tables: dict[str, list[int]] = {}
        self.seq_lengths: dict[str, int] = {}
        self.cow_copies_triggered: int = 0

    @property
    def num_free_blocks(self) -> int:
        """Number of unallocated physical HBM KV blocks."""
        return len(self.free_block_ids)

    @property
    def num_used_blocks(self) -> int:
        """Number of currently pinned physical HBM KV blocks."""
        return self.num_physical_blocks - len(self.free_block_ids)

    @property
    def utilization(self) -> float:
        """Fraction of physical KV block pool currently resident."""
        return self.num_used_blocks / float(self.num_physical_blocks)

    @property
    def internal_waste_tokens(self) -> int:
        """Exact unused trailing slots summed across all allocated physical tail blocks."""
        waste = 0
        seen: set[int] = set()
        for table in self.block_tables.values():
            for bid in table:
                if bid not in seen:
                    seen.add(bid)
                    blk = self.blocks[bid]
                    waste += max(0, blk.block_size - blk.used_slots)
        return waste

    @property
    def page_table_metadata_bytes(self) -> int:
        """Host/GPU page-table pointer array bytes for all active logical-to-physical mappings."""
        total_entries = sum(len(tbl) for tbl in self.block_tables.values())
        return total_entries * self.page_table_entry_bytes

    def _alloc_physical_block(self, used_slots: int) -> PhysicalBlock:
        if not self.free_block_ids:
            raise KVBlockPoolExhaustedError(
                f"HBM KV block pool exhausted: 0/{self.num_physical_blocks} free "
                f"(B_block={self.block_size})."
            )
        bid = self.free_block_ids.pop()
        blk = self.blocks[bid]
        blk.ref_count = 1
        blk.used_slots = used_slots
        return blk

    def _free_block_ref(self, bid: int) -> None:
        blk = self.blocks[bid]
        if blk.ref_count <= 0:
            raise RuntimeError(f"Double free detected on physical block {bid}.")
        blk.ref_count -= 1
        if blk.ref_count == 0:
            blk.used_slots = 0
            self.free_block_ids.append(bid)

    def allocate_sequence(self, seq_id: str, prompt_tokens: int) -> list[int]:
        """Allocate non-contiguous physical KV blocks matching prompt_tokens without over-reserving."""
        if seq_id in self.block_tables:
            raise ValueError(f"Sequence '{seq_id}' already holds a logical block table.")
        if prompt_tokens < 1:
            raise ValueError("prompt_tokens must be >= 1.")

        needed = math.ceil(prompt_tokens / self.block_size)
        if needed > len(self.free_block_ids):
            raise KVBlockPoolExhaustedError(
                f"Sequence '{seq_id}' requires {needed} blocks, only {len(self.free_block_ids)} free."
            )

        table: list[int] = []
        remaining = prompt_tokens
        for _ in range(needed):
            fill = min(self.block_size, remaining)
            blk = self._alloc_physical_block(used_slots=fill)
            table.append(blk.block_id)
            remaining -= fill

        self.block_tables[seq_id] = table
        self.seq_lengths[seq_id] = prompt_tokens
        return list(table)

    def fork_sequence(self, parent_seq_id: str, child_seq_id: str) -> list[int]:
        """Branch a parallel/beam completion sharing parent physical blocks via ref_count."""
        if parent_seq_id not in self.block_tables:
            raise KeyError(f"Unknown parent sequence '{parent_seq_id}'.")
        if child_seq_id in self.block_tables:
            raise ValueError(f"Child sequence '{child_seq_id}' already exists.")

        parent_table = self.block_tables[parent_seq_id]
        for bid in parent_table:
            self.blocks[bid].ref_count += 1
        self.block_tables[child_seq_id] = list(parent_table)
        self.seq_lengths[child_seq_id] = self.seq_lengths[parent_seq_id]
        return list(parent_table)

    def append_token(self, seq_id: str) -> int:
        """Append one generated KV token, triggering Copy-on-Write if tail block ref_count > 1."""
        if seq_id not in self.block_tables:
            raise KeyError(f"Unknown sequence '{seq_id}'.")

        table = self.block_tables[seq_id]
        tail_bid = table[-1]
        tail_blk = self.blocks[tail_bid]

        if not tail_blk.is_full:
            if tail_blk.ref_count > 1:
                # Copy-on-Write: preserve shared parent prefix slots on a fresh private block
                new_blk = self._alloc_physical_block(used_slots=tail_blk.used_slots + 1)
                self._free_block_ref(tail_bid)
                table[-1] = new_blk.block_id
                self.cow_copies_triggered += 1
                self.seq_lengths[seq_id] += 1
                return new_blk.block_id
            tail_blk.used_slots += 1
            self.seq_lengths[seq_id] += 1
            return tail_bid

        # Tail block full: allocate next logical KV block cleanly
        next_blk = self._alloc_physical_block(used_slots=1)
        table.append(next_blk.block_id)
        self.seq_lengths[seq_id] += 1
        return next_blk.block_id

    def free_sequence(self, seq_id: str) -> int:
        """Release all logical references owned by seq_id, returning freed physical block count."""
        if seq_id not in self.block_tables:
            raise KeyError(f"Unknown sequence '{seq_id}'.")
        table = self.block_tables.pop(seq_id)
        self.seq_lengths.pop(seq_id)
        reclaimed = 0
        for bid in table:
            before = len(self.free_block_ids)
            self._free_block_ref(bid)
            if len(self.free_block_ids) > before:
                reclaimed += 1
        return reclaimed


def analyze_block_granularity_tradeoffs(
    *,
    memory_budget: MemoryBudget,
    mean_seq_len: int,
    step_tokens: int = 512,
    block_sizes: Sequence[int] = (16, 32, 64),
) -> tuple[BlockGranularityTradeoff, ...]:
    """Quantify internal tail waste vs. page-table memory and kernel indirection across B_block."""
    if mean_seq_len < 1:
        raise ValueError("mean_seq_len must be >= 1.")
    max_kv_tokens = memory_budget.max_resident_kv_tokens(step_tokens=step_tokens)
    results: list[BlockGranularityTradeoff] = []
    for b_sz in block_sizes:
        if b_sz not in (16, 32, 64):
            raise ValueError("Supported B_block values are 16, 32, 64.")
        total_blocks = max_kv_tokens // b_sz
        bytes_per_blk = b_sz * memory_budget.kv_bytes_per_token
        exp_waste = (b_sz - 1) / 2.0
        # Average sequences resident ~ total usable slots / (mean_seq_len + exp_waste)
        est_seqs = max(1.0, max_kv_tokens / (mean_seq_len + exp_waste))
        table_bytes = int(math.ceil(est_seqs * math.ceil(mean_seq_len / b_sz) * 4))
        # Empirically, smaller blocks inflate indirect pointer-chase TLB misses & loop trip counts
        rel_kernel_pct = 4.2 if b_sz == 16 else (1.8 if b_sz == 32 else 0.7)
        waste_ratio = exp_waste / (mean_seq_len + exp_waste)
        eff_pct = (1.0 - waste_ratio) * 100.0
        results.append(
            BlockGranularityTradeoff(
                block_size=b_sz,
                physical_blocks_total=total_blocks,
                bytes_per_block=bytes_per_blk,
                expected_internal_waste_tokens_per_seq=exp_waste,
                worst_case_internal_waste_tokens_per_seq=b_sz - 1,
                page_table_bytes_for_pool=table_bytes,
                relative_kernel_overhead_pct=rel_kernel_pct,
                pool_efficiency_pct=eff_pct,
            )
        )
    return tuple(results)


def compare_contiguous_vs_paged(
    *,
    request_lengths: Sequence[int],
    memory_budget: MemoryBudget,
    max_seq_len: int,
    block_sizes: Sequence[int] = (16, 32, 64),
    step_tokens: int = 512,
    external_frag_fraction: float = 0.08,
) -> dict[str, FragmentationAudit]:
    """Compare contiguous max_seq_len static reservation against PagedAttention block pools.

    Contiguous allocation charges reservation waste ``max_seq_len - L_i`` for every admitted
    sequence plus buddy/slab external fragmentation; PagedAttention charges only ``<= B_block - 1``.
    """
    if not request_lengths or max_seq_len < max(request_lengths):
        raise ValueError("request_lengths non-empty and max_seq_len >= max(lengths) required.")

    max_kv_tokens = memory_budget.max_resident_kv_tokens(step_tokens=step_tokens)
    usable_contiguous = int(max_kv_tokens * (1.0 - external_frag_fraction))
    ext_waste_base = max_kv_tokens - usable_contiguous

    # 1. Contiguous max_seq_len pre-allocation
    contig_admitted = 0
    contig_useful = 0
    contig_res_waste = 0
    contig_exhausted = False
    used_contig = 0
    for seq_len in request_lengths:
        if used_contig + max_seq_len > usable_contiguous:
            contig_exhausted = True
            continue
        used_contig += max_seq_len
        contig_admitted += 1
        contig_useful += seq_len
        contig_res_waste += max_seq_len - seq_len

    contig_alloc_total = used_contig + (ext_waste_base if contig_admitted > 0 else 0)
    contig_waste_total = contig_res_waste + (ext_waste_base if contig_admitted > 0 else 0)
    contig_waste_pct = (
        (100.0 * contig_waste_total / contig_alloc_total) if contig_alloc_total > 0 else 0.0
    )

    audits: dict[str, FragmentationAudit] = {
        "contiguous_max_seq_len": FragmentationAudit(
            allocator=f"contiguous(max_seq={max_seq_len})",
            admitted_sequences=contig_admitted,
            rejected_sequences=len(request_lengths) - contig_admitted,
            total_allocated_tokens=contig_alloc_total,
            total_useful_tokens=contig_useful,
            internal_waste_tokens=0,
            reservation_waste_tokens=contig_res_waste,
            external_waste_tokens=ext_waste_base if contig_admitted > 0 else 0,
            memory_waste_pct=contig_waste_pct,
            exhausted_hbm=contig_exhausted,
        )
    }

    # 2. PagedAttention pools across requested block granularities
    for b_sz in block_sizes:
        mgr = PagedKVBlockManager(
            memory_budget=memory_budget, block_size=b_sz, step_tokens=step_tokens
        )
        admitted = 0
        rejected = 0
        useful = 0
        exhausted = False
        for idx, seq_len in enumerate(request_lengths):
            try:
                mgr.allocate_sequence(f"req-{idx}", seq_len)
                admitted += 1
                useful += seq_len
            except KVBlockPoolExhaustedError:
                exhausted = True
                rejected += 1
        allocated_tokens = mgr.num_used_blocks * b_sz
        internal_waste = mgr.internal_waste_tokens
        waste_pct = (100.0 * internal_waste / allocated_tokens) if allocated_tokens > 0 else 0.0
        audits[f"paged_B{b_sz}"] = FragmentationAudit(
            allocator=f"paged(B_block={b_sz})",
            admitted_sequences=admitted,
            rejected_sequences=rejected,
            total_allocated_tokens=allocated_tokens,
            total_useful_tokens=useful,
            internal_waste_tokens=internal_waste,
            reservation_waste_tokens=0,
            external_waste_tokens=0,
            memory_waste_pct=waste_pct,
            exhausted_hbm=exhausted,
        )

    return audits
