"""RadixAttention prefix trie, LRU eviction with ref-count pinning, and tool-order sensitivity."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence


class PinnedBlockEvictionError(RuntimeError):
    """Raised when an eviction target holds active request pins or pinned descendants."""


@dataclass(frozen=True, slots=True)
class PrefixHandle:
    """Active lease pinning matched and appended prefix KV blocks for one request."""

    request_id: str
    tokens: tuple[int, ...]
    matched_tokens: int
    total_tokens: int
    path_node_ids: tuple[int, ...]
    hit_blocks: int
    missed_blocks: int

    @property
    def cache_hit_ratio(self) -> float:
        """Fraction of prompt tokens satisfied directly from cached RadixTree blocks."""
        if self.total_tokens <= 0:
            return 0.0
        return self.matched_tokens / float(self.total_tokens)


@dataclass
class RadixNode:
    """Edge-compressed RadixAttention trie node mapping token segments to cached KV blocks."""

    node_id: int
    edge_tokens: tuple[int, ...]
    parent_id: int | None = None
    children: dict[int, RadixNode] = field(default_factory=dict)
    ref_count: int = 0
    last_access_step: int = 0
    block_ids: tuple[int, ...] = ()

    @property
    def token_count(self) -> int:
        """Number of prompt/completion tokens represented along this node's incoming edge."""
        return len(self.edge_tokens)

    @property
    def is_leaf(self) -> bool:
        """True iff this node currently has no descendant edges."""
        return len(self.children) == 0

    @property
    def is_evictable(self) -> bool:
        """Only unpinned leaf nodes (ref_count == 0 and no children) are valid LRU targets."""
        return self.node_id != 0 and self.ref_count == 0 and len(self.children) == 0


@dataclass(frozen=True, slots=True)
class ToolOrderCacheAudit:
    """Measured prefix cache reuse when parallel tool outputs arrive jittered vs. canonicalized."""

    num_turns: int
    tools_per_turn: int
    tokens_per_tool_result: int
    system_prompt_tokens: int
    jittered_hit_ratio: float
    canonical_hit_ratio: float
    jittered_prefill_tokens_recomputed: int
    canonical_prefill_tokens_recomputed: int
    prefill_compute_reduction_pct: float


class RadixPrefixTree:
    """LRU-evicting prefix cache trie enforcing ancestor pinning via reference counts."""

    def __init__(
        self,
        *,
        max_cached_tokens: int,
        kv_bytes_per_token: int,
        block_size: int = 16,
    ) -> None:
        if max_cached_tokens <= 0 or kv_bytes_per_token <= 0 or block_size <= 0:
            raise ValueError("max_cached_tokens, kv_bytes_per_token, and block_size must be positive.")

        self.max_cached_tokens = max_cached_tokens
        self.kv_bytes_per_token = kv_bytes_per_token
        self.block_size = block_size

        self._next_node_id: int = 1
        self._next_block_id: int = 0
        self._step_counter: int = 0
        self.root: RadixNode = RadixNode(node_id=0, edge_tokens=(), parent_id=None)
        self._nodes: dict[int, RadixNode] = {0: self.root}
        self._active_handles: dict[str, PrefixHandle] = {}
        self.total_cached_tokens: int = 0
        self.evicted_tokens_total: int = 0

    @property
    def cached_bytes(self) -> int:
        """Total HBM/DRAM KV footprint currently stored across all RadixTree edges."""
        return self.total_cached_tokens * self.kv_bytes_per_token

    @property
    def pinned_node_count(self) -> int:
        """Count of trie nodes directly referenced by active in-flight requests."""
        return sum(1 for node in self._nodes.values() if node.ref_count > 0)

    def _alloc_blocks(self, num_tokens: int) -> tuple[int, ...]:
        count = math.ceil(num_tokens / self.block_size)
        start = self._next_block_id
        self._next_block_id += count
        return tuple(range(start, start + count))

    def _collect_path(self, leaf_node_id: int) -> tuple[int, ...]:
        path: list[int] = []
        curr_id: int | None = leaf_node_id
        while curr_id is not None and curr_id != 0:
            path.append(curr_id)
            curr_id = self._nodes[curr_id].parent_id
        path.reverse()
        return tuple(path)

    def _split_node(self, node: RadixNode, split_idx: int) -> RadixNode:
        """Split an internal edge at `split_idx` while preserving active downstream pins."""
        prefix_tokens = node.edge_tokens[:split_idx]
        suffix_tokens = node.edge_tokens[split_idx:]
        parent = self._nodes[node.parent_id] if node.parent_id is not None else self.root

        split_node = RadixNode(
            node_id=self._next_node_id,
            edge_tokens=prefix_tokens,
            parent_id=parent.node_id,
            # Preserve all active pins passing through this newly separated prefix edge
            ref_count=node.ref_count,
            last_access_step=node.last_access_step,
            block_ids=self._alloc_blocks(len(prefix_tokens)),
        )
        self._next_node_id += 1
        self._nodes[split_node.node_id] = split_node

        # Reparent original node under split_node holding only the remaining suffix
        node.edge_tokens = suffix_tokens
        node.parent_id = split_node.node_id
        node.block_ids = self._alloc_blocks(len(suffix_tokens))

        split_node.children[suffix_tokens[0]] = node
        parent.children[prefix_tokens[0]] = split_node
        return split_node

    def match_prefix(self, tokens: Sequence[int]) -> tuple[int, int]:
        """Longest-prefix match returning (matched_token_count, deepest_matched_node_id) without mutating."""
        if not tokens:
            return 0, 0
        curr = self.root
        idx = 0
        n = len(tokens)
        while idx < n:
            first_tok = tokens[idx]
            child = curr.children.get(first_tok)
            if child is None:
                break
            edge = child.edge_tokens
            common = 0
            limit = min(len(edge), n - idx)
            while common < limit and edge[common] == tokens[idx + common]:
                common += 1
            idx += common
            if common < len(edge):
                # Partial edge match still constitutes matched tokens; return child for read inspection
                return idx, child.node_id
            curr = child
        return idx, curr.node_id

    def insert_and_pin(self, request_id: str, tokens: Sequence[int]) -> PrefixHandle:
        """Match cached prefix, split/insert uncached suffix, pin entire path, and return PrefixHandle."""
        if request_id in self._active_handles:
            raise ValueError(f"Request '{request_id}' already holds an active PrefixHandle.")
        if not tokens:
            raise ValueError("Cannot insert empty token sequence into RadixPrefixTree.")

        self._step_counter += 1
        step = self._step_counter
        tok_tuple = tuple(tokens)
        n = len(tok_tuple)

        curr = self.root
        idx = 0
        matched = 0
        while idx < n:
            child = curr.children.get(tok_tuple[idx])
            if child is None:
                break
            edge = child.edge_tokens
            common = 0
            limit = min(len(edge), n - idx)
            while common < limit and edge[common] == tok_tuple[idx + common]:
                common += 1
            matched += common
            idx += common
            if common < len(edge):
                curr = self._split_node(child, common)
                break
            curr.last_access_step = step
            curr = child

        matched_path_ids = self._collect_path(curr.node_id)
        for nid in matched_path_ids:
            node = self._nodes[nid]
            node.ref_count += 1
            node.last_access_step = step

        if idx < n:
            rem_tokens = tok_tuple[idx:]
            needed = len(rem_tokens)
            try:
                if self.total_cached_tokens + needed > self.max_cached_tokens:
                    self._evict_to_fit(needed)
            except BaseException:
                # Exception-safe rollback of newly acquired matched prefix pins
                for nid in matched_path_ids:
                    self._nodes[nid].ref_count -= 1
                raise

            new_node = RadixNode(
                node_id=self._next_node_id,
                edge_tokens=rem_tokens,
                parent_id=curr.node_id,
                ref_count=1,
                last_access_step=step,
                block_ids=self._alloc_blocks(len(rem_tokens)),
            )
            self._next_node_id += 1
            self._nodes[new_node.node_id] = new_node
            curr.children[rem_tokens[0]] = new_node
            self.total_cached_tokens += len(rem_tokens)
            curr = new_node

        path_ids = self._collect_path(curr.node_id)

        hit_blocks = matched // self.block_size
        total_blocks = math.ceil(n / self.block_size)
        missed_blocks = max(0, total_blocks - hit_blocks)
        handle = PrefixHandle(
            request_id=request_id,
            tokens=tok_tuple,
            matched_tokens=matched,
            total_tokens=n,
            path_node_ids=path_ids,
            hit_blocks=hit_blocks,
            missed_blocks=missed_blocks,
        )
        self._active_handles[request_id] = handle
        return handle

    def ancestor_path(self, request_id: str) -> tuple[int, ...]:
        """Return live root-to-terminal node IDs for an active request (post-any splits)."""
        handle = self._active_handles[request_id]
        return self._collect_path(handle.path_node_ids[-1])

    def release_handle(self, request_id: str) -> None:
        """Decrement ref_count pins along the live root-to-terminal prefix path."""
        handle = self._active_handles.pop(request_id, None)
        if handle is None:
            raise KeyError(f"Unknown active PrefixHandle '{request_id}'.")
        self._step_counter += 1
        step = self._step_counter
        live_path = self._collect_path(handle.path_node_ids[-1])
        for nid in live_path:
            node = self._nodes.get(nid)
            if node is not None:
                if node.ref_count <= 0:
                    raise RuntimeError(f"Reference count underflow on RadixNode {nid}.")
                node.ref_count -= 1
                node.last_access_step = step

    def evict_lru_tokens(self, target_free_tokens: int) -> int:
        """Evict coldest unpinned leaves bottom-up; returns total tokens reclaimed."""
        if target_free_tokens <= 0:
            return 0
        reclaimed = 0
        while reclaimed < target_free_tokens:
            evictable = [node for node in self._nodes.values() if node.is_evictable]
            if not evictable:
                break
            victim = min(evictable, key=lambda n: (n.last_access_step, n.node_id))
            reclaimed += self._remove_leaf_node(victim)
        return reclaimed

    def _evict_to_fit(self, incoming_tokens: int) -> None:
        excess = (self.total_cached_tokens + incoming_tokens) - self.max_cached_tokens
        if excess <= 0:
            return
        reclaimed = self.evict_lru_tokens(excess)
        if self.total_cached_tokens + incoming_tokens - reclaimed > self.max_cached_tokens and (
            self.total_cached_tokens + incoming_tokens > self.max_cached_tokens
        ):
            # If evictable leaves were exhausted while pins still exceed capacity, raise PinnedBlockEvictionError
            raise PinnedBlockEvictionError(
                f"Cannot reclaim {excess} tokens for incoming suffix of {incoming_tokens} tokens: "
                f"remaining cached tokens are pinned by {self.pinned_node_count} active tree nodes."
            )

    def _remove_leaf_node(self, node: RadixNode) -> int:
        parent_id = node.parent_id
        if parent_id is not None and parent_id in self._nodes:
            parent = self._nodes[parent_id]
            if node.edge_tokens:
                parent.children.pop(node.edge_tokens[0], None)
        freed = node.token_count
        self.total_cached_tokens -= freed
        self.evicted_tokens_total += freed
        del self._nodes[node.node_id]
        return freed

    def force_evict_node(self, node_id: int) -> int:
        """Negative-control / explicit eviction attempting to drop a node; rejects pinned/internal nodes."""
        node = self._nodes.get(node_id)
        if node is None:
            raise KeyError(f"Unknown RadixNode {node_id}.")
        if node_id == 0:
            raise PinnedBlockEvictionError("Cannot evict RadixPrefixTree root node 0.")
        if node.ref_count > 0:
            raise PinnedBlockEvictionError(
                f"Refused eviction of RadixNode {node_id}: active ref_count={node.ref_count}."
            )
        if not node.is_leaf:
            raise PinnedBlockEvictionError(
                f"Refused eviction of internal RadixNode {node_id}: node has {len(node.children)} child subtrees."
            )
        return self._remove_leaf_node(node)


def simulate_tool_result_cache_sensitivity(
    *,
    num_turns: int = 8,
    tools_per_turn: int = 4,
    tokens_per_tool_result: int = 256,
    system_prompt_tokens: int = 2048,
    assistant_tokens_per_turn: int = 128,
    max_cached_tokens: int = 131_072,
    kv_bytes_per_token: int = 65_536,
    seed: int = 42,
) -> ToolOrderCacheAudit:
    """Demonstrate parallel tool return-order jitter destroying RadixTree prefix hits vs. canonical sorting."""
    if num_turns < 2 or tools_per_turn < 2:
        raise ValueError("num_turns and tools_per_turn must be >= 2.")

    rng = random.Random(seed)
    sys_tokens = [10_000 + i for i in range(system_prompt_tokens)]

    # Generate deterministic shared parallel tool-result payloads plus unique per-request tails
    tool_blocks: list[list[int]] = []
    token_cursor = 50_000
    for _ in range(tools_per_turn):
        block = list(range(token_cursor, token_cursor + tokens_per_tool_result))
        token_cursor += tokens_per_tool_result
        tool_blocks.append(block)

    def _run_batch(canonical: bool, num_requests: int) -> tuple[float, int]:
        tree = RadixPrefixTree(
            max_cached_tokens=max_cached_tokens,
            kv_bytes_per_token=kv_bytes_per_token,
            block_size=16,
        )
        matched_total = 0
        prompt_total = 0
        nonlocal token_cursor

        for req_idx in range(num_requests):
            prompt: list[int] = list(sys_tokens)
            indices = list(range(tools_per_turn))
            if not canonical and req_idx > 0:
                # Simulate arrival-order permutation across concurrent parallel tool executions
                rng.shuffle(indices)
                while indices == list(range(tools_per_turn)):
                    rng.shuffle(indices)
            for idx in indices:
                prompt.extend(tool_blocks[idx])
            # Append per-request unique query/turn tokens
            unique_tail = list(range(token_cursor, token_cursor + assistant_tokens_per_turn))
            token_cursor += assistant_tokens_per_turn
            prompt.extend(unique_tail)

            handle = tree.insert_and_pin(f"req_{req_idx}", prompt)
            if req_idx > 0:
                matched_total += handle.matched_tokens
                prompt_total += handle.total_tokens
            tree.release_handle(handle.request_id)

        hit_ratio = matched_total / float(prompt_total) if prompt_total > 0 else 0.0
        recomputed = prompt_total - matched_total
        return hit_ratio, recomputed

    j_hit, j_recomp = _run_batch(canonical=False, num_requests=num_turns)
    c_hit, c_recomp = _run_batch(canonical=True, num_requests=num_turns)
    reduction_pct = ((j_recomp - c_recomp) / float(j_recomp) * 100.0) if j_recomp > 0 else 0.0

    return ToolOrderCacheAudit(
        num_turns=num_turns,
        tools_per_turn=tools_per_turn,
        tokens_per_tool_result=tokens_per_tool_result,
        system_prompt_tokens=system_prompt_tokens,
        jittered_hit_ratio=j_hit,
        canonical_hit_ratio=c_hit,
        jittered_prefill_tokens_recomputed=j_recomp,
        canonical_prefill_tokens_recomputed=c_recomp,
        prefill_compute_reduction_pct=reduction_pct,
    )
