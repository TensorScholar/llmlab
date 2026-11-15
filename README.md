# `llmlab` (`v0.1.2`) — Reference Implementation

[![CI](https://github.com/TensorScholar/llmlab/actions/workflows/ci.yml/badge.svg)](https://github.com/TensorScholar/llmlab/actions/workflows/ci.yml)
[![Status: Public, CI-Verified](https://img.shields.io/badge/Status-Public%2C%20CI--Verified-brightgreen.svg)](https://github.com/TensorScholar/llmlab)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official companion reference repository for the upcoming book *Inside Production LLM Systems: Engineering Latency, Cost, Reliability, and Security for Production AI Infrastructure* by Muhammad Atashi, featuring the runnable reference implementations for **Part 1 (Chapters 1–4)** and **Chapter 10: Durable and Multi-Agent Execution**.

---

## Architectural Overview

`llmlab` (`v0.1.2`) is engineered with a **100% local-first, zero-cloud-dependency** runtime (Python 3.11+, PostgreSQL 16, SQLAlchemy 2.0 `asyncio`, `asyncpg`, and Pydantic v2). Every analytical model, contract enforcement hook, and state-machine invariant maps 1-to-1 to the formal definitions and code listings in the book:

### Part 1 — Inference & Distributed Systems Foundations (Chapters 1–4, Pure Python)

* **`llmlab/tracing/` (Chapter 1 — Distributed Systems Framing, Four Budgets & Five Contracts):**
  * `spans.py`: `TraceSpan`, `TraceGraph`, `StreamingLatencyBreakdown`, `CriticalPathReport`, temporal containment checks, and DAG critical-path decomposition.
  * `budgets.py`: `FourBudgetLedger`, `LatencyBudget`, `MemoryBudget`, `FinancialBudget`, `TrustBudget`, and cross-budget coupling equations.
  * `contracts.py`: Runtime enforcement of the five contracts (`DeadlineContract`, `ResourceContract`, `StateContract`, `EffectContract`, `AuthorityContract`).
* **`llmlab/memory/` (Chapter 2 — Transformer Inference Mechanics & the Memory Wall):**
  * `hardware.py`: GPU accelerator profiles (`H100_SXM5`, `H200_SXM`, `B200_SXM`, `ACCELERATORS`) and Roofline compute/bandwidth ridge models.
  * `kv_cache.py`: Exact KV-cache and static weight byte accounting across `MHA`, `GQA`, and `MLA` (with up-projection absorption) plus quantization layouts (`FP16`, `BF16`, `FP8`, `INT8`, `INT4`, `NVFP4`).
  * `roofline.py`: `RooflineAnalyzer`, `StepRooflineResult`, critical batch-size solvers, and `MoEAllToAllResult` interconnect bounds.
* **`llmlab/loadgen/` (Chapter 3 — Latency Decomposition, Tail SLOs & Capacity Planning):**
  * `metrics.py`: Non-interpolated empirical order-statistic quantiles (`empirical_quantile`), joint-SLO goodput accounting (`summarize_goodput`), and Pareto frontier extraction (`extract_pareto_frontier`).
  * `queueing.py`: Heavy-tailed Kingman VUT queueing (`kingman_estimate`), Little's Law KV-cache saturation coupling (`solve_coupled_kv_saturation`), and head-of-line stall injection (`inject_hol_stall_breakdown`).
  * `generator.py`: Deterministic open-loop Poisson/burst (`simulate_open_loop`) and closed-loop (`simulate_closed_loop`, `compare_open_vs_closed_omission`) continuous-batching simulators.
  * `capacity.py`: Dual-constraint cluster sizing (`plan_cluster_capacity`, `CapacityPlan`, `WorkloadProfile`).
* **`llmlab/serving/` (Chapter 4 — High-Throughput LLM Serving Architectures):**
  * `paged_attention.py`: `PagedKVBlockManager`, virtual KV-cache block tables, Copy-on-Write branching, and fragmentation audits (`compare_contiguous_vs_paged`, `analyze_block_granularity_tradeoffs`).
  * `scheduler.py`: `UnifiedTokenScheduler`, chunked-prefill evaluation (`evaluate_chunk_sizes`, `dynamic_slack_chunk_size`), quadratic KV re-read accounting (`quadratic_kv_reread_bytes`), and preemption tradeoffs (`compare_preemption_policies`).
  * `radix_cache.py`: `RadixPrefixTree`, LRU eviction with ref-count pinning, and tool-order cache sensitivity (`simulate_tool_result_cache_sensitivity`).
  * `disaggregation.py`: Prefill-Decode (P/D) disaggregation fabric profiles (`FABRIC_NVLINK4_H100`, `FABRIC_NVLINK5_B200`, `FABRIC_RDMA_*`), KV migration checks (`evaluate_kv_migration`), and topology sizing (`plan_disaggregated_topology`).

### Part 3 — Durable Agentic Execution (Chapter 10, Python 3.11+ & PostgreSQL 16)

* **`sql/01_schema.sql` (Section 10.3.2 — Physical Storage Schema & Structural Invariants):** Defines the append-only `checkpoints` lineage table, the `checkpoint_writes` delta and side-effect ledger, the monotonic `fencing_token_seq` sequence, partial unique indexes for idempotency keys, and `CHECK` constraints enforcing root/child lineage and suspension state consistency.
* **`llmlab/durable/graph.py` (Section 10.1.3 — Static Cyclic Graph Verification):** Implements `GraphTopology`, `tarjan_scc` for $O(|V| + |E|)$ Strongly Connected Components (SCC) decomposition, entry reachability auditing from $v_{\text{start}}$, and condensation DAG sink validation (`validate_graph_termination`) to reject deadlocked cycles (`NonTerminatingCycleError`) and unreachable subgraphs (`OrphanComponentError`) at compile time.
* **`llmlab/durable/cas.py` (Sections 10.3.3 & 10.3.4 — Optimistic Concurrency Control & Commutative Rebasing):** Implements atomic branch head advancement via row-locked Compare-and-Swap (`CAS_HEAD_SQL`), split-brain zombie rejection using monotonic fencing tokens (`LeaseLost`), and conflict inspection (`READ_HEAD_SQL`) that distinguishes non-commutative races (`ConcurrencyConflict`) from join-semilattice transitions eligible for automatic state rebasing (`RetryFromVersion`).
* **`llmlab/durable/idempotency.py` (Section 10.4 — Deterministic Idempotency & Two-Phase Side Effects):** Implements length-prefixed `HMAC-SHA256` idempotency key derivation (`idempotency_key`) and the 3-state physical (`APPLIED`, `INTENT`, `COMMITTED`) to 5-state logical (`Attempted`, `Dispatched`, `Confirmed`, `Ambiguous`, `Reconciled`) execution protocol (`execute_idempotent_effect`, `RECORD_INTENT_SQL`, `CONFIRM_EFFECT_SQL`) across `IDEMPOTENT_BY_KEY`, `STATUS_QUERYABLE`, and `NON_IDEMPOTENT` external providers (`UnknownSideEffectOutcome` / `UnknownOutcomeError`).
* **`llmlab/durable/suspension.py` (Section 10.5 — Zero-Compute Human-in-the-Loop Suspension):** Implements zero-worker-compute `SUSPENDED` checkpoint transitions (`SUSPEND_HEAD_SQL`), audience-, version-, and expiry-bound HMAC resumption tokens stored exclusively by SHA-256 digest (`issue_resumption_token`), constant-time single-use CAS redemption (`verify_and_resume_workflow`, `RESUME_CAS_SQL`), and lock-free batch expiration reclamation using `FOR UPDATE SKIP LOCKED` (`sweep_expired_suspensions`, `SWEEP_EXPIRED_SUSPENSIONS_SQL`).
* **`llmlab/durable/envelope.py` (Section 10.6.2 — Typed Inter-Agent Envelopes & Capability Authorization):** Implements frozen, size-bounded (`MAX_ARGUMENT_BYTES = 8192`) Pydantic v2 message envelopes (`AgentEnvelope`, `ToolCallRequest`) with `ConfigDict(extra="forbid", frozen=True)` and server-side topology edge and tool capability enforcement via `AgentDirectory` and `authorize`.

---

## Repository Structure

```text
.
├── .github/workflows/ci.yml
├── .gitignore
├── CHANGELOG.md
├── LICENSE
├── Makefile
├── README.md
├── docker-compose.yml
├── pyproject.toml
├── assets/
│   └── figures/
│       ├── fig-10-1-state-engine-color.png
│       ├── fig-10-1-state-engine-print.png
│       ├── fig-10-2-forking-color.png
│       ├── fig-10-2-forking-print.png
│       ├── fig-10-3-dual-write-color.png
│       ├── fig-10-3-dual-write-print.png
│       ├── fig-10-4-fencing-color.png
│       └── fig-10-4-fencing-print.png
├── docs/
│   └── reviewer-quickstart.md
├── sql/
│   └── 01_schema.sql
├── llmlab/
│   ├── __init__.py
│   ├── py.typed
│   ├── tracing/                # Chapter 1: Trace DAGs, 4 budgets, 5 contracts
│   │   ├── __init__.py
│   │   ├── budgets.py
│   │   ├── contracts.py
│   │   └── spans.py
│   ├── memory/                 # Chapter 2: Hardware profiles, KV cache, Roofline
│   │   ├── __init__.py
│   │   ├── hardware.py
│   │   ├── kv_cache.py
│   │   └── roofline.py
│   ├── loadgen/                # Chapter 3: Queueing, simulators, goodput, sizing
│   │   ├── __init__.py
│   │   ├── capacity.py
│   │   ├── generator.py
│   │   ├── metrics.py
│   │   └── queueing.py
│   ├── serving/                # Chapter 4: PagedAttention, vLLM V1, Radix, P/D
│   │   ├── __init__.py
│   │   ├── disaggregation.py
│   │   ├── paged_attention.py
│   │   ├── radix_cache.py
│   │   └── scheduler.py
│   └── durable/                # Chapter 10: Durable & multi-agent runtime
│       ├── __init__.py
│       ├── cas.py
│       ├── envelope.py
│       ├── graph.py
│       ├── idempotency.py
│       └── suspension.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_tracing_contracts.py
    ├── test_memory_roofline.py
    ├── test_loadgen_queueing.py
    ├── test_serving_architectures.py
    ├── test_graph_termination.py
    ├── test_cas_concurrency.py
    ├── test_idempotency_crash.py
    └── test_suspension_tokens.py
```

---

## Quickstart & Verification Commands

Two verification paths are provided depending on whether you want instant CPU-only verification of the analytical/serving/graph modules or full PostgreSQL-backed concurrency testing:

### 1. Fast CPU-Only Verification (No Docker or PostgreSQL Required)

Verify strict static typing across all 5 packages and run the 29 pure-Python tests covering **Part 1 (Chapters 1–4)** plus **Chapter 10 static graph & envelope checks**:

```bash
# 1. Install llmlab in editable mode with development dependencies
pip install -e ".[dev]"

# 2. Run mypy --strict + non-DB test suite (29 tests across 5 test modules)
make verify-part1
```

### 2. Full Database-Backed Verification (Docker + PostgreSQL 16)

Reproduce the complete static type check alongside all 41 tests (Part 1 analytical/serving suites plus Chapter 10 PostgreSQL 16 transactional CAS, crash-recovery idempotency, and suspension-token tests):

```bash
# 1. Start the local PostgreSQL 16 container and initialize sql/01_schema.sql
docker compose up -d

# 2. Install llmlab in editable mode with development dependencies
pip install -e ".[dev]"

# 3. Execute the complete verification harness (mypy --strict + all 41 pytest tests)
make verify-book
```

For a step-by-step reviewer manifest, see [`docs/reviewer-quickstart.md`](docs/reviewer-quickstart.md).

<details>
<summary>Run the underlying checks individually</summary>

```bash
# Strict static type analysis across all 5 packages and test modules
mypy --strict llmlab tests

# Non-DB pure-Python subset (Part 1 Chapters 1-4 + Chapter 10 graph/envelope tests)
pytest -v \
  tests/test_tracing_contracts.py \
  tests/test_memory_roofline.py \
  tests/test_loadgen_queueing.py \
  tests/test_serving_architectures.py \
  tests/test_graph_termination.py

# Complete unit and PostgreSQL concurrency/crash-recovery integration suite (41 tests)
pytest -v tests/
```

</details>
