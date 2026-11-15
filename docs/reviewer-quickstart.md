# Reviewer Quickstart & Reproducibility Manifest

Welcome, O'Reilly Editorial & Technical Reviewers. This suite lets you independently verify the analytical physics, capacity math, serving simulations, and durable execution invariants for **Part 1 (Chapters 1–4)** and **Chapter 10** without requiring a GPU or paid model APIs.

---

## Option A: Fast CPU-Only Verification (No Docker Required)

Verify strict static typing across all 5 packages and run the 29 pure-Python tests covering **Part 1 (Chapters 1–4)** plus **Chapter 10 graph and envelope verification** in under a second:

### Prerequisites
- Python 3.11+

### Execution
Run these commands from the repository root:

```bash
# 1. Install package in editable mode with development test dependencies
pip install -e ".[dev]"

# 2. Execute strict mypy + the 5 non-DB test modules (29 tests)
make verify-part1
```

### What This Verifies
1. **Chapter 1 (`llmlab.tracing`):** Temporal containment in trace DAGs, critical-path attribution, four-budget coupling equations, and five-contract violation controls.
2. **Chapter 2 (`llmlab.memory`):** Accelerator ridge points (`H100`, `H200`, `B200`), exact `MHA`/`GQA`/`MLA` KV-cache byte accounting, Roofline compute vs. bandwidth regimes, and MoE All-to-All bounds.
3. **Chapter 3 (`llmlab.loadgen`):** Kingman VUT heavy-tailed queueing, closed-loop coordinated omission vs. open-loop replay, HoL tail-SLO collapse, coupled KV saturation, and dual-constraint cluster sizing.
4. **Chapter 4 (`llmlab.serving`):** `PagedAttention` Copy-on-Write block accounting, `vLLM V1` chunked-prefill vs. preemption crossovers, `RadixAttention` ref-count pinning, and Prefill-Decode disaggregation fabric sizing.
5. **Chapter 10 Static Guarantees (`llmlab.durable`):** Tarjan SCC condensation cycle rejection and Pydantic v2 envelope/capability authorization.
6. **Type Safety:** 100% clean static analysis under `mypy --strict llmlab tests`.

---

## Option B: Full Database-Backed Verification (Docker + PostgreSQL 16)

Verify all 41 tests, combining the 29 pure-Python checks above with the 12 PostgreSQL 16 transactional concurrency, crash-recovery, and suspension-token tests from **Chapter 10**:

### Prerequisites
- Python 3.11+
- Docker Engine / Docker Desktop

### Execution
Run these commands from the repository root:

```bash
# 1. Start isolated PostgreSQL 16 service with exact schema & credentials
docker compose up -d

# 2. Install package in editable mode with development test dependencies (if not already installed)
pip install -e ".[dev]"

# 3. Execute the complete verification harness (mypy --strict + all 41 pytest tests)
make verify-book
```

### Additional Invariants Verified
1. **CAS Race Condition:** Two concurrent workers racing on `expected_version` resolve to exactly 1 winner, while stale fencing tokens fail with `LeaseLost` and commutative updates rebase.
2. **Intent/Commit Crash Recovery:** Simulated SIGKILL between outbox `INTENT` and `COMMITTED` states safely reconciles via provider-capability protocols.
3. **Zero-Compute Suspension & Expiry:** Single-use HMAC resumption tokens draw monotonic `nextval('fencing_token_seq')` tokens and expired checkpoints are safely swept via `FOR UPDATE SKIP LOCKED`.

### Teardown
When finished, stop the isolated container:

```bash
docker compose down -v
```
