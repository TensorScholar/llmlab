.PHONY: verify-part1 verify-book

verify-part1:
	@echo "============================================================"
	@echo " EVIDENCE CLASS: PURE-PYTHON ANALYTICAL & CONTRACT CHECKS"
	@echo " Target: llmlab Part 1 (Ch. 1-4) & Ch. 10 Graph/Envelope Suite"
	@echo "============================================================"
	@echo "[1/2] Running strict static type analysis (mypy)..."
	mypy --strict llmlab tests
	@echo "[2/2] Executing non-DB verification suite (no Docker required)..."
	pytest -v tests/test_tracing_contracts.py tests/test_memory_roofline.py tests/test_loadgen_queueing.py tests/test_serving_architectures.py tests/test_graph_termination.py
	@echo "============================================================"
	@echo " PART 1 + STATIC VERIFICATION SUCCESSFUL"
	@echo " PostgreSQL runtime tests: NOT RUN (run 'make verify-book')"
	@echo "============================================================"

verify-book:
	@echo "============================================================"
	@echo " EVIDENCE CLASS: LOCAL INTEGRATION + ANALYTICAL CHECKS"
	@echo " Target: llmlab Full Suite (Part 1 Ch. 1-4 + Ch. 10)"
	@echo "============================================================"
	@echo "[1/2] Running strict static type analysis (mypy)..."
	mypy --strict llmlab tests
	@echo "[2/2] Executing full analytical & PostgreSQL test suite..."
	pytest -v tests/
	@echo "============================================================"
	@echo " VERIFICATION SUCCESSFUL"
	@echo " GPU throughput: NOT TESTED (Out of scope for local suite)"
	@echo " Power-loss durability: NOT TESTED (Relies on OS fsync)"
	@echo "============================================================"
