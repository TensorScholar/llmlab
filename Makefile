.PHONY: verify-book

verify-book:
	@echo "============================================================"
	@echo " EVIDENCE CLASS: LOCAL INTEGRATION + ANALYTICAL CHECKS"
	@echo " Target: llmlab Chapter 10 Verification Suite"
	@echo "============================================================"
	@echo "[1/2] Running strict static type analysis (mypy)..."
	mypy --strict llmlab tests
	@echo "[2/2] Executing distributed concurrency & crash-recovery tests..."
	pytest -v tests/
	@echo "============================================================"
	@echo " VERIFICATION SUCCESSFUL"
	@echo " GPU throughput: NOT TESTED (Out of scope for local suite)"
	@echo " Power-loss durability: NOT TESTED (Relies on OS fsync)"
	@echo "============================================================"
