SHELL := /bin/bash
.PHONY: test test-failed test-unit test-scenarios test-integration test-quick

test:
	@source .venv/bin/activate && PYTHONPATH=. pytest tests/unit tests/scenarios tests/integration --tb=short -v

test-failed:
	@source .venv/bin/activate && PYTHONPATH=. pytest tests/unit tests/scenarios tests/integration --lf --tb=short -v

test-unit:
	@source .venv/bin/activate && PYTHONPATH=. pytest tests/unit --tb=short -v

test-scenarios:
	@source .venv/bin/activate && PYTHONPATH=. pytest tests/scenarios --tb=short -v

test-integration:
	@source .venv/bin/activate && PYTHONPATH=. pytest tests/integration --tb=short -v

test-quick:
	@source .venv/bin/activate && PYTHONPATH=. pytest tests/unit tests/scenarios tests/integration -x --tb=short

test-summary:
	@source .venv/bin/activate && PYTHONPATH=. pytest tests/unit tests/scenarios tests/integration --tb=no -q | grep -E "(FAILED|PASSED|ERROR)" | sort | uniq -c | sort -nr