#!/bin/bash
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=.
pytest tests/unit tests/scenarios tests/integration --tb=short -v "$@"