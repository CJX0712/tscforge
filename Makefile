# TscForge development Makefile
# Author: 晨星 (CJX0712)
PY ?= python
VENV ?= .venv

.PHONY: help venv install dev install-tier0 lint format test bench demo clean

help:
	@echo "TscForge targets:"
	@echo "  venv          create a virtual environment"
	@echo "  install       install runtime + dev dependencies"
	@echo "  install-tier0 install optional Tier-0 backends (aeon, tslearn)"
	@echo "  lint          run ruff check"
	@echo "  format        run ruff format"
	@echo "  test          run the test suite with coverage"
	@echo "  bench         run the full benchmark (numpy backend)"
	@echo "  demo          run the interactive demo"
	@echo "  clean         remove caches and build artifacts"

venv:
	$(PY) -m venv $(VENV)

install:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e .
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install pytest pytest-cov ruff

install-tier0:
	$(PY) -m pip install aeon==1.6.0 tslearn==0.9.0

lint:
	$(PY) -m ruff check tscforge tests scripts

format:
	$(PY) -m ruff format tscforge tests scripts
	$(PY) -m ruff check --fix tscforge tests scripts || true

test:
	$(PY) -m pytest --cov=tscforge --cov-report=term-missing tests

bench:
	$(PY) scripts/run_full_bench.py numpy all no

demo:
	$(PY) -m tscforge.examples.run_demo

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
