.DEFAULT_GOAL := help

.PHONY: help install sync lint format check test test-fast clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install with pip (fallback if uv is not available)
	pip install -e ".[dev]"

sync: ## Install dependencies with uv (recommended)
	uv sync --group dev

lint: ## Run ruff linter
	uv run ruff check .

format: ## Format code with ruff
	uv run ruff format .

check: ## Run linter and format check (CI-friendly, no changes)
	uv run ruff check .
	uv run ruff format --check .

test: ## Run the full test suite
	uv run pytest tests/

test-fast: ## Run fast tests only (no MCMC sampling)
	uv run pytest tests/ -m "not slow"

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info src/*.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
