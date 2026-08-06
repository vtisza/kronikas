# Contributing to kronikas

Thank you for your interest in contributing to **kronikas**! This document
explains how to set up a development environment, run tests, and submit changes.

## Getting started

1. Fork and clone the repository:

   ```bash
   git clone https://github.com/<your-username>/kronikas.git
   cd kronikas
   ```

2. Install dependencies with [uv](https://docs.astral.sh/uv/) (recommended):

   ```bash
   uv sync --group dev
   ```

   Or use the Makefile shortcut:

   ```bash
   make sync
   ```

   **Without uv** — a plain pip workflow also works:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

## Makefile targets

Run `make` (or `make help`) to see all available targets:

| Target | Description |
|---|---|
| `make sync` | Install dependencies with uv (recommended) |
| `make install` | Install with pip (fallback if uv is not available) |
| `make lint` | Run ruff linter |
| `make format` | Format code with ruff |
| `make check` | Run linter and format check (no changes, CI-friendly) |
| `make typecheck` | Run mypy over the package |
| `make test` | Run the full test suite |
| `make test-fast` | Run fast tests only (no MCMC sampling) |
| `make clean` | Remove build artifacts and caches |

## Project layout

```
src/kronikas/
    __init__.py      # Public API
    backtest.py      # Refit at past dates and score against the true result
    cli.py           # `kronikas` command-line entry point
    config.py        # ModelConfig / PollsterPrior dataclasses
    data.py          # CSV and DataFrame loading, validation, filtering
    diagnostics.py   # Sampler convergence diagnostics
    forecast.py      # ElectionForecast orchestrator
    model.py         # PyMC model building, inference, and result extraction
    py.typed         # PEP 561 marker
tests/
    conftest.py      # Shared fixtures
    test_*.py        # Test modules
```

## Running tests

```bash
make test-fast   # Fast tests only (no MCMC sampling) — use during development
make test        # Full suite including inference tests
```

Tests that run MCMC are marked `@pytest.mark.slow`. `make test-fast` deselects
them, which makes it a poor gate on its own: it skips the entire sampling path.
Run `make test` before opening a pull request. CI runs both, and the full suite
takes a few minutes.

Please make sure all tests pass before submitting a pull request.

## Making changes

1. Create a feature branch from `main`:

   ```bash
   git checkout -b my-feature main
   ```

2. Make your changes. Keep commits focused and write clear commit messages.

3. Add or update tests for any new or changed behaviour. Tests live in the
   `tests/` directory and use pytest.

4. Run `make check` and `make typecheck` to ensure your code passes linting,
   formatting, and type checking.

5. Run `make test` to verify nothing is broken.

6. Add an entry to `CHANGELOG.md` under "Unreleased".

7. Push your branch and open a pull request against `main`.

## Code style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and
formatting. Ruff is installed as part of the dev dependencies.

```bash
make lint        # Check for lint errors
make format      # Format code
make check       # Lint + format check (no modifications)
```

Or call ruff directly:

```bash
uv run ruff check .           # Lint
uv run ruff check --fix .     # Lint with auto-fix
uv run ruff format .          # Format
```

Please run `make check` before submitting a pull request.

The Ruff configuration lives in `pyproject.toml` and enforces:

- Target Python 3.10+.
- 88-character line length.
- Rules: pyflakes (`F`), pycodestyle (`E`), isort (`I`), pyupgrade (`UP`),
  flake8-bugbear (`B`), and flake8-simplify (`SIM`).

Additional style guidelines:

- Keep functions and methods focused — prefer small, well-named functions.
- Use type hints for public API signatures. The package ships `py.typed`, so
  annotations are part of the public contract; `make typecheck` must pass.

## Reporting bugs and requesting features

Open an issue on GitHub with a clear description. For bugs, include:

- Steps to reproduce the problem.
- Expected vs. actual behaviour.
- Python version and OS.
- A minimal CSV sample if the issue involves data loading or model fitting.

## License

By contributing you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE) that covers this project.
