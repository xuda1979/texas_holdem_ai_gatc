# Contributing

Thanks for your interest in improving Texas Hold'em AI GATC. This document
describes the workflow we use for changes large and small.

## Development setup

1. Clone the repository.
2. Create a virtual environment and install the development dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```

3. Install the pre-commit hooks (optional but recommended):

   ```bash
   pre-commit install
   ```

## Running the tests

Before opening a pull request, run the full test suite and the linters:

```bash
pytest -q
ruff check src tests tools
mypy src
```

The project keeps 380+ unit and integration tests green on `main`. Tests
under `tests/e2e/` require `E2E_BASE_URL` to be set (used by the container
workflow in `.github/workflows/e2e-tests.yml`).

## Code style

- Line length is 100 (configured in `ruff.toml` and `pyproject.toml`).
- Imports are sorted with `isort` (first-party packages are `gatc_poker`,
  `gatc_holdem`, `poker_ai`).
- Type annotations are required on public functions; `mypy` runs with
  `ignore_missing_imports = true`.
- Avoid leaving `print` statements in committed code — use the logging
  utilities in `poker_ai.monitoring`.

## Pull request checklist

- [ ] Tests pass locally (`pytest -q`).
- [ ] `ruff check src tests tools` reports no new errors.
- [ ] Public API changes are documented in `README.md` and, where relevant,
      `docs/common_workflows.md`.
- [ ] New behavior is covered by a test.
- [ ] Commit messages follow the conventional-commits style used in `git log`
      (e.g. `feat:`, `fix:`, `docs:`, `chore:`, `test:`).

## Reporting bugs

Open a GitHub issue with:

1. The exact command you ran.
2. The Python version and OS.
3. A minimal reproduction (a script or a test case).
4. The full traceback, if any.

## Licensing

By contributing, you agree that your contributions will be licensed under the
MIT License (see `LICENSE`).
