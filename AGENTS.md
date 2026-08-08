# AGENTS.md

This file defines repository-wide rules for coding agents working on
`kronikas`. Human instructions in the current request take precedence. For
general contribution policy, also read `CONTRIBUTING.md`.

## Project context

`kronikas` is a typed Python library and CLI for hierarchical Bayesian election
forecasting. Source code lives under `src/kronikas/`; tests live under `tests/`.
The supported runtime range is Python 3.10 through 3.12.

Important modules:

- `data.py`: input loading, normalization, filtering, and validation.
- `config.py`: public configuration and prior dataclasses.
- `model.py`: PyMC model construction, sampling, and result extraction.
- `forecast.py`: high-level forecast orchestration.
- `backtesting.py`: historical refits and scoring.
- `diagnostics.py`: convergence checks and warnings.
- `cli.py`: command-line interface and JSON output.
- `__init__.py`: package-level public API.

### Architecture boundaries

Preserve the repository's existing dependency direction:

```text
config     data     diagnostics
   \         |         /
            model
          /       \
     forecast   backtesting
          \       /
             cli

__init__.py re-exports the supported public surface; it is not a logic layer.
```

- Keep validation and normalization of poll inputs in `data.py`; keep prior and
  sampler configuration in `config.py`; keep convergence interpretation in
  `diagnostics.py`.
- Keep PyMC graph construction, inference, posterior transformation, and
  `ForecastResult` behavior in `model.py` unless a cohesive new module is
  warranted.
- `forecast.py` and `backtesting.py` orchestrate lower layers. They may compose
  data and model operations, but should not duplicate their validation or
  statistical transformations.
- `cli.py` adapts command-line arguments and output to library APIs. Do not put
  behavior exclusively in the CLI when Python callers also need it.
- `__init__.py` should contain metadata handling and explicit re-exports only.
  Core modules must not depend on package-level re-exports.
- Avoid circular imports and generic `utils.py` dumping grounds. Put a helper
  with the concept it serves; extract a focused module only when ownership is
  genuinely shared.
- Mirror source responsibilities in tests. Cross-module end-to-end behavior may
  live in the test file for the public entry point that owns it.

## Working rules

- Inspect the relevant implementation, tests, and documentation before making
  changes. Do not guess at existing behavior.
- Keep changes narrowly scoped to the request. Preserve unrelated user edits
  and untracked files; never clean or reset the worktree to make it tidy.
- Prefer the smallest coherent fix. Avoid opportunistic refactors, dependency
  upgrades, mass formatting, or generated-file churn.
- Use `rg` or `rg --files` for repository searches.
- Do not edit release metadata, publish packages, create tags, push branches,
  or open pull requests unless explicitly asked.
- Do not hand-edit `uv.lock`. Change dependencies in `pyproject.toml`, then use
  `uv lock` or `uv sync` so the lockfile is updated by uv.
- Never weaken validation, tests, typing, or convergence checks merely to make
  a failure disappear. Fix the underlying behavior or explain the blocker.

## Development setup and commands

Use uv unless the task explicitly requires the documented pip fallback:

```bash
uv sync --group dev
```

Prefer the repository's Make targets:

```bash
make check       # Ruff lint and formatting checks
make typecheck   # mypy over src/kronikas
make test-fast   # tests excluding @pytest.mark.slow MCMC tests
make test        # full suite, including sampling
```

During iteration, run the narrowest relevant pytest selection, for example:

```bash
uv run pytest tests/test_data.py -q
uv run pytest tests/test_model.py::TestBuildModel -q
```

`make test-fast` is useful feedback but is not a complete validation gate: it
omits the end-to-end PyMC sampling path. Run `make test` before every pull
request, as required by `CONTRIBUTING.md`, and whenever model construction,
sampling, result extraction, diagnostics, or sampler configuration changes.

## Code conventions

- Follow Ruff configuration in `pyproject.toml`: Python 3.10 syntax, an
  88-character line length, and the configured `E`, `F`, `I`, `UP`, `B`, and
  `SIM` rules.
- Add precise type hints to public APIs. This package ships `py.typed`, so
  public annotations are part of the compatibility contract.
- Keep functions focused and use descriptive names. Match existing module and
  test organization before introducing a new abstraction.
- Validate external inputs at their boundary. Reject non-finite, malformed,
  out-of-range, or temporally invalid data with actionable errors.
- Preserve deterministic ordering and reproducibility. Thread random seeds
  through sampling code and use fixed seeds in tests.
- Avoid importing optional plotting, notebook, or application dependencies into
  the core package.

### Naming conventions

- Use `snake_case` for modules, functions, methods, local variables, pytest
  fixtures, and CLI implementation helpers.
- Use `PascalCase` for classes, dataclasses, exceptions, warnings, and test
  classes. Prefix test classes with `Test`.
- Use `UPPER_SNAKE_CASE` only for true module-level constants. Do not disguise
  mutable global state as a constant.
- Name tests `test_<behavior>`, not `test_<method_name>` when the observable
  behavior can be stated more clearly. A test name should identify the
  condition and expected outcome without requiring its body to be read.
- Use domain terms already established by the public API: candidate, pollster,
  house effect, shared bias, reference date, election day, draw, chain, and
  time step. Do not introduce synonyms for the same concept in neighboring
  modules.
- Include units or scale in names when ambiguity would be dangerous, such as
  `_days`, `_percent`, `_probability`, `_idx`, or `_samples`.
- Boolean names should read as predicates, normally beginning with `is_`,
  `has_`, `can_`, `should_`, or `include_`.
- Private implementation details start with `_`. Do not make a public-looking
  name and rely only on omission from `__all__` to signal that it is private.
- CLI flags use lowercase kebab-case. Python parameters and configuration
  fields use the equivalent snake_case spelling.
- New branch names should be short, lowercase, and descriptive, using a prefix
  such as `fix/`, `feat/`, `docs/`, `test/`, `refactor/`, or `chore/`, followed
  by hyphenated words; for example, `fix/reject-future-polls`.

## Statistical and domain invariants

Changes to the model must preserve these invariants unless the request
explicitly changes the documented behavior:

- Candidate vote shares are compositional: each posterior draw is non-negative
  and sums to 100 percent.
- Polls after the applicable reference date must not leak into a forecast or
  backtest; observations after the election are invalid.
- The time grid ends on election day, and time-scale priors remain consistent
  when grid resolution changes.
- Multi-pollster house effects remain identifiable and zero-sum across both
  candidates and pollsters. Single-pollster behavior must stay explicit.
- Shared polling bias is a scenario/prior assumption, not information learned
  from one election without identification.
- Warmup samples are excluded from public posterior data. Candidate, pollster,
  chain, draw, and time dimensions must remain aligned when reshaping arrays.
- `win_probabilities` means election-day vote-share plurality probability; do
  not describe it as a general probability of winning office.
- Backtests must use only information available at each historical cutoff.

For numerical code, assert shapes and coordinates at boundaries and prefer
stable vectorized NumPy/PyTensor operations over Python loops in model graphs.
Do not make stochastic tests pass by broadening tolerances without a statistical
justification.

### Numerical verification

- Assert exact structural properties exactly: names, shapes, dimensions,
  coordinates, ordering, date boundaries, and presence of trace groups.
- Use `pytest.approx` for scalar floating-point comparisons and
  `numpy.testing.assert_allclose` for arrays. Supply an explicit absolute or
  relative tolerance when the meaningful scale is not obvious.
- Derive tolerances from numerical precision, Monte Carlo error, or a stated
  domain bound. Do not copy a loose tolerance from an unrelated test.
- Test mathematical invariants directly, including non-negativity, sums to 1
  or 100, monotonicity, zero-sum constraints, and agreement between summary
  values and their source samples.
- For transformations, test simple hand-calculated cases, boundary cases, and
  round trips where an inverse exists. Check array axes and labels, not only
  aggregate values.
- Use deterministic seeds for regression tests. When making a statistical
  quality claim, use enough chains/draws or repeated seeds to support it and
  report the diagnostic evidence; the tiny `fast_config` is insufficient.
- Compare a model change to the previous behavior on a small representative
  dataset when its consequences are not captured by local invariants. Explain
  expected differences rather than snapshotting unexplained posterior values.

### Performance and resource use

- Optimize measured bottlenecks. For a performance claim, record a reproducible
  before/after command, representative input, Python version, dependency
  environment, and wall-time or memory result.
- Do not set a rigid performance threshold from a noisy local MCMC run. Prefer
  stable unit-level benchmarks or compare multiple runs when sampling dominates
  variance.
- Avoid material increases in model graph size, posterior storage, sampling
  time, or memory without calling them out. Consider candidate, pollster, time,
  chain, and draw dimensions when judging scaling behavior.
- Prefer vectorized NumPy/PyTensor/xarray operations, but do not sacrifice shape
  clarity, numerical stability, diagnostics, or correctness for small speedups.
- Keep fast tests fast and full sampling coverage meaningful. Do not reduce
  production defaults or diagnostic rigor to shorten CI.

## Tests and documentation

- Every bug fix needs a regression test that fails before the fix. New or
  changed behavior needs focused tests for the normal path and relevant edge
  cases.
- Keep most tests deterministic and fast. Mark tests that actually sample with
  `@pytest.mark.slow`; do not label a non-sampling test slow to hide it from the
  regular suite.
- Use the shared fixtures in `tests/conftest.py` where appropriate. The
  `fast_config` fixture is for plumbing tests, not evidence of convergence or
  forecast quality.
- Test public behavior rather than private implementation details unless the
  internal mathematical structure is itself the contract.
- Update `README.md`, CLI help, docstrings, and `CONTRIBUTING.md` when their
  documented behavior or workflow changes.
- Add a concise entry under `CHANGELOG.md`'s `Unreleased` section for every pull
  request, as required by `CONTRIBUTING.md`. Describe the user or contributor
  impact rather than implementation details; ask the maintainer before omitting
  an entry for purely internal maintenance.

### Documentation requirements

- Document the reason, user-visible semantics, accepted units, defaults,
  exceptions, and important edge cases of a public API. Do not merely restate
  its signature.
- Keep examples executable and deterministic. Use an explicit `today` and
  random seed when an example's output would otherwise change over time.
- Update every affected documentation surface in the same change: Python
  docstrings, README examples, CLI `--help`, changelog, and contribution
  guidance as applicable.
- Explain statistical assumptions and limitations precisely. Distinguish
  posterior estimates from observed data, plurality probability from winning
  office, scenario analysis from identified model parameters, and uncertainty
  intervals from guarantees.
- When changing a CLI or serialization format, document compatibility and
  migration behavior. Include an example of the new invocation or payload when
  it helps users migrate.
- Link to the canonical explanation instead of maintaining multiple long,
  easily divergent copies. Keep short command references local where that is
  more useful to readers.
- Use comments for non-obvious intent, mathematical reasoning, numerical
  constraints, or external invariants. Do not narrate straightforward code.
- Public docstrings and documentation use complete sentences and the same
  terminology as the implementation.

## Public API and compatibility

- Treat names imported by `src/kronikas/__init__.py`, serialized result files,
  CLI flags/output, and documented DataFrame schemas as public interfaces.
- When adding a package-level public symbol, export it from `__init__.py`, add
  it to the alphabetically sorted `__all__`, document it, and test it.
- Do not let a package-level export shadow a submodule name.
- Prefer backward-compatible evolution. If a breaking change is explicitly
  required, update tests, documentation, changelog, and serialization handling
  together.
- Maintain compatibility across all supported Python versions and avoid relying
  on dependency behavior outside the bounds in `pyproject.toml`.

### Deprecation policy

- Do not remove or silently reinterpret a public name, parameter, CLI flag,
  output field, DataFrame column, metric, or persisted field without an
  explicitly approved breaking change.
- Prefer an additive replacement plus a compatibility alias, as used for legacy
  metric names. The alias must delegate to one authoritative implementation and
  have a focused compatibility test.
- When active migration is required, emit a targeted `DeprecationWarning` from
  Python APIs with the replacement and planned removal context. For CLI users,
  print an actionable warning to stderr without corrupting JSON stdout.
- Document a deprecation in the docstring/help text, README where relevant, and
  `CHANGELOG.md`. Do not remove it until a release plan authorizes the matching
  Semantic Versioning impact.
- Do not use deprecation machinery for private helpers. Rename or remove private
  code atomically with its callers and tests.

### Serialization and schema evolution

- Treat `ForecastResult.save()` netCDF files, `ForecastResult.to_dict()`, CLI
  JSON, and documented DataFrame schemas as versioned compatibility surfaces.
- Preserve save/load round trips for metadata, dates, samples, coordinates,
  estimates, scenarios, and election-day grid alignment. Keep rejecting files
  that do not carry kronikas metadata with an actionable error.
- Add an explicit schema/version marker before making a persisted format change
  that cannot be interpreted unambiguously from existing metadata. Loaders
  should validate required fields and either migrate a supported older schema
  or fail clearly; never silently guess.
- Prefer additive JSON changes. Do not rename, remove, change the type of, or
  change units for an existing field without migration and compatibility tests.
- JSON payloads must contain ordinary JSON-compatible values, stable key
  meanings, ISO-8601 dates, and documented percentage/probability units. Do not
  expose NumPy scalars, arbitrary objects, or implementation-only trace data.
- Any schema change needs round-trip tests for the new format and fixtures or
  constructed cases for every older format the loader promises to support.

## Errors, warnings, and user-facing output

Match the existing error taxonomy:

- `TypeError` means a Python caller supplied an unsupported value type.
- `ValueError` means the type is accepted but the value, range, shape, date, or
  combination is invalid.
- `KeyError` means a requested named entity, such as a candidate, is unknown.
- `FileNotFoundError` means a requested input path does not exist.
- `RuntimeError` means an operation is unavailable for the fitted result or
  model state, such as an unidentifiable quantity or missing posterior samples.
- `argparse.ArgumentTypeError` is for command-line token conversion. Let
  `argparse` render usage errors rather than duplicating parser behavior.

Raise before doing expensive sampling when inputs are invalid. Error messages
must identify the offending field/value where safe, state the violated
constraint, and tell the user the accepted form when it is not obvious.

- Warn only when work can continue with a meaningful, documented result. Use a
  specific warning category for programmatically important conditions;
  convergence problems use `ConvergenceWarning`.
- Preserve the distinction between a detected convergence problem and an
  unverified single-chain run. Do not report either as proof of convergence.
- Do not catch unexpected programming errors merely to return a plausible
  result. The CLI currently maps successful execution to status 0, a produced
  forecast with convergence problems to 1, and invalid input/missing files to
  2; preserve and test these meanings.
- Use `--quiet` for scheduled or machine-readable CLI runs, as shown in the
  README. Machine-readable JSON on stdout must remain valid JSON with no
  progress, diagnostics, warnings, or decorative text mixed into it. Send
  errors and warnings to stderr and honor quiet operation.
- Human-readable output should be concise, deterministic in ordering where
  practical, understandable without color, and usable in a plain terminal.
  Include labels and units rather than relying on position or styling alone.
- Treat CLI flag spelling, exit statuses, JSON keys/types, and the distinction
  between stdout and stderr as tested interfaces.

## Security, privacy, and data handling

- Treat CSV, DataFrame, netCDF, JSON, paths, candidate names, and pollster names
  as untrusted input. Validate content and bounds before using them to allocate
  large arrays, construct model coordinates, or write files.
- Do not add pickle or executable deserialization for convenience. Loading a
  persisted result must not evaluate user-controlled code.
- Never log or commit credentials, tokens, environment variables, private
  paths, unpublished polling microdata, respondent-level records, or proprietary
  datasets. Test fixtures should be synthetic and contain no personal data.
- Poll aggregates can still be sensitive before publication. Minimize copied
  data, avoid embedding input rows in errors or snapshots, and sanitize any
  reproducer included in an issue or PR.
- Write only to an explicitly requested destination. Do not overwrite an input
  dataset or existing result implicitly, create missing parent trees without a
  documented contract, or derive output paths from unsanitized labels.
- Keep secrets out of command lines and captured test output. Use established
  CI secret mechanisms only when release work is explicitly authorized.
- Review dependency additions for maintenance, provenance, license, known
  vulnerabilities, and transitive cost. Never work around integrity or TLS
  verification to install a package.

## Git and branch workflow

- Branch from the current `main` unless the maintainer specifies another base.
  Keep one logical change per branch and pull request.
- Before editing, inspect `git status` and distinguish pre-existing work from
  changes made for the task. Never include unrelated files in a commit.
- Keep the branch current with its base using the maintainer's preferred
  workflow. Do not rewrite published history, force-push, merge `main`, or
  rebase a shared branch unless explicitly authorized.
- Resolve conflicts semantically: understand both sides and preserve intended
  behavior. Never choose an entire side mechanically for a conflicted source,
  test, lock, or documentation file.
- Do not mix dependency upgrades, formatting sweeps, generated artifacts, or
  renames with a functional change unless they are required for it.
- Do not commit caches, local environments, editor settings, credentials,
  datasets containing sensitive information, or build outputs. If a recurring
  local artifact is not already ignored, propose an appropriate `.gitignore`
  entry rather than committing it.

### Commit conventions

- Make commits cohesive and reviewable. Each commit should leave the repository
  in a coherent state when practical, with its relevant tests included.
- Write imperative commit subjects that describe the outcome, normally no more
  than 72 characters. Suggested prefixes are `feat:`, `fix:`, `docs:`,
  `test:`, `refactor:`, `perf:`, `build:`, `ci:`, and `chore:`.
- Use a commit body when the motivation, modeling choice, migration, tradeoff,
  or compatibility impact is not obvious from the diff. Explain why the change
  is needed rather than listing edited files.
- Reference an issue in the commit body when useful, but do not use an issue
  number as the whole subject.
- Do not create fixup commits, amend commits, squash history, or sign commits
  on the user's behalf unless asked. Never attribute generated work to a human
  who did not author or approve it.

## Pull requests

Only create, update, or submit a pull request when explicitly requested. When
asked to prepare one, make it independently understandable to a reviewer who
has not read the originating conversation.

### PR title and message

- Use a concise, imperative title describing the user-visible outcome. Follow
  the commit prefixes above when the repository does not provide a different
  PR title convention.
- Start the body with a brief summary of the problem and the resulting behavior.
- Include these sections when applicable:

  - `Summary`: two or three bullets describing the coherent change.
  - `Why`: motivation, issue context, and rejected alternatives when relevant.
  - `Testing`: exact commands run and their results. Explicitly list important
    checks not run and why.
  - `Compatibility and risk`: public API, CLI, serialization, performance,
    statistical, dependency, and migration effects.
  - `Documentation`: files updated, or why no documentation change is needed.
  - `Issue`: `Fixes #<number>` only when merging should close that issue;
    otherwise use `Refs #<number>`.

- Do not claim that a test, benchmark, manual check, or platform validation ran
  unless it actually ran. Do not paste large logs; summarize the result and
  include the actionable failure when a check is incomplete.
- Call out changes to priors, likelihoods, parameterization, numerical
  stability, convergence behavior, or interpretation of output explicitly.
  Include evidence appropriate to the claim rather than relying only on a
  green unit test.

### PR quality and review readiness

A pull request is ready for review only when:

- Its scope matches its title and linked issue, and unrelated changes have been
  removed.
- The diff is small enough to review coherently or is separated into logical,
  well-explained commits.
- New behavior and bug fixes have focused tests, including important boundary
  and failure cases.
- Lint, formatting, typing, and relevant test gates pass. Required CI failures
  are resolved or clearly explained as unrelated and pre-existing.
- Public interfaces, documentation, examples, and the `Unreleased` changelog
  agree with the implementation.
- Backward compatibility and migration consequences are understood and stated.
- No secrets, private data, debug output, commented-out experiments, temporary
  workarounds, or accidental generated files are present.
- Reviewer attention is directed to mathematical assumptions, risky migrations,
  or other decisions that need human judgment.

Keep draft status while required work or known decisions remain. Do not request
review merely to discover failures that can be found locally.

### Responding to review

- Treat each comment as a request to understand, not automatically as an order
  to change code. Reproduce the concern and verify the relevant behavior before
  editing.
- Address the underlying issue consistently across code, tests, and docs. Avoid
  a narrow patch that leaves equivalent cases broken.
- Reply with a concise explanation of the resolution and point to the relevant
  test or rationale. If declining a suggestion, explain the concrete tradeoff
  respectfully and request maintainer direction when needed.
- Do not resolve another person's review thread, dismiss requested changes, or
  mark a draft ready without explicit authorization.
- Re-run checks affected by review edits and update the PR's testing summary if
  the validation evidence changed.

## Issues, dependencies, CI, and releases

- For a bug, capture the observed behavior, minimal reproduction, expected
  behavior, environment, and impact before implementing a fix. Preserve a
  sanitized reproducer as a regression test when feasible.
- For a feature, state the user problem and acceptance criteria. Avoid expanding
  scope into adjacent features without approval.
- Add or upgrade a dependency only when its value outweighs maintenance,
  compatibility, supply-chain, installation-size, and runtime costs. Prefer the
  standard library or an existing dependency for small needs.
- Keep runtime and development dependencies in the correct groups and verify
  lower-bound and supported-Python compatibility when dependency behavior is
  material.
- Inspect the first actionable CI error instead of repeatedly rerunning a
  failing workflow. Treat intermittent stochastic or sampling failures as
  defects to diagnose, not noise to retry until green.
- Workflow permission, publishing, package-signing, PyPI, tag, GitHub Release,
  and `CITATION.cff` changes are release-sensitive. Do not modify or trigger
  them unless the task explicitly includes release work.
- Versioning follows Semantic Versioning and is derived with `setuptools-scm`.
  Do not add a hard-coded package version to source files.

### Design decisions

- Record rationale for substantial changes to model parameterization, priors,
  likelihoods, public APIs, persistence schemas, module boundaries, or supported
  platforms. Include alternatives, compatibility impact, and validation plan.
- Use the issue or PR description for a contained decision. Propose a short
  repository design note only when the decision is long-lived, cross-cutting,
  or too detailed to recover reliably from a diff. Do not create process-heavy
  records for routine fixes.
- Keep the implementation, tests, documentation, and recorded decision in sync
  when review changes the chosen design.

## Instruction scope

- This root `AGENTS.md` applies to the entire repository. A closer
  `AGENTS.md` may add or override rules only for files under its own directory.
- Add a nested instruction file only when a subtree has a genuinely different
  toolchain, generated-code policy, test workflow, or domain contract. Do not
  duplicate the root file merely for visibility.
- Explicit maintainer/user instructions take precedence over repository files.
  Among repository instruction files, follow the closest applicable file. Call
  out a material ambiguity before making a change that would be hard to reverse.
- Rules from a directory do not automatically apply to sibling directories.
  When a change spans scopes, satisfy each applicable file for the files it
  governs.

## Agent handoff

The final handoff must be self-contained and include:

- The outcome and important behavior changes, followed by the main files
  changed.
- Exact checks executed and whether each passed. State why any expected check
  was skipped.
- Compatibility, migration, numerical, performance, security, or data risks
  that remain relevant.
- Assumptions made, unresolved decisions, and focused follow-up work, if any.
- Existing unrelated worktree changes that were deliberately preserved when
  they could otherwise be confused with the agent's edits.

Do not dump a chronological activity log or large command output. Lead with the
result, link to relevant files and lines, and distinguish completed work from
recommendations that were not implemented.

## Completion checklist

Before handing off a code change:

1. Review `git diff` and `git status`; confirm only intended files changed.
2. Run the most relevant focused tests while iterating.
3. Run `make check` and `make typecheck` for Python changes.
4. Run `make test-fast` while iterating as appropriate, then run `make test`
   before every pull request.
5. Confirm documentation is current and the pull request has an `Unreleased`
   changelog entry, unless the maintainer explicitly approved omitting it.
6. Report what changed, which checks ran, and any checks not run or remaining
   risks. Do not claim success for a command that was not executed.
7. If a PR was requested, confirm its title and body accurately describe the
   final diff, issue linkage, compatibility impact, and validation evidence.
