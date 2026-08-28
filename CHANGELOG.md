# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A guided, no-code workflow at `.claude/skills/election-forecast/`: a
  plain-language `forecast.yaml` settings file (election, poll CSV, opinion
  volatility, per-pollster leans and trust, industry-wide polling error), a
  runner with a `--check` mode that validates settings and data before
  sampling, and a self-contained HTML report covering win probabilities,
  forecast ranges, the trend with its polls, house effects, the break-even
  polling error and a plain-language model-health verdict. Usable directly
  from a terminal, or as a skill an AI assistant follows to conduct the whole
  interview.

### Fixed in the current review

- Forecast runs now exclude polls after `today` (or election day, whichever is
  earlier); lower-level model construction rejects post-election observations.
- House effects are zero-sum across both candidates and pollsters in an
  identifiable `ZeroSumNormal` parameterisation, even without shared bias.
- Unknown or zero-only shared-bias keys no longer activate a different model.
- Shared-bias spread is jointly calibrated in vote-share percentage points,
  and saved scenario results reload with their adjusted samples intact.
- Shared-bias break-even now tests whether the leader beats every candidate,
  not merely the original runner-up.
- Initial support comes from the earliest occupied time node and the largest
  early candidate is used as the internal log-ratio reference.
- Poll, prior, sampler, and actual-result inputs reject non-finite or otherwise
  invalid values; same-date poll ordering is stable.
- The Dirichlet concentration floor is smooth, and an optional
  `undecided_column` scales effective sample size by the decided fraction.
- Supported Python is explicitly constrained to the CI-tested 3.10–3.12 range,
  and duplicate citation-release automation was removed.

### Changed in the current review

- Backtests report CRPS and call their single-election interval statistic a
  hit rate rather than claiming it establishes calibration (`coverage_90`
  remains as a compatibility alias). The old
  `kronikas.backtest` module path was renamed to `kronikas.backtesting`; import
  `backtest` from the package root or the new module path.

### Fixed

- **The time grid now ends exactly on election day.** It is anchored backwards
  from the election rather than forwards from the first poll. Previously the
  final node could land up to `time_step_days - 1` days *after* the election,
  so `election_day_estimates` described the wrong date and its credible
  interval carried a spurious extra step of random-walk drift. The grid now
  starts on or just before the first poll instead.
- **`sigma_walk_prior` is no longer sensitive to the grid resolution.** The
  prior is expressed per `walk_reference_days` (7 by default) and rescaled to
  the actual step length, so changing `time_step_days` changes the resolution
  of the trend without also changing how volatile the prior says it is.
  Previously, halving the step size implicitly doubled the weekly prior
  variance. With the shipped defaults the per-step scale is unchanged.
- **`mu_house` is converted relative to each candidate's own support level.**
  A fixed 50 % baseline understated small-party biases roughly four-fold: a
  +3 pp bias is a logit shift of 0.12 at 50 % support but 0.50 at 5 %. The
  conversion is unchanged for a candidate polling at 50 %.
- Polls dated after `election_date`, and a `today` outside the modelled window,
  now raise a warning instead of being silently clamped into the grid.
- `uv sync --group dev`, as documented in the README and CONTRIBUTING, now
  works: a PEP 735 `[dependency-groups]` entry was added alongside the existing
  `dev` extra, which continues to serve `pip install -e ".[dev]"`.

### Added

- **Backtesting** (`kronikas.backtesting`): refit as of past dates using only the
  polls available then, and score election-day forecasts for accuracy (MAE,
  RMSE) and calibration (90 % interval coverage). Bias is reported per
  candidate, since pooled signed error is identically zero for compositional
  data.
- **Command-line interface**: `kronikas forecast` and `kronikas backtest`, with
  JSON output for scheduled jobs. `forecast` exits non-zero on a convergence
  problem.
- **Convergence diagnostics** (`kronikas.diagnostics`): every `ForecastResult`
  carries `SamplingDiagnostics` (max R-hat, min bulk/tail ESS, divergences),
  and a `ConvergenceWarning` is raised when the posterior looks unconverged.
  A single-chain run is reported as unverified rather than failed.
- `ForecastResult.threshold_probabilities(threshold)`: probability that each
  candidate's vote share reaches a given level — electoral thresholds, or any
  other target.
- `ForecastResult.lead_probability(candidate, opponent)` for head-to-head
  comparisons.
- `ForecastResult.save()` / `ForecastResult.load()`: persist and restore a full
  forecast as netCDF, and `to_dict()` for a small JSON-serialisable summary.
- `polls_from_dataframe()` and `ElectionForecast.from_dataframe()`: use polls
  already in memory without a CSV round-trip.
- `PollData.up_to()`, `PollData.poll_dates`, `PollData.last_poll_date`.
- **Shared polling error** (`SharedBiasPrior`, `ModelConfig.shared_bias`): an
  explicit prior on the bias every pollster shares, which is not identifiable
  from a single election's polls and which the model otherwise passes into the
  forecast one-for-one while *narrowing* the interval, because it reads
  pollster agreement as precision. Both the centre and the spread are settable
  per candidate — the error is not assumed symmetric about zero. Defaults to
  `None`, reproducing earlier behavior apart from the now-unconditional
  identification constraint on house effects. The scale must be supplied
  rather than learned: a
  hierarchical scale collapses to zero for the same reason.
- `ForecastResult.assume_shared_bias()` and `shared_bias_breakeven()`: report
  what an industry-wide error would cost, with no refit, plus the smallest
  uniform error that erases the leader's advantage. A `--shared-bias` flag
  exposes both on the CLI.
- `ModelConfig.compute_log_likelihood` to populate the trace's
  `log_likelihood` group for `arviz.loo` / `arviz.waic`.
- `ModelConfig` now validates its numeric fields on construction.
- `py.typed` marker, packaged in the wheel, so downstream type checkers can see
  the package's annotations.
- Test coverage for `latent_trend_dataframe()`, which previously had none, and
  for the CLI's CSV-schema pass-through (renamed columns, `--date-format`,
  `--decimal`, `--candidate-column`, `--seed`).

### Changed

- `win_probabilities` is documented as the probability of holding a **plurality
  of the vote share**, not of winning office under a non-plurality electoral
  system. The summary heading now reads "Plurality probabilities (election
  day)".
- `ForecastResult.trace` is annotated as `arviz.InferenceData`, and the sample
  arrays as `np.ndarray | None`, instead of `Any` and a mistyped default.
- API documentation is phrased in party-system-neutral terms.
- `ModelConfig.pollster_priors` and `shared_bias` are now documented in the
  class docstring, and `__all__` is sorted and asserted to stay that way.
- The backtesting module is named `kronikas.backtesting`, so it no longer
  collides with the `backtest()` function it exports. Previously
  `from .backtest import backtest` rebound `kronikas.backtest` from the module
  to the function, so `import kronikas.backtest as m` yielded the function and
  `pydoc.locate("kronikas.backtest")` — the resolution Sphinx, pdoc and
  mkdocstrings use — returned the wrong object. The public API is unchanged:
  `from kronikas import backtest` still gives the function. Two guard tests now
  assert that no submodule is shadowed by a package-level export and that every
  submodule resolves for documentation tooling.
- CI now runs a Python 3.10/3.11/3.12 matrix plus macOS, type checks with mypy,
  verifies the built wheel ships `py.typed`, and runs the **full** test suite
  including MCMC sampling — which `make test-fast` deselects entirely, leaving
  the sampling path previously untested on every push.

## [0.1.0]

Initial release.
