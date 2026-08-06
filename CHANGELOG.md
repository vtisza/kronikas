# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

- **Backtesting** (`kronikas.backtest`): refit as of past dates using only the
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
- `ModelConfig.compute_log_likelihood` to populate the trace's
  `log_likelihood` group for `arviz.loo` / `arviz.waic`.
- `ModelConfig` now validates its numeric fields on construction.
- `py.typed` marker, packaged in the wheel, so downstream type checkers can see
  the package's annotations.

### Changed

- `win_probabilities` is documented as the probability of holding a **plurality
  of the vote share**, not of winning office under a non-plurality electoral
  system. The summary heading now reads "Plurality probabilities (election
  day)".
- `ForecastResult.trace` is annotated as `arviz.InferenceData`, and the sample
  arrays as `np.ndarray | None`, instead of `Any` and a mistyped default.
- API documentation is phrased in party-system-neutral terms.
- CI now runs a Python 3.10/3.11/3.12 matrix plus macOS, type checks with mypy,
  verifies the built wheel ships `py.typed`, and runs the **full** test suite
  including MCMC sampling — which `make test-fast` deselects entirely, leaving
  the sampling path previously untested on every push.

## [0.1.0]

Initial release.
