[![PyPI version](https://img.shields.io/pypi/v/kronikas.svg)](https://pypi.org/project/kronikas/)
[![Python versions](https://img.shields.io/pypi/pyversions/kronikas.svg)](https://pypi.org/project/kronikas/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/vtisza/kronikas/actions/workflows/ci.yml/badge.svg)](https://github.com/vtisza/kronikas/actions)
[![Downloads](https://img.shields.io/pypi/dm/kronikas.svg)](https://pypi.org/project/kronikas/)
[![DOI](https://zenodo.org/badge/1188801535.svg)](https://doi.org/10.5281/zenodo.19163741)

# kronikas

**Principled election forecasting from opinion polls, powered by hierarchical Bayesian inference.**

Most poll aggregators reduce rich, noisy data into a single point estimate and call it a day. *kronikas* does the opposite: it builds a full generative model of how public opinion evolves, learns each pollster's systematic biases, and propagates every source of uncertainty into honest probability distributions. The result is not just "Party A is at 42 %," but "Party A wins with 73 % probability, and here is the full distribution behind that number." If you are a political scientist, data journalist, election analyst, or anyone who needs defensible, reproducible forecasts from polling data, kronikas gives you a statistically rigorous engine you can trust, and customize, in a single `pip install`.

## Why kronikas?

- 🎯 **Probability over point estimates.** Full posterior distributions and win probabilities, not just a number. Every forecast comes with calibrated uncertainty so you know exactly what you don't know.
- 🔧 **Systematic bias correction.** Automatically detects and corrects per-pollster house effects, cleanly separating genuine opinion shifts from firm-specific methodological bias.
- 📐 **Structurally sound.** Dirichlet observations and softmax constraints guarantee that predicted vote shares are always non-negative and sum to exactly 100 %. No ad-hoc normalization needed.
- ⚙️ **Highly configurable.** Flexible priors, per-pollster overrides, adjustable time grids, correlated random walks, and an escape hatch to any `pymc.sample()` kwarg. Shape the model to your domain knowledge.

## Used in production

kronikas is the forecasting engine behind **[százkilencvenkilenc.hu](https://www.szazkilencvenkilenc.hu/)**, where it powers live Hungarian election forecasts and tracks real-world polling shifts as they happen.

## Installation

The fastest way to get started:

```bash
pip install kronikas
```

For development, with [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync --group dev
```

Or with pip:

```bash
pip install -e ".[dev]"
```

## Quick start

### 1. Prepare a CSV

Each row is one poll.  Required columns: **date**, **pollster**,
**sample_size**, plus one column per candidate with their support value
(any scale; values are normalised to 100 %).

```csv
date,pollster,sample_size,Alice,Bob,Carol
2024-01-15,PollCo,1000,45,40,10
2024-02-01,SurveyInc,1200,44,42,11
2024-02-15,PollCo,800,46,39,12
2024-03-01,SurveyInc,1500,43,43,10
2024-03-15,PollCo,1000,47,38,13
```

### 2. Run a forecast

```python
from kronikas import ElectionForecast, ModelConfig

forecast = ElectionForecast(
    polls_csv="polls.csv",
    election_date="2024-11-05",
    # today defaults to date.today(); override for reproducibility:
    today="2024-03-20",
)
result = forecast.run()
print(result.summary())
```

### 3. Inspect the output

`result` is a `ForecastResult` with:

```python
# Per-candidate estimates for today and election day
for est in result.today_estimates:
    print(f"{est.name}: {est.mean:.1f}% (90% CI: {est.ci_lower:.1f}%-{est.ci_upper:.1f}%)")

# Plurality probabilities (election day): P(this candidate polls highest)
for name, prob in result.win_probabilities.items():
    print(f"{name}: {prob:.1%}")

# Probability of clearing a vote-share threshold (e.g. an electoral threshold)
for name, prob in result.threshold_probabilities(5.0).items():
    print(f"{name} reaches 5%: {prob:.1%}")

# Head-to-head comparison
print(result.lead_probability("Alice", "Bob"))

# Sampler convergence diagnostics
print(result.diagnostics.summary())

# Full ArviZ InferenceData for custom analysis
result.trace
```

> **What `win_probabilities` measures.** It is the probability of holding a
> **plurality of the vote share** — of polling higher than every other
> candidate on election day. Under electoral systems that are not simple
> national plurality (runoffs, district seat allocation, electoral colleges,
> coalition formation), that is not the probability of winning office. Treat it
> as a vote-share statistic and build any seat or office model on top of
> `party_forecast_dataframe()`.

### 4. Party forecast as a DataFrame

`party_forecast_dataframe()` returns all posterior draws as a
`pandas.DataFrame`: one row per draw, one column per party named after
the party.  Values are vote shares in percentage points.

```python
# Posterior draws at the reference date (today)
df_today = result.party_forecast_dataframe(day="today")

# Posterior draws at election day
df_election = result.party_forecast_dataframe(day="election_day")

# df.columns == ["Alice", "Bob", "Carol"]
# df.shape   == (num_draws * num_chains, num_candidates)
# Each row sums to 100.0 (percentage points)
```

Warmup iterations are never included. Only the post-tuning posterior
draws that PyMC keeps after sampling.

Use these DataFrames for custom downstream analysis:

```python
import matplotlib.pyplot as plt

# Histogram of Alice's election-day support
df_election["Alice"].plot.hist(bins=40, edgecolor="white")
plt.xlabel("Vote share (%)")
plt.title("Alice – election-day forecast")
plt.show()

# Correlations between parties
print(df_election.corr().round(2))

# Probability that Alice leads Bob on election day
p = (df_election["Alice"] > df_election["Bob"]).mean()
print(f"P(Alice > Bob) = {p:.1%}")
```

### 5. House effects as a DataFrame

`house_effects_dataframe()` returns all posterior draws of the per-pollster
house effects as a `pandas.DataFrame`.  Each value is the percentage-point
deviation from a neutral equal-support baseline (all candidates at 1/*K*)
produced by that pollster's bias term.  Positive values mean the pollster
over-estimates a candidate; negative values mean under-estimation.  Within
each draw and pollster, values across all candidates sum to zero.

The DataFrame uses a two-level column `MultiIndex`: the outer level is the
pollster name and the inner level is the candidate name.

```python
df_he = result.house_effects_dataframe()
# df_he.columns is a MultiIndex with levels ["pollster", "candidate"]
# df_he.shape == (num_draws * num_chains, num_pollsters * num_candidates)

# Inspect PollCo's bias for all candidates
print(df_he["PollCo"].mean())

# Plot the posterior distribution of PollCo's Alice bias
df_he["PollCo"]["Alice"].plot.hist(bins=40, edgecolor="white")
```

A `RuntimeError` is raised when the model was run with a single pollster
(house effects are not identifiable in that case).

### 6. Save a forecast and reload it later

Sampling takes minutes, so a run worth keeping should not have to be repeated.
`save()` writes the full posterior to netCDF; `load()` rebuilds an equivalent
`ForecastResult`.

```python
result.save("forecast-2024-03-20.nc")

from kronikas import ForecastResult
restored = ForecastResult.load("forecast-2024-03-20.nc")
```

For publishing just the numbers, `to_dict()` returns a small JSON-serialisable
summary with no posterior draws attached:

```python
import json

payload = result.to_dict(thresholds=[5.0])
print(json.dumps(payload, indent=2))
```

### 7. Customise priors and sampler

```python
config = ModelConfig(
    # --- Sampler ---
    num_tune=2000,                 # warmup iterations per chain
    num_draws=2000,                # posterior draws per chain
    num_chains=4,                  # independent MCMC chains
    cores=4,                       # CPU cores for parallel sampling
    target_accept=0.99,            # higher = fewer divergences
    init_method="adapt_full",      # better for correlated posteriors
    progressbar=True,              # show progress bar

    # --- Time grid ---
    time_step_days=3,              # finer time resolution

    # --- Priors (logit scale) ---
    sigma_walk_prior=0.03,         # smoother trend, per `walk_reference_days`
    sigma_house_prior=0.2,         # tighter house-effect prior
    initial_sigma=0.3,             # tighter prior on initial support
    kappa_log_sigma=0.3,           # tighter poll-precision prior

    # --- Escape hatch for any pymc.sample() kwarg ---
    sampler_kwargs={"nuts_sampler": "nutpie"},
)

result = ElectionForecast(
    polls_csv="polls.csv",
    election_date="2024-11-05",
    config=config,
).run()
```

## CSV format

| Column | Type | Description |
|---|---|---|
| `date` | date string | Poll date (any format `pandas.to_datetime` can parse, or specify `date_format`). |
| `pollster` | string | Polling firm identifier. |
| `sample_size` | positive int | Number of respondents. |
| *candidate columns* | numeric | Raw support for each candidate (normalised internally). |

Column names for `date`, `pollster`, and `sample_size` can be overridden:

```python
ElectionForecast(
    polls_csv="polls.csv",
    election_date="2024-11-05",
    date_column="poll_date",
    pollster_column="firm",
    sample_size_column="n",
    candidate_columns=["Dem", "Rep"],   # explicit subset
    date_format="%d/%m/%Y",             # non-ISO dates
    decimal=",",                        # European-style decimal separator
)
```

### European-style CSVs

Some locales write numbers with a comma as the decimal point (e.g. `45,3`
instead of `45.3`).  Use the `decimal` parameter to tell the reader which
character to treat as the decimal separator:

```python
ElectionForecast(
    polls_csv="polls_eu.csv",
    election_date="2024-11-05",
    decimal=",",
)
```

`decimal` defaults to `"."` and accepts any single character.

### Polls already in memory

When the data comes from a database or an upstream cleaning step, skip the CSV
round-trip:

```python
from kronikas import ElectionForecast, polls_from_dataframe

forecast = ElectionForecast.from_dataframe(polls_frame, election_date="2024-11-05")
result = forecast.run()

# Or validate/normalise a frame on its own:
poll_data = polls_from_dataframe(polls_frame)
```

`from_dataframe` accepts the same column-name overrides as the constructor, and
never mutates the frame you pass it.

## Command line

Installing the package puts a `kronikas` executable on your path, so a
scheduled job can produce machine-readable output without a Python wrapper:

```bash
# Human-readable summary
kronikas forecast polls.csv --election-date 2024-11-05

# JSON to a file, with threshold probabilities, quiet enough for cron
kronikas forecast polls.csv \
    --election-date 2024-11-05 \
    --threshold 5 --threshold 10 \
    --json forecast.json \
    --save-trace forecast.nc \
    --quiet

# Score the model against a past election
kronikas backtest polls.csv \
    --election-date 2024-11-05 \
    --as-of 2024-08-01 --as-of 2024-10-01 \
    --actual "Alice=48.2,Bob=47.1,Carol=4.7"
```

`kronikas forecast` exits non-zero when the sampler reports a convergence
problem, so a scheduled run fails loudly instead of publishing bad numbers.
Run `kronikas forecast --help` for the full option list, including sampler
settings and CSV schema overrides.

## Backtesting

A forecast that has never been scored is an assertion, not a measurement.
`backtest()` replays a campaign: for each *as-of* date it discards every later
poll, refits, and records what the model would have said about election day
knowing only what was available then.

```python
from datetime import date
from kronikas import backtest

report = backtest(
    "polls.csv",
    election_date=date(2024, 11, 5),
    as_of_dates=[date(2024, 8, 1), date(2024, 9, 1), date(2024, 10, 1)],
    actual={"Alice": 48.2, "Bob": 47.1, "Carol": 4.7},
)

print(report.summary())
report.to_dataframe()   # tidy: one row per (as-of date, candidate)
report.metrics()        # MAE, RMSE, 90% coverage, per-candidate bias
```

`actual` accepts any scale — percentages, fractions, or raw vote counts — and
is normalised the same way polls are.

Two notes on reading the output. **Coverage** is the calibration check that
matters most: 90 % credible intervals should contain the truth about 90 % of
the time, and much less means the model is overconfident. **Bias is reported
per candidate, never pooled** — forecast and actual shares both sum to 100, so
signed errors cancel exactly across candidates and a pooled mean would be
identically zero.

Each as-of date costs one full MCMC fit, so lower `num_draws` for exploratory
runs.

## Convergence diagnostics

Every result carries the headline sampler statistics, and a
`ConvergenceWarning` is raised when something looks wrong — so a script cannot
quietly go on to print confident-looking numbers from chains that never mixed.

```python
result = forecast.run()

print(result.diagnostics.summary())
result.diagnostics.converged      # False if R-hat, ESS, or divergences look bad
result.diagnostics.issues         # problems detected
result.diagnostics.notes          # caveats, e.g. R-hat needs >= 2 chains
```

Thresholds are configurable via `ModelConfig(r_hat_threshold=..., ess_threshold=...)`.

A single-chain run reports `converged=True` with a note, not a failure: one
chain leaves convergence *unverified* rather than demonstrating a problem. Use
`num_chains >= 2` to actually check it.

For model comparison, set `compute_log_likelihood=True` to populate the trace's
`log_likelihood` group for `arviz.loo` / `arviz.waic`:

```python
config = ModelConfig(compute_log_likelihood=True)
result = ElectionForecast("polls.csv", "2024-11-05", config=config).run()

import arviz as az
print(az.loo(result.trace))
```

## Model

The model has three components:

1. **Latent support (logistic-normal random walk).**
   Candidate proportions are parameterised as K-1 log-ratios relative to a
   reference candidate.  These evolve as a Gaussian random walk on a
   discretised time grid (default: weekly steps).  Softmax maps the
   log-ratios back to the probability simplex, guaranteeing non-negative
   shares that sum to 1.

   The grid is anchored **backwards from election day**, so its final node
   falls exactly on the election date and the election-day forecast is not
   contaminated by an extra partial step of drift.  The grid therefore starts
   on or just before the first poll.

   The random-walk prior `sigma_walk_prior` is expressed per
   `walk_reference_days` (7 by default) rather than per time step, so changing
   `time_step_days` changes the resolution of the trend without also changing
   how volatile the prior says it is.

2. **House effects.**
   Each pollster gets a bias term in log-ratio space, drawn from a
   `Normal(mu_house, sigma_house)` prior. House effects are strictly
   zero-sum constrained across all K candidates for each pollster, ensuring
   predictions correctly sum back to 100% without an overall scalar shift.
   The mean `mu_house` defaults to zero (no assumed direction of bias) but
   can be set per pollster and per candidate to encode prior knowledge about a
   specific pollster's lean.
   When only a single pollster is present, house effects are omitted (not
   identifiable).  Per-pollster prior overrides can replace the hierarchical
   `sigma_house` with a fixed SD for individual pollsters (see
   [Per-pollster priors](#per-pollster-priors)).

3. **Dirichlet observations.**
   Each poll is modelled as
   `Dirichlet(kappa_scale * sample_size * latent_proportions)`.
   The learnt `kappa_scale` absorbs overdispersion beyond pure multinomial
   sampling (design effects, non-response, etc.).  When per-pollster
   `kappa_log_sigma` overrides are specified, each pollster receives its own
   `kappa_scale`.

Non-centred parameterisation is used for the random walk to avoid
divergences.

## Configuration reference

All fields on `ModelConfig` with their defaults:

**Sampler settings**

| Parameter | Default | Description |
|---|---|---|
| `num_tune` | 1500 | Warmup (tuning) iterations per chain |
| `num_draws` | 1000 | Posterior draws per chain (total samples = draws × chains) |
| `num_chains` | 2 | Independent MCMC chains (≥ 2 recommended for R-hat) |
| `cores` | None | CPU cores for parallel sampling (None = auto-detect) |
| `target_accept` | 0.95 | NUTS target acceptance rate (0.90–0.99) |
| `random_seed` | 42 | Reproducibility seed |
| `init_method` | `"jitter+adapt_diag"` | NUTS initialisation (`"adapt_diag"`, `"adapt_full"`, …) |
| `progressbar` | True | Show progress bar during sampling |
| `compute_log_likelihood` | False | Store pointwise log-likelihood for `arviz.loo` / `waic` |
| `sampler_kwargs` | `{}` | Extra kwargs forwarded to `pymc.sample()` |

**Time discretisation**

| Parameter | Default | Description |
|---|---|---|
| `time_step_days` | 7 | Time-grid granularity in days (grid ends exactly on election day) |

**Priors (logit / log-ratio scale)**

| Parameter | Default | Description |
|---|---|---|
| `sigma_walk_prior` | 0.05 | HalfNormal scale for random-walk SD, per `walk_reference_days` (~1 pp/week at 50 %) |
| `walk_reference_days` | 7 | Calendar window `sigma_walk_prior` refers to; keeps implied volatility grid-invariant |
| `sigma_house_prior` | 0.3 | HalfNormal scale for house-effect SD (~5 pp max bias) |
| `initial_sigma` | 0.5 | Normal SD for initial latent support |
| `kappa_log_sigma` | 0.5 | SD of log-normal prior on poll precision scaling factor |
| `r_hat_threshold` | 1.01 | R-hat above this triggers a `ConvergenceWarning` |
| `ess_threshold` | 400.0 | Minimum bulk ESS below which a `ConvergenceWarning` is raised |
| `correlated_walk` | False | Enables LKJ-correlated random walk innovations rather than independent ones |
| `lkj_eta` | 2.0 | Shape parameter for LKJ matrix prior (used when `correlated_walk=True`) |

**Per-pollster overrides**

| Parameter | Default | Description |
|---|---|---|
| `pollster_priors` | `{}` | Dict mapping pollster name to `PollsterPrior` (see below) |

## Per-pollster priors

Use `PollsterPrior` to set different priors for individual pollsters.  This
is useful when you have external knowledge about a pollster's reliability or
known biases.

```python
from kronikas import ElectionForecast, ModelConfig, PollsterPrior

config = ModelConfig(
    pollster_priors={
        # PollCo has a known small bias, constrain its house effect
        "PollCo": PollsterPrior(sigma_house=0.1),
        # SurveyInc uses an online panel, allow more overdispersion
        "SurveyInc": PollsterPrior(kappa_log_sigma=1.0),
    },
)

result = ElectionForecast(
    polls_csv="polls.csv",
    election_date="2024-11-05",
    config=config,
).run()
```

Each `PollsterPrior` field is optional; `None` (the default) inherits the
global value from `ModelConfig`:

| Field | Default | Description |
|---|---|---|
| `sigma_house` | None (uses `sigma_house_prior`) | Fixed house-effect SD for this pollster in logit space. Lower = more trusted. |
| `kappa_log_sigma` | None (uses `kappa_log_sigma`) | SD of log-normal prior on this pollster's precision scaling. Higher = allow more overdispersion. |
| `mu_house` | None (all zeros) | Dict mapping candidate name to expected bias in **percentage points**. Positive = over-estimates, negative = under-estimates. Omitted candidates default to 0 pp. Converted to logit space relative to that candidate's own support level, so the bias must not push it outside (0 %, 100 %). |

**How it works:**

- **House effects:** Pollsters with a `sigma_house` override use that value
  directly as the SD for their house-effect prior, bypassing the
  hierarchical `sigma_house` parameter.  Pollsters without an override
  continue to share the learnt hierarchical `sigma_house`.  If *all*
  pollsters have overrides, the hierarchical `sigma_house` is omitted
  entirely.
- **Kappa (precision):** When any pollster has a `kappa_log_sigma` override,
  the model switches from a single shared `kappa_log` to per-pollster
  `kappa_log` values.  Pollsters without overrides use the global
  `kappa_log_sigma` as their prior SD.
- **Unknown names:** Pollster names in `pollster_priors` that don't match
  any pollster in the data trigger a warning and are ignored.

### Setting prior means for pollster–party bias

Use `mu_house` when you have external knowledge that a pollster
systematically leans toward or against a specific candidate.  Values are
in **percentage points**. Specify the expected bias directly.  You only
need to list the candidates you want to set; the rest default to 0 pp.

```python
from kronikas import ElectionForecast, ModelConfig, PollsterPrior

config = ModelConfig(
    pollster_priors={
        # PollCo is believed to over-estimate Alice by 3 pp
        "PollCo": PollsterPrior(mu_house={"Alice": 3}),

        # SurveyInc tends to under-estimate Bob by 4 pp; also allow a wider SD
        "SurveyInc": PollsterPrior(
            mu_house={"Bob": -4},
            sigma_house=0.4,
        ),

        # YouGov: set means for two candidates, keep default sigma
        "YouGov": PollsterPrior(mu_house={"Alice": 2, "Bob": -2}),
    },
)

result = ElectionForecast(
    polls_csv="polls.csv",
    election_date="2024-11-05",
    config=config,
).run()
```

Pollsters without a `mu_house` entry keep the default zero mean; only
the pollsters you explicitly configure are affected.  Values are converted
to logit space internally using a 50 % support baseline.

## Lower-level API

For more control, use the building blocks directly:

```python
from kronikas import ModelConfig, load_polls
from kronikas.model import build_model, run_inference, extract_results
from datetime import date

poll_data = load_polls("polls.csv")
config = ModelConfig(num_draws=500)

model, metadata = build_model(poll_data, date(2024, 11, 5), date.today(), config)
trace = run_inference(model, config)
result = extract_results(trace, poll_data, metadata)

# Direct access to ArviZ trace
import arviz as az
az.summary(result.trace)
az.plot_trace(result.trace, var_names=["sigma_walk", "kappa_log"])
```

## Contributing

We welcome contributions of all kinds: bug reports, feature ideas, documentation improvements, and code. Whether you're fixing a typo or building a new feature, we'd love to have you involved.

👉 **See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, coding guidelines, and how to submit a pull request.**

## Citation

[![DOI](https://zenodo.org/badge/1188801535.svg)](https://doi.org/10.5281/zenodo.19163741)

If you use kronikas in your research, please cite it:

```bibtex
@software{Tisza_kronikas_2026,
  author = {Tisza, Viktor},
  title = {kronikas},
  month = {3},
  year = {2026},
  publisher = {Zenodo},
  version = {0.1.0},
  doi = {10.5281/zenodo.19163741},
  url = {https://github.com/vtisza/kronikas}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
