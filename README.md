# kronikas

[![PyPI version](https://img.shields.io/pypi/v/kronikas.svg)](https://pypi.org/project/kronikas/)
[![Python versions](https://img.shields.io/pypi/pyversions/kronikas.svg)](https://pypi.org/project/kronikas/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/vtisza/kronikas/actions/workflows/tests.yml/badge.svg)](https://github.com/vtisza/kronikas/actions)
[![Downloads](https://img.shields.io/pypi/dm/kronikas.svg)](https://pypi.org/project/kronikas/)
[![DOI](https://zenodo.org/badge/1188801535.svg)](https://doi.org/10.5281/zenodo.19163741)

Hierarchical Bayesian election forecasting from opinion polls.

## Overview

**kronikas** reads a CSV of opinion polls and fits a hierarchical Bayesian
model to produce probabilistic forecasts for both the current date and election
day.  The model includes:

- **Logistic-normal random walk** for latent candidate support in log-ratio
  space, mapped to the probability simplex via softmax.
- **Dirichlet observation model** that structurally guarantees predicted
  shares sum to 100 %, with concentration tied to sample size.
- **House effects** to capture systematic pollster biases (automatically
  omitted when only one pollster is present).

Inference is performed with the NUTS sampler via [PyMC](https://www.pymc.io/).

## Installation

With [uv](https://docs.astral.sh/uv/) (recommended):

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
(any scale -- values are normalised to 100 %).

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

# Win probabilities (election day)
for name, prob in result.win_probabilities.items():
    print(f"{name}: {prob:.1%}")

# Full ArviZ InferenceData for custom analysis
result.trace
```

### 4. Party forecast as a DataFrame

`party_forecast_dataframe()` returns all posterior draws as a
`pandas.DataFrame` — one row per draw, one column per party named after
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

Warmup iterations are never included — only the post-tuning posterior
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

The DataFrame uses a two-level column `MultiIndex` — the outer level is the
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

### 6. Customise priors and sampler

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
    sigma_walk_prior=0.03,         # smoother trend
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

## Model

The model has three components:

1. **Latent support (logistic-normal random walk).**
   Candidate proportions are parameterised as K-1 log-ratios relative to a
   reference candidate.  These evolve as a Gaussian random walk on a
   discretised time grid (default: weekly steps).  Softmax maps the
   log-ratios back to the probability simplex, guaranteeing non-negative
   shares that sum to 1.

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
| `num_draws` | 1000 | Posterior draws per chain (total samples = draws x chains) |
| `num_chains` | 2 | Independent MCMC chains (>= 2 recommended for R-hat) |
| `cores` | None | CPU cores for parallel sampling (None = auto-detect) |
| `target_accept` | 0.95 | NUTS target acceptance rate (0.90-0.99) |
| `random_seed` | 42 | Reproducibility seed |
| `init_method` | `"jitter+adapt_diag"` | NUTS initialisation (`"adapt_diag"`, `"adapt_full"`, ...) |
| `progressbar` | True | Show progress bar during sampling |
| `sampler_kwargs` | `{}` | Extra kwargs forwarded to `pymc.sample()` |

**Time discretisation**

| Parameter | Default | Description |
|---|---|---|
| `time_step_days` | 7 | Time-grid granularity in days |

**Priors (logit / log-ratio scale)**

| Parameter | Default | Description |
|---|---|---|
| `sigma_walk_prior` | 0.05 | HalfNormal scale for random-walk SD (~1 pp/week at 50 %) |
| `sigma_house_prior` | 0.3 | HalfNormal scale for house-effect SD (~5 pp max bias) |
| `initial_sigma` | 0.5 | Normal SD for initial latent support |
| `kappa_log_sigma` | 0.5 | SD of log-normal prior on poll precision scaling factor |
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
        # PollCo has a known small bias — constrain its house effect
        "PollCo": PollsterPrior(sigma_house=0.1),
        # SurveyInc uses an online panel — allow more overdispersion
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
| `mu_house` | None (all zeros) | Dict mapping candidate name to expected bias in **percentage points**. Positive = over-estimates, negative = under-estimates. Omitted candidates default to 0 pp. Must be in (-50, 50). |

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
in **percentage points** — specify the expected bias directly.  You only
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

Pollsters without a `mu_house` entry keep the default zero mean — only
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

## Running tests

```bash
# Fast tests only (no MCMC sampling)
uv run pytest tests/ -m "not slow"

# Full suite including inference tests
uv run pytest tests/
```

## License

Apache 2.0
