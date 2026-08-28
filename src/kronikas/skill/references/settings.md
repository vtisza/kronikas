# Settings reference

Every setting in `forecast.yaml`, what it means in plain language, and what it
does to the model underneath. Start from
`assets/forecast.template.yaml`.

**Paths are relative to the settings file.** If `forecast.yaml` and
`polls.csv` sit side by side, `file: polls.csv` is right, whatever directory
you run the command from.

The file is YAML: `setting: value`, two-space indentation for nested blocks,
`#` starts a comment. A misspelled setting name is an error, not a silent
default — the check step names it and stops.

---

## `election`

| Setting | Default | Meaning |
|---|---|---|
| `date` | **required** | The day votes are counted, `YYYY-MM-DD`. The model's time grid ends exactly here, so the election-day estimate carries no stray extra drift. |
| `as_of` | today | Pretend it is this date. Polls after it are dropped from the fit. This is how you replay a past race. |
| `name` | none | Title shown on the report. Cosmetic. |

## `polls`

| Setting | Default | Meaning |
|---|---|---|
| `file` | **required** | Path to the CSV. One row per poll. |
| `columns.date` | `date` | Column holding the poll date. |
| `columns.pollster` | `pollster` | Column holding the firm's name. |
| `columns.sample_size` | `sample_size` | Column holding how many people were asked. |
| `columns.undecided` | none | Undecided / don't-know column. Named here, it is kept out of the party shares **and** cuts the poll's effective sample size by the decided fraction. Omitting a column that exists overstates the poll's precision. |
| `parties` | all other numeric columns | Restrict the model to these columns. Use it when the file has extras like `Other` or `Total` you do not want modelled. |
| `date_format` | auto | `strftime` pattern for unusual dates, e.g. `"%d/%m/%Y"`. |
| `decimal` | `.` | Use `","` for European-style numbers (`45,3`). |

Party numbers may be on any scale — percentages, fractions, raw counts. They
are normalised to sum to 100 internally.

## `beliefs`

This block is where a person's knowledge enters the model. Everything here is
an assumption you are choosing to make; the report prints all of it back.

### `volatility`

How fast real opinion can move, before any polls are seen.

| Value | Effect | Use when |
|---|---|---|
| `calm` | trend bends slowly, narrower ranges | entrenched electorate, quiet period |
| `normal` | about a point a week | the default |
| `volatile` | trend bends fast, wider ranges | active campaign, scandal, new entrant |

A number instead of a word sets `sigma_walk_prior` directly (logit units per
week: `calm` = 0.02, `normal` = 0.05, `volatile` = 0.10).

### `pollsters.<Firm>`

Only for knowledge from *outside* this poll file. The model already learns how
firms differ from one another; this is for saying something the data cannot.

| Setting | Meaning |
|---|---|
| `leans: {Party: +2}` | You believe this firm reports that party 2 points too high. Negative means too low. Sets the prior mean of the firm's house effect (`PollsterPrior.mu_house`). Range −50 to 50. |
| `trust: high \| normal \| low` | How far the firm may stray from the industry average. `high` pins it close (`sigma_house` 0.10), `low` lets it wander (0.60) and so quietly reduces its pull on the answer. `normal` learns it from the data. |
| `noisiness: low \| normal \| high` | How much to discount the firm's stated sample size. `high` (1.0) suits cheap online panels whose numbers bounce more than their sample size explains. |

Prefer `trust: low` over deleting a firm's polls. Deleting says "these are
worthless"; distrusting says "these count for less", which is nearly always
what is actually meant.

### `industry_error`

The error every firm shares. **Read this even if you skip everything else.**

A bias common to the whole industry is invisible to any model fitted to one
election's polls: shifting the true trend one way and every house effect the
other fits the data exactly as well. Worse, the model gauges how far firms may
stray from how much they differ *from each other* — so when they all lean the
same way, it concludes they are all accurate and reports a *narrower* range
around the wrong answer. On a synthetic dead heat where every firm shaded 3
points one way, kronikas put that side ahead with 97.8 % probability and its
90 % range excluded the truth.

| Setting | Default | Meaning |
|---|---|---|
| `uncertainty_pp` | `0` | How wrong the whole industry could be, in points, either direction. **`0` is not neutral** — it asserts collective perfection. Historical error in comparable races runs 2–3. |
| `expected: {Party: 2}` | none | A directional belief: positive means the polls have that party too high. Moves the headline. |
| `party_uncertainty: {Party: 3}` | uses `uncertainty_pp` | Per-party spread, for when one party is harder to poll than the rest. |

The number must come from outside: past elections, published post-mortems, a
known coverage problem. It cannot be estimated from the polls it applies to.

## `run`

| Setting | Default | Meaning |
|---|---|---|
| `effort` | `standard` | `quick` (750 warmup + 1000 draws x 2 chains, exploratory), `standard` (1500 + 1000 x 2), `thorough` (2000 + 2000 x 4, and a stricter acceptance rate). |
| `time_step_days` | `7` | How finely the trend is tracked. Smaller resolves faster swings and costs time. Changing it does **not** change how volatile the model thinks opinion is. |
| `seed` | `42` | Same seed and same inputs give the same numbers, every run. Change it only to check the answer is stable. |
| `progress` | `true` | Show the sampler's progress bar. |

Use `quick` while iterating. Move to `standard` before showing anyone, and
`thorough` if the model health section complains or the race is close enough
that the third decimal matters.

## `report`

| Setting | Default | Meaning |
|---|---|---|
| `output_dir` | `forecast-output` | Where every file is written. |
| `thresholds_pp` | none | Report each party's chance of clearing these shares, e.g. `[5]` for a 5 % seat threshold. |
| `industry_error_scenarios_pp` | none | Show the result under an industry-wide error of this many points, moved from the front-runner to the runner-up. `[1, 2, 3]` is a good default. |
| `party_colors` | automatic | `{Party: "#3366cc"}` to match a house style. |

## `advanced`

Anything here is passed straight to kronikas' `ModelConfig`, overriding
everything above. `correlated_walk`, `num_draws`, `target_accept`,
`initial_sigma`, `sampler_kwargs`, and the rest are listed in the project
README's configuration reference. An unknown name is rejected rather than
ignored.

---

## Replaying a past election

Set `election.date` to the real election day and `election.as_of` to a date
before it. Polls after `as_of` are dropped, so the forecast sees only what was
knowable then, and you can compare its numbers to what actually happened.

For a proper scored backtest across several dates at once — mean absolute
error, CRPS, interval hit rate — use the package's own command, which refits
at each date:

```bash
kronikas backtest polls.csv \
    --election-date 2026-04-12 \
    --as-of 2026-01-01 --as-of 2026-02-01 --as-of 2026-03-01 \
    --actual "Progress=41.8,Unity=39.2,Greens=19.0"
```

Each date costs a full fit, so start with `--draws 500`. And read the result
carefully: one election's interval hit rate describes that election, it does
not establish that the model is calibrated.
