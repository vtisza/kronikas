# When something goes wrong

Each entry: what the user sees, what it actually means, what to do.

---

## "The sampler did not settle" / red model health

The run exits non-zero and the report's health section is red. The internal
checks say the computation has not stabilised, so the numbers are unreliable
— separately from any question about the polls.

**Say so plainly.** Do not quote the figures "for now". Then, in order:

1. Raise the effort: `run.effort: thorough`. Fixes most cases on its own.
2. Widen the time grid: `time_step_days: 7` or `14`. A grid finer than the
   polls can support gives the model more unknowns than evidence.
3. If the parties include a very small one (under ~2 %), consider whether the
   data can really carry it, or whether it belongs in an "Other" column.
4. Still red: `advanced: {target_accept: 0.995, init_method: adapt_full}`.

Typical wording:

> "The model has not settled yet — the internal consistency checks failed, so
> I don't trust these numbers. I'm re-running it more carefully; it will take
> a few more minutes."

## "R-hat" / "effective sample size" / "divergences" in the health section

- **R-hat above 1.01** — the independent runs disagree. More effort.
- **Effective sample size below 400** — the answer is stable but grainy;
  probabilities may wobble a point or two between runs. More draws.
- **Divergences above zero** — the geometry is awkward for the sampler. Raise
  `target_accept` toward 0.99; suspect a grid that is too fine.

## "kronikas is not installed in this Python environment"

The script ran under the wrong interpreter. Use the path printed by the setup
script (`<workdir>/.kronikas-venv/bin/python`), not `python` or `python3`.

## Install fails: "no matching distribution found for kronikas"

The Python is too new or too old. kronikas needs 3.10–3.12. See
`setup.md` for installing a supported version.

## "Your settings mention the party 'X', but the poll file has ..."

A name in `beliefs` does not match a column in the CSV. The message suggests
the likely intended spelling. Match the CSV exactly — case and spacing
included — rather than renaming columns in the user's data file.

## "unknown setting(s) ..."

A misspelled key in `forecast.yaml`. The message lists what is allowed at that
level. Do not work around it by moving the setting elsewhere; check
`settings.md` for where it actually belongs.

## The poll file will not load

- *"could not convert string to float"* — a comma decimal separator
  (`polls.decimal: ","`), a thousands separator, or a stray `%` in the cells.
- *dates land in the wrong year* — set `polls.date_format`, e.g. `"%d/%m/%Y"`.
- *a column is treated as a party that should not be* — list the real parties
  under `polls.parties`.
- *"sample_size must be positive"* — a blank or zero sample size. Ask the user
  for the real figure; do not fill it in yourself.

## "Only one pollster, house effects not estimated"

Correct behaviour, not a bug. With a single firm there is no way to tell that
firm's habits from real movement in opinion. The forecast still runs; the
house-effects section explains its own absence. Worth telling the user that
one firm's systematic error passes straight into the answer here, and that
`beliefs.industry_error.uncertainty_pp` is the only defence available.

## Every party shows a 0 % or 100 % chance

Usually a real gap far larger than the uncertainty — a party 20 points clear
with three months of stable polls. Check the trend chart looks sane and the
parties are not mislabelled. Then say it honestly: on this data the ordering
is not in doubt, and the interesting question is the margin, not the winner.

## It is taking forever

`standard` on a few dozen polls is a few minutes; `thorough` several times
that. Beyond that: too fine a `time_step_days`, many parties, or a very long
poll history. Drop to `quick` while iterating, and only spend the full run
once the settings are settled.

A pip-installed PyTensor without a tuned BLAS also samples slowly. A conda
install is meaningfully faster; `setup.md` has the command.

## The form's Download button does nothing

Some browsers block downloads from a `file://` page. Use **Copy** instead and
paste the settings into a new file called `forecast.yaml`, or paste them back
to your assistant and let it write the file.

## The form does not list one of my parties

It lists exactly what the poll file contains. A missing party means the column
is not being read as a party — usually a stray `Total` or `Other` column
confusing the reader, or a non-numeric cell. Check the file, then rebuild the
page; do not add the name by hand, because the run will reject a party that
the data does not have.

## The report opens but looks empty

`report.html` needs its sibling files only for *rebuilding*, not for viewing —
it is self-contained. An empty page instead means the run failed before the
report was written; check the terminal output for the real error.

## Re-running gives slightly different numbers

Only if `run.seed` changed, or the settings did. Same seed and same inputs
reproduce exactly. Small differences between seeds are the sampler's own
noise; large ones mean the fit is not settled — treat it as a red health
section.
