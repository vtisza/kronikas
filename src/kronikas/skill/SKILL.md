---
name: election-forecast
description: Turn opinion polls into a full election forecast with kronikas — install it, collect the poll data, capture the user's assumptions about pollster bias, run the Bayesian model, and produce a visual report. Use when someone wants to forecast or simulate an election, aggregate or average polls, estimate a party's chance of finishing first, adjust for house effects or a polling error, backtest a past race, or asks "who is going to win" from polling data. Written for users with no statistics or programming background.
---

# Election forecast, end to end

Take someone from "I have some poll numbers" to a finished forecast they can
read, question, and defend. Assume no statistics and no programming. You do
the mechanics; they supply the data and the judgement calls.

`$SKILL` below means this skill's own directory — `kronikas skill path` prints
the packaged copy. `$KRONIKAS` means the `kronikas` command from the
environment step 1 sets up; if it is not on the PATH, use
`<workdir>/.kronikas-venv/bin/kronikas`.

## How to run this conversation

- **Ask with widgets wherever your client has them.** Claude Code's
  `AskUserQuestion` renders clickable options; use it for every question that
  has fixed answers, batching up to four at a time, recommended option first.
  `references/interview.md` gives each question as a ready-made spec — header,
  wording, options, and what each one does. A click beats a paragraph, and it
  removes the "hmm, medium I guess" answer entirely.
- **Type only what cannot be clicked**: dates, file paths, party names, poll
  numbers. Never offer a menu of guessed dates.
- **Ask one small thing at a time** otherwise, and put the default inside the
  question so silence is a valid answer.
- **Never invent poll numbers, dates, or sample sizes.** If a value is
  missing, ask. A fabricated poll silently corrupts every number downstream.
- **No jargon in what you say.** Not "prior", "posterior", "R-hat" — say "the
  assumption you gave me", "the range of outcomes", "a sign the model did not
  settle". `references/interpreting.md` has the plain-language phrasing.
- **Say what a number does not mean**, especially "chance of finishing first"
  vs "chance of taking office". They are different, and the difference has
  embarrassed better forecasters than this one.
- Work in a directory of the user's choosing (default: the current one).
  Everything — settings, data, results — lives there, so it is easy to keep.

## Step 1 — Install it (once per machine)

If `kronikas` already runs, skip this. Otherwise:

```bash
bash $SKILL/scripts/setup_kronikas.sh --dir <workdir>
```

It picks a Python between 3.10 and 3.12, builds a virtual environment at
`<workdir>/.kronikas-venv`, installs kronikas, and prints the paths to use.
`--source` clones the GitHub repository and installs from that instead of
from PyPI.

By hand it is `pip install kronikas` into any Python 3.10–3.12 environment.
Either way it takes a few minutes (PyMC is a large dependency) — say so before
you start, rather than leaving the user watching a silent terminal. Windows
without Git Bash, or any failure: `references/setup.md`.

## Step 2 — Get the polls into a CSV

The model needs one row per poll:

```csv
date,pollster,sample_size,PartyA,PartyB,PartyC
2026-01-15,Meridian,1000,45,40,10
```

- **They already have a file** → read the first few rows yourself, tell them
  what you found (how many polls, which firms, what date range), and map any
  differently-named columns in the settings file rather than editing their data.
- **They have numbers in a message, a spreadsheet or a web page** → write the
  CSV for them, then show it back and get an explicit confirmation before
  running anything. Ask for the sample size of every poll; if a poll genuinely
  has none, ask for a typical size for that firm and note in the report that
  it was assumed.
- **Undecided voters**: if there is such a column, keep it and name it under
  `polls.columns.undecided`. Dropping it makes every poll look more precise
  than it is.
- **They just want to see it work** → copy `$SKILL/assets/polls.example.csv`
  and `$SKILL/assets/forecast.example.yaml` into the working directory and run
  those. About a minute end to end.

Details of every column option: `references/settings.md`.

## Step 3 — Write the settings file

Copy `$SKILL/assets/forecast.template.yaml` to `<workdir>/forecast.yaml` and
fill it in from their answers. The questions, each written as a widget spec
you can hand straight to `AskUserQuestion`:
**`references/interview.md`**. What each setting does:
**`references/settings.md`**.

When the beliefs get fiddly — more than a couple of firms to configure, or a
lean to set per party — build them a form instead of asking twenty questions:

```bash
$KRONIKAS form <workdir>/polls.csv --election-date <date>
```

It reads their poll file and writes `settings-builder.html` with a control for
every party and every firm actually in it — sliders, steppers, and radio
buttons, with the finished YAML updating live at the bottom of the page. They
fill it in, hit **Download**, and you carry on from the file they saved. Tell
them to open it: it is a local page, so nothing is uploaded anywhere.

The two questions that matter most, and that nobody else will ask them:

1. **Does any single firm lean?** — `beliefs.pollsters.<Firm>.leans`, in
   percentage points. Only from outside knowledge; the model already learns
   differences *between* firms on its own.
2. **Could every firm be wrong the same way?** —
   `beliefs.industry_error.uncertainty_pp`. This one cannot be measured from
   the polls, ever: when everybody leans the same way the model reads their
   agreement as accuracy and gets confident *and* wrong. Left unset, the
   forecast asserts the industry is collectively perfect. Historically that
   error runs 2–3 points. Recommend `2.5` unless they have a better number,
   and tell them what you did.

## Step 4 — Check before you spend the time

```bash
$KRONIKAS guided <workdir>/forecast.yaml --check
```

Validates the settings and the data without sampling, and reads back — in
plain language — exactly what the model has been told. **Show that read-back
to the user and get a "yes, that's right" before running.** Fix any misspelled
party or pollster name here; the check names the likely intended spelling.

## Step 5 — Run it

```bash
$KRONIKAS guided <workdir>/forecast.yaml
```

Minutes, not seconds — `quick` effort is about a minute, `standard` several,
`thorough` longer on a big file. Tell them before you start; do not run it in
the background and go quiet.

Writes to `report.output_dir`:

| File | What it is |
|---|---|
| `report.html` | the visual report — **this is the deliverable** |
| `report_data.json` | everything the report draws, for rebuilding it |
| `estimates.csv`, `trend.csv`, `draws_election_day.csv` | for spreadsheets |
| `forecast.json` | compact summary for publishing |

Restyle or rebuild the page without refitting:
`$KRONIKAS report <outdir>/report_data.json`.

A non-zero exit means the sampler did not settle. Say so plainly, do not
quote the numbers as if they were fine, and see `references/troubleshooting.md`.

## Step 6 — Read the report with them

Open `report.html`, then walk them through it in this order — headline, the
ranges, the trend, the break-even error, model health. Use
`references/interpreting.md` for the wording, including what to say when the
answer is "this race is too close to call" and how to handle "so who wins?".

Point at the break-even number every time. A 1.5-point break-even and a
9-point break-even are the same forecast in the headline and completely
different in reality.

## Step 7 — What they will ask next

- *"What if Party X is really 3 points lower?"* → add it to
  `beliefs.industry_error.expected` and re-run, or read the scenario table
  already in the report.
- *"Would this have worked last time?"* → set `election.as_of` to a date
  before a past election and compare with the real result;
  `references/settings.md` covers backtesting properly.
- *"Can I share this?"* → `report.html` is a single self-contained file. It
  works over email, with no internet connection and nothing to install.

## Never

- Never quote a forecast from a run whose model health section is red.
- Never fill in a missing poll, sample size or date yourself.
- Never describe "chance of finishing first" as the chance of governing.
- Never leave the industry-wide error question unasked. Silence on it is
  itself an assumption, and the least defensible one available.
