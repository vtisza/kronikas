# The interview

Every answer the settings file needs, as a question a non-technical user can
answer in one click or one word.

## Ask with widgets, not with walls of text

**If your client can render a structured question — Claude Code's
`AskUserQuestion`, or any multiple-choice / form control — use it for every
question below that has fixed options.** Each one is written as a ready-made
widget spec: a short header, the question, and 2–4 labelled options with the
consequence of each. Put the recommended option first and mark it
`(Recommended)`. The user gets a click; you get an unambiguous answer instead
of "hmm, medium I guess".

Rules that matter more than they look:

- **Batch related questions** — up to four in one widget. One exchange for the
  whole of round 3 beats four round-trips.
- **Never widget a free value.** Dates, file paths, party names and poll
  numbers are typed, not chosen. Asking "which of these four dates" for an
  election date is worse than asking.
- **Every option needs its consequence in the description**, not a restatement
  of the label. "Widens the ranges and lets the line bend faster" tells them
  something; "opinion moves a lot" does not.
- **Skip what you can already see.** If the file has one pollster, do not ask
  which firms they distrust.
- Without a widget-capable client, ask the same questions in prose, offering
  the default inside the question so that silence is a valid answer.

There is also a browser form for the fiddly per-pollster part —
`kronikas form polls.csv --election-date YYYY-MM-DD` builds a page with a
control for every party and firm found in their file, and hands back finished
YAML. Offer it when the interview would otherwise run to a dozen numbers.

---

## Round 1 — the race (typed, not clicked)

> **When is the election — the day the votes are counted?**

→ `election.date`. Required. "Next April" is not enough; the model anchors its
entire time grid on the exact day.


> **Which country or region is this election in?**

→ `election.country`. Typed, not clicked. It appears on the report so a reader
knows which race they are looking at, and it tells you what to expect in the
data — a European file will often use `45,3` rather than `45.3`.

**It does not give you a polling-error figure.** If you happen to know a
well-established number for that country's recent industry-wide error, propose
it in round 3 **and name where it comes from**. If you cannot name a source,
use the default and say so. An invented country-specific number is worse than
a stated generic one, because it looks researched.

**Header:** `System`

> How do votes turn into power there?

| Option | Description |
|---|---|
| Proportional, party lists | Seats shared roughly in proportion; the largest party often does not govern. → `election.system: list-pr` |
| One seat per district | Seats won district by district; the national vote share and the seat count can point different ways. → `districts` |
| Districts plus lists | Some seats local, some allocated from lists. → `mixed` |
| Two rounds | This forecast is the first round only; the second turns on where eliminated candidates' voters go. → `runoff` |
| One nationwide contest | Most votes wins outright — the one case where these shares nearly are the answer. → `plurality` |
| An electoral college | The national vote share does not decide it. → `electoral-college` |

Anything else they say is accepted verbatim; the report falls back to the
general caveat. **This changes no number** — it changes what the report says
"chance of finishing first" is worth, which is the most misread thing in any
forecast. Ask it, and if they pick a system with a threshold, that is your cue
for the threshold question in round 4.

> **Do you have the polls in a file, or shall we build one together?**

If a file exists, read it and report what you found *before* asking anything
else: how many polls, the date range, the party columns, the polling firms.
Confirm the party columns are parties — a `Total` or `Other` column is common
and easy to misread as a candidate.

If you are building the file: for every poll, date, firm, sample size, and each
party's number. **Always ask for the sample size.** Without it every poll is
treated as equally informative, which is wrong and cannot be repaired later.

---

## Round 2 — the data questions worth a widget

Ask these only when the file shows a reason to.

**Header:** `Undecideds` — *only if a column looks like undecided/don't-know*

> There's a column called "Undecided". How should the model treat it?

| Option | Description |
|---|---|
| Handle it properly (Recommended) | Keeps it out of the party shares and reduces each poll's effective sample size to match. The honest treatment. → `polls.columns.undecided` |
| Ignore that column | Excludes it entirely. The remaining shares are rescaled, which makes each poll look more precise than it really is. |

**Header:** `Extra cols` — *only if a non-party numeric column is present*

> Which of these columns are actual parties? Set to multi-select, listing the
> detected columns.

→ `polls.parties`.

---

## Round 3 — the beliefs (widget these together)

**Header:** `Volatility`

> How fast does opinion move in this race?

| Option | Description |
|---|---|
| Normal (Recommended) | About a point a week. The right answer for an ordinary campaign. → `beliefs.volatility: normal` |
| Volatile | A race in motion — a scandal, a new entrant, numbers that have visibly swung within a month. Widens the ranges and lets the trend line bend faster. |
| Calm | An entrenched electorate in a quiet period. Trusts the polls more and reports narrower ranges. |

**Header:** `Pollsters`

> Do you know of a firm here that consistently reports a party higher or lower
> than the others do?

| Option | Description |
|---|---|
| No — let the model work it out (Recommended) | House effects are estimated from how the firms differ from each other. This is the right answer unless you know something the polls cannot show. |
| Yes, one or more leans | You'll tell me which firm, which party, and by how many points. Only use outside knowledge — a methodology review, a record against past results. |
| One firm has a poor record | Sets that firm's trust to low: its polls still count, but count for less. Better than deleting them, which is a stronger claim. |

If they pick "yes", follow up per firm — and if there are more than two firms
to configure, offer the browser form instead of a chain of questions.
→ `beliefs.pollsters.<Firm>.leans` / `.trust`

**Header:** `Poll error`

Ask this one **every time**, and explain before offering the options: when all
the pollsters miss the same way, no model can see it from the polls. They
agree with each other, and agreement reads as accuracy — so the model gets
*more* confident, not less.

> How wrong could the whole polling industry be, all at once?

| Option | Description |
|---|---|
| About 2.5 points (Recommended) | Matches the industry-wide error in recent comparable elections. → `beliefs.industry_error.uncertainty_pp: 2.5` |
| About 4 points | For a race with hard-to-reach voters, a new voting system, or a track record of misses. |
| Assume the polls are collectively unbiased | Zero. Not a neutral choice — it asserts the industry is collectively perfect, which is the assumption behind most famous forecasting failures. |
| I think they lean a specific way | You name the party and direction; it moves the headline, and the report records that it came from you. → `beliefs.industry_error.expected` |

---

## Round 4 — practicalities (widget these together)

**Header:** `Effort`

> How careful should this run be?

| Option | Description |
|---|---|
| Standard (Recommended) | A few minutes. Right for anything you'll show another person. → `run.effort: standard` |
| Quick | About a minute, coarser numbers. For a first look while you're still adjusting settings. |
| Thorough | Several times longer. Use when the race is close or a standard run reported trouble. |

**Header:** `Threshold`

> Is there a share a party must clear to win any seats at all?

| Option | Description |
|---|---|
| No threshold (Recommended for two-party races) | Skip it. |
| 5% | Common in list-PR systems. → `report.thresholds_pp: [5]` |
| Another number | You'll type it. |

---

## Then

Write `forecast.yaml`, run `kronikas guided forecast.yaml --check`, and show
them the plain-language read-back it prints. Get an explicit "yes, that's
right" before spending the sampling time.
