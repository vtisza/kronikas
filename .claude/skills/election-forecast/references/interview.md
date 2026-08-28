# The interview

Everything the settings file needs, as questions a non-technical user can
answer. Ask in this order. Skip anything you already know. Offer the default
in the question itself so silence is a valid answer, and write the answers
into `forecast.yaml` as you go.

Do not ask all of these at once. Five short exchanges beat one form.

---

## Round 1 — the race

> **When is the election?** (the day votes are counted)

→ `election.date`. Required. If they give "next April", ask for the exact day;
the model anchors its whole time grid on it.

> **Should the forecast use everything up to today, or pretend it is an
> earlier date?**

→ `election.as_of`. Default today. An earlier date is how you check the model
against a race that already happened — polls after that date are ignored.

> **What should I call this in the report?**

→ `election.name`. Cosmetic. Skip if they do not care.

---

## Round 2 — the polls

> **Do you have the polls in a file, or shall I build one with you?**

If a file exists, read it and report back what you found before asking
anything else: number of polls, date range, party columns, polling firms.
Confirm the party columns are the parties and not something else (a "Total"
or "Other" column is common and easy to misread).

If you are building it: for every poll ask date, firm, sample size, and each
party's number. **Ask for the sample size.** Without it every poll gets
treated as equally informative, which is wrong and unfixable later.

> **Is there an undecided or "don't know" column?**

→ `polls.columns.undecided`. If yes, name it. It is then excluded from the
party shares *and* shrinks the poll's effective sample size, which is the
honest treatment. Leaving it in as if it were a party is wrong; deleting it
makes the poll look sharper than it is.

> **Do the numbers use a comma for decimals (45,3)?**

→ `polls.decimal: ","`. Common in European files. Only ask if you see it.

---

## Round 3 — how fast opinion moves

> **Between polls, does opinion in this race move a lot, a little, or
> normally? Normal means roughly a point a week.**

→ `beliefs.volatility`: `calm` | `normal` | `volatile`.

- `calm` — entrenched electorate, nothing much happening. Trusts the polls
  more and gives narrower ranges.
- `normal` — the default. Use it unless they have a reason.
- `volatile` — a campaign in motion, a scandal, a new entrant. Widens the
  ranges and lets the line bend faster.

Guidance if they hesitate: an ordinary campaign is `normal`; a race where the
numbers have visibly swung inside a month is `volatile`.

---

## Round 4 — individual polling firms

> **Do you know of any firm here that consistently reports a particular party
> higher or lower than the others do?**

Default: no, and the model will work it out. Only record a lean when they
have a reason from *outside* this poll file — a published methodology review,
a track record against past results.

If yes, for each firm:

> **Which party, and by how many points, in which direction?**

→ `beliefs.pollsters.<Firm>.leans: {Party: +2}`. Positive means the firm
reports that party **too high**.

> **How much do you trust this firm overall — high, normal, or low?**

→ `beliefs.pollsters.<Firm>.trust`. `high` holds the firm close to the
industry average; `low` lets it wander further and quietly reduces its
influence on the answer. Use `low` for a firm with a poor record rather than
deleting its polls: dropping data is a stronger claim than distrusting it.

> **Does this firm's sample size look optimistic — do its numbers bounce
> around more than its stated sample would explain?**

→ `beliefs.pollsters.<Firm>.noisiness: high`. Typical for cheap online panels.
Only ask this if they have already flagged the firm.

---

## Round 5 — the question nobody asks

> **Could every firm be wrong in the same direction at once?**

Ask this one every time, and explain why before they answer:

> When all the pollsters miss the same way — as they have in several recent
> elections — no model can see it from the polls. They all agree, and the
> model reads agreement as accuracy: it gets *more* confident, not less.
> The only fix is for you to tell it how big that shared error could be.
> Historically it runs about 2 to 3 points. Shall I use 2.5?

→ `beliefs.industry_error.uncertainty_pp`. Recommend `2.5`. Zero is not a
neutral choice — it is the claim that the industry is collectively perfect.

If they also believe the error has a *direction*:

> **Which party do you think the polls are overstating, and by how much?**

→ `beliefs.industry_error.expected: {Party: 2}`. Positive means the polls have
that party too high. Record it as their belief, and make sure the report says
so — it moves the headline number, and a reader deserves to know why.

---

## Round 6 — practicalities

> **A rough answer in a minute, or a careful one in several?**

→ `run.effort`: `quick` for exploring, `standard` for anything shown to other
people, `thorough` when the model complains or the race is close.

> **Is there a threshold that matters — a share a party must clear to win
> seats at all?**

→ `report.thresholds_pp: [5]`. Common in list-PR systems.

Then write the file, run `--check`, and read the summary back to them.
