# Reading the report out loud

Wording for walking a non-technical user through `report.html`. Use their
party names, their numbers, and the shortest true sentence available.

---

## The headline

> "Progress is ahead. If the election were held on the modelled date and the
> polls are roughly right, it finishes first about two times in three."

Two things to add immediately, every time:

- **Two thirds is not a certainty.** One in three is how often it *does not*
  happen. If they want a feel for it: a one-in-three chance is about the same
  as rolling a 5 or a 6 on a die.
- **"Finishes first" means most votes, not most power.** Under runoffs,
  district seats, an electoral college, or coalition arithmetic, the party
  with the most votes may well not govern. This model knows nothing about
  seats. If they need seats, the vote-share draws in
  `draws_election_day.csv` are the input to a separate seat model.

## The ranges

> "The best single guess is 41.6 %. The bar says the true share should land
> between 37 and 46 nine times in ten. That width is the honest part — a
> single number would just be hiding it."

If the bars for the top two parties overlap a lot:

> "These two overlap heavily. That is the model saying the race is genuinely
> open, not that it is confused."

## The trend

> "The line is where the model thinks real support actually was at each point.
> The dots are the individual polls. The line does not chase every dot on
> purpose — some of that scatter is sampling noise and some is one firm's
> habits, and separating the two is most of what this model does."

## The break-even error

The single most useful number in the report, and the one nobody else shows.

> "If every polling firm were wrong in the same direction by 1.8 points, the
> lead disappears. Errors of that size have happened in real elections
> repeatedly. So the lead is real, but it is not safe."

versus

> "It would take a 7-point across-the-board polling error to overturn this.
> That would be extraordinary. This lead is comfortable."

Why it cannot simply be measured, if they ask:

> "When every firm misses the same way, the polls all agree with each other —
> and a model reads agreement as accuracy. It would become *more* confident,
> not less. Nothing in the data can catch that, which is why we state the size
> we are worried about instead of pretending to measure it."

## House effects

> "This shows how each firm sits relative to the others: to the right means it
> reports that party higher than its peers do. It is a comparison *between*
> firms only. If all of them lean the same way, that shows up nowhere on this
> chart — the break-even number above is where that risk lives."

## Model health

Green:

> "The fit is clean. The remaining uncertainty is about the world, not about
> the computation."

Red:

> "The model did not settle — the internal checks failed, so these numbers are
> not reliable yet. Let me re-run it at a higher effort setting before either
> of us reads anything into them."

Then actually re-run. Do not quote a red run's numbers, not even provisionally.
See `troubleshooting.md`.

---

## Questions you will get

**"So who's going to win?"**

> "On these polls, Progress is the favourite — roughly a two-in-three chance of
> getting the most votes. It is a favourite, not a lock, and a normal-sized
> polling error would be enough to flip it."

Never answer with a name alone.

**"Why doesn't this match the average of the polls?"**

> "Three reasons. Recent polls count for more than old ones. Bigger samples
> count for more than small ones. And each firm's habitual lean is estimated
> and taken out. A plain average does none of that."

**"Can you make it more confident?"**

> "Only by telling it something we do not actually know. The width comes from
> the polls disagreeing and from time left to run. I can narrow it by claiming
> opinion is calmer than it is, or that the polling industry cannot be
> collectively wrong — but that would be manufacturing confidence, not
> finding it."

**"The polls are all rubbish / all biased."**

That is a substantive claim and the model can take it — as a number.

> "Which way, and how many points? I will put it in as a stated assumption,
> and the report will show that it came from you rather than from the data."

→ `beliefs.industry_error.expected`.

**"Is this the same as [well-known forecaster]?"**

> "Same family of method — a Bayesian poll aggregator with pollster bias
> correction. The differences are in the extras they carry: economic
> fundamentals, seat-level models, turnout adjustments. This one models vote
> share from polls and is explicit about what it does not include."

---

## Words to avoid, and what to say instead

| Do not say | Say |
|---|---|
| prior | the assumption you gave me |
| posterior / credible interval | the range of outcomes |
| convergence / R-hat | whether the calculation settled |
| house effect | how far a firm sits from the others |
| plurality probability | chance of getting the most votes |
| MCMC draws | simulated elections |
