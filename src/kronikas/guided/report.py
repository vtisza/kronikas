#!/usr/bin/env python3
"""Turn ``report_data.json`` into a single self-contained HTML report.

Driven by ``kronikas report``, and by ``kronikas guided`` at the end of a run.

Standard library only, and every chart is inline SVG, so the file opens in
any browser with no internet connection, no plotting library, and nothing
left to install.  Rebuilding the report never refits the model.
"""

from __future__ import annotations

import html
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

# --- small helpers ---------------------------------------------------------


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%" if value >= 0.005 or value == 0 else "<1%"


def _pp(value: float) -> str:
    return f"{value:.1f}"


def _parse_date(text: str) -> date:
    return datetime.strptime(text[:10], "%Y-%m-%d").date()


def _nice_bounds(low: float, high: float) -> tuple[float, float, float]:
    """Round a data range outwards to a readable axis with a sane step."""
    if high <= low:
        high = low + 1.0
    span = high - low
    for step in (1, 2, 5, 10, 20, 25, 50):
        if span / step <= 6:
            break
    axis_low = step * ((low - span * 0.08) // step)
    axis_high = step * -(-(high + span * 0.08) // step)
    return float(axis_low), float(axis_high), float(step)


# --- charts ----------------------------------------------------------------


def _probability_bars(data: dict[str, Any]) -> str:
    probs = data["win_probabilities"]
    order = sorted(probs, key=lambda name: -probs[name])
    rows = []
    for name in order:
        share = max(0.0, min(1.0, probs[name]))
        color = data["colors"].get(name, "#3366cc")
        rows.append(
            f'<div class="barrow"><div class="barname">{_esc(name)}</div>'
            f'<div class="bartrack"><div class="barfill" style="width:'
            f'{share * 100:.2f}%;background:{_esc(color)}"></div></div>'
            f'<div class="barvalue">{_pct(probs[name])}</div></div>'
        )
    return '<div class="bars">' + "".join(rows) + "</div>"


def _range_chart(estimates: list[dict[str, Any]], colors: dict[str, str]) -> str:
    """Dot-and-interval chart: mean with the 90 % credible interval."""
    if not estimates:
        return ""
    ordered = sorted(estimates, key=lambda e: -e["mean"])
    low = min(e["ci_lower"] for e in ordered)
    high = max(e["ci_upper"] for e in ordered)
    axis_low, axis_high, step = _nice_bounds(low, high)
    width, row_h, pad_l, pad_r, pad_t = 760, 34, 130, 40, 12
    height = pad_t + row_h * len(ordered) + 34
    plot_w = width - pad_l - pad_r

    def x_of(value: float) -> float:
        return pad_l + plot_w * (value - axis_low) / (axis_high - axis_low)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" '
        f'role="img" aria-label="Forecast ranges by party">'
    ]
    tick = axis_low
    while tick <= axis_high + 1e-9:
        x = x_of(tick)
        parts.append(
            f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" '
            f'y2="{pad_t + row_h * len(ordered):.1f}" class="grid"/>'
            f'<text x="{x:.1f}" y="{height - 12}" class="tick mid">{tick:g}%</text>'
        )
        tick += step
    for index, estimate in enumerate(ordered):
        y = pad_t + row_h * index + row_h / 2
        color = colors.get(estimate["name"], "#3366cc")
        parts.append(
            f'<text x="{pad_l - 12}" y="{y + 4:.1f}" class="rowlabel">'
            f"{_esc(estimate['name'])}</text>"
            f'<line x1="{x_of(estimate["ci_lower"]):.1f}" y1="{y:.1f}" '
            f'x2="{x_of(estimate["ci_upper"]):.1f}" y2="{y:.1f}" '
            f'stroke="{_esc(color)}" stroke-width="8" stroke-linecap="round" '
            f'opacity="0.28"/>'
            f'<circle cx="{x_of(estimate["mean"]):.1f}" cy="{y:.1f}" r="6" '
            f'fill="{_esc(color)}"/>'
            f'<text x="{x_of(estimate["ci_upper"]) + 10:.1f}" y="{y + 4:.1f}" '
            f'class="rowvalue">{_pp(estimate["mean"])}%</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _trend_chart(data: dict[str, Any]) -> str:
    """Latent support over time, with the polls that informed it."""
    trend = data.get("trend") or {}
    dates = [_parse_date(d) for d in trend.get("dates", [])]
    series = trend.get("series", {})
    if len(dates) < 2 or not series:
        return ""
    polls = data.get("polls", [])
    colors = data["colors"]

    values = [v for s in series.values() for v in s["lo"] + s["hi"]]
    for poll in polls:
        values.extend(poll["shares"].values())
    axis_low, axis_high, step = _nice_bounds(min(values), max(values))
    axis_low = max(0.0, axis_low)

    start, end = dates[0], dates[-1]
    total_days = max((end - start).days, 1)
    width, height = 900, 400
    pad_l, pad_r, pad_t, pad_b = 46, 18, 16, 40
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b

    def x_of(day: date) -> float:
        return pad_l + plot_w * max(0.0, min(1.0, (day - start).days / total_days))

    def y_of(value: float) -> float:
        span = axis_high - axis_low or 1.0
        return pad_t + plot_h * (1 - (value - axis_low) / span)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" '
        f'aria-label="Support over time">'
    ]
    tick = axis_low
    while tick <= axis_high + 1e-9:
        y = y_of(tick)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
            f'class="grid"/><text x="{pad_l - 8}" y="{y + 4:.1f}" '
            f'class="tick end">{tick:g}%</text>'
        )
        tick += step
    n_ticks = min(6, len(dates))
    for i in range(n_ticks):
        day = dates[round(i * (len(dates) - 1) / max(n_ticks - 1, 1))]
        parts.append(
            f'<text x="{x_of(day):.1f}" y="{height - 14}" class="tick mid">'
            f"{day.day} {day:%b}</text>"
        )

    for party, points in series.items():
        color = colors.get(party, "#3366cc")
        upper = " ".join(
            f"{x_of(d):.1f},{y_of(v):.1f}"
            for d, v in zip(dates, points["hi"], strict=False)
        )
        lower = " ".join(
            f"{x_of(d):.1f},{y_of(v):.1f}"
            for d, v in zip(reversed(dates), reversed(points["lo"]), strict=False)
        )
        parts.append(
            f'<polygon points="{upper} {lower}" fill="{_esc(color)}" opacity="0.13"/>'
        )
    for poll in polls:
        day = _parse_date(poll["date"])
        if day < start or day > end:
            continue
        for party, value in poll["shares"].items():
            if party not in series:
                continue
            parts.append(
                f'<circle cx="{x_of(day):.1f}" cy="{y_of(value):.1f}" r="2.6" '
                f'fill="{_esc(colors.get(party, "#3366cc"))}" opacity="0.5"/>'
            )
    for party, points in series.items():
        line = " ".join(
            f"{x_of(d):.1f},{y_of(v):.1f}"
            for d, v in zip(dates, points["mean"], strict=False)
        )
        parts.append(
            f'<polyline points="{line}" fill="none" '
            f'stroke="{_esc(colors.get(party, "#3366cc"))}" stroke-width="2.4"/>'
        )
    election = _parse_date(data["election"]["date"])
    if start <= election <= end:
        x = x_of(election)
        parts.append(
            f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" '
            f'y2="{pad_t + plot_h}" class="marker"/>'
            f'<text x="{x - 6:.1f}" y="{pad_t + 12}" class="tick end">'
            f"election day</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _legend(data: dict[str, Any]) -> str:
    items = "".join(
        f'<span class="key"><i style="background:'
        f'{_esc(data["colors"].get(party, "#3366cc"))}"></i>{_esc(party)}</span>'
        for party in data["parties"]
    )
    return f'<div class="legend">{items}</div>'


def _house_chart(data: dict[str, Any]) -> str:
    house = data.get("house_effects")
    if not house:
        return ""
    parties, pollsters = house["parties"], house["pollsters"]
    means = house["mean"]
    limit = max(
        (abs(v) for row in means for v in row),
        default=1.0,
    )
    limit = max(limit * 1.25, 1.0)
    width, pad_l, pad_r = 760, 130, 24
    row_h, group_gap = 16, 16
    height = 30 + sum(row_h * len(parties) + group_gap for _ in pollsters)
    plot_w = width - pad_l - pad_r
    centre = pad_l + plot_w / 2

    def x_of(value: float) -> float:
        return centre + (plot_w / 2) * (value / limit)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" '
        f'aria-label="House effects by pollster">',
        f'<line x1="{centre:.1f}" y1="10" x2="{centre:.1f}" '
        f'y2="{height - 20}" class="marker"/>',
    ]
    y = 18
    for p_index, pollster in enumerate(pollsters):
        parts.append(
            f'<text x="{pad_l - 12}" y="{y + row_h * len(parties) / 2:.1f}" '
            f'class="rowlabel">{_esc(pollster)}</text>'
        )
        for c_index, party in enumerate(parties):
            value = means[p_index][c_index]
            colour = data["colors"].get(party, "#3366cc")
            x0, x1 = sorted((centre, x_of(value)))
            parts.append(
                f'<rect x="{x0:.1f}" y="{y + 3:.1f}" width="{max(x1 - x0, 1):.1f}" '
                f'height="{row_h - 6}" fill="{_esc(colour)}" opacity="0.8"><title>'
                f"{_esc(pollster)} / {_esc(party)}: {value:+.1f} pp</title></rect>"
            )
            y += row_h
        y += group_gap
    parts.append(
        f'<text x="{pad_l}" y="{height - 6}" class="tick start">'
        f"understates by {limit:.1f} pp</text>"
        f'<text x="{width - pad_r}" y="{height - 6}" class="tick end">'
        f"overstates by {limit:.1f} pp</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


# --- sections --------------------------------------------------------------


def _headline(data: dict[str, Any]) -> str:
    probs = data["win_probabilities"]
    if not probs:
        return ""
    leader = max(probs, key=lambda name: probs[name])
    estimates = {e["name"]: e for e in data["election_day_estimates"]}
    lead = estimates[leader]
    ordered = sorted(data["election_day_estimates"], key=lambda e: -e["mean"])
    margin = ordered[0]["mean"] - ordered[1]["mean"] if len(ordered) > 1 else 0.0
    days = data["election"]["days_to_go"]
    when = (
        f"{days} days before the vote"
        if days > 0
        else "on the day of the vote"
        if days == 0
        else f"{abs(days)} days after the vote"
    )
    breakeven = data.get("breakeven_pp")
    if breakeven is None:
        fragility = (
            "The lead survives every industry-wide polling error the model "
            "considered, so the ordering is not in doubt on this data."
        )
    else:
        # How alarming the break-even figure is depends entirely on its size,
        # so the sentence has to change with it.
        if breakeven < 1.0:
            verdict = (
                "That is smaller than the error in a routine election, so this "
                "is a toss-up rather than a lead."
            )
        elif breakeven < 3.0:
            verdict = "Errors of that size happen in real elections regularly."
        else:
            verdict = "An error that large would be unusual."
        fragility = (
            f"An industry-wide polling error of about "
            f"<strong>{breakeven:.1f} points</strong> would be enough to erase "
            f"the lead. {verdict}"
        )
    return f"""
    <section class="headline">
      <p class="kicker">Forecast for {_esc(data["election"]["date"])}
        &middot; made {when} using polls up to
        {_esc(data["election"]["as_of"])}</p>
      <h1>{_esc(leader)} leads, with a {_pct(probs[leader])} chance of
        finishing first</h1>
      <p class="lede">Central estimate {_pp(lead["mean"])}% of the vote
        (90% range {_pp(lead["ci_lower"])}–{_pp(lead["ci_upper"])}%),
        {_pp(abs(margin))} points ahead of the runner-up. {fragility}</p>
    </section>
    """


def _threshold_section(data: dict[str, Any]) -> str:
    thresholds = data.get("threshold_probabilities") or {}
    if not thresholds:
        return ""
    rows = []
    for level, probs in thresholds.items():
        cells = "".join(
            f"<td>{_pct(probs[p])}</td>" for p in data["parties"] if p in probs
        )
        rows.append(f"<tr><th>{_esc(level)}%</th>{cells}</tr>")
    heads = "".join(f"<th>{_esc(p)}</th>" for p in data["parties"])
    return f"""
    <section>
      <h2>Chance of clearing a threshold</h2>
      <p class="note">Probability that a party's election-day vote share lands
        at or above the level in the first column. Useful for electoral
        thresholds that decide whether a party gets seats at all.</p>
      <table><thead><tr><th>At least</th>{heads}</tr></thead>
      <tbody>{"".join(rows)}</tbody></table>
    </section>
    """


def _scenario_section(data: dict[str, Any]) -> str:
    scenarios = data.get("scenarios") or []
    breakeven = data.get("breakeven_pp")
    if not scenarios and breakeven is None:
        return ""
    rows = []
    for scenario in scenarios:
        probs = scenario["win_probabilities"]
        cells = "".join(f"<td>{_pct(probs.get(p, 0.0))}</td>" for p in data["parties"])
        rows.append(f"<tr><th>{scenario['pp']:g} pp</th>{cells}</tr>")
    heads = "".join(f"<th>{_esc(p)}</th>" for p in data["parties"])
    table = (
        f"<table><thead><tr><th>Error</th>{heads}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        if rows
        else ""
    )
    breakeven_line = (
        f"<p><strong>Break-even error: {breakeven:.1f} percentage points.</strong> "
        "That is the smallest across-the-board polling error that would make "
        "the race a coin flip.</p>"
        if breakeven is not None
        else "<p>No industry-wide error up to 25 points would overturn the "
        "leader's advantage.</p>"
    )
    return f"""
    <section>
      <h2>How much would a polling error cost?</h2>
      <p class="note">If every pollster is wrong in the same direction, no
        model can detect it from the polls — the mistake looks like agreement,
        which looks like precision. So instead of pretending to measure it,
        this asks what it would take to change the answer. Each row moves that
        many points from the front-runner to the runner-up.</p>
      {breakeven_line}
      {table}
    </section>
    """


def _house_section(data: dict[str, Any]) -> str:
    if not data.get("house_effects"):
        reason = (
            "Only one pollster is present, so a pollster's lean cannot be told "
            "apart from a real change in opinion. House effects were not "
            "estimated."
            if len(data.get("pollsters", [])) <= 1
            else "House effects were not available for this run."
        )
        return (
            "<section><h2>Pollster house effects</h2>"
            f'<p class="note">{reason}</p></section>'
        )
    return f"""
    <section>
      <h2>Pollster house effects</h2>
      <p class="note">How far each firm's results sit from the industry
        average, after accounting for when its polls were taken. Bars to the
        right mean the firm reports that party higher than its peers do; to
        the left, lower. These are differences <em>between</em> pollsters, so
        they say nothing about whether the industry as a whole is off.</p>
      {_legend(data)}
      {_house_chart(data)}
    </section>
    """


def _health_section(data: dict[str, Any]) -> str:
    diagnostics = data.get("diagnostics") or {}
    converged = diagnostics.get("converged")
    issues = diagnostics.get("issues") or []
    notes = diagnostics.get("notes") or []
    warnings = data.get("warnings") or []
    if converged is None:
        state, verdict = "warn", "Convergence could not be checked."
    elif converged and not issues:
        state, verdict = (
            "ok",
            "The sampler converged. Nothing in the fit looks broken.",
        )
    else:
        state, verdict = (
            "bad",
            "The sampler did NOT converge. Do not publish these numbers.",
        )
    bullets = "".join(f"<li>{_esc(item)}</li>" for item in issues + notes + warnings)
    detail = f"<ul>{bullets}</ul>" if bullets else ""
    stats = []
    if diagnostics.get("max_r_hat") is not None:
        stats.append(f"R-hat {diagnostics['max_r_hat']:.3f} (want under 1.01)")
    if diagnostics.get("min_ess_bulk") is not None:
        stats.append(
            f"effective sample size {diagnostics['min_ess_bulk']:.0f} (want 400+)"
        )
    if diagnostics.get("n_divergences") is not None:
        stats.append(f"{diagnostics['n_divergences']} divergent transitions (want 0)")
    return f"""
    <section>
      <h2>Model health</h2>
      <p class="verdict {state}">{_esc(verdict)}</p>
      <p class="note">{_esc(" · ".join(stats))}</p>
      {detail}
    </section>
    """


def _settings_section(data: dict[str, Any]) -> str:
    readback = "".join(
        f"<li>{_esc(line)}</li>" for line in data.get("settings_readback", [])
    )
    overview = "".join(
        f"<li>{_esc(line)}</li>" for line in data.get("data_overview", [])
    )
    return f"""
    <section>
      <h2>What this forecast was told</h2>
      <p class="note">A forecast is only as good as its inputs and its
        assumptions. Both are printed here so anyone reading the numbers can
        check them.</p>
      <div class="cols">
        <div><h3>Your settings</h3><ul>{readback}</ul></div>
        <div><h3>Your data</h3><ul>{overview}</ul></div>
      </div>
    </section>
    """


CAVEATS = """
    <section>
      <h2>What these numbers do and do not say</h2>
      <ul>
        <li><strong>"Chance of finishing first" is about vote share, not
          power.</strong> It is the probability that a party polls higher than
          every other on election day. Under runoffs, district seats, electoral
          colleges or coalition maths, that is not the probability of taking
          office.</li>
        <li><strong>The ranges are wide on purpose.</strong> The 90% range is
          where the vote share is expected to land nine times out of ten, given
          the polls and the assumptions listed above.</li>
        <li><strong>A bias shared by every pollster is invisible here.</strong>
          It cannot be measured from one election's polls. The break-even
          number above is the honest way to read that risk.</li>
        <li><strong>Nothing outside the polls is modelled.</strong> Turnout
          surprises, late scandals, and shifts in who answers surveys are not
          in the data and so are not in the forecast.</li>
      </ul>
    </section>
"""

STYLE = """
:root {
  color-scheme: light dark;
  --bg: #ffffff; --panel: #f7f8fa; --ink: #14171f; --muted: #5b6472;
  --line: #dfe3ea; --accent: #14171f;
  --ok: #0f7b3f; --warn: #a25c00; --bad: #b02020;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14161b; --panel: #1c1f26; --ink: #eef1f6; --muted: #9aa4b2;
    --line: #2b303a; --accent: #eef1f6;
    --ok: #4ec27f; --warn: #e0a44a; --bad: #ff7b7b;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        Helvetica, Arial, sans-serif;
}
main { max-width: 980px; margin: 0 auto; padding: 40px 22px 80px; }
h1 { font-size: 2.1rem; line-height: 1.2; margin: 6px 0 14px; letter-spacing: -0.02em; }
h2 { font-size: 1.25rem; margin: 0 0 6px; letter-spacing: -0.01em; }
h3 { font-size: 0.95rem; margin: 0 0 6px; color: var(--muted);
     text-transform: uppercase; letter-spacing: 0.06em; }
section { border-top: 1px solid var(--line); padding: 26px 0; }
section.headline { border-top: none; padding-top: 0; }
.kicker { color: var(--muted); font-size: 0.9rem; margin: 0; }
.lede { font-size: 1.05rem; max-width: 62ch; }
.note { color: var(--muted); font-size: 0.9rem; max-width: 70ch; }
.chart { width: 100%; height: auto; display: block; margin: 10px 0 4px; }
.grid { stroke: var(--line); stroke-width: 1; }
.marker { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 4 3; }
.tick { fill: var(--muted); font-size: 11px; }
.tick.mid { text-anchor: middle; }
.tick.end { text-anchor: end; }
.tick.start { text-anchor: start; }
.rowlabel { fill: var(--ink); font-size: 13px; text-anchor: end; }
.rowvalue { fill: var(--muted); font-size: 12px; }
.bars { margin: 12px 0 4px; }
.barrow { display: grid; grid-template-columns: 150px 1fr 60px;
          align-items: center; gap: 12px; margin: 7px 0; }
.barname { font-size: 0.95rem; }
.bartrack { background: var(--panel); border-radius: 4px; height: 22px; }
.barfill { height: 22px; border-radius: 4px; }
.barvalue { text-align: right; font-variant-numeric: tabular-nums; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 10px 0 0;
          font-size: 0.85rem; color: var(--muted); }
.key i { display: inline-block; width: 11px; height: 11px; border-radius: 3px;
         margin-right: 6px; vertical-align: -1px; }
table { border-collapse: collapse; width: 100%; margin-top: 12px;
        font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--line); }
thead th { color: var(--muted); font-weight: 600; font-size: 0.85rem; }
tbody th { text-align: left; font-weight: 600; }
.verdict { font-weight: 600; margin: 4px 0; }
.verdict.ok { color: var(--ok); }
.verdict.warn { color: var(--warn); }
.verdict.bad { color: var(--bad); }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 26px; }
.cols ul { padding-left: 18px; font-size: 0.9rem; }
ul { max-width: 72ch; }
footer { color: var(--muted); font-size: 0.82rem; border-top: 1px solid var(--line);
         padding-top: 16px; }
.scroll { overflow-x: auto; }
@media (max-width: 680px) {
  .cols { grid-template-columns: 1fr; }
  .barrow { grid-template-columns: 96px 1fr 52px; }
}
"""


def render(data: dict[str, Any]) -> str:
    """Build the complete HTML document."""
    title = (
        data["election"].get("name") or f"Election forecast {data['election']['date']}"
    )
    election_ranges = _range_chart(data["election_day_estimates"], data["colors"])
    today_ranges = _range_chart(data["today_estimates"], data["colors"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{STYLE}</style></head>
<body><main>
{_headline(data)}
<section>
  <h2>Chance of finishing first</h2>
  <p class="note">Share of simulated election days on which each party polls
    higher than every other party.</p>
  {_probability_bars(data)}
</section>
<section>
  <h2>Election-day vote share</h2>
  <p class="note">Dot is the central estimate; the bar is the 90% range.
    Overlapping bars mean the race is genuinely open.</p>
  <div class="scroll">{election_ranges}</div>
</section>
<section>
  <h2>Where support stands today</h2>
  <p class="note">The same picture for {_esc(data["election"]["as_of"])}, before
    any drift between now and election day is added.</p>
  <div class="scroll">{today_ranges}</div>
</section>
<section>
  <h2>Support over time</h2>
  <p class="note">Lines are the estimated true level of support; shaded bands
    are the 90% range; dots are the individual polls behind it.</p>
  {_legend(data)}
  <div class="scroll">{_trend_chart(data)}</div>
</section>
{_threshold_section(data)}
{_scenario_section(data)}
{_house_section(data)}
{_health_section(data)}
{_settings_section(data)}
{CAVEATS}
<footer>Produced by kronikas {_esc(data.get("kronikas_version", ""))} on
{_esc(data.get("generated_at", ""))}. Rebuild this page from
report_data.json without refitting the model.</footer>
</main></body></html>
"""


def build_from_data(data: dict[str, Any], out_path: str | Path) -> Path:
    """Render an already-loaded payload to *out_path*."""
    out_path = Path(out_path)
    out_path.write_text(render(data), encoding="utf-8")
    return out_path


def build(data_path: str | Path, out_path: str | Path | None = None) -> Path:
    """Read ``report_data.json`` and write ``report.html`` beside it."""
    data_path = Path(data_path)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    return build_from_data(data, out_path or data_path.with_name("report.html"))
