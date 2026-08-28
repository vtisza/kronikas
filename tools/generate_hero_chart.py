#!/usr/bin/env python3
"""Regenerate the fan-chart geometry embedded in the landing page hero.

The hero figure in ``index.html`` is a synthetic three-party forecast:
observed polls scattered around a latent trend, with credible intervals that
stay tight where polls constrain them and fan out toward election day.  The
geometry is generated rather than hand-drawn so the curve shapes are plausible
and the whole thing is reproducible from a seed.

Usage::

    python tools/generate_hero_chart.py > frag.svg

Then replace everything between the ``HERO CHART BEGIN`` and ``HERO CHART END``
comments in ``index.html`` with the output.  The surrounding axes, grid,
markers and end-of-line labels are hand-written in the HTML; if you change
``W``, ``H``, ``X_NOW`` or the y-scale below, update those to match — the
script prints the relevant coordinates to stderr.

Nothing here is a forecast of any real election.
"""

from __future__ import annotations

import math
import random
import sys

W, H = 1200.0, 520.0  # plot area, in the SVG user units of the hero viewBox
N = 81  # vertices per path
X_NOW = 0.58  # fraction of the width where "today" sits
PAD_T, PAD_B = 34.0, 46.0
Y_MIN, Y_MAX = 4.0, 52.0  # vote-share window, in points
SEED = 20240315

BANDS = ((1.0, "90"), (0.6, "70"), (0.3, "50"))
SERIES = (
    {"key": "flame", "start": 41.0, "step": 0.42, "drift": +2.6, "width": 1.5},
    {"key": "tide", "start": 36.5, "step": 0.38, "drift": -1.4, "width": 1.5},
    {"key": "gold", "start": 13.5, "step": 0.22, "drift": -0.7, "width": 0.9},
)
# The three shares are renormalised to this total; the remainder stands in for
# other parties and undecideds.
TOTAL = 91.0


def smooth(values: list[float], passes: int = 3, half: int = 2) -> list[float]:
    """Box-smooth a sequence, so the walk reads as a trend rather than noise."""
    n = len(values)
    for _ in range(passes):
        values = [
            sum(values[max(0, i - half) : min(n, i + half + 1)])
            / len(values[max(0, i - half) : min(n, i + half + 1)])
            for i in range(n)
        ]
    return values


def walk(rng: random.Random, start: float, n: int, step: float, pull: float = 0.06):
    """A mean-reverting Gaussian random walk, smoothed."""
    out, v = [], start
    for _ in range(n):
        v += rng.gauss(0, step) + pull * (start - v)
        out.append(v)
    return smooth(out)


def sx(i: int) -> float:
    return i / (N - 1) * W


def sy(v: float) -> float:
    return PAD_T + (Y_MAX - v) / (Y_MAX - Y_MIN) * (H - PAD_T - PAD_B)


def fmt(x: float) -> str:
    return f"{x:.0f}" if abs(x - round(x)) < 0.05 else f"{x:.1f}"


def line_path(points) -> str:
    """Catmull-Rom through the points, emitted as cubic Beziers."""
    d = [f"M{fmt(points[0][0])},{fmt(points[0][1])}"]
    last = len(points) - 1
    for i in range(last):
        p0 = points[max(0, i - 1)]
        p1, p2 = points[i], points[i + 1]
        p3 = points[min(last, i + 2)]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d.append(
            f"C{fmt(c1[0])},{fmt(c1[1])} {fmt(c2[0])},{fmt(c2[1])}"
            f" {fmt(p2[0])},{fmt(p2[1])}"
        )
    return "".join(d)


def main() -> None:
    rng = random.Random(SEED)
    i_now = round(X_NOW * (N - 1))

    series = [dict(s) for s in SERIES]
    for s in series:
        base = walk(rng, s["start"], N, s["step"])
        s["mean"] = [v + s["drift"] * (i / (N - 1)) ** 1.6 for i, v in enumerate(base)]

    for i in range(N):
        total = sum(s["mean"][i] for s in series)
        for s in series:
            s["mean"][i] = s["mean"][i] * TOTAL / total

    def halfwidth(i: int, base: float) -> float:
        """Interval half-width: tight where polls pin it, widening after today."""
        past = base * (0.55 + 0.45 * math.exp(-i / 26.0))
        ahead = max(0, i - i_now) / (N - 1 - i_now)
        return past + base * 2.5 * math.sqrt(ahead)

    out: list[str] = ['<g class="bands">']
    for s in series:
        for scale, name in BANDS:
            top = [
                (sx(i), sy(s["mean"][i] + halfwidth(i, s["width"]) * scale))
                for i in range(N)
            ]
            bottom = [
                (sx(i), sy(s["mean"][i] - halfwidth(i, s["width"]) * scale))
                for i in range(N)
            ]
            d = line_path(top) + "L" + line_path(bottom[::-1])[1:] + "Z"
            out.append(
                f'<path class="ci ci--{name}" fill="var(--c-{s["key"]})" d="{d}"/>'
            )
    for s in series:
        points = [(sx(i), sy(s["mean"][i])) for i in range(N)]
        colour = f"var(--c-{s['key']})"
        out.append(f'<path class="trend" stroke="{colour}" d="{line_path(points)}"/>')
    out.append("</g>")

    out.append('<g class="polls">')
    for s in series:
        r = 2.9 if s["key"] != "gold" else 2.4
        for i in range(2, i_now, 2):
            if rng.random() > 0.62:
                continue
            jitter = rng.gauss(0, 1.35 if s["key"] != "gold" else 0.8)
            cx = sx(i) + rng.uniform(-4, 4)
            cy = sy(s["mean"][i] + jitter)
            out.append(
                f'<circle class="poll" fill="var(--c-{s["key"]})" '
                f'cx="{fmt(cx)}" cy="{fmt(cy)}" r="{r}"/>'
            )
    out.append("</g>")

    print("\n".join(out))

    print(f"today x={fmt(sx(i_now))}  election-day x={fmt(W)}", file=sys.stderr)
    for v in (50, 40, 30, 20, 10):
        print(f"  gridline {v}% -> y={sy(v):.1f}", file=sys.stderr)
    for s in series:
        hw = halfwidth(N - 1, s["width"])
        print(
            f"  {s['key']:6s} election day {s['mean'][-1]:.1f}%"
            f" (90% {s['mean'][-1] - hw:.1f}-{s['mean'][-1] + hw:.1f})"
            f" label y={sy(s['mean'][-1]):.1f}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
