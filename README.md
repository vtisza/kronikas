# kronikas — project site

**This branch is not part of the codebase.** It is an orphan branch: it
shares no history with `main` and contains no package source, tests or CI.
It holds only the public landing page for
[kronikas](https://github.com/vtisza/kronikas), served by GitHub Pages at
<https://vtisza.github.io/kronikas/>.

Keeping the site here rather than on `main` means the page never reaches the
package: it is not linted by `ruff check .`, not type-checked, and not
bundled into the sdist that goes to PyPI.

## Publishing

Repository **Settings → Pages → Build and deployment**:

- Source: *Deploy from a branch*
- Branch: `gh-pages` / `(root)`

Every push to this branch republishes the site. There is no build step and
no workflow — `index.html` is served exactly as committed. `.nojekyll` stops
GitHub from running the page through Jekyll.

## Layout

```
index.html                the page; chart geometry is inlined
assets/styles.css         all styling
assets/logo-mark.svg      the logo mark on its own, for reuse
assets/favicon.svg        the mark on an ink tile, for browser tabs
assets/og-image.png       1200x630 social preview card
assets/report-preview.png the guided workflow's report, in section 05
tools/generate_hero_chart.py   regenerates the hero fan chart
```

Working on it locally: open `index.html` in a browser, or
`python -m http.server` from this directory. What you see is what ships.

## The mark

A stem and two arms that read as a **K** and, at the same time, as a fan
chart: the stem is the fixed record of polls already taken, and the arms are
posterior trajectories diverging into the future. The mark is inlined in
`index.html` as an SVG `<symbol>` so its stem picks up `currentColor` and
inverts cleanly on the dark sections.

Palette: ink `#131a24`, paper `#f6f1e7`, vermillion `#e04e2a` (the trend),
teal `#1e8e86` and gold `#d9a03c` (the other series). Display face is
Fraunces, body Inter, code JetBrains Mono.

## Regenerating the hero chart

The fan chart is synthetic data — plausible poll scatter with credible
intervals that tighten where polls constrain them and widen toward election
day — generated from a fixed seed rather than drawn by hand:

```bash
python tools/generate_hero_chart.py > frag.svg
```

Paste the output between the `HERO CHART BEGIN` and `HERO CHART END`
comments in `index.html`. The script prints the axis and label coordinates
to stderr; the surrounding gridlines, markers and end-of-line values in the
HTML are hand-written and must be kept in step with them.

Nothing on the page is a forecast of any real election, and the figure says
so in its caption.

## Regenerating the report preview

`assets/report-preview.png` in section 05 is a real run, not a mockup — the
sample polls that ship with the package, forecast by the guided workflow:

```bash
pip install kronikas
cp "$(kronikas skill path)"/assets/{polls.example.csv,forecast.example.yaml} .
kronikas guided forecast.example.yaml
```

Screenshot the top of the resulting `report.html` at 1180x880 (capture at
2x and downscale, so the type stays crisp), cropping just below the
election-day range chart. Because it is a real run the numbers move between
runs; the caption claims only that it is a real run, so that is fine.

## Keeping it honest

Because this branch is never reviewed alongside a code change, the page can
drift from what the package actually does. Worth re-reading after anything
that changes the public API, the CLI flags, or the claims in the README —
the code samples and terminal output on the page are meant to match real
`kronikas` behaviour.
