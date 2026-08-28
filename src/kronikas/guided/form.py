"""A browser form for writing the settings file.

Some of the interview is a poor fit for a conversation: a lean per party per
pollster is a dozen small numbers, and typing them one question at a time is
miserable for everyone.  This builds a page with a real control for every
party and firm found in the user's own poll file — radio buttons, steppers, a
slider — with the resulting YAML updating live and a button to save it.

Self-contained: inline CSS and JS, no network, nothing uploaded.  The page is
generated from the poll file, so it can only offer names that actually exist,
which removes the most common settings error before it happens.
"""

from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path

from ..data import PollData

# Reuse the report's palette so the two pages look like one product.
from .runner import DEFAULT_COLORS

STYLE = """
:root {
  color-scheme: light dark;
  --bg: #ffffff; --panel: #f7f8fa; --ink: #14171f; --muted: #5b6472;
  --line: #dfe3ea; --accent: #0072b2; --accent-ink: #ffffff;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14161b; --panel: #1c1f26; --ink: #eef1f6; --muted: #9aa4b2;
    --line: #2b303a; --accent: #4c9fd8; --accent-ink: #0d0f13;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        Helvetica, Arial, sans-serif;
}
main { max-width: 900px; margin: 0 auto; padding: 36px 22px 260px; }
h1 { font-size: 1.9rem; margin: 0 0 8px; letter-spacing: -0.02em; }
h2 { font-size: 1.15rem; margin: 0 0 4px; letter-spacing: -0.01em; }
h3 { font-size: 0.95rem; margin: 14px 0 6px; }
section { border-top: 1px solid var(--line); padding: 24px 0; }
section:first-of-type { border-top: none; }
.note { color: var(--muted); font-size: 0.9rem; max-width: 68ch; margin: 4px 0 14px; }
.field { margin: 14px 0; }
label.lead { display: block; font-weight: 600; margin-bottom: 6px; }
input[type=text], input[type=date], input[type=number] {
  font: inherit; padding: 8px 10px; border: 1px solid var(--line);
  border-radius: 7px; background: var(--bg); color: var(--ink); max-width: 260px;
}
input[type=number] { width: 92px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { position: relative; }
.chip input { position: absolute; opacity: 0; inset: 0; cursor: pointer; }
.chip span {
  display: block; padding: 7px 14px; border: 1px solid var(--line);
  border-radius: 999px; cursor: pointer; background: var(--panel);
  font-size: 0.92rem;
}
.chip input:checked + span {
  background: var(--accent); color: var(--accent-ink); border-color: var(--accent);
  font-weight: 600;
}
.chip input:focus-visible + span {
  outline: 2px solid var(--accent); outline-offset: 2px;
}
.hint { color: var(--muted); font-size: 0.85rem; margin-top: 6px; max-width: 62ch; }
.card {
  border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px;
  margin: 12px 0; background: var(--panel);
}
.card h3 { margin-top: 0; display: flex; align-items: center; gap: 8px; }
.leans { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px; }
.lean { display: flex; align-items: center; gap: 7px; font-size: 0.9rem; }
.swatch { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }
input[type=range] { width: 100%; max-width: 420px; accent-color: var(--accent); }
.slider-value { font-weight: 600; margin-left: 10px; }
.out {
  position: fixed; left: 0; right: 0; bottom: 0; background: var(--panel);
  border-top: 1px solid var(--line); padding: 12px 22px 16px; max-height: 42vh;
  overflow: auto;
}
.out-inner { max-width: 900px; margin: 0 auto; }
.out header { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
.out h2 { flex: 1; font-size: 0.95rem; margin: 0; }
button {
  font: inherit; font-weight: 600; padding: 8px 14px; border-radius: 8px;
  border: 1px solid var(--accent); background: var(--accent);
  color: var(--accent-ink); cursor: pointer;
}
button.secondary {
  background: transparent; color: var(--ink); border-color: var(--line);
}
pre {
  margin: 0; font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  white-space: pre-wrap; color: var(--ink);
}
.summary { background: var(--panel); border-radius: 10px; padding: 12px 16px;
           font-size: 0.9rem; color: var(--muted); }
@media (max-width: 620px) { main { padding-bottom: 320px; } }
"""

SCRIPT = """
const q = (id) => document.getElementById(id);
const val = (name) => {
  const picked = document.querySelector(`input[name="${name}"]:checked`);
  return picked ? picked.value : "";
};
const num = (id) => {
  const raw = q(id) ? q(id).value.trim() : "";
  if (raw === "") return null;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : null;
};

function pairs(entries) {
  // Quote the key: a party named "Something, Inc" would otherwise split the
  // flow mapping at its own comma.
  return (
    "{" +
    entries.map(([k, v]) => `${JSON.stringify(k)}: ${v}`).join(", ") +
    "}"
  );
}

function buildYaml() {
  const lines = [];
  lines.push("election:");
  const name = q("election-name").value.trim();
  if (name) lines.push(`  name: ${JSON.stringify(name)}`);
  lines.push(`  date: ${q("election-date").value || "YYYY-MM-DD"}`);
  const asOf = q("as-of").value;
  if (asOf) lines.push(`  as_of: ${asOf}`);

  lines.push("");
  lines.push("polls:");
  lines.push(`  file: ${POLL_FILE}`);
  if (DECIMAL !== ".") lines.push(`  decimal: ${JSON.stringify(DECIMAL)}`);

  const beliefs = ["", "beliefs:", `  volatility: ${val("volatility")}`];

  const firmBlocks = [];
  for (const firm of POLLSTERS) {
    const key = firm.id;
    const trust = val(`trust-${key}`);
    const noise = val(`noise-${key}`);
    const leans = [];
    for (const party of PARTIES) {
      const lean = num(`lean-${key}-${party.id}`);
      if (lean) leans.push([party.name, lean]);
    }
    if (trust === "normal" && noise === "normal" && leans.length === 0) continue;
    const block = [`    ${JSON.stringify(firm.name)}:`];
    if (leans.length) block.push(`      leans: ${pairs(leans)}`);
    if (trust !== "normal") block.push(`      trust: ${trust}`);
    if (noise !== "normal") block.push(`      noisiness: ${noise}`);
    firmBlocks.push(block.join("\\n"));
  }
  if (firmBlocks.length) {
    beliefs.push("  pollsters:");
    beliefs.push(firmBlocks.join("\\n"));
  }

  const spread = Number(q("industry").value);
  const expected = [];
  for (const party of PARTIES) {
    const lean = num(`industry-${party.id}`);
    if (lean) expected.push([party.name, lean]);
  }
  if (spread > 0 || expected.length) {
    beliefs.push("  industry_error:");
    if (expected.length) beliefs.push(`    expected: ${pairs(expected)}`);
    beliefs.push(`    uncertainty_pp: ${spread}`);
  }
  lines.push(...beliefs);

  lines.push("");
  lines.push("run:");
  lines.push(`  effort: ${val("effort")}`);

  lines.push("");
  lines.push("report:");
  lines.push("  output_dir: forecast-output");
  const threshold = num("threshold");
  if (threshold) lines.push(`  thresholds_pp: [${threshold}]`);
  const scenarios = Array.from(
    document.querySelectorAll('input[name="scenario"]:checked')
  ).map((el) => el.value);
  if (scenarios.length) {
    lines.push(`  industry_error_scenarios_pp: [${scenarios.join(", ")}]`);
  }
  return lines.join("\\n") + "\\n";
}

function refresh() {
  q("yaml").textContent = buildYaml();
  const spread = Number(q("industry").value);
  q("industry-value").textContent =
    spread === 0 ? "0 pp — assumes the industry is collectively perfect"
                 : `${spread} pp`;
  q("industry-hint").textContent =
    spread === 0
      ? "That is a strong claim, and the one behind most famous forecasting misses."
      : spread < 2
      ? "Lower than the industry-wide error seen in recent comparable elections."
      : spread <= 3
      ? "In line with the industry-wide error seen in recent elections."
      : "Cautious — appropriate where polling has a track record of missing.";
}

document.addEventListener("input", refresh);
document.addEventListener("change", refresh);

q("copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(q("yaml").textContent);
    q("copy").textContent = "Copied";
    setTimeout(() => (q("copy").textContent = "Copy"), 1500);
  } catch (err) {
    q("copy").textContent = "Select the text below";
  }
});

q("download").addEventListener("click", () => {
  const blob = new Blob([q("yaml").textContent], { type: "text/yaml" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "forecast.yaml";
  link.click();
  URL.revokeObjectURL(link.href);
});

refresh();
"""


def _esc(value: object) -> str:
    return html.escape(str(value))


def _js(value: object) -> str:
    """JSON for embedding in a <script> block.

    A party literally named ``</script>`` would otherwise close the block and
    let the rest of the name run as markup, so the characters that can start a
    tag are escaped to their JSON unicode form: still valid JSON, inert HTML.
    """
    return (
        json.dumps(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _slug(value: str) -> str:
    """A safe DOM id fragment for an arbitrary party or pollster name."""
    return "".join(char if char.isalnum() else "-" for char in value)


def _chips(name: str, options: list[tuple[str, str]], selected: str) -> str:
    return (
        '<div class="chips">'
        + "".join(
            f'<label class="chip"><input type="radio" name="{_esc(name)}" '
            f'value="{_esc(value)}"{" checked" if value == selected else ""}>'
            f"<span>{_esc(label)}</span></label>"
            for value, label in options
        )
        + "</div>"
    )


def _pollster_card(firm: str, parties: list[str], colors: dict[str, str]) -> str:
    key = _slug(firm)
    leans = "".join(
        f'<label class="lean"><span class="swatch" style="background:'
        f'{_esc(colors[party])}"></span>{_esc(party)}'
        f'<input type="number" id="lean-{_esc(key)}-{_esc(_slug(party))}" '
        f'step="0.5" placeholder="0"></label>'
        for party in parties
    )
    return f"""
    <div class="card">
      <h3>{_esc(firm)}</h3>
      <div class="field">
        <label class="lead">How much do you trust it?</label>
        {
        _chips(
            f"trust-{key}",
            [
                ("normal", "Normal"),
                ("high", "High — hold it close to the others"),
                ("low", "Low — let it stray, and count it for less"),
            ],
            "normal",
        )
    }
      </div>
      <div class="field">
        <label class="lead">Do its numbers bounce more than its sample size
          explains?</label>
        {
        _chips(
            f"noise-{key}",
            [
                ("normal", "No"),
                ("high", "Yes — cheap online panel"),
                ("low", "Unusually steady"),
            ],
            "normal",
        )
    }
      </div>
      <div class="field">
        <label class="lead">Known lean, in percentage points
          (positive = reports that party too high)</label>
        <div class="leans">{leans}</div>
      </div>
    </div>
    """


def render(
    poll_data: PollData,
    *,
    election_date: date | None = None,
    polls_filename: str = "polls.csv",
    decimal: str = ".",
) -> str:
    """Build the settings-builder page for a specific poll file."""
    parties = list(poll_data.candidates)
    pollsters = list(poll_data.pollsters)
    colors = {
        party: DEFAULT_COLORS[index % len(DEFAULT_COLORS)]
        for index, party in enumerate(parties)
    }
    industry_rows = "".join(
        f'<label class="lean"><span class="swatch" style="background:'
        f'{_esc(colors[party])}"></span>{_esc(party)}'
        f'<input type="number" id="industry-{_esc(_slug(party))}" step="0.5" '
        f'placeholder="0"></label>'
        for party in parties
    )

    def _names(values: list[str]) -> str:
        return _js([{"name": value, "id": _slug(value)} for value in values])

    constants = (
        f"const PARTIES = {_names(parties)};\n"
        f"const POLLSTERS = {_names(pollsters)};\n"
        f"const POLL_FILE = {_js(polls_filename)};\n"
        f"const DECIMAL = {_js(decimal)};\n"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Set up your forecast</title>
<style>{STYLE}</style></head>
<body><main>
<h1>Set up your forecast</h1>
<p class="note">Every control below was built from your own poll file, so the
names match what is actually in it. Nothing is uploaded anywhere — this page
runs entirely in your browser. When you are done, download the file and run
<code>kronikas guided forecast.yaml</code>.</p>
<p class="summary">Read {len(poll_data.dates)} polls from
<strong>{_esc(polls_filename)}</strong>, {poll_data.first_poll_date} to
{poll_data.last_poll_date}. Parties: {_esc(", ".join(parties))}.
Pollsters: {_esc(", ".join(pollsters))}.</p>

<section>
  <h2>The race</h2>
  <div class="field">
    <label class="lead" for="election-date">Election day — when the votes are
      counted</label>
    <input type="date" id="election-date"
           value="{_esc(election_date.isoformat() if election_date else "")}">
  </div>
  <div class="field">
    <label class="lead" for="election-name">What to call it on the report
      (optional)</label>
    <input type="text" id="election-name" placeholder="General election 2026">
  </div>
  <div class="field">
    <label class="lead" for="as-of">Forecast as if it were … (optional)</label>
    <input type="date" id="as-of">
    <p class="hint">Leave empty to use today. Set an earlier date to ignore
      every poll after it — that is how you replay a race that already
      happened.</p>
  </div>
</section>

<section>
  <h2>How fast does opinion move?</h2>
  <p class="note">This is what the model assumes before it sees a single poll.
    Normal allows roughly a point a week.</p>
  {
        _chips(
            "volatility",
            [
                ("normal", "Normal"),
                ("volatile", "Volatile — a campaign in motion"),
                ("calm", "Calm — entrenched, quiet"),
            ],
            "normal",
        )
    }
</section>

<section>
  <h2>What you know about the polling firms</h2>
  <p class="note">Leave every one of these alone unless you know something
    from <em>outside</em> these polls. The model already works out how the
    firms differ from each other; this is for what it cannot see.</p>
  {"".join(_pollster_card(firm, parties, colors) for firm in pollsters)}
</section>

<section>
  <h2>How wrong could every firm be at once?</h2>
  <p class="note">When all the pollsters miss in the same direction, no model
    can detect it: they agree with each other, and agreement reads as
    accuracy — so the forecast gets <em>more</em> confident, not less. This is
    the one number that has to come from you.</p>
  <input type="range" id="industry" min="0" max="5" step="0.5" value="2.5">
  <span class="slider-value" id="industry-value">2.5 pp</span>
  <p class="hint" id="industry-hint"></p>
  <h3>And if you think they lean a particular way (optional)</h3>
  <p class="hint">Percentage points, positive meaning the polls have that
    party too high.</p>
  <div class="leans">{industry_rows}</div>
</section>

<section>
  <h2>Running it</h2>
  <div class="field">
    <label class="lead">How careful should the run be?</label>
    {
        _chips(
            "effort",
            [
                ("standard", "Standard — a few minutes"),
                ("quick", "Quick — about a minute, coarser"),
                ("thorough", "Thorough — slower, for close races"),
            ],
            "standard",
        )
    }
  </div>
  <div class="field">
    <label class="lead" for="threshold">Seat threshold, if the system has one
      (optional)</label>
    <input type="number" id="threshold" step="0.5" placeholder="5">
    <p class="hint">The share a party must clear to win any seats at all.</p>
  </div>
  <div class="field">
    <label class="lead">Show what a polling error of this size would do</label>
    <div class="chips">
      <label class="chip"><input type="checkbox" name="scenario" value="1" checked>
        <span>1 pp</span></label>
      <label class="chip"><input type="checkbox" name="scenario" value="2" checked>
        <span>2 pp</span></label>
      <label class="chip"><input type="checkbox" name="scenario" value="3" checked>
        <span>3 pp</span></label>
    </div>
  </div>
</section>
</main>

<div class="out"><div class="out-inner">
  <header>
    <h2>forecast.yaml</h2>
    <button class="secondary" id="copy" type="button">Copy</button>
    <button id="download" type="button">Download</button>
  </header>
  <pre id="yaml"></pre>
</div></div>

<script>
{constants}{SCRIPT}
</script>
</body></html>
"""


def build(
    poll_data: PollData,
    out_path: str | Path,
    *,
    election_date: date | None = None,
    polls_filename: str = "polls.csv",
    decimal: str = ".",
) -> Path:
    """Write the settings-builder page to *out_path*."""
    out_path = Path(out_path)
    out_path.write_text(
        render(
            poll_data,
            election_date=election_date,
            polls_filename=polls_filename,
            decimal=decimal,
        ),
        encoding="utf-8",
    )
    return out_path
