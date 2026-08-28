"""Tests for the guided (non-technical) forecasting workflow.

A settings file a non-programmer can fill in, a runner, a self-contained HTML
report, and the packaged skill an AI assistant follows.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from kronikas.guided import report as make_report
from kronikas.guided import settings, skill

ASSETS = Path(settings.__file__).resolve().parent.parent / "skill" / "assets"


# --- the YAML subset -------------------------------------------------------


def test_parses_nested_blocks_lists_and_flow_collections():
    parsed = settings.parse_simple_yaml(
        """
# a comment
election:
  date: 2026-04-12
  name: "Spring vote"      # trailing comment
beliefs:
  volatility: volatile
  industry_error:
    expected: {PartyA: 2, PartyB: -2}
    uncertainty_pp: 2.5
report:
  thresholds_pp: [5, 10]
  colors:
    - one
    - two
run:
  seed: 7
  progress: false
  effort: null
"""
    )
    assert parsed["election"] == {"date": "2026-04-12", "name": "Spring vote"}
    assert parsed["beliefs"]["industry_error"]["expected"] == {
        "PartyA": 2,
        "PartyB": -2,
    }
    assert parsed["beliefs"]["industry_error"]["uncertainty_pp"] == 2.5
    assert parsed["report"]["thresholds_pp"] == [5, 10]
    assert parsed["report"]["colors"] == ["one", "two"]
    assert parsed["run"] == {"seed": 7, "progress": False, "effort": None}


def test_hash_inside_a_quoted_value_is_not_a_comment():
    parsed = settings.parse_simple_yaml('report:\n  party_colors: {A: "#ff0000"}\n')
    assert parsed["report"]["party_colors"] == {"A": "#ff0000"}


def test_shipped_settings_files_parse_and_normalise():
    for name in ("forecast.template.yaml", "forecast.example.yaml"):
        plan = settings.normalise(
            settings.read_settings_file(ASSETS / name), base_dir=ASSETS
        )
        assert plan.election_date.year >= 2026


# --- normalisation ---------------------------------------------------------


def _plan(text: str, tmp_path: Path) -> object:
    path = tmp_path / "forecast.yaml"
    path.write_text(text, encoding="utf-8")
    return settings.load_plan(path)


MINIMAL = "election:\n  date: 2026-04-12\npolls:\n  file: polls.csv\n"


def test_minimal_settings_get_documented_defaults(tmp_path):
    plan = _plan(MINIMAL, tmp_path)
    assert plan.election_date == date(2026, 4, 12)
    assert plan.polls_path == tmp_path / "polls.csv"  # relative to the file
    assert plan.effort == "standard"
    assert plan.volatility == "normal"
    assert plan.sigma_walk_prior == settings.VOLATILITY_PRESETS["normal"]
    assert plan.time_step_days == 7
    assert plan.seed == 42
    assert plan.industry_uncertainty_pp == 0.0
    assert plan.output_dir == tmp_path / "forecast-output"


def test_word_presets_resolve_to_numbers(tmp_path):
    plan = _plan(
        MINIMAL
        + """
beliefs:
  volatility: volatile
  pollsters:
    Meridian:
      trust: high
      noisiness: high
      leans: {PartyA: 2.5}
run:
  effort: thorough
""",
        tmp_path,
    )
    assert plan.sigma_walk_prior == settings.VOLATILITY_PRESETS["volatile"]
    assert plan.sampler["num_draws"] == 2000
    assert plan.sampler["num_chains"] == 4
    belief = plan.pollsters["Meridian"]
    assert belief.sigma_house == settings.TRUST_PRESETS["high"]
    assert belief.kappa_log_sigma == settings.NOISINESS_PRESETS["high"]
    assert belief.leans == {"PartyA": 2.5}
    assert plan.named_parties == {"PartyA"}


def test_a_number_may_replace_a_word_preset(tmp_path):
    plan = _plan(MINIMAL + "beliefs:\n  volatility: 0.09\n", tmp_path)
    assert plan.volatility == "custom"
    assert plan.sigma_walk_prior == 0.09


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("polls:\n  file: p.csv\n", "election.date is required"),
        ("election:\n  date: 2026-04-12\n", "polls.file is required"),
        (MINIMAL + "beleifs:\n  volatility: calm\n", "unknown setting"),
        (MINIMAL + "beliefs:\n  volatility: sleepy\n", "beliefs.volatility"),
        (MINIMAL + "run:\n  effort: fastest\n", "run.effort"),
        ("election:\n  date: next April\npolls:\n  file: p.csv\n", "YYYY-MM-DD"),
        (
            "election:\n  date: 2026-04-12\n  as_of: 2026-05-01\n"
            "polls:\n  file: p.csv\n",
            "after election.date",
        ),
        (
            MINIMAL + "beliefs:\n  pollsters:\n    A:\n      leans: {P: 80}\n",
            "outside the allowed range",
        ),
        (
            MINIMAL + "beliefs:\n  industry_error:\n    uncertainty_pp: -1\n",
            "cannot be negative",
        ),
    ],
)
def test_bad_settings_are_refused_with_a_useful_message(text, fragment, tmp_path):
    with pytest.raises(settings.SettingsError, match=fragment):
        _plan(text, tmp_path)


# --- translation to kronikas objects ---------------------------------------


def test_beliefs_become_kronikas_priors(tmp_path):
    from kronikas import ModelConfig

    plan = _plan(
        MINIMAL
        + """
beliefs:
  pollsters:
    Meridian:
      leans: {PartyA: 3}
      trust: low
  industry_error:
    expected: {PartyA: 2}
    uncertainty_pp: 2.5
    party_uncertainty: {PartyB: 3.5}
run:
  effort: quick
  seed: 11
""",
        tmp_path,
    )
    config = settings.build_model_config(plan)
    assert isinstance(config, ModelConfig)
    assert config.num_draws == 1000
    assert config.random_seed == 11
    prior = config.pollster_priors["Meridian"]
    assert prior.mu_house == {"PartyA": 3.0}
    assert prior.sigma_house == settings.TRUST_PRESETS["low"]
    assert prior.kappa_log_sigma is None  # normal noisiness stays hierarchical
    assert config.shared_bias.mean == {"PartyA": 2.0}
    assert config.shared_bias.default_sd == 2.5
    assert config.shared_bias.sd == {"PartyB": 3.5}


def test_no_industry_belief_leaves_shared_bias_unset(tmp_path):
    config = settings.build_model_config(_plan(MINIMAL, tmp_path))
    assert config.shared_bias is None
    assert config.pollster_priors == {}


def test_advanced_block_reaches_model_config_and_rejects_typos(tmp_path):
    config = settings.build_model_config(
        _plan(
            MINIMAL + "advanced:\n  correlated_walk: true\n  num_draws: 123\n", tmp_path
        )
    )
    assert config.correlated_walk is True
    assert config.num_draws == 123

    with pytest.raises(settings.SettingsError, match="unknown setting"):
        settings.build_model_config(
            _plan(MINIMAL + "advanced:\n  corelated_walk: true\n", tmp_path)
        )


def test_readback_states_the_assumption_the_user_did_not_make(tmp_path):
    text = "\n".join(settings.describe(_plan(MINIMAL, tmp_path)))
    assert "cannot measure this from polls alone" in text
    assert "12 April 2026" in text


# --- the report ------------------------------------------------------------


def _payload() -> dict:
    return {
        "generated_at": "2026-03-06T10:00:00",
        "kronikas_version": "0.1.0",
        "election": {
            "name": "Sample election",
            "date": "2026-04-12",
            "as_of": "2026-03-06",
            "days_to_go": 37,
        },
        "parties": ["Progress", "Unity"],
        "pollsters": ["Meridian", "Northgate"],
        "colors": {"Progress": "#3366cc", "Unity": "#dc3912"},
        "today_estimates": [
            {
                "name": "Progress",
                "mean": 41.0,
                "median": 41.0,
                "ci_lower": 38.0,
                "ci_upper": 44.0,
            },
            {
                "name": "Unity",
                "mean": 39.0,
                "median": 39.0,
                "ci_lower": 36.0,
                "ci_upper": 42.0,
            },
        ],
        "election_day_estimates": [
            {
                "name": "Progress",
                "mean": 41.5,
                "median": 41.5,
                "ci_lower": 37.5,
                "ci_upper": 45.5,
            },
            {
                "name": "Unity",
                "mean": 39.5,
                "median": 39.5,
                "ci_lower": 36.0,
                "ci_upper": 43.0,
            },
        ],
        "win_probabilities": {"Progress": 0.68, "Unity": 0.32},
        "threshold_probabilities": {"5": {"Progress": 1.0, "Unity": 1.0}},
        "trend": {
            "dates": ["2026-02-01", "2026-03-01", "2026-04-12"],
            "series": {
                "Progress": {
                    "mean": [40, 41, 41.5],
                    "lo": [37, 38, 37.5],
                    "hi": [43, 44, 45.5],
                },
                "Unity": {
                    "mean": [40, 39.5, 39.5],
                    "lo": [37, 36, 36],
                    "hi": [43, 43, 43],
                },
            },
        },
        "polls": [
            {
                "date": "2026-02-10",
                "pollster": "Meridian",
                "sample_size": 1000,
                "shares": {"Progress": 41.0, "Unity": 39.0},
            }
        ],
        "house_effects": {
            "pollsters": ["Meridian", "Northgate"],
            "parties": ["Progress", "Unity"],
            "mean": [[0.8, -0.8], [-1.2, 1.2]],
            "lo": [[0.1, -1.5], [-2.0, 0.4]],
            "hi": [[1.5, -0.1], [-0.4, 2.0]],
        },
        "scenarios": [
            {
                "pp": 2.0,
                "win_probabilities": {"Progress": 0.3, "Unity": 0.7},
                "estimates": [],
            }
        ],
        "breakeven_pp": 1.4,
        "diagnostics": {
            "max_r_hat": 1.004,
            "min_ess_bulk": 900.0,
            "n_divergences": 0,
            "converged": True,
            "issues": [],
            "notes": [],
            "text": "",
        },
        "settings_readback": ["Election day: 12 April 2026"],
        "data_overview": ["41 polls"],
        "warnings": [],
    }


def test_report_is_one_self_contained_page(tmp_path):
    out = make_report.build_from_data(_payload(), tmp_path / "report.html")
    page = out.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>")
    # No external resources: it has to open offline, from an email attachment.
    assert "<script" not in page
    assert "http://" not in page and "https://" not in page.replace(
        "https://python.arviz.org", ""
    )
    assert "<svg" in page


def test_report_states_the_headline_and_its_caveats(tmp_path):
    page = make_report.build_from_data(_payload(), tmp_path / "r.html").read_text(
        encoding="utf-8"
    )
    assert "Progress leads" in page
    assert "68%" in page
    assert "1.4 points" in page  # break-even
    assert "is about vote share" in page  # plurality caveat
    assert "The sampler converged" in page


def test_report_flags_a_run_that_did_not_converge(tmp_path):
    payload = _payload()
    payload["diagnostics"].update(
        converged=False, issues=["R-hat 1.2 exceeds 1.01; chains have not mixed."]
    )
    page = make_report.build_from_data(payload, tmp_path / "r.html").read_text(
        encoding="utf-8"
    )
    assert "did NOT converge" in page
    assert "verdict bad" in page


def test_report_explains_a_missing_house_effect_section(tmp_path):
    payload = _payload()
    payload["house_effects"] = None
    payload["pollsters"] = ["Meridian"]
    page = make_report.build_from_data(payload, tmp_path / "r.html").read_text(
        encoding="utf-8"
    )
    assert "Only one pollster" in page


def test_party_names_are_escaped(tmp_path):
    payload = _payload()
    payload["parties"] = ["<script>x</script>", "Unity"]
    payload["colors"]["<script>x</script>"] = "#123456"
    page = make_report.build_from_data(payload, tmp_path / "r.html").read_text(
        encoding="utf-8"
    )
    assert "<script>x</script>" not in page
    assert "&lt;script&gt;" in page


def test_report_can_be_rebuilt_from_the_written_json(tmp_path):
    data_path = tmp_path / "report_data.json"
    data_path.write_text(json.dumps(_payload()), encoding="utf-8")
    out = make_report.build(data_path)
    assert out == tmp_path / "report.html"
    assert out.read_text(encoding="utf-8").count("<svg") >= 3


def test_example_polls_load_with_the_example_settings():
    from kronikas import load_polls

    plan = settings.normalise(
        settings.read_settings_file(ASSETS / "forecast.example.yaml"), base_dir=ASSETS
    )
    poll_data = load_polls(plan.polls_path, **plan.loader_kwargs)
    assert set(plan.named_parties) <= set(poll_data.candidates)
    assert set(plan.pollsters) <= set(poll_data.pollsters)
    assert len(poll_data.dates) > 20


# --- the packaged skill ----------------------------------------------------


def test_skill_ships_inside_the_package():
    root = skill.packaged_path()
    assert (root / "SKILL.md").is_file()
    assert (root / "assets" / "forecast.template.yaml").is_file()
    assert len(list((root / "references").glob("*.md"))) >= 5


def test_skill_frontmatter_is_usable_by_an_assistant():
    text = (skill.packaged_path() / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    frontmatter = text.split("---\n")[1]
    assert "name: election-forecast" in frontmatter
    # The description is what makes an assistant pick the skill up at all.
    assert "description:" in frontmatter
    assert "forecast" in frontmatter.split("description:")[1].lower()


def test_skill_install_copies_the_whole_tree(tmp_path):
    installed = skill.copy_to(tmp_path)
    assert installed == tmp_path / "election-forecast"
    assert (installed / "SKILL.md").is_file()
    assert (installed / "references" / "interview.md").is_file()
    assert (installed / "assets" / "polls.example.csv").is_file()


def test_skill_install_refuses_to_clobber_an_edited_copy(tmp_path):
    installed = skill.copy_to(tmp_path)
    (installed / "SKILL.md").write_text("my edits", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--force"):
        skill.copy_to(tmp_path)
    assert (installed / "SKILL.md").read_text(encoding="utf-8") == "my edits"

    skill.copy_to(tmp_path, force=True)
    assert "election-forecast" in (installed / "SKILL.md").read_text(encoding="utf-8")


# --- the commands the skill tells people to run ----------------------------


def _subcommands() -> set[str]:
    import argparse

    from kronikas.cli import build_parser

    actions = [
        action
        for action in build_parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    return set(actions[0].choices)


def test_every_command_the_skill_documents_actually_exists():
    """Guards against the docs drifting away from the CLI."""
    import re

    available = _subcommands()
    root = skill.packaged_path()
    documented = set()
    for path in [root / "SKILL.md", *sorted((root / "references").glob("*.md"))]:
        text = path.read_text(encoding="utf-8")
        documented |= set(
            re.findall(r"kronikas (guided|report|skill|forecast|backtest)\b", text)
        )
    assert documented, "the skill documents no commands at all"
    assert documented <= available


def test_guided_and_skill_commands_are_wired(tmp_path):
    parser = _subcommands()
    assert {"guided", "report", "skill"} <= parser


def test_cli_reports_a_bad_settings_file_without_a_traceback(tmp_path, capsys):
    from kronikas.cli import main

    bad = tmp_path / "forecast.yaml"
    bad.write_text("election:\n  date: whenever\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        main(["guided", str(bad)])
    assert exit_info.value.code == 2
    assert "YYYY-MM-DD" in capsys.readouterr().err


# --- the browser form ------------------------------------------------------


@pytest.fixture
def poll_data():
    from kronikas import load_polls

    return load_polls(ASSETS / "polls.example.csv")


def test_form_offers_a_control_for_every_party_and_pollster(poll_data, tmp_path):
    from kronikas.guided import form

    page = form.build(
        poll_data, tmp_path / "settings-builder.html", election_date=date(2026, 4, 12)
    ).read_text(encoding="utf-8")

    for firm in poll_data.pollsters:
        assert f'name="trust-{firm}"' in page
        assert f'name="noise-{firm}"' in page
        for party in poll_data.candidates:
            assert f'id="lean-{firm}-{party}"' in page
    for party in poll_data.candidates:
        assert f'id="industry-{party}"' in page
    assert 'value="2026-04-12"' in page
    # Self-contained: it has to work from a file:// URL with no network.
    assert "http://" not in page and "https://" not in page
    assert "<script" in page and "src=" not in page.split("<script")[1][:200]


def test_form_only_offers_names_that_exist_in_the_poll_file(poll_data, tmp_path):
    from kronikas.guided import form

    page = form.build(poll_data, tmp_path / "f.html").read_text(encoding="utf-8")
    assert "PartyA" not in page  # no placeholder names leak through
    assert page.count('class="card"') == len(poll_data.pollsters)


def test_form_escapes_hostile_names(tmp_path):
    import numpy as np

    from kronikas.data import PollData
    from kronikas.guided import form

    hostile = PollData(
        dates=np.array([0]),
        pollster_ids=np.array([0]),
        sample_sizes=np.array([1000.0]),
        poll_values=np.array([[60.0, 40.0]]),
        candidates=["<script>x</script>", "Unity"],
        pollsters=["A & B"],
        first_poll_date=date(2026, 1, 1),
    )
    page = form.build(hostile, tmp_path / "f.html").read_text(encoding="utf-8")
    assert "<script>x</script>" not in page
    assert "&lt;script&gt;" in page
    assert "A &amp; B" in page


def _chrome() -> str | None:
    """Any local Chromium, for checking the page's JavaScript actually runs."""
    import glob
    import shutil

    for candidate in ("chromium", "chromium-browser", "google-chrome"):
        found = shutil.which(candidate)
        if found:
            return found
    matches = glob.glob("/opt/pw-browsers/chromium*/chrome-linux/chrome")
    return matches[0] if matches else None


@pytest.mark.skipif(_chrome() is None, reason="no local Chromium to run the page in")
def test_the_yaml_the_form_generates_is_settings_the_runner_accepts(
    poll_data, tmp_path
):
    """The page builds YAML in JavaScript; this proves the two agree."""
    import html as html_module
    import re
    import subprocess

    from kronikas.guided import form

    page = form.build(
        poll_data, tmp_path / "form.html", election_date=date(2026, 4, 12)
    ).read_text(encoding="utf-8")
    # Fill it in the way a user would, then let the page rebuild its output.
    firm, party = poll_data.pollsters[0], poll_data.candidates[0]
    filled = page.replace(
        "</body>",
        f"""<script>
        const trust = 'input[name="trust-{firm}"][value="low"]';
        document.querySelector(trust).checked = true;
        document.getElementById("lean-{firm}-{party}").value = "1.5";
        document.getElementById("industry-{party}").value = "1";
        document.getElementById("threshold").value = "5";
        document.dispatchEvent(new Event("input"));
        </script></body>""",
    )
    filled_path = tmp_path / "filled.html"
    filled_path.write_text(filled, encoding="utf-8")
    (tmp_path / "polls.csv").write_bytes((ASSETS / "polls.example.csv").read_bytes())

    dom = subprocess.run(
        [
            str(_chrome()),
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--dump-dom",
            filled_path.as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    match = re.search(r'<pre id="yaml">(.*?)</pre>', dom, re.S)
    assert match, "the page produced no YAML"
    generated = html_module.unescape(match.group(1))

    (tmp_path / "forecast.yaml").write_text(generated, encoding="utf-8")
    plan = settings.load_plan(tmp_path / "forecast.yaml")
    assert plan.election_date == date(2026, 4, 12)
    assert plan.pollsters[firm].trust == "low"
    assert plan.pollsters[firm].leans == {party: 1.5}
    assert plan.industry_expected == {party: 1.0}
    assert plan.industry_uncertainty_pp == 2.5
    assert plan.thresholds_pp == [5.0]
    # And it survives the whole way to a model configuration.
    settings.build_model_config(plan)
