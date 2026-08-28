"""Plain-language settings file -> kronikas configuration.

The guided workflow asks a non-technical user a handful of questions and
writes the answers to ``forecast.yaml``.  This module turns that file into
the objects ``kronikas`` actually wants, and refuses clearly when an answer
does not make sense — before an expensive MCMC fit rather than after it.

Two layers, deliberately separate:

* :func:`normalise` is pure Python.  It validates the file, resolves word
  presets ("quick", "volatile", "high trust") into numbers, and returns a
  :class:`Plan`.  No kronikas import, so it can be tested on its own.
* :func:`build_model_config` turns a :class:`Plan` into a ``ModelConfig``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

# --- presets ---------------------------------------------------------------
# Every preset resolves to a plain number that the settings file may also
# supply directly, so the words are a convenience and never a ceiling.

EFFORT_PRESETS: dict[str, dict[str, Any]] = {
    # "quick" keeps a full 1000 draws per chain: fewer than that and the
    # effective-sample-size check fails on almost every run, crying wolf about
    # convergence when the fit is merely coarse. Warmup is what gets cut.
    "quick": {
        "num_tune": 750,
        "num_draws": 1000,
        "num_chains": 2,
        "target_accept": 0.90,
    },
    "standard": {
        "num_tune": 1500,
        "num_draws": 1000,
        "num_chains": 2,
        "target_accept": 0.95,
    },
    "thorough": {
        "num_tune": 2000,
        "num_draws": 2000,
        "num_chains": 4,
        "target_accept": 0.99,
    },
}

# Random-walk scale, in logit units per week. ~1 pp/week at 50 % for "normal".
VOLATILITY_PRESETS: dict[str, float] = {
    "calm": 0.02,
    "normal": 0.05,
    "volatile": 0.10,
}

# Fixed house-effect SD in logit space. None keeps the hierarchical estimate,
# i.e. "let the data decide how far this pollster may stray".
TRUST_PRESETS: dict[str, float | None] = {
    "high": 0.10,
    "normal": None,
    "low": 0.60,
}

# SD of the prior on poll precision. Higher allows more overdispersion, i.e.
# "treat this firm's stated sample size with more suspicion".
NOISINESS_PRESETS: dict[str, float | None] = {
    "low": 0.25,
    "normal": None,
    "high": 1.00,
}

TOP_LEVEL_KEYS = {"election", "polls", "beliefs", "run", "report", "advanced"}
ELECTION_KEYS = {"name", "date", "as_of"}
POLLS_KEYS = {"file", "columns", "parties", "date_format", "decimal"}
COLUMN_KEYS = {"date", "pollster", "sample_size", "undecided"}
BELIEF_KEYS = {"volatility", "pollsters", "industry_error"}
POLLSTER_KEYS = {"leans", "trust", "noisiness"}
INDUSTRY_KEYS = {"expected", "uncertainty_pp", "party_uncertainty"}
RUN_KEYS = {"effort", "time_step_days", "seed", "progress"}
REPORT_KEYS = {
    "output_dir",
    "thresholds_pp",
    "industry_error_scenarios_pp",
    "party_colors",
}


class SettingsError(Exception):
    """A settings file the user needs to fix, described in their words."""


# --- tiny YAML reader ------------------------------------------------------


def _strip_comment(line: str) -> str:
    """Drop a trailing ``#`` comment, ignoring ``#`` inside quotes."""
    out: list[str] = []
    quote: str | None = None
    for index, char in enumerate(line):
        if quote:
            out.append(char)
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
            out.append(char)
            continue
        if char == "#" and (index == 0 or line[index - 1] in " \t"):
            break
        out.append(char)
    return "".join(out).rstrip()


def _split_flow(text: str) -> list[str]:
    """Split a comma-separated flow list, respecting quotes."""
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for char in text:
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
            current.append(char)
            continue
        if char == ",":
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))
    return [part.strip() for part in parts if part.strip()]


def _scalar(text: str, lineno: int) -> Any:
    """Convert one YAML scalar (or flow collection) to a Python value."""
    text = text.strip()
    if not text or text in {"null", "~"}:
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        return [_scalar(item, lineno) for item in _split_flow(text[1:-1])]
    if text.startswith("{") and text.endswith("}"):
        mapping: dict[str, Any] = {}
        for item in _split_flow(text[1:-1]):
            key, sep, value = item.partition(":")
            if not sep:
                raise SettingsError(
                    f"Line {lineno}: {item!r} should look like 'Name: value'."
                )
            mapping[_scalar(key, lineno)] = _scalar(value, lineno)
        return mapping
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _parse_block(
    lines: list[tuple[int, int, str]], start: int, indent: int
) -> tuple[Any, int]:
    """Parse one indented block, returning the value and the next line index."""
    if lines[start][2].startswith("- "):
        items: list[Any] = []
        index = start
        while index < len(lines) and lines[index][0] == indent:
            lineno, _, text = lines[index][1], lines[index][0], lines[index][2]
            if not text.startswith("- "):
                break
            items.append(_scalar(text[2:], lineno))
            index += 1
        return items, index

    mapping: dict[str, Any] = {}
    index = start
    while index < len(lines) and lines[index][0] == indent:
        _, lineno, text = lines[index]
        key, sep, rest = text.partition(":")
        if not sep:
            raise SettingsError(
                f"Line {lineno}: {text!r} should look like 'setting: value'."
            )
        key = str(_scalar(key, lineno))
        rest = rest.strip()
        index += 1
        if rest:
            mapping[key] = _scalar(rest, lineno)
            continue
        if index < len(lines) and lines[index][0] > indent:
            mapping[key], index = _parse_block(lines, index, lines[index][0])
        else:
            mapping[key] = None
    return mapping, index


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the subset of YAML the settings template uses.

    Mappings, nested mappings, lists of scalars, ``[a, b]`` and ``{a: 1}``
    flow collections, comments and blank lines.  Anything fancier raises
    :class:`SettingsError` telling the user what shape was expected.
    """
    lines: list[tuple[int, int, str]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        cleaned = _strip_comment(raw.replace("\t", "    "))
        if not cleaned.strip():
            continue
        lines.append((len(cleaned) - len(cleaned.lstrip(" ")), number, cleaned.strip()))
    if not lines:
        return {}
    value, _ = _parse_block(lines, 0, lines[0][0])
    if not isinstance(value, dict):
        raise SettingsError("The settings file must start with settings, not a list.")
    return value


def read_settings_file(path: str | Path) -> dict[str, Any]:
    """Read ``forecast.yaml`` (or ``.json``) into a plain dictionary."""
    path = Path(path)
    if not path.exists():
        raise SettingsError(f"No settings file at {path}.")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SettingsError(f"{path} is not valid JSON: {exc}") from None
    else:
        try:
            import yaml
        except ImportError:
            # PyYAML is not a kronikas dependency, and the template needs only
            # a small slice of YAML, so fall back to reading it directly.
            loaded = parse_simple_yaml(text)
        else:
            try:
                loaded = yaml.safe_load(text) or {}
            except yaml.YAMLError as exc:
                raise SettingsError(f"{path} is not valid YAML: {exc}") from None
    if not isinstance(loaded, dict):
        raise SettingsError(f"{path} does not contain a settings block.")
    return loaded


# --- the plan --------------------------------------------------------------


@dataclass
class PollsterBelief:
    """What the user said about one polling firm."""

    name: str
    leans: dict[str, float] = field(default_factory=dict)
    trust: str = "normal"
    noisiness: str = "normal"
    sigma_house: float | None = None
    kappa_log_sigma: float | None = None


@dataclass
class Plan:
    """Everything a run needs, with every preset already resolved."""

    election_date: date
    polls_path: Path
    election_name: str | None = None
    as_of: date | None = None
    loader_kwargs: dict[str, Any] = field(default_factory=dict)
    volatility: str = "normal"
    sigma_walk_prior: float = VOLATILITY_PRESETS["normal"]
    pollsters: dict[str, PollsterBelief] = field(default_factory=dict)
    industry_expected: dict[str, float] = field(default_factory=dict)
    industry_uncertainty_pp: float = 0.0
    industry_party_uncertainty: dict[str, float] = field(default_factory=dict)
    effort: str = "standard"
    sampler: dict[str, Any] = field(default_factory=dict)
    time_step_days: int = 7
    seed: int = 42
    progress: bool = True
    advanced: dict[str, Any] = field(default_factory=dict)
    output_dir: Path = Path("forecast-output")
    thresholds_pp: list[float] = field(default_factory=list)
    scenarios_pp: list[float] = field(default_factory=list)
    party_colors: dict[str, str] = field(default_factory=dict)

    @property
    def named_parties(self) -> set[str]:
        """Every party the user mentioned anywhere in their beliefs."""
        names: set[str] = set(self.industry_expected)
        names |= set(self.industry_party_uncertainty)
        for belief in self.pollsters.values():
            names |= set(belief.leans)
        return names


# --- validation helpers ----------------------------------------------------


def _check_keys(block: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise SettingsError(
            f"{where}: unknown setting(s) {', '.join(repr(u) for u in unknown)}. "
            f"Allowed here: {', '.join(sorted(allowed))}."
        )


def _as_block(value: Any, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SettingsError(f"{where} should be a block of settings, not {value!r}.")
    return value


def _as_date(value: Any, where: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value.strip(), "%Y-%m-%d").date()
        except ValueError:
            pass
    raise SettingsError(f"{where} must be a date written as YYYY-MM-DD, got {value!r}.")


def _as_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SettingsError(f"{where} must be a number, got {value!r}.")
    return float(value)


def _as_pp_map(value: Any, where: str, *, allow_negative: bool = True) -> dict:
    block = _as_block(value, where)
    out: dict[str, float] = {}
    for party, raw in block.items():
        number = _as_number(raw, f"{where}: {party}")
        if not allow_negative and number < 0:
            raise SettingsError(f"{where}: {party} cannot be negative ({number}).")
        out[str(party)] = number
    return out


def _preset(
    value: Any, presets: dict[str, Any], where: str, default: str
) -> tuple[str, Any]:
    """Resolve a word preset, or accept a number the user typed instead."""
    if value is None:
        return default, presets[default]
    if isinstance(value, str):
        word = value.strip().lower()
        if word not in presets:
            raise SettingsError(
                f"{where} must be one of {', '.join(sorted(presets))} "
                f"(or a number), got {value!r}."
            )
        return word, presets[word]
    return "custom", _as_number(value, where)


def _as_float_list(value: Any, where: str) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if not isinstance(value, list):
        raise SettingsError(f"{where} must be a list of numbers, got {value!r}.")
    return [_as_number(item, where) for item in value]


# --- normalisation ---------------------------------------------------------


def normalise(raw: dict[str, Any], *, base_dir: Path | None = None) -> Plan:
    """Validate a settings dictionary and resolve it into a :class:`Plan`."""
    base_dir = Path(base_dir or ".")
    _check_keys(raw, TOP_LEVEL_KEYS, "settings file")

    election = _as_block(raw.get("election"), "election")
    _check_keys(election, ELECTION_KEYS, "election")
    if election.get("date") is None:
        raise SettingsError(
            "election.date is required — the day the votes are counted, "
            "written as YYYY-MM-DD."
        )
    election_date = _as_date(election["date"], "election.date")
    as_of = (
        _as_date(election["as_of"], "election.as_of")
        if election.get("as_of") is not None
        else None
    )
    if as_of is not None and as_of > election_date:
        raise SettingsError(
            f"election.as_of ({as_of}) is after election.date ({election_date}); "
            "the forecast cannot start after the election."
        )

    polls = _as_block(raw.get("polls"), "polls")
    _check_keys(polls, POLLS_KEYS, "polls")
    if not polls.get("file"):
        raise SettingsError("polls.file is required — the path to your poll CSV.")
    polls_path = Path(str(polls["file"]))
    if not polls_path.is_absolute():
        polls_path = base_dir / polls_path

    columns = _as_block(polls.get("columns"), "polls.columns")
    _check_keys(columns, COLUMN_KEYS, "polls.columns")
    parties = polls.get("parties")
    if parties is not None and not isinstance(parties, list):
        raise SettingsError("polls.parties must be a list of column names.")
    decimal = polls.get("decimal") or "."
    if len(str(decimal)) != 1:
        raise SettingsError(
            f"polls.decimal must be a single character, got {decimal!r}."
        )
    loader_kwargs = {
        "date_column": str(columns.get("date") or "date"),
        "pollster_column": str(columns.get("pollster") or "pollster"),
        "sample_size_column": str(columns.get("sample_size") or "sample_size"),
        "undecided_column": (
            str(columns["undecided"]) if columns.get("undecided") else None
        ),
        "candidate_columns": [str(p) for p in parties] if parties else None,
        "date_format": (
            str(polls["date_format"]) if polls.get("date_format") else None
        ),
        "decimal": str(decimal),
    }

    beliefs = _as_block(raw.get("beliefs"), "beliefs")
    _check_keys(beliefs, BELIEF_KEYS, "beliefs")
    volatility, sigma_walk = _preset(
        beliefs.get("volatility"), VOLATILITY_PRESETS, "beliefs.volatility", "normal"
    )
    if sigma_walk <= 0:
        raise SettingsError("beliefs.volatility as a number must be above zero.")

    pollster_block = _as_block(beliefs.get("pollsters"), "beliefs.pollsters")
    pollster_beliefs: dict[str, PollsterBelief] = {}
    for name, value in pollster_block.items():
        where = f"beliefs.pollsters.{name}"
        block = _as_block(value, where)
        _check_keys(block, POLLSTER_KEYS, where)
        trust, sigma_house = _preset(
            block.get("trust"), TRUST_PRESETS, f"{where}.trust", "normal"
        )
        noisiness, kappa = _preset(
            block.get("noisiness"), NOISINESS_PRESETS, f"{where}.noisiness", "normal"
        )
        leans = _as_pp_map(block.get("leans"), f"{where}.leans")
        for party, pp in leans.items():
            if not -50.0 < pp < 50.0:
                raise SettingsError(
                    f"{where}.leans: {party} is {pp} pp, which is outside the "
                    "allowed range of -50 to 50 percentage points."
                )
        pollster_beliefs[str(name)] = PollsterBelief(
            name=str(name),
            leans=leans,
            trust=trust,
            noisiness=noisiness,
            sigma_house=sigma_house,
            kappa_log_sigma=kappa,
        )

    industry = _as_block(beliefs.get("industry_error"), "beliefs.industry_error")
    _check_keys(industry, INDUSTRY_KEYS, "beliefs.industry_error")
    industry_expected = _as_pp_map(
        industry.get("expected"), "beliefs.industry_error.expected"
    )
    industry_uncertainty = (
        _as_number(industry["uncertainty_pp"], "beliefs.industry_error.uncertainty_pp")
        if industry.get("uncertainty_pp") is not None
        else 0.0
    )
    if industry_uncertainty < 0:
        raise SettingsError("beliefs.industry_error.uncertainty_pp cannot be negative.")
    industry_party_uncertainty = _as_pp_map(
        industry.get("party_uncertainty"),
        "beliefs.industry_error.party_uncertainty",
        allow_negative=False,
    )

    run = _as_block(raw.get("run"), "run")
    _check_keys(run, RUN_KEYS, "run")
    effort_word = run.get("effort")
    if effort_word is not None and (
        not isinstance(effort_word, str)
        or effort_word.strip().lower() not in EFFORT_PRESETS
    ):
        raise SettingsError(
            f"run.effort must be one of {', '.join(sorted(EFFORT_PRESETS))}, "
            f"got {effort_word!r}."
        )
    effort = (effort_word or "standard").strip().lower()
    time_step = run.get("time_step_days")
    time_step_days = (
        int(_as_number(time_step, "run.time_step_days")) if time_step else 7
    )
    if time_step_days < 1:
        raise SettingsError("run.time_step_days must be at least 1.")
    seed = (
        int(_as_number(run["seed"], "run.seed")) if run.get("seed") is not None else 42
    )
    progress = run.get("progress")
    if progress is None:
        progress = True
    if not isinstance(progress, bool):
        raise SettingsError("run.progress must be true or false.")

    report = _as_block(raw.get("report"), "report")
    _check_keys(report, REPORT_KEYS, "report")
    output_dir = Path(str(report.get("output_dir") or "forecast-output"))
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    thresholds = _as_float_list(report.get("thresholds_pp"), "report.thresholds_pp")
    scenarios = _as_float_list(
        report.get("industry_error_scenarios_pp"),
        "report.industry_error_scenarios_pp",
    )
    colors_block = _as_block(report.get("party_colors"), "report.party_colors")
    party_colors = {str(k): str(v) for k, v in colors_block.items()}

    advanced = _as_block(raw.get("advanced"), "advanced")

    return Plan(
        election_date=election_date,
        polls_path=polls_path,
        election_name=str(election["name"]) if election.get("name") else None,
        as_of=as_of,
        loader_kwargs=loader_kwargs,
        volatility=volatility,
        sigma_walk_prior=float(sigma_walk),
        pollsters=pollster_beliefs,
        industry_expected=industry_expected,
        industry_uncertainty_pp=industry_uncertainty,
        industry_party_uncertainty=industry_party_uncertainty,
        effort=effort,
        sampler=dict(EFFORT_PRESETS[effort]),
        time_step_days=time_step_days,
        seed=seed,
        progress=bool(progress),
        advanced=advanced,
        output_dir=output_dir,
        thresholds_pp=thresholds,
        scenarios_pp=sorted({abs(pp) for pp in scenarios if pp}),
        party_colors=party_colors,
    )


def load_plan(path: str | Path) -> Plan:
    """Read and validate a settings file in one step."""
    path = Path(path)
    return normalise(read_settings_file(path), base_dir=path.parent)


# --- kronikas objects ------------------------------------------------------


def build_model_config(plan: Plan):  # type: ignore[no-untyped-def]
    """Translate a :class:`Plan` into a ``kronikas.ModelConfig``."""
    from kronikas import ModelConfig, PollsterPrior, SharedBiasPrior

    pollster_priors = {
        name: PollsterPrior(
            sigma_house=belief.sigma_house,
            kappa_log_sigma=belief.kappa_log_sigma,
            mu_house=dict(belief.leans) or None,
        )
        for name, belief in plan.pollsters.items()
        if belief.sigma_house is not None
        or belief.kappa_log_sigma is not None
        or belief.leans
    }

    shared_bias = None
    if (
        plan.industry_expected
        or plan.industry_uncertainty_pp
        or plan.industry_party_uncertainty
    ):
        shared_bias = SharedBiasPrior(
            mean=dict(plan.industry_expected),
            sd=dict(plan.industry_party_uncertainty),
            default_sd=plan.industry_uncertainty_pp,
        )
        if shared_bias.is_inert():
            shared_bias = None

    fields = {
        **plan.sampler,
        "random_seed": plan.seed,
        "time_step_days": plan.time_step_days,
        "sigma_walk_prior": plan.sigma_walk_prior,
        "progressbar": plan.progress,
        "pollster_priors": pollster_priors,
        "shared_bias": shared_bias,
    }
    known = {f.name for f in ModelConfig.__dataclass_fields__.values()}
    unknown = sorted(set(plan.advanced) - known)
    if unknown:
        raise SettingsError(
            "advanced: unknown setting(s) "
            f"{', '.join(repr(u) for u in unknown)}. See the configuration "
            "reference in the kronikas README for the full list."
        )
    fields.update(plan.advanced)
    return ModelConfig(**fields)


def describe(plan: Plan) -> list[str]:
    """Plain-language read-back of every choice, for confirmation and the report."""
    lines = [
        # %-d is glibc-only; build the day number by hand so the same text
        # appears on Windows.
        f"Election day: {plan.election_date.day} "
        f"{plan.election_date:%B %Y}"
        + (f" ({plan.election_name})" if plan.election_name else ""),
        f"Forecast made as if it were: {plan.as_of or 'today'}",
        f"Polls read from: {plan.polls_path}",
        f"Opinion moves: {plan.volatility} "
        f"(random-walk scale {plan.sigma_walk_prior:g} per week)",
        f"Sampling effort: {plan.effort} "
        f"({plan.sampler['num_draws']} draws x {plan.sampler['num_chains']} chains)",
        f"Trend resolution: one step every {plan.time_step_days} day(s)",
    ]
    if plan.pollsters:
        described = []
        for belief in plan.pollsters.values():
            bits = []
            if belief.leans:
                bits.append(
                    "leans "
                    + ", ".join(
                        f"{party} {pp:+g} pp" for party, pp in belief.leans.items()
                    )
                )
            if belief.trust != "normal":
                bits.append(f"{belief.trust} trust")
            if belief.noisiness != "normal":
                bits.append(f"{belief.noisiness} noise")
            described.append(f"{belief.name} ({', '.join(bits) or 'no adjustment'})")
        lines.append("Pollster adjustments you supplied: " + "; ".join(described))
    else:
        lines.append(
            "No pollster-specific beliefs: house effects are learnt from the data."
        )
    if plan.industry_expected or plan.industry_uncertainty_pp:
        expected = (
            ", ".join(f"{p} {v:+g} pp" for p, v in plan.industry_expected.items())
            or "no directional lean"
        )
        lines.append(
            f"Industry-wide polling error: {expected}; "
            f"uncertainty {plan.industry_uncertainty_pp:g} pp"
        )
    else:
        lines.append(
            "Industry-wide polling error: assumed zero with no uncertainty "
            "(the model cannot measure this from polls alone)."
        )
    return lines
