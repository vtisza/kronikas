"""Package-level smoke tests."""

import pytest

import kronikas
from kronikas import ModelConfig, PollsterPrior


def test_version():
    assert hasattr(kronikas, "__version__")
    assert isinstance(kronikas.__version__, str)


def test_public_exports():
    for name in [
        "BacktestPoint",
        "BacktestResult",
        "CandidateEstimate",
        "ConvergenceWarning",
        "ElectionForecast",
        "ForecastResult",
        "ModelConfig",
        "PollData",
        "PollsterPrior",
        "SamplingDiagnostics",
        "SharedBiasPrior",
        "backtest",
        "compute_diagnostics",
        "load_polls",
        "polls_from_dataframe",
    ]:
        assert hasattr(kronikas, name), f"{name} missing from the package namespace"


def test_all_is_complete_and_sorted():
    """__all__ must resolve, and stay alphabetical so diffs stay readable."""
    assert kronikas.__all__ == sorted(kronikas.__all__)
    for name in kronikas.__all__:
        assert hasattr(kronikas, name), f"__all__ lists unresolvable name {name!r}"


def test_cli_entry_point_is_importable():
    from kronikas.cli import build_parser, main

    assert callable(main)
    assert build_parser().prog == "kronikas"


def test_no_submodule_is_shadowed_by_an_export():
    """Re-exporting a name that matches a submodule hides the module.

    `from .backtest import backtest` used to rebind `kronikas.backtest` from the
    module to the function, so `import kronikas.backtest as m` returned the
    function and `pydoc.locate("kronikas.backtest")` — the resolution Sphinx,
    pdoc and mkdocstrings all use — documented the wrong object.
    """
    import pkgutil
    import types

    shadowed = []
    for info in pkgutil.iter_modules(kronikas.__path__):
        attr = getattr(kronikas, info.name, None)
        if attr is not None and not isinstance(attr, types.ModuleType):
            shadowed.append(f"kronikas.{info.name} -> {type(attr).__name__}")
    assert not shadowed, (
        "submodule(s) shadowed by a package-level export: " + ", ".join(shadowed)
    )


def test_submodules_resolve_for_documentation_tools():
    import importlib
    import pydoc

    for name in ("backtesting", "config", "data", "diagnostics", "forecast", "model"):
        dotted = f"kronikas.{name}"
        assert importlib.import_module(dotted) is pydoc.locate(dotted), (
            f"{dotted} does not resolve to its module"
        )


class TestModelConfigDefaults:
    def test_sampler_defaults(self):
        cfg = ModelConfig()
        assert cfg.num_tune == 1500
        assert cfg.num_draws == 1000
        assert cfg.num_chains == 2
        assert cfg.cores is None
        assert cfg.target_accept == 0.95
        assert cfg.init_method == "jitter+adapt_diag"
        assert cfg.progressbar is True
        assert cfg.sampler_kwargs == {}

    def test_prior_defaults(self):
        cfg = ModelConfig()
        assert cfg.sigma_walk_prior == 0.05
        assert cfg.sigma_house_prior == 0.3
        assert cfg.initial_sigma == 0.5
        assert cfg.kappa_log_sigma == 0.5

    def test_all_fields_overridable(self):
        cfg = ModelConfig(
            num_tune=500,
            num_draws=200,
            num_chains=4,
            cores=2,
            target_accept=0.99,
            random_seed=7,
            init_method="adapt_full",
            progressbar=False,
            sampler_kwargs={"nuts_sampler": "nutpie"},
            time_step_days=3,
            sigma_walk_prior=0.01,
            sigma_house_prior=0.1,
            initial_sigma=0.2,
            kappa_log_sigma=0.3,
        )
        assert cfg.cores == 2
        assert cfg.init_method == "adapt_full"
        assert cfg.progressbar is False
        assert cfg.sampler_kwargs == {"nuts_sampler": "nutpie"}
        assert cfg.time_step_days == 3

    def test_pollster_priors_default_empty(self):
        cfg = ModelConfig()
        assert cfg.pollster_priors == {}

    def test_pollster_priors_overridable(self):
        cfg = ModelConfig(
            pollster_priors={
                "FirmA": PollsterPrior(sigma_house=0.1, kappa_log_sigma=0.2),
                "FirmB": PollsterPrior(sigma_house=0.05),
            },
        )
        assert len(cfg.pollster_priors) == 2
        assert cfg.pollster_priors["FirmA"].sigma_house == 0.1
        assert cfg.pollster_priors["FirmA"].kappa_log_sigma == 0.2
        assert cfg.pollster_priors["FirmB"].kappa_log_sigma is None

    def test_pollster_priors_independent_across_instances(self):
        a = ModelConfig()
        b = ModelConfig()
        a.pollster_priors["X"] = PollsterPrior(sigma_house=0.1)
        assert "X" not in b.pollster_priors

    def test_sampler_kwargs_independent_across_instances(self):
        a = ModelConfig()
        b = ModelConfig()
        a.sampler_kwargs["x"] = 1
        assert "x" not in b.sampler_kwargs

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("num_draws", 0),
            ("num_chains", 0),
            ("cores", 0),
            ("target_accept", 1.0),
            ("kappa_log_sigma", float("nan")),
            ("lkj_eta", 0.0),
        ],
    )
    def test_invalid_numeric_configuration_is_rejected(self, field, value):
        with pytest.raises(ValueError):
            ModelConfig(**{field: value})

    def test_sampler_kwargs_cannot_duplicate_explicit_options(self):
        with pytest.raises(ValueError, match="duplicates"):
            ModelConfig(sampler_kwargs={"draws": 10})
