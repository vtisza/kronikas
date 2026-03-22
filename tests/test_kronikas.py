"""Package-level smoke tests."""

import kronikas
from kronikas import ModelConfig, PollsterPrior


def test_version():
    assert hasattr(kronikas, "__version__")
    assert isinstance(kronikas.__version__, str)


def test_public_exports():
    assert hasattr(kronikas, "ElectionForecast")
    assert hasattr(kronikas, "ModelConfig")
    assert hasattr(kronikas, "PollData")
    assert hasattr(kronikas, "PollsterPrior")
    assert hasattr(kronikas, "ForecastResult")
    assert hasattr(kronikas, "CandidateEstimate")
    assert hasattr(kronikas, "load_polls")


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
