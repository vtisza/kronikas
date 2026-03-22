"""kronikas – Hierarchical Bayesian election forecasting."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kronikas")
except PackageNotFoundError:
    __version__ = "unknown"

from .config import ModelConfig, PollsterPrior
from .data import PollData, load_polls
from .forecast import ElectionForecast
from .model import CandidateEstimate, ForecastResult

__all__ = [
    "CandidateEstimate",
    "ElectionForecast",
    "ForecastResult",
    "ModelConfig",
    "PollData",
    "PollsterPrior",
    "load_polls",
]
