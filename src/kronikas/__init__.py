"""kronikas – Hierarchical Bayesian election forecasting."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kronikas")
except PackageNotFoundError:
    __version__ = "unknown"

from .backtest import BacktestPoint, BacktestResult, backtest
from .config import ModelConfig, PollsterPrior, SharedBiasPrior
from .data import PollData, load_polls, polls_from_dataframe
from .diagnostics import ConvergenceWarning, SamplingDiagnostics, compute_diagnostics
from .forecast import ElectionForecast
from .model import CandidateEstimate, ForecastResult

__all__ = [
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
]
