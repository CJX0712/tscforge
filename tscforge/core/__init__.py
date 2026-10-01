"""Core layer: types, errors, config, seed, interfaces.

Author: 晨星 (CJX0712)
"""

from tscforge.core.config import TscConfig
from tscforge.core.errors import (
    BackendUnavailableError,
    ConfigError,
    DataGenError,
    TscForgeError,
)
from tscforge.core.interfaces import Classifier
from tscforge.core.seed import get_rng, set_global_seed
from tscforge.core.types import SplitName, TimeSeriesBatch

__all__ = [
    "TscConfig",
    "TscForgeError",
    "BackendUnavailableError",
    "ConfigError",
    "DataGenError",
    "Classifier",
    "get_rng",
    "set_global_seed",
    "SplitName",
    "TimeSeriesBatch",
]
