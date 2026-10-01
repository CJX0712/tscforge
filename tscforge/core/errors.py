"""Error hierarchy (E100~E500).

Author: 晨星 (CJX0712)
"""

from __future__ import annotations


class TscForgeError(Exception):
    """Base error for the whole TscForge system. (E100)"""


class BackendUnavailableError(TscForgeError):
    """A Tier-0 optional backend is missing or broken; caller must degrade. (E200)"""


class ConfigError(TscForgeError):
    """Invalid configuration value or ENV override. (E300)"""


class DataGenError(TscForgeError):
    """Synthetic data generation failure (bad knob values etc.). (E400)"""
