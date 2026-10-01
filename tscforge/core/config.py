"""Frozen configuration with ENV overrides.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace

from tscforge.core.errors import ConfigError

# ENV prefix: TSCFORGE_<FIELD_UPPER>. Only scalar fields are overridable.
_ENV_SCALARS = {
    "TSCFORGE_N_KERNELS": ("n_kernels", int),
    "TSCFORGE_SEED": ("seed", int),
    "TSCFORGE_BACKEND": ("backend", str),
    "TSCFORGE_DTW_BAND": ("dtw_band", int),
}

_BOOL_FIELDS = {"use_multi_pool", "use_dtw_features", "use_norm_gate"}


@dataclass(frozen=True)
class TscConfig:
    """Frozen system configuration."""

    n_kernels: int = 1000
    use_multi_pool: bool = True
    use_dtw_features: bool = True
    dtw_band: int = 32
    dtw_k: int = 4
    use_norm_gate: bool = True
    backend: str = "auto"
    seed: int = 42
    dtw_weight_grid: tuple = (0.0, 0.5, 1.0, 2.0)

    def __post_init__(self) -> None:
        if self.n_kernels <= 0:
            raise ConfigError(f"n_kernels must be > 0, got {self.n_kernels}")
        if self.dtw_band <= 0:
            raise ConfigError(f"dtw_band must be > 0, got {self.dtw_band}")
        if self.seed < 0:
            raise ConfigError(f"seed must be >= 0, got {self.seed}")
        if self.backend not in ("auto", "numpy", "aeon", "tslearn", "numba"):
            raise ConfigError(f"unknown backend {self.backend!r}")

    def with_overrides(self, **kwargs) -> TscConfig:
        """Return a new frozen config with the given fields replaced."""
        valid = {f.name for f in fields(self)}
        unknown = set(kwargs) - valid
        if unknown:
            raise ConfigError(f"unknown config fields: {sorted(unknown)}")
        return replace(self, **kwargs)

    @classmethod
    def from_env(cls) -> TscConfig:
        """Build config from defaults, overridden by TSCFORGE_* environment variables."""
        values = {}
        for env_key, (field_name, cast) in _ENV_SCALARS.items():
            raw = os.environ.get(env_key)
            if raw is None:
                continue
            try:
                values[field_name] = cast(raw)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"invalid {env_key}={raw!r}: {exc}") from exc
        for field_name in sorted(_BOOL_FIELDS):
            raw = os.environ.get(f"TSCFORGE_{field_name.upper()}")
            if raw is None:
                continue
            low = raw.strip().lower()
            if low not in ("0", "1", "true", "false"):
                raise ConfigError(f"invalid boolean for {field_name}: {raw!r}")
            values[field_name] = low in ("1", "true")
        return cls(**values)
