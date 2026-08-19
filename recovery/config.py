"""Read and validate the transparent analysis settings.

Input: ``settings.toml``. Output: an immutable ``Settings`` object. This file
contains no RT data and performs no network, shell, or output operations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True, slots=True)
class Settings:
    auto_threshold: float = 0.85
    review_threshold: float = 0.70
    auto_margin: float = 0.05
    primary_min_months: int = 12
    primary_max_months: int = 36
    prior_history_months: int = 24
    min_cell_n: int = 10
    diagnostic_seed: int = 20260618
    locked_seed: int = 20260619
    sample_auto: int = 500
    sample_review: int = 300
    sample_fallback: int = 200
    min_test_rows: int = 1_000
    min_test_each_class: int = 100
    min_calibration_each_class: int = 50
    isotonic_each_class: int = 200
    auc_floor: float = 0.70
    match_precision_floor: float = 0.98
    match_precision_lower_ci_floor: float = 0.95
    max_calibration_gap: float = 0.03
    min_calibration_slope: float = 0.80
    max_calibration_slope: float = 1.20
    bootstrap_replicates: int = 1_000

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS = {"matching", "cohort", "review_sample", "acceptance", "runtime"}


def load_settings(path: str | Path) -> Settings:
    """Load known keys from TOML and reject unsafe or contradictory values."""

    source = Path(path)
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    unknown_sections = set(raw) - _SECTIONS
    if unknown_sections:
        raise ValueError(f"unknown settings section(s): {sorted(unknown_sections)}")
    flat: dict[str, Any] = {}
    for section in _SECTIONS:
        values = raw.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"settings section [{section}] must be a table")
        flat.update(values)
    known = set(Settings.__dataclass_fields__)
    unknown = set(flat) - known
    if unknown:
        raise ValueError(f"unknown setting(s): {sorted(unknown)}")
    settings = Settings(**flat)
    _validate(settings)
    return settings


def _validate(settings: Settings) -> None:
    if not 0 <= settings.review_threshold < settings.auto_threshold <= 1:
        raise ValueError("matching thresholds must satisfy 0 <= review < auto <= 1")
    if not 0 <= settings.auto_margin <= 1:
        raise ValueError("auto_margin must be between 0 and 1")
    if not 0 < settings.primary_min_months < settings.primary_max_months:
        raise ValueError("primary cohort months must satisfy 0 < min < max")
    if settings.prior_history_months <= 0:
        raise ValueError("prior_history_months must be positive")
    if min(settings.sample_auto, settings.sample_review, settings.sample_fallback) < 0:
        raise ValueError("review sample sizes cannot be negative")
    if sum((settings.sample_auto, settings.sample_review, settings.sample_fallback)) != 1_000:
        raise ValueError("review sample allocations must total 1,000")
    if settings.diagnostic_seed == settings.locked_seed:
        raise ValueError("diagnostic_seed and locked_seed must differ")
    if not 0 < settings.match_precision_lower_ci_floor <= settings.match_precision_floor <= 1:
        raise ValueError("match precision floors must be in (0, 1] and correctly ordered")
    if settings.bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates must be at least 100")
