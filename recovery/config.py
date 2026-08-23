"""Reads settings.toml and checks every fixed number used by the matching and model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any
import tomllib


# Settings

@dataclass(frozen=True, slots=True)
class Settings:
    primary_min_months: int = 1
    primary_max_months: int = 48
    prior_history_months: int = 24
    model_seed: int = 20260619
    min_cell_n: int = 10
    diagnostic_seed: int = 20260618
    sample_size: int = 1_000
    min_test_rows: int = 1_000
    min_test_companies: int = 500
    min_test_each_class: int = 100
    min_calibration_each_class: int = 50
    isotonic_each_class: int = 200
    auc_floor: float = 0.70
    max_calibration_gap: float = 0.03
    min_calibration_slope: float = 0.80
    max_calibration_slope: float = 1.20
    bootstrap_replicates: int = 1_000

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTION_FIELDS: dict[str, tuple[str, ...]] = {
    "cohort": (
        "primary_min_months",
        "primary_max_months",
        "prior_history_months",
        "model_seed",
    ),
    "pair_sample": ("diagnostic_seed", "sample_size"),
    "acceptance": (
        "min_test_rows",
        "min_test_companies",
        "min_test_each_class",
        "min_calibration_each_class",
        "isotonic_each_class",
        "auc_floor",
        "max_calibration_gap",
        "min_calibration_slope",
        "max_calibration_slope",
        "bootstrap_replicates",
    ),
    "runtime": ("min_cell_n",),
}

_INTEGER_FIELDS = frozenset(
    name
    for name, field in Settings.__dataclass_fields__.items()
    if field.type == "int"
)
_FLOAT_FIELDS = frozenset(
    name
    for name, field in Settings.__dataclass_fields__.items()
    if field.type == "float"
)


def _typed_value(name: str, value: object) -> int | float:
    if name in _INTEGER_FIELDS:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        return value
    if name in _FLOAT_FIELDS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{name} must be finite")
        return numeric
    raise ValueError(f"unknown setting: {name}")


# Read and check settings.toml

def load_settings(path: str | Path) -> Settings:
    """Load known keys from TOML and reject unsafe or contradictory values."""

    source = Path(path)
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    unknown_sections = set(raw) - set(_SECTION_FIELDS)
    if unknown_sections:
        raise ValueError(f"unknown settings section(s): {sorted(unknown_sections)}")
    flat: dict[str, Any] = {}
    field_sections = {
        field: section
        for section, fields in _SECTION_FIELDS.items()
        for field in fields
    }
    sections: dict[str, dict[str, object]] = {}
    for section in _SECTION_FIELDS:
        if section not in raw:
            raise ValueError(f"settings file is missing required section [{section}]")
        values = raw[section]
        if not isinstance(values, dict):
            raise ValueError(f"settings section [{section}] must be a table")
        sections[section] = values

    for section, fields in _SECTION_FIELDS.items():
        values = sections[section]
        unexpected = set(values) - set(fields)
        if unexpected:
            misplaced = {
                name: field_sections[name]
                for name in sorted(unexpected)
                if name in field_sections
            }
            if misplaced:
                details = ", ".join(
                    f"{name} belongs in [{expected}]"
                    for name, expected in misplaced.items()
                )
                raise ValueError(f"misplaced setting(s) in [{section}]: {details}")
            raise ValueError(
                f"unknown setting(s) in [{section}]: {sorted(unexpected)}"
            )

    for section, fields in _SECTION_FIELDS.items():
        values = sections[section]
        missing = set(fields) - set(values)
        if missing:
            raise ValueError(
                f"settings section [{section}] is missing required setting(s): "
                f"{sorted(missing)}"
            )
        for name in fields:
            flat[name] = _typed_value(name, values[name])
    settings = Settings(**flat)
    _validate(settings)
    return settings


def _validate(settings: Settings) -> None:
    if (settings.primary_min_months, settings.primary_max_months) != (1, 48):
        raise ValueError("primary cohort window must remain frozen at 1 to 48 months")
    if settings.prior_history_months != 24:
        raise ValueError("prior_history_months must be 24 for the frozen feature schema")
    if settings.sample_size != 1_000:
        raise ValueError("linkage-validation sample size must be 1,000")
    if not 0 <= settings.diagnostic_seed <= 2**32 - 1:
        raise ValueError("diagnostic_seed must be between 0 and 2^32 - 1")
    if not 0 <= settings.model_seed <= 2**32 - 1:
        raise ValueError("model_seed must be between 0 and 2^32 - 1")
    if settings.diagnostic_seed == settings.model_seed:
        raise ValueError("diagnostic_seed and model_seed must differ")
    for name in (
        "min_test_rows",
        "min_test_companies",
        "min_test_each_class",
        "min_calibration_each_class",
        "isotonic_each_class",
        "min_cell_n",
    ):
        if getattr(settings, name) < 1:
            raise ValueError(f"{name} must be positive")
    if settings.isotonic_each_class < settings.min_calibration_each_class:
        raise ValueError(
            "isotonic_each_class must be at least min_calibration_each_class"
        )
    if settings.min_test_companies > settings.min_test_rows:
        raise ValueError("min_test_companies must not exceed min_test_rows")
    if not 0.0 <= settings.auc_floor <= 1.0:
        raise ValueError("auc_floor must be between 0 and 1")
    if not 0.0 <= settings.max_calibration_gap <= 1.0:
        raise ValueError("max_calibration_gap must be between 0 and 1")
    if not (
        0.0
        < settings.min_calibration_slope
        <= settings.max_calibration_slope
    ):
        raise ValueError(
            "calibration slopes must satisfy 0 < min_calibration_slope "
            "<= max_calibration_slope"
        )
    if settings.bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates must be at least 100")
