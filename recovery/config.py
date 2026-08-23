"""Read and validate the run settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True, slots=True)
class Settings:
    diagnostic_seed: int = 20260618
    sample_size: int = 1_000
    min_cell_n: int = 10

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTION_FIELDS: dict[str, tuple[str, ...]] = {
    "pair_sample": ("diagnostic_seed", "sample_size"),
    "runtime": ("min_cell_n",),
}


def load_settings(path: str | Path) -> Settings:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)

    unknown_sections = set(raw) - set(_SECTION_FIELDS)
    if unknown_sections:
        raise ValueError(f"unknown settings section(s): {sorted(unknown_sections)}")

    values: dict[str, int] = {}
    for section, fields in _SECTION_FIELDS.items():
        if section not in raw:
            raise ValueError(f"settings file is missing required section [{section}]")
        section_values = raw[section]
        if not isinstance(section_values, dict):
            raise ValueError(f"settings section [{section}] must be a table")

        unknown = set(section_values) - set(fields)
        if unknown:
            raise ValueError(f"unknown setting(s) in [{section}]: {sorted(unknown)}")
        missing = set(fields) - set(section_values)
        if missing:
            raise ValueError(
                f"settings section [{section}] is missing required setting(s): "
                f"{sorted(missing)}"
            )

        for name in fields:
            value = section_values[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            values[name] = value

    settings = Settings(**values)
    _validate(settings)
    return settings


def _validate(settings: Settings) -> None:
    expected = Settings()
    for name in Settings.__dataclass_fields__:
        if getattr(settings, name) != getattr(expected, name):
            raise ValueError(f"{name} must remain {getattr(expected, name)}")
