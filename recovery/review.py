"""Validate RT's manual match review and emit aggregate-only quality results.

Input rows remain RT-internal and may contain identifiers.  The returned and
written results contain tier-level counts and Wilson intervals only.  This
module performs no network or shell operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Final

import pandas as pd

try:  # Package import in tests; direct import when recovery/ is on sys.path.
    from .config import Settings
except ImportError:  # pragma: no cover - exercised by the Windows launcher.
    from config import Settings  # type: ignore[no-redef]


DECISIONS: Final = ("correct", "incorrect", "uncertain")
REVIEW_TIERS: Final = ("auto", "review", "fallback_review")
_TIER_VALUE_ALIASES: Final = {"fallback": "fallback_review"}
_Z_95: Final = 1.959963984540054

_TIER_ALIASES: Final = (
    "review_tier",
    "sample_tier",
    "review_stratum",
    "match_tier",
    "tier",
)
_DECISION_ALIASES: Final = (
    "review_decision",
    "manual_decision",
    "manual_review",
    "review_outcome",
    "decision",
    "outcome",
)
_ROW_ID_ALIASES: Final = ("review_row_id", "sample_id", "row_id")


class ReviewFormatError(ValueError):
    """The completed review file does not meet its locked schema."""


@dataclass(frozen=True, slots=True)
class ReviewResult:
    """Aggregate review evidence; no reviewed row or identifier is retained."""

    stats: pd.DataFrame
    gate_passed: bool
    gate_reasons: tuple[str, ...]
    total_rows: int
    uncertain_rows: int


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return the two-sided 95% Wilson score interval for a binomial rate."""

    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Wilson inputs must satisfy 0 <= successes <= total")
    if total == 0:
        return float("nan"), float("nan")
    observed = successes / total
    z2 = _Z_95**2
    denominator = 1.0 + z2 / total
    centre = (observed + z2 / (2.0 * total)) / denominator
    radius = (
        _Z_95
        * sqrt(observed * (1.0 - observed) / total + z2 / (4.0 * total**2))
        / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def parse_completed_review(
    source: str | Path | pd.DataFrame,
    settings: Settings,
    *,
    tier_column: str | None = None,
    decision_column: str | None = None,
) -> ReviewResult:
    """Parse the locked 1,000-row review and evaluate the auto-match gate.

    Decisions are case-insensitive but must be exactly ``correct``,
    ``incorrect`` or ``uncertain``.  An uncertain decision is conservatively
    counted as a non-success in the observed precision and Wilson interval.
    """

    rows = _read_review(source)
    tier_col = _resolve_column(rows, tier_column, _TIER_ALIASES, "review tier")
    decision_col = _resolve_column(
        rows, decision_column, _DECISION_ALIASES, "review decision"
    )
    row_id_col = _optional_column(rows, _ROW_ID_ALIASES)

    if row_id_col is not None:
        row_ids = rows[row_id_col].astype("string").str.strip()
        if row_ids.isna().any() or row_ids.eq("").any():
            raise ReviewFormatError(f"{row_id_col!r} contains a blank review-row ID")
        if row_ids.duplicated().any():
            raise ReviewFormatError(f"{row_id_col!r} contains duplicate review-row IDs")

    tiers = (
        rows[tier_col]
        .astype("string")
        .str.strip()
        .str.casefold()
        .replace(_TIER_VALUE_ALIASES)
    )
    decisions = rows[decision_col].astype("string").str.strip().str.casefold()
    bad_tiers = sorted(set(tiers.dropna()) - set(REVIEW_TIERS))
    if tiers.isna().any() or tiers.eq("").any() or bad_tiers:
        raise ReviewFormatError(
            "review tiers must be auto, review or fallback_review"
            + (f"; unexpected values: {bad_tiers}" if bad_tiers else "")
        )
    bad_decisions = sorted(set(decisions.dropna()) - set(DECISIONS))
    if decisions.isna().any() or decisions.eq("").any() or bad_decisions:
        raise ReviewFormatError(
            "review decisions must be correct, incorrect or uncertain"
            + (f"; unexpected values: {bad_decisions}" if bad_decisions else "")
        )

    actual_by_tier = {tier: int(value) for tier, value in tiers.value_counts().items()}
    expected_by_tier = _expected_allocation(rows, tiers, settings)
    expected_total = sum(expected_by_tier.values())
    if len(rows) != expected_total:
        raise ReviewFormatError(
            f"completed review allocation totals {expected_total} rows; found {len(rows)}"
        )
    allocation_errors = {
        tier: (expected, int(actual_by_tier.get(tier, 0)))
        for tier, expected in expected_by_tier.items()
        if int(actual_by_tier.get(tier, 0)) != expected
    }
    if allocation_errors:
        details = ", ".join(
            f"{tier} expected {expected}, found {actual}"
            for tier, (expected, actual) in allocation_errors.items()
        )
        raise ReviewFormatError(f"review allocation does not match settings: {details}")

    aggregate_rows: list[dict[str, object]] = []
    for tier in REVIEW_TIERS:
        selected = decisions[tiers.eq(tier)]
        n_total = int(len(selected))
        n_correct = int(selected.eq("correct").sum())
        n_incorrect = int(selected.eq("incorrect").sum())
        n_uncertain = int(selected.eq("uncertain").sum())
        observed = n_correct / n_total if n_total else float("nan")
        lower, upper = wilson_interval(n_correct, n_total)
        aggregate_rows.append(
            {
                "tier": tier,
                "n_reviewed": n_total,
                "n_correct": n_correct,
                "n_incorrect": n_incorrect,
                "n_uncertain": n_uncertain,
                "observed_precision": observed,
                "wilson_lower_95": lower,
                "wilson_upper_95": upper,
            }
        )

    stats = pd.DataFrame(aggregate_rows)
    uncertain_rows = int(stats["n_uncertain"].sum())
    auto = stats.loc[stats["tier"].eq("auto")].iloc[0]
    reasons: list[str] = []
    if float(auto["observed_precision"]) < settings.match_precision_floor:
        reasons.append(
            "auto observed precision is below "
            f"{settings.match_precision_floor:.3f}"
        )
    if float(auto["wilson_lower_95"]) <= settings.match_precision_lower_ci_floor:
        reasons.append(
            "auto Wilson lower bound is not above "
            f"{settings.match_precision_lower_ci_floor:.3f}"
        )
    gate_passed = not reasons
    stats["auto_gate_applies"] = stats["tier"].eq("auto")
    stats["auto_gate_passed"] = stats["auto_gate_applies"] & gate_passed
    return ReviewResult(
        stats=stats,
        gate_passed=gate_passed,
        gate_reasons=tuple(reasons),
        total_rows=len(rows),
        uncertain_rows=uncertain_rows,
    )


def _expected_allocation(
    rows: pd.DataFrame, tiers: pd.Series, settings: Settings
) -> dict[str, int]:
    """Use the sample's locked allocation when a short tier was redistributed."""

    allocation_column = next(
        (column for column in rows.columns if _canon(column) == "sample_allocation"),
        None,
    )
    if allocation_column is None:
        return {
            "auto": settings.sample_auto,
            "review": settings.sample_review,
            "fallback_review": settings.sample_fallback,
        }
    expected: dict[str, int] = {}
    values = pd.to_numeric(rows[allocation_column], errors="coerce")
    if values.isna().any() or values.lt(0).any() or (values % 1 != 0).any():
        raise ReviewFormatError("sample_allocation must contain non-negative whole numbers")
    for tier in REVIEW_TIERS:
        tier_values = values[tiers.eq(tier)].astype(int).unique()
        if len(tier_values) > 1:
            raise ReviewFormatError(f"sample_allocation is inconsistent within tier {tier}")
        expected[tier] = int(tier_values[0]) if len(tier_values) else 0
    if sum(expected.values()) != 1_000:
        raise ReviewFormatError("sample allocation must total exactly 1,000 rows")
    return expected


def write_review_aggregates(
    result: ReviewResult,
    outdir: str | Path,
    *,
    min_cell_n: int = 10,
    basename: str = "E2_review_quality",
) -> tuple[Path, Path]:
    """Write identifier-free CSV and text summaries of a completed review."""

    if min_cell_n < 1:
        raise ValueError("min_cell_n must be positive")
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    sensitive_counts = result.stats[["n_correct", "n_incorrect", "n_uncertain"]]
    small_outcome = ((sensitive_counts > 0) & (sensitive_counts < min_cell_n)).any(
        axis=1
    )
    public = result.stats.loc[
        result.stats["n_reviewed"].ge(min_cell_n) & ~small_outcome
    ].copy()
    suppressed = int(len(result.stats) - len(public))
    csv_path = destination / f"{basename}.csv"
    txt_path = destination / f"{basename}.txt"
    public.to_csv(csv_path, index=False, float_format="%.6f")

    lines = [
        "RT MATCH-REVIEW QUALITY (AGGREGATE ONLY)",
        f"Completed rows: {result.total_rows}",
        "Uncertain decision present: " + ("yes" if result.uncertain_rows else "no"),
        "AUTO GATE: " + ("PASS" if result.gate_passed else "FAIL"),
    ]
    if result.gate_reasons:
        lines.extend(f"  - {reason}" for reason in result.gate_reasons)
    for row in public.itertuples(index=False):
        lines.append(
            f"{row.tier}: n={row.n_reviewed}; observed={row.observed_precision:.6f}; "
            f"Wilson 95%=[{row.wilson_lower_95:.6f}, {row.wilson_upper_95:.6f}]"
        )
    if suppressed:
        lines.append(f"Tier rows below minimum cell n={min_cell_n} were suppressed.")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, txt_path


def _read_review(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        return pd.read_csv(path, dtype="string", keep_default_na=False)
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, dtype="string", keep_default_na=False)
    raise ReviewFormatError("completed review must be a CSV or XLSX file")


def _canon(value: object) -> str:
    return "_".join(str(value).strip().casefold().replace("-", " ").split())


def _resolve_column(
    rows: pd.DataFrame,
    explicit: str | None,
    aliases: tuple[str, ...],
    label: str,
) -> str:
    lookup = {_canon(column): str(column) for column in rows.columns}
    if explicit is not None:
        found = lookup.get(_canon(explicit))
        if found is None:
            raise ReviewFormatError(f"missing {label} column {explicit!r}")
        return found
    found_columns = [lookup[alias] for alias in aliases if alias in lookup]
    if not found_columns:
        raise ReviewFormatError(
            f"missing {label} column; accepted headings: {', '.join(aliases)}"
        )
    if len(set(found_columns)) > 1:
        raise ReviewFormatError(f"multiple possible {label} columns: {found_columns}")
    return found_columns[0]


def _optional_column(rows: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lookup = {_canon(column): str(column) for column in rows.columns}
    found = [lookup[alias] for alias in aliases if alias in lookup]
    if len(set(found)) > 1:
        raise ReviewFormatError(f"multiple possible review-row ID columns: {found}")
    return found[0] if found else None
