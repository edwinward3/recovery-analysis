"""Write the matching check outputs."""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator
import hashlib
import json
import os
import platform
import re
import sys
import time

import pandas as pd


# Output folders and run times

@dataclass(frozen=True, slots=True)
class RunPaths:
    root: Path
    results: Path
    working: Path


@dataclass(slots=True)
class RunRecorder:
    stages: list[dict[str, Any]] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str) -> Iterator[dict[str, Any]]:
        record: dict[str, Any] = {"stage": name, "started_utc": _utc_now()}
        started = time.perf_counter()
        try:
            yield record
            record["status"] = "ok"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            record["peak_memory_mb"] = round(peak_memory_mb(), 1)
            self.stages.append(record)

def create_run_paths(base: str | Path, run_id: str | None = None) -> RunPaths:
    """Create a new, non-overwriting run directory."""

    if run_id is not None and (
        not isinstance(run_id, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_id)
    ):
        raise ValueError("run_id must contain 1-64 safe filename characters")
    token = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = Path(base) / f"run_{token}"
    if root.exists():
        raise FileExistsError(f"output directory already exists: {root}")
    results = root / "results"
    working = root / "working_files"
    for path in (results, working):
        path.mkdir(parents=True, exist_ok=False)
    return RunPaths(root=root, results=results, working=working)


def source_fingerprint(package_dir: str | Path, settings_path: str | Path) -> str:
    """Hash the Python files and settings, but not the data or Python setup."""

    digest = hashlib.sha256()
    sources = sorted(Path(package_dir).glob("*.py")) + [Path(settings_path)]
    for path in sources:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, default=_json_default)
    Path(path).write_text(text + "\n", encoding="utf-8")


# E1 data audit

def write_e1(
    egress: Path,
    audit_counts: pd.DataFrame,
    funnel: pd.DataFrame,
) -> list[str]:
    files = []
    files.append(_write_csv(egress / "E1_data_audit.csv", audit_counts))
    files.append(_write_csv(egress / "E1_data_funnel.csv", funnel))
    return files


def build_data_audit_counts(
    judgments: pd.DataFrame,
    audit: Any | None = None,
) -> pd.DataFrame:
    """Return identifier-free E1 marginals and the required combined audit."""

    frame = judgments.copy()
    dates = pd.to_datetime(
        frame["JudgmentDate"], format="mixed", dayfirst=True, errors="coerce"
    )
    frame["judgment_year"] = dates.dt.year.astype("Int64").astype("string").fillna("missing")
    amount = pd.to_numeric(
        frame.get("Amount", pd.Series(pd.NA, index=frame.index)), errors="coerce"
    )
    frame["amount_band"] = pd.cut(
        amount,
        bins=[-float("inf"), 500, 1_000, 5_000, 25_000, float("inf")],
        labels=["under_500", "500_999", "1000_4999", "5000_24999", "25000_plus"],
        right=False,
    ).astype("string").fillna("amount_missing")
    inserted_difference = pd.to_numeric(
        frame.get(
            "date_inserted_minus_judgment_days",
            pd.Series(pd.NA, index=frame.index),
        ),
        errors="coerce",
    )
    frame["date_inserted_minus_judgment_days_band"] = pd.cut(
        inserted_difference,
        bins=[-float("inf"), 0, 1, 2, 6, 31, float("inf")],
        labels=[
            "before_judgment",
            "same_day",
            "one_day",
            "2_to_5_days",
            "6_to_30_days",
            "31_plus_days",
        ],
        right=False,
    ).astype("string").fillna("missing")
    age = pd.to_numeric(
        frame.get("age_at_observation_months", pd.Series(pd.NA, index=frame.index)),
        errors="coerce",
    )
    frame["observation_age_band"] = pd.cut(
        age,
        bins=[-float("inf"), 1, 12, 24, 36, 48, 72, float("inf")],
        labels=[
            "up_to_1_month",
            "over_1_to_12_months",
            "over_12_to_24_months",
            "over_24_to_36_months",
            "over_36_to_48_months",
            "over_48_to_72_months",
            "over_72_months",
        ],
        right=False,
    ).astype("string").fillna("missing")

    rows: list[dict[str, Any]] = []

    def add(dimension: str, values: pd.Series) -> None:
        counts = values.astype("string").fillna("missing").value_counts(sort=False)
        denominator = max(int(counts.sum()), 1)
        rows.extend(
            {
                "dimension": dimension,
                "value": str(value),
                "rows": int(count),
                "share": int(count) / denominator,
            }
            for value, count in counts.items()
        )

    for column in (
        "JudgmentStatus",
        "DefendantType",
        "Jurisdiction",
        "judgment_year",
        "amount_band",
        "date_inserted_minus_judgment_days_band",
        "observation_age_band",
    ):
        add(column, frame[column])
    if "Date Inserted" in frame:
        inserted = pd.to_datetime(
            frame["Date Inserted"], format="mixed", dayfirst=True, errors="coerce"
        ).dropna()
        if not inserted.empty:
            for dimension, value in (
                ("Date Inserted (literal) distinct values", str(int(inserted.nunique()))),
                ("Date Inserted (literal) minimum", inserted.min().date().isoformat()),
                ("Date Inserted (literal) maximum", inserted.max().date().isoformat()),
            ):
                rows.append(
                    {
                        "dimension": dimension,
                        "value": value,
                        "rows": int(len(frame)),
                        "share": 1.0,
                    }
                )
    combined = frame[
        ["JudgmentStatus", "DefendantType", "Jurisdiction", "judgment_year"]
    ].astype("string").fillna("missing").agg(" | ".join, axis=1)
    add("status_x_type_x_jurisdiction_x_vintage", combined)
    support = int(len(frame))
    raw_schema = tuple(getattr(audit, "raw_header_schema", ())) if audit else ()
    if raw_schema:
        extra_position = 0
        for original, standardised in raw_schema:
            if standardised == "<unrecognised>":
                extra_position += 1
                original = f"extra_column_{extra_position}"
            rows.append(
                {
                    "dimension": "source_column",
                    "value": f"{original} -> {standardised}",
                    "rows": support,
                    "share": 1.0,
                }
            )
    else:
        for column in frame.columns:
            derived_prefixes = (
                "date_inserted_minus_judgment_days",
                "age_at_",
                "judgment_year",
                "amount_band",
                "observation_age_band",
            )
            if column.startswith(derived_prefixes):
                continue
            rows.append(
                {
                    "dimension": "input_column",
                    "value": str(column),
                    "rows": support,
                    "share": 1.0,
                }
            )
    if audit is not None:
        rows.append(
            {
                "dimension": "data_construct",
                "value": str(getattr(audit, "data_construct", "unknown")),
                "rows": support,
                "share": 1.0,
            }
        )
        for column in getattr(audit, "event_date_columns_present", ()):
            rows.append(
                {
                    "dimension": "event_or_snapshot_column_present",
                    "value": str(column),
                    "rows": support,
                    "share": 1.0,
                }
            )
        if not raw_schema:
            for position, _header in enumerate(
                getattr(audit, "extra_headers", ()), start=1
            ):
                rows.append(
                    {
                        "dimension": "extra_input_column_not_used",
                        "value": f"extra_column_{position}",
                        "rows": support,
                        "share": 1.0,
                    }
                )
        for column in getattr(audit, "absent_optional_columns", ()):
            rows.append(
                {
                    "dimension": "optional_column_absent_and_defaulted",
                    "value": str(column),
                    "rows": support,
                    "share": 1.0,
                }
            )
        for column, attribute in (
            ("Satisfaction Date", "satisfaction_date_present_rows"),
            ("Cancellation Date", "cancellation_date_present_rows"),
            ("Cancellation Reason", "cancellation_reason_present_rows"),
            ("Status Effective Date", "status_effective_date_present_rows"),
            ("Snapshot Date", "snapshot_date_present_rows"),
        ):
            count = int(getattr(audit, attribute, 0))
            rows.append(
                {
                    "dimension": "optional_field_populated",
                    "value": column,
                    "rows": count,
                    "share": count / max(support, 1),
                }
            )
        rows.append(
            {
                "dimension": "historical_snapshots_detected",
                "value": str(
                    bool(getattr(audit, "historical_snapshots_available", False))
                ).lower(),
                "rows": support,
                "share": 1.0,
            }
        )
        judgment_dates = pd.to_datetime(frame["JudgmentDate"], errors="coerce")
        for column in (
            "Satisfaction Date",
            "Cancellation Date",
            "Status Effective Date",
        ):
            if column not in frame:
                continue
            event_dates = pd.to_datetime(frame[column], errors="coerce")
            valid = event_dates.notna() & judgment_dates.notna()
            if not valid.any():
                continue
            delays = (event_dates.loc[valid] - judgment_dates.loc[valid]).dt.days
            event_support = int(valid.sum())
            for statistic, value in (
                ("minimum", int(delays.min())),
                ("median", float(delays.median())),
                ("maximum", int(delays.max())),
            ):
                rows.append(
                    {
                        "dimension": f"{column} minus JudgmentDate {statistic}",
                        "value": str(value),
                        "rows": event_support,
                        "share": event_support / max(support, 1),
                    }
                )
        observed_text = getattr(audit, "observation_date", None)
        if observed_text:
            observed = pd.Timestamp(observed_text)
            corporate_ew = frame["DefendantType"].eq("Corporate") & frame[
                "Jurisdiction"
            ].eq("England and Wales")
            target_support = int(corporate_ew.sum())
            satisfaction_dates = pd.to_datetime(
                frame.get(
                    "Satisfaction Date",
                    pd.Series(pd.NaT, index=frame.index),
                ),
                errors="coerce",
            )
            for months in (1, 12, 24):
                horizon = judgment_dates.map(
                    lambda value: value + pd.DateOffset(months=months)
                    if pd.notna(value)
                    else pd.NaT
                )
                eligible = corporate_ew & horizon.le(observed)
                label = (
                    "corporate_EW_older_than_one_calendar_month"
                    if months == 1
                    else f"corporate_EW_with_{months}_month_followup"
                )
                rows.append(
                    {
                        "dimension": "follow_up_check",
                        "value": label,
                        "rows": int(eligible.sum()),
                        "share": int(eligible.sum()) / max(target_support, 1),
                    }
                )
                if satisfaction_dates.notna().any():
                    within_horizon = eligible & satisfaction_dates.notna() & (
                        satisfaction_dates.le(horizon)
                    )
                    rows.append(
                        {
                            "dimension": "follow_up_check",
                            "value": f"recorded_satisfaction_within_{months}_months",
                            "rows": int(within_horizon.sum()),
                            "share": int(within_horizon.sum()) / max(target_support, 1),
                        }
                    )
        for value, count in (
            ("invalid_amount", getattr(audit, "invalid_amount_rows", 0)),
            ("missing_company_name", getattr(audit, "missing_company_name_rows", 0)),
            ("missing_postcode", getattr(audit, "missing_postcode_rows", 0)),
            (
                "date_inserted_before_judgment",
                getattr(audit, "date_inserted_before_judgment_rows", 0),
            ),
            (
                "date_inserted_after_observation_date",
                getattr(audit, "date_inserted_after_observation_rows", 0),
            ),
        ):
            rows.append(
                {
                    "dimension": "data_quality_issue",
                    "value": value,
                    "rows": int(count),
                    "share": int(count) / max(support, 1),
                }
            )
    return pd.DataFrame(rows).sort_values(["dimension", "value"], kind="stable")


# E2 matching results

def write_e2(egress: Path, diagnostics: dict[str, pd.DataFrame]) -> list[str]:
    names = {
        "tier_counts": "E2_match_coverage.csv",
        "unmatched_reasons": "E2_unmatched_reasons.csv",
        "method_counts": "E2_match_methods.csv",
        "selection_profile": "E2_linkage_profile.csv",
        "linkage_checks": "E2_linkage_checks.csv",
    }
    files = []
    for key, filename in names.items():
        table = diagnostics.get(key, pd.DataFrame())
        files.append(_write_csv(egress / filename, table))
    return files


# E5 run record and summary

def write_e5(
    egress: Path,
    recorder: RunRecorder,
    manifest: dict[str, Any],
    *,
    min_cell_n: int = 10,
) -> list[str]:
    """Write E5 after hiding small counts."""

    log = _public_run_log(pd.DataFrame(recorder.stages), min_cell_n)
    log_path = egress / "E5_run_log.csv"
    log.to_csv(log_path, index=False)
    manifest = _public_manifest(manifest, min_cell_n)
    manifest["runtime"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pid": os.getpid(),
        "peak_memory_mb": round(peak_memory_mb(), 1),
    }
    manifest_path = egress / "E5_run_manifest.json"
    write_json(manifest_path, manifest)
    return [log_path.name, manifest_path.name]


def write_summary(path: str | Path, context: dict[str, Any]) -> None:
    """Write the matching summary without row identifiers."""

    _write_matching_summary(path, context)


def _write_matching_summary(path: str | Path, context: dict[str, Any]) -> None:
    """Write the matching summary."""

    counts = context.get("counts", {})
    match = context.get("match", {})
    inserted = context.get("date_inserted", {})
    optional = context.get("optional_fields", {})
    min_cell_n = int(context.get("min_cell_n", 10))
    file_structure = {
        "status_only_unique_judgment_rows": "status only; one row per judgment",
        "status_with_event_dates_unique_judgment_rows": (
            "status and event dates; one row per judgment"
        ),
        "status_with_event_dates_and_snapshot_date": (
            "status and event dates; one dated row per judgment"
        ),
        "status_with_cancellation_reason_unique_judgment_rows": (
            "status and cancellation reason; one row per judgment"
        ),
        "status_with_cancellation_reason_and_snapshot_date": (
            "status, cancellation reason and snapshot date; one row per judgment"
        ),
        "status_with_snapshot_date_unique_judgment_rows": (
            "status and snapshot date; one row per judgment"
        ),
    }.get(str(context.get("data_construct", "unknown")), "unknown")
    lines = [
        "CONFIDENTIAL — SEND ONLY TO EDWIN",
        "RT MATCHING CHECK",
        "=================",
        "Status                        MATCHING COMPLETE",
        f"RT extract date                {context.get('observation_date', 'unknown')}",
        f"Companies House file date      {context.get('companies_house_date', 'unknown')}",
        f"File structure                 {file_structure}",
        "Stock or historical extract    not established by the file columns",
        "",
        "Optional RT fields",
        *_optional_field_lines(optional, min_cell_n),
        "",
        "Rows checked",
        f"  All RT rows                  {_fmt_count(counts.get('rows_read'), min_cell_n)}",
        "  Corporate E&W rows           "
        f"{_fmt_count(counts.get('matching_decisions'), min_cell_n)}",
        "  Missing company name (all RT) "
        f"{_fmt_count(counts.get('missing_company_name'), min_cell_n)}",
        "  Missing postcode (all RT)     "
        f"{_fmt_count(counts.get('missing_postcode'), min_cell_n)}",
        "  Date Inserted before judgment (all RT) "
        f"{_fmt_count(counts.get('date_inserted_before_judgment'), min_cell_n)}",
        "",
        "Date Inserted (RT registration date)",
        f"  Distinct values              {_fmt(inserted.get('distinct_values'))}",
        f"  Minimum value                {_fmt(inserted.get('minimum'))}",
        f"  Maximum value                {_fmt(inserted.get('maximum'))}",
        "",
        "Exact-name results for corporate E&W records",
        f"  Records checked              {_fmt_count(match.get('denominator'), min_cell_n)}",
        "  One exact live-company match "
        f"{_fmt_count(match.get('exact_unique'), min_cell_n)}",
        f"  No match                     {_fmt_count(match.get('unmatched'), min_cell_n)}",
        f"  Match rate                   {_fmt_percent(match.get('coverage'))}",
        "",
        "This run",
        "  - Checked every valid RT row.",
        "  - Made two review samples for Edwin.",
        "",
        "Limits",
        "  - After basic name cleaning, a match needs exactly one company in",
        "    the supplied live-company file with a matching current or former",
        "    name that was valid on the judgment date.",
        "  - Postcode is shown as a check. It never creates or chooses a match.",
        "  - Date Inserted is the RT registration date; its delay from",
        "    JudgmentDate is reported.",
        "  - The Companies House file contains live companies only.",
        "  - Send the ZIP file only to Edwin.",
        "",
        "Files",
        "  E1 file and row checks",
        "  E2 matching results and reasons for no match",
        "  E5 run details",
        f"Edwin's matched sample: {context.get('accepted_file', 'not produced')}",
        f"Edwin's unmatched sample: {context.get('unmatched_file', 'not produced')}",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# Memory and formatting

def peak_memory_mb() -> float:
    """Return process peak resident memory using only the standard library."""

    if os.name == "nt":
        try:
            return _windows_peak_memory_mb()
        except Exception:
            return float("nan")
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / 1024 if sys.platform != "darwin" else value / (1024 * 1024)
    except Exception:
        return float("nan")


def _windows_peak_memory_mb() -> float:
    """Read the current process peak with correctly typed 64-bit WinAPI calls."""

    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = wintypes.HANDLE
    get_process_memory = psapi.GetProcessMemoryInfo
    get_process_memory.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    get_process_memory.restype = wintypes.BOOL

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not get_process_memory(
        get_current_process(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    peak = counters.PeakWorkingSetSize / (1024 * 1024)
    if peak <= 0:
        raise OSError("Windows returned a non-positive peak working set")
    return peak


def _write_csv(path: Path, table: pd.DataFrame) -> str:
    table.to_csv(path, index=False)
    return path.name


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if isinstance(value, (int, float)) and float(value).is_integer():
        return f"{int(value):,}"
    return str(value)


def _fmt_count(value: Any, min_cell_n: int = 10) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        count = int(value)
    except (TypeError, ValueError):
        return str(value)
    return f"<{min_cell_n}" if 0 < count < min_cell_n else f"{count:,}"


def _fmt_percent(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{100 * float(value):.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _optional_field_lines(optional: Any, min_cell_n: int) -> list[str]:
    lines: list[str] = []
    for name in (
        "Satisfaction Date",
        "Cancellation Date",
        "Cancellation Reason",
        "Status Effective Date",
        "Snapshot Date",
    ):
        details = optional.get(name, {}) if isinstance(optional, dict) else {}
        presence = "present" if details.get("present") else "absent"
        count = _fmt_count(details.get("rows", 0), min_cell_n)
        lines.append(f"  {name:<23} {presence}; {count} filled")
    return lines


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot serialise {type(value).__name__}")


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


_RUN_LOG_COUNT_COLUMNS = frozenset(
    {
        "ch_rows_read",
        "ch_rows_retained",
        "judgments_matched",
        "accepted_sample_rows",
        "unmatched_sample_rows",
        "sample_rows",
        "suppressed_rows",
    }
)


def _redact_small_count(value: Any, min_cell_n: int) -> Any:
    """Replace a positive observed count below the disclosure floor."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if numeric.is_integer() and 0 < numeric < min_cell_n:
        return f"<{min_cell_n}"
    return value


def _public_run_log(log: pd.DataFrame, min_cell_n: int) -> pd.DataFrame:
    public = log.copy()
    for column in _RUN_LOG_COUNT_COLUMNS & set(public.columns):
        public[column] = public[column].map(
            lambda value: _redact_small_count(value, min_cell_n)
        )
    return public


def _public_manifest(manifest: dict[str, Any], min_cell_n: int) -> dict[str, Any]:
    """Redact small counts from the public run details."""

    public = deepcopy(manifest)
    stats = public.get("ch_index_stats", {})
    if isinstance(stats, dict):
        for key, value in list(stats.items()):
            if key != "analysis_fingerprint":
                stats[key] = _redact_small_count(value, min_cell_n)

    disclosure = public.get("disclosure", {})
    suppressed = disclosure.get("suppressed_rows", []) if isinstance(disclosure, dict) else []
    for row in suppressed:
        if isinstance(row, dict) and "rows" in row:
            row["rows"] = _redact_small_count(row["rows"], min_cell_n)
    return public
