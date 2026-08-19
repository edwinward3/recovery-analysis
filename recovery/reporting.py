"""Write aggregate E1–E5 reports and segregate confidential artefacts.

Inputs are in-memory aggregate tables plus run metadata. Outputs go either to
``egress_candidate`` or ``rt_internal``. This module performs no network or
shell operations and never places row-level predictions in the egress area.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
import hashlib
import json
import os
import platform
import sys
import time

import pandas as pd

from .config import Settings


@dataclass(frozen=True, slots=True)
class RunPaths:
    root: Path
    egress: Path
    internal: Path
    models: Path


@dataclass(slots=True)
class RunRecorder:
    stages: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

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

    def warn(self, message: str) -> None:
        self.warnings.append(str(message))


def create_run_paths(base: str | Path, stage: str, run_id: str | None = None) -> RunPaths:
    """Create a new, non-overwriting run directory."""

    if stage not in {"diagnostic", "locked"}:
        raise ValueError("stage must be 'diagnostic' or 'locked'")
    token = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = Path(base) / f"{stage}_{token}"
    if root.exists():
        raise FileExistsError(f"output directory already exists: {root}")
    egress = root / "egress_candidate"
    internal = root / "rt_internal"
    models = internal / "models"
    for path in (egress, internal, models):
        path.mkdir(parents=True, exist_ok=False)
    return RunPaths(root=root, egress=egress, internal=internal, models=models)


def source_fingerprint(package_dir: str | Path, settings_path: str | Path) -> str:
    """Hash the readable source and settings to identify the exact run inputs."""

    digest = hashlib.sha256()
    sources = sorted(Path(package_dir).glob("*.py")) + [Path(settings_path)]
    for path in sources:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


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
    lag = pd.to_numeric(
        frame.get("registration_lag_days", pd.Series(pd.NA, index=frame.index)),
        errors="coerce",
    )
    frame["registration_lag_band"] = pd.cut(
        lag,
        bins=[-float("inf"), 1, 2, 6, 31, float("inf")],
        labels=["same_day", "one_day", "2_to_5_days", "6_to_30_days", "31_plus_days"],
        right=False,
    ).astype("string").fillna("missing")
    age = pd.to_numeric(
        frame.get("age_at_observation_months", pd.Series(pd.NA, index=frame.index)),
        errors="coerce",
    )
    frame["observation_age_band"] = pd.cut(
        age,
        bins=[-float("inf"), 12, 36, float("inf")],
        labels=["under_12_months", "12_to_36_months", "over_36_months"],
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
        "registration_lag_band",
        "observation_age_band",
    ):
        add(column, frame[column])
    combined = frame[
        ["JudgmentStatus", "DefendantType", "Jurisdiction", "judgment_year"]
    ].astype("string").fillna("missing").agg(" | ".join, axis=1)
    add("status_x_type_x_jurisdiction_x_vintage", combined)
    support = int(len(frame))
    for column in frame.columns:
        if column.startswith(("registration_lag_", "age_at_", "judgment_year", "amount_band", "observation_age_band")):
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
        for header in getattr(audit, "extra_headers", ()):
            rows.append(
                {
                    "dimension": "extra_input_column_not_used",
                    "value": str(header),
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
        for value, count in (
            ("invalid_amount", getattr(audit, "invalid_amount_rows", 0)),
            ("missing_company_name", getattr(audit, "missing_company_name_rows", 0)),
            ("missing_postcode", getattr(audit, "missing_postcode_rows", 0)),
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


def write_e2(egress: Path, diagnostics: dict[str, pd.DataFrame]) -> list[str]:
    names = {
        "tier_counts": "E2_match_coverage.csv",
        "unmatched_reasons": "E2_unmatched_reasons.csv",
        "method_counts": "E2_match_methods.csv",
        "by_defendant_type": "E2_match_by_defendant_type.csv",
        "by_judgment_vintage": "E2_match_by_judgment_vintage.csv",
        "guard_counts": "E2_incorporation_guards.csv",
    }
    files = []
    for key, filename in names.items():
        table = diagnostics.get(key, pd.DataFrame())
        files.append(_write_csv(egress / filename, table))
    return files


def build_model_tables(
    evaluation: Any,
    sensitivities: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Flatten the four aggregate model results into reviewable E3/E4 tables."""

    comparison: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    lift: list[dict[str, Any]] = []
    for key, run in sorted(evaluation.runs.items()):
        test = run.test_metrics_calibrated
        intervals = run.bootstrap_intervals
        comparison.append(
            {
                "model": key,
                "family": run.family,
                "algorithm": run.algorithm,
                "role": "primary_candidate" if run.family == "prospective" else "exploratory_only",
                "validation_selected": evaluation.champions.get(run.family) == run.algorithm,
                "calibration_method": run.calibration.method,
                "test_rows": test.get("n"),
                "test_positive": test.get("n_positive"),
                "test_negative": test.get("n_negative"),
                "validation_roc_auc": run.validation_metrics.get("roc_auc"),
                "validation_brier": run.validation_metrics.get("brier"),
                "test_roc_auc": test.get("roc_auc"),
                "test_roc_auc_lower_95": intervals.get("roc_auc", {}).get("lower"),
                "test_roc_auc_upper_95": intervals.get("roc_auc", {}).get("upper"),
                "test_average_precision": test.get("average_precision"),
                "test_average_precision_lower_95": intervals.get(
                    "average_precision", {}
                ).get("lower"),
                "test_average_precision_upper_95": intervals.get(
                    "average_precision", {}
                ).get("upper"),
                "test_brier": test.get("brier"),
                "test_brier_lower_95": intervals.get("brier", {}).get("lower"),
                "test_brier_upper_95": intervals.get("brier", {}).get("upper"),
                "training_prevalence_brier": test.get("null_brier"),
                "test_log_loss": test.get("log_loss"),
                "test_log_loss_lower_95": intervals.get("log_loss", {}).get("lower"),
                "test_log_loss_upper_95": intervals.get("log_loss", {}).get("upper"),
                "mean_prediction": test.get("mean_prediction"),
                "observed_rate": test.get("base_rate"),
                "calibration_gap_lower_95": intervals.get("calibration_gap", {}).get(
                    "lower"
                ),
                "calibration_gap_upper_95": intervals.get("calibration_gap", {}).get(
                    "upper"
                ),
                "calibration_intercept": test.get("calibration_intercept"),
                "calibration_slope": test.get("calibration_slope"),
            }
        )
        for row in run.reliability_bins:
            calibration.append({"model": key, **row})
        support = int(test.get("n", 0))
        for row in run.feature_effects:
            effects.append(
                {
                    "model": key,
                    "family": run.family,
                    "algorithm": run.algorithm,
                    "support_rows": support,
                    "feature_name": row["feature"],
                    "effect_type": row["effect_type"],
                    "value": row["value"],
                    "absolute_share": row["absolute_share"],
                    "direction": row["direction"],
                }
            )
        top = test.get("top_decile", {})
        lift.append(
            {
                "model": key,
                "selected": top.get("selected"),
                "selected_positive": top.get("selected_positive"),
                "selected_negative": top.get("selected_negative"),
                "positive_rate": top.get("positive_rate"),
                "base_rate": test.get("base_rate"),
                "lift": top.get("lift"),
                "lift_lower_95": intervals.get("top_decile_lift", {}).get("lower"),
                "lift_upper_95": intervals.get("top_decile_lift", {}).get("upper"),
                "recall": top.get("recall"),
            }
        )

    split_rows = [
        {"split": split, **counts}
        for split, counts in evaluation.cohort.split_counts.items()
    ]
    return {
        "comparison": pd.DataFrame(comparison),
        "calibration": pd.DataFrame(calibration),
        "feature_effects": pd.DataFrame(effects),
        "split_counts": pd.DataFrame(split_rows),
        "sensitivities": (
            sensitivities.copy() if sensitivities is not None else pd.DataFrame()
        ),
        "lift": pd.DataFrame(lift),
    }


def build_population_sensitivities(
    judgments: pd.DataFrame,
    matches: pd.DataFrame,
    observation_date: str | pd.Timestamp,
    settings: Settings,
) -> pd.DataFrame:
    """Describe locked and alternative populations without fitting extra models."""

    columns = [
        "ID",
        "JudgmentDate",
        "JudgmentStatus",
        "DefendantType",
        "Jurisdiction",
        "Amount",
    ]
    left = judgments[[column for column in columns if column in judgments]].copy()
    if "Amount" not in left:
        left["Amount"] = pd.NA
    right = matches[["ID", "tier", "matched_company_number"]].copy()
    frame = left.merge(right, on="ID", how="left", validate="one_to_one")
    frame["JudgmentDate"] = pd.to_datetime(
        frame["JudgmentDate"], format="mixed", dayfirst=True, errors="coerce"
    )
    frame["Amount"] = pd.to_numeric(frame["Amount"], errors="coerce")
    observed = pd.Timestamp(observation_date).normalize()
    lower = observed - pd.DateOffset(months=settings.primary_max_months)
    upper = observed - pd.DateOffset(months=settings.primary_min_months)
    primary_age = frame["JudgmentDate"].between(lower, upper, inclusive="both")
    aged_12_plus = frame["JudgmentDate"] <= upper
    corporate = frame["DefendantType"].eq("Corporate")
    noncorporate = frame["DefendantType"].eq("Non-Corporate")
    ew = frame["Jurisdiction"].eq("England and Wales")
    scotland = frame["Jurisdiction"].eq("Scotland")
    labelled = frame["JudgmentStatus"].isin(("Satisfied", "Unsatisfied"))
    cancelled = frame["JudgmentStatus"].eq("Cancelled")
    auto = frame["tier"].eq("auto")
    auto_review = frame["tier"].isin(("auto", "review"))

    rows: list[dict[str, Any]] = []

    def add(name: str, mask: pd.Series, *, earliest: bool) -> None:
        selected = frame.loc[mask].sort_values(
            ["JudgmentDate", "matched_company_number", "ID"], kind="stable"
        )
        if earliest:
            matched = selected["matched_company_number"].fillna("").ne("")
            selected = pd.concat(
                [
                    selected.loc[matched].drop_duplicates(
                        "matched_company_number", keep="first"
                    ),
                    selected.loc[~matched],
                ],
                ignore_index=True,
            )
        positive = int(selected["JudgmentStatus"].eq("Satisfied").sum())
        negative = int(selected["JudgmentStatus"].eq("Unsatisfied").sum())
        denominator = positive + negative
        rows.append(
            {
                "analysis": "population",
                "stratum": name,
                "rows": int(len(selected)),
                "unique_companies": int(
                    selected["matched_company_number"].replace("", pd.NA).nunique()
                ),
                "positive": positive,
                "negative": negative,
                "recorded_satisfied_rate": positive / denominator if denominator else pd.NA,
            }
        )

    base = corporate & ew & labelled
    add("primary_12_36_auto_unique_earliest", base & primary_age & auto, earliest=True)
    add("primary_12_36_auto_with_repeats", base & primary_age & auto, earliest=False)
    add("aged_12_plus_auto_unique_earliest", base & aged_12_plus & auto, earliest=True)
    add("aged_12_plus_auto_with_repeats", base & aged_12_plus & auto, earliest=False)
    add("primary_12_36_auto_plus_review", base & primary_age & auto_review, earliest=True)
    add(
        "noncorporate_12_36_kept_separate",
        noncorporate & ew & labelled & primary_age,
        earliest=False,
    )
    add(
        "scotland_12_36_kept_separate",
        corporate & scotland & labelled & primary_age,
        earliest=False,
    )
    add(
        "cancelled_12_36_kept_separate",
        corporate & ew & cancelled & primary_age,
        earliest=False,
    )

    eligible = frame.loc[base & primary_age].copy()
    for tier, selected in eligible.groupby("tier", dropna=False, sort=True):
        add(f"match_tier_{tier}", frame.index.isin(selected.index), earliest=False)

    auto_primary = frame.loc[base & primary_age & auto].copy()
    auto_primary["judgment_year"] = auto_primary["JudgmentDate"].dt.year.astype("Int64")
    amount_bands = pd.cut(
        auto_primary["Amount"],
        bins=[-float("inf"), 500, 1_000, 5_000, 25_000, float("inf")],
        labels=["under_500", "500_999", "1000_4999", "5000_24999", "25000_plus"],
        right=False,
    ).astype("string").fillna("amount_missing")
    auto_primary["amount_band"] = amount_bands
    for dimension in ("judgment_year", "amount_band"):
        for value, selected in auto_primary.groupby(dimension, dropna=False, sort=True):
            positive = int(selected["JudgmentStatus"].eq("Satisfied").sum())
            negative = int(selected["JudgmentStatus"].eq("Unsatisfied").sum())
            rows.append(
                {
                    "analysis": dimension,
                    "stratum": str(value),
                    "rows": int(len(selected)),
                    "unique_companies": int(selected["matched_company_number"].nunique()),
                    "positive": positive,
                    "negative": negative,
                    "recorded_satisfied_rate": (
                        positive / (positive + negative) if positive + negative else pd.NA
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_e3_e4(
    egress: Path,
    model_tables: dict[str, pd.DataFrame],
    limitations: str,
) -> list[str]:
    names = {
        "comparison": "E3_model_comparison.csv",
        "calibration": "E3_calibration.csv",
        "feature_effects": "E3_feature_effects.csv",
        "split_counts": "E3_split_counts.csv",
        "sensitivities": "E4_sensitivities.csv",
        "lift": "E4_lift.csv",
    }
    files = []
    for key, filename in names.items():
        files.append(_write_csv(egress / filename, model_tables.get(key, pd.DataFrame())))
    limitation_path = egress / "E4_limitations.txt"
    limitation_path.write_text(limitations.rstrip() + "\n", encoding="utf-8")
    files.append(limitation_path.name)
    return files


def write_e5(
    egress: Path,
    recorder: RunRecorder,
    manifest: dict[str, Any],
) -> list[str]:
    log = pd.DataFrame(recorder.stages)
    log_path = egress / "E5_run_log.csv"
    log.to_csv(log_path, index=False)
    manifest = dict(manifest)
    manifest["warnings"] = recorder.warnings
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
    """Write the deliberately short cover page; detailed values remain E1–E5."""

    counts = context.get("counts", {})
    match = context.get("match", {})
    primary = context.get("primary", {})
    exploratory = context.get("exploratory", {})
    gates = context.get("gates", {})
    splits = context.get("splits", {})
    lines = [
        "DRAFT — RT REVIEW REQUIRED — NOT AUTHORISED FOR EXTERNAL USE",
        "RECOVERY ANALYSIS — SUMMARY",
        "===========================",
        f"Run stage                     {context.get('stage', 'unknown')}",
        f"Run status                    {context.get('status', 'PROVISIONAL')}",
        f"Statuses observed on           {context.get('observation_date', 'unknown')}",
        "",
        "Data funnel",
        f"  Judgments read               {_fmt_count(counts.get('rows_read'))}",
        f"  Corporate E&W, valid label   {_fmt_count(counts.get('corporate_ew_labelled'))}",
        f"  Aged 12–36 months            {_fmt_count(counts.get('primary_age_eligible'))}",
        f"  Auto-matched eligible        {_fmt_count(counts.get('auto_matched_eligible'))}",
        f"  Unique companies/model rows  {_fmt_count(counts.get('model_rows'))}",
        f"  Satisfied / Unsatisfied      {_fmt_count(counts.get('satisfied'))} / {_fmt_count(counts.get('unsatisfied'))}",
        "",
        "Matching coverage (coverage, not accuracy)",
        f"  Corporate denominator        {_fmt_count(match.get('denominator'))}",
        f"  Auto                          {_fmt_count(match.get('auto'))}",
        f"  Review                        {_fmt_count(match.get('review'))}",
        f"  Fallback review               {_fmt_count(match.get('fallback_review'))}",
        f"  Unmatched                     {_fmt_count(match.get('unmatched'))}",
        "",
        "Primary judgment-time model",
        f"  Locked algorithm              {_fmt(primary.get('champion'))}",
        f"  Train rows / positives        {_split_count(splits, 'train')}",
        f"  Validation rows / positives   {_split_count(splits, 'validation')}",
        f"  Calibration rows / positives  {_split_count(splits, 'calibration')}",
        f"  Test rows / positives         {_split_count(splits, 'test')}",
        f"  ROC-AUC (95% CI)              {_metric_ci(primary, 'roc_auc')}",
        f"  Average precision             {_fmt_float(primary.get('average_precision'))}",
        f"  Brier / prevalence baseline   {_fmt_float(primary.get('brier'))} / {_fmt_float(primary.get('baseline_brier'))}",
        f"  Mean predicted / observed     {_fmt_float(primary.get('mean_predicted'))} / {_fmt_float(primary.get('observed_rate'))}",
        f"  Primary model gates           {_fmt(gates.get('primary_model'))}",
        "",
        "EXPLORATORY — CURRENT-SNAPSHOT FEATURES — NOT PROSPECTIVE",
        f"  ROC-AUC                       {_fmt_float(exploratory.get('roc_auc'))}",
        "",
        "Decisive caveats",
        "  - Date Inserted measures registration delay; it is not used for seasoning.",
        "  - Satisfied/Unsatisfied is register status when this fresh extract was run.",
        "  - Companies House bulk contains live companies and cannot match every defendant.",
        "  - Review and fallback matches do not enter the headline model.",
        "  - Current-snapshot features may post-date the judgment and are exploratory only.",
        "  - RT review and permission are required before any result or model artefact leaves.",
        "",
        "Detailed aggregate reports in egress_candidate",
        "  E1 data audit and exclusion funnel",
        "  E2 matching coverage and diagnosis",
        "  E3 model comparison, calibration and effects",
        "  E4 population sensitivities, lift and limitations",
        "  E5 timings, memory, fingerprints and artefact manifest",
        f"RT-internal pair file: {context.get('pair_file', 'not produced')} (DO NOT EGRESS)",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    return f"<{min_cell_n}" if 0 <= count < min_cell_n else f"{count:,}"


def _split_count(splits: dict[str, Any], split: str) -> str:
    values = splits.get(split, {})
    return f"{_fmt_count(values.get('rows'))} / {_fmt_count(values.get('positive'))}"


def _fmt_float(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _metric_ci(metrics: dict[str, Any], stem: str) -> str:
    value = _fmt_float(metrics.get(stem))
    lo = _fmt_float(metrics.get(f"{stem}_ci_low"))
    hi = _fmt_float(metrics.get(f"{stem}_ci_high"))
    return value if lo == "n/a" or hi == "n/a" else f"{value} ({lo}–{hi})"


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot serialise {type(value).__name__}")


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
