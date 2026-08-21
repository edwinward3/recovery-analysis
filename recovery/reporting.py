"""Writes SUMMARY.txt and the detailed count, matching, model and run files."""

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
import sys
import time

import pandas as pd

from .config import Settings


# Output folders and run times

@dataclass(frozen=True, slots=True)
class RunPaths:
    root: Path
    results: Path
    working: Path
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
    results = root / "results"
    working = root / "working_files"
    models = working / "models"
    for path in (results, working, models):
        path.mkdir(parents=True, exist_ok=False)
    return RunPaths(root=root, results=results, working=working, models=models)


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
    lag = pd.to_numeric(
        frame.get("registration_lag_days", pd.Series(pd.NA, index=frame.index)),
        errors="coerce",
    )
    frame["registration_lag_band"] = pd.cut(
        lag,
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
        derived_prefixes = (
            "registration_lag_",
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
        for position, _header in enumerate(
            getattr(audit, "extra_headers", ()), start=1
        ):
            rows.append(
                {
                    "dimension": "extra_input_column_not_used",
                    # Replace unknown headings because they may contain client wording.
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
        "by_defendant_type": "E2_match_by_defendant_type.csv",
        "by_judgment_vintage": "E2_match_by_judgment_vintage.csv",
        "guard_counts": "E2_incorporation_guards.csv",
    }
    files = []
    for key, filename in names.items():
        table = diagnostics.get(key, pd.DataFrame())
        files.append(_write_csv(egress / filename, table))
    return files


# E3 and E4 model results

def build_model_tables(
    evaluation: Any,
    sensitivities: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Flatten the four aggregate model results into E3/E4 tables."""

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
    exact_unique = frame["tier"].eq("exact_unique")

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
    add(
        "primary_12_36_exact_unique_earliest",
        base & primary_age & exact_unique,
        earliest=True,
    )
    add(
        "primary_12_36_exact_unique_with_repeats",
        base & primary_age & exact_unique,
        earliest=False,
    )
    add(
        "aged_12_plus_exact_unique_earliest",
        base & aged_12_plus & exact_unique,
        earliest=True,
    )
    add(
        "aged_12_plus_exact_unique_with_repeats",
        base & aged_12_plus & exact_unique,
        earliest=False,
    )
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

    exact_primary = frame.loc[base & primary_age & exact_unique].copy()
    exact_primary["judgment_year"] = exact_primary["JudgmentDate"].dt.year.astype("Int64")
    amount_bands = pd.cut(
        exact_primary["Amount"],
        bins=[-float("inf"), 500, 1_000, 5_000, 25_000, float("inf")],
        labels=["under_500", "500_999", "1000_4999", "5000_24999", "25000_plus"],
        right=False,
    ).astype("string").fillna("amount_missing")
    exact_primary["amount_band"] = amount_bands
    for dimension in ("judgment_year", "amount_band"):
        for value, selected in exact_primary.groupby(dimension, dropna=False, sort=True):
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


# E5 run record and summary

def write_e5(
    egress: Path,
    recorder: RunRecorder,
    manifest: dict[str, Any],
    *,
    min_cell_n: int = 10,
) -> list[str]:
    """Write E5 after applying its schema-specific public redactions."""

    log = _public_run_log(pd.DataFrame(recorder.stages), min_cell_n)
    log_path = egress / "E5_run_log.csv"
    log.to_csv(log_path, index=False)
    manifest = _public_manifest(manifest, min_cell_n)
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
    """Write a short cover page for either the matching or model stage."""

    if context.get("scope") == "matching_only":
        _write_matching_summary(path, context)
        return

    counts = context.get("counts", {})
    match = context.get("match", {})
    primary = context.get("primary", {})
    exploratory = context.get("exploratory", {})
    gates = context.get("gates", {})
    splits = context.get("splits", {})
    min_cell_n = int(context.get("min_cell_n", 10))
    lines = [
        "RT INTERNAL — NOT AUTHORISED FOR EXTERNAL USE",
        "RECOVERY ANALYSIS — SUMMARY",
        "===========================",
        f"Run stage                     {context.get('stage', 'unknown')}",
        f"Run status                    {context.get('status', 'PROVISIONAL')}",
        f"Statuses observed on           {context.get('observation_date', 'unknown')}",
        "",
        "Data funnel",
        f"  Judgments read               {_fmt_count(counts.get('rows_read'), min_cell_n)}",
        "  Corporate E&W, valid label   "
        f"{_fmt_count(counts.get('corporate_ew_labelled'), min_cell_n)}",
        "  Aged 12–36 months            "
        f"{_fmt_count(counts.get('primary_age_eligible'), min_cell_n)}",
        "  Unique exact-name eligible   "
        f"{_fmt_count(counts.get('exact_unique_matched_eligible'), min_cell_n)}",
        f"  Unique companies/model rows  {_fmt_count(counts.get('model_rows'), min_cell_n)}",
        "  Satisfied / Unsatisfied      "
        f"{_fmt_count(counts.get('satisfied'), min_cell_n)} / "
        f"{_fmt_count(counts.get('unsatisfied'), min_cell_n)}",
        "",
        "Exact-name matching coverage",
        f"  Corporate denominator        {_fmt_count(match.get('denominator'), min_cell_n)}",
        "  Unique exact-name matches     "
        f"{_fmt_count(match.get('exact_unique'), min_cell_n)}",
        f"  Unmatched                     {_fmt_count(match.get('unmatched'), min_cell_n)}",
        "",
        "Primary judgment-time model",
        f"  Locked algorithm              {_fmt(primary.get('champion'))}",
        f"  Train rows / positives        {_split_count(splits, 'train', min_cell_n)}",
        f"  Validation rows / positives   {_split_count(splits, 'validation', min_cell_n)}",
        f"  Calibration rows / positives  {_split_count(splits, 'calibration', min_cell_n)}",
        f"  Test rows / positives         {_split_count(splits, 'test', min_cell_n)}",
        f"  ROC-AUC (95% CI)              {_metric_ci(primary, 'roc_auc')}",
        f"  Average precision             {_fmt_float(primary.get('average_precision'))}",
        "  Brier / prevalence baseline   "
        f"{_fmt_float(primary.get('brier'))} / "
        f"{_fmt_float(primary.get('baseline_brier'))}",
        "  Mean predicted / observed     "
        f"{_fmt_float(primary.get('mean_predicted'))} / "
        f"{_fmt_float(primary.get('observed_rate'))}",
        f"  Primary model gates           {_fmt(gates.get('primary_model'))}",
        "",
        "EXPLORATORY — CURRENT-SNAPSHOT FEATURES — NOT PROSPECTIVE",
        f"  ROC-AUC                       {_fmt_float(exploratory.get('roc_auc'))}",
        "",
        "Decisive caveats",
        "  - Date Inserted measures registration delay; it is not used for seasoning.",
        "  - Satisfied/Unsatisfied is register status when this fresh extract was run.",
        "  - Companies House bulk contains live companies and cannot match every defendant.",
        "  - Only unique exact normalized-name matches enter the model.",
        "  - Current-snapshot features may post-date the judgment and are exploratory only.",
        "  - RT permission is required before any result or model artefact leaves.",
        "",
        "Other reports in this folder",
        "  E1 data audit and exclusion funnel",
        "  E2 matching coverage and diagnosis",
        "  E3 model comparison, calibration and effects",
        "  E4 population sensitivities, lift and limitations",
        "  E5 timings, memory, fingerprints and artefact manifest",
        f"Matching pairs: {context.get('pair_file', 'not produced')}",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_matching_summary(path: str | Path, context: dict[str, Any]) -> None:
    """Write Run 1's full-dataset matching summary without model results."""

    counts = context.get("counts", {})
    match = context.get("match", {})
    min_cell_n = int(context.get("min_cell_n", 10))
    lines = [
        "RT INTERNAL — NOT AUTHORISED FOR EXTERNAL USE",
        "RECOVERY ANALYSIS — RUN 1 MATCHING SUMMARY",
        "==========================================",
        f"Run status                    {context.get('status', 'PROVISIONAL')}",
        f"RT extract date                {context.get('observation_date', 'unknown')}",
        "",
        "Full-dataset matching",
        f"  Judgments read               {_fmt_count(counts.get('rows_read'), min_cell_n)}",
        "  Matching decisions           "
        f"{_fmt_count(counts.get('matching_decisions'), min_cell_n)}",
        "  Missing company name         "
        f"{_fmt_count(counts.get('missing_company_name'), min_cell_n)}",
        f"  Missing postcode             {_fmt_count(counts.get('missing_postcode'), min_cell_n)}",
        "  Inserted before judgment     "
        f"{_fmt_count(counts.get('date_inserted_before_judgment'), min_cell_n)}",
        "",
        "Exact-name matching",
        f"  Full-dataset denominator     {_fmt_count(match.get('denominator'), min_cell_n)}",
        "  Unique exact-name matches     "
        f"{_fmt_count(match.get('exact_unique'), min_cell_n)}",
        f"  Unmatched                     {_fmt_count(match.get('unmatched'), min_cell_n)}",
        f"  Exact-name coverage           {_fmt_percent(match.get('coverage'))}",
        "",
        "What this run did",
        "  - Every valid row was sent through matching, with no age, status,",
        "    defendant-type or jurisdiction filter.",
        "  - No satisfaction model was trained or assessed.",
        "  - It created a separate file containing 1,000 matching pairs.",
        "",
        "Important limits",
        "  - A match requires one unique, date-valid exact normalized name.",
        "  - Postcode is reported but never creates or chooses a match.",
        "  - The free Companies House file contains a present-day live-company snapshot.",
        "  - Address changes, dissolved companies and non-company defendants can remain unmatched.",
        "  - RT permission is required before any result leaves.",
        "",
        "Other reports in this folder",
        "  E1 data audit and matching funnel",
        "  E2 matching coverage, methods and unmatched reasons",
        "  E5 timings, memory, settings and run record",
        f"Matching pairs: {context.get('pair_file', 'not produced')}",
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
    return f"<{min_cell_n}" if 0 <= count < min_cell_n else f"{count:,}"


def _split_count(
    splits: dict[str, Any], split: str, min_cell_n: int = 10
) -> str:
    values = splits.get(split, {})
    return (
        f"{_fmt_count(values.get('rows'), min_cell_n)} / "
        f"{_fmt_count(values.get('positive'), min_cell_n)}"
    )


def _fmt_float(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_percent(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{100 * float(value):.1f}%"
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
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


_RUN_LOG_COUNT_COLUMNS = frozenset(
    {
        "ch_rows_read",
        "ch_rows_retained",
        "judgments_matched",
        "model_rows",
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
    """Redact the declared CH, cohort and disclosure counts from public E5."""

    public = deepcopy(manifest)
    stats = public.get("ch_index_stats", {})
    if isinstance(stats, dict):
        for key, value in list(stats.items()):
            if key != "analysis_fingerprint":
                stats[key] = _redact_small_count(value, min_cell_n)

    acceptance = public.get("model_only_acceptance")
    if isinstance(acceptance, dict):
        acceptance.pop("cohort_test_counts", None)
        guards = acceptance.get("guards")
        if isinstance(guards, dict):
            guards.pop("training_prevalence", None)

    disclosure = public.get("disclosure", {})
    suppressed = disclosure.get("suppressed_rows", []) if isinstance(disclosure, dict) else []
    for row in suppressed:
        if isinstance(row, dict) and "rows" in row:
            row["rows"] = _redact_small_count(row["rows"], min_cell_n)
    return public
