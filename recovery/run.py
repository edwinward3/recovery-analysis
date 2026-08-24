"""Run the matching check."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from typing import Any, Callable, Sequence
import argparse
import hashlib
import re
import shutil
import sys
import time
import zipfile

import pandas as pd

from .config import Settings, load_settings
from .data import DataAudit, read_rt_extract
from .disclosure import stage_egress
from .matching import (
    ACCEPTED_LINKAGE_VALIDATION_FILENAME,
    UNMATCHED_LINKAGE_VALIDATION_FILENAME,
    CHIndex,
    accepted_validation_sample,
    build_relevant_ch_index,
    match_diagnostics,
    match_judgments,
    unmatched_validation_sample,
)
from .outcomes import (
    FIXED_HORIZONS,
    LANDMARK_MONTHS,
    aalen_johansen_monthly,
    cross_sectional_status_aggregates,
    mature_fixed_horizon_outcomes,
    outcome_validity_gate,
    registration_working_day_aggregates,
)
from .prediction import (
    BOOTSTRAP_REPLICATES,
    CANCELLATION_TREATMENT,
    CAPACITIES,
    JUDGMENT_AGE_BASELINE,
    MIN_NONTRAIN_CLASS,
    MODEL_NAMES,
    PRIMARY_ESTIMAND,
    SAFE_FEATURES,
    SPLIT_SHARES,
    build_12_month_landmark_cohort,
    run_12_month_prediction,
)
from .reporting import (
    RunPaths,
    RunRecorder,
    build_artifact_manifest,
    build_data_audit_counts,
    build_linkage_comparison,
    build_output_dictionary,
    build_validation_sampling,
    create_run_paths,
    source_fingerprint,
    write_e1,
    write_e2,
    write_e5,
    write_summary,
    write_tables,
)


MAX_CH_SNAPSHOT_LAG_DAYS = 35
ELAPSED_UPDATE_SECONDS = 300
_GLOBAL_OUTCOME_BLOCKERS = frozenset(
    {
        "snapshot_date_does_not_match_extract",
        "snapshot_date_missing",
        "snapshot_date_not_single_value",
        "unknown_outcome_or_history_header",
    }
)


class RunFailure(RuntimeError):
    """A problem that stops the run."""


def analyze(
    *,
    judgments_path: str | Path,
    companies_house_path: str | Path,
    observation_date: str | pd.Timestamp | None,
    companies_house_date: str | pd.Timestamp | None = None,
    settings_path: str | Path,
    output_base: str | Path,
    run_id: str | None = None,
    _match_validator: Callable[[pd.DataFrame], None] | None = None,
) -> RunPaths:
    """Run the matching check."""

    if observation_date is None:
        raise RunFailure("RT extract date is required; it must not default to today")
    if companies_house_date is None:
        raise RunFailure("Companies House file date is required")
    observed = _declared_date(observation_date, "RT extract date")
    ch_observed = _declared_date(companies_house_date, "Companies House file date")
    companies_file = Path(companies_house_path).resolve()
    _validate_ch_snapshot(companies_file, observed, ch_observed)
    settings_source = Path(settings_path).resolve()
    settings = load_settings(settings_source)

    paths: RunPaths | None = None
    try:
        paths = create_run_paths(output_base, run_id)
        result = _analyze_created_run(
            judgments_path=judgments_path,
            companies_house_path=companies_file,
            observed=observed,
            ch_observed=ch_observed,
            settings_source=settings_source,
            settings=settings,
            paths=paths,
            _match_validator=_match_validator,
        )
        return result
    except BaseException as exc:
        if paths is not None and paths.root.exists():
            try:
                shutil.rmtree(paths.root)
            except OSError as cleanup_error:
                raise RunFailure(
                    "the run failed and its incomplete files could not be removed: "
                    f"{cleanup_error}"
                ) from exc
        raise


def _analyze_created_run(
    *,
    judgments_path: str | Path,
    companies_house_path: Path,
    observed: pd.Timestamp,
    ch_observed: pd.Timestamp,
    settings_source: Path,
    settings: Settings,
    paths: RunPaths,
    _match_validator: Callable[[pd.DataFrame], None] | None,
) -> RunPaths:
    aggregate = paths.root / ".aggregate_staging"
    aggregate.mkdir()
    recorder = RunRecorder()
    judgments_file = Path(judgments_path).resolve()

    with recorder.stage("E1_read_validate_schema"):
        judgments, audit = read_rt_extract(judgments_file, observed)

    with recorder.stage("CH_stream_index") as record:
        ch_index = build_relevant_ch_index(judgments, companies_house_path)
        record["ch_rows_read"] = ch_index.stats.get("ch_rows_read")
        record["ch_rows_retained"] = ch_index.stats.get("ch_rows_retained")

    with recorder.stage("E2_exact_linkage") as record:
        matches = match_judgments(judgments, ch_index)
        if _match_validator is not None:
            _match_validator(matches)
        linkage_judgments, linkage_matches = _linkage_target(judgments, matches)
        diagnostics = match_diagnostics(linkage_judgments, linkage_matches)
        record["judgments_matched"] = int(
            linkage_matches["tier"].eq("exact_unique").sum()
        )
    with recorder.stage("E2_probability_validation_samples") as record:
        accepted_sample = accepted_validation_sample(
            linkage_judgments,
            linkage_matches,
            settings,
            seed=settings.diagnostic_seed,
        )
        unmatched_sample = unmatched_validation_sample(
            linkage_judgments,
            linkage_matches,
            settings.sample_size,
            seed=settings.diagnostic_seed,
        )
        accepted_sample.to_csv(
            paths.working / ACCEPTED_LINKAGE_VALIDATION_FILENAME,
            index=False,
            encoding="utf-8-sig",
        )
        unmatched_sample.to_csv(
            paths.working / UNMATCHED_LINKAGE_VALIDATION_FILENAME,
            index=False,
            encoding="utf-8-sig",
        )
        record["accepted_sample_rows"] = len(accepted_sample)
        record["unmatched_sample_rows"] = len(unmatched_sample)
        record["sample_seed"] = settings.diagnostic_seed

    with recorder.stage("E1_registration_delay") as record:
        try:
            registration = registration_working_day_aggregates(
                linkage_judgments, observed
            )
            registration_tables = _registration_tables(registration)
            registration_status = "completed"
        except ValueError as exc:
            registration_tables = {
                "E1_registration_gate.csv": pd.DataFrame(
                    [{"status": "not_run", "reason": str(exc)}]
                )
            }
            registration_status = "not_run"
        record["status_selected"] = registration_status

    with recorder.stage("E3_outcomes") as record:
        outcome_gate = outcome_validity_gate(linkage_judgments, observed)
        global_outcome_issues = _GLOBAL_OUTCOME_BLOCKERS.intersection(
            audit.outcome_issues
        )
        if global_outcome_issues:
            outcome_gate = {
                **outcome_gate,
                "design": "blocked",
                "invalid_counts": {
                    **outcome_gate["invalid_counts"],
                    **{
                        f"{issue}_rows": audit.outcome_issues[issue]
                        for issue in sorted(global_outcome_issues)
                    },
                },
                "reasons": tuple(
                    f"{issue}={audit.outcome_issues[issue]}"
                    for issue in sorted(global_outcome_issues)
                ),
            }
        outcome_tables = _outcome_tables(
            linkage_judgments,
            observed,
            outcome_gate,
        )
        outcome_status = {
            "longitudinal": "longitudinal",
            "cross_sectional": "cross-sectional only",
            "blocked": "not run",
        }[str(outcome_gate["design"])]
        record["analysis_selected"] = outcome_status

    with recorder.stage("E4_prediction") as record:
        prediction_tables, prediction_status = _prediction_tables(
            linkage_judgments,
            linkage_matches,
            observed,
            outcome_gate,
            settings,
        )
        record["analysis_selected"] = prediction_status

    with recorder.stage("aggregate_reports"):
        audit_counts = build_data_audit_counts(judgments, audit)
        funnel = _matching_funnel(judgments, linkage_matches)
        write_e1(aggregate, audit_counts, funnel)
        write_tables(aggregate, registration_tables)
        write_e2(aggregate, diagnostics)
        write_tables(
            aggregate,
            {
                "E2_population_comparison.csv": build_linkage_comparison(
                    linkage_judgments, linkage_matches
                ),
                "E2_validation_sampling.csv": build_validation_sampling(
                    linkage_matches,
                    len(accepted_sample),
                    len(unmatched_sample),
                    settings.diagnostic_seed,
                ),
                **outcome_tables,
                **prediction_tables,
            },
        )
        write_summary(
            aggregate / "SUMMARY.txt",
            _summary_context(
                audit=audit,
                ch_observed=ch_observed,
                judgments=judgments,
                matches=matches,
                settings=settings,
                outcome_status=outcome_status,
                prediction_status=prediction_status,
            ),
        )

    pd.DataFrame(
        columns=["artifact_name", "bytes", "sha256", "rows"]
    ).to_csv(aggregate / "E5_artifact_manifest.csv", index=False)
    build_output_dictionary(aggregate).to_csv(
        aggregate / "E5_output_dictionary.csv", index=False
    )
    allowlist = _analysis_allowlist(aggregate)
    allowlist.update(
        {
            "E5_run_log.csv": None,
            "E5_run_manifest.json": None,
        }
    )
    working_files = sorted(
        path.relative_to(paths.working).as_posix()
        for path in paths.working.rglob("*")
        if path.is_file()
    )
    manifest = _run_manifest(
        paths=paths,
        observed=observed,
        ch_observed=ch_observed,
        audit=audit,
        ch_index=ch_index,
        settings=settings,
        settings_source=settings_source,
        judgments_file=judgments_file,
        companies_file=companies_house_path,
        working_files=working_files,
        allowlist=allowlist,
        outcome_status=outcome_status,
        prediction_status=prediction_status,
    )
    known_identifiers = _bounded_known_identifiers(
        pd.concat([accepted_sample, unmatched_sample], ignore_index=True, sort=False)
    )
    report_allowlist = {
        name: policy
        for name, policy in allowlist.items()
        if name
        not in {
            "E5_artifact_manifest.csv",
            "E5_run_log.csv",
            "E5_run_manifest.json",
        }
    }
    with recorder.stage("E5_disclosure_gate") as record:
        with TemporaryDirectory(prefix=".disclosure-preview-", dir=paths.root) as temp:
            preview = stage_egress(
                aggregate,
                Path(temp) / "egress",
                allowlist=report_allowlist,
                min_cell_n=settings.min_cell_n,
                known_identifiers=known_identifiers,
            )
        record["staged_files"] = len(allowlist)
        record["suppressed_rows"] = sum(count for _, count in preview.suppressed_rows)
    manifest["disclosure"].update(
        {
            "status": "pass",
            "staged_files": sorted(allowlist),
            "suppressed_rows": [
                {"file": filename, "rows": count}
                for filename, count in preview.suppressed_rows
            ],
        }
    )
    write_e5(aggregate, recorder, manifest, min_cell_n=settings.min_cell_n)
    build_output_dictionary(aggregate).to_csv(
        aggregate / "E5_output_dictionary.csv", index=False
    )
    full_preview_allowlist = {
        name: policy
        for name, policy in allowlist.items()
        if name != "E5_artifact_manifest.csv"
    }
    with TemporaryDirectory(prefix=".disclosure-final-", dir=paths.root) as temp:
        preview_root = Path(temp) / "egress"
        final_preview = stage_egress(
            aggregate,
            preview_root,
            allowlist=full_preview_allowlist,
            min_cell_n=settings.min_cell_n,
            known_identifiers=known_identifiers,
        )
        if tuple(final_preview.suppressed_rows) != tuple(preview.suppressed_rows):
            raise RunFailure("the final reports differed from the disclosure preview")
        build_artifact_manifest(preview_root).to_csv(
            aggregate / "E5_artifact_manifest.csv", index=False
        )
    disclosure = stage_egress(
        aggregate,
        paths.results,
        allowlist=allowlist,
        min_cell_n=settings.min_cell_n,
        known_identifiers=known_identifiers,
    )
    if (
        tuple(disclosure.suppressed_rows) != tuple(final_preview.suppressed_rows)
        or set(disclosure.staged_files) != set(allowlist)
    ):
        raise RunFailure("final disclosure copy differed from its checked preview")
    _verify_artifact_manifest(paths.results)
    shutil.rmtree(aggregate)
    return paths


def _linkage_target(
    judgments: pd.DataFrame, matches: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = judgments["DefendantType"].eq("Corporate") & judgments["Jurisdiction"].eq(
        "England and Wales"
    )
    target = judgments.loc[mask].copy()
    identifiers = set(target["ID"].astype(str))
    decisions = matches.loc[matches["ID"].astype(str).isin(identifiers)].copy()
    if len(target) != len(decisions):
        raise RunFailure(
            "corporate England and Wales records do not have one match decision each"
        )
    return target, decisions


def _matching_funnel(
    judgments: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"stage": "judgments_read", "rows": int(len(judgments))},
            {"stage": "matching_decisions", "rows": int(len(matches))},
        ]
    )


def _outcome_tables(
    judgments: pd.DataFrame,
    observed: pd.Timestamp,
    gate: dict[str, object],
) -> dict[str, pd.DataFrame]:
    reasons = "; ".join(str(value).split("=", 1)[0] for value in gate["reasons"])
    selected = str(gate["design"])
    if selected not in {"longitudinal", "cross_sectional", "blocked"}:
        raise RunFailure(f"unknown outcome analysis: {selected}")
    gate_table = pd.DataFrame(
        [
            {
                "selected_analysis": selected,
                "longitudinal_status": (
                    "completed" if selected == "longitudinal" else "not_run"
                ),
                "reason": reasons,
                "population_rows": gate["rows"],
                "landmark_at_risk_rows": gate["landmark_at_risk_rows"],
                "mature_12_month_rows": gate["mature_12_month_rows"],
                "mature_24_month_rows": gate["mature_24_month_rows"],
                "satisfaction_date_supplied": gate[
                    "satisfaction_date_source_present"
                ],
                "cancellation_date_supplied": gate[
                    "cancellation_date_source_present"
                ],
                "extract_date": gate["extract_date"],
                "population": "corporate_England_and_Wales_records_present_at_extract",
                "retention_or_removal_confirmation": "required",
                "uncertainty": "exact_counts_for_supplied_extract_no_sampling_interval",
                **gate["invalid_counts"],
            }
        ]
    )
    tables = {"E3_outcome_gate.csv": gate_table}
    if selected == "blocked":
        return tables
    if selected == "cross_sectional":
        tables["E3_status_at_extract.csv"] = cross_sectional_status_aggregates(
            judgments, observed
        )
        return tables

    curve = aalen_johansen_monthly(judgments, observed)
    tables["E3_cumulative_incidence.csv"] = curve[
        [
            "month",
            "at_risk",
            "satisfaction_events",
            "cancellation_events",
            "censored",
            "satisfaction_cif",
            "cancellation_cif",
            "event_free_survival",
            "complete_month",
            "extract_date",
            "time_origin",
        ]
    ]
    fixed = mature_fixed_horizon_outcomes(judgments, observed)
    parts = []
    for status in ("satisfied", "cancelled", "unsatisfied"):
        part = fixed[
            [
                "horizon_months",
                "judgment_cohort",
                "eligible_rows",
                f"{status}_rows",
                f"{status}_share",
                "excluded_late_registration_rows",
                "excluded_cancelled_by_landmark_rows",
                "excluded_unobservable_landmark_rows",
                "excluded_immature_rows",
                "extract_date",
                "time_origin",
            ]
        ].rename(
            columns={f"{status}_rows": "rows", f"{status}_share": "share"}
        )
        part.insert(2, "status", status.capitalize())
        parts.append(part)
    status_order = pd.CategoricalDtype(
        ["Satisfied", "Cancelled", "Unsatisfied"], ordered=True
    )
    fixed_long = pd.concat(parts, ignore_index=True)
    fixed_long["status"] = fixed_long["status"].astype(status_order)
    tables["E3_fixed_horizon.csv"] = fixed_long.sort_values(
        ["horizon_months", "judgment_cohort", "status"], kind="stable"
    ).reset_index(drop=True)
    return tables


def _registration_tables(values: dict[str, object]) -> dict[str, pd.DataFrame]:
    valid = int(values["valid_registration_rows"])
    total = int(values["england_wales_rows"])
    same_calendar = int(values["same_calendar_day_rows"])
    next_calendar = int(values["next_calendar_day_rows"])
    within_working = int(values["within_one_working_day_rows"])
    nonworking = int(values["inserted_on_non_working_day_rows"])
    groups = {
        "validity": {
            "all_records": total,
            "valid_for_delay_calculation": valid,
            "excluded_date_anomaly": total - valid,
        },
        "calendar_day_delay": {
            "valid_records": valid,
            "same_day": same_calendar,
            "next_day": next_calendar,
            "later": valid - same_calendar - next_calendar,
        },
        "working_day_delay": {
            "valid_records": valid,
            "within_one_working_day": within_working,
            "more_than_one_working_day": valid - within_working,
        },
        "registration_day": {
            "valid_records": valid,
            "working_day": valid - nonworking,
            "non_working_day": nonworking,
        },
    }
    rows = []
    for dimension, counts in groups.items():
        denominator = max(counts["valid_records"] if "valid_records" in counts else total, 1)
        rows.extend(
            {
                "dimension": dimension,
                "measure": measure,
                "rows": count,
                "share": count / denominator,
            }
            for measure, count in counts.items()
        )
    statistics = pd.DataFrame(
        [
            {
                "rows": valid,
                "calendar_day_lag_median": values["calendar_day_lag_median"],
                "calendar_day_lag_p95": values["calendar_day_lag_p95"],
                "calendar_day_lag_max": values["calendar_day_lag_max"],
                "working_day_lag_median": values["working_day_lag_median"],
                "working_day_lag_p95": values["working_day_lag_p95"],
                "working_day_lag_max": values["working_day_lag_max"],
                "holiday_calendar_source": values["holiday_calendar_source"],
                "holiday_calendar_start": values["holiday_calendar_start"],
                "holiday_calendar_end": values["holiday_calendar_end"],
                "extract_date": values["extract_date"],
            }
        ]
    )
    return {
        "E1_registration_gate.csv": pd.DataFrame(
            [{"status": "completed", "reason": ""}]
        ),
        "E1_registration_counts.csv": pd.DataFrame(rows),
        "E1_registration_statistics.csv": statistics,
    }


def _prediction_tables(
    judgments: pd.DataFrame,
    matches: pd.DataFrame,
    observed: pd.Timestamp,
    outcome_gate: dict[str, object],
    settings: Settings,
) -> tuple[dict[str, pd.DataFrame], str]:
    if outcome_gate["design"] != "longitudinal":
        gate = pd.DataFrame(
            [
                {
                    "check": "longitudinal_outcome",
                    "status": "fail",
                    "detail": "a valid 12-month outcome is not available",
                    "rows": pd.NA,
                }
            ]
        )
        return {"E4_prediction_gate.csv": gate}, "not run"
    try:
        cohort = build_12_month_landmark_cohort(judgments, matches, observed)
    except ValueError as exc:
        gate = pd.DataFrame(
            [
                {
                    "check": "cohort_construction",
                    "status": "fail",
                    "detail": str(exc),
                    "rows": pd.NA,
                }
            ]
        )
        return {"E4_prediction_gate.csv": gate}, "not run"
    results = run_12_month_prediction(
        cohort,
        feature_columns=tuple(sorted(SAFE_FEATURES)),
        min_reporting_count=settings.min_cell_n,
        seed=settings.diagnostic_seed,
    )
    gate = results["gate"].copy()
    gate = gate.rename(columns={"gate_id": "check"})
    design_rows = pd.DataFrame(
        [
            {
                "check": "primary_estimand",
                "status": "pass",
                "detail": PRIMARY_ESTIMAND,
                "rows": pd.NA,
            },
            {
                "check": "cancellation_treatment",
                "status": "pass",
                "detail": CANCELLATION_TREATMENT,
                "rows": pd.NA,
            },
            {
                "check": "judgment_age_baseline",
                "status": "pass",
                "detail": JUDGMENT_AGE_BASELINE,
                "rows": pd.NA,
            },
            {
                "check": "calibration_curve_uncertainty",
                "status": "pass",
                "detail": "descriptive_curve; clustered_intervals_reported_for_scalar_metrics",
                "rows": pd.NA,
            },
        ]
    )
    gate = pd.concat([design_rows, gate], ignore_index=True)
    gate["rows"] = pd.to_numeric(gate["rows"], errors="coerce").astype("Int64")
    gate["population"] = "exact_linked_current_live_company_subpopulation"
    gate["selection_note"] = "conditional_on_current_live_company_file"
    names = {
        "gate": "E4_prediction_gate.csv",
        "split_summary": "E4_split_summary.csv",
        "performance": "E4_model_performance.csv",
        "ranking": "E4_ranking.csv",
        "improvement": "E4_paired_improvement.csv",
        "calibration_curve": "E4_calibration_curve.csv",
    }
    tables = {
        names[key]: gate if key == "gate" else table
        for key, table in results.items()
        if key == "gate" or not table.empty
    }
    flow = cohort.attrs["cohort_flow"].copy()
    split = results["split_summary"]
    if not split.empty:
        removed = split.loc[split["split"].eq("removed_spanning_boundaries")]
        if not removed.empty:
            flow = pd.concat(
                [
                    flow,
                    pd.DataFrame(
                        [
                            {
                                "stage": "excluded_company_spans_split_boundary",
                                "rows": removed.iloc[0]["rows"],
                                "companies": removed.iloc[0]["companies"],
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        analysed = split.loc[split["split"].isin(("train", "validation", "calibration", "final_test"))]
        flow = pd.concat(
            [
                flow,
                pd.DataFrame(
                    [
                        {
                            "stage": "final_analysed_cohort",
                            "rows": pd.to_numeric(analysed["rows"]).sum(),
                            "companies": pd.to_numeric(analysed["companies"]).sum(),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    tables["E4_cohort_flow.csv"] = flow
    completed = not gate.empty and gate["status"].eq("pass").all()
    return tables, "completed" if completed else "not run"


def _summary_context(
    *,
    audit: DataAudit,
    ch_observed: pd.Timestamp,
    judgments: pd.DataFrame,
    matches: pd.DataFrame,
    settings: Settings,
    outcome_status: str,
    prediction_status: str,
) -> dict[str, Any]:
    inserted = pd.to_datetime(judgments["Date Inserted"], errors="raise")
    corporate_ew = judgments["DefendantType"].eq("Corporate") & judgments[
        "Jurisdiction"
    ].eq("England and Wales")
    target_matches = matches.loc[corporate_ew]
    exact = int(target_matches["tier"].eq("exact_unique").sum())
    denominator = int(len(target_matches))
    return {
        "observation_date": audit.observation_date,
        "companies_house_date": ch_observed.date().isoformat(),
        "data_construct": audit.data_construct,
        "optional_fields": {
            "Satisfaction Date": {
                "present": "Satisfaction Date" not in audit.absent_optional_columns,
                "rows": audit.satisfaction_date_present_rows,
            },
            "Cancellation Date": {
                "present": "Cancellation Date" not in audit.absent_optional_columns,
                "rows": audit.cancellation_date_present_rows,
            },
            "Cancellation Reason": {
                "present": "Cancellation Reason" not in audit.absent_optional_columns,
                "rows": audit.cancellation_reason_present_rows,
            },
            "Status Effective Date": {
                "present": "Status Effective Date" not in audit.absent_optional_columns,
                "rows": audit.status_effective_date_present_rows,
            },
            "Snapshot Date": {
                "present": "Snapshot Date" not in audit.absent_optional_columns,
                "rows": audit.snapshot_date_present_rows,
            },
        },
        "min_cell_n": settings.min_cell_n,
        "date_inserted": {
            "distinct_values": audit.date_inserted_distinct,
            "minimum": inserted.min().date().isoformat(),
            "maximum": inserted.max().date().isoformat(),
        },
        "counts": {
            "rows_read": audit.rows,
            "matching_decisions": denominator,
            "missing_company_name": audit.missing_company_name_rows,
            "missing_postcode": audit.missing_postcode_rows,
            "date_inserted_before_judgment": audit.date_inserted_before_judgment_rows,
        },
        "match": {
            "denominator": denominator,
            "exact_unique": exact,
            "unmatched": denominator - exact,
            "coverage": exact / denominator if denominator else 0.0,
        },
        "outcome": {"status": outcome_status},
        "prediction": {"status": prediction_status},
    }


def _run_manifest(
    *,
    paths: RunPaths,
    observed: pd.Timestamp,
    ch_observed: pd.Timestamp,
    audit: DataAudit,
    ch_index: CHIndex,
    settings: Settings,
    settings_source: Path,
    judgments_file: Path,
    companies_file: Path,
    working_files: list[str],
    allowlist: dict[str, str | Sequence[str] | None],
    outcome_status: str,
    prediction_status: str,
) -> dict[str, Any]:
    return {
        "schema_version": 7,
        "status": "CONFIDENTIAL - SEND ONLY TO EDWIN",
        "run_id": paths.root.name,
        "schema_construct": audit.data_construct,
        "observation_date": observed.date().isoformat(),
        "companies_house_snapshot_date": ch_observed.date().isoformat(),
        "companies_house_snapshot_lag_days": int((observed - ch_observed).days),
        "fingerprints": {
            "rt_raw_file": audit.raw_source_sha256,
            "rt_raw_header_schema": audit.raw_header_schema_sha256,
            "rt_analysis_content": audit.analysis_fingerprint,
            "rt_provenance": audit.provenance_fingerprint,
            "companies_house_raw_file": _file_sha256(companies_file),
            "companies_house_analysis_content": ch_index.stats.get("analysis_fingerprint"),
            "code_and_settings": source_fingerprint(Path(__file__).parent, settings_source),
            "settings_file": _file_sha256(settings_source),
        },
        "settings": settings.as_dict(),
        "matching_rule": "unique_date_valid_exact_normalized_name_v1",
        "analysis_status": {
            "outcomes": outcome_status,
            "prediction": prediction_status,
        },
        "academic_design": {
            "landmark_months_after_judgment": LANDMARK_MONTHS,
            "fixed_horizon_months": list(FIXED_HORIZONS),
            "prediction_split_shares": list(SPLIT_SHARES),
            "prediction_minimum_events_and_non_events": MIN_NONTRAIN_CLASS,
            "prediction_bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "prediction_capacities": list(CAPACITIES),
            "prediction_features": sorted(SAFE_FEATURES),
            "prediction_models": list(MODEL_NAMES),
        },
        "linkage_validation_seed": settings.diagnostic_seed,
        "ch_index_stats": dict(ch_index.stats),
        "package_versions": _package_versions(),
        "artifact_manifest": {
            "reports": sorted(allowlist),
            "working_files": working_files,
        },
        "input_formats": {
            "judgments": judgments_file.suffix.casefold(),
            "companies_house": companies_file.suffix.casefold(),
        },
        "disclosure": {
            "status": "pending",
            "minimum_cell": settings.min_cell_n,
            "explicit_allowlist": True,
            "identifier_scan_required": True,
        },
    }


def _analysis_allowlist(
    root: Path,
) -> dict[str, str | Sequence[str] | None]:
    allowed: dict[str, str | Sequence[str] | None] = {}
    for path in sorted(item for item in root.iterdir() if item.is_file()):
        if path.name == "SUMMARY.txt":
            allowed[path.name] = None
            continue
        if not re.fullmatch(r"E[1-5]_[A-Za-z0-9_]+\.(?:csv|json|txt|log)", path.name):
            raise RunFailure(f"unexpected aggregate output file: {path.name}")
        if path.suffix.casefold() != ".csv" or path.name.startswith("E5_"):
            allowed[path.name] = None
            continue
        header = pd.read_csv(path, nrows=0)
        counts = tuple(
            str(column)
            for column in header.columns
            if _is_count_column(str(column))
        )
        allowed[path.name] = counts or None
    if "SUMMARY.txt" not in allowed:
        raise RunFailure("SUMMARY.txt was not written")
    return allowed


def _is_count_column(column: str) -> bool:
    key = re.sub(r"[^a-z0-9]+", "_", column.casefold()).strip("_")
    return (
        key == "rows"
        or key.endswith("_rows")
        or key.endswith("_events")
        or key
        in {
            "at_risk",
            "cancellations",
            "censored",
            "companies",
            "events",
            "non_events",
            "reviewed",
            "reviewed_count",
            "events_captured",
            "cancellations_captured",
        }
    )


def _bounded_known_identifiers(sample: pd.DataFrame) -> tuple[str, ...]:
    values: list[str] = []
    for column in (
        "source_company_name",
        "source_trading_name",
        "matched_company_name",
    ):
        if column in sample:
            values.extend(sample[column].astype("string").fillna("").tolist())
    return tuple(
        dict.fromkeys(
            value.strip()
            for value in values
            if len(value.strip()) >= 8 and any(character.isalpha() for character in value)
        )
    )


def _declared_date(value: str | pd.Timestamp, label: str) -> pd.Timestamp:
    if isinstance(value, str) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        raise RunFailure(f"{label} must use YYYY-MM-DD")
    try:
        timestamp = pd.Timestamp(value.strip() if isinstance(value, str) else value)
    except (TypeError, ValueError) as exc:
        raise RunFailure(f"{label} is invalid") from exc
    if pd.isna(timestamp):
        raise RunFailure(f"{label} is invalid")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.normalize()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_artifact_manifest(results: Path) -> tuple[Path, ...]:
    manifest_name = "E5_artifact_manifest.csv"
    manifest_path = results / manifest_name
    entries = tuple(results.iterdir())
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or any(not path.is_file() or path.is_symlink() for path in entries)
    ):
        raise RunFailure("the results folder contains an unlisted file or folder")
    table = pd.read_csv(manifest_path)
    required = {"artifact_name", "bytes", "sha256", "rows"}
    if not required.issubset(table.columns):
        raise RunFailure("the artifact manifest is incomplete")
    if table["artifact_name"].duplicated().any():
        raise RunFailure("the artifact manifest contains duplicate files")
    names = tuple(table["artifact_name"].astype(str))
    if any(
        not name
        or name == manifest_name
        or Path(name).name != name
        or "/" in name
        or "\\" in name
        for name in names
    ):
        raise RunFailure("the artifact manifest contains an invalid file name")
    expected = set(names)
    actual = {
        path.name
        for path in entries
        if path.is_file() and path.name != manifest_name
    }
    if actual != expected:
        raise RunFailure("the artifact manifest does not list the final files")
    for row in table.itertuples(index=False):
        path = results / str(row.artifact_name)
        if path.stat().st_size != int(row.bytes) or _file_sha256(path) != str(row.sha256):
            raise RunFailure(f"artifact check failed: {path.name}")
        if path.suffix.casefold() == ".csv" and not pd.isna(row.rows):
            if len(pd.read_csv(path)) != int(row.rows):
                raise RunFailure(f"artifact row check failed: {path.name}")
    return (manifest_path, *(results / name for name in sorted(expected)))


def _validate_ch_snapshot(
    path: Path, observed: pd.Timestamp, ch_observed: pd.Timestamp
) -> None:
    lag = int((observed - ch_observed).days)
    if lag < 0:
        raise RunFailure("Companies House file date is after the RT extract date")
    if lag > MAX_CH_SNAPSHOT_LAG_DAYS:
        raise RunFailure(
            f"Companies House file is more than {MAX_CH_SNAPSHOT_LAG_DAYS} days "
            "older than the RT extract"
        )
    embedded = sorted(set(re.findall(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)", path.name)))
    if not embedded:
        raise RunFailure(
            "Companies House filename must contain its date as YYYY-MM-DD"
        )
    if len(embedded) > 1:
        raise RunFailure(f"Companies House filename contains conflicting dates: {embedded}")
    if embedded and pd.Timestamp(embedded[0]).normalize() != ch_observed:
        raise RunFailure("Companies House file date does not match its filename")


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in (
        "numpy",
        "pandas",
        "openpyxl",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "joblib",
        "threadpoolctl",
        "python-dateutil",
        "tzdata",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def package_results(paths: RunPaths) -> Path:
    """Put the checked aggregate files in one ZIP."""

    files = _verify_artifact_manifest(paths.results)
    token = paths.root.name.removeprefix("run_")
    archive = paths.root.parent / f"SEND_TO_EDWIN_{token}.zip"
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    existing = [path for path in (archive, checksum) if path.exists()]
    if existing:
        raise RunFailure(f"output package already exists: {existing[0]}")
    if not files:
        raise RunFailure("there are no checked result files to package")
    archive_names = {
        path: (Path(paths.root.name) / "results" / path.name).as_posix()
        for path in files
    }
    expected_content = {
        archive_names[path]: (path.stat().st_size, _file_sha256(path)) for path in files
    }
    published: list[Path] = []
    try:
        with TemporaryDirectory(prefix=".package-", dir=archive.parent) as temporary:
            temporary_root = Path(temporary)
            staged_archive = temporary_root / archive.name
            staged_checksum = temporary_root / checksum.name
            with zipfile.ZipFile(
                staged_archive,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                allowZip64=True,
            ) as package:
                for path in files:
                    package.write(path, archive_names[path])
            with zipfile.ZipFile(staged_archive) as package:
                members = [entry for entry in package.infolist() if not entry.is_dir()]
                actual = {entry.filename for entry in members}
                if (
                    len(members) != len(expected_content)
                    or actual != set(expected_content)
                    or package.testzip() is not None
                ):
                    raise RunFailure("the result ZIP did not pass its final check")
                for entry in members:
                    digest = hashlib.sha256()
                    size = 0
                    with package.open(entry) as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            size += len(chunk)
                            digest.update(chunk)
                    if (size, digest.hexdigest()) != expected_content[entry.filename]:
                        raise RunFailure("the result ZIP did not pass its final check")

            digest = _file_sha256(staged_archive)
            staged_checksum.write_text(
                f"{digest}  {archive.name}\n", encoding="ascii"
            )
            checksum_parts = staged_checksum.read_text(encoding="ascii").split()
            if (
                checksum_parts != [digest, archive.name]
                or digest != _file_sha256(staged_archive)
            ):
                raise RunFailure("the result checksum did not pass its final check")

            staged_checksum.replace(checksum)
            published.append(checksum)
            staged_archive.replace(archive)
            published.append(archive)
    except BaseException as exc:
        cleanup_errors: list[str] = []
        for path in reversed(published):
            try:
                path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                cleanup_errors.append(f"{path.name}: {cleanup_error}")
        if cleanup_errors:
            raise RunFailure(
                "packaging failed and incomplete output could not be removed: "
                + "; ".join(cleanup_errors)
            ) from exc
        raise
    return archive.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Registry Trust data check")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--judgments", required=True)
    analyze_parser.add_argument("--companies-house", required=True)
    analyze_parser.add_argument("--observation-date", required=True)
    analyze_parser.add_argument("--companies-house-date", required=True)
    analyze_parser.add_argument("--settings", default="settings.toml")
    analyze_parser.add_argument("--output-base", default="outputs")
    return parser


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _elapsed_updates(
    stop: Event,
    started: float,
    interval_seconds: float = ELAPSED_UPDATE_SECONDS,
) -> None:
    while not stop.wait(interval_seconds):
        print(
            f"Still running. Elapsed time: {_format_elapsed(time.monotonic() - started)}",
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    started = time.monotonic()
    stop = Event()
    updates = Thread(
        target=_elapsed_updates,
        args=(stop, started),
        name="elapsed-time",
        daemon=True,
    )
    try:
        updates.start()
    except (OSError, RuntimeError):
        updates = None
    archive: Path | None = None
    error: Exception | None = None
    try:
        paths = analyze(
            judgments_path=args.judgments,
            companies_house_path=args.companies_house,
            observation_date=args.observation_date,
            companies_house_date=args.companies_house_date,
            settings_path=args.settings,
            output_base=args.output_base,
        )
        archive = package_results(paths)
    except Exception as exc:
        error = exc
    finally:
        stop.set()
        if updates is not None:
            updates.join(1)
    elapsed = f"Elapsed time: {_format_elapsed(time.monotonic() - started)}"
    if error is not None:
        print(f"STOP: {type(error).__name__}: {error}", file=sys.stderr)
        print(elapsed, file=sys.stderr)
        return 2
    print("RUN COMPLETE")
    print(elapsed)
    print(f"SEND THIS FILE TO EDWIN: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
