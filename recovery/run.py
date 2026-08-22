"""Run the diagnostic, outcome-blind development, and one-time locked release."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping, Sequence
import argparse
import hashlib
import json
import re
import shutil
import sys

import numpy as np
import pandas as pd

from .config import Settings, load_settings
from .data import DataAudit, read_rt_extract
from .disclosure import stage_egress
from .locking import (
    MAX_CH_SNAPSHOT_LAG_DAYS,
    file_sha256,
    finalize_release_receipt,
    reserve_release,
    verify_manifest_sources,
    verify_release_approval,
)
from .matching import (
    ACCEPTED_LINKAGE_VALIDATION_FILENAME,
    UNMATCHED_LINKAGE_VALIDATION_FILENAME,
    CHIndex,
    accepted_validation_sample,
    build_relevant_ch_index,
    match_diagnostics,
    match_judgments,
    summarize_linkage_validation,
    unmatched_validation_sample,
    validate_linkage_adjudications,
)
from .models import (
    ModelDevelopment,
    ModelEvaluation,
    develop_models,
    evaluate_locked_models,
    materialize_locked_test_outcomes,
    prepare_model_cohort,
    write_model_artifacts,
)
from .reporting import (
    RunPaths,
    RunRecorder,
    build_data_audit_counts,
    build_development_tables,
    build_model_tables,
    build_population_sensitivities,
    create_run_paths,
    source_fingerprint,
    write_development_tables,
    write_e1,
    write_e2,
    write_e3_e4,
    write_e5,
    write_linkage_validation,
    write_summary,
)


PAIR_FILENAME = ACCEPTED_LINKAGE_VALIDATION_FILENAME
UNMATCHED_FILENAME = UNMATCHED_LINKAGE_VALIDATION_FILENAME
DEVELOPMENT_SPECIFICATION_FILENAME = "development_specification.json"
RT_CONSTRUCT = "current_register_stock_single_snapshot"
LOCKED_STAGE_BLOCK_MESSAGE = (
    "locked evaluation requires completed two-arm linkage adjudication, an exact "
    "frozen development specification, and an unused manifest-bound approval"
)


class RunFailure(RuntimeError):
    """A clear, operator-facing failure that stops an official run."""


def analyze(
    *,
    stage: str,
    judgments_path: str | Path,
    companies_house_path: str | Path,
    observation_date: str | pd.Timestamp | None,
    companies_house_date: str | pd.Timestamp | None = None,
    settings_path: str | Path,
    output_base: str | Path,
    accepted_adjudications_path: str | Path | None = None,
    unmatched_adjudications_path: str | Path | None = None,
    development_specification_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    approval_path: str | Path | None = None,
    key_path: str | Path | None = None,
    release_registry: str | Path | None = None,
    recall_denominator_supported: bool = False,
    run_id: str | None = None,
    _match_validator: Callable[[pd.DataFrame], None] | None = None,
) -> RunPaths:
    """Execute one stage; only ``locked`` can materialize held-out outcomes."""

    if stage not in {"diagnostic", "development", "locked"}:
        raise RunFailure("stage must be diagnostic, development or locked")
    if stage == "locked":
        early_gate = (
            accepted_adjudications_path,
            unmatched_adjudications_path,
            development_specification_path,
            manifest_path,
            approval_path,
            key_path,
        )
        if any(value is None for value in early_gate):
            # Fail before dates, settings, data, or output paths are touched.
            raise RunFailure(LOCKED_STAGE_BLOCK_MESSAGE)
    if observation_date is None:
        raise RunFailure("RT observation date is required; it must not default to today")
    if companies_house_date is None:
        raise RunFailure("Companies House snapshot date is required")
    observed = _declared_date(observation_date, "RT observation date")
    ch_observed = _declared_date(companies_house_date, "Companies House snapshot date")
    companies_file = Path(companies_house_path).resolve()
    _validate_ch_snapshot(companies_file, observed, ch_observed)
    settings_source = Path(settings_path).resolve()
    settings = load_settings(settings_source)

    completed_paths: tuple[Path, Path] | None = None
    if stage in {"development", "locked"}:
        if accepted_adjudications_path is None or unmatched_adjudications_path is None:
            raise RunFailure(
                "development and locked stages require completed accepted and unmatched "
                "linkage-adjudication files"
            )
        completed_paths = (
            Path(accepted_adjudications_path).resolve(),
            Path(unmatched_adjudications_path).resolve(),
        )
        for path in completed_paths:
            if not path.is_file():
                raise RunFailure(f"completed linkage-adjudication file is missing: {path}")

    receipt: Path | None = None
    authorization: Mapping[str, Any] | None = None
    frozen_specification: Path | None = None
    if stage == "locked":
        required = {
            "development specification": development_specification_path,
            "design manifest": manifest_path,
            "release approval": approval_path,
            "release key": key_path,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RunFailure(LOCKED_STAGE_BLOCK_MESSAGE + "; missing " + ", ".join(missing))
        frozen_specification = Path(development_specification_path).resolve()
        if not frozen_specification.is_file():
            raise RunFailure(f"frozen development specification is missing: {frozen_specification}")
        repository = Path(__file__).resolve().parents[1]
        design_sources = (repository / "STUDY_DESIGN.md", repository / "requirements.lock")
        for source in design_sources:
            if not source.is_file():
                raise RunFailure(f"release-bound design source is missing: {source}")
        bound_files = {
            "accepted_adjudications": completed_paths[0],
            "unmatched_adjudications": completed_paths[1],
            "development_specification": frozen_specification,
        }
        verify_manifest_sources(
            manifest_path=manifest_path,
            judgments_path=judgments_path,
            companies_house_path=companies_file,
            observation_date=observed,
            companies_house_date=ch_observed,
            settings_path=settings_source,
            package_dir=Path(__file__).resolve().parent,
            extra_sources=design_sources,
            bound_files=bound_files,
            recall_denominator_supported=recall_denominator_supported,
        )
        authorization = verify_release_approval(
            manifest_path=manifest_path,
            approval_path=approval_path,
            key_path=key_path,
        )
        registry = (
            Path(release_registry).resolve()
            if release_registry is not None
            else repository / ".release_receipts"
        )
        # Consume before parsing RT statuses. A failed attempt also stays consumed.
        receipt = reserve_release(registry_dir=registry, authorization=authorization)

    paths: RunPaths | None = None
    try:
        paths = create_run_paths(output_base, stage, run_id)
        result = _analyze_created_run(
            stage=stage,
            judgments_path=judgments_path,
            companies_house_path=companies_file,
            observed=observed,
            ch_observed=ch_observed,
            settings_source=settings_source,
            settings=settings,
            paths=paths,
            completed_paths=completed_paths,
            frozen_specification=frozen_specification,
            authorization=authorization,
            recall_denominator_supported=recall_denominator_supported,
            _match_validator=_match_validator,
        )
        if receipt is not None:
            finalize_release_receipt(receipt, status="completed", run_id=result.root.name)
        return result
    except BaseException as exc:
        if receipt is not None:
            try:
                finalize_release_receipt(
                    receipt,
                    status="failed",
                    run_id=paths.root.name if paths is not None else None,
                )
            except Exception:
                pass
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
    stage: str,
    judgments_path: str | Path,
    companies_house_path: Path,
    observed: pd.Timestamp,
    ch_observed: pd.Timestamp,
    settings_source: Path,
    settings: Settings,
    paths: RunPaths,
    completed_paths: tuple[Path, Path] | None,
    frozen_specification: Path | None,
    authorization: Mapping[str, Any] | None,
    recall_denominator_supported: bool,
    _match_validator: Callable[[pd.DataFrame], None] | None,
) -> RunPaths:
    aggregate = paths.root / ".aggregate_staging"
    aggregate.mkdir()
    recorder = RunRecorder()
    judgments_file = Path(judgments_path).resolve()

    with recorder.stage("E1_read_validate_schema"):
        judgments, audit = read_rt_extract(judgments_file, observed)
        if stage != "diagnostic":
            _require_cross_sectional_fallback(audit)

    with recorder.stage("CH_stream_index") as record:
        ch_index = build_relevant_ch_index(judgments, companies_house_path)
        record["ch_rows_read"] = ch_index.stats.get("ch_rows_read")
        record["ch_rows_retained"] = ch_index.stats.get("ch_rows_retained")

    with recorder.stage("E2_exact_linkage") as record:
        matches = match_judgments(judgments, ch_index)
        if _match_validator is not None:
            _match_validator(matches)
        diagnostics = match_diagnostics(judgments, matches)
        record["judgments_matched"] = int(matches["tier"].eq("exact_unique").sum())

    linkage_judgments, linkage_matches = _linkage_target(judgments, matches)
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
            paths.working / PAIR_FILENAME, index=False, encoding="utf-8-sig"
        )
        unmatched_sample.to_csv(
            paths.working / UNMATCHED_FILENAME, index=False, encoding="utf-8-sig"
        )
        record["accepted_sample_rows"] = len(accepted_sample)
        record["unmatched_sample_rows"] = len(unmatched_sample)
        record["sample_seed"] = settings.diagnostic_seed

    linkage_summaries: dict[str, pd.DataFrame] | None = None
    if completed_paths is not None:
        with recorder.stage("E2_validate_double_adjudication"):
            completed = _load_completed_linkage_validation(
                accepted_sample,
                unmatched_sample,
                accepted_path=completed_paths[0],
                unmatched_path=completed_paths[1],
            )
            linkage_summaries = summarize_linkage_validation(
                completed,
                recall_denominator_supported=recall_denominator_supported,
            )

    development: ModelDevelopment | None = None
    development_envelope: dict[str, Any] | None = None
    evaluation: ModelEvaluation | None = None
    model_artifacts: dict[str, str] = {}
    if stage in {"development", "locked"}:
        with recorder.stage("E3_prepare_cross_sectional_cohort") as record:
            cohort = prepare_model_cohort(judgments, matches, observed, settings)
            record["model_rows"] = len(cohort.frame)
            record["model_companies"] = cohort.frame["matched_company_number"].nunique()
        with recorder.stage("E3_develop_without_test_outcomes"):
            development = develop_models(cohort, settings)
        development_envelope = _development_specification(
            development=development,
            audit=audit,
            ch_index=ch_index,
            observation_date=observed,
            companies_house_date=ch_observed,
            accepted_adjudications=completed_paths[0],
            unmatched_adjudications=completed_paths[1],
        )
        _write_json(paths.working / DEVELOPMENT_SPECIFICATION_FILENAME, development_envelope)

    if stage == "locked":
        if frozen_specification is None or development is None or development_envelope is None:
            raise RunFailure("internal locked-release invariant failed")
        _verify_development_specification(frozen_specification, development_envelope)
        with recorder.stage("E3_materialize_once_locked_test"):
            locked_outcomes = materialize_locked_test_outcomes(judgments, development.cohort)
        with recorder.stage("E3_evaluate_frozen_models_once"):
            evaluation = evaluate_locked_models(development, locked_outcomes, settings)
            model_artifacts = write_model_artifacts(evaluation, paths.models)

    with recorder.stage("aggregate_reports"):
        audit_counts = build_data_audit_counts(judgments, audit)
        if development is None:
            funnel = _matching_funnel(
                judgments, matches, len(accepted_sample), len(unmatched_sample)
            )
        else:
            funnel = pd.DataFrame(
                [
                    {"stage": name, "rows": int(rows)}
                    for name, rows in development.cohort.funnel.items()
                ]
            )
        write_e1(aggregate, audit_counts, funnel)
        write_e2(aggregate, diagnostics)
        if linkage_summaries is not None:
            write_linkage_validation(aggregate, linkage_summaries)
        sensitivities = None
        if development is not None:
            sensitivities = build_population_sensitivities(
                judgments, matches, observed, settings
            )
        limitations = _limitations_text(stage)
        if stage == "development":
            write_development_tables(
                aggregate,
                build_development_tables(development, sensitivities),
                limitations,
            )
        elif stage == "locked":
            write_e3_e4(
                aggregate,
                build_model_tables(evaluation, sensitivities),
                limitations,
            )
        write_summary(
            aggregate / "SUMMARY.txt",
            _summary_context(
                stage=stage,
                audit=audit,
                ch_observed=ch_observed,
                judgments=judgments,
                matches=matches,
                development=development,
                evaluation=evaluation,
                linkage_summaries=linkage_summaries,
                settings=settings,
            ),
        )

    allowlist = _analysis_allowlist(stage)
    working_files = sorted(
        path.relative_to(paths.working).as_posix()
        for path in paths.working.rglob("*")
        if path.is_file()
    )
    manifest = _run_manifest(
        paths=paths,
        stage=stage,
        observed=observed,
        ch_observed=ch_observed,
        audit=audit,
        ch_index=ch_index,
        settings=settings,
        settings_source=settings_source,
        judgments_file=judgments_file,
        companies_file=companies_house_path,
        working_files=working_files,
        model_artifacts=model_artifacts,
        evaluation=evaluation,
        authorization=authorization,
        allowlist=allowlist,
    )
    known_identifiers = _bounded_known_identifiers(
        pd.concat([accepted_sample, unmatched_sample], ignore_index=True, sort=False)
    )
    report_allowlist = {
        name: policy
        for name, policy in allowlist.items()
        if name not in {"E5_run_log.csv", "E5_run_manifest.json"}
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
    disclosure = stage_egress(
        aggregate,
        paths.results,
        allowlist=allowlist,
        min_cell_n=settings.min_cell_n,
        known_identifiers=known_identifiers,
    )
    if (
        tuple(disclosure.suppressed_rows) != tuple(preview.suppressed_rows)
        or set(disclosure.staged_files) != set(allowlist)
    ):
        raise RunFailure("final disclosure copy differed from its checked preview")
    shutil.rmtree(aggregate)
    if stage != "locked":
        paths.models.rmdir()
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
        raise RunFailure("linkage target does not have exactly one decision per RT row")
    return target, decisions


def _load_completed_linkage_validation(
    accepted_expected: pd.DataFrame,
    unmatched_expected: pd.DataFrame,
    *,
    accepted_path: Path,
    unmatched_path: Path,
) -> pd.DataFrame:
    accepted = pd.read_csv(accepted_path, dtype="string", keep_default_na=False)
    unmatched = pd.read_csv(unmatched_path, dtype="string", keep_default_na=False)
    accepted = validate_linkage_adjudications(accepted)
    unmatched = validate_linkage_adjudications(unmatched)
    if set(accepted["validation_arm"]) != {"accepted"}:
        raise RunFailure("accepted adjudication file contains the wrong validation arm")
    if set(unmatched["validation_arm"]) != {"unmatched"}:
        raise RunFailure("unmatched adjudication file contains the wrong validation arm")
    _verify_sample_membership(accepted_expected, accepted, "accepted")
    _verify_sample_membership(unmatched_expected, unmatched, "unmatched")
    return pd.concat([accepted, unmatched], ignore_index=True, sort=False)


def _verify_sample_membership(
    expected: pd.DataFrame, completed: pd.DataFrame, arm: str
) -> None:
    columns = (
        "ID",
        "validation_arm",
        "sampling_stratum",
        "stratum_population_n",
        "stratum_sample_n",
        "inclusion_probability",
        "sampling_weight",
        "matched_company_number",
    )
    missing = set(columns) - set(completed.columns)
    if missing:
        raise RunFailure(f"{arm} adjudications are missing frozen fields: {sorted(missing)}")
    left = expected.loc[:, [column for column in columns if column in expected]].copy()
    if "matched_company_number" not in left:
        left["matched_company_number"] = ""
    right = completed.loc[:, columns].copy()
    for frame in (left, right):
        frame["ID"] = frame["ID"].astype("string").fillna("").str.strip()
        frame["validation_arm"] = (
            frame["validation_arm"].astype("string").fillna("").str.strip().str.lower()
        )
        frame["sampling_stratum"] = (
            frame["sampling_stratum"].astype("string").fillna("").str.strip()
        )
        frame["matched_company_number"] = (
            frame["matched_company_number"]
            .astype("string")
            .fillna("")
            .str.replace(r"\s+", "", regex=True)
            .str.upper()
        )
    if set(left["ID"]) != set(right["ID"]) or len(left) != len(right):
        raise RunFailure(f"{arm} adjudication membership differs from the frozen sample")
    merged = left.merge(right, on="ID", suffixes=("_expected", "_completed"), validate="one_to_one")
    for column in ("validation_arm", "sampling_stratum", "matched_company_number"):
        if not merged[f"{column}_expected"].eq(merged[f"{column}_completed"]).all():
            raise RunFailure(f"{arm} adjudication changed frozen field {column}")
    for column in (
        "stratum_population_n",
        "stratum_sample_n",
        "inclusion_probability",
        "sampling_weight",
    ):
        expected_values = pd.to_numeric(merged[f"{column}_expected"], errors="coerce")
        completed_values = pd.to_numeric(merged[f"{column}_completed"], errors="coerce")
        if expected_values.isna().any() or completed_values.isna().any() or not np.allclose(
            expected_values, completed_values, rtol=1e-10, atol=1e-12
        ):
            raise RunFailure(f"{arm} adjudication changed frozen field {column}")


def _require_cross_sectional_fallback(audit: DataAudit) -> None:
    event_columns = set(audit.event_date_columns_present).intersection(
        {"Satisfaction Date", "Cancellation Date", "Status Effective Date"}
    )
    if event_columns:
        raise RunFailure(
            "event-date field(s) are available: "
            f"{sorted(event_columns)}; stop and implement the preferred event-time/fixed-horizon design"
        )
    if audit.historical_snapshots_available:
        raise RunFailure(
            "historical snapshots are available; stop and implement a longitudinal design"
        )
    if not audit.data_construct.startswith("cross_sectional_status"):
        raise RunFailure(f"unsupported RT data construct: {audit.data_construct}")


def _development_specification(
    *,
    development: ModelDevelopment,
    audit: DataAudit,
    ch_index: CHIndex,
    observation_date: pd.Timestamp,
    companies_house_date: pd.Timestamp,
    accepted_adjudications: Path,
    unmatched_adjudications: Path,
) -> dict[str, Any]:
    specification = {
        "schema_version": 1,
        "design": "cross_sectional_recorded_satisfaction_at_extract",
        "observation_date": observation_date.date().isoformat(),
        "companies_house_snapshot_date": companies_house_date.date().isoformat(),
        "rt_provenance_sha256": audit.provenance_fingerprint,
        "ch_analysis_sha256": ch_index.stats.get("analysis_fingerprint"),
        "accepted_adjudications_sha256": file_sha256(accepted_adjudications),
        "unmatched_adjudications_sha256": file_sha256(unmatched_adjudications),
        "development": development.to_public_dict(),
        "calibration_plan": {
            key: {
                "method": calibration.method,
                "n_positive": calibration.n_positive,
                "n_negative": calibration.n_negative,
            }
            for key, calibration in sorted(development.calibrations.items())
        },
        "test_outcomes_accessed": False,
    }
    safe = _json_safe(specification)
    digest = hashlib.sha256(_canonical_json(safe)).hexdigest()
    return {"specification": safe, "specification_sha256": digest}


def _verify_development_specification(
    frozen_path: Path, current: dict[str, Any]
) -> None:
    try:
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunFailure(f"cannot read frozen development specification: {exc}") from exc
    if not isinstance(frozen, dict):
        raise RunFailure("frozen development specification is not a JSON object")
    specification = frozen.get("specification")
    recorded = frozen.get("specification_sha256")
    if not isinstance(specification, dict) or not isinstance(recorded, str):
        raise RunFailure("frozen development specification has an invalid structure")
    calculated = hashlib.sha256(_canonical_json(specification)).hexdigest()
    if calculated != recorded:
        raise RunFailure("frozen development specification hash is invalid")
    if _canonical_json(frozen) != _canonical_json(current):
        raise RunFailure(
            "rerun development differs from the manifest-bound frozen specification"
        )


def _matching_funnel(
    judgments: pd.DataFrame,
    matches: pd.DataFrame,
    accepted_rows: int,
    unmatched_rows: int,
) -> pd.DataFrame:
    tiers = matches["tier"].value_counts()
    return pd.DataFrame(
        [
            {"stage": "judgments_read", "rows": int(len(judgments))},
            {"stage": "matching_decisions", "rows": int(len(matches))},
            {"stage": "unique_exact_name", "rows": int(tiers.get("exact_unique", 0))},
            {"stage": "unmatched", "rows": int(tiers.get("unmatched", 0))},
            {"stage": "accepted_validation_sample", "rows": int(accepted_rows)},
            {"stage": "unmatched_validation_sample", "rows": int(unmatched_rows)},
        ]
    )


def _summary_context(
    *,
    stage: str,
    audit: DataAudit,
    ch_observed: pd.Timestamp,
    judgments: pd.DataFrame,
    matches: pd.DataFrame,
    development: ModelDevelopment | None,
    evaluation: ModelEvaluation | None,
    linkage_summaries: dict[str, pd.DataFrame] | None,
    settings: Settings,
) -> dict[str, Any]:
    corporate_ew = judgments["DefendantType"].eq("Corporate") & judgments[
        "Jurisdiction"
    ].eq("England and Wales")
    target_matches = matches.loc[corporate_ew]
    exact = int(target_matches["tier"].eq("exact_unique").sum())
    denominator = int(len(target_matches))
    if stage == "diagnostic":
        return {
            "scope": "matching_only",
            "stage": stage,
            "status": "DIAGNOSTIC COMPLETE; ADJUDICATION REQUIRED",
            "observation_date": audit.observation_date,
            "companies_house_date": ch_observed.date().isoformat(),
            "data_construct": audit.data_construct,
            "min_cell_n": settings.min_cell_n,
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
            "accepted_file": PAIR_FILENAME,
            "unmatched_file": UNMATCHED_FILENAME,
        }
    if development is None:
        raise RunFailure("model summary requires a development object")
    counts = {
        "rows_read": audit.rows,
        "corporate_ew_labelled": development.cohort.funnel.get(
            "satisfied_or_unsatisfied_corporate_ew"
        ),
        "primary_age_eligible": development.cohort.funnel.get(
            "primary_age_window_corporate_ew_labelled"
        ),
        "model_rows": len(development.cohort.frame),
        "model_companies": development.cohort.frame["matched_company_number"].nunique(),
    }
    linkage = _linkage_context(linkage_summaries)
    primary: dict[str, Any] = {
        "champion": development.frozen_evaluation_keys[-1],
        "specification_hash": development.specification_hash,
    }
    status = "DEVELOPMENT FROZEN; TEST NOT ACCESSED"
    if evaluation is not None:
        key = development.frozen_evaluation_keys[-1]
        run = evaluation.runs[key]
        metrics = run.test_metrics_calibrated
        auc_interval = run.bootstrap_intervals.get("roc_auc", {})
        primary.update(
            {
                "roc_auc": metrics.get("roc_auc"),
                "roc_auc_ci_low": auc_interval.get("lower"),
                "roc_auc_ci_high": auc_interval.get("upper"),
                "average_precision": metrics.get("average_precision"),
                "brier": metrics.get("brier"),
                "calibration_intercept": metrics.get("calibration_intercept"),
                "calibration_slope": metrics.get("calibration_slope"),
                "internal_screen": evaluation.primary_acceptance.get("status"),
            }
        )
        status = "LOCKED TEST EVALUATED ONCE"
    return {
        "scope": "model",
        "stage": stage,
        "status": status,
        "observation_date": audit.observation_date,
        "companies_house_date": ch_observed.date().isoformat(),
        "data_construct": audit.data_construct,
        "min_cell_n": settings.min_cell_n,
        "counts": counts,
        "linkage": linkage,
        "splits": development.cohort.split_counts,
        "primary": primary,
    }


def _linkage_context(
    summaries: dict[str, pd.DataFrame] | None,
) -> dict[str, Any]:
    if summaries is None:
        return {}
    estimates = summaries["estimates"].set_index("measure")
    return {
        "precision": estimates.loc["accepted_match_precision", "estimate"],
        "missed": estimates.loc["unmatched_missed_link_prevalence", "estimate"],
        "recall": estimates.loc["linkage_recall", "estimate"],
        "recall_status": estimates.loc["linkage_recall", "status"],
    }


def _run_manifest(
    *,
    paths: RunPaths,
    stage: str,
    observed: pd.Timestamp,
    ch_observed: pd.Timestamp,
    audit: DataAudit,
    ch_index: CHIndex,
    settings: Settings,
    settings_source: Path,
    judgments_file: Path,
    companies_file: Path,
    working_files: list[str],
    model_artifacts: dict[str, str],
    evaluation: ModelEvaluation | None,
    authorization: Mapping[str, Any] | None,
    allowlist: dict[str, str | Sequence[str] | None],
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "status": "RT INTERNAL - NOT AUTHORISED FOR EXTERNAL USE",
        "run_id": paths.root.name,
        "stage": stage,
        "data_construct_attestation": RT_CONSTRUCT,
        "schema_construct": audit.data_construct,
        "observation_date": observed.date().isoformat(),
        "companies_house_snapshot_date": ch_observed.date().isoformat(),
        "companies_house_snapshot_lag_days": int((observed - ch_observed).days),
        "fingerprints": {
            "rt_raw_file": audit.raw_source_sha256,
            "rt_raw_header_schema": audit.raw_header_schema_sha256,
            "rt_analysis_content": audit.analysis_fingerprint,
            "rt_provenance": audit.provenance_fingerprint,
            "companies_house_raw_file": file_sha256(companies_file),
            "companies_house_analysis_content": ch_index.stats.get("analysis_fingerprint"),
            "code_and_settings": source_fingerprint(Path(__file__).parent, settings_source),
            "settings_file": file_sha256(settings_source),
        },
        "settings": settings.as_dict(),
        "matching_rule": "unique_date_valid_exact_normalized_name_v1",
        "linkage_validation_seed": settings.diagnostic_seed,
        "model_only_acceptance": (
            evaluation.primary_acceptance if evaluation is not None else None
        ),
        "release": (
            {
                "approval_id": authorization.get("approval_id"),
                "manifest_sha256": authorization.get("manifest_sha256"),
                "single_use": True,
            }
            if authorization is not None
            else None
        ),
        "model_artifact_sha256": {
            Path(path).name: file_sha256(path)
            for path in sorted(model_artifacts.values())
        },
        "ch_index_stats": dict(ch_index.stats),
        "package_versions": _package_versions(),
        "artifact_manifest": {
            "reports": sorted(allowlist),
            "working_files": working_files,
            "model_weights_public": False,
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


def _analysis_allowlist(stage: str) -> dict[str, str | Sequence[str] | None]:
    common: dict[str, str | Sequence[str] | None] = {
        "SUMMARY.txt": None,
        "E1_data_audit.csv": "rows",
        "E1_data_funnel.csv": "rows",
        "E2_match_coverage.csv": "rows",
        "E2_unmatched_reasons.csv": "rows",
        "E2_match_methods.csv": "rows",
        "E2_match_by_defendant_type.csv": "rows",
        "E2_match_by_judgment_vintage.csv": "rows",
        "E2_incorporation_guards.csv": "rows",
        "E5_run_log.csv": None,
        "E5_run_manifest.json": None,
    }
    if stage == "diagnostic":
        return common
    linkage = {
        "E2_linkage_validation_estimates.csv": ("population_n", "sample_n"),
        "E2_linkage_reviewer_agreement.csv": "reviewed_n",
        "E2_linkage_strata.csv": ("stratum_population_n", "stratum_sample_n"),
        "E3_split_counts.csv": ("rows", "unique_companies"),
        "E4_population_comparison.csv": (
            "rows",
            "unique_companies",
            "positive",
            "negative",
        ),
        "E4_limitations.txt": None,
    }
    if stage == "development":
        return {
            **common,
            **linkage,
            "E3_development_models.csv": (
                "validation_rows",
                "validation_positive",
                "validation_negative",
            ),
        }
    return {
        **common,
        **linkage,
        "E3_model_comparison.csv": ("test_rows", "test_positive", "test_negative"),
        "E3_calibration.csv": ("rows", "positive", "negative"),
        "E3_incremental_vs_age.csv": None,
        "E3_operational_ranking.csv": (
            "selected",
            "selected_positive",
            "selected_negative",
        ),
    }


def _limitations_text(stage: str) -> str:
    return "\n".join(
        [
            "RT INTERNAL - NOT AUTHORISED FOR EXTERNAL USE",
            f"Run stage: {stage}",
            "Outcome is recorded Satisfied versus Unsatisfied at the extract date.",
            "This is not cash recovery, partial recovery, LGD, return, or future satisfaction.",
            "No satisfaction date is available; this is cross-sectional, not fixed-horizon.",
            "The strict post-one-month cohort is conditional on remaining registered.",
            "Judgment age is mandatory and the age-only model is the primary comparator.",
            "The Companies House bulk snapshot contains live companies only.",
            "Only unique exact date-valid normalized-name links enter prediction.",
            "Repeated judgments are retained, grouped, and resampled by company.",
            "Prior-history features describe observable retained linked records only.",
            "Current Companies House snapshot features are development-only, not locked-tested.",
            "AUC 0.70 is an internal screen and is not a publication criterion.",
        ]
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
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise RunFailure(f"{label} is invalid") from exc
    if pd.isna(timestamp):
        raise RunFailure(f"{label} is invalid")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.normalize()


def _validate_ch_snapshot(
    path: Path, observed: pd.Timestamp, ch_observed: pd.Timestamp
) -> None:
    lag = int((observed - ch_observed).days)
    if lag < 0:
        raise RunFailure("Companies House snapshot post-dates the RT observation date")
    if lag > MAX_CH_SNAPSHOT_LAG_DAYS:
        raise RunFailure(
            f"Companies House snapshot lag exceeds {MAX_CH_SNAPSHOT_LAG_DAYS} days"
        )
    embedded = sorted(set(re.findall(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)", path.name)))
    if len(embedded) > 1:
        raise RunFailure(f"Companies House filename contains conflicting dates: {embedded}")
    if embedded and pd.Timestamp(embedded[0]).normalize() != ch_observed:
        raise RunFailure("declared Companies House date does not match its filename")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "openpyxl",
        "narwhals",
        "tzdata",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Registry Trust analysis")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument(
        "--stage", choices=("diagnostic", "development", "locked"), required=True
    )
    analyze_parser.add_argument("--judgments", required=True)
    analyze_parser.add_argument("--companies-house", required=True)
    analyze_parser.add_argument("--observation-date", required=True)
    analyze_parser.add_argument("--companies-house-date", required=True)
    analyze_parser.add_argument("--settings", default="settings.toml")
    analyze_parser.add_argument("--output-base", default="outputs")
    analyze_parser.add_argument("--accepted-adjudications")
    analyze_parser.add_argument("--unmatched-adjudications")
    analyze_parser.add_argument("--development-specification")
    analyze_parser.add_argument("--manifest")
    analyze_parser.add_argument("--approval")
    analyze_parser.add_argument("--key-file")
    analyze_parser.add_argument("--release-registry")
    analyze_parser.add_argument("--recall-denominator-supported", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        paths = analyze(
            stage=args.stage,
            judgments_path=args.judgments,
            companies_house_path=args.companies_house,
            observation_date=args.observation_date,
            companies_house_date=args.companies_house_date,
            settings_path=args.settings,
            output_base=args.output_base,
            accepted_adjudications_path=args.accepted_adjudications,
            unmatched_adjudications_path=args.unmatched_adjudications,
            development_specification_path=args.development_specification,
            manifest_path=args.manifest,
            approval_path=args.approval,
            key_path=args.key_file,
            release_registry=args.release_registry,
            recall_denominator_supported=args.recall_denominator_supported,
        )
        print(f"OPEN THIS SUMMARY: {paths.results / 'SUMMARY.txt'}")
        if args.stage == "diagnostic":
            print(f"ACCEPTED AUDIT: {paths.working / PAIR_FILENAME}")
            print(f"UNMATCHED AUDIT: {paths.working / UNMATCHED_FILENAME}")
            print("No model was trained.")
        elif args.stage == "development":
            print(
                "FROZEN DEVELOPMENT SPECIFICATION: "
                f"{paths.working / DEVELOPMENT_SPECIFICATION_FILENAME}"
            )
            print("Locked test outcomes were not accessed.")
        else:
            print("The manifest-bound locked test was evaluated once.")
        return 0
    except Exception as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
