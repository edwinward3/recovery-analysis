"""Shared code that runs the matching and later satisfaction model."""

from __future__ import annotations

from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Sequence
import argparse
import hashlib
import shutil
import sys

import pandas as pd

from .config import Settings, load_settings
from .data import DataAudit, read_rt_extract
from .disclosure import stage_egress
from .matching import (
    CHIndex,
    build_relevant_ch_index,
    match_diagnostics,
    match_judgments,
    pair_sample,
)
from .models import (
    ModelEvaluation,
    fit_evaluate_models,
    prepare_model_cohort,
    write_model_artifacts,
)
from .reporting import (
    RunPaths,
    RunRecorder,
    build_data_audit_counts,
    build_model_tables,
    build_population_sensitivities,
    create_run_paths,
    source_fingerprint,
    write_e1,
    write_e2,
    write_e3_e4,
    write_e5,
    write_summary,
)


# Output file names

PAIR_FILENAME = "matching_pairs_1000.csv"


class RunFailure(RuntimeError):
    """A clear, operator-facing failure that stops an official run."""


# Run 1 matching and Run 2 modelling

def analyze(
    *,
    stage: str,
    judgments_path: str | Path,
    companies_house_path: str | Path,
    observation_date: str | date | None,
    settings_path: str | Path,
    output_base: str | Path,
    run_id: str | None = None,
    _match_validator: Callable[[pd.DataFrame], None] | None = None,
) -> RunPaths:
    """Run full-data matching, then add modelling only for a locked Run 2."""

    if stage not in {"diagnostic", "locked"}:
        raise RunFailure("stage must be diagnostic or locked")
    if stage == "locked" and (
        observation_date is None or not str(observation_date).strip()
    ):
        raise RunFailure("Run 2 requires the explicit RT extract date")
    settings_source = Path(settings_path).resolve()
    settings = load_settings(settings_source)
    observed = pd.Timestamp(observation_date or date.today()).normalize()
    paths = create_run_paths(output_base, stage, run_id)
    try:
        return _analyze_created_run(
            stage=stage,
            judgments_path=judgments_path,
            companies_house_path=companies_house_path,
            settings_source=settings_source,
            settings=settings,
            observed=observed,
            paths=paths,
            _match_validator=_match_validator,
        )
    except BaseException as exc:
        try:
            if paths.root.exists():
                shutil.rmtree(paths.root)
        except OSError as cleanup_error:
            raise RunFailure(
                f"the run failed and its incomplete files could not be removed: "
                f"{cleanup_error}"
            ) from exc
        raise


def _analyze_created_run(
    *,
    stage: str,
    judgments_path: str | Path,
    companies_house_path: str | Path,
    settings_source: Path,
    settings: Settings,
    observed: pd.Timestamp,
    paths: RunPaths,
    _match_validator: Callable[[pd.DataFrame], None] | None,
) -> RunPaths:
    """Write one already-created run, leaving no folder behind on failure."""

    aggregate = paths.root / ".aggregate_staging"
    aggregate.mkdir()
    recorder = RunRecorder()
    judgments_file = Path(judgments_path)
    companies_file = Path(companies_house_path)

    with recorder.stage("E1_read_and_validate"):
        judgments, audit = read_rt_extract(judgments_file, observed)

    with recorder.stage("CH_stream_and_index") as record:
        ch_index = build_relevant_ch_index(judgments, companies_file)
        record["ch_rows_read"] = ch_index.stats.get("ch_rows_read")
        record["ch_rows_retained"] = ch_index.stats.get("ch_rows_retained")

    with recorder.stage("E2_match") as record:
        matches = match_judgments(judgments, ch_index)
        if _match_validator is not None:
            _match_validator(matches)
        diagnostics = match_diagnostics(judgments, matches)
        record["judgments_matched"] = int(matches["tier"].ne("unmatched").sum())

    with recorder.stage("matching_pair_sample") as record:
        seed = settings.diagnostic_seed if stage == "diagnostic" else settings.locked_seed
        sample = pair_sample(judgments, matches, settings, seed=seed)
        pair_path = paths.working / PAIR_FILENAME
        sample.to_csv(pair_path, index=False, encoding="utf-8-sig")
        record["sample_rows"] = len(sample)
        record["sample_seed"] = seed

    evaluation: ModelEvaluation | None = None
    model_artifacts: dict[str, str] = {}
    if stage == "locked":
        with recorder.stage("E3_prepare_cohort") as record:
            cohort = prepare_model_cohort(judgments, matches, observed, settings)
            record["model_rows"] = len(cohort.frame)

        with recorder.stage("E3_fit_four_models"):
            evaluation = fit_evaluate_models(cohort, settings)
            model_artifacts = write_model_artifacts(evaluation, paths.models)

    report_stage = (
        "E1_E2_matching_reports"
        if stage == "diagnostic"
        else "E1_to_E4_aggregate_reports"
    )
    with recorder.stage(report_stage):
        audit_counts = build_data_audit_counts(judgments, audit)
        if evaluation is None:
            funnel = _matching_funnel(judgments, matches, len(sample))
        else:
            funnel = pd.DataFrame(
                [
                    {"stage": name, "rows": int(rows)}
                    for name, rows in evaluation.cohort.funnel.items()
                ]
            )
        write_e1(aggregate, audit_counts, funnel)
        write_e2(aggregate, diagnostics)
        if evaluation is None:
            summary_context = _diagnostic_summary_context(
                audit, matches, PAIR_FILENAME, settings
            )
        else:
            sensitivities = build_population_sensitivities(
                judgments, matches, observed, settings
            )
            model_tables = build_model_tables(evaluation, sensitivities)
            write_e3_e4(aggregate, model_tables, _limitations_text(stage))
            summary_context = _locked_summary_context(
                stage,
                audit,
                judgments,
                matches,
                evaluation,
                PAIR_FILENAME,
                settings,
            )
        write_summary(aggregate / "SUMMARY.txt", summary_context)

    code_fingerprint = source_fingerprint(Path(__file__).parent, settings_source)
    settings_fingerprint = _small_file_sha256(settings_source)
    model_only = evaluation.primary_acceptance if evaluation is not None else None
    model_artifact_fingerprints = {
        Path(path).name: _small_file_sha256(Path(path))
        for path in sorted(model_artifacts.values())
    }
    allowlist = _analysis_allowlist(stage)
    manifest = _run_manifest(
        paths=paths,
        stage=stage,
        observed=observed,
        audit=audit,
        ch_index=ch_index,
        settings=settings,
        code_fingerprint=code_fingerprint,
        settings_fingerprint=settings_fingerprint,
        seed=seed,
        model_only=model_only,
        model_artifacts=model_artifacts,
        model_artifact_fingerprints=model_artifact_fingerprints,
        allowlist=allowlist,
        judgments_suffix=judgments_file.suffix,
        companies_suffix=companies_file.suffix,
    )
    with recorder.stage("disclosure_configuration") as record:
        record["allowlisted_files"] = len(allowlist)
        record["minimum_aggregate_cell"] = settings.min_cell_n
    known_identifiers = _bounded_known_identifiers(sample)
    report_allowlist = {
        name: policy
        for name, policy in allowlist.items()
        if name not in {"E5_run_log.csv", "E5_run_manifest.json"}
    }
    with recorder.stage("E5_disclosure_gate") as record:
        # Check a temporary copy first; the final files are then copied one by one.
        with TemporaryDirectory(prefix=".disclosure-preview-", dir=paths.root) as temp:
            preview = stage_egress(
                aggregate,
                Path(temp) / "egress",
                allowlist=report_allowlist,
                min_cell_n=settings.min_cell_n,
                known_identifiers=known_identifiers,
            )
        record["staged_files"] = len(allowlist)
        record["suppressed_rows"] = sum(
            count for _, count in preview.suppressed_rows
        )
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
    if stage == "diagnostic":
        paths.models.rmdir()
    return paths


# Build the summary and run record

def _matching_funnel(
    judgments: pd.DataFrame,
    matches: pd.DataFrame,
    sample_rows: int,
) -> pd.DataFrame:
    """Show that every valid input row received one matching decision."""

    tiers = matches["tier"].value_counts()
    return pd.DataFrame(
        [
            {"stage": "judgments_read", "rows": int(len(judgments))},
            {"stage": "matching_decisions", "rows": int(len(matches))},
            {
                "stage": "unique_exact_name",
                "rows": int(tiers.get("exact_unique", 0)),
            },
            {"stage": "unmatched", "rows": int(tiers.get("unmatched", 0))},
            {"stage": "matching_pair_examples", "rows": int(sample_rows)},
        ]
    )


def _diagnostic_summary_context(
    audit: DataAudit,
    matches: pd.DataFrame,
    pair_file: str,
    settings: Settings,
) -> dict[str, Any]:
    """Build the Run 1 cover-page values from the full matching population."""

    tiers = matches["tier"].value_counts()
    denominator = int(len(matches))
    exact_unique = int(tiers.get("exact_unique", 0))
    return {
        "scope": "matching_only",
        "stage": "diagnostic",
        "status": "MATCHING COMPLETE",
        "observation_date": audit.observation_date,
        "min_cell_n": settings.min_cell_n,
        "counts": {
            "rows_read": audit.rows,
            "matching_decisions": denominator,
            "missing_company_name": audit.missing_company_name_rows,
            "missing_postcode": audit.missing_postcode_rows,
            "date_inserted_before_judgment": (
                audit.date_inserted_before_judgment_rows
            ),
        },
        "match": {
            "denominator": denominator,
            "exact_unique": exact_unique,
            "unmatched": int(tiers.get("unmatched", 0)),
            "coverage": _safe_coverage(
                exact_unique,
                denominator,
                (exact_unique,),
                settings.min_cell_n,
            ),
        },
        "pair_file": pair_file,
    }


def _locked_summary_context(
    stage: str,
    audit: DataAudit,
    judgments: pd.DataFrame,
    matches: pd.DataFrame,
    evaluation: ModelEvaluation,
    pair_file: str,
    settings: Settings,
) -> dict[str, Any]:
    cohort = evaluation.cohort
    primary_key = f"prospective.{evaluation.champions['prospective']}"
    snapshot_key = f"snapshot_exploratory.{evaluation.champions['snapshot_exploratory']}"
    primary_run = evaluation.runs[primary_key]
    snapshot_run = evaluation.runs[snapshot_key]
    primary = primary_run.test_metrics_calibrated
    intervals = primary_run.bootstrap_intervals.get("roc_auc", {})
    corporate = judgments["DefendantType"].eq("Corporate")
    tiers = matches.loc[corporate, "tier"].value_counts()
    model_only = evaluation.primary_acceptance
    gate_text = (
        "PASS"
        if model_only["passed"]
        else "FAIL - " + ", ".join(model_only["reasons"])
    )
    return {
        "scope": "model",
        "stage": stage,
        "status": model_only["status"].upper(),
        "observation_date": audit.observation_date,
        "min_cell_n": settings.min_cell_n,
        "counts": {
            "rows_read": audit.rows,
            "corporate_ew_labelled": cohort.funnel.get(
                "satisfied_or_unsatisfied_corporate_ew"
            ),
            "primary_age_eligible": cohort.funnel.get(
                "seasoned_12_36_corporate_ew_labelled"
            ),
            "exact_unique_matched_eligible": cohort.funnel.get(
                "exact_unique_matched_eligible"
            ),
            "model_rows": len(cohort.frame),
            "satisfied": int(cohort.frame["label"].sum()),
            "unsatisfied": int(len(cohort.frame) - cohort.frame["label"].sum()),
        },
        "match": {
            "denominator": int(corporate.sum()),
            "exact_unique": int(tiers.get("exact_unique", 0)),
            "unmatched": int(tiers.get("unmatched", 0)),
        },
        "splits": cohort.split_counts,
        "primary": {
            "champion": evaluation.champions["prospective"],
            "roc_auc": primary.get("roc_auc"),
            "roc_auc_ci_low": intervals.get("lower"),
            "roc_auc_ci_high": intervals.get("upper"),
            "average_precision": primary.get("average_precision"),
            "brier": primary.get("brier"),
            "baseline_brier": primary.get("null_brier"),
            "mean_predicted": primary.get("mean_prediction"),
            "observed_rate": primary.get("base_rate"),
        },
        "exploratory": {
            "roc_auc": snapshot_run.test_metrics_calibrated.get("roc_auc")
        },
        "gates": {"primary_model": gate_text},
        "pair_file": pair_file,
    }


def _run_manifest(
    *,
    paths: RunPaths,
    stage: str,
    observed: pd.Timestamp,
    audit: DataAudit,
    ch_index: CHIndex,
    settings: Settings,
    code_fingerprint: str,
    settings_fingerprint: str,
    seed: int,
    model_only: dict[str, Any] | None,
    model_artifacts: dict[str, str],
    model_artifact_fingerprints: dict[str, str],
    allowlist: dict[str, str | Sequence[str] | None],
    judgments_suffix: str,
    companies_suffix: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "status": "RT INTERNAL - NOT AUTHORISED FOR EXTERNAL USE",
        "run_id": paths.root.name,
        "stage": stage,
        "observation_date": observed.date().isoformat(),
        "input_formats": {
            "judgments": judgments_suffix.casefold(),
            "companies_house": companies_suffix.casefold(),
        },
        "fingerprints": {
            "rt_analysis_content": audit.analysis_fingerprint,
            "ch_analysis_content": ch_index.stats.get("analysis_fingerprint"),
            "code_and_settings": code_fingerprint,
            "settings_file": settings_fingerprint,
        },
        "settings": settings.as_dict(),
        "matching_rule": "unique_date_valid_exact_normalized_name_v1",
        "sample_seed": seed,
        "model_artifact_sha256": model_artifact_fingerprints,
        "ch_index_stats": dict(ch_index.stats),
        "model_only_acceptance": model_only,
        "package_versions": _package_versions(),
        "artifact_manifest": {
            "reports": sorted(allowlist),
            "working_files": [
                PAIR_FILENAME,
                *sorted(
                    f"models/{Path(value).name}"
                    for value in model_artifacts.values()
                ),
            ],
            "models_require_separate_rt_approval": stage == "locked",
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
    return {
        **common,
        "E3_model_comparison.csv": ("test_rows", "test_positive", "test_negative"),
        "E3_calibration.csv": ("rows", "positive", "negative"),
        "E3_feature_effects.csv": "support_rows",
        "E3_split_counts.csv": ("rows", "unique_companies", "positive", "negative"),
        "E4_sensitivities.csv": ("rows", "unique_companies", "positive", "negative"),
        "E4_lift.csv": ("selected", "selected_positive", "selected_negative"),
        "E4_limitations.txt": None,
    }


def _limitations_text(stage: str) -> str:
    return "\n".join(
        [
            "RT INTERNAL - NOT AUTHORISED FOR EXTERNAL USE",
            f"Run stage: {stage}",
            "The label is recorded Satisfied versus Unsatisfied on the observation date.",
            "It is not a payment-within-12-months label and does not measure partial recovery.",
            "Date Inserted is used only to audit registration delay.",
            "The primary model uses judgment-time-reconstructable fields only.",
            "Current Companies House fields are retrospective and exploratory.",
            "The free Companies House bulk file omits dissolved defendants.",
            "E4 population sensitivities are descriptive and are not extra fitted models.",
            "Only unique exact normalized-name matches enter the headline model.",
        ]
    )


# File fingerprints and small helpers

def _bounded_known_identifiers(sample: pd.DataFrame) -> tuple[str, ...]:
    values: list[str] = []
    columns = (
        "source_company_name",
        "source_trading_name",
        "matched_company_name",
    )
    for column in columns:
        if column in sample:
            values.extend(sample[column].astype("string").fillna("").tolist())
    distinctive = (
        value.strip()
        for value in values
        if len(value.strip()) >= 8 and any(character.isalpha() for character in value)
    )
    return tuple(dict.fromkeys(distinctive))


def _safe_coverage(
    numerator: int,
    denominator: int,
    components: Sequence[int],
    min_cell_n: int,
) -> float | None:
    """Suppress a rate that could reconstruct a small positive component."""

    if not denominator:
        return 0.0
    if any(0 < value < min_cell_n for value in components):
        return None
    return numerator / denominator


def _small_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_versions() -> dict[str, str]:
    names = (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "openpyxl",
        "narwhals",
        "tzdata",
    )
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


# Command line used by RUN.bat

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline Registry Trust recovery analysis"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    analyze_parser = commands.add_parser(
        "analyze", help="run diagnostic or locked analysis"
    )
    analyze_parser.add_argument(
        "--stage", choices=("diagnostic", "locked"), required=True
    )
    analyze_parser.add_argument("--judgments", required=True)
    analyze_parser.add_argument("--companies-house", required=True)
    analyze_parser.add_argument("--observation-date")
    analyze_parser.add_argument("--settings", default="settings.toml")
    analyze_parser.add_argument("--output-base", default="outputs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        paths = analyze(
            stage=args.stage,
            judgments_path=args.judgments,
            companies_house_path=args.companies_house,
            observation_date=args.observation_date,
            settings_path=args.settings,
            output_base=args.output_base,
        )
        print(f"OPEN THIS SUMMARY: {paths.results / 'SUMMARY.txt'}")
        print(f"MATCHING PAIRS: {paths.working / PAIR_FILENAME}")
        if args.stage == "diagnostic":
            print("No satisfaction model was run.")
        else:
            print("The satisfaction models were also run.")
        return 0
    except Exception as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
