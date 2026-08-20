"""Run the two RT stages and check a completed 1,000-pair review.

Run 1 sends every valid judgment through matching, writes the matching reports
and stops. Run 2 repeats matching from scratch and then runs the deferred
satisfaction analysis. Named rows and models stay in ``rt_internal``; only
checked aggregate files can reach ``egress_candidate``. Nothing here connects
to the internet or starts another program.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence
import argparse
import hashlib
import json
import re
import sys
import unicodedata

import pandas as pd

from .config import Settings, load_settings
from .data import DataAudit, read_rt_extract
from .disclosure import stage_egress, validate_egress
from .matching import (
    CHIndex,
    build_relevant_ch_index,
    match_diagnostics,
    match_judgments,
    review_sample,
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
    write_json,
    write_summary,
)
from .review import ReviewResult, parse_completed_review, write_review_aggregates


# ===== filenames and values that bind a review to its original run =====

PAIR_FILENAME = "RT_INTERNAL_match_pairs_1000.csv"
STATE_FILENAME = "RT_INTERNAL_review_state.json"
MATCH_FILENAME = "RT_INTERNAL_match_table.csv.gz"
SPLIT_FILENAME = "RT_INTERNAL_split_membership.csv"
_MUTABLE_REVIEW_COLUMNS = frozenset({"review_decision", "review_notes"})
_NUMERIC_REVIEW_COLUMNS = frozenset(
    {
        "score",
        "runner_up_score",
        "margin",
        "postcode_candidate_count",
        "exact_name_candidate_count",
        "rejected_post_incorporation",
        "sample_seed",
        "sample_allocation",
        "sampling_weight",
        "Mortgages.NumMortCharges",
        "Mortgages.NumMortOutstanding",
        "Mortgages.NumMortPartSatisfied",
        "Mortgages.NumMortSatisfied",
    }
)
_DATE_REVIEW_COLUMNS = frozenset(
    {
        "JudgmentDate",
        "IncorporationDate",
        "Accounts.NextDueDate",
        "Accounts.LastMadeUpDate",
    }
)
_BOOL_REVIEW_COLUMNS = frozenset({"postcode_agrees", "incorporation_date_missing"})
_MODEL_REASON_CODES = frozenset(
    {
        "test_rows_below_minimum",
        "test_class_count_below_minimum",
        "calibration_underpowered",
        "auc_below_floor",
        "auc_lower_ci_not_above_chance",
        "calibration_gap_above_maximum",
        "calibration_slope_outside_range",
        "brier_not_better_than_training_prevalence",
    }
)


class RunFailure(RuntimeError):
    """A clear, operator-facing failure that stops an official run."""


@dataclass(frozen=True, slots=True)
class ReviewOutcome:
    match_review: ReviewResult
    combined_passed: bool
    combined_reasons: tuple[str, ...]
    output_dir: Path


# ===== run the full-data matcher, with models added only in Run 2 =====

def analyze(
    *,
    stage: str,
    judgments_path: str | Path,
    companies_house_path: str | Path,
    observation_date: str | date | None,
    settings_path: str | Path,
    output_base: str | Path,
    run_id: str | None = None,
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
    aggregate = paths.internal / "aggregate_staging"
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
        matches = match_judgments(judgments, ch_index, settings)
        diagnostics = match_diagnostics(judgments, matches)
        record["judgments_matched"] = int(matches["tier"].ne("unmatched").sum())
        matches.to_csv(
            paths.internal / MATCH_FILENAME,
            index=False,
            compression="gzip",
            encoding="utf-8",
        )

    with recorder.stage("RT_internal_review_sample") as record:
        seed = settings.diagnostic_seed if stage == "diagnostic" else settings.locked_seed
        sample = review_sample(judgments, matches, settings, seed=seed)
        if len(sample) != 1_000:
            raise RunFailure(
                "fewer than 1,000 proposed matches are available for the required review sample"
            )
        pair_path = paths.internal / PAIR_FILENAME
        sample.to_csv(pair_path, index=False, encoding="utf-8-sig")
        # The provenance baseline is the exact serialized file RT will edit,
        # not pandas' pre-CSV objects (which may format dates differently).
        review_baseline = _read_review_frame(pair_path)
        record["sample_rows"] = len(sample)
        record["sample_seed"] = seed

    evaluation: ModelEvaluation | None = None
    model_artifacts: dict[str, str] = {}
    if stage == "locked":
        with recorder.stage("E3_prepare_cohort") as record:
            cohort = prepare_model_cohort(judgments, matches, observed, settings)
            record["model_rows"] = len(cohort.frame)
            cohort.frame[
                ["ID", "matched_company_number", "JudgmentDate", "label", "split"]
            ].to_csv(
                paths.internal / SPLIT_FILENAME,
                index=False,
                encoding="utf-8-sig",
            )

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
    model_only = (
        _model_only_acceptance(evaluation) if evaluation is not None else None
    )
    model_artifact_fingerprints = {
        Path(path).name: _small_file_sha256(Path(path))
        for path in sorted(model_artifacts.values())
    }
    review_state = {
        "schema_version": 1,
        "data_classification": "RT INTERNAL - DO NOT EGRESS",
        "run_id": paths.root.name,
        "stage": stage,
        "observation_date": observed.date().isoformat(),
        "sample_seed": seed,
        "sample_rows": len(sample),
        "sample_allocation": {
            str(key): int(value)
            for key, value in sample.groupby("review_tier").size().items()
        },
        "immutable_columns": sorted(
            set(review_baseline.columns) - _MUTABLE_REVIEW_COLUMNS
        ),
        "sample_digest": review_sample_digest(review_baseline),
        "sampling_design": "equal_probability_systematic_stratified_v1",
        "settings_fingerprint": settings_fingerprint,
        "code_fingerprint": code_fingerprint,
        "rt_analysis_fingerprint": audit.analysis_fingerprint,
        "ch_analysis_fingerprint": ch_index.stats.get("analysis_fingerprint"),
        "model_artifact_fingerprints": model_artifact_fingerprints,
        "model_only_acceptance": model_only,
    }
    review_state_fingerprint = _json_fingerprint(review_state)

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
        review_state_fingerprint=review_state_fingerprint,
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
        # Preview the aggregate reports in a temporary directory first. This
        # gives E5 its final disclosure receipt before the one atomic copy into
        # egress_candidate; a failed final scan therefore leaves egress empty.
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
        paths.egress,
        allowlist=allowlist,
        min_cell_n=settings.min_cell_n,
        known_identifiers=known_identifiers,
    )
    if (
        tuple(disclosure.suppressed_rows) != tuple(preview.suppressed_rows)
        or set(disclosure.staged_files) != set(allowlist)
    ):
        raise RunFailure("final disclosure copy differed from its checked preview")
    write_json(paths.internal / STATE_FILENAME, review_state)
    return paths


# ===== turn RT's completed pair file into aggregate match-quality results =====

def review_completed_sample(
    *,
    review_file: str | Path,
    settings_path: str | Path,
    output_dir: str | Path,
) -> ReviewOutcome:
    """Verify immutable sample provenance and write aggregate review results."""

    review_path = Path(review_file).resolve()
    state_path = review_path.parent / STATE_FILENAME
    if not review_path.is_file():
        raise RunFailure(f"completed review file does not exist: {review_path}")
    if not state_path.is_file():
        raise RunFailure(
            f"missing adjacent {STATE_FILENAME}; review the original RT-internal sample"
        )
    settings_source = Path(settings_path).resolve()
    settings = load_settings(settings_source)
    frame = _read_review_frame(review_path)
    state, manifest = _load_and_validate_review_state(
        state_path,
        review_path,
        frame,
        settings_source,
        settings,
    )
    review = parse_completed_review(
        frame,
        settings,
        tier_column="review_tier",
        decision_column="review_decision",
    )

    locked = state.get("stage") == "locked"
    model_only = state.get("model_only_acceptance") if locked else None
    reasons = list(model_only.get("reasons", [])) if model_only else []
    if not review.gate_passed:
        reasons.extend(f"match_review: {reason}" for reason in review.gate_reasons)
    combined_passed = bool(
        locked and model_only and model_only.get("passed") and not reasons
    )
    if (
        locked
        and model_only
        and not model_only.get("passed")
        and not model_only.get("reasons")
    ):
        reasons.append("model_only_acceptance_failed")
        combined_passed = False
    if not locked:
        reasons.append("satisfaction_model_deferred")

    destination = Path(output_dir)
    with TemporaryDirectory(prefix="review-aggregate-") as temporary:
        staging = Path(temporary)
        write_review_aggregates(review, staging, min_cell_n=settings.min_cell_n)
        auto = review.stats.loc[review.stats["tier"].eq("auto")].iloc[0]
        auto_outcome_counts = (
            int(auto["n_correct"]),
            int(auto["n_incorrect"]),
            int(auto["n_uncertain"]),
        )
        auto_detail_suppressed = any(
            0 < count < settings.min_cell_n for count in auto_outcome_counts
        )
        public_auto_precision = (
            None if auto_detail_suppressed else float(auto["observed_precision"])
        )
        public_auto_wilson_lower = (
            None if auto_detail_suppressed else float(auto["wilson_lower_95"])
        )
        public_auto_wilson_upper = (
            None if auto_detail_suppressed else float(auto["wilson_upper_95"])
        )
        match_status = {
            "schema_version": 1,
            "run_id": state.get("run_id"),
            "run_stage": state.get("stage"),
            "observation_date": state.get("observation_date"),
            "rt_analysis_fingerprint": state.get("rt_analysis_fingerprint"),
            "ch_analysis_fingerprint": state.get("ch_analysis_fingerprint"),
            "code_fingerprint": state.get("code_fingerprint"),
            "settings_fingerprint": state.get("settings_fingerprint"),
            "review_sample_fingerprint": state.get("sample_digest"),
            # The reviewed CSV remains the decision record. A public digest of
            # its low-entropy answer vector could reveal rare outcomes.
            "review_decision_fingerprint": None,
            "match_gate_passed": review.gate_passed,
            "auto_rows_reviewed": int(auto["n_reviewed"]),
            "auto_quality_detail_suppressed": auto_detail_suppressed,
            "auto_observed_precision": public_auto_precision,
            "auto_wilson_lower_95": public_auto_wilson_lower,
            "auto_wilson_upper_95": public_auto_wilson_upper,
            "analysis_disclosure_status": manifest["disclosure"]["status"],
        }
        write_json(staging / "MATCH_REVIEW_STATUS.json", match_status)
        match_status_lines = [
            "DRAFT - RT REVIEW REQUIRED - NOT AUTHORISED FOR EXTERNAL USE",
            "MATCH REVIEW STATUS",
            "===================",
            f"Run ID: {state.get('run_id')}",
            f"Run stage: {state.get('stage')}",
            f"Observation date: {state.get('observation_date')}",
            f"Auto match-review gate: {'PASS' if review.gate_passed else 'FAIL'}",
            "Auto quality detail suppressed: "
            + ("yes" if auto_detail_suppressed else "no"),
            "Auto precision: "
            + (
                "SUPPRESSED (small nonzero outcome cell)"
                if public_auto_precision is None
                else f"{public_auto_precision:.6f}"
            ),
            "Auto Wilson lower 95%: "
            + (
                "SUPPRESSED (small nonzero outcome cell)"
                if public_auto_wilson_lower is None
                else f"{public_auto_wilson_lower:.6f}"
            ),
        ]
        if not locked:
            match_status_lines.append(
                "Next step: agree the satisfaction analysis with RT before Run 2."
            )
        (staging / "MATCH_REVIEW_STATUS.txt").write_text(
            "\n".join(match_status_lines) + "\n", encoding="utf-8"
        )
        allowlist: dict[str, str | Sequence[str] | None] = {
            "E2_review_quality.csv": (
                "n_reviewed",
                "n_correct",
                "n_incorrect",
                "n_uncertain",
            ),
            "E2_review_quality.txt": None,
            "MATCH_REVIEW_STATUS.json": None,
            "MATCH_REVIEW_STATUS.txt": None,
        }
        if locked and model_only is not None:
            combined = {
                **match_status,
                "model_artifact_fingerprints": state.get(
                    "model_artifact_fingerprints"
                ),
                "model_only_gate_passed": bool(model_only.get("passed")),
                "combined_final_passed": combined_passed,
                "combined_reasons": reasons,
                "model_family": model_only.get("family"),
                "model_algorithm": model_only.get("algorithm"),
            }
            write_json(staging / "FINAL_STATUS.json", combined)
            final_lines = [
                "DRAFT - RT REVIEW REQUIRED - NOT AUTHORISED FOR EXTERNAL USE",
                "LOCKED ANALYSIS FINAL STATUS",
                "============================",
                f"Run ID: {state.get('run_id')}",
                f"Observation date: {state.get('observation_date')}",
                f"Auto match-review gate: {'PASS' if review.gate_passed else 'FAIL'}",
                "Auto quality detail suppressed: "
                + ("yes" if auto_detail_suppressed else "no"),
                "Auto precision: "
                + (
                    "SUPPRESSED (small nonzero outcome cell)"
                    if public_auto_precision is None
                    else f"{public_auto_precision:.6f}"
                ),
                "Auto Wilson lower 95%: "
                + (
                    "SUPPRESSED (small nonzero outcome cell)"
                    if public_auto_wilson_lower is None
                    else f"{public_auto_wilson_lower:.6f}"
                ),
                f"Model-only gates: {'PASS' if model_only.get('passed') else 'FAIL'}",
                f"Combined final status: {'PASS' if combined_passed else 'FAIL'}",
                "Reasons: " + (", ".join(reasons) if reasons else "none"),
            ]
            (staging / "FINAL_STATUS.txt").write_text(
                "\n".join(final_lines) + "\n", encoding="utf-8"
            )
            allowlist.update(
                {"FINAL_STATUS.json": None, "FINAL_STATUS.txt": None}
            )
        stage_egress(
            staging,
            destination,
            allowlist=allowlist,
            min_cell_n=settings.min_cell_n,
        )
    return ReviewOutcome(review, combined_passed, tuple(reasons), destination)


# ===== protect the sampled rows while allowing RT to fill in two answer columns =====

def review_sample_digest(frame: pd.DataFrame) -> str:
    """Hash immutable review cells independent of row/column order and CSV typing."""

    columns = sorted(set(map(str, frame.columns)) - _MUTABLE_REVIEW_COLUMNS)
    if "review_row_id" not in columns:
        raise RunFailure("review sample is missing review_row_id")
    if frame["review_row_id"].astype("string").duplicated().any():
        raise RunFailure("review_row_id values are not unique")
    rows = frame.sort_values("review_row_id", kind="stable")
    digest = hashlib.sha256()
    digest.update("\x1f".join(columns).encode("utf-8"))
    for values in rows.loc[:, columns].itertuples(index=False, name=None):
        canonical = [
            _canonical_review_value(column, value)
            for column, value in zip(columns, values)
        ]
        digest.update("\x1f".join(canonical).encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def _load_and_validate_review_state(
    state_path: Path,
    review_path: Path,
    frame: pd.DataFrame,
    settings_source: Path,
    settings: Settings,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind one completed review to the exact originating analysis."""

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunFailure("review-state file is not readable JSON") from exc
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise RunFailure("unsupported review-state schema")
    if review_path.parent.name.casefold() != "rt_internal":
        raise RunFailure("completed review must remain inside its original rt_internal folder")
    run_root = review_path.parent.parent
    manifest_path = run_root / "egress_candidate" / "E5_run_manifest.json"
    if not manifest_path.is_file():
        raise RunFailure("the analysis E5 manifest is missing beside this review run")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunFailure("the analysis E5 manifest is not readable JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RunFailure("unsupported E5 manifest schema")

    stage = state.get("stage")
    if stage not in {"diagnostic", "locked"}:
        raise RunFailure("review state has an invalid run stage")
    egress = run_root / "egress_candidate"
    expected_files = set(_analysis_allowlist(stage))
    entries = list(egress.iterdir()) if egress.is_dir() else []
    actual_files = {path.name for path in entries}
    if (
        actual_files != expected_files
        or any(not path.is_file() or path.is_symlink() for path in entries)
    ):
        raise RunFailure("analysis egress files differ from the saved run stage")
    validate_egress(
        egress,
        known_identifiers=_bounded_known_identifiers(frame),
    )
    expected_seed = (
        settings.diagnostic_seed if stage == "diagnostic" else settings.locked_seed
    )
    if state.get("sample_seed") != expected_seed:
        raise RunFailure("review sample seed does not match its declared run stage")
    if int(state.get("sample_rows", -1)) != 1_000 or len(frame) != 1_000:
        raise RunFailure("review state and completed sample must each contain 1,000 rows")

    immutable = sorted(set(map(str, frame.columns)) - _MUTABLE_REVIEW_COLUMNS)
    if state.get("immutable_columns") != immutable:
        raise RunFailure("review sample columns differ from the saved schema")
    if review_sample_digest(frame) != state.get("sample_digest"):
        raise RunFailure(
            "the review sample rows changed; only review_decision and review_notes may be edited"
        )
    tiers = frame["review_tier"].astype("string").str.strip().str.casefold()
    tiers = tiers.replace({"fallback": "fallback_review"})
    actual_allocation = {
        str(key): int(value) for key, value in tiers.value_counts().items()
    }
    if state.get("sample_allocation") != actual_allocation:
        raise RunFailure("review sample allocation differs from the saved state")
    if "sample_seed" not in frame or not frame["sample_seed"].astype(str).eq(
        str(expected_seed)
    ).all():
        raise RunFailure("review sample seed column differs from the saved state")
    if state.get("sampling_design") != "equal_probability_systematic_stratified_v1":
        raise RunFailure("review state has an unsupported sampling design")
    if "sampling_design" not in frame or not frame["sampling_design"].eq(
        state["sampling_design"]
    ).all():
        raise RunFailure("review sample design column differs from the saved state")

    if _small_file_sha256(settings_source) != state.get("settings_fingerprint"):
        raise RunFailure("settings.toml differs from the settings used for this sample")
    if source_fingerprint(Path(__file__).parent, settings_source) != state.get(
        "code_fingerprint"
    ):
        raise RunFailure("analysis source differs from the source used for this sample")
    for key in (
        "settings_fingerprint",
        "code_fingerprint",
        "rt_analysis_fingerprint",
        "ch_analysis_fingerprint",
        "sample_digest",
    ):
        if not _is_sha256(state.get(key)):
            raise RunFailure(f"review state has an invalid {key}")

    model_hashes = state.get("model_artifact_fingerprints")
    model_only = state.get("model_only_acceptance")
    if stage == "diagnostic":
        if model_only is not None or model_hashes != {}:
            raise RunFailure("matching-only review state must not contain model results")
    else:
        if not isinstance(model_only, dict):
            raise RunFailure("review state is missing model-only acceptance")
        reasons = model_only.get("reasons")
        if not isinstance(reasons, list) or not all(
            isinstance(reason, str) and reason in _MODEL_REASON_CODES
            for reason in reasons
        ):
            raise RunFailure("review state contains invalid model-acceptance reasons")
        passed = model_only.get("passed")
        if not isinstance(passed, bool) or passed != (not reasons):
            raise RunFailure("review state model pass flag is inconsistent with its reasons")
        if model_only.get("status") != ("pass" if passed else "fail"):
            raise RunFailure("review state model status is inconsistent")
        if model_only.get("family") != "prospective" or model_only.get(
            "algorithm"
        ) not in {"logistic", "lightgbm"}:
            raise RunFailure("review state identifies an invalid primary model")
        if not isinstance(model_hashes, dict) or not model_hashes:
            raise RunFailure("review state is missing model artefact fingerprints")
        for filename, expected in model_hashes.items():
            if Path(str(filename)).name != filename or not _is_sha256(expected):
                raise RunFailure("review state has an invalid model artefact fingerprint")
            artifact = review_path.parent / "models" / filename
            if not artifact.is_file() or _small_file_sha256(artifact) != expected:
                raise RunFailure(
                    f"saved model artefact differs or is missing: {filename}"
                )

    binding = manifest.get("review_binding", {})
    if binding.get("review_state_sha256") != _json_fingerprint(state):
        raise RunFailure("review state does not match its E5 manifest binding")
    if binding.get("model_artifact_sha256") != model_hashes:
        raise RunFailure("model artefact bindings differ between review state and E5")
    fingerprints = manifest.get("fingerprints", {})
    expected_fingerprints = {
        "rt_analysis_content": state["rt_analysis_fingerprint"],
        "ch_analysis_content": state["ch_analysis_fingerprint"],
        "code_and_settings": state["code_fingerprint"],
        "settings_file": state["settings_fingerprint"],
    }
    if any(fingerprints.get(key) != value for key, value in expected_fingerprints.items()):
        raise RunFailure("review-state fingerprints differ from the E5 manifest")
    if any(
        (
            manifest.get("run_id") != state.get("run_id"),
            manifest.get("stage") != stage,
            manifest.get("sample_seed") != expected_seed,
            manifest.get("observation_date") != state.get("observation_date"),
            not _same_public_model_acceptance(
                manifest.get("model_only_acceptance"), model_only
            ),
            manifest.get("disclosure", {}).get("status") != "pass",
        )
    ):
        raise RunFailure("review state is inconsistent with its completed E5 manifest")
    if state.get("run_id") != run_root.name:
        raise RunFailure("review state run ID does not match its containing run folder")
    return state, manifest


def _canonical_review_value(column: str, value: Any) -> str:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if column in _NUMERIC_REVIEW_COLUMNS:
        try:
            numeric = Decimal(text)
        except InvalidOperation as exc:
            raise RunFailure(f"review column {column} contains a non-numeric value") from exc
        return format(numeric.normalize(), "f")
    if column in _DATE_REVIEW_COLUMNS:
        # Pandas' mixed parser applies ``dayfirst`` even to ISO-looking values:
        # for example, 2024-01-02 can become 1 February.  Review provenance must
        # instead treat the pipeline's YYYY-MM-DD serialization unambiguously,
        # while still accepting the equivalent UK date Excel may write back.
        iso_date = re.fullmatch(
            r"(\d{4}-\d{2}-\d{2})(?:[ T]00:00:00(?:\.0+)?)?", text
        )
        uk_date = re.fullmatch(
            r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?:[ T]00:00:00(?:\.0+)?)?",
            text,
        )
        if iso_date:
            parsed = pd.to_datetime(
                iso_date.group(1), format="%Y-%m-%d", errors="coerce"
            )
        elif uk_date:
            day, month, year = uk_date.groups()
            parsed = pd.to_datetime(
                f"{day}/{month}/{year}", format="%d/%m/%Y", errors="coerce"
            )
        else:
            parsed = pd.NaT
        if pd.isna(parsed):
            raise RunFailure(f"review column {column} contains an invalid date")
        return pd.Timestamp(parsed).date().isoformat()
    if column in _BOOL_REVIEW_COLUMNS:
        folded = text.casefold()
        if folded in {"true", "1", "yes"}:
            return "true"
        if folded in {"false", "0", "no"}:
            return "false"
        raise RunFailure(f"review column {column} contains an invalid boolean")
    if column == "matched_company_number" and text.isdigit():
        return text.zfill(8)
    return text


def _read_review_frame(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() == ".csv":
        return pd.read_csv(
            path,
            dtype="string",
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    if path.suffix.casefold() in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, dtype="string", keep_default_na=False)
    raise RunFailure("completed review must be CSV or XLSX")


def _model_only_acceptance(evaluation: ModelEvaluation) -> dict[str, Any]:
    result = deepcopy(evaluation.primary_acceptance)
    result["reasons"] = [
        reason for reason in result.get("reasons", []) if reason != "match_audit_not_supplied"
    ]
    result["passed"] = not result["reasons"]
    result["status"] = "pass" if result["passed"] else "fail"
    return result


# ===== build the short summaries and audit manifest for each stage =====

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
            {"stage": "auto", "rows": int(tiers.get("auto", 0))},
            {"stage": "review", "rows": int(tiers.get("review", 0))},
            {
                "stage": "fallback_review",
                "rows": int(tiers.get("fallback_review", 0)),
            },
            {"stage": "unmatched", "rows": int(tiers.get("unmatched", 0))},
            {"stage": "manual_review_sample", "rows": int(sample_rows)},
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
    postcode_proposed = int(tiers.get("auto", 0) + tiers.get("review", 0))
    proposed = int(postcode_proposed + tiers.get("fallback_review", 0))
    auto = int(tiers.get("auto", 0))
    review = int(tiers.get("review", 0))
    fallback = int(tiers.get("fallback_review", 0))
    return {
        "scope": "matching_only",
        "stage": "diagnostic",
        "status": "PROVISIONAL - MATCH REVIEW PENDING",
        "observation_date": audit.observation_date,
        "min_cell_n": settings.min_cell_n,
        "counts": {
            "rows_read": audit.rows,
            "matching_decisions": denominator,
            "missing_company_name": audit.missing_company_name_rows,
            "missing_postcode": audit.missing_postcode_rows,
        },
        "match": {
            "denominator": denominator,
            "auto": auto,
            "review": review,
            "fallback_review": fallback,
            "unmatched": int(tiers.get("unmatched", 0)),
            "postcode_proposed": postcode_proposed,
            "postcode_coverage": _safe_coverage(
                postcode_proposed,
                denominator,
                (auto, review),
                settings.min_cell_n,
            ),
            "proposed": proposed,
            "coverage": _safe_coverage(
                proposed,
                denominator,
                (auto, review, fallback),
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
    model_only = _model_only_acceptance(evaluation)
    gate_text = (
        "PASS EXCEPT MATCH AUDIT"
        if model_only["passed"]
        else "FAIL - " + ", ".join(model_only["reasons"])
    )
    return {
        "scope": "model",
        "stage": stage,
        "status": "PROVISIONAL - MATCH REVIEW PENDING",
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
            "auto_matched_eligible": cohort.funnel.get("auto_matched_eligible"),
            "model_rows": len(cohort.frame),
            "satisfied": int(cohort.frame["label"].sum()),
            "unsatisfied": int(len(cohort.frame) - cohort.frame["label"].sum()),
        },
        "match": {
            "denominator": int(corporate.sum()),
            "auto": int(tiers.get("auto", 0)),
            "review": int(tiers.get("review", 0)),
            "fallback_review": int(tiers.get("fallback_review", 0)),
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
    review_state_fingerprint: str,
    allowlist: dict[str, str | Sequence[str] | None],
    judgments_suffix: str,
    companies_suffix: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "review_status": "DRAFT - RT REVIEW REQUIRED - NOT AUTHORISED FOR EXTERNAL USE",
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
        "sample_seed": seed,
        "review_binding": {
            "review_state_sha256": review_state_fingerprint,
            "model_artifact_sha256": model_artifact_fingerprints,
        },
        "ch_index_stats": dict(ch_index.stats),
        "model_only_acceptance": model_only,
        "package_versions": _package_versions(),
        "artifact_manifest": {
            "egress_allowlist": sorted(allowlist),
            "rt_internal": [
                PAIR_FILENAME,
                STATE_FILENAME,
                MATCH_FILENAME,
                *([SPLIT_FILENAME] if stage == "locked" else []),
                *sorted(Path(value).name for value in model_artifacts.values()),
            ],
            "rt_internal_models_require_separate_rt_approval": stage == "locked",
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
            "DRAFT - RT REVIEW REQUIRED - NOT AUTHORISED FOR EXTERNAL USE",
            f"Run stage: {stage}",
            "The label is recorded Satisfied versus Unsatisfied on the observation date.",
            "It is not a payment-within-12-months label and does not measure partial recovery.",
            "Date Inserted is used only to audit registration delay.",
            "The primary model uses judgment-time-reconstructable fields only.",
            "Current Companies House fields are retrospective and exploratory.",
            "The free Companies House bulk file omits dissolved defendants.",
            "E4 population sensitivities are descriptive and are not extra fitted models.",
            "Review and fallback matches do not enter the headline model.",
        ]
    )


# ===== disclosure-safe fingerprints and small utility functions =====

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


def _same_public_model_acceptance(public: Any, internal: Any) -> bool:
    """Compare only the non-sensitive acceptance fields carried in public E5."""

    if internal is None:
        return public is None
    if not isinstance(public, dict) or not isinstance(internal, dict):
        return False
    keys = ("status", "passed", "family", "algorithm", "reasons")
    if any(public.get(key) != internal.get(key) for key in keys):
        return False
    if not isinstance(public.get("guards"), dict) or not isinstance(
        internal.get("guards"), dict
    ):
        return public.get("guards") == internal.get("guards")
    public_guards = dict(public["guards"])
    internal_guards = dict(internal["guards"])
    internal_guards.pop("training_prevalence", None)
    return public_guards == internal_guards


def _small_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _package_versions() -> dict[str, str]:
    names = (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "openpyxl",
        "rapidfuzz",
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


# ===== command-line entry point used by the two Windows launchers =====

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
    review_parser = commands.add_parser(
        "review", help="validate a completed RT match review"
    )
    review_parser.add_argument("--review-file", required=True)
    review_parser.add_argument("--settings", default="settings.toml")
    review_parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "analyze":
            paths = analyze(
                stage=args.stage,
                judgments_path=args.judgments,
                companies_house_path=args.companies_house,
                observation_date=args.observation_date,
                settings_path=args.settings,
                output_base=args.output_base,
            )
            print(f"ANALYSIS COMPLETE: {paths.root}")
            if args.stage == "diagnostic":
                print("RUN 1: MATCHING ONLY - no satisfaction model was trained")
            else:
                print("RUN 2: LOCKED MATCHING AND SATISFACTION MODELS")
            print("STATUS: PROVISIONAL - RT match review is still required")
            return 0
        outcome = review_completed_sample(
            review_file=args.review_file,
            settings_path=args.settings,
            output_dir=args.output_dir,
        )
        print(f"MATCH REVIEW: {'PASS' if outcome.match_review.gate_passed else 'FAIL'}")
        if "satisfaction_model_deferred" in outcome.combined_reasons:
            print("RUN 1 COMPLETE: discuss satisfaction with RT before Run 2")
        else:
            print(
                f"COMBINED FINAL STATUS: {'PASS' if outcome.combined_passed else 'FAIL'}"
            )
        return 0 if outcome.match_review.gate_passed else 3
    except Exception as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
