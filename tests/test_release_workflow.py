"""Synthetic integration test for the frozen one-time release workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from recovery.locking import (
    ReleaseLockError,
    build_design_manifest,
    create_release_approval,
    generate_release_key,
    write_design_manifest,
)
from recovery.run import (
    DEVELOPMENT_SPECIFICATION_FILENAME,
    PAIR_FILENAME,
    UNMATCHED_FILENAME,
    analyze,
)
from recovery.synthetic import make_synthetic_bundle, write_bundle


ROOT = Path(__file__).resolve().parents[1]


def _complete_reviews(path: Path, truth_path: Path, arm: str) -> None:
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    if arm == "accepted":
        label = pd.Series("correct_match", index=frame.index, dtype="string")
        company = pd.Series("", index=frame.index, dtype="string")
    else:
        truth = pd.read_csv(
            truth_path,
            dtype={"ID": "string", "expected_company_number": "string"},
        )
        frame = frame.merge(truth, on="ID", how="left", validate="one_to_one")
        missed = frame["corruption_class"].isin(
            ("same_postcode_wrong_name", "same_postcode_ambiguous")
        )
        label = pd.Series(
            pd.NA, index=frame.index, dtype="string"
        ).mask(missed, "missed_match").fillna("true_unmatched")
        company = frame["expected_company_number"].where(missed, "").astype("string")
        frame = frame.drop(
            columns=("expected_company_number", "corruption_class"), errors="ignore"
        )
    frame["reviewer_1_label"] = label
    frame["reviewer_1_company_number"] = company
    frame["reviewer_2_label"] = label
    frame["reviewer_2_company_number"] = company
    frame["adjudicated_label"] = label
    frame["adjudicated_company_number"] = company
    frame["adjudication_notes"] = ""
    frame.to_csv(path, index=False)


def test_development_freeze_and_locked_release_are_separate_and_single_use(
    tmp_path: Path,
) -> None:
    bundle = make_synthetic_bundle(1_600, include_prior_rows=False)
    judgments, companies, truth = write_bundle(bundle, tmp_path / "inputs", excel=False)
    settings = tmp_path / "settings.toml"
    settings.write_text(
        (ROOT / "settings.toml")
        .read_text(encoding="utf-8")
        .replace("bootstrap_replicates = 1000", "bootstrap_replicates = 100"),
        encoding="utf-8",
    )

    diagnostic = analyze(
        stage="diagnostic",
        judgments_path=judgments,
        companies_house_path=companies,
        observation_date=bundle.observation_date,
        companies_house_date=bundle.observation_date,
        settings_path=settings,
        output_base=tmp_path / "outputs",
        run_id="diagnostic",
    )
    accepted = diagnostic.working / PAIR_FILENAME
    unmatched = diagnostic.working / UNMATCHED_FILENAME
    _complete_reviews(accepted, truth, "accepted")
    _complete_reviews(unmatched, truth, "unmatched")

    development = analyze(
        stage="development",
        judgments_path=judgments,
        companies_house_path=companies,
        observation_date=bundle.observation_date,
        companies_house_date=bundle.observation_date,
        settings_path=settings,
        output_base=tmp_path / "outputs",
        accepted_adjudications_path=accepted,
        unmatched_adjudications_path=unmatched,
        run_id="development",
    )
    specification = development.working / DEVELOPMENT_SPECIFICATION_FILENAME
    assert specification.is_file()
    assert "TEST NOT ACCESSED" in (
        development.results / "SUMMARY.txt"
    ).read_text(encoding="utf-8")
    assert not (development.results / "E3_model_comparison.csv").exists()

    manifest_path = tmp_path / "frozen_manifest.json"
    manifest = build_design_manifest(
        judgments_path=judgments,
        companies_house_path=companies,
        observation_date=bundle.observation_date,
        companies_house_date=bundle.observation_date,
        settings_path=settings,
        package_dir=ROOT / "recovery",
        bound_files={
            "accepted_adjudications": accepted,
            "unmatched_adjudications": unmatched,
            "development_specification": specification,
        },
    )
    write_design_manifest(manifest_path, manifest)
    key = generate_release_key(tmp_path / "custodian.release-key")
    approval = tmp_path / "approval.json"
    create_release_approval(
        manifest_path=manifest_path,
        key_path=key,
        approval_path=approval,
        approval_id="synthetic-release-0001",
    )

    registry = tmp_path / "receipts"
    locked = analyze(
        stage="locked",
        judgments_path=judgments,
        companies_house_path=companies,
        observation_date=bundle.observation_date,
        companies_house_date=bundle.observation_date,
        settings_path=settings,
        output_base=tmp_path / "outputs",
        accepted_adjudications_path=accepted,
        unmatched_adjudications_path=unmatched,
        development_specification_path=specification,
        manifest_path=manifest_path,
        approval_path=approval,
        key_path=key,
        release_registry=registry,
        run_id="locked",
    )
    comparison = pd.read_csv(locked.results / "E3_model_comparison.csv")
    assert set(comparison["role"]) == {
        "baseline",
        "age_only_baseline",
        "frozen_primary",
    }
    capacities = {"1pct", "2pct", "5pct", "10pct", "20pct"}
    evaluation = json.loads(
        (locked.models / "model_evaluation.json").read_text(encoding="utf-8")
    )
    for run in evaluation["runs"].values():
        assert set(run["test_metrics_calibrated"]["capacity_metrics"]) == capacities

    # The synthetic test set is deliberately small. Its public ranking rows may
    # all be removed by the minimum-cell disclosure rule; the internal locked
    # evaluation above must nevertheless contain every prespecified capacity.
    ranking = pd.read_csv(locked.results / "E3_operational_ranking.csv")
    assert set(ranking["capacity"]).issubset(capacities)
    receipt = pd.read_json(registry / "synthetic-release-0001.json", typ="series")
    assert receipt["status"] == "completed"

    with pytest.raises(ReleaseLockError, match="already been consumed"):
        analyze(
            stage="locked",
            judgments_path=judgments,
            companies_house_path=companies,
            observation_date=bundle.observation_date,
            companies_house_date=bundle.observation_date,
            settings_path=settings,
            output_base=tmp_path / "outputs",
            accepted_adjudications_path=accepted,
            unmatched_adjudications_path=unmatched,
            development_specification_path=specification,
            manifest_path=manifest_path,
            approval_path=approval,
            key_path=key,
            release_registry=registry,
            run_id="locked_again",
        )
    assert not (tmp_path / "outputs" / "locked_locked_again").exists()
