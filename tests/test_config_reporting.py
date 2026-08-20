"""Test settings, run folders, aggregate summaries and synthetic input files.

Inputs are fixed settings and fake in-memory records. Outputs exist only in
temporary test folders and contain no confidential or row-level RT data.
"""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from recovery.config import Settings, load_settings
from recovery.reporting import (
    RunRecorder,
    build_data_audit_counts,
    build_population_sensitivities,
    create_run_paths,
    peak_memory_mb,
    source_fingerprint,
    write_e5,
    write_summary,
)
from recovery.synthetic import make_synthetic_bundle, write_bundle
from recovery.selftest import main as selftest_main


def test_repository_settings_are_valid() -> None:
    settings = load_settings(Path(__file__).parents[1] / "settings.toml")
    assert settings.auto_threshold == 0.85
    assert settings.sample_auto + settings.sample_review + settings.sample_fallback == 1_000


def test_peak_memory_measurement_is_positive() -> None:
    peak = peak_memory_mb()
    assert math.isfinite(peak)
    assert peak > 0


def test_settings_reject_unknown_or_bad_values(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[matching]\nauto_threshold=0.5\nreview_threshold=0.7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="thresholds"):
        load_settings(bad)
    same_seed = tmp_path / "same-seed.toml"
    same_seed.write_text(
        "[pair_sample]\ndiagnostic_seed=7\nlocked_seed=7\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must differ"):
        load_settings(same_seed)


def test_run_paths_do_not_overwrite(tmp_path: Path) -> None:
    paths = create_run_paths(tmp_path, "diagnostic", "known")
    assert paths.results.is_dir()
    assert paths.working.is_dir()
    assert paths.models.is_dir()
    with pytest.raises(FileExistsError):
        create_run_paths(tmp_path, "diagnostic", "known")


def test_source_fingerprint_changes_with_settings(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "one.py").write_text("VALUE = 1\n", encoding="utf-8")
    settings = tmp_path / "settings.toml"
    settings.write_text("[x]\na=1\n", encoding="utf-8")
    first = source_fingerprint(package, settings)
    settings.write_text("[x]\na=2\n", encoding="utf-8")
    assert source_fingerprint(package, settings) != first


def test_summary_is_short_and_explicit(tmp_path: Path) -> None:
    path = tmp_path / "SUMMARY.txt"
    write_summary(
        path,
        {
            "stage": "diagnostic",
            "status": "PROVISIONAL",
            "observation_date": "2026-06-01",
            "counts": {"rows_read": 10, "model_rows": 4},
            "match": {"denominator": 8, "auto": 4, "unmatched": 4},
            "primary": {"champion": "logistic", "roc_auc": 0.71},
            "exploratory": {"roc_auc": 0.80},
            "gates": {"primary_model": "PROVISIONAL"},
            "pair_file": "match_pairs_1000.csv",
        },
    )
    text = path.read_text(encoding="utf-8")
    assert "Date Inserted measures registration delay" in text
    assert "CURRENT-SNAPSHOT FEATURES" in text
    assert "<10" in text
    assert len(text.splitlines()) < 60


def test_public_e5_redacts_small_observed_counts(tmp_path: Path) -> None:
    recorder = RunRecorder(
        stages=[
            {
                "stage": "E2_match",
                "status": "ok",
                "judgments_matched": 4,
                "elapsed_seconds": 0.2,
                "peak_memory_mb": 100.0,
            }
        ]
    )
    manifest = {
        "schema_version": 1,
        "ch_index_stats": {
            "ch_rows_read": 100,
            "companies_retained": 3,
            "analysis_fingerprint": "a" * 64,
        },
        "model_only_acceptance": {
            "status": "fail",
            "passed": False,
            "family": "prospective",
            "algorithm": "logistic",
            "reasons": ["test_class_count_below_minimum"],
            "guards": {"training_prevalence": 0.01, "auc_floor": 0.70},
            "cohort_test_counts": {"rows": 100, "positive": 1, "negative": 99},
        },
        "disclosure": {
            "status": "pass",
            "suppressed_rows": [{"file": "E4.csv", "rows": 2}],
        },
    }
    write_e5(tmp_path, recorder, manifest, min_cell_n=10)
    log = (tmp_path / "E5_run_log.csv").read_text(encoding="utf-8")
    public = json.loads((tmp_path / "E5_run_manifest.json").read_text())
    assert "<10" in log
    assert public["ch_index_stats"]["companies_retained"] == "<10"
    assert "cohort_test_counts" not in public["model_only_acceptance"]
    assert "training_prevalence" not in public["model_only_acceptance"]["guards"]
    assert public["disclosure"]["suppressed_rows"][0]["rows"] == "<10"


def test_extra_input_heading_is_not_copied_to_public_audit() -> None:
    judgments = pd.DataFrame(
        {
            "JudgmentStatus": ["Satisfied"] * 10,
            "DefendantType": ["Corporate"] * 10,
            "Jurisdiction": ["England and Wales"] * 10,
            "JudgmentDate": pd.to_datetime(["2024-01-01"] * 10),
        }
    )
    audit = type(
        "Audit",
        (),
        {
            "extra_headers": ("PRIVATE CLIENT NAME",),
            "absent_optional_columns": (),
            "invalid_amount_rows": 0,
            "missing_company_name_rows": 0,
            "missing_postcode_rows": 0,
        },
    )()
    table = build_data_audit_counts(judgments, audit)
    assert "PRIVATE CLIENT NAME" not in table["value"].astype(str).tolist()
    assert "extra_column_1" in table["value"].astype(str).tolist()


def test_population_sensitivities_expose_repeated_and_long_window_counts() -> None:
    judgments = pd.DataFrame(
        {
            "ID": ["J1", "J2", "J3"],
            "JudgmentDate": pd.to_datetime(["2024-07-01", "2025-01-01", "2022-01-01"]),
            "JudgmentStatus": ["Satisfied", "Unsatisfied", "Unsatisfied"],
            "DefendantType": ["Corporate"] * 3,
            "Jurisdiction": ["England and Wales"] * 3,
            "Amount": [500, 1_500, 3_000],
        }
    )
    matches = pd.DataFrame(
        {
            "ID": ["J1", "J2", "J3"],
            "tier": ["auto"] * 3,
            "matched_company_number": ["C1", "C1", "C2"],
        }
    )
    table = build_population_sensitivities(
        judgments, matches, "2026-06-01", Settings()
    ).set_index("stratum")
    assert table.loc["primary_12_36_auto_with_repeats", "rows"] == 2
    assert table.loc["primary_12_36_auto_unique_earliest", "rows"] == 1
    assert table.loc["aged_12_plus_auto_unique_earliest", "rows"] == 2


def test_synthetic_dates_reproduce_rt_semantics(tmp_path: Path) -> None:
    bundle = make_synthetic_bundle(120)
    target = bundle.judgments[bundle.judgments["ID"].str.startswith("J-")].copy()
    judgment_date = pd.to_datetime(target["JudgmentDate"], dayfirst=True)
    inserted = pd.to_datetime(target["Date Inserted"], dayfirst=True)
    assert ((inserted - judgment_date).dt.days == 1).all()
    age_months = (pd.Timestamp(bundle.observation_date) - judgment_date).dt.days / 30.4375
    assert age_months.between(12, 36).all()
    audited = bundle.judgments.copy()
    audited["registration_lag_days"] = 1
    audited["age_at_observation_months"] = age_months.reindex(audited.index)
    audit_table = build_data_audit_counts(audited)
    assert "status_x_type_x_jurisdiction_x_vintage" in set(audit_table["dimension"])
    judgment_path, ch_path, truth_path = write_bundle(bundle, tmp_path, excel=True)
    assert judgment_path.suffix == ".xlsx"
    assert ch_path.suffix == ".zip"
    assert truth_path.exists()


def test_default_settings_dataclass_is_self_consistent() -> None:
    settings = Settings()
    assert settings.min_calibration_each_class < settings.isotonic_each_class


def test_scale_fixture_can_request_exact_judgment_count() -> None:
    bundle = make_synthetic_bundle(500, include_prior_rows=False)
    assert len(bundle.judgments) == 500


def test_selftest_write_inputs_options_are_wired(tmp_path: Path) -> None:
    assert selftest_main(
        [
            "--write-inputs",
            str(tmp_path),
            "--n-companies",
            "100",
            "--format",
            "csv",
            "--no-prior-rows",
        ]
    ) == 0
    rows = pd.read_csv(tmp_path / "synthetic judgments.csv")
    assert len(rows) == 100
    assert (tmp_path / "BasicCompanyDataAsOneFile-2026-06-01.zip").is_file()
