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
    _fmt_count,
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


def _settings_variant(
    tmp_path: Path,
    name: str,
    old: str,
    new: str,
) -> Path:
    text = (Path(__file__).parents[1] / "settings.toml").read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / name
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


def test_repository_settings_are_valid() -> None:
    settings = load_settings(Path(__file__).parents[1] / "settings.toml")
    assert settings.sample_size == 1_000
    assert settings.diagnostic_seed == 20260618
    assert settings.model_seed == 20260619


def test_peak_memory_measurement_is_positive() -> None:
    peak = peak_memory_mb()
    assert math.isfinite(peak)
    assert peak > 0


def test_settings_reject_unknown_or_bad_values(tmp_path: Path) -> None:
    bad = _settings_variant(
        tmp_path,
        "bad.toml",
        "sample_size = 1000",
        "sample_size = 500",
    )
    with pytest.raises(ValueError, match="sample size"):
        load_settings(bad)
    same_seed = _settings_variant(
        tmp_path,
        "same-seed.toml",
        "model_seed = 20260619",
        "model_seed = 20260618",
    )
    with pytest.raises(ValueError, match="must differ"):
        load_settings(same_seed)


def test_settings_are_complete_and_bound_to_their_declared_sections(
    tmp_path: Path,
) -> None:
    missing = _settings_variant(
        tmp_path,
        "missing.toml",
        "min_cell_n = 10\n",
        "",
    )
    with pytest.raises(ValueError, match="missing required setting.*min_cell_n"):
        load_settings(missing)

    misplaced = _settings_variant(
        tmp_path,
        "misplaced.toml",
        "auc_floor = 0.70\n",
        "",
    )
    with misplaced.open("a", encoding="utf-8") as handle:
        handle.write("auc_floor = 0.70\n")
    with pytest.raises(ValueError, match=r"misplaced setting.*belongs in \[acceptance\]"):
        load_settings(misplaced)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "primary_min_months = 1",
            "primary_min_months = 2",
            "window must remain frozen",
        ),
        (
            "primary_max_months = 48",
            "primary_max_months = 60",
            "window must remain frozen",
        ),
        ("min_test_rows = 1000", "min_test_rows = true", "must be an integer"),
        (
            "min_test_companies = 500",
            "min_test_companies = 1001",
            "must not exceed min_test_rows",
        ),
        ("model_seed = 20260619", "model_seed = -1", "model_seed must be"),
        (
            "isotonic_each_class = 200",
            "isotonic_each_class = 25",
            "at least min_calibration_each_class",
        ),
        ("auc_floor = 0.70", "auc_floor = 1.01", "between 0 and 1"),
        (
            "min_calibration_slope = 0.80",
            "min_calibration_slope = 1.30",
            "calibration slopes",
        ),
        (
            "bootstrap_replicates = 1000",
            "bootstrap_replicates = 99",
            "at least 100",
        ),
        ("min_cell_n = 10", "min_cell_n = 0", "min_cell_n must be positive"),
    ],
)
def test_settings_fixed_values_types_and_ranges_fail_closed(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    path = _settings_variant(tmp_path, "invalid.toml", old, new)

    with pytest.raises(ValueError, match=message):
        load_settings(path)


def test_run_paths_do_not_overwrite(tmp_path: Path) -> None:
    paths = create_run_paths(tmp_path, "diagnostic", "known")
    assert paths.results.is_dir()
    assert paths.working.is_dir()
    assert paths.models.is_dir()
    with pytest.raises(FileExistsError):
        create_run_paths(tmp_path, "diagnostic", "known")

    with pytest.raises(ValueError, match="safe filename"):
        create_run_paths(tmp_path, "diagnostic", "../../outside")
    assert not (tmp_path.parent / "outside").exists()


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
            "scope": "matching_only",
            "stage": "diagnostic",
            "status": "PROVISIONAL",
            "observation_date": "2026-06-01",
            "companies_house_date": "2026-06-01",
            "data_construct": "status_only_unique_judgment_rows",
            "satisfaction_date_field": "absent",
            "satisfaction_date_rows": 0,
            "min_cell_n": 10,
            "date_inserted": {
                "distinct_values": 2,
                "minimum": "2026-05-31",
                "maximum": "2026-06-01",
            },
            "counts": {"rows_read": 10, "model_rows": 4},
            "match": {"denominator": 8, "exact_unique": 4, "unmatched": 4},
            "accepted_file": "linkage_validation_accepted.csv",
            "unmatched_file": "linkage_validation_unmatched.csv",
        },
    )
    text = path.read_text(encoding="utf-8")
    assert "No model was run" in text
    assert "Companies House file contains live companies only" in text
    assert "One unique exact match" in text
    assert "Date Inserted (as supplied)" in text
    assert "Satisfaction Date field        absent" in text
    assert "Satisfaction Dates filled in   0" in text
    assert "Date Inserted is reported as supplied" in text
    assert "Minimum value                2026-05-31" in text
    assert "Corporate E&W rows" in text
    assert "Full-dataset" not in text
    assert "<10" in text
    assert len(text.splitlines()) < 60


def test_count_format_hides_only_positive_small_counts() -> None:
    assert _fmt_count(0, min_cell_n=10) == "0"
    assert _fmt_count(1, min_cell_n=10) == "<10"
    assert _fmt_count(9, min_cell_n=10) == "<10"
    assert _fmt_count(10, min_cell_n=10) == "10"


def test_data_audit_reports_date_inserted_literals() -> None:
    judgments = pd.DataFrame(
        {
            "Date Inserted": pd.to_datetime(
                ["2026-05-31", "2026-06-01", "2026-06-01"]
            ),
            "JudgmentDate": pd.to_datetime(["2025-01-01"] * 3),
            "JudgmentStatus": ["Satisfied", "Unsatisfied", "Cancelled"],
            "DefendantType": ["Corporate"] * 3,
            "Jurisdiction": ["England and Wales"] * 3,
        }
    )

    table = build_data_audit_counts(judgments).set_index("dimension")

    assert table.loc["Date Inserted (literal) distinct values", "value"] == "2"
    assert table.loc["Date Inserted (literal) minimum", "value"] == "2026-05-31"
    assert table.loc["Date Inserted (literal) maximum", "value"] == "2026-06-01"


def test_public_e5_redacts_small_observed_counts(tmp_path: Path) -> None:
    recorder = RunRecorder(
        stages=[
            {
                "stage": "E2_match",
                "status": "ok",
                "judgments_matched": 4,
                "accepted_sample_rows": 3,
                "unmatched_sample_rows": 0,
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
    run_log = pd.read_csv(tmp_path / "E5_run_log.csv", dtype="string")
    assert run_log.loc[0, "accepted_sample_rows"] == "<10"
    assert run_log.loc[0, "unmatched_sample_rows"] == "0"
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


def test_population_comparison_keeps_repeats_and_defined_age_window() -> None:
    judgments = pd.DataFrame(
        {
            "ID": ["J1", "J2", "J3", "J4", "J5"],
            "JudgmentDate": pd.to_datetime(
                [
                    "2024-07-01",
                    "2025-01-01",
                    "2022-01-01",
                    "2025-02-01",
                    "2025-03-01",
                ]
            ),
            "JudgmentStatus": [
                "Satisfied",
                "Unsatisfied",
                "Unsatisfied",
                "Unsatisfied",
                "Cancelled",
            ],
            "DefendantType": ["Corporate"] * 5,
            "Jurisdiction": ["England and Wales"] * 5,
            "Amount": [500, 1_500, 3_000, 10_000, 400],
        }
    )
    matches = pd.DataFrame(
        {
            "ID": ["J1", "J2", "J3", "J4", "J5"],
            "tier": [
                "exact_unique",
                "exact_unique",
                "exact_unique",
                "unmatched",
                "unmatched",
            ],
            "matched_company_number": ["C1", "C1", "C2", "", ""],
        }
    )
    raw = build_population_sensitivities(
        judgments, matches, "2026-06-01", Settings()
    )
    table = raw.set_index("stratum")
    assert table.loc["all_corporate_england_wales_register_stock", "rows"] == 5
    assert (
        table.loc[
            "all_corporate_england_wales_register_stock",
            "binary_status_denominator",
        ]
        == 4
    )
    assert table.loc["post_one_to_48_month_binary_status", "rows"] == 3
    assert table.loc["included_unique_exact_live_company", "rows"] == 2
    assert (
        table.loc[
            "included_unique_exact_live_company",
            "distinct_linked_entities",
        ]
        == 1
    )
    assert table.loc["excluded_not_unique_exact_linked_to_live_bulk", "rows"] == 1
    groups = {
        "all_primary_age_eligible",
        "included_unique_exact_live_company",
        "excluded_not_unique_exact_linked_to_live_bulk",
    }
    for analysis in (
        "selection_by_judgment_year",
        "selection_by_amount_band",
        "selection_by_age_band",
    ):
        observed = set(
            raw.loc[raw["analysis"].eq(analysis), "stratum"].str.split("|").str[0]
        )
        assert observed == groups
    amount = raw.loc[raw["analysis"].eq("selection_by_amount_band")].set_index(
        "stratum"
    )
    assert amount.loc[
        "included_unique_exact_live_company|1000_4999", "rows"
    ] == 1
    assert amount.loc[
        "excluded_not_unique_exact_linked_to_live_bulk|5000_24999", "rows"
    ] == 1
    assert amount.loc["all_primary_age_eligible|1000_4999", "rows"] == 1


def test_synthetic_dates_reproduce_rt_semantics(tmp_path: Path) -> None:
    bundle = make_synthetic_bundle(120)
    target = bundle.judgments[bundle.judgments["ID"].str.startswith("J-")].copy()
    judgment_date = pd.to_datetime(target["JudgmentDate"], dayfirst=True)
    inserted = pd.to_datetime(target["Date Inserted"], dayfirst=True)
    assert ((inserted - judgment_date).dt.days == 1).all()
    age_months = (pd.Timestamp(bundle.observation_date) - judgment_date).dt.days / 30.4375
    assert age_months.between(12, 36).all()
    audited = bundle.judgments.copy()
    audited["date_inserted_minus_judgment_days"] = 1
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
