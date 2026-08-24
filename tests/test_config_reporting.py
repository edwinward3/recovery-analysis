"""Test settings, output files and synthetic inputs.

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
    build_output_dictionary,
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
    assert settings.min_cell_n == 10


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
    with pytest.raises(ValueError, match="sample_size must remain 1000"):
        load_settings(bad)


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

    unknown = _settings_variant(
        tmp_path,
        "unknown.toml",
        "sample_size = 1000",
        "sample_size = 1000\nextra = 1",
    )
    with pytest.raises(ValueError, match="unknown setting"):
        load_settings(unknown)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("sample_size = 1000", "sample_size = true", "must be an integer"),
        ("diagnostic_seed = 20260618", "diagnostic_seed = 1", "must remain"),
        ("min_cell_n = 10", "min_cell_n = 0", "must remain"),
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
    paths = create_run_paths(tmp_path, "known")
    assert paths.results.is_dir()
    assert paths.working.is_dir()
    with pytest.raises(FileExistsError):
        create_run_paths(tmp_path, "known")

    with pytest.raises(ValueError, match="safe filename"):
        create_run_paths(tmp_path, "../../outside")
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
            "observation_date": "2026-06-01",
            "companies_house_date": "2026-06-01",
            "data_construct": "status_only_unique_judgment_rows",
            "optional_fields": {
                "Satisfaction Date": {"present": False, "rows": 0},
                "Cancellation Date": {"present": False, "rows": 0},
                "Cancellation Reason": {"present": False, "rows": 0},
                "Status Effective Date": {"present": False, "rows": 0},
                "Snapshot Date": {"present": False, "rows": 0},
            },
            "min_cell_n": 10,
            "date_inserted": {
                "distinct_values": 2,
                "minimum": "2026-05-31",
                "maximum": "2026-06-01",
            },
            "counts": {"rows_read": 10},
            "match": {"denominator": 8, "exact_unique": 4, "unmatched": 4},
            "accepted_file": "linkage_validation_accepted.csv",
            "unmatched_file": "linkage_validation_unmatched.csv",
        },
    )
    text = path.read_text(encoding="utf-8")
    assert "ANALYSIS COMPLETE" in text
    assert "Made two private review samples. They remain with RT" in text
    assert "Companies House file contains live companies only" in text
    assert "One exact live-company match" in text
    assert "Date Inserted (RT registration date)" in text
    assert "Satisfaction Date" in text and "absent; 0 filled" in text
    assert "Stock or historical extract" in text
    assert "Date Inserted is the RT registration date" in text
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


def test_summary_hides_match_complements_and_rate(tmp_path: Path) -> None:
    path = tmp_path / "SUMMARY.txt"
    write_summary(
        path,
        {
            "min_cell_n": 10,
            "counts": {},
            "match": {
                "denominator": 100,
                "exact_unique": 95,
                "unmatched": 5,
                "coverage": 0.95,
            },
        },
    )

    text = path.read_text(encoding="utf-8")
    assert "One exact live-company match suppressed" in text
    assert "No match                     suppressed" in text
    assert "Match rate                   suppressed" in text
    assert "95.0%" not in text


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


def test_output_dictionary_is_built_from_written_columns(tmp_path: Path) -> None:
    pd.DataFrame(columns=["non_events", "satisfaction_cif"]).to_csv(
        tmp_path / "E4_results.csv", index=False
    )

    dictionary = build_output_dictionary(tmp_path).set_index("column_name")

    assert set(dictionary.index) == {"non_events", "satisfaction_cif"}
    assert "cancellations stay separate" in dictionary.loc["non_events", "definition"]
    assert "recorded satisfaction" in dictionary.loc["satisfaction_cif", "definition"]


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
    assert settings.sample_size == 1_000
    assert settings.min_cell_n == 10


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
