"""Test offline RT input handling and conservative Companies House matching.

Inputs are synthetic rows and temporary CSV/XLSX/ZIP files. Outputs are test
assertions only; no confidential data, network calls or persistent files occur.
"""

from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd
import pytest

from recovery.config import Settings
from recovery.data import iter_ch_chunks, read_rt_extract
from recovery.matching import (
    accepted_validation_sample,
    build_relevant_ch_index,
    match_diagnostics,
    match_judgments,
    normalize_name,
)


OBSERVATION_DATE = "2026-06-01"


def _rt_row(
    identifier: str,
    name: str,
    postcode: str,
    judgment_date: str = "01/01/2024",
    *,
    trading_name: str = "",
) -> dict[str, object]:
    judgment = pd.to_datetime(judgment_date, dayfirst=True)
    return {
        "ID": identifier,
        "Date Inserted": (judgment + pd.Timedelta(days=2)).strftime("%d/%m/%Y"),
        "JudgmentDate": judgment_date,
        "JudgmentStatus": "Unsatisfied",
        "DefendantType": "Corporate",
        "Jurisdiction": "England and Wales",
        "Defendant Company Name": name,
        "Defendant_Postcode": postcode,
        "Amount": "£1,250.50",
        "Defendant Trading Name": trading_name,
        "Defendant Address": "",
    }


def _ch_row(
    number: str,
    name: str,
    postcode: str,
    incorporation: str = "01/01/2010",
    *,
    status: str = "Active",
    former_name: str = "",
    former_change: str = "",
) -> dict[str, object]:
    return {
        "CompanyName": name,
        "CompanyNumber": number,
        "RegAddress.PostCode": postcode,
        "IncorporationDate": incorporation,
        "CompanyStatus": status,
        "CompanyCategory": "Private Limited Company",
        "Accounts.NextDueDate": "01/09/2026",
        "Mortgages.NumMortCharges": "1",
        "Mortgages.NumMortSatisfied": "0",
        "PreviousName_1.CompanyName": former_name,
        "PreviousName_1.CONDATE": former_change,
    }


def _write_inputs(
    directory: Path,
    judgment_rows: list[dict[str, object]],
    company_rows: list[dict[str, object]],
    *,
    zipped: bool = True,
) -> tuple[pd.DataFrame, Path]:
    rt_path = directory / "judgments.csv"
    pd.DataFrame(judgment_rows).to_csv(rt_path, index=False)
    judgments, _ = read_rt_extract(rt_path, OBSERVATION_DATE)
    ch_csv = directory / "BasicCompanyDataAsOneFile.csv"
    pd.DataFrame(company_rows).to_csv(ch_csv, index=False)
    if not zipped:
        return judgments, ch_csv
    ch_zip = directory / "BasicCompanyDataAsOneFile.zip"
    with zipfile.ZipFile(ch_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(ch_csv, arcname=ch_csv.name)
    return judgments, ch_zip


def test_rt_dates_keep_inserted_difference_distinct_from_observation_age(
    tmp_path: Path,
) -> None:
    rows = [_rt_row("J-1", "Example Limited", "SW1A 1AA", "01/01/2024")]
    rows[0]["Date Inserted"] = "11/01/2024"
    path = tmp_path / "rt.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    frame, audit = read_rt_extract(path, "2024-07-01")

    assert frame.loc[0, "date_inserted_minus_judgment_days"] == 10
    assert frame.loc[0, "age_at_observation_days"] == 182
    assert frame.loc[0, "age_at_observation_months"] == pytest.approx(182 / 30.44)
    assert audit.date_inserted_minus_judgment_days_median == 10
    assert audit.observation_date == "2024-07-01"
    assert frame.loc[0, "Amount"] == pytest.approx(1250.5)


def test_real_excel_dates_are_not_swapped_when_day_and_month_are_ambiguous(
    tmp_path: Path,
) -> None:
    pytest.importorskip("openpyxl")
    row = _rt_row("J-1", "Example Limited", "SW1A 1AA")
    row["JudgmentDate"] = pd.Timestamp("2025-05-01")
    row["Date Inserted"] = pd.Timestamp("2025-05-02")
    path = tmp_path / "real-excel-dates.xlsx"
    pd.DataFrame([row]).to_excel(path, index=False)

    frame, audit = read_rt_extract(path, "2025-07-03")

    assert frame.loc[0, "JudgmentDate"] == pd.Timestamp("2025-05-01")
    assert frame.loc[0, "Date Inserted"] == pd.Timestamp("2025-05-02")
    assert frame.loc[0, "date_inserted_minus_judgment_days"] == 1
    assert audit.date_inserted_before_judgment_rows == 0


def test_date_inserted_anomalies_are_audited_without_assuming_field_meaning(
    tmp_path: Path,
) -> None:
    row = _rt_row("J-1", "Example Limited", "SW1A 1AA", "02/05/2025")
    row["Date Inserted"] = "01/05/2025"
    path = tmp_path / "registration-anomaly.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    frame, audit = read_rt_extract(path, "2025-07-03")

    assert frame.loc[0, "date_inserted_minus_judgment_days"] == -1
    assert audit.date_inserted_before_judgment_rows == 1


def test_judgment_after_extract_date_still_stops_the_run(tmp_path: Path) -> None:
    row = _rt_row("J-1", "Example Limited", "SW1A 1AA", "04/07/2025")
    row["Date Inserted"] = "05/07/2025"
    path = tmp_path / "future-judgment.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="JudgmentDate is after the RT extract date"):
        read_rt_extract(path, "2025-07-03")


def test_rt_csv_and_xlsx_validation_reject_duplicate_ids(tmp_path: Path) -> None:
    rows = [
        _rt_row("DUPLICATE", "One Limited", "AA1 1AA"),
        _rt_row("DUPLICATE", "Two Limited", "BB1 1BB"),
    ]
    csv_path = tmp_path / "duplicate.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="ID must be unique"):
        read_rt_extract(csv_path, OBSERVATION_DATE)

    pytest.importorskip("openpyxl")
    xlsx_path = tmp_path / "duplicate.xlsx"
    pd.DataFrame(rows).to_excel(xlsx_path, index=False)
    with pytest.raises(ValueError, match="ID must be unique"):
        read_rt_extract(xlsx_path, OBSERVATION_DATE)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("ID", "", "ID has"),
        ("JudgmentDate", "not-a-date", "JudgmentDate"),
        ("JudgmentStatus", "Pending", "JudgmentStatus"),
        ("DefendantType", "Mystery", "DefendantType"),
        ("Jurisdiction", "Unknown", "Jurisdiction"),
    ],
)
def test_rt_validation_rejects_missing_or_unexpected_values(
    tmp_path: Path, column: str, value: str, message: str
) -> None:
    row = _rt_row("J-1", "Example Limited", "AA1 1AA")
    row[column] = value
    path = tmp_path / f"bad-{column}.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        read_rt_extract(path, OBSERVATION_DATE)


def test_missing_or_invalid_amounts_are_audited_not_silently_zeroed(
    tmp_path: Path,
) -> None:
    rows = [
        _rt_row("J-1", "One Limited", "AA1 1AA"),
        _rt_row("J-2", "Two Limited", "BB1 1BB"),
        _rt_row("J-3", "Three Limited", "CC1 1CC"),
        _rt_row("J-4", "Four Limited", "DD1 1DD"),
    ]
    rows[0]["Amount"] = ""
    rows[1]["Amount"] = "not-an-amount"
    rows[2]["Amount"] = "inf"
    rows[3]["Amount"] = "-1"
    path = tmp_path / "amounts.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    frame, audit = read_rt_extract(path, OBSERVATION_DATE)
    assert frame["Amount"].isna().all()
    assert audit.invalid_amount_rows == 3


def test_companies_house_status_header_is_required(tmp_path: Path) -> None:
    judgments = pd.DataFrame(
        [_rt_row("J-1", "Example Limited", "AA1 1AA")]
    )
    company = _ch_row("00000001", "EXAMPLE LIMITED", "AA1 1AA")
    del company["CompanyStatus"]
    path = tmp_path / "missing-status.csv"
    pd.DataFrame([company]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required.*CompanyStatus"):
        build_relevant_ch_index(judgments, path)


def test_companies_house_zip_with_multiple_csvs_stops_for_review(
    tmp_path: Path,
) -> None:
    path = tmp_path / "multiple.csv.files.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("first.csv", "CompanyName,CompanyNumber\nA,1\n")
        archive.writestr("second.csv", "CompanyName,CompanyNumber\nB,2\n")
    with pytest.raises(ValueError, match="exactly one CSV"):
        list(iter_ch_chunks(path))


@pytest.mark.parametrize(
    "status",
    [
        "Active",
        "Active - Proposal to Strike off",
        "Administration",
        "ADMINISTRATION ORDER",
        "ADMINISTRATIVE RECEIVER",
        "In Administration",
        "In Administration/Administrative Receiver",
        "In Administration/Receiver Manager",
        "Liquidation",
        "Live but Receiver Manager on at least one charge",
        "RECEIVER MANAGER / ADMINISTRATIVE RECEIVER",
        "RECEIVERSHIP",
        "Registered",
        "Voluntary Arrangement",
        "VOLUNTARY ARRANGEMENT / ADMINISTRATIVE RECEIVER",
        "VOLUNTARY ARRANGEMENT / RECEIVER MANAGER",
    ],
)
def test_recognised_live_company_statuses_are_not_reduced_to_active(
    tmp_path: Path,
    status: str,
) -> None:
    judgments = pd.DataFrame(
        [_rt_row("J-1", "Example Limited", "AA1 1AA")]
    )
    path = tmp_path / "live-status.csv"
    pd.DataFrame(
        [_ch_row("00000001", "EXAMPLE LIMITED", "AA1 1AA", status=status)]
    ).to_csv(path, index=False)

    index = build_relevant_ch_index(judgments, path)

    assert index.companies["00000001"].current_snapshot["CompanyStatus"] == status


@pytest.mark.parametrize("status", ["", "Dissolved", "not a real status"])
def test_blank_non_live_or_unknown_company_status_fails_closed(
    tmp_path: Path,
    status: str,
) -> None:
    judgments = pd.DataFrame(
        [_rt_row("J-1", "Example Limited", "AA1 1AA")]
    )
    path = tmp_path / "bad-status.csv"
    pd.DataFrame(
        [_ch_row("00000001", "EXAMPLE LIMITED", "AA1 1AA", status=status)]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="CompanyStatus.*non-live"):
        build_relevant_ch_index(judgments, path)


@pytest.mark.parametrize("zipped", [False, True])
def test_duplicate_raw_companies_house_headers_fail_before_pandas_mangling(
    tmp_path: Path,
    zipped: bool,
) -> None:
    judgments = pd.DataFrame(
        [_rt_row("J-1", "Example Limited", "AA1 1AA")]
    )
    company = _ch_row("00000001", "EXAMPLE LIMITED", "AA1 1AA")
    frame = pd.DataFrame(
        [[*company.values(), "shadow number"]],
        columns=[*company.keys(), "CompanyNumber"],
    )
    csv_path = tmp_path / "duplicate-company-header.csv"
    frame.to_csv(csv_path, index=False)
    path = csv_path
    if zipped:
        path = tmp_path / "duplicate-company-header.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(csv_path, arcname=csv_path.name)

    with pytest.raises(ValueError, match="duplicate raw header.*CompanyNumber"):
        build_relevant_ch_index(judgments, path)


def test_only_unique_exact_names_match_and_postcode_never_selects(tmp_path: Path) -> None:
    judgment_rows = [
        _rt_row("J-EXACT", "Alpha Ltd.", "AA1 1AA"),
        _rt_row("J-POSTCODE-TRAP", "Omega Trading Ltd", "BB1 1BB"),
        _rt_row("J-DIFFERENT-POSTCODE", "Gamma Ltd", "ZZ1 1ZZ"),
        _rt_row("J-NOT-UNIQUE", "Delta Limited", "YY1 1YY"),
        _rt_row("J-TRADING", "Unrelated Style", "TT1 1TT", trading_name="Trader Ltd"),
        _rt_row("J-MISSING-POSTCODE", "Epsilon Limited", ""),
        _rt_row("J-CONFLICTING-SOURCES", "Alpha Ltd", "AA1 1AA", trading_name="Gamma Ltd"),
        _rt_row("J-MISSING-INC-COMPETITOR", "Blocker Limited", "BL1 1BL"),
        _rt_row("J-INCOMPLETE-COMPETITOR", "History Blocker Limited", "HB1 1HB"),
    ]
    company_rows = [
        _ch_row("00000001", "ALPHA LIMITED", "AA1 1AA"),
        _ch_row("00000002", "OMEGA TRADING SERVICES LIMITED", "BB1 1BB"),
        _ch_row("00000004", "GAMMA LIMITED", "GG1 1GG"),
        _ch_row("00000005", "DELTA LTD", "YY1 1YY"),
        _ch_row("00000006", "DELTA LIMITED", "DE1 1DE"),
        _ch_row("00000007", "TRADER LIMITED", "TT1 1TT"),
        _ch_row("00000008", "EPSILON LIMITED", "EE1 1EE"),
        _ch_row("00000009", "BLOCKER LIMITED", "BL1 1BL"),
        _ch_row("00000010", "BLOCKER LTD", "BX1 1BX", ""),
        _ch_row("00000011", "HISTORY BLOCKER LIMITED", "HB1 1HB"),
        _ch_row(
            "00000012",
            "HISTORY BLOCKER LTD",
            "HX1 1HX",
            former_name="OLDER HISTORY BLOCKER LIMITED",
            former_change="",
        ),
    ]
    judgments, ch_path = _write_inputs(tmp_path, judgment_rows, company_rows)
    index = build_relevant_ch_index(judgments, ch_path, chunksize=2)
    matched = match_judgments(judgments, index).set_index("ID")

    assert index.stats["ch_rows_read"] == len(company_rows)
    assert matched.loc["J-EXACT", "matched_company_number"] == "00000001"
    assert matched.loc["J-EXACT", "tier"] == "exact_unique"
    assert matched.loc["J-POSTCODE-TRAP", "tier"] == "unmatched"
    assert matched.loc["J-DIFFERENT-POSTCODE", "matched_company_number"] == "00000004"
    assert matched.loc["J-DIFFERENT-POSTCODE", "tier"] == "exact_unique"
    assert not matched.loc["J-DIFFERENT-POSTCODE", "postcode_agrees"]
    assert matched.loc["J-NOT-UNIQUE", "tier"] == "unmatched"
    assert matched.loc["J-NOT-UNIQUE", "exact_name_candidate_count"] == 2
    assert matched.loc["J-TRADING", "matched_company_number"] == "00000007"
    assert matched.loc["J-TRADING", "matched_on"] == "trading_name"
    assert matched.loc["J-MISSING-POSTCODE", "tier"] == "exact_unique"
    assert matched.loc["J-MISSING-POSTCODE", "reason"] == "unique_exact_name_postcode_missing"
    assert matched.loc["J-CONFLICTING-SOURCES", "tier"] == "unmatched"
    assert matched.loc["J-CONFLICTING-SOURCES", "exact_name_candidate_count"] == 2
    assert matched.loc["J-MISSING-INC-COMPETITOR", "tier"] == "unmatched"
    assert matched.loc["J-MISSING-INC-COMPETITOR", "incorporation_date_missing"]
    assert matched.loc["J-INCOMPLETE-COMPETITOR", "tier"] == "unmatched"
    assert matched.loc["J-INCOMPLETE-COMPETITOR", "name_history_incomplete"]


def test_companies_house_iso_dates_are_not_read_day_first(tmp_path: Path) -> None:
    judgments = pd.DataFrame(
        [_rt_row("J-ISO", "ISO Example Limited", "IS1 1IS", "03/01/2024")]
    )
    companies = pd.DataFrame(
        [_ch_row("00000100", "ISO EXAMPLE LIMITED", "IS1 1IS", "2024-01-02")]
    )
    ch_path = tmp_path / "companies.csv"
    companies.to_csv(ch_path, index=False)

    index = build_relevant_ch_index(judgments, ch_path)
    matched = match_judgments(judgments, index).set_index("ID")

    assert index.companies["00000100"].incorporation_date == pd.Timestamp("2024-01-02")
    assert matched.loc["J-ISO", "tier"] == "exact_unique"


def test_date_valid_former_names_and_incorporation_guard(tmp_path: Path) -> None:
    judgment_rows = [
        _rt_row("J-FORMER", "Old Echo Limited", "EE1 1EE", "01/06/2020"),
        _rt_row("J-CURRENT", "New Echo Limited", "EE1 1EE", "01/06/2022"),
        _rt_row("J-CURRENT-EARLY", "New Echo Limited", "EE1 1EE", "01/06/2020"),
        _rt_row("J-WRONG-DATE", "Old Echo Limited", "EE1 1EE", "01/06/2022"),
        _rt_row("J-PRE-INC", "Future Limited", "FF1 1FF", "01/01/2010"),
        _rt_row("J-NO-INC", "No Date Limited", "NN1 1NN", "01/01/2024"),
        _rt_row("J-INCOMPLETE-HISTORY", "Incomplete New Limited", "II1 1II", "01/01/2024"),
    ]
    company_rows = [
        _ch_row(
            "10000001",
            "NEW ECHO LIMITED",
            "EE1 1EE",
            "01/01/2010",
            former_name="OLD ECHO LIMITED",
            former_change="01/01/2021",
        ),
        _ch_row("10000002", "FUTURE LIMITED", "FF1 1FF", "01/01/2011"),
        _ch_row("10000003", "NO DATE LIMITED", "NN1 1NN", ""),
        _ch_row(
            "10000004",
            "INCOMPLETE NEW LIMITED",
            "II1 1II",
            "01/01/2010",
            former_name="INCOMPLETE OLD LIMITED",
            former_change="",
        ),
    ]
    judgments, ch_path = _write_inputs(tmp_path, judgment_rows, company_rows, zipped=False)
    matched = match_judgments(
        judgments,
        build_relevant_ch_index(judgments, ch_path, chunksize=2),
    ).set_index("ID")

    assert matched.loc["J-FORMER", "matched_company_number"] == "10000001"
    assert matched.loc["J-FORMER", "matched_name_kind"] == "former"
    assert matched.loc["J-FORMER", "tier"] == "exact_unique"
    assert matched.loc["J-CURRENT", "matched_name_kind"] == "current"
    assert matched.loc["J-CURRENT", "tier"] == "exact_unique"
    assert matched.loc["J-CURRENT-EARLY", "tier"] == "unmatched"
    assert matched.loc["J-WRONG-DATE", "tier"] == "unmatched"
    assert matched.loc["J-PRE-INC", "tier"] == "unmatched"
    assert matched.loc["J-PRE-INC", "rejected_post_incorporation"] == 1
    assert matched.loc["J-NO-INC", "tier"] == "unmatched"
    assert matched.loc["J-NO-INC", "reason"] == "exact_name_missing_incorporation_date"
    assert matched.loc["J-NO-INC", "incorporation_date_missing"]
    assert matched.loc["J-INCOMPLETE-HISTORY", "tier"] == "unmatched"


def test_diagnostics_are_aggregate_and_sample_is_deterministic() -> None:
    settings = Settings()
    tiers = ["exact_unique"] * 1_050
    ids = [f"J-{position:04d}" for position in range(len(tiers))]
    judgments = pd.DataFrame(
        {
            "ID": ids,
            "JudgmentDate": pd.Timestamp("2024-01-01"),
            "DefendantType": "Corporate",
            "Jurisdiction": "England and Wales",
            "Defendant Company Name": [f"SOURCE {position} LIMITED" for position in range(len(tiers))],
            "Defendant Trading Name": "",
            "Defendant_Postcode": "AA1 1AA",
        }
    )
    matches = pd.DataFrame(
        {
            "ID": ids,
            "tier": tiers,
            "reason": "fixture",
            "matched_company_number": [f"{position:08d}" for position in range(len(tiers))],
            "matched_company_name": [f"MATCH {position} LIMITED" for position in range(len(tiers))],
            "matched_name": [f"MATCH {position} LIMITED" for position in range(len(tiers))],
            "matched_name_kind": "current",
            "matched_on": "company_name",
            "postcode_agrees": True,
            "source_company_name": [
                f"CANONICAL SOURCE {position} LIMITED"
                for position in range(len(tiers))
            ],
            "source_trading_name": [
                f"CANONICAL TRADE {position}" for position in range(len(tiers))
            ],
            "source_postcode": "CC1 1CC",
        }
    )

    first = accepted_validation_sample(
        judgments, matches, settings, seed=settings.diagnostic_seed
    )
    second = accepted_validation_sample(
        judgments, matches, settings, seed=settings.diagnostic_seed
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 1_000
    assert first["tier"].value_counts().to_dict() == {"exact_unique": 1_000}
    expected_sources = matches.set_index("ID")[[
        "source_company_name",
        "source_trading_name",
        "source_postcode",
    ]]
    actual_sources = first.set_index("ID")[expected_sources.columns]
    pd.testing.assert_frame_equal(actual_sources, expected_sources.loc[actual_sources.index])

    diagnostics = match_diagnostics(judgments, matches)
    assert set(diagnostics) >= {"tier_counts", "unmatched_reasons", "method_counts"}
    assert all("ID" not in table.columns for table in diagnostics.values())


def test_name_normalisation_handles_suffixes_and_formatting() -> None:
    assert normalize_name("A&B, Ltd.") == "A AND B"
    assert normalize_name("Limited Edition Designs Ltd") == "LIMITED EDITION DESIGNS"
    assert normalize_name("Example Ltd (in liquidation)") == "EXAMPLE"
