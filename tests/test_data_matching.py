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
from recovery.data import read_rt_extract
from recovery.matching import (
    build_relevant_ch_index,
    match_diagnostics,
    match_judgments,
    normalize_name,
    review_sample,
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
    former_name: str = "",
    former_change: str = "",
) -> dict[str, object]:
    return {
        "CompanyName": name,
        "CompanyNumber": number,
        "RegAddress.PostCode": postcode,
        "IncorporationDate": incorporation,
        "CompanyStatus": "Active",
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


def test_rt_dates_distinguish_registration_lag_from_observation_age(tmp_path: Path) -> None:
    rows = [_rt_row("J-1", "Example Limited", "SW1A 1AA", "01/01/2024")]
    rows[0]["Date Inserted"] = "11/01/2024"
    path = tmp_path / "rt.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    frame, audit = read_rt_extract(path, "01/07/2024")

    assert frame.loc[0, "registration_lag_days"] == 10
    assert frame.loc[0, "age_at_observation_days"] == 182
    assert frame.loc[0, "age_at_observation_months"] == pytest.approx(182 / 30.44)
    assert audit.registration_lag_days_median == 10
    assert audit.observation_date == "2024-07-01"
    assert frame.loc[0, "Amount"] == pytest.approx(1250.5)


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


def test_missing_and_malformed_amounts_are_audited_not_silently_zeroed(
    tmp_path: Path,
) -> None:
    rows = [
        _rt_row("J-1", "One Limited", "AA1 1AA"),
        _rt_row("J-2", "Two Limited", "BB1 1BB"),
    ]
    rows[0]["Amount"] = ""
    rows[1]["Amount"] = "not-an-amount"
    path = tmp_path / "amounts.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    frame, audit = read_rt_extract(path, OBSERVATION_DATE)
    assert frame["Amount"].isna().all()
    assert audit.invalid_amount_rows == 1


def test_exact_ground_truth_ambiguity_and_unique_fallback(tmp_path: Path) -> None:
    judgment_rows = [
        _rt_row("J-AUTO", "Alpha Ltd.", "AA1 1AA"),
        _rt_row("J-AMBIG", "Beta Limited", "BB1 1BB"),
        _rt_row("J-FALLBACK", "Gamma Ltd", "ZZ1 1ZZ"),
        _rt_row("J-NOT-UNIQUE", "Delta Limited", "YY1 1YY"),
        _rt_row("J-TRADING", "Unrelated Style", "TT1 1TT", trading_name="Trader Ltd"),
    ]
    company_rows = [
        _ch_row("00000001", "ALPHA LIMITED", "AA1 1AA"),
        _ch_row("00000002", "BETA LTD", "BB1 1BB"),
        _ch_row("00000003", "BETA LIMITED", "BB1 1BB"),
        _ch_row("00000004", "GAMMA LIMITED", "GG1 1GG"),
        _ch_row("00000005", "DELTA LTD", "DD1 1DD"),
        _ch_row("00000006", "DELTA LIMITED", "DE1 1DE"),
        _ch_row("00000007", "TRADER LIMITED", "TT1 1TT"),
    ]
    judgments, ch_path = _write_inputs(tmp_path, judgment_rows, company_rows)
    index = build_relevant_ch_index(judgments, ch_path, chunksize=2)
    matched = match_judgments(judgments, index, Settings()).set_index("ID")

    assert index.stats["ch_rows_read"] == len(company_rows)
    assert matched.loc["J-AUTO", "matched_company_number"] == "00000001"
    assert matched.loc["J-AUTO", "tier"] == "auto"
    assert matched.loc["J-AMBIG", "matched_company_number"] == "00000002"
    assert matched.loc["J-AMBIG", "tier"] == "review"
    assert matched.loc["J-AMBIG", "margin"] == 0
    assert matched.loc["J-FALLBACK", "matched_company_number"] == "00000004"
    assert matched.loc["J-FALLBACK", "tier"] == "fallback_review"
    assert not matched.loc["J-FALLBACK", "postcode_agrees"]
    assert matched.loc["J-NOT-UNIQUE", "tier"] == "unmatched"
    assert matched.loc["J-NOT-UNIQUE", "exact_name_candidate_count"] == 2
    assert matched.loc["J-TRADING", "matched_company_number"] == "00000007"
    assert matched.loc["J-TRADING", "matched_on"] == "trading_name"


def test_date_valid_former_names_and_incorporation_guard(tmp_path: Path) -> None:
    judgment_rows = [
        _rt_row("J-FORMER", "Old Echo Limited", "EE1 1EE", "01/06/2020"),
        _rt_row("J-CURRENT", "New Echo Limited", "EE1 1EE", "01/06/2022"),
        _rt_row("J-WRONG-DATE", "Old Echo Limited", "EE1 1EE", "01/06/2022"),
        _rt_row("J-PRE-INC", "Future Limited", "FF1 1FF", "01/01/2010"),
        _rt_row("J-NO-INC", "No Date Limited", "NN1 1NN", "01/01/2024"),
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
    ]
    judgments, ch_path = _write_inputs(tmp_path, judgment_rows, company_rows, zipped=False)
    matched = match_judgments(
        judgments,
        build_relevant_ch_index(judgments, ch_path, chunksize=2),
        Settings(),
    ).set_index("ID")

    assert matched.loc["J-FORMER", "matched_company_number"] == "10000001"
    assert matched.loc["J-FORMER", "matched_name_kind"] == "former"
    assert matched.loc["J-FORMER", "tier"] == "auto"
    assert matched.loc["J-CURRENT", "matched_name_kind"] == "current"
    assert matched.loc["J-CURRENT", "tier"] == "auto"
    assert matched.loc["J-WRONG-DATE", "tier"] == "unmatched"
    assert matched.loc["J-PRE-INC", "tier"] == "unmatched"
    assert matched.loc["J-PRE-INC", "rejected_post_incorporation"] == 1
    assert matched.loc["J-NO-INC", "tier"] == "review"
    assert matched.loc["J-NO-INC", "reason"] == "incorporation_date_missing"
    assert matched.loc["J-NO-INC", "incorporation_date_missing"]


def test_diagnostics_are_aggregate_and_sample_is_deterministic_with_redistribution() -> None:
    settings = Settings()
    tiers = ["auto"] * 600 + ["review"] * 400 + ["fallback_review"] * 50
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
            "score": 1.0,
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

    first = review_sample(judgments, matches, settings)
    second = review_sample(judgments, matches, settings)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 1_000
    assert first["review_tier"].value_counts().to_dict() == {
        "auto": 600,
        "review": 350,
        "fallback_review": 50,
    }
    assert first["review_decision"].eq("").all()
    assert first["review_row_id"].is_unique
    assert first["sampling_design"].eq(
        "equal_probability_systematic_stratified_v1"
    ).all()
    assert first.groupby("review_tier")["sampling_weight"].nunique().eq(1).all()
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


def test_name_normalisation_only_strips_terminal_legal_suffixes() -> None:
    assert normalize_name("A&B, Ltd.") == "A AND B"
    assert normalize_name("Limited Edition Designs Ltd") == "LIMITED EDITION DESIGNS"
    assert normalize_name("Example Ltd (in liquidation)") == "EXAMPLE"
