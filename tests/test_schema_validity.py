"""Synthetic tests for the fail-closed RT schema and provenance audit.

No private or locked data are read by this module.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from recovery.data import read_rt_extract


OBSERVED = "2024-12-31"


def _row(
    identifier: str = "J-1",
    *,
    status: str = "Unsatisfied",
) -> dict[str, object]:
    return {
        "ID": identifier,
        "Date Inserted": "03/01/2024",
        "JudgmentDate": "01/01/2024",
        "JudgmentStatus": status,
        "DefendantType": "Corporate",
        "Jurisdiction": "England and Wales",
        "Defendant Company Name": "Example Limited",
        "Defendant_Postcode": "AA1 1AA",
        "Amount": "1250",
        "Defendant Trading Name": "",
        "Defendant Address": "1 Example Street",
    }


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_sample_schema_is_explicitly_cross_sectional_and_needs_no_event_dates(
    tmp_path: Path,
) -> None:
    # A Satisfied row is valid without a satisfaction date only when the source
    # schema contains no such field, as in the supplied eleven-column sample.
    path = _write(tmp_path / "rt.csv", [_row(status="Satisfied")])

    frame, audit = read_rt_extract(path, OBSERVED)

    assert audit.data_construct == "status_only_unique_judgment_rows"
    assert audit.event_date_columns_present == ()
    assert not audit.historical_snapshots_available
    assert frame.attrs["data_construct"] == "status_only_unique_judgment_rows"
    assert frame["Satisfaction Date"].isna().all()
    assert frame["Cancellation Date"].isna().all()
    assert frame["Status Effective Date"].isna().all()
    assert frame["Snapshot Date"].isna().all()
    assert frame["Cancellation Reason"].eq("").all()
    assert {
        "Satisfaction Date",
        "Cancellation Date",
        "Cancellation Reason",
        "Status Effective Date",
        "Snapshot Date",
    }.issubset(audit.absent_optional_columns)


def test_event_and_snapshot_aliases_are_preserved_parsed_and_audited(
    tmp_path: Path,
) -> None:
    satisfied = _row("J-S", status="Satisfied")
    satisfied.update(
        {
            "Date Satisfied": "01/03/2024",
            "Date Cancelled": "",
            "Reason for Cancellation": "",
            "Status Date": "02/03/2024",
            "Extract Date": OBSERVED,
        }
    )
    cancelled = _row("J-C", status="Cancelled")
    cancelled.update(
        {
            "Date Satisfied": "",
            "Date Cancelled": "01/04/2024",
            "Reason for Cancellation": "set aside",
            "Status Date": "01/04/2024",
            "Extract Date": OBSERVED,
        }
    )
    unsatisfied = _row("J-U", status="Unsatisfied")
    unsatisfied.update(
        {
            "Date Satisfied": "",
            "Date Cancelled": "",
            "Reason for Cancellation": "",
            "Status Date": "03/01/2024",
            "Extract Date": OBSERVED,
        }
    )
    path = _write(tmp_path / "dated.csv", [satisfied, cancelled, unsatisfied])

    frame, audit = read_rt_extract(path, OBSERVED)

    assert audit.data_construct == "status_with_event_dates_and_snapshot_date"
    assert audit.event_date_columns_present == (
        "Satisfaction Date",
        "Cancellation Date",
        "Status Effective Date",
    )
    assert audit.satisfaction_date_present_rows == 1
    assert audit.cancellation_date_present_rows == 1
    assert audit.cancellation_reason_present_rows == 1
    assert audit.status_effective_date_present_rows == 3
    assert audit.snapshot_date_present_rows == 3
    assert frame.loc[0, "Satisfaction Date"] == pd.Timestamp("2024-03-01")
    assert frame.loc[1, "Cancellation Date"] == pd.Timestamp("2024-04-01")
    assert frame.loc[1, "Cancellation Reason"] == "set aside"
    assert "Date Satisfied" in audit.raw_headers
    assert ("Date Satisfied", "Satisfaction Date") in audit.raw_header_schema
    assert ("Extract Date", "Snapshot Date") in audit.raw_header_schema


def test_exact_extra_headers_stay_internal_and_affect_schema_provenance(
    tmp_path: Path,
) -> None:
    first = _row()
    first["Source Batch"] = "A"
    first_path = _write(tmp_path / "first.csv", [first])
    first_frame, first_audit = read_rt_extract(first_path, OBSERVED)

    second = _row()
    second["Source Batch Code"] = "A"
    second_path = _write(tmp_path / "second.csv", [second])
    second_frame, second_audit = read_rt_extract(second_path, OBSERVED)

    assert first_audit.extra_headers == ("Source Batch",)
    assert ("Source Batch", "<unrecognised>") in first_audit.raw_header_schema
    assert "Source Batch" not in first_frame.columns
    assert "Source Batch Code" not in second_frame.columns
    assert first_audit.analysis_fingerprint == second_audit.analysis_fingerprint
    assert first_audit.raw_header_schema_sha256 != second_audit.raw_header_schema_sha256
    assert first_audit.provenance_fingerprint != second_audit.provenance_fingerprint


def test_raw_file_hash_and_all_retained_columns_are_fingerprinted(tmp_path: Path) -> None:
    first = _row(status="Cancelled")
    first.update(
        {
            "Cancellation Date": "01/04/2024",
            "Cancellation Reason": "set aside",
        }
    )
    first_path = _write(tmp_path / "first.csv", [first])
    _, first_audit = read_rt_extract(first_path, OBSERVED)

    second = dict(first)
    second["Cancellation Reason"] = "entered in error"
    second_path = _write(tmp_path / "second.csv", [second])
    _, second_audit = read_rt_extract(second_path, OBSERVED)

    assert first_audit.raw_source_sha256 == hashlib.sha256(
        first_path.read_bytes()
    ).hexdigest()
    assert len(first_audit.raw_header_schema_sha256) == 64
    assert len(first_audit.analysis_fingerprint) == 64
    assert len(first_audit.provenance_fingerprint) == 64
    assert first_audit.raw_header_schema_sha256 == second_audit.raw_header_schema_sha256
    assert first_audit.analysis_fingerprint != second_audit.analysis_fingerprint
    assert first_audit.provenance_fingerprint != second_audit.provenance_fingerprint


@pytest.mark.parametrize(
    "header",
    ["Payment Confirmed At", "Recovery Outcome", "Recovery Date"],
)
def test_unrecognised_outcome_or_history_header_fails_closed(
    tmp_path: Path,
    header: str,
) -> None:
    row = _row()
    row[header] = "01/03/2024"
    path = _write(tmp_path / "unknown-outcome.csv", [row])

    with pytest.raises(ValueError, match="unrecognised outcome/history-related"):
        read_rt_extract(path, OBSERVED)


@pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
def test_duplicate_raw_rt_headers_fail_before_pandas_can_mangle_them(
    tmp_path: Path,
    suffix: str,
) -> None:
    if suffix == ".xlsx":
        pytest.importorskip("openpyxl")
    row = _row()
    frame = pd.DataFrame(
        [[*row.values(), "shadow identifier"]],
        columns=[*row.keys(), "ID"],
    )
    path = tmp_path / f"duplicate-header{suffix}"
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_excel(path, index=False)

    with pytest.raises(ValueError, match="duplicate raw header.*ID"):
        read_rt_extract(path, OBSERVED)


def test_rt_workbook_with_additional_sheet_stops_for_schema_review(
    tmp_path: Path,
) -> None:
    pytest.importorskip("openpyxl")
    path = tmp_path / "multiple-sheets.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([_row()]).to_excel(writer, index=False, sheet_name="Judgments")
        pd.DataFrame({"Status History": ["hidden extra structure"]}).to_excel(
            writer, index=False, sheet_name="History"
        )

    with pytest.raises(ValueError, match="exactly one worksheet"):
        read_rt_extract(path, OBSERVED)


@pytest.mark.parametrize("observation_date", ["31/12/2024", " 2024-12-31"])
def test_external_observation_date_requires_strict_iso_format(
    tmp_path: Path,
    observation_date: str,
) -> None:
    path = _write(tmp_path / "rt.csv", [_row()])

    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        read_rt_extract(path, observation_date)


def test_cancellation_reason_does_not_imply_event_dates_or_history(tmp_path: Path) -> None:
    row = _row(status="Cancelled")
    row["Cancellation Reason"] = "entered in error"
    path = _write(tmp_path / "cancellation-history.csv", [row])

    frame, audit = read_rt_extract(path, OBSERVED)

    assert audit.data_construct == (
        "status_with_cancellation_reason_unique_judgment_rows"
    )
    assert frame.attrs["data_construct"] == (
        "status_with_cancellation_reason_unique_judgment_rows"
    )


@pytest.mark.parametrize(
    ("status", "extra", "message"),
    [
        (
            "Satisfied",
            {"Satisfaction Date": "31/12/2023"},
            "Satisfaction Date is before JudgmentDate",
        ),
        (
            "Satisfied",
            {"Satisfaction Date": "01/01/2025"},
            "Satisfaction Date is after the RT extract date",
        ),
        (
            "Satisfied",
            {"Satisfaction Date": ""},
            "missing for 1 Satisfied",
        ),
        (
            "Unsatisfied",
            {"Satisfaction Date": "01/03/2024"},
            "populated for 1 Unsatisfied",
        ),
        (
            "Cancelled",
            {"Cancellation Date": ""},
            "missing for 1 Cancelled",
        ),
        (
            "Satisfied",
            {"Cancellation Date": "01/04/2024"},
            "populated for 1 non-Cancelled",
        ),
        (
            "Unsatisfied",
            {"Cancellation Reason": "set aside"},
            "Cancellation Reason is populated",
        ),
        (
            "Satisfied",
            {
                "Satisfaction Date": "01/03/2024",
                "Status Effective Date": "01/02/2024",
            },
            "Status Effective Date precedes",
        ),
        (
            "Unsatisfied",
            {"Snapshot Date": "30/12/2024"},
            "does not match RT extract date",
        ),
    ],
)
def test_optional_timing_and_status_contradictions_fail_closed(
    tmp_path: Path,
    status: str,
    extra: dict[str, str],
    message: str,
) -> None:
    row = _row(status=status)
    row.update(extra)
    path = _write(tmp_path / "contradiction.csv", [row])

    with pytest.raises(ValueError, match=message):
        read_rt_extract(path, OBSERVED)


def test_repeated_ids_with_snapshot_dates_are_identified_as_history(
    tmp_path: Path,
) -> None:
    first = _row("J-1")
    first["Snapshot Date"] = OBSERVED
    second = dict(first)
    path = _write(tmp_path / "snapshots.csv", [first, second])

    with pytest.raises(ValueError, match="indicate historical snapshots"):
        read_rt_extract(path, OBSERVED)


def test_snapshot_column_must_be_complete_and_single_valued(tmp_path: Path) -> None:
    first = _row("J-1")
    first["Snapshot Date"] = OBSERVED
    second = _row("J-2")
    second["Snapshot Date"] = ""
    missing_path = _write(tmp_path / "missing-snapshot.csv", [first, second])
    with pytest.raises(ValueError, match="Snapshot Date is present.*missing"):
        read_rt_extract(missing_path, OBSERVED)

    second["Snapshot Date"] = "2024-12-30"
    mixed_path = _write(tmp_path / "mixed-snapshot.csv", [first, second])
    with pytest.raises(ValueError, match="2 distinct values"):
        read_rt_extract(mixed_path, OBSERVED)
