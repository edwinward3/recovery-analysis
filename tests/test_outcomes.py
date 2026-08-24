"""Synthetic boundary tests for aggregate RT outcome definitions."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from recovery.outcomes import (
    HOLIDAY_CALENDAR_SOURCE,
    aalen_johansen_monthly,
    cross_sectional_status_aggregates,
    england_wales_bank_holidays,
    mature_fixed_horizon_outcomes,
    one_calendar_month_landmark,
    outcome_validity_gate,
    registration_working_day_aggregates,
)


def _row(
    status: str = "Unsatisfied",
    *,
    judgment: str = "2020-01-31",
    inserted: str = "2020-02-03",
    satisfaction: str | None = None,
    cancellation: str | None = None,
    jurisdiction: str = "England and Wales",
) -> dict[str, object]:
    return {
        "ID": "private-id-not-for-output",
        "JudgmentDate": pd.Timestamp(judgment),
        "Date Inserted": pd.Timestamp(inserted),
        "JudgmentStatus": status,
        "Satisfaction Date": pd.Timestamp(satisfaction) if satisfaction else pd.NaT,
        "Cancellation Date": pd.Timestamp(cancellation) if cancellation else pd.NaT,
        "Jurisdiction": jurisdiction,
    }


def _frame(rows: list[dict[str, object]], *, dated_schema: bool = True) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if dated_schema:
        frame.attrs["raw_header_schema"] = (
            ("Satisfaction Date", "Satisfaction Date"),
            ("Cancellation Date", "Cancellation Date"),
        )
    return frame


def test_embedded_holidays_cover_every_year_and_label_one_off_dates() -> None:
    holidays = england_wales_bank_holidays()

    assert set(holidays["date"].dt.year) == set(range(2019, 2028))
    assert holidays["source"].eq(HOLIDAY_CALENDAR_SOURCE).all()
    names = holidays.set_index("date")["holiday"]
    assert names[pd.Timestamp("2020-05-08")].endswith("(VE Day)")
    assert "Jubilee" in names[pd.Timestamp("2022-06-03")]
    assert "Queen Elizabeth II" in names[pd.Timestamp("2022-09-19")]
    assert "Coronation" in names[pd.Timestamp("2023-05-08")]


@pytest.mark.parametrize(
    ("judgment", "landmark"),
    [
        ("2024-01-31", "2024-02-29"),
        ("2023-01-31", "2023-02-28"),
        ("2024-08-31", "2024-09-30"),
        ("2024-02-29", "2024-03-29"),
    ],
)
def test_one_month_landmark_uses_calendar_arithmetic(
    judgment: str,
    landmark: str,
) -> None:
    assert one_calendar_month_landmark(judgment) == pd.Timestamp(landmark)


def test_registration_aggregates_respect_holidays_and_exclude_anomalies() -> None:
    rows = [
        _row(judgment="2019-04-18", inserted="2019-04-23"),
        _row(judgment="2022-09-16", inserted="2022-09-20"),
        _row(judgment="2023-05-05", inserted="2023-05-09"),
        _row(judgment="2024-05-07", inserted="2024-05-09"),
        _row(judgment="2024-06-03", inserted="2024-06-02"),
        _row(judgment="2024-06-03", inserted="2024-06-04", jurisdiction="Scotland"),
        _row(judgment="2024-07-01", inserted="2024-08-01"),
    ]
    result = registration_working_day_aggregates(
        pd.DataFrame(rows), pd.Timestamp("2024-07-31")
    )

    assert result["holiday_calendar_source"] == HOLIDAY_CALENDAR_SOURCE
    assert result["england_wales_rows"] == 6
    assert result["excluded_other_jurisdiction_rows"] == 1
    assert result["valid_registration_rows"] == 4
    assert result["insertion_before_judgment_rows"] == 1
    assert result["insertion_after_extract_rows"] == 1
    assert result["within_one_working_day_rows"] == 3
    assert result["more_than_one_working_day_rows"] == 1
    assert result["within_one_working_day_share"] == pytest.approx(0.75)
    assert result["working_day_lag_max"] == 2
    assert "ID" not in result
    assert "private-id-not-for-output" not in result.values()


def test_registration_calendar_fails_closed_outside_embedded_coverage() -> None:
    frame = pd.DataFrame(
        [_row(judgment="2018-12-31", inserted="2019-01-02")]
    )
    with pytest.raises(ValueError, match="outside the embedded 2019-2027"):
        registration_working_day_aggregates(frame, "2024-12-31")


def test_status_only_schema_selects_cross_sectional_design() -> None:
    frame = _frame([_row(status="Satisfied")], dated_schema=False)
    frame = frame.drop(columns=["Satisfaction Date", "Cancellation Date"])

    gate = outcome_validity_gate(frame, "2022-12-31")

    assert gate["design"] == "cross_sectional"
    assert gate["reasons"] == (
        "Satisfaction Date was not supplied",
        "Cancellation Date was not supplied",
    )


def test_complete_exclusive_event_dates_select_longitudinal_design() -> None:
    frame = _frame(
        [
            _row(status="Satisfied", satisfaction="2020-03-29"),
            _row(status="Cancelled", cancellation="2020-03-15"),
            _row(),
        ]
    )

    gate = outcome_validity_gate(frame, "2022-12-31")

    assert gate["design"] == "longitudinal"
    assert gate["landmark_at_risk_rows"] == 3
    assert gate["mature_24_month_rows"] == 3
    assert not any(gate["invalid_counts"].values())


@pytest.mark.parametrize(
    ("rows", "invalid_key", "expected_design"),
    [
        (
            [
                _row(
                    status="Satisfied",
                    satisfaction="2020-03-29",
                    cancellation="2020-04-01",
                )
            ],
            "both_event_dates_rows",
            "cross_sectional",
        ),
        (
            [_row(status="Satisfied", satisfaction="2020-02-29")],
            "satisfaction_on_or_before_landmark_rows",
            "cross_sectional",
        ),
        (
            [_row(status="Satisfied", satisfaction=None)],
            "satisfied_without_date_rows",
            "cross_sectional",
        ),
        (
            [_row(status="Cancelled", cancellation=None)],
            "cancelled_without_date_rows",
            "cross_sectional",
        ),
        (
            [_row(inserted="2020-01-30")],
            "insertion_before_judgment_rows",
            "cross_sectional",
        ),
        (
            [_row(inserted="2023-01-01")],
            "insertion_after_extract_rows",
            "blocked",
        ),
        (
            [
                _row(
                    status="Satisfied",
                    inserted="2020-03-15",
                    satisfaction="2020-03-10",
                )
            ],
            "event_before_insertion_rows",
            "cross_sectional",
        ),
    ],
)
def test_validity_gate_selects_safe_fallback_for_timing_or_state_contradictions(
    rows: list[dict[str, object]],
    invalid_key: str,
    expected_design: str,
) -> None:
    gate = outcome_validity_gate(_frame(rows), "2022-12-31")

    assert gate["design"] == expected_design
    assert gate["invalid_counts"][invalid_key] == 1
    assert f"{invalid_key}=1" in gate["reasons"]


def test_validity_gate_returns_blocked_aggregate_for_missing_required_data() -> None:
    gate = outcome_validity_gate(pd.DataFrame({"JudgmentStatus": ["Unsatisfied"]}), "2024-01-01")

    assert gate["design"] == "blocked"
    assert gate["rows"] == 1
    assert "missing required" in gate["reasons"][0]


def test_status_effective_date_is_reported_but_not_used_without_semantics() -> None:
    frame = _frame([_row(status="Satisfied", satisfaction="2020-03-29")])
    frame["Status Effective Date"] = pd.Timestamp("2020-03-01")

    gate = outcome_validity_gate(frame, "2022-12-31")

    assert gate["design"] == "longitudinal"
    assert gate["invalid_counts"]["status_effective_before_event_rows"] == 1
    assert not gate["reasons"]


def test_aalen_johansen_competing_risks_use_exact_months() -> None:
    # The Jan-31 judgment has a Feb-29 landmark in this leap year.
    frame = _frame(
        [
            _row(status="Satisfied", satisfaction="2020-03-29"),
            _row(status="Cancelled", cancellation="2020-03-14"),
            _row(status="Satisfied", satisfaction="2020-04-29"),
            _row(),
        ]
    )

    curve = aalen_johansen_monthly(frame, "2020-06-30").set_index("month")

    assert curve.loc[0, "at_risk"] == 4
    assert curve.loc[1, "satisfaction_events"] == 1
    assert curve.loc[1, "cancellation_events"] == 1
    assert curve.loc[1, "satisfaction_cif"] == pytest.approx(0.25)
    assert curve.loc[1, "cancellation_cif"] == pytest.approx(0.25)
    assert curve.loc[1, "event_free_survival"] == pytest.approx(0.5)
    assert curve.loc[2, "satisfaction_events"] == 1
    assert curve.loc[2, "satisfaction_cif"] == pytest.approx(0.5)
    assert curve.loc[2, "cancellation_cif"] == pytest.approx(0.25)
    assert curve.loc[2, "event_free_survival"] == pytest.approx(0.25)
    totals = (
        curve["satisfaction_cif"]
        + curve["cancellation_cif"]
        + curve["event_free_survival"]
    )
    assert np.allclose(totals, 1.0)
    assert "ID" not in curve.columns


def test_aalen_johansen_keeps_terminal_first_month_when_all_records_have_events() -> None:
    frame = _frame(
        [
            _row(
                judgment="2020-01-01",
                inserted="2020-01-02",
                status="Satisfied",
                satisfaction="2020-02-15",
            ),
            _row(
                judgment="2020-01-01",
                inserted="2020-01-02",
                status="Cancelled",
                cancellation="2020-02-20",
            ),
        ]
    )

    curve = aalen_johansen_monthly(frame, "2021-01-01").set_index("month")

    assert tuple(curve.index) == (0, 1)
    assert curve.loc[1, "at_risk"] == 2
    assert curve.loc[1, "satisfaction_events"] == 1
    assert curve.loc[1, "cancellation_events"] == 1
    assert curve.loc[1, "satisfaction_cif"] == pytest.approx(0.5)
    assert curve.loc[1, "cancellation_cif"] == pytest.approx(0.5)
    assert curve.loc[1, "event_free_survival"] == pytest.approx(0.0)


def test_aalen_johansen_excludes_pre_landmark_cancellation_and_late_registration() -> None:
    frame = _frame(
        [
            _row(status="Cancelled", cancellation="2020-02-20"),
            _row(inserted="2020-03-01"),
            _row(),
        ]
    )

    curve = aalen_johansen_monthly(frame, "2020-06-30")

    assert curve.loc[0, "at_risk"] == 1
    assert curve["satisfaction_events"].sum() == 0
    assert curve["cancellation_events"].sum() == 0


def test_aalen_johansen_rejects_status_only_data() -> None:
    frame = _frame([_row()], dated_schema=False).drop(
        columns=["Satisfaction Date", "Cancellation Date"]
    )
    with pytest.raises(ValueError, match="cross_sectional"):
        aalen_johansen_monthly(frame, "2022-12-31")


def test_fixed_horizons_are_mature_exclusive_and_use_cutoff_dates() -> None:
    rows = [
        _row(status="Satisfied", satisfaction="2021-02-28"),
        _row(status="Satisfied", satisfaction="2021-03-01"),
        _row(status="Cancelled", cancellation="2021-02-28"),
        _row(status="Cancelled", cancellation="2021-03-01"),
        _row(),
        _row(status="Cancelled", cancellation="2020-02-20"),
        _row(inserted="2020-03-01"),
        _row(judgment="2021-12-31", inserted="2022-01-03"),
    ]
    table = mature_fixed_horizon_outcomes(
        _frame(rows), "2022-03-01"
    )
    assert set(table["judgment_cohort"]) == {"all", "2020Q1"}
    table = table.loc[table["judgment_cohort"].eq("all")].set_index(
        "horizon_months"
    )

    assert tuple(table.index) == (12, 24)
    assert table.loc[12, "eligible_rows"] == 5
    assert table.loc[12, "satisfied_rows"] == 1
    assert table.loc[12, "cancelled_rows"] == 1
    assert table.loc[12, "unsatisfied_rows"] == 3
    assert table.loc[24, "eligible_rows"] == 5
    assert table.loc[24, "satisfied_rows"] == 2
    assert table.loc[24, "cancelled_rows"] == 2
    assert table.loc[24, "unsatisfied_rows"] == 1
    assert table.loc[12, "excluded_cancelled_by_landmark_rows"] == 1
    assert table.loc[12, "excluded_late_registration_rows"] == 1
    assert table.loc[12, "excluded_immature_rows"] == 1
    for horizon in (12, 24):
        shares = table.loc[
            horizon,
            ["satisfied_share", "cancelled_share", "unsatisfied_share"],
        ].astype(float)
        assert shares.sum() == pytest.approx(1.0)
    assert "ID" not in table.columns


def test_fixed_horizons_report_nan_shares_when_no_cohort_is_mature() -> None:
    table = mature_fixed_horizon_outcomes(_frame([_row()]), "2020-03-01")
    table = table.loc[table["judgment_cohort"].eq("all")]

    assert table["eligible_rows"].eq(0).all()
    assert table["satisfied_share"].map(math.isnan).all()


def test_cross_sectional_statuses_remain_three_separate_states() -> None:
    frame = _frame(
        [
            _row(status="Satisfied", satisfaction="2020-03-29"),
            _row(),
            _row(),
            _row(status="Cancelled", cancellation="2020-03-15"),
        ]
    )

    table = cross_sectional_status_aggregates(frame, "2022-12-31")
    assert set(table["dimension"]) == {"overall", "judgment_quarter"}
    table = table.loc[table["dimension"].eq("overall")].set_index("status")

    assert table.loc["Satisfied", "rows"] == 1
    assert table.loc["Unsatisfied", "rows"] == 2
    assert table.loc["Cancelled", "rows"] == 1
    assert table.loc["Cancelled", "share"] == pytest.approx(0.25)
    assert table["share"].sum() == pytest.approx(1.0)
    assert table["estimand"].eq("status_among_records_present_at_extract").all()
    assert "ID" not in table.columns


def test_cross_sectional_aggregate_rejects_future_judgments() -> None:
    frame = _frame([_row(judgment="2025-01-02", inserted="2025-01-03")])
    with pytest.raises(ValueError, match="after the extract"):
        cross_sectional_status_aggregates(frame, "2025-01-01")
