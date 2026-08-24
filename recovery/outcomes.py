"""Registry Trust outcome summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


STATUSES: tuple[str, ...] = ("Satisfied", "Unsatisfied", "Cancelled")
LANDMARK_MONTHS = 1
FIXED_HORIZONS = (12, 24)
PRIMARY_HORIZON_MONTHS = FIXED_HORIZONS[0]
RETENTION_MONTHS = 72
HOLIDAY_CALENDAR_SOURCE = (
    "GOV.UK bank holidays in England and Wales; embedded calendar 2019-2027"
)

# Official England-and-Wales dates, including one-off national bank holidays.
_HOLIDAYS: tuple[tuple[str, str], ...] = (
    ("2019-01-01", "New Year's Day"),
    ("2019-04-19", "Good Friday"),
    ("2019-04-22", "Easter Monday"),
    ("2019-05-06", "Early May bank holiday"),
    ("2019-05-27", "Spring bank holiday"),
    ("2019-08-26", "Summer bank holiday"),
    ("2019-12-25", "Christmas Day"),
    ("2019-12-26", "Boxing Day"),
    ("2020-01-01", "New Year's Day"),
    ("2020-04-10", "Good Friday"),
    ("2020-04-13", "Easter Monday"),
    ("2020-05-08", "Early May bank holiday (VE Day)"),
    ("2020-05-25", "Spring bank holiday"),
    ("2020-08-31", "Summer bank holiday"),
    ("2020-12-25", "Christmas Day"),
    ("2020-12-28", "Boxing Day substitute"),
    ("2021-01-01", "New Year's Day"),
    ("2021-04-02", "Good Friday"),
    ("2021-04-05", "Easter Monday"),
    ("2021-05-03", "Early May bank holiday"),
    ("2021-05-31", "Spring bank holiday"),
    ("2021-08-30", "Summer bank holiday"),
    ("2021-12-27", "Christmas Day substitute"),
    ("2021-12-28", "Boxing Day substitute"),
    ("2022-01-03", "New Year's Day substitute"),
    ("2022-04-15", "Good Friday"),
    ("2022-04-18", "Easter Monday"),
    ("2022-05-02", "Early May bank holiday"),
    ("2022-06-02", "Spring bank holiday"),
    ("2022-06-03", "Platinum Jubilee bank holiday"),
    ("2022-08-29", "Summer bank holiday"),
    ("2022-09-19", "State Funeral of Queen Elizabeth II"),
    ("2022-12-26", "Boxing Day"),
    ("2022-12-27", "Christmas Day substitute"),
    ("2023-01-02", "New Year's Day substitute"),
    ("2023-04-07", "Good Friday"),
    ("2023-04-10", "Easter Monday"),
    ("2023-05-01", "Early May bank holiday"),
    ("2023-05-08", "Coronation bank holiday"),
    ("2023-05-29", "Spring bank holiday"),
    ("2023-08-28", "Summer bank holiday"),
    ("2023-12-25", "Christmas Day"),
    ("2023-12-26", "Boxing Day"),
    ("2024-01-01", "New Year's Day"),
    ("2024-03-29", "Good Friday"),
    ("2024-04-01", "Easter Monday"),
    ("2024-05-06", "Early May bank holiday"),
    ("2024-05-27", "Spring bank holiday"),
    ("2024-08-26", "Summer bank holiday"),
    ("2024-12-25", "Christmas Day"),
    ("2024-12-26", "Boxing Day"),
    ("2025-01-01", "New Year's Day"),
    ("2025-04-18", "Good Friday"),
    ("2025-04-21", "Easter Monday"),
    ("2025-05-05", "Early May bank holiday"),
    ("2025-05-26", "Spring bank holiday"),
    ("2025-08-25", "Summer bank holiday"),
    ("2025-12-25", "Christmas Day"),
    ("2025-12-26", "Boxing Day"),
    ("2026-01-01", "New Year's Day"),
    ("2026-04-03", "Good Friday"),
    ("2026-04-06", "Easter Monday"),
    ("2026-05-04", "Early May bank holiday"),
    ("2026-05-25", "Spring bank holiday"),
    ("2026-08-31", "Summer bank holiday"),
    ("2026-12-25", "Christmas Day"),
    ("2026-12-28", "Boxing Day substitute"),
    ("2027-01-01", "New Year's Day"),
    ("2027-03-26", "Good Friday"),
    ("2027-03-29", "Easter Monday"),
    ("2027-05-03", "Early May bank holiday"),
    ("2027-05-31", "Spring bank holiday"),
    ("2027-08-30", "Summer bank holiday"),
    ("2027-12-27", "Christmas Day substitute"),
    ("2027-12-28", "Boxing Day substitute"),
)


def england_wales_bank_holidays() -> pd.DataFrame:
    """Return the embedded holiday calendar with its public source label."""

    return pd.DataFrame(_HOLIDAYS, columns=["date", "holiday"]).assign(
        date=lambda table: pd.to_datetime(table["date"]),
        source=HOLIDAY_CALENDAR_SOURCE,
    )


def _extract_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid extract date: {value!r}") from exc
    if pd.isna(stamp):
        raise ValueError("extract date is missing")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(None)
    return stamp.normalize()


def one_calendar_month_landmark(value: str | pd.Timestamp) -> pd.Timestamp:
    """Return the date exactly one calendar month later."""

    stamp = _extract_timestamp(value)
    return (stamp + pd.DateOffset(months=LANDMARK_MONTHS)).normalize()


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"judgments are missing required column(s): {missing}")


def _required_dates(frame: pd.DataFrame, column: str) -> pd.Series:
    parsed = pd.to_datetime(frame[column], errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{column} has {int(parsed.isna().sum())} missing/invalid row(s)")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_convert(None)
    return parsed.dt.normalize()


def _optional_dates(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    raw = frame[column]
    populated = raw.notna() & raw.astype("string").str.strip().ne("")
    parsed = pd.to_datetime(raw.where(populated), errors="coerce")
    invalid = populated & parsed.isna()
    if invalid.any():
        raise ValueError(f"{column} has {int(invalid.sum())} invalid row(s)")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_convert(None)
    return parsed.dt.normalize()


def _add_calendar_months(values: pd.Series, months: int | np.ndarray) -> pd.Series:
    """Add calendar months to a series of dates."""

    dates = pd.Series(values.to_numpy(), dtype="datetime64[ns]")
    if dates.empty:
        return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    offsets = np.broadcast_to(np.asarray(months, dtype=np.int64), len(dates))
    ordinal = (
        dates.dt.year.to_numpy(dtype=np.int64) * 12
        + dates.dt.month.to_numpy(dtype=np.int64)
        - 1
        + offsets
    )
    years = ordinal // 12
    month_numbers = ordinal % 12 + 1
    first = pd.to_datetime(
        {"year": years, "month": month_numbers, "day": np.ones(len(dates), int)}
    )
    days = np.minimum(
        dates.dt.day.to_numpy(dtype=np.int64),
        first.dt.days_in_month.to_numpy(dtype=np.int64),
    )
    result = pd.to_datetime(
        {"year": years, "month": month_numbers, "day": days}
    )
    result.index = values.index
    return result


def _calendar_month_durations(start: pd.Series, end: pd.Series) -> np.ndarray:
    """Express elapsed time in exact fractional calendar months."""

    starts = pd.Series(start.to_numpy(), dtype="datetime64[ns]")
    ends = pd.Series(end.to_numpy(), dtype="datetime64[ns]")
    if (ends < starts).any():
        raise ValueError("follow-up ends before the one-month landmark")
    rough = (
        (ends.dt.year - starts.dt.year) * 12 + ends.dt.month - starts.dt.month
    ).to_numpy(dtype=np.int64)
    anniversary = _add_calendar_months(starts, rough)
    adjust = anniversary.gt(ends).to_numpy()
    rough[adjust] -= 1
    anniversary = _add_calendar_months(starts, rough)
    next_anniversary = _add_calendar_months(starts, rough + 1)
    elapsed = (ends - anniversary).dt.days.to_numpy(dtype=float)
    span = (next_anniversary - anniversary).dt.days.to_numpy(dtype=float)
    return rough.astype(float) + elapsed / span


def _source_column_present(
    frame: pd.DataFrame,
    column: str,
    parsed: pd.Series,
) -> bool:
    absent = frame.attrs.get("absent_optional")
    if isinstance(absent, (tuple, list, set)):
        return column not in set(absent)
    schema = frame.attrs.get("raw_header_schema")
    if isinstance(schema, (tuple, list)):
        return any(
            isinstance(pair, (tuple, list)) and len(pair) == 2 and pair[1] == column
            for pair in schema
        )
    return column in frame and bool(parsed.notna().any())


def _prepare(
    judgments: pd.DataFrame,
) -> tuple[pd.DataFrame, bool, bool]:
    _require_columns(
        judgments,
        ("JudgmentDate", "Date Inserted", "JudgmentStatus"),
    )
    if judgments.empty:
        raise ValueError("judgments contain no rows")
    frame = judgments.reset_index(drop=True).copy()
    status = frame["JudgmentStatus"].astype("string").fillna("").str.strip()
    invalid_status = ~status.isin(STATUSES)
    if invalid_status.any():
        raise ValueError(
            f"JudgmentStatus has {int(invalid_status.sum())} unsupported row(s)"
        )
    frame["_status"] = status.astype(str)
    frame["_judgment"] = _required_dates(frame, "JudgmentDate")
    frame["_inserted"] = _required_dates(frame, "Date Inserted")
    frame["_satisfaction"] = _optional_dates(frame, "Satisfaction Date")
    frame["_cancellation"] = _optional_dates(frame, "Cancellation Date")
    frame["_status_effective"] = _optional_dates(frame, "Status Effective Date")
    frame["_landmark"] = _add_calendar_months(frame["_judgment"], LANDMARK_MONTHS)
    frame["_retention_end"] = _add_calendar_months(
        frame["_judgment"], RETENTION_MONTHS
    )
    satisfaction_source = _source_column_present(
        judgments, "Satisfaction Date", frame["_satisfaction"]
    )
    cancellation_source = _source_column_present(
        judgments, "Cancellation Date", frame["_cancellation"]
    )
    return frame, satisfaction_source, cancellation_source


def _evaluate_gate(
    frame: pd.DataFrame,
    extract: pd.Timestamp,
    satisfaction_source: bool,
    cancellation_source: bool,
) -> dict[str, object]:
    status = frame["_status"]
    satisfaction = frame["_satisfaction"]
    cancellation = frame["_cancellation"]
    both = satisfaction.notna() & cancellation.notna()
    invalid_counts = {
        "judgment_after_extract_rows": int(frame["_judgment"].gt(extract).sum()),
        "insertion_before_judgment_rows": int(
            frame["_inserted"].lt(frame["_judgment"]).sum()
        ),
        "insertion_after_extract_rows": int(frame["_inserted"].gt(extract).sum()),
        "satisfaction_date_unparseable_rows": int(
            frame.get(
                "_invalid_satisfaction_date",
                pd.Series(False, index=frame.index),
            ).sum()
        ),
        "cancellation_date_unparseable_rows": int(
            frame.get(
                "_invalid_cancellation_date",
                pd.Series(False, index=frame.index),
            ).sum()
        ),
        "status_effective_date_unparseable_rows": int(
            frame.get(
                "_invalid_status_effective_date",
                pd.Series(False, index=frame.index),
            ).sum()
        ),
        "satisfied_without_date_rows": int(
            (status.eq("Satisfied") & satisfaction.isna()).sum()
        ),
        "non_satisfied_with_satisfaction_date_rows": int(
            (~status.eq("Satisfied") & satisfaction.notna()).sum()
        ),
        "cancelled_without_date_rows": int(
            (status.eq("Cancelled") & cancellation.isna()).sum()
        ),
        "non_cancelled_with_cancellation_date_rows": int(
            (~status.eq("Cancelled") & cancellation.notna()).sum()
        ),
        "both_event_dates_rows": int(both.sum()),
        "event_before_judgment_rows": int(
            (
                satisfaction.lt(frame["_judgment"])
                | cancellation.lt(frame["_judgment"])
            ).sum()
        ),
        "event_before_insertion_rows": int(
            (
                satisfaction.lt(frame["_inserted"])
                | cancellation.lt(frame["_inserted"])
            ).sum()
        ),
        "event_after_extract_rows": int(
            (satisfaction.gt(extract) | cancellation.gt(extract)).sum()
        ),
        "event_after_retention_rows": int(
            (
                satisfaction.gt(frame["_retention_end"])
                | cancellation.gt(frame["_retention_end"])
            ).sum()
        ),
        "satisfaction_on_or_before_landmark_rows": int(
            (satisfaction.notna() & satisfaction.le(frame["_landmark"])).sum()
        ),
        "status_effective_before_event_rows": int(
            (
                frame["_status_effective"].notna()
                & (
                    (satisfaction.notna() & frame["_status_effective"].lt(satisfaction))
                    | (cancellation.notna() & frame["_status_effective"].lt(cancellation))
                )
            ).sum()
        ),
    }
    core_keys = (
        "judgment_after_extract_rows",
        "insertion_after_extract_rows",
    )
    optional_keys = (
        "insertion_before_judgment_rows",
        "both_event_dates_rows",
        "event_before_judgment_rows",
        "event_before_insertion_rows",
        "event_after_extract_rows",
        "event_after_retention_rows",
        "satisfaction_on_or_before_landmark_rows",
    )
    if satisfaction_source:
        optional_keys += (
            "satisfaction_date_unparseable_rows",
            "satisfied_without_date_rows",
            "non_satisfied_with_satisfaction_date_rows",
        )
    if cancellation_source:
        optional_keys += (
            "cancellation_date_unparseable_rows",
            "cancelled_without_date_rows",
            "non_cancelled_with_cancellation_date_rows",
        )
    core_failures = [key for key in core_keys if invalid_counts[key] > 0]
    optional_failures = [key for key in optional_keys if invalid_counts[key] > 0]
    missing_sources = [
        column
        for column, present in (
            ("Satisfaction Date", satisfaction_source),
            ("Cancellation Date", cancellation_source),
        )
        if not present
    ]
    if core_failures:
        design = "blocked"
        reasons = tuple(
            f"{key}={invalid_counts[key]}" for key in dict.fromkeys(core_failures)
        )
    elif optional_failures or missing_sources:
        design = "cross_sectional"
        reasons = tuple(
            f"{key}={invalid_counts[key]}"
            for key in dict.fromkeys(optional_failures)
        ) + tuple(f"{column} was not supplied" for column in missing_sources)
    else:
        design = "longitudinal"
        reasons = ()

    registered = frame["_inserted"].le(frame["_landmark"])
    early_cancellation = cancellation.notna() & cancellation.le(frame["_landmark"])
    observable = frame["_landmark"].le(
        pd.concat(
            [
                pd.Series(extract, index=frame.index),
                frame["_retention_end"],
            ],
            axis=1,
        ).min(axis=1)
    )
    at_risk = registered & ~early_cancellation & observable
    mature_counts: dict[str, int] = {}
    for months in FIXED_HORIZONS:
        cutoff = _add_calendar_months(frame["_landmark"], months)
        mature_counts[f"mature_{months}_month_rows"] = int(
            (at_risk & cutoff.le(extract) & cutoff.le(frame["_retention_end"])).sum()
        )
    return {
        "design": design,
        "extract_date": extract.date().isoformat(),
        "rows": int(len(frame)),
        "satisfaction_date_source_present": satisfaction_source,
        "cancellation_date_source_present": cancellation_source,
        "landmark_at_risk_rows": int(at_risk.sum()),
        **mature_counts,
        "invalid_counts": invalid_counts,
        "reasons": reasons,
    }


def outcome_validity_gate(
    judgments: pd.DataFrame,
    extract_date: str | pd.Timestamp,
) -> dict[str, object]:
    """Choose longitudinal, cross-sectional, or blocked analysis."""

    extract = _extract_timestamp(extract_date)
    try:
        frame, satisfaction_source, cancellation_source = _prepare(judgments)
    except ValueError as exc:
        return {
            "design": "blocked",
            "extract_date": extract.date().isoformat(),
            "rows": int(len(judgments)),
            "satisfaction_date_source_present": False,
            "cancellation_date_source_present": False,
            "landmark_at_risk_rows": 0,
            "mature_12_month_rows": 0,
            "mature_24_month_rows": 0,
            "invalid_counts": {},
            "reasons": (str(exc),),
        }
    return _evaluate_gate(
        frame,
        extract,
        satisfaction_source,
        cancellation_source,
    )


def registration_working_day_aggregates(
    judgments: pd.DataFrame,
    extract_date: str | pd.Timestamp,
) -> dict[str, object]:
    """Summarise E&W registration delay without exposing judgment rows."""

    _require_columns(
        judgments,
        ("JudgmentDate", "Date Inserted", "Jurisdiction"),
    )
    extract = _extract_timestamp(extract_date)
    frame = judgments.reset_index(drop=True)
    judgment = _required_dates(frame, "JudgmentDate")
    inserted = _required_dates(frame, "Date Inserted")
    jurisdiction = frame["Jurisdiction"].astype("string").fillna("").str.strip()
    england_wales = jurisdiction.eq("England and Wales")
    negative = inserted.lt(judgment) & england_wales
    after_extract = inserted.gt(extract) & england_wales
    future_judgment = judgment.gt(extract) & england_wales
    valid = england_wales & ~negative & ~after_extract & ~future_judgment
    valid_judgment = judgment.loc[valid]
    valid_inserted = inserted.loc[valid]
    if not valid_judgment.empty:
        calendar_start = pd.Timestamp("2019-01-01")
        calendar_end = pd.Timestamp("2027-12-31")
        if valid_judgment.min() < calendar_start or valid_inserted.max() > calendar_end:
            raise ValueError(
                "registration dates fall outside the embedded 2019-2027 "
                "England-and-Wales holiday calendar"
            )
    holidays = england_wales_bank_holidays()["date"].to_numpy(dtype="datetime64[D]")
    begin = (valid_judgment + pd.Timedelta(days=1)).to_numpy(dtype="datetime64[D]")
    end = (valid_inserted + pd.Timedelta(days=1)).to_numpy(dtype="datetime64[D]")
    working_lag = np.busday_count(begin, end, holidays=holidays).astype(int)
    calendar_lag = (valid_inserted - valid_judgment).dt.days.to_numpy(dtype=int)
    insertion_is_working_day = np.is_busday(
        valid_inserted.to_numpy(dtype="datetime64[D]"), holidays=holidays
    )
    denominator = len(calendar_lag)

    def quantile(values: np.ndarray, probability: float) -> float:
        return float(np.quantile(values, probability)) if len(values) else float("nan")

    return {
        "extract_date": extract.date().isoformat(),
        "holiday_calendar_source": HOLIDAY_CALENDAR_SOURCE,
        "holiday_calendar_start": "2019-01-01",
        "holiday_calendar_end": "2027-12-31",
        "england_wales_rows": int(england_wales.sum()),
        "excluded_other_jurisdiction_rows": int((~england_wales).sum()),
        "valid_registration_rows": int(denominator),
        "insertion_before_judgment_rows": int(negative.sum()),
        "insertion_after_extract_rows": int(after_extract.sum()),
        "judgment_after_extract_rows": int(future_judgment.sum()),
        "same_calendar_day_rows": int((calendar_lag == 0).sum()),
        "next_calendar_day_rows": int((calendar_lag == 1).sum()),
        "same_working_day_rows": int((working_lag == 0).sum()),
        "next_working_day_rows": int((working_lag == 1).sum()),
        "within_one_working_day_rows": int((working_lag <= 1).sum()),
        "more_than_one_working_day_rows": int((working_lag > 1).sum()),
        "inserted_on_non_working_day_rows": int((~insertion_is_working_day).sum()),
        "within_one_working_day_share": (
            float((working_lag <= 1).mean()) if denominator else float("nan")
        ),
        "calendar_day_lag_median": quantile(calendar_lag, 0.5),
        "calendar_day_lag_p95": quantile(calendar_lag, 0.95),
        "calendar_day_lag_max": (
            int(calendar_lag.max()) if denominator else float("nan")
        ),
        "working_day_lag_median": quantile(working_lag, 0.5),
        "working_day_lag_p95": quantile(working_lag, 0.95),
        "working_day_lag_max": (
            int(working_lag.max()) if denominator else float("nan")
        ),
    }


def _require_longitudinal(
    frame: pd.DataFrame,
    extract: pd.Timestamp,
    satisfaction_source: bool,
    cancellation_source: bool,
) -> dict[str, object]:
    gate = _evaluate_gate(
        frame,
        extract,
        satisfaction_source,
        cancellation_source,
    )
    if gate["design"] != "longitudinal":
        raise ValueError(
            f"longitudinal outcomes are unavailable: {gate['design']}; "
            f"{'; '.join(gate['reasons'])}"
        )
    return gate


def _landmark_risk_set(frame: pd.DataFrame, extract: pd.Timestamp) -> pd.DataFrame:
    censor = pd.concat(
        [pd.Series(extract, index=frame.index), frame["_retention_end"]], axis=1
    ).min(axis=1)
    eligible = (
        frame["_inserted"].le(frame["_landmark"])
        & ~(
            frame["_cancellation"].notna()
            & frame["_cancellation"].le(frame["_landmark"])
        )
        & frame["_landmark"].le(censor)
    )
    result = frame.loc[eligible].copy().reset_index(drop=True)
    result["_censor"] = censor.loc[eligible].to_numpy()
    return result


def aalen_johansen_monthly(
    judgments: pd.DataFrame,
    extract_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Calculate monthly satisfaction and cancellation incidence."""

    extract = _extract_timestamp(extract_date)
    frame, satisfaction_source, cancellation_source = _prepare(judgments)
    _require_longitudinal(
        frame,
        extract,
        satisfaction_source,
        cancellation_source,
    )
    risk = _landmark_risk_set(frame, extract)
    columns = [
        "month",
        "at_risk",
        "satisfaction_events",
        "cancellation_events",
        "censored",
        "satisfaction_cif",
        "cancellation_cif",
        "event_free_survival",
        "complete_month",
        "extract_date",
        "time_origin",
    ]
    if risk.empty:
        return pd.DataFrame(columns=columns)

    satisfaction_event = risk["_satisfaction"].notna() & risk[
        "_satisfaction"
    ].le(risk["_censor"])
    cancellation_event = risk["_cancellation"].notna() & risk[
        "_cancellation"
    ].le(risk["_censor"])
    end = risk["_censor"].copy()
    end.loc[satisfaction_event] = risk.loc[satisfaction_event, "_satisfaction"]
    end.loc[cancellation_event] = risk.loc[cancellation_event, "_cancellation"]
    event = np.select(
        [satisfaction_event, cancellation_event],
        ["satisfaction", "cancellation"],
        default="censored",
    )
    durations = _calendar_month_durations(risk["_landmark"], end)
    possible_follow_up = _calendar_month_durations(
        risk["_landmark"], risk["_censor"]
    )
    timeline = pd.DataFrame({"time": durations, "event": event})
    counts = (
        timeline.assign(rows=1)
        .pivot_table(
            index="time",
            columns="event",
            values="rows",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=["satisfaction", "cancellation", "censored"], fill_value=0)
        .sort_index()
    )
    total_at_time = counts.sum(axis=1).to_numpy(dtype=int)
    at_risk = len(timeline) - np.r_[0, np.cumsum(total_at_time)[:-1]]
    satisfaction_count = counts["satisfaction"].to_numpy(dtype=float)
    cancellation_count = counts["cancellation"].to_numpy(dtype=float)
    event_count = satisfaction_count + cancellation_count
    survival_factor = 1.0 - event_count / at_risk
    survival_after = np.cumprod(survival_factor)
    survival_before = np.r_[1.0, survival_after[:-1]]
    satisfaction_cif = np.cumsum(survival_before * satisfaction_count / at_risk)
    cancellation_cif = np.cumsum(survival_before * cancellation_count / at_risk)

    max_month = int(np.ceil(durations.max()))
    months = np.arange(max_month + 1, dtype=int)
    positions = np.searchsorted(counts.index.to_numpy(dtype=float), months, side="right") - 1
    valid_position = positions >= 0
    month_satisfaction = np.zeros(len(months), dtype=float)
    month_cancellation = np.zeros(len(months), dtype=float)
    month_survival = np.ones(len(months), dtype=float)
    month_satisfaction[valid_position] = satisfaction_cif[positions[valid_position]]
    month_cancellation[valid_position] = cancellation_cif[positions[valid_position]]
    month_survival[valid_position] = survival_after[positions[valid_position]]

    interval = np.ceil(durations).astype(int)
    interval_counts = (
        pd.DataFrame({"month": interval, "event": event})
        .assign(rows=1)
        .pivot_table(
            index="month",
            columns="event",
            values="rows",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=months, fill_value=0)
        .reindex(columns=["satisfaction", "cancellation", "censored"], fill_value=0)
    )
    sorted_durations = np.sort(durations)
    risk_at_month = np.empty(len(months), dtype=int)
    risk_at_month[0] = len(durations)
    if len(months) > 1:
        risk_at_month[1:] = len(durations) - np.searchsorted(
            sorted_durations, months[:-1], side="right"
        )
    result = pd.DataFrame(
        {
            "month": months,
            "at_risk": risk_at_month.astype(int),
            "satisfaction_events": interval_counts["satisfaction"].to_numpy(int),
            "cancellation_events": interval_counts["cancellation"].to_numpy(int),
            "censored": interval_counts["censored"].to_numpy(int),
            "satisfaction_cif": month_satisfaction,
            "cancellation_cif": month_cancellation,
            "event_free_survival": month_survival,
            "complete_month": months <= int(np.floor(possible_follow_up.min())),
            "extract_date": extract.date().isoformat(),
            "time_origin": "one_calendar_month_landmark",
        },
        columns=columns,
    )
    return result.reset_index(drop=True)


def mature_fixed_horizon_outcomes(
    judgments: pd.DataFrame,
    extract_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return exclusive 12- and 24-month landmark outcome aggregates."""

    extract = _extract_timestamp(extract_date)
    frame, satisfaction_source, cancellation_source = _prepare(judgments)
    _require_longitudinal(
        frame,
        extract,
        satisfaction_source,
        cancellation_source,
    )
    registered = frame["_inserted"].le(frame["_landmark"])
    early_cancellation = frame["_cancellation"].notna() & frame[
        "_cancellation"
    ].le(frame["_landmark"])
    observable = frame["_landmark"].le(extract) & frame["_landmark"].le(
        frame["_retention_end"]
    )
    landmark_eligible = registered & ~early_cancellation & observable
    quarters = frame["_judgment"].dt.to_period("Q").astype(str)
    rows: list[dict[str, object]] = []
    for months in FIXED_HORIZONS:
        cutoff = _add_calendar_months(frame["_landmark"], months)
        mature = landmark_eligible & cutoff.le(extract) & cutoff.le(
            frame["_retention_end"]
        )
        cohort_masks = [("all", pd.Series(True, index=frame.index))]
        cohort_masks.extend(
            (quarter, quarters.eq(quarter))
            for quarter in sorted(quarters.loc[mature].unique())
        )
        for cohort, cohort_mask in cohort_masks:
            cohort_mature = mature & cohort_mask
            satisfied = cohort_mature & frame["_satisfaction"].notna() & frame[
                "_satisfaction"
            ].le(cutoff)
            cancelled = cohort_mature & frame["_cancellation"].notna() & frame[
                "_cancellation"
            ].le(cutoff)
            if (satisfied & cancelled).any():
                raise ValueError("satisfaction and cancellation outcomes overlap")
            unsatisfied = cohort_mature & ~satisfied & ~cancelled
            eligible_rows = int(cohort_mature.sum())
            rows.append(
                {
                    "horizon_months": months,
                    "judgment_cohort": cohort,
                    "eligible_rows": eligible_rows,
                    "satisfied_rows": int(satisfied.sum()),
                    "cancelled_rows": int(cancelled.sum()),
                    "unsatisfied_rows": int(unsatisfied.sum()),
                    "satisfied_share": (
                        float(satisfied.sum() / eligible_rows)
                        if eligible_rows
                        else float("nan")
                    ),
                    "cancelled_share": (
                        float(cancelled.sum() / eligible_rows)
                        if eligible_rows
                        else float("nan")
                    ),
                    "unsatisfied_share": (
                        float(unsatisfied.sum() / eligible_rows)
                        if eligible_rows
                        else float("nan")
                    ),
                    "excluded_late_registration_rows": int(
                        (cohort_mask & ~registered).sum()
                    ),
                    "excluded_cancelled_by_landmark_rows": int(
                        (cohort_mask & early_cancellation).sum()
                    ),
                    "excluded_unobservable_landmark_rows": int(
                        (cohort_mask & ~observable).sum()
                    ),
                    "excluded_immature_rows": int(
                        (cohort_mask & landmark_eligible & ~mature).sum()
                    ),
                    "extract_date": extract.date().isoformat(),
                    "time_origin": "one_calendar_month_landmark",
                }
            )
    return pd.DataFrame(rows)


def cross_sectional_status_aggregates(
    judgments: pd.DataFrame,
    extract_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Describe current status without implying event timing."""

    _require_columns(judgments, ("JudgmentDate", "JudgmentStatus"))
    if judgments.empty:
        raise ValueError("judgments contain no rows")
    extract = _extract_timestamp(extract_date)
    frame = judgments.reset_index(drop=True)
    judgment = _required_dates(frame, "JudgmentDate")
    if judgment.gt(extract).any():
        raise ValueError(
            f"JudgmentDate is after the extract date for "
            f"{int(judgment.gt(extract).sum())} row(s)"
        )
    status = frame["JudgmentStatus"].astype("string").fillna("").str.strip()
    invalid = ~status.isin(STATUSES)
    if invalid.any():
        raise ValueError(f"JudgmentStatus has {int(invalid.sum())} unsupported row(s)")
    groups = {
        "overall": pd.Series("all", index=frame.index),
        "judgment_quarter": judgment.dt.to_period("Q").astype(str),
    }
    tables = []
    for dimension, strata in groups.items():
        counts = (
            pd.DataFrame({"stratum": strata, "status": status})
            .groupby(["stratum", "status"], observed=True)
            .size()
            .reindex(
                pd.MultiIndex.from_product(
                    [sorted(strata.unique()), STATUSES],
                    names=["stratum", "status"],
                ),
                fill_value=0,
            )
            .rename("rows")
            .reset_index()
        )
        counts["share"] = counts["rows"] / counts.groupby("stratum")[
            "rows"
        ].transform("sum")
        counts.insert(0, "dimension", dimension)
        tables.append(counts)
    result = pd.concat(tables, ignore_index=True)
    result["extract_date"] = extract.date().isoformat()
    result["estimand"] = "status_among_records_present_at_extract"
    return result
