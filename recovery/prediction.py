"""Prediction checks for the paper."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .outcomes import (
    LANDMARK_MONTHS,
    PRIMARY_HORIZON_MONTHS,
    RETENTION_MONTHS,
    england_wales_bank_holidays,
    outcome_validity_gate,
)


SEED = 20260618
NO_PRIOR_RECENCY_DAYS = 367.0
MIN_NONTRAIN_CLASS = 100
BOOTSTRAP_REPLICATES = 500
SPLIT_SHARES = (0.50, 0.20, 0.15, 0.15)
MODEL_NAMES = (
    "prevalence",
    "nonlinear_company_age_amount_logistic",
    "penalized_logistic",
    "lightgbm",
)
IMPROVEMENT_COMPARISONS = (
    ("nonlinear_company_age_amount_logistic", "prevalence"),
    ("penalized_logistic", "prevalence"),
    ("lightgbm", "prevalence"),
    ("penalized_logistic", "nonlinear_company_age_amount_logistic"),
    ("lightgbm", "nonlinear_company_age_amount_logistic"),
)
SAFE_FEATURES = frozenset(
    {
        "judgment_amount",
        "company_age_years",
        "registration_delay_calendar_days",
        "registered_by_next_working_day",
        "prior_rt_12m_count",
        "prior_rt_12m_amount",
        "prior_rt_12m_recency_days",
    }
)
CAPACITIES = (0.01, 0.05, 0.10, 0.20)
PRIMARY_ESTIMAND = "recorded_satisfaction_within_12_months_after_one_month_landmark"
CANCELLATION_TREATMENT = "competing_event_reported_separately"
JUDGMENT_AGE_BASELINE = "prevalence_at_constant_one_month_judgment_age"

_GATE_COLUMNS = ("gate_id", "status", "detail", "rows")
_SPLIT_COLUMNS = (
    "split",
    "start_date",
    "end_date",
    "rows",
    "companies",
    "events",
    "non_events",
    "cancellations",
)
_PERFORMANCE_COLUMNS = (
    "model",
    "metric",
    "estimate",
    "ci_low",
    "ci_high",
    "bootstrap_successes",
    "rows",
    "companies",
    "events",
    "non_events",
    "horizon_months",
    "bootstrap_unit",
)
_RANKING_COLUMNS = (
    "model",
    "capacity_fraction",
    "reviewed_rows",
    "events_captured",
    "events_captured_ci_low",
    "events_captured_ci_high",
    "precision",
    "precision_ci_low",
    "precision_ci_high",
    "recall",
    "recall_ci_low",
    "recall_ci_high",
    "lift",
    "lift_ci_low",
    "lift_ci_high",
    "cancellations_captured",
    "cancellations_captured_ci_low",
    "cancellations_captured_ci_high",
    "cancellation_rate",
    "cancellation_rate_ci_low",
    "cancellation_rate_ci_high",
    "suppressed",
    "horizon_months",
)
_IMPROVEMENT_COLUMNS = (
    "model",
    "baseline",
    "metric",
    "improvement",
    "ci_low",
    "ci_high",
    "bootstrap_successes",
    "horizon_months",
    "bootstrap_unit",
)
_CALIBRATION_CURVE_COLUMNS = (
    "model",
    "predicted_probability",
    "observed_probability",
    "effective_rows",
    "smoother",
    "bandwidth",
    "horizon_months",
)


def build_12_month_landmark_cohort(
    judgments: pd.DataFrame,
    matches: pd.DataFrame,
    extract_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Build the exact-linked cohort at the one-month landmark."""

    gate = outcome_validity_gate(judgments, extract_date)
    if gate["design"] != "longitudinal":
        raise ValueError("a valid longitudinal outcome construct is required")
    judgment_columns = {
        "ID",
        "JudgmentDate",
        "Date Inserted",
        "Amount",
        "Satisfaction Date",
        "Cancellation Date",
    }
    match_columns = {"ID", "tier", "matched_company_number", "IncorporationDate"}
    if judgment_columns - set(judgments) or match_columns - set(matches):
        raise ValueError("linked inputs are missing required cohort columns")
    if judgments["ID"].astype(str).duplicated().any() or matches["ID"].astype(str).duplicated().any():
        raise ValueError("linked inputs require unique judgment IDs")

    frame = judgments.loc[:, sorted(judgment_columns)].merge(
        matches.loc[:, sorted(match_columns)], on="ID", how="inner", validate="one_to_one"
    )
    frame = frame.loc[frame["tier"].eq("exact_unique")].reset_index(drop=True)
    if frame.empty:
        result = _empty_landmark_cohort()
        result.attrs["cohort_flow"] = pd.DataFrame(
            [
                {"stage": "exact_linked_start", "rows": 0, "companies": 0},
                {"stage": "mature_landmark_cohort", "rows": 0, "companies": 0},
            ]
        )
        return result
    company = frame["matched_company_number"].astype("string").fillna("").str.strip()
    if company.eq("").any():
        raise ValueError("an accepted exact link is missing its company identifier")
    judgment = _dates(frame["JudgmentDate"])
    inserted = _dates(frame["Date Inserted"])
    incorporation = _dates(frame["IncorporationDate"])
    satisfaction = _dates(frame["Satisfaction Date"])
    cancellation = _dates(frame["Cancellation Date"])
    if judgment.isna().any() or inserted.isna().any() or incorporation.isna().any():
        raise ValueError("linked cohort contains an invalid required date")
    amount = pd.to_numeric(frame["Amount"], errors="coerce")
    if amount.isna().any() or amount.lt(0).any():
        raise ValueError("linked cohort contains an invalid judgment amount")
    extract = pd.Timestamp(extract_date).normalize()
    landmark = judgment + pd.DateOffset(months=LANDMARK_MONTHS)
    cutoff = landmark + pd.DateOffset(months=PRIMARY_HORIZON_MONTHS)
    retention_end = judgment + pd.DateOffset(months=RETENTION_MONTHS)
    exclusions = (
        ("excluded_registered_after_landmark", inserted.gt(landmark)),
        (
            "excluded_satisfied_by_landmark",
            satisfaction.notna() & satisfaction.le(landmark),
        ),
        (
            "excluded_cancelled_by_landmark",
            cancellation.notna() & cancellation.le(landmark),
        ),
        ("excluded_without_12_month_followup", cutoff.gt(extract)),
        ("excluded_beyond_retention_window", cutoff.gt(retention_end)),
    )
    remaining = pd.Series(True, index=frame.index)
    flow = [
        {
            "stage": "exact_linked_start",
            "rows": int(len(frame)),
            "companies": int(company.nunique()),
        }
    ]
    for stage, condition in exclusions:
        excluded = remaining & condition
        flow.append(
            {
                "stage": stage,
                "rows": int(excluded.sum()),
                "companies": int(company.loc[excluded].nunique()),
            }
        )
        remaining &= ~condition
    eligible = remaining
    flow.append(
        {
            "stage": "mature_landmark_cohort",
            "rows": int(eligible.sum()),
            "companies": int(company.loc[eligible].nunique()),
        }
    )

    calendar_delay = (inserted - judgment).dt.days
    holidays = england_wales_bank_holidays()["date"].to_numpy(dtype="datetime64[D]")
    if judgment.min() < pd.Timestamp("2019-01-01") or inserted.max() > pd.Timestamp("2027-12-31"):
        raise ValueError("registration dates fall outside the embedded holiday calendar")
    working_delay = np.busday_count(
        (judgment + pd.Timedelta(days=1)).to_numpy(dtype="datetime64[D]"),
        (inserted + pd.Timedelta(days=1)).to_numpy(dtype="datetime64[D]"),
        holidays=holidays,
    )
    history = pd.DataFrame(
        {
            "company_id": company,
            "judgment": judgment,
            "inserted": inserted,
            "landmark": landmark,
            "amount": amount.astype(float),
        }
    )
    prior_count, prior_amount, prior_recency = _prior_rt_12m(history)
    result = pd.DataFrame(
        {
            "company_id": company,
            "index_date": landmark,
            "feature_as_of_date": landmark,
            "outcome_observed_through": cutoff,
            "satisfied_within_12_months": (
                satisfaction.notna() & satisfaction.gt(landmark) & satisfaction.le(cutoff)
            ).astype(np.int8),
            "cancelled_within_12_months": (
                cancellation.notna() & cancellation.gt(landmark) & cancellation.le(cutoff)
            ).astype(np.int8),
            "judgment_amount": amount.astype(float),
            "company_age_years": (landmark - incorporation).dt.days / 365.25,
            "registration_delay_calendar_days": calendar_delay.astype(float),
            "registered_by_next_working_day": (working_delay <= 1).astype(np.int8),
            "prior_rt_12m_count": prior_count,
            "prior_rt_12m_amount": prior_amount,
            "prior_rt_12m_recency_days": np.nan_to_num(
                prior_recency, nan=NO_PRIOR_RECENCY_DAYS
            ),
        }
    )
    overlap = result["satisfied_within_12_months"].eq(1) & result[
        "cancelled_within_12_months"
    ].eq(1)
    if overlap.any():
        raise ValueError("satisfaction and cancellation outcomes overlap")
    cohort = result.loc[eligible].reset_index(drop=True)
    cohort.attrs["cohort_flow"] = pd.DataFrame(flow)
    return cohort


def _empty_landmark_cohort() -> pd.DataFrame:
    columns = (
        "company_id",
        "index_date",
        "feature_as_of_date",
        "outcome_observed_through",
        "satisfied_within_12_months",
        "cancelled_within_12_months",
        *sorted(SAFE_FEATURES),
    )
    return pd.DataFrame(columns=columns)


def _prior_rt_12m(history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = (
        history.reset_index(names="__row")
        .sort_values(["company_id", "judgment", "__row"], kind="stable")
        .reset_index(drop=True)
    )
    lower = ordered["landmark"] - pd.DateOffset(months=PRIMARY_HORIZON_MONTHS)
    counts = np.zeros(len(ordered), dtype=float)
    amounts = np.zeros(len(ordered), dtype=float)
    recency = np.full(len(ordered), np.nan)
    companies = ordered["company_id"].astype(str).to_numpy()
    boundaries = np.r_[0, np.flatnonzero(companies[1:] != companies[:-1]) + 1, len(ordered)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop - start == 1:
            continue
        dates = ordered.loc[start : stop - 1, "judgment"].to_numpy(dtype="datetime64[ns]")
        inserted = ordered.loc[start : stop - 1, "inserted"].to_numpy(
            dtype="datetime64[ns]"
        )
        low = lower.iloc[start:stop].to_numpy(dtype="datetime64[ns]")
        high = ordered.loc[start : stop - 1, "landmark"].to_numpy(dtype="datetime64[ns]")
        values = ordered.loc[start : stop - 1, "amount"].to_numpy(dtype=float)
        coordinates = np.unique(dates)
        positions = np.searchsorted(coordinates, dates)
        activation_order = np.argsort(inserted, kind="stable")
        focal_order = np.argsort(high, kind="stable")
        active = np.zeros(stop - start, dtype=bool)
        count_tree = np.zeros(len(coordinates) + 1, dtype=np.int64)
        amount_tree = np.zeros(len(coordinates) + 1, dtype=float)

        def add(tree: np.ndarray, position: int, value: float | int) -> None:
            index = position + 1
            while index < len(tree):
                tree[index] += value
                index += index & -index

        def prefix(tree: np.ndarray, end: int) -> float:
            total = 0.0
            index = end
            while index:
                total += tree[index]
                index -= index & -index
            return total

        def last_position(order: int) -> int:
            index = 0
            step = 1 << (len(coordinates).bit_length() - 1)
            remaining = order
            while step:
                candidate = index + step
                if candidate < len(count_tree) and count_tree[candidate] < remaining:
                    index = candidate
                    remaining -= int(count_tree[candidate])
                step >>= 1
            return index

        activation = 0
        for focal in focal_order:
            while (
                activation < len(activation_order)
                and inserted[activation_order[activation]] <= high[focal]
            ):
                candidate = int(activation_order[activation])
                add(count_tree, int(positions[candidate]), 1)
                add(amount_tree, int(positions[candidate]), values[candidate])
                active[candidate] = True
                activation += 1

            left = int(np.searchsorted(coordinates, low[focal], side="left"))
            right = int(np.searchsorted(coordinates, high[focal], side="left"))
            own_available = active[focal] and left <= positions[focal] < right
            if own_available:
                add(count_tree, int(positions[focal]), -1)
                add(amount_tree, int(positions[focal]), -values[focal])

            count = int(prefix(count_tree, right) - prefix(count_tree, left))
            counts[start + focal] = count
            amounts[start + focal] = prefix(amount_tree, right) - prefix(
                amount_tree, left
            )
            if count:
                final = last_position(int(prefix(count_tree, right)))
                recency[start + focal] = (
                    high[focal] - coordinates[final]
                ).astype("timedelta64[D]").astype(float)

            if own_available:
                add(count_tree, int(positions[focal]), 1)
                add(amount_tree, int(positions[focal]), values[focal])
    original = ordered["__row"].to_numpy(dtype=int)
    output_count = np.empty(len(ordered), dtype=float)
    output_amount = np.empty(len(ordered), dtype=float)
    output_recency = np.empty(len(ordered), dtype=float)
    output_count[original] = np.clip(counts, 0, None)
    output_amount[original] = np.clip(amounts, 0, None)
    output_recency[original] = recency
    return output_count, output_amount, output_recency


def run_12_month_prediction(
    cohort: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    company_column: str = "company_id",
    index_date_column: str = "index_date",
    feature_date_column: str = "feature_as_of_date",
    followup_column: str = "outcome_observed_through",
    outcome_column: str = "satisfied_within_12_months",
    cancellation_column: str = "cancelled_within_12_months",
    company_age_column: str = "company_age_years",
    amount_column: str = "judgment_amount",
    min_nontrain_class: int = MIN_NONTRAIN_CLASS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    min_reporting_count: int = 10,
    seed: int = SEED,
) -> dict[str, pd.DataFrame]:
    """Fit the four models and return aggregate tables."""

    tables = _empty_tables()
    gates: list[dict[str, Any]] = []

    if not isinstance(cohort, pd.DataFrame):
        return _failed(tables, gates, "cohort_type", "cohort is not a DataFrame")
    if min_nontrain_class < 1 or bootstrap_replicates < 1 or min_reporting_count < 1:
        return _failed(tables, gates, "settings", "count settings must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        return _failed(tables, gates, "settings", "seed must be an integer")

    features = tuple(feature_columns)
    if not features or len(set(features)) != len(features):
        return _failed(
            tables, gates, "feature_allowlist", "feature list is empty or duplicated"
        )
    if set(features) != SAFE_FEATURES or len(features) != len(SAFE_FEATURES):
        return _failed(
            tables,
            gates,
            "feature_allowlist",
            "requested features do not equal the locked scientific feature set",
        )
    if company_age_column not in features or amount_column not in features:
        return _failed(
            tables,
            gates,
            "simple_baseline_features",
            "company age and amount must both be requested",
        )
    gates.append(_gate("feature_allowlist", "pass", "all requested features allowlisted"))

    required = {
        company_column,
        index_date_column,
        feature_date_column,
        followup_column,
        outcome_column,
        cancellation_column,
        *features,
    }
    missing = sorted(required - set(cohort.columns))
    if missing:
        return _failed(
            tables,
            gates,
            "required_columns",
            f"{len(missing)} required column(s) are absent",
        )
    if cohort.empty:
        return _failed(tables, gates, "cohort_rows", "cohort contains no rows")
    gates.append(_gate("required_columns", "pass", "required columns are present"))

    frame = cohort.loc[:, sorted(required)].copy()
    companies = frame[company_column].astype("string").fillna("").str.strip()
    if companies.eq("").any():
        return _failed(
            tables,
            gates,
            "company_identifier",
            "company identifier is missing",
            int(companies.eq("").sum()),
        )
    frame[company_column] = companies
    gates.append(_gate("company_identifier", "pass", "company identifiers are complete"))

    dates = _dates(frame[index_date_column])
    feature_dates = _dates(frame[feature_date_column])
    followup_dates = _dates(frame[followup_column])
    invalid_dates = dates.isna() | feature_dates.isna() | followup_dates.isna()
    if invalid_dates.any():
        return _failed(
            tables,
            gates,
            "date_fields",
            "one or more required dates are invalid",
            int(invalid_dates.sum()),
        )
    if feature_dates.gt(dates).any():
        return _failed(
            tables,
            gates,
            "point_in_time_features",
            "feature date is after prediction date",
            int(feature_dates.gt(dates).sum()),
        )
    horizon = dates + pd.DateOffset(months=PRIMARY_HORIZON_MONTHS)
    incomplete = followup_dates.lt(horizon)
    if incomplete.any():
        return _failed(
            tables,
            gates,
            "equal_followup",
            f"{PRIMARY_HORIZON_MONTHS} calendar months of outcome follow-up are incomplete",
            int(incomplete.sum()),
        )
    frame[index_date_column] = dates
    gates.extend(
        [
            _gate("date_fields", "pass", "required dates are valid"),
            _gate("point_in_time_features", "pass", "features are dated at or before prediction"),
            _gate(
                "equal_followup",
                "pass",
                f"all labels have {PRIMARY_HORIZON_MONTHS} calendar months follow-up",
            ),
        ]
    )

    outcome = pd.to_numeric(frame[outcome_column], errors="coerce")
    invalid_outcome = outcome.isna() | ~outcome.isin((0, 1))
    if invalid_outcome.any():
        return _failed(
            tables,
            gates,
            "binary_outcome",
            "outcome must be complete and binary",
            int(invalid_outcome.sum()),
        )
    frame[outcome_column] = outcome.astype(np.int8)
    cancellation = pd.to_numeric(frame[cancellation_column], errors="coerce")
    invalid_cancellation = cancellation.isna() | ~cancellation.isin((0, 1))
    overlap = outcome.eq(1) & cancellation.eq(1)
    if invalid_cancellation.any() or overlap.any():
        return _failed(
            tables,
            gates,
            "competing_cancellation",
            "cancellation must be complete, binary, and exclusive of satisfaction",
            int((invalid_cancellation | overlap).sum()),
        )
    frame[cancellation_column] = cancellation.astype(np.int8)
    gates.append(
        _gate("binary_outcome", "pass", "satisfaction and cancellation are exclusive")
    )

    bad_numeric = 0
    for column in features:
        raw = frame[column]
        parsed = pd.to_numeric(raw, errors="coerce")
        populated = raw.notna() & raw.astype("string").fillna("").str.strip().ne("")
        bad_numeric += int((populated & parsed.isna()).sum())
        bad_numeric += int(np.isinf(parsed.to_numpy(dtype=float, na_value=np.nan)).sum())
        frame[column] = parsed.replace([np.inf, -np.inf], np.nan)
    if bad_numeric:
        return _failed(
            tables,
            gates,
            "numeric_features",
            "numeric features contain invalid populated values",
            bad_numeric,
        )
    all_missing = [
        column
        for column in features
        if frame[column].isna().all()
    ]
    if all_missing:
        return _failed(
            tables,
            gates,
            "numeric_features",
            f"{len(all_missing)} numeric feature(s) are entirely missing",
        )
    indicator = frame["registered_by_next_working_day"].dropna()
    if not indicator.isin((0, 1)).all():
        return _failed(
            tables,
            gates,
            "feature_values",
            "next-working-day indicator must be binary",
        )
    gates.append(_gate("feature_values", "pass", "feature values pass type gates"))

    split, _ = _whole_date_split(dates)
    if split is None:
        return _failed(
            tables,
            gates,
            "chronological_split",
            "four nonempty whole-date periods cannot be constructed",
        )
    frame["__split"] = split
    spanning = frame.groupby(company_column, sort=False)["__split"].nunique().gt(1)
    spanning_ids = set(spanning.index[spanning])
    remove = frame[company_column].isin(spanning_ids)
    removed_rows = int(remove.sum())
    frame = frame.loc[~remove].reset_index(drop=True)
    gates.extend(
        [
            _gate("chronological_split", "pass", "50/20/15/15 whole-date boundaries set"),
            _gate(
                "company_boundary_overlap",
                "pass",
                "companies crossing periods were removed",
                len(spanning_ids),
            ),
        ]
    )

    split_summary = _split_summary(
        frame,
        company_column=company_column,
        date_column=index_date_column,
        outcome_column=outcome_column,
        cancellation_column=cancellation_column,
    )
    removed = pd.DataFrame(
        [
            {
                "split": "removed_spanning_boundaries",
                "start_date": "",
                "end_date": "",
                "rows": removed_rows,
                "companies": len(spanning_ids),
                "events": pd.NA,
                "non_events": pd.NA,
                "cancellations": pd.NA,
            }
        ]
    )
    combined_summary = pd.concat([split_summary, removed], ignore_index=True)[
        list(_SPLIT_COLUMNS)
    ]
    for column in ("rows", "companies", "events", "non_events", "cancellations"):
        combined_summary[column] = pd.to_numeric(
            combined_summary[column], errors="coerce"
        ).astype("Int64")
    tables["split_summary"] = combined_summary

    balance_failure = _balance_failure(split_summary, min_nontrain_class)
    if balance_failure:
        return _failed(
            tables,
            gates,
            "class_balance",
            balance_failure,
        )
    gates.append(
        _gate(
            "class_balance",
            "pass",
            f"each nontraining period has at least {min_nontrain_class} events and non-events",
        )
    )

    try:
        from lightgbm import LGBMClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except (ImportError, OSError) as exc:
        return _failed(
            tables,
            gates,
            "model_dependencies",
            f"model dependencies unavailable ({type(exc).__name__})",
        )
    gates.append(_gate("model_dependencies", "pass", "model dependencies available"))

    def preprocessor() -> Any:
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )

    def full_logistic(c_value: float) -> Any:
        return Pipeline(
            [
                ("features", preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        C=c_value,
                        solver="lbfgs",
                        max_iter=2_000,
                        random_state=seed,
                    ),
                ),
            ]
        )

    def lightgbm(n_estimators: int) -> Any:
        return Pipeline(
            [
                ("features", preprocessor()),
                (
                    "model",
                    LGBMClassifier(
                        objective="binary",
                        n_estimators=n_estimators,
                        learning_rate=0.05,
                        num_leaves=15,
                        min_child_samples=40,
                        subsample=1.0,
                        colsample_bytree=1.0,
                        reg_lambda=1.0,
                        random_state=seed,
                        n_jobs=1,
                        deterministic=True,
                        force_col_wise=True,
                        verbosity=-1,
                    ),
                ),
            ]
        )

    masks = {name: frame["__split"].eq(name) for name in _split_names()}
    train = frame.loc[masks["train"]]
    validation = frame.loc[masks["validation"]]
    development = frame.loc[masks["train"] | masks["validation"]]
    calibration = frame.loc[masks["calibration"]]
    test = frame.loc[masks["final_test"]]
    y_train = train[outcome_column].to_numpy(dtype=np.int8)
    y_validation = validation[outcome_column].to_numpy(dtype=np.int8)
    y_development = development[outcome_column].to_numpy(dtype=np.int8)
    y_calibration = calibration[outcome_column].to_numpy(dtype=np.int8)
    y_test = test[outcome_column].to_numpy(dtype=np.int8)
    cancellation_test = test[cancellation_column].to_numpy(dtype=np.int8)

    try:
        simple_train = _simple_features(train, company_age_column, amount_column)
        simple_development = _simple_features(development, company_age_column, amount_column)

        simple = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        max_iter=2_000,
                        random_state=seed,
                    ),
                ),
            ]
        )
        simple.fit(simple_train, y_train)

        best_c = _select_candidate(
            (0.1, 1.0, 10.0),
            lambda value: full_logistic(float(value)),
            train.loc[:, features],
            y_train,
            validation.loc[:, features],
            y_validation,
        )
        best_trees = _select_candidate(
            (100, 200),
            lambda value: lightgbm(int(value)),
            train.loc[:, features],
            y_train,
            validation.loc[:, features],
            y_validation,
        )

        simple.fit(simple_development, y_development)
        full = full_logistic(float(best_c)).fit(
            development.loc[:, features], y_development
        )
        boosted = lightgbm(int(best_trees)).fit(
            development.loc[:, features], y_development
        )
        prevalence = float(y_development.mean())
    except Exception as exc:
        return _failed(
            tables,
            gates,
            "model_fit",
            f"model fitting failed ({type(exc).__name__})",
        )
    gates.extend(
        [
            _gate("model_fit", "pass", "all four candidate models fitted"),
            _gate(
                "penalized_logistic_selection",
                "pass",
                f"C={best_c} selected by validation Brier score",
            ),
            _gate(
                "lightgbm_selection",
                "pass",
                f"n_estimators={best_trees} selected by validation Brier score",
            ),
        ]
    )

    raw_calibration = {
        "prevalence": np.full(len(calibration), prevalence),
        "nonlinear_company_age_amount_logistic": simple.predict_proba(
            _simple_features(calibration, company_age_column, amount_column)
        )[:, 1],
        "penalized_logistic": full.predict_proba(calibration.loc[:, features])[:, 1],
        "lightgbm": boosted.predict_proba(calibration.loc[:, features])[:, 1],
    }
    raw_test = {
        "prevalence": np.full(len(test), prevalence),
        "nonlinear_company_age_amount_logistic": simple.predict_proba(
            _simple_features(test, company_age_column, amount_column)
        )[:, 1],
        "penalized_logistic": full.predict_proba(test.loc[:, features])[:, 1],
        "lightgbm": boosted.predict_proba(test.loc[:, features])[:, 1],
    }
    predictions: dict[str, np.ndarray] = {}
    for name in MODEL_NAMES:
        intercept, slope = _fit_recalibrator(y_calibration, raw_calibration[name])
        predictions[name] = _expit(
            intercept + slope * _logit(raw_test[name])
        )
    gates.append(_gate("calibration", "pass", "calibration used only its held-out period"))

    evaluation = _aggregate_evaluation(
        y_test,
        cancellation_test,
        test[company_column].astype(str).to_numpy(),
        predictions,
        bootstrap_replicates=bootstrap_replicates,
        min_reporting_count=min_reporting_count,
        seed=seed,
    )
    tables.update(evaluation)
    gates.append(
        _gate(
            "final_test",
            "pass",
            "final test evaluated once after fitting and calibration",
            len(test),
        )
    )
    tables["gate"] = pd.DataFrame(gates, columns=_GATE_COLUMNS)
    return tables


def _empty_tables() -> dict[str, pd.DataFrame]:
    return {
        "gate": pd.DataFrame(columns=_GATE_COLUMNS),
        "split_summary": pd.DataFrame(columns=_SPLIT_COLUMNS),
        "performance": pd.DataFrame(columns=_PERFORMANCE_COLUMNS),
        "ranking": pd.DataFrame(columns=_RANKING_COLUMNS),
        "improvement": pd.DataFrame(columns=_IMPROVEMENT_COLUMNS),
        "calibration_curve": pd.DataFrame(columns=_CALIBRATION_CURVE_COLUMNS),
    }


def _gate(gate_id: str, status: str, detail: str, rows: int | None = None) -> dict[str, Any]:
    return {"gate_id": gate_id, "status": status, "detail": detail, "rows": rows}


def _failed(
    tables: dict[str, pd.DataFrame],
    gates: list[dict[str, Any]],
    gate_id: str,
    detail: str,
    rows: int | None = None,
) -> dict[str, pd.DataFrame]:
    gates.append(_gate(gate_id, "fail", detail, rows))
    tables["gate"] = pd.DataFrame(gates, columns=_GATE_COLUMNS)
    return tables


def _dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(
        values,
        format="mixed",
        dayfirst=True,
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None).dt.normalize()


def _split_names() -> tuple[str, ...]:
    return ("train", "validation", "calibration", "final_test")


def _whole_date_split(dates: pd.Series) -> tuple[pd.Series | None, tuple[pd.Timestamp, ...]]:
    counts = dates.value_counts().sort_index()
    if len(counts) < 4:
        return None, ()
    cumulative = counts.cumsum().to_numpy()
    total = int(cumulative[-1])
    chosen: list[int] = []
    previous = -1
    for position, fraction in enumerate(np.cumsum(SPLIT_SHARES)[:-1]):
        low = previous + 1
        high = len(counts) - (4 - position)
        candidates = np.arange(low, high + 1)
        if not len(candidates):
            return None, ()
        distance = np.abs(cumulative[candidates] - fraction * total)
        selected = int(candidates[int(np.argmin(distance))])
        chosen.append(selected)
        previous = selected
    boundaries = tuple(pd.Timestamp(counts.index[index]) for index in chosen)
    if not (boundaries[0] < boundaries[1] < boundaries[2]):
        return None, ()
    split = pd.Series("final_test", index=dates.index, dtype="string")
    split.loc[dates.le(boundaries[2])] = "calibration"
    split.loc[dates.le(boundaries[1])] = "validation"
    split.loc[dates.le(boundaries[0])] = "train"
    return split, boundaries


def _split_summary(
    frame: pd.DataFrame,
    *,
    company_column: str,
    date_column: str,
    outcome_column: str,
    cancellation_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in _split_names():
        part = frame.loc[frame["__split"].eq(name)]
        events = int(part[outcome_column].sum())
        rows.append(
            {
                "split": name,
                "start_date": part[date_column].min().date().isoformat() if len(part) else "",
                "end_date": part[date_column].max().date().isoformat() if len(part) else "",
                "rows": int(len(part)),
                "companies": int(part[company_column].nunique()),
                "events": events,
                "non_events": int(len(part) - events),
                "cancellations": int(part[cancellation_column].sum()),
            }
        )
    return pd.DataFrame(rows, columns=_SPLIT_COLUMNS)


def _balance_failure(summary: pd.DataFrame, minimum: int) -> str | None:
    indexed = summary.set_index("split")
    if (indexed.loc["train", ["events", "non_events"]].astype(int) < 2).any():
        return "training period does not contain both outcome classes"
    nontrain = indexed.loc[list(_split_names()[1:]), ["events", "non_events"]].astype(int)
    if (nontrain < minimum).any().any():
        return "a nontraining period has too few events or non-events"
    return None


def _simple_features(
    frame: pd.DataFrame, company_age_column: str, amount_column: str
) -> np.ndarray:
    age = pd.to_numeric(frame[company_age_column], errors="coerce").to_numpy(dtype=float)
    amount = pd.to_numeric(frame[amount_column], errors="coerce").to_numpy(dtype=float)
    log_amount = np.log1p(np.clip(amount, 0, None))
    return np.column_stack(
        [
            age,
            age**2,
            age**3,
            log_amount,
            log_amount**2,
            log_amount**3,
        ]
    )


def _select_candidate(
    values: Sequence[float | int],
    factory: Any,
    x_train: Any,
    y_train: np.ndarray,
    x_validation: Any,
    y_validation: np.ndarray,
) -> float | int:
    scored: list[tuple[float, float | int]] = []
    for value in values:
        model = factory(value)
        model.fit(x_train, y_train)
        prediction = model.predict_proba(x_validation)[:, 1]
        scored.append((float(np.mean((y_validation - prediction) ** 2)), value))
    return min(scored, key=lambda item: (item[0], values.index(item[1])))[1]


def _aggregate_evaluation(
    outcome: np.ndarray,
    cancellation: np.ndarray,
    companies: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    bootstrap_replicates: int,
    min_reporting_count: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    codes, unique_companies = pd.factorize(companies, sort=True)
    n_companies = len(unique_companies)
    point_weights = np.ones(len(outcome), dtype=float)
    orders = {name: _score_order(prediction) for name, prediction in predictions.items()}
    point_metrics = {
        name: _metrics(outcome, prediction, point_weights, orders[name])
        for name, prediction in predictions.items()
    }
    point_ranking = {
        name: _rankings(outcome, cancellation, point_weights, orders[name])
        for name in predictions
    }
    metric_samples = {
        name: {metric: [] for metric in point_metrics[name]} for name in MODEL_NAMES
    }
    ranking_samples = {
        name: {
            capacity: {metric: [] for metric in point_ranking[name][capacity]}
            for capacity in CAPACITIES
        }
        for name in MODEL_NAMES
    }
    rng = np.random.default_rng(seed + 41)
    for _ in range(bootstrap_replicates):
        draws = rng.integers(0, n_companies, size=n_companies)
        company_weights = np.bincount(draws, minlength=n_companies).astype(float)
        weights = company_weights[codes]
        for name, prediction in predictions.items():
            values = _metrics(outcome, prediction, weights, orders[name])
            for metric, value in values.items():
                metric_samples[name][metric].append(value)
            ranked_all = _rankings(outcome, cancellation, weights, orders[name])
            for capacity, ranked in ranked_all.items():
                for metric, value in ranked.items():
                    ranking_samples[name][capacity][metric].append(value)

    events = int(outcome.sum())
    performance_rows: list[dict[str, Any]] = []
    for name in MODEL_NAMES:
        for metric, estimate in point_metrics[name].items():
            low, high, successful = _interval(metric_samples[name][metric])
            performance_rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_successes": successful,
                    "rows": len(outcome),
                    "companies": n_companies,
                    "events": events,
                    "non_events": len(outcome) - events,
                    "horizon_months": PRIMARY_HORIZON_MONTHS,
                    "bootstrap_unit": "company",
                }
            )

    ranking_rows: list[dict[str, Any]] = []
    for name in MODEL_NAMES:
        for capacity in CAPACITIES:
            point = point_ranking[name][capacity]
            intervals = {
                metric: _interval(ranking_samples[name][capacity][metric])[:2]
                for metric in point
            }
            captured = point["events_captured"]
            reviewed = point["reviewed_rows"]
            selected_non_events = reviewed - captured
            cancellations = point["cancellations_captured"]
            suppress = (
                (0 < captured < min_reporting_count)
                or (0 < cancellations < min_reporting_count)
                or (
                0 < selected_non_events < min_reporting_count
                )
            )
            ranking_rows.append(
                {
                    "model": name,
                    "capacity_fraction": capacity,
                    "reviewed_rows": reviewed,
                    "events_captured": np.nan if suppress else captured,
                    "events_captured_ci_low": np.nan
                    if suppress
                    else intervals["events_captured"][0],
                    "events_captured_ci_high": np.nan
                    if suppress
                    else intervals["events_captured"][1],
                    "precision": np.nan if suppress else point["precision"],
                    "precision_ci_low": np.nan if suppress else intervals["precision"][0],
                    "precision_ci_high": np.nan if suppress else intervals["precision"][1],
                    "recall": np.nan if suppress else point["recall"],
                    "recall_ci_low": np.nan if suppress else intervals["recall"][0],
                    "recall_ci_high": np.nan if suppress else intervals["recall"][1],
                    "lift": np.nan if suppress else point["lift"],
                    "lift_ci_low": np.nan if suppress else intervals["lift"][0],
                    "lift_ci_high": np.nan if suppress else intervals["lift"][1],
                    "cancellations_captured": np.nan if suppress else cancellations,
                    "cancellations_captured_ci_low": np.nan
                    if suppress
                    else intervals["cancellations_captured"][0],
                    "cancellations_captured_ci_high": np.nan
                    if suppress
                    else intervals["cancellations_captured"][1],
                    "cancellation_rate": np.nan if suppress else point["cancellation_rate"],
                    "cancellation_rate_ci_low": np.nan
                    if suppress
                    else intervals["cancellation_rate"][0],
                    "cancellation_rate_ci_high": np.nan
                    if suppress
                    else intervals["cancellation_rate"][1],
                    "suppressed": suppress,
                    "horizon_months": PRIMARY_HORIZON_MONTHS,
                }
            )
    improvement_rows: list[dict[str, Any]] = []
    for name, baseline in IMPROVEMENT_COMPARISONS:
        for metric in ("auc", "brier"):
            direction = 1.0 if metric == "auc" else -1.0
            improvement = direction * (
                point_metrics[name][metric] - point_metrics[baseline][metric]
            )
            paired = direction * (
                np.asarray(metric_samples[name][metric])
                - np.asarray(metric_samples[baseline][metric])
            )
            low, high, successful = _interval(paired)
            improvement_rows.append(
                {
                    "model": name,
                    "baseline": baseline,
                    "metric": metric,
                    "improvement": improvement,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_successes": successful,
                    "horizon_months": PRIMARY_HORIZON_MONTHS,
                    "bootstrap_unit": "company",
                }
            )
    ranking = pd.DataFrame(ranking_rows, columns=_RANKING_COLUMNS)
    ranking["reviewed_rows"] = pd.to_numeric(
        ranking["reviewed_rows"], errors="coerce"
    ).astype("Int64")
    return {
        "performance": pd.DataFrame(performance_rows, columns=_PERFORMANCE_COLUMNS),
        "ranking": ranking,
        "improvement": pd.DataFrame(improvement_rows, columns=_IMPROVEMENT_COLUMNS),
        "calibration_curve": _flexible_calibration_curves(outcome, predictions),
    }


def _score_order(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ascending = np.argsort(prediction, kind="mergesort")
    descending = ascending[::-1]
    ascending_starts = np.r_[
        0, np.flatnonzero(np.diff(prediction[ascending])) + 1
    ]
    descending_starts = np.r_[
        0, np.flatnonzero(np.diff(prediction[descending])) + 1
    ]
    return ascending, ascending_starts, descending, descending_starts


def _metrics(
    outcome: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
    order: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, float]:
    positive = float(np.dot(weights, outcome))
    negative = float(np.dot(weights, 1 - outcome))
    total = positive + negative
    if total <= 0 or positive <= 0 or negative <= 0:
        return {
            "auc": np.nan,
            "brier": np.nan,
            "calibration_intercept": np.nan,
            "calibration_slope": np.nan,
        }
    intercept, slope = _calibration_metrics(outcome, prediction, weights)
    return {
        "auc": _weighted_auc(outcome, weights, order),
        "brier": float(np.dot(weights, (outcome - prediction) ** 2) / total),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def _weighted_auc(
    outcome: np.ndarray,
    weights: np.ndarray,
    score_order: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> float:
    order, starts = score_order[:2]
    positive = np.add.reduceat(weights[order] * outcome[order], starts)
    negative = np.add.reduceat(weights[order] * (1 - outcome[order]), starts)
    total_positive = positive.sum()
    total_negative = negative.sum()
    if total_positive <= 0 or total_negative <= 0:
        return np.nan
    concordant = np.sum(positive * (np.cumsum(negative) - negative + 0.5 * negative))
    return float(concordant / (total_positive * total_negative))


def _rankings(
    outcome: np.ndarray,
    cancellation: np.ndarray,
    weights: np.ndarray,
    score_order: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[float, dict[str, float]]:
    order, starts = score_order[2:]
    group_rows = np.add.reduceat(weights[order], starts)
    group_events = np.add.reduceat(weights[order] * outcome[order], starts)
    group_cancellations = np.add.reduceat(weights[order] * cancellation[order], starts)
    total_rows = float(group_rows.sum())
    total_events = float(group_events.sum())
    cumulative_rows = np.cumsum(group_rows)
    cumulative_events = np.cumsum(group_events)
    cumulative_cancellations = np.cumsum(group_cancellations)
    result: dict[float, dict[str, float]] = {}
    for capacity in CAPACITIES:
        reviewed = max(1.0, float(np.floor(capacity * total_rows)))
        position = min(int(np.searchsorted(cumulative_rows, reviewed)), len(group_rows) - 1)
        prior_rows = float(cumulative_rows[position - 1]) if position else 0.0
        prior_events = float(cumulative_events[position - 1]) if position else 0.0
        prior_cancellations = (
            float(cumulative_cancellations[position - 1]) if position else 0.0
        )
        fraction = (reviewed - prior_rows) / float(group_rows[position])
        captured = prior_events + fraction * float(group_events[position])
        cancelled = prior_cancellations + fraction * float(group_cancellations[position])
        prevalence = total_events / total_rows if total_rows else np.nan
        precision = captured / reviewed if reviewed else np.nan
        result[capacity] = {
            "reviewed_rows": reviewed,
            "events_captured": captured,
            "precision": precision,
            "recall": captured / total_events if total_events else np.nan,
            "lift": precision / prevalence if prevalence else np.nan,
            "cancellations_captured": cancelled,
            "cancellation_rate": cancelled / reviewed if reviewed else np.nan,
        }
    return result


def _flexible_calibration_curves(
    outcome: np.ndarray, predictions: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in MODEL_NAMES:
        prediction = predictions[name]
        log_odds = _logit(prediction)
        spread = float(np.std(log_odds))
        if spread < 1e-10:
            grid = np.array([float(prediction[0])])
            bandwidth = np.nan
        else:
            grid = np.unique(np.quantile(prediction, np.linspace(0.02, 0.98, 25)))
            bandwidth = max(0.20, 1.06 * spread * len(prediction) ** -0.2)
        for probability in grid:
            if np.isnan(bandwidth):
                weight = np.ones(len(outcome))
            else:
                distance = (_logit(np.array([probability]))[0] - log_odds) / bandwidth
                weight = np.exp(-0.5 * distance**2)
            total = float(weight.sum())
            effective = total**2 / float(np.dot(weight, weight))
            rows.append(
                {
                    "model": name,
                    "predicted_probability": probability,
                    "observed_probability": float(np.dot(weight, outcome) / total),
                    "effective_rows": effective,
                    "smoother": "gaussian_kernel_logit_scale",
                    "bandwidth": bandwidth,
                    "horizon_months": PRIMARY_HORIZON_MONTHS,
                }
            )
    return pd.DataFrame(rows, columns=_CALIBRATION_CURVE_COLUMNS)


def _fit_recalibrator(outcome: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    z = _logit(prediction)
    if np.ptp(z) < 1e-10:
        prevalence = float(np.clip(outcome.mean(), 1e-6, 1 - 1e-6))
        return float(_logit(np.array([prevalence]))[0] - z[0]), 1.0
    return _fit_logistic_line(outcome, z, np.ones(len(outcome), dtype=float))


def _calibration_metrics(
    outcome: np.ndarray, prediction: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    z = _logit(prediction)
    prevalence = float(np.dot(weights, outcome) / weights.sum())
    if np.ptp(z[weights > 0]) < 1e-10:
        intercept = float(_logit(np.array([prevalence]))[0] - z[weights > 0][0])
        return intercept, np.nan
    return _fit_logistic_line(outcome, z, weights)


def _fit_logistic_line(
    outcome: np.ndarray, log_odds: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    x = np.column_stack([np.ones(len(log_odds)), log_odds])
    prevalence = float(np.dot(weights, outcome) / weights.sum())
    beta = np.array([_logit(np.array([prevalence]))[0], 0.0], dtype=float)
    for _ in range(30):
        fitted = _expit(x @ beta)
        variance = np.clip(fitted * (1 - fitted), 1e-8, None)
        gradient = x.T @ (weights * (outcome - fitted))
        information = x.T @ (x * (weights * variance)[:, None])
        information.flat[::3] += 1e-10
        try:
            change = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        beta += change
        if np.max(np.abs(change)) < 1e-8:
            break
    return float(beta[0]), float(beta[1])


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def _expit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=float), -35, 35)
    return 1 / (1 + np.exp(-clipped))


def _interval(values: Sequence[float]) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan, 0
    low, high = np.quantile(finite, (0.025, 0.975))
    return float(low), float(high), int(len(finite))
