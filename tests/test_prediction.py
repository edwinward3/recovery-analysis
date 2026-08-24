"""Synthetic checks for the aggregate-only prediction branch."""

from __future__ import annotations

import builtins

import numpy as np
import pandas as pd

from recovery.prediction import (
    MODEL_NAMES,
    SAFE_FEATURES,
    build_12_month_landmark_cohort,
    run_12_month_prediction,
)


FEATURES = (
    "judgment_amount",
    "company_age_years",
    "registration_delay_calendar_days",
    "registered_by_next_working_day",
    "prior_rt_12m_count",
    "prior_rt_12m_amount",
    "prior_rt_12m_recency_days",
)


def _cohort(n_dates: int = 80, rows_per_date: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(20260618)
    rows = n_dates * rows_per_date
    position = np.arange(rows)
    dates = pd.Timestamp("2015-01-01") + pd.to_timedelta(
        np.repeat(np.arange(n_dates), rows_per_date) * 14, unit="D"
    )
    age = 1 + position % 48
    amount = np.exp(rng.normal(np.log(2_000), 0.8, rows))
    probability = 1 / (
        1
        + np.exp(
            -(
                -0.6
                + 0.025 * age
                - 0.00005 * amount
                + 0.25 * (position % rows_per_date < rows_per_date / 2)
            )
        )
    )
    outcome = rng.binomial(1, probability)
    # Guarantee both classes on every date without making a feature deterministic.
    for start in range(0, rows, rows_per_date):
        outcome[start] = 0
        outcome[start + 1] = 1
    companies = np.array([f"C{value:07d}" for value in position], dtype=object)
    companies[0] = "SPANNING"
    companies[-1] = "SPANNING"
    return pd.DataFrame(
        {
            "company_id": companies,
            "index_date": dates,
            "feature_as_of_date": dates,
            "outcome_observed_through": dates + pd.DateOffset(months=13),
            "satisfied_within_12_months": outcome,
            "cancelled_within_12_months": ((outcome == 0) & (position % 7 == 0)).astype(int),
            "judgment_amount": amount,
            "company_age_years": 1 + position % 30,
            "registration_delay_calendar_days": position % 5,
            "registered_by_next_working_day": (position % 5 <= 1).astype(int),
            "prior_rt_12m_count": position % 4,
            "prior_rt_12m_amount": (position % 4) * 750.0,
            "prior_rt_12m_recency_days": np.where(position % 4, 20 + position % 200, np.nan),
        }
    )


def test_invalid_cohort_stops_before_lazy_model_imports(monkeypatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn.") or name == "lightgbm":
            raise AssertionError("model dependency imported before the gate")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result = run_12_month_prediction(pd.DataFrame(), feature_columns=FEATURES)

    assert result["gate"].iloc[-1]["status"] == "fail"
    assert result["performance"].empty
    assert result["ranking"].empty


def test_unsafe_feature_is_rejected_without_echoing_values() -> None:
    result = run_12_month_prediction(
        pd.DataFrame(),
        feature_columns=(*FEATURES, "matched_company_number"),
    )

    failure = result["gate"].iloc[-1]
    assert failure["gate_id"] == "feature_allowlist"
    assert failure["status"] == "fail"
    assert "matched_company_number" not in failure["detail"]


def test_nontraining_class_gate_prevents_model_fit() -> None:
    cohort = _cohort()
    cohort["satisfied_within_12_months"] = 0

    result = run_12_month_prediction(
        cohort,
        feature_columns=FEATURES,
        min_nontrain_class=2,
        bootstrap_replicates=2,
    )

    failure = result["gate"].iloc[-1]
    assert failure["gate_id"] == "class_balance"
    assert failure["status"] == "fail"
    assert result["performance"].empty


def test_prediction_outputs_only_deterministic_aggregate_tables() -> None:
    cohort = _cohort()
    arguments = {
        "feature_columns": FEATURES,
        "min_nontrain_class": 10,
        "bootstrap_replicates": 8,
        "min_reporting_count": 1,
    }

    first = run_12_month_prediction(cohort, **arguments)
    second = run_12_month_prediction(cohort, **arguments)

    assert set(FEATURES) == SAFE_FEATURES
    assert set(first) == {
        "gate",
        "split_summary",
        "performance",
        "ranking",
        "improvement",
        "calibration_curve",
    }
    assert all(isinstance(table, pd.DataFrame) for table in first.values())
    assert set(first["gate"]["status"]) == {"pass"}
    assert set(first["performance"]["model"]) == set(MODEL_NAMES)
    assert set(first["performance"]["metric"]) == {
        "auc",
        "brier",
        "calibration_intercept",
        "calibration_slope",
    }
    assert set(first["ranking"]["capacity_fraction"]) == {0.01, 0.05, 0.10, 0.20}
    assert len(first["performance"]) == 16
    assert len(first["ranking"]) == 16
    assert len(first["improvement"]) == 10
    assert set(first["improvement"]["baseline"]) == {
        "prevalence",
        "nonlinear_company_age_amount_logistic",
    }
    assert not first["calibration_curve"].empty
    assert "cancellation_rate" in first["ranking"]
    assert not any("company_id" in table.columns for table in first.values())

    splits = first["split_summary"].set_index("split")
    assert splits.loc["removed_spanning_boundaries", "companies"] == 1
    assert splits.loc["removed_spanning_boundaries", "rows"] == 2
    assert list(splits.loc[["train", "validation", "calibration", "final_test"], "start_date"]) == sorted(
        splits.loc[["train", "validation", "calibration", "final_test"], "start_date"]
    )
    prevalence_auc = first["performance"].loc[
        lambda table: table["model"].eq("prevalence") & table["metric"].eq("auc"),
        "estimate",
    ].iloc[0]
    assert prevalence_auc == 0.5

    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_landmark_builder_derives_only_locked_point_in_time_features() -> None:
    judgments = pd.DataFrame(
        {
            "ID": ["J1", "J2", "J3"],
            "JudgmentDate": pd.to_datetime(["2020-01-01", "2020-06-01", "2020-03-01"]),
            "Date Inserted": pd.to_datetime(["2020-01-02", "2020-06-02", "2020-03-02"]),
            "JudgmentStatus": ["Unsatisfied", "Satisfied", "Cancelled"],
            "Amount": [100.0, 300.0, 500.0],
            "Satisfaction Date": [pd.NaT, pd.Timestamp("2021-01-01"), pd.NaT],
            "Cancellation Date": [pd.NaT, pd.NaT, pd.Timestamp("2020-10-01")],
        }
    )
    matches = pd.DataFrame(
        {
            "ID": ["J1", "J2", "J3"],
            "tier": "exact_unique",
            "matched_company_number": ["00000001", "00000001", "00000002"],
            "IncorporationDate": pd.to_datetime(
                ["2010-01-01", "2010-01-01", "2012-01-01"]
            ),
        }
    )

    cohort = build_12_month_landmark_cohort(judgments, matches, "2022-12-31")

    metadata = {
        "company_id",
        "index_date",
        "feature_as_of_date",
        "outcome_observed_through",
        "satisfied_within_12_months",
        "cancelled_within_12_months",
    }
    assert set(cohort) == metadata | SAFE_FEATURES
    assert cohort.loc[1, "prior_rt_12m_count"] == 1
    assert cohort.loc[1, "prior_rt_12m_amount"] == 100.0
    assert cohort.loc[1, "satisfied_within_12_months"] == 1
    assert cohort.loc[2, "cancelled_within_12_months"] == 1
    flow = cohort.attrs["cohort_flow"].set_index("stage")
    assert flow.loc["exact_linked_start", "rows"] == 3
    assert flow.loc["mature_landmark_cohort", "rows"] == 3
    assert flow.loc[
        [
            "excluded_registered_after_landmark",
            "excluded_satisfied_by_landmark",
            "excluded_cancelled_by_landmark",
            "excluded_without_12_month_followup",
            "excluded_beyond_retention_window",
        ],
        "rows",
    ].sum() == 0


def test_prior_history_excludes_judgments_not_registered_by_focal_landmark() -> None:
    judgments = pd.DataFrame(
        {
            "ID": ["J1", "J2"],
            "JudgmentDate": pd.to_datetime(["2020-01-01", "2020-01-15"]),
            "Date Inserted": pd.to_datetime(["2020-01-02", "2020-03-01"]),
            "JudgmentStatus": ["Unsatisfied", "Unsatisfied"],
            "Amount": [100.0, 900.0],
            "Satisfaction Date": [pd.NaT, pd.NaT],
            "Cancellation Date": [pd.NaT, pd.NaT],
        }
    )
    judgments.attrs["raw_header_schema"] = (
        ("Satisfaction Date", "Satisfaction Date"),
        ("Cancellation Date", "Cancellation Date"),
    )
    matches = pd.DataFrame(
        {
            "ID": ["J1", "J2"],
            "tier": ["exact_unique", "exact_unique"],
            "matched_company_number": ["00000001", "00000001"],
            "IncorporationDate": pd.to_datetime(["2010-01-01", "2010-01-01"]),
        }
    )

    cohort = build_12_month_landmark_cohort(judgments, matches, "2022-12-31")

    assert len(cohort) == 1
    assert cohort.loc[0, "prior_rt_12m_count"] == 0
    assert cohort.loc[0, "prior_rt_12m_amount"] == 0
    assert cohort.loc[0, "prior_rt_12m_recency_days"] == 367
