"""Tests for the status-at-extract model design."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from recovery.config import Settings
from recovery.models import (
    CAPACITY_FRACTIONS,
    assign_company_group_splits,
    bootstrap_metrics,
    develop_models,
    evaluate_locked_models,
    evaluate_predictions,
    materialize_locked_test_outcomes,
    paired_cluster_bootstrap_metrics,
    prepare_model_cohort,
)


def _source_rows(company_count: int = 240) -> tuple[pd.DataFrame, pd.DataFrame]:
    observation = pd.Timestamp("2026-06-01")
    judgments: list[dict] = []
    matches: list[dict] = []
    for company_position in range(company_count):
        company = f"C{company_position:05d}"
        age = 2 + company_position % 46
        for repeat in range(2 if company_position % 7 == 0 else 1):
            identifier = f"J{company_position:05d}-{repeat}"
            date = observation - pd.DateOffset(months=age) - pd.DateOffset(days=repeat)
            judgments.append(
                {
                    "ID": identifier,
                    "JudgmentDate": date,
                    "JudgmentStatus": (
                        "Satisfied" if (company_position + repeat) % 3 == 0 else "Unsatisfied"
                    ),
                    "DefendantType": "Corporate",
                    "Jurisdiction": "England and Wales",
                    "Amount": 100 + company_position,
                }
            )
            matches.append(
                {
                    "ID": identifier,
                    "matched_company_number": company,
                    "tier": "exact_unique",
                    "incorporation_date": "2010-01-01",
                    "n_charges": company_position % 4,
                    "accounts_overdue": company_position % 2,
                }
            )
    return pd.DataFrame(judgments), pd.DataFrame(matches)


class CohortValidityTests(TestCase):
    def test_all_eligible_repeats_are_retained_and_test_labels_are_masked(self) -> None:
        judgments, matches = _source_rows(120)
        cohort = prepare_model_cohort(
            judgments, matches, "2026-06-01", Settings()
        )

        expected = len(judgments)
        self.assertEqual(len(cohort.frame), expected)
        self.assertGreater(cohort.funnel["eligible_repeat_judgments"], 0)
        self.assertTrue(
            cohort.frame.groupby("matched_company_number")["split"].nunique().eq(1).all()
        )
        self.assertTrue(cohort.frame.loc[cohort.frame["split"].eq("test"), "label"].isna().all())
        self.assertTrue(cohort.frame.loc[~cohort.frame["split"].eq("test"), "label"].notna().all())
        self.assertIn("observation_age_hinge_24m_cubed", cohort.frame)
        self.assertIn(
            "observable_retained_prior_judgment_count_24m", cohort.frame
        )
        self.assertNotIn("prior_judgment_count_24m", cohort.frame)
        for counts in cohort.split_counts.values():
            self.assertNotIn("positive", counts)
            self.assertNotIn("negative", counts)

    def test_one_month_boundary_is_strict_and_48_month_boundary_is_inclusive(self) -> None:
        observation = pd.Timestamp("2026-06-01")
        dates = [
            observation - pd.DateOffset(months=1),
            observation - pd.DateOffset(months=1) - pd.DateOffset(days=1),
            observation - pd.DateOffset(months=48),
            observation - pd.DateOffset(months=48) - pd.DateOffset(days=1),
        ]
        judgments = pd.DataFrame(
            {
                "ID": ["one", "post-one", "forty-eight", "too-old"],
                "JudgmentDate": dates,
                "JudgmentStatus": ["Satisfied", "Unsatisfied", "Satisfied", "Unsatisfied"],
                "DefendantType": ["Corporate"] * 4,
                "Jurisdiction": ["England and Wales"] * 4,
                "Amount": [100] * 4,
            }
        )
        matches = pd.DataFrame(
            {
                "ID": judgments["ID"],
                "matched_company_number": ["C1", "C2", "C3", "C4"],
                "tier": ["exact_unique"] * 4,
                "incorporation_date": ["2010-01-01"] * 4,
            }
        )
        cohort = prepare_model_cohort(judgments, matches, observation, Settings())
        self.assertEqual(set(cohort.frame["ID"]), {"post-one", "forty-eight"})

    def test_group_partition_is_label_blind_and_order_invariant(self) -> None:
        frame = pd.DataFrame(
            {
                "ID": [f"J{i:03d}" for i in range(200)],
                "matched_company_number": [f"C{i // 2:03d}" for i in range(200)],
                "observation_age_months": [2 + (i // 2) % 46 for i in range(200)],
                "label": [i % 2 for i in range(200)],
            }
        )
        first = assign_company_group_splits(frame, 99)
        changed = frame.sample(frac=1, random_state=4).copy()
        changed["label"] = 1 - changed["label"]
        second = assign_company_group_splits(changed, 99)
        first_map = first.set_index("ID")["split"].sort_index()
        second_map = second.set_index("ID")["split"].sort_index()
        pd.testing.assert_series_equal(first_map, second_map)
        self.assertTrue(
            first.groupby("matched_company_number")["split"].nunique().eq(1).all()
        )


class MetricValidityTests(TestCase):
    def test_multiple_capacities_and_clustered_intervals(self) -> None:
        labels = np.array([0, 1] * 60)
        probabilities = np.where(labels == 1, 0.8, 0.2)
        clusters = np.array([f"C{i // 2:03d}" for i in range(len(labels))])
        metrics = evaluate_predictions(labels, probabilities, training_prevalence=0.5)
        self.assertEqual(
            set(metrics["capacity_metrics"]),
            {f"{int(value * 100)}pct" for value in CAPACITY_FRACTIONS},
        )
        intervals = bootstrap_metrics(
            labels,
            probabilities,
            100,
            7,
            training_prevalence=0.5,
            clusters=clusters,
        )
        self.assertIn("capacity_1pct_lift", intervals)
        paired = paired_cluster_bootstrap_metrics(
            labels, probabilities, np.full(len(labels), 0.5), clusters, 100, 7
        )
        self.assertGreater(paired["delta_roc_auc"]["lower"], 0)


class _FakeBooster:
    def dump_model(self):
        return {"tree_info": []}


class _FakeClassifier:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.best_iteration_ = min(10, kwargs.get("n_estimators", 10))
        self.booster_ = _FakeBooster()

    def fit(self, matrix, labels, **kwargs):
        self.model = LogisticRegression(max_iter=1000).fit(matrix, labels)
        self.feature_importances_ = np.ones(matrix.shape[1])
        return self

    def predict_proba(self, matrix):
        return self.model.predict_proba(matrix)

    def get_params(self, deep=True):
        return dict(self.params)


class _FakeLightGBM:
    LGBMClassifier = _FakeClassifier

    @staticmethod
    def early_stopping(rounds, verbose=False):
        return SimpleNamespace(rounds=rounds, verbose=verbose)


class LockedEvaluationTests(TestCase):
    def test_development_and_locked_evaluation_are_separate(self) -> None:
        judgments, matches = _source_rows()
        settings = Settings(
            min_test_rows=10,
            min_test_each_class=2,
            min_calibration_each_class=2,
            isotonic_each_class=10,
            auc_floor=0.0,
            max_calibration_gap=1.0,
            min_calibration_slope=-100.0,
            max_calibration_slope=100.0,
            bootstrap_replicates=100,
        )
        cohort = prepare_model_cohort(judgments, matches, "2026-06-01", settings)
        with patch("recovery.models._require_lightgbm", return_value=_FakeLightGBM):
            development = develop_models(cohort, settings)
        self.assertFalse(development.to_public_dict()["test_outcomes_accessed"])
        self.assertEqual(
            development.frozen_evaluation_keys[0], "age_only.logistic"
        )
        self.assertNotIn(
            "snapshot_exploratory." + development.champions["snapshot_exploratory"],
            development.frozen_evaluation_keys,
        )
        outcomes = materialize_locked_test_outcomes(judgments, cohort)
        evaluation = evaluate_locked_models(development, outcomes, settings)
        self.assertEqual(set(evaluation.runs), set(development.frozen_evaluation_keys))
        self.assertTrue(all(not key.startswith("snapshot") for key in evaluation.runs))
        primary_key = development.frozen_evaluation_keys[-1]
        self.assertIn(
            "paired_company_clustered_intervals",
            evaluation.runs[primary_key].comparison_to_age_only,
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
