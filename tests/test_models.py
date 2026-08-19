"""Test cohort timing, four fits, calibration, gates and safe model artefacts.

Inputs are synthetic in-memory rows; temporary outputs contain no RT data.
The tests perform no network or shell operations.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from recovery.config import Settings
from recovery.models import (
    FEATURE_FAMILIES,
    LightGBMUnavailableError,
    ModelDataError,
    PreparedCohort,
    assign_chronological_splits,
    assess_acceptance,
    bootstrap_metrics,
    choose_champion,
    evaluate_predictions,
    fit_calibrator,
    fit_evaluate_models,
    prepare_model_cohort,
    write_model_artifacts,
    _select_model_matches,
)


def _match_row(identifier: str, company: str, tier: str = "auto") -> dict:
    return {
        "ID": identifier,
        "matched_company_number": company,
        "match_tier": tier,
        "incorporation_date": "01/01/2010",
        "any_charges": 0,
        "n_charges": 0,
        "pct_charges_satisfied": 0.0,
        "accounts_overdue": 0,
        "company_status_active": 1,
    }


class CohortTests(TestCase):
    def test_primary_filters_earliest_company_and_strict_prior_history(self) -> None:
        rows = [
            # Two strictly-prior events in the 24-month window. Their statuses
            # are deliberately non-primary; history must not use prior status.
            ("A-boundary", "01/01/2023", "Cancelled", "Corporate", "England and Wales", 10),
            ("A-prior", "01/12/2024", "Cancelled", "Corporate", "England and Wales", 20),
            ("A-target", "01/01/2025", "Satisfied", "Corporate", "England and Wales", 100),
            # Same-day is not strictly prior; later row is dropped because the
            # target is the earliest eligible judgment for company A.
            ("A-same-day", "01/01/2025", "Cancelled", "Corporate", "England and Wales", 30),
            ("A-later", "01/02/2025", "Unsatisfied", "Corporate", "England and Wales", 200),
            ("B-review", "01/01/2025", "Satisfied", "Corporate", "England and Wales", 100),
            ("C-noncorp", "01/01/2025", "Satisfied", "Non-Corporate", "England and Wales", 100),
            ("D-scotland", "01/01/2025", "Satisfied", "Corporate", "Scotland", 100),
            ("E-recent", "01/01/2026", "Satisfied", "Corporate", "England and Wales", 100),
            ("F-old", "01/01/2022", "Satisfied", "Corporate", "England and Wales", 100),
        ]
        judgments = pd.DataFrame(
            rows,
            columns=(
                "ID",
                "JudgmentDate",
                "JudgmentStatus",
                "DefendantType",
                "Jurisdiction",
                "Amount",
            ),
        ).iloc[[2, 4, 0, 9, 1, 6, 3, 5, 7, 8]].reset_index(drop=True)
        matches = pd.DataFrame(
            [
                _match_row(
                    identifier,
                    "A" if identifier.startswith("A-") else identifier,
                    "review" if identifier == "B-review" else "auto",
                )
                for identifier in judgments["ID"]
            ]
        )
        cohort = prepare_model_cohort(
            judgments,
            matches,
            "2026-06-01",
            Settings(),
        )
        self.assertEqual(cohort.frame["ID"].tolist(), ["A-target"])
        target = cohort.frame.iloc[0]
        self.assertEqual(int(target["label"]), 1)
        self.assertEqual(int(target["prior_judgment_count_24m"]), 2)
        self.assertEqual(float(target["prior_judgment_value_24m"]), 30.0)
        self.assertEqual(float(target["days_since_prior_judgment_24m"]), 31.0)
        self.assertEqual(int(target["no_prior_judgment_24m"]), 0)

    def test_matcher_native_tier_and_raw_snapshot_columns_are_supported(self) -> None:
        judgments = pd.DataFrame(
            [
                {
                    "ID": "J1",
                    "JudgmentDate": "01/01/2025",
                    "JudgmentStatus": "Unsatisfied",
                    "DefendantType": "Corporate",
                    "Jurisdiction": "England and Wales",
                    "Amount": 500,
                }
            ]
        )
        matches = pd.DataFrame(
            [
                {
                    "ID": "J1",
                    "matched_company_number": "00000001",
                    "tier": "auto",
                    "IncorporationDate": "01/01/2010",
                    "Mortgages.NumMortCharges": "4",
                    "Mortgages.NumMortSatisfied": "1",
                    "Accounts.NextDueDate": "01/12/2025",
                    "CompanyStatus": "Active",
                }
            ]
        )
        cohort = prepare_model_cohort(judgments, matches, "2026-06-01", Settings())
        row = cohort.frame.iloc[0]
        self.assertEqual(float(row["snapshot_any_charges"]), 1.0)
        self.assertEqual(float(row["snapshot_n_charges"]), 4.0)
        self.assertEqual(float(row["snapshot_pct_charges_satisfied"]), 0.25)
        self.assertEqual(float(row["snapshot_accounts_overdue"]), 1.0)
        self.assertEqual(float(row["snapshot_company_status_active"]), 1.0)

    def test_model_match_selection_is_narrow_and_preserves_tier_alias(self) -> None:
        matches = pd.DataFrame(
            {
                "ID": ["J1"],
                "matched_company_number": ["00000001"],
                "match_tier": [" auto "],
                "IncorporationDate": ["01/01/2010"],
                "Mortgages.NumMortCharges": ["2"],
                "source_company_name": ["MUST NOT ENTER MODEL MERGE"],
                "matched_company_postcode": ["AA1 1AA"],
            }
        )

        selected = _select_model_matches(matches)

        self.assertEqual(
            selected.columns.tolist(),
            [
                "ID",
                "matched_company_number",
                "tier",
                "IncorporationDate",
                "Mortgages.NumMortCharges",
            ],
        )
        self.assertEqual(selected.loc[0, "tier"], " auto ")

    def test_disagreeing_tier_aliases_are_rejected(self) -> None:
        matches = pd.DataFrame(
            {
                "ID": ["J1"],
                "matched_company_number": ["00000001"],
                "tier": ["auto"],
                "match_tier": ["review"],
            }
        )
        with self.assertRaisesRegex(ModelDataError, "disagree"):
            _select_model_matches(matches)

    def test_chronological_split_is_common_and_company_unique(self) -> None:
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        frame = pd.DataFrame(
            {
                "ID": [f"J{i:02d}" for i in range(20)],
                "matched_company_number": [f"C{i:02d}" for i in range(20)],
                "JudgmentDate": dates,
            }
        )
        split = assign_chronological_splits(frame)
        self.assertEqual(split["split"].value_counts().to_dict(), {
            "train": 12,
            "validation": 3,
            "calibration": 2,
            "test": 3,
        })
        positions = {
            name: i
            for i, name in enumerate(("train", "validation", "calibration", "test"))
        }
        encoded = split["split"].map(positions).to_numpy()
        self.assertTrue(np.all(encoded[:-1] <= encoded[1:]))

    def test_chronological_split_keeps_identical_dates_together(self) -> None:
        frame = pd.DataFrame(
            {
                "ID": [f"J{i:02d}" for i in range(16)],
                "matched_company_number": [f"C{i:02d}" for i in range(16)],
                "JudgmentDate": pd.to_datetime(
                    ["2020-01-01"] * 5
                    + ["2020-01-02"] * 4
                    + ["2020-01-03"] * 3
                    + ["2020-01-04"] * 2
                    + ["2020-01-05"] * 2
                ),
            }
        )
        split = assign_chronological_splits(frame)
        self.assertTrue(
            (split.groupby("JudgmentDate")["split"].nunique() == 1).all()
        )


class SelectionAndCalibrationTests(TestCase):
    def test_conservative_champion_rule(self) -> None:
        logistic = {"brier": 0.10, "roc_auc": 0.70}
        self.assertEqual(
            choose_champion(logistic, {"brier": 0.09, "roc_auc": 0.71}),
            "lightgbm",
        )
        self.assertEqual(
            choose_champion(logistic, {"brier": 0.0995, "roc_auc": 0.71}),
            "logistic",
        )
        self.assertEqual(
            choose_champion(logistic, {"brier": 0.09, "roc_auc": 0.69}),
            "logistic",
        )
        self.assertEqual(
            choose_champion(logistic, {"brier": 0.11, "roc_auc": 0.75}),
            "logistic",
        )

    def test_calibration_power_rules(self) -> None:
        settings = Settings(min_calibration_each_class=50, isotonic_each_class=200)
        iso = fit_calibrator(
            np.r_[np.zeros(200), np.ones(200)],
            np.linspace(0.01, 0.99, 400),
            settings,
            seed=1,
        )
        self.assertEqual(iso.method, "isotonic")
        platt = fit_calibrator(
            np.r_[np.zeros(50), np.ones(50)],
            np.linspace(0.01, 0.99, 100),
            settings,
            seed=1,
        )
        self.assertEqual(platt.method, "platt")
        underpowered = fit_calibrator(
            np.r_[np.zeros(49), np.ones(49)],
            np.linspace(0.01, 0.99, 98),
            settings,
            seed=1,
        )
        self.assertEqual(underpowered.method, "underpowered")
        np.testing.assert_allclose(
            underpowered.predict([0.2, 0.8]),
            [0.2, 0.8],
        )

    def test_metrics_and_bootstrap_have_declared_schema(self) -> None:
        y = np.array([0, 1] * 60)
        p = np.where(y == 1, 0.8, 0.2)
        metrics = evaluate_predictions(y, p, training_prevalence=0.5)
        self.assertGreater(metrics["roc_auc"], 0.99)
        self.assertGreater(metrics["brier_improvement_vs_null"], 0)
        intervals = bootstrap_metrics(
            y,
            p,
            100,
            10,
            training_prevalence=0.5,
        )
        self.assertIn("roc_auc", intervals)
        self.assertIn("brier_improvement_vs_null", intervals)
        self.assertGreater(intervals["brier_improvement_vs_null"]["lower"], 0)

    def test_acceptance_uses_primary_test_and_match_audit_guards(self) -> None:
        run = SimpleNamespace(
            family="prospective",
            algorithm="logistic",
            calibration=SimpleNamespace(powered=True),
            test_metrics_calibrated={
                "n": 2_000,
                "n_positive": 200,
                "n_negative": 1_800,
                "roc_auc": 0.75,
                "calibration_gap": 0.01,
                "calibration_slope": 1.0,
                "brier_improvement_vs_null": 0.02,
            },
            bootstrap_intervals={
                "roc_auc": {"lower": 0.65, "upper": 0.82},
                "brier_improvement_vs_null": {"lower": 0.01, "upper": 0.03},
            },
        )
        settings = Settings()
        cohort = _prepared_numeric_cohort(100)
        accepted = assess_acceptance(
            run,
            cohort,
            settings,
            training_prevalence=0.10,
            match_precision=0.99,
            match_precision_lower_ci=0.97,
        )
        self.assertTrue(accepted["passed"])
        rejected = assess_acceptance(
            run,
            cohort,
            settings,
            training_prevalence=0.10,
            match_precision=None,
            match_precision_lower_ci=None,
        )
        self.assertFalse(rejected["passed"])
        self.assertIn("match_audit_not_supplied", rejected["reasons"])

    def test_missing_lightgbm_is_a_clear_failure(self) -> None:
        tiny = _prepared_numeric_cohort(80)
        for dependency_error in (
            ImportError("package absent"),
            OSError("native libomp runtime absent"),
        ):
            with self.subTest(error=type(dependency_error).__name__), patch(
                "recovery.models.import_module",
                side_effect=dependency_error,
            ):
                with self.assertRaisesRegex(
                    LightGBMUnavailableError,
                    "no estimator fallback is permitted",
                ):
                    fit_evaluate_models(tiny, _test_settings())

    def test_single_class_training_split_stops_before_fitting(self) -> None:
        cohort = _prepared_numeric_cohort(100)
        cohort.frame.loc[cohort.frame["split"].eq("train"), "label"] = 0
        with patch("recovery.models._require_lightgbm", return_value=_FakeLightGBMModule):
            with self.assertRaisesRegex(ModelDataError, "only one outcome class"):
                fit_evaluate_models(cohort, _test_settings())


class _FakeBooster:
    def dump_model(self) -> dict:
        return {"tree_info": [], "name": "test-double"}


class _FakeLGBMClassifier:
    def __init__(self, **kwargs) -> None:
        self.params = kwargs
        self.best_iteration_ = kwargs.get("n_estimators", 10)
        self.booster_ = _FakeBooster()
        self._model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    def fit(self, matrix, labels, **kwargs):
        self._model.fit(matrix, labels)
        if kwargs.get("eval_X") is not None:
            self.best_iteration_ = min(10, self.params.get("n_estimators", 10))
        return self

    def predict_proba(self, matrix):
        return self._model.predict_proba(matrix)

    def get_params(self, deep=True):
        return dict(self.params)


class _FakeLightGBMModule:
    LGBMClassifier = _FakeLGBMClassifier

    @staticmethod
    def early_stopping(rounds, verbose=False):
        return SimpleNamespace(rounds=rounds, verbose=verbose)


class EndToEndTests(TestCase):
    def test_two_families_two_algorithms_and_json_only_artifacts(self) -> None:
        cohort = _prepared_numeric_cohort(400)
        settings = _test_settings()
        with warnings.catch_warnings(), patch(
            "recovery.models._require_lightgbm", return_value=_FakeLightGBMModule
        ):
            warnings.simplefilter("ignore", RuntimeWarning)
            evaluation = fit_evaluate_models(
                cohort,
                settings,
                match_precision=0.99,
                match_precision_lower_ci=0.97,
            )
        self.assertEqual(
            set(evaluation.runs),
            {
                "prospective.logistic",
                "prospective.lightgbm",
                "snapshot_exploratory.logistic",
                "snapshot_exploratory.lightgbm",
            },
        )
        self.assertIn(evaluation.champions["prospective"], {"logistic", "lightgbm"})
        with TemporaryDirectory() as directory:
            first = write_model_artifacts(evaluation, directory)
            first_json = Path(first["evaluation"]).read_text(encoding="utf-8")
            second = write_model_artifacts(evaluation, directory)
            second_json = Path(second["evaluation"]).read_text(encoding="utf-8")
            self.assertEqual(first_json, second_json)
            payload = json.loads(first_json)
            self.assertEqual(payload["schema_version"], 1)
            files = [path.name for path in Path(directory).iterdir()]
            self.assertTrue(all(not name.endswith((".pkl", ".pickle")) for name in files))
            self.assertTrue(
                all(
                    path.suffix in {".json", ".txt"}
                    for path in Path(directory).iterdir()
                )
            )


def _test_settings() -> Settings:
    return Settings(
        min_test_rows=20,
        min_test_each_class=5,
        min_calibration_each_class=5,
        isotonic_each_class=20,
        auc_floor=0.40,
        max_calibration_gap=0.50,
        min_calibration_slope=0.0,
        max_calibration_slope=10.0,
        bootstrap_replicates=100,
    )


def _prepared_numeric_cohort(n: int) -> PreparedCohort:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2019-01-01", periods=n, freq="D")
    signal = rng.normal(size=n)
    labels = (signal + rng.normal(scale=0.75, size=n) > 0).astype(int)
    frame = pd.DataFrame(
        {
            "ID": [f"J{i:05d}" for i in range(n)],
            "matched_company_number": [f"C{i:05d}" for i in range(n)],
            "JudgmentDate": dates,
            "label": labels,
            "company_age_at_judgment_years": 5 + signal,
            "company_age_at_judgment_missing": np.zeros(n),
            "log1p_judgment_amount": 6 + 0.25 * signal,
            "judgment_amount_missing": np.zeros(n),
            "prior_judgment_count_24m": np.maximum(0, np.round(signal + 1)),
            "prior_judgment_value_24m": np.maximum(0, 100 * (signal + 1)),
            "days_since_prior_judgment_24m": np.where(signal > -1, 30, np.nan),
            "no_prior_judgment_24m": (signal <= -1).astype(int),
            "snapshot_any_charges": (signal > 0).astype(int),
            "snapshot_n_charges": np.maximum(0, np.round(signal + 1)),
            "snapshot_pct_charges_satisfied": np.clip((signal + 2) / 4, 0, 1),
            "snapshot_accounts_overdue": (signal < -0.5).astype(int),
            "snapshot_company_status_active": (signal > -1.5).astype(int),
        }
    )
    frame = assign_chronological_splits(frame)
    split_counts = {}
    for split in ("train", "validation", "calibration", "test"):
        part = frame.loc[frame["split"] == split]
        positive = int(part["label"].sum())
        split_counts[split] = {
            "rows": len(part),
            "unique_companies": len(part),
            "positive": positive,
            "negative": len(part) - positive,
        }
    return PreparedCohort(
        frame=frame,
        observation_date=pd.Timestamp("2026-06-01"),
        feature_families=dict(FEATURE_FAMILIES),
        funnel={"earliest_eligible_unique_companies": n},
        split_counts=split_counts,
    )


if __name__ == "__main__":
    import unittest

    unittest.main()
