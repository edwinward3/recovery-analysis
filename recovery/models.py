"""Run 2 only. Builds the company facts, fits the four models and checks the results."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import json
import math
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .config import Settings


# Model splits, features and input fields

SPLIT_ORDER: tuple[str, ...] = ("train", "validation", "calibration", "test")
SPLIT_FRACTIONS: tuple[float, ...] = (0.60, 0.15, 0.10, 0.15)

PROSPECTIVE_FEATURES: tuple[str, ...] = (
    "company_age_at_judgment_years",
    "company_age_at_judgment_missing",
    "log1p_judgment_amount",
    "judgment_amount_missing",
    "prior_judgment_count_24m",
    "prior_judgment_value_24m",
    "days_since_prior_judgment_24m",
    "no_prior_judgment_24m",
)

SNAPSHOT_ADDITIONAL_FEATURES: tuple[str, ...] = (
    "snapshot_any_charges",
    "snapshot_n_charges",
    "snapshot_pct_charges_satisfied",
    "snapshot_accounts_overdue",
    "snapshot_company_status_active",
)

FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "prospective": PROSPECTIVE_FEATURES,
    "snapshot_exploratory": PROSPECTIVE_FEATURES + SNAPSHOT_ADDITIONAL_FEATURES,
}

_REQUIRED_JUDGMENT_COLUMNS = (
    "ID",
    "JudgmentDate",
    "JudgmentStatus",
    "DefendantType",
    "Jurisdiction",
)
_REQUIRED_MATCH_COLUMNS = ("ID", "matched_company_number", "tier")

_INCORPORATION_ALIASES = (
    "incorporation_date",
    "IncorporationDate",
    "ch_incorporation_date",
    "CH_incorporation_date",
)

_SNAPSHOT_ALIASES: dict[str, tuple[str, ...]] = {
    "snapshot_any_charges": ("snapshot_any_charges", "any_charges"),
    "snapshot_n_charges": (
        "snapshot_n_charges",
        "n_charges",
        "n_mort_charges",
        "Mortgages.NumMortCharges",
    ),
    "snapshot_pct_charges_satisfied": (
        "snapshot_pct_charges_satisfied",
        "pct_charges_satisfied",
    ),
    "snapshot_accounts_overdue": (
        "snapshot_accounts_overdue",
        "accounts_overdue",
    ),
    "snapshot_company_status_active": (
        "snapshot_company_status_active",
        "company_status_active",
    ),
}

_MODEL_MATCH_OPTIONAL_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *_INCORPORATION_ALIASES,
            *(alias for aliases in _SNAPSHOT_ALIASES.values() for alias in aliases),
            "Mortgages.NumMortSatisfied",
            "Accounts.NextDueDate",
            "CompanyStatus",
        )
    )
)


class ModelDataError(ValueError):
    """Raised when input data cannot support a defensible model run."""


class LightGBMUnavailableError(RuntimeError):
    """Raised when the required LightGBM package is unavailable."""


# Model data and results

@dataclass(slots=True)
class PreparedCohort:
    """Internal modelling rows plus aggregate construction metadata.

    ``frame`` contains identifiers and must remain inside RT's environment.
    ``to_public_dict`` intentionally exposes counts and schema only.
    """

    frame: pd.DataFrame
    observation_date: pd.Timestamp
    feature_families: dict[str, tuple[str, ...]]
    funnel: dict[str, int]
    split_counts: dict[str, dict[str, int]]
    warnings: list[str] = field(default_factory=list)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "observation_date": self.observation_date.date().isoformat(),
            "feature_families": {
                name: list(columns) for name, columns in self.feature_families.items()
            },
            "funnel": dict(self.funnel),
            "split_counts": self.split_counts,
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class CalibrationResult:
    method: str
    n_positive: int
    n_negative: int
    model: Any = field(default=None, repr=False)

    @property
    def powered(self) -> bool:
        return self.method in {"isotonic", "platt"}

    def predict(self, probabilities: Sequence[float]) -> np.ndarray:
        p = _clip_probabilities(np.asarray(probabilities, dtype=float))
        if self.method == "isotonic":
            return _clip_probabilities(np.asarray(self.model.predict(p), dtype=float))
        if self.method == "platt":
            logits = _logit(p).reshape(-1, 1)
            return _clip_probabilities(self.model.predict_proba(logits)[:, 1])
        return p

    def to_public_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "method": self.method,
            "powered": self.powered,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }
        if self.method == "isotonic":
            out["x_thresholds"] = self.model.X_thresholds_.tolist()
            out["y_thresholds"] = self.model.y_thresholds_.tolist()
        elif self.method == "platt":
            out["coefficient"] = float(self.model.coef_[0, 0])
            out["intercept"] = float(self.model.intercept_[0])
        return out


@dataclass(slots=True)
class FittedNumericModel:
    family: str
    algorithm: str
    feature_names: tuple[str, ...]
    imputer: SimpleImputer
    estimator: Any = field(repr=False)
    scaler: StandardScaler | None = field(default=None, repr=False)
    best_iteration: int | None = None

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.imputer.transform(frame.loc[:, self.feature_names])
        if self.scaler is not None:
            matrix = self.scaler.transform(matrix)
        return _clip_probabilities(self.estimator.predict_proba(matrix)[:, 1])

    def to_model_dict(self) -> dict[str, Any]:
        indicator = getattr(self.imputer, "indicator_", None)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "family": self.family,
            "algorithm": self.algorithm,
            "features": list(self.feature_names),
            "imputer": {
                "strategy": "median",
                "statistics": np.asarray(self.imputer.statistics_, dtype=float).tolist(),
                "indicator_feature_indices": (
                    np.asarray(indicator.features_, dtype=int).tolist()
                    if indicator is not None
                    else []
                ),
            },
            "best_iteration": self.best_iteration,
        }
        if self.scaler is not None:
            payload["scaler"] = {
                "mean": np.asarray(self.scaler.mean_, dtype=float).tolist(),
                "scale": np.asarray(self.scaler.scale_, dtype=float).tolist(),
            }
        if self.algorithm == "logistic":
            payload["estimator"] = {
                "type": "sklearn_logistic_regression",
                "coefficient": np.asarray(self.estimator.coef_, dtype=float).tolist(),
                "intercept": np.asarray(self.estimator.intercept_, dtype=float).tolist(),
                "classes": np.asarray(self.estimator.classes_).tolist(),
                "C": float(self.estimator.C),
            }
        elif self.algorithm == "lightgbm":
            payload["estimator"] = {
                "type": "lightgbm_booster_json",
                "parameters": self.estimator.get_params(),
                "booster": self.estimator.booster_.dump_model(),
            }
        else:  # pragma: no cover - construction prevents this
            raise RuntimeError(f"unknown algorithm: {self.algorithm}")
        return payload


@dataclass(slots=True)
class ModelRun:
    family: str
    algorithm: str
    validation_metrics: dict[str, Any]
    test_metrics_raw: dict[str, Any]
    test_metrics_calibrated: dict[str, Any]
    bootstrap_intervals: dict[str, dict[str, float | None]]
    reliability_bins: list[dict[str, Any]]
    feature_effects: list[dict[str, Any]]
    calibration: CalibrationResult
    model: FittedNumericModel = field(repr=False)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "algorithm": self.algorithm,
            "validation_metrics": self.validation_metrics,
            "test_metrics_raw": self.test_metrics_raw,
            "test_metrics_calibrated": self.test_metrics_calibrated,
            "bootstrap_intervals": self.bootstrap_intervals,
            "reliability_bins": self.reliability_bins,
            "feature_effects": self.feature_effects,
            "calibration": self.calibration.to_public_dict(),
        }


@dataclass(slots=True)
class ModelEvaluation:
    cohort: PreparedCohort
    runs: dict[str, ModelRun]
    champions: dict[str, str]
    primary_acceptance: dict[str, Any]
    training_prevalence: float

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "cohort": self.cohort.to_public_dict(),
            "training_prevalence": self.training_prevalence,
            "champions": dict(self.champions),
            "primary_acceptance": self.primary_acceptance,
            "runs": {
                name: run.to_public_dict() for name, run in sorted(self.runs.items())
            },
        }


# Build the Run 2 sample and date splits

def prepare_model_cohort(
    judgments: pd.DataFrame,
    matches: pd.DataFrame,
    observation_date: str | pd.Timestamp,
    settings: Settings,
) -> PreparedCohort:
    """Construct the locked primary cohort and common chronological split.

    Parameters
    ----------
    judgments:
        Standardised judgment rows. Required columns are ``ID``,
        ``JudgmentDate``, ``JudgmentStatus``, ``DefendantType`` and
        ``Jurisdiction``; ``Amount`` is optional.
    matches:
        One row per ID with ``matched_company_number``, matcher-native ``tier``
        and any Companies House/incorporation/snapshot feature columns.
    observation_date:
        One RT-confirmed extract-level status date. It is never inferred from
        per-row insertion dates.
    settings:
        Validated analysis settings from :mod:`recovery.config`.
    """

    _require_columns(judgments, _REQUIRED_JUDGMENT_COLUMNS, "judgments")
    _require_unique(judgments, "ID", "judgments")
    model_matches = _select_model_matches(matches)
    _require_columns(model_matches, _REQUIRED_MATCH_COLUMNS, "matches")
    _require_unique(model_matches, "ID", "matches")
    raw_tiers = model_matches["tier"].astype("string").str.strip().str.casefold()
    if raw_tiers.isna().any() or raw_tiers.eq("").any():
        raise ModelDataError("matches.tier contains missing values")
    tiers = set(raw_tiers)
    unsupported_tiers = tiers - {"exact_unique", "unmatched"}
    if unsupported_tiers:
        raise ModelDataError(
            f"matches contain unsupported legacy tier(s): {sorted(unsupported_tiers)}"
        )
    exact_rows = raw_tiers.eq("exact_unique")
    company_numbers = model_matches["matched_company_number"].astype("string").str.strip()
    if (exact_rows & (company_numbers.isna() | company_numbers.eq(""))).any():
        raise ModelDataError("exact_unique matches require a company number")
    match_incorporation_col = _first_present(
        model_matches.columns, _INCORPORATION_ALIASES
    )
    if match_incorporation_col is None:
        raise ModelDataError("exact_unique matches require an incorporation date column")
    match_incorporation = pd.to_datetime(
        model_matches[match_incorporation_col], errors="coerce", dayfirst=True
    )
    if (exact_rows & match_incorporation.isna()).any():
        raise ModelDataError("exact_unique matches require a valid incorporation date")

    obs = _as_timestamp(observation_date, "observation_date")
    judgment_columns = [
        *_REQUIRED_JUDGMENT_COLUMNS,
        *(["Amount"] if "Amount" in judgments.columns else []),
    ]
    left = judgments.loc[:, judgment_columns].copy()
    if "Amount" not in left.columns:
        left["Amount"] = np.nan
    left["JudgmentDate"] = pd.to_datetime(
        left["JudgmentDate"], errors="coerce", dayfirst=True
    )
    left["_amount_numeric"] = pd.to_numeric(left["Amount"], errors="coerce")
    left.loc[left["_amount_numeric"] < 0, "_amount_numeric"] = np.nan

    merged = left.merge(model_matches, on="ID", how="left", validate="one_to_one")
    funnel = {
        "judgments_read": int(len(left)),
        "valid_judgment_date": int(merged["JudgmentDate"].notna().sum()),
    }

    corporate = merged["DefendantType"].astype(str).str.strip().str.casefold() == "corporate"
    ew = (
        merged["Jurisdiction"].astype(str).str.strip().str.casefold()
        == "england and wales"
    )
    labelled = merged["JudgmentStatus"].isin(("Satisfied", "Unsatisfied"))
    exact_unique = (
        merged["tier"].astype(str).str.strip().str.casefold() == "exact_unique"
    )
    has_company = merged["matched_company_number"].notna() & (
        merged["matched_company_number"].astype(str).str.strip() != ""
    )

    lower_date = obs - pd.DateOffset(months=settings.primary_max_months)
    upper_date = obs - pd.DateOffset(months=settings.primary_min_months)
    seasoned = merged["JudgmentDate"].between(lower_date, upper_date, inclusive="both")

    funnel.update(
        {
            "corporate": int(corporate.sum()),
            "england_and_wales_corporate": int((corporate & ew).sum()),
            "satisfied_or_unsatisfied_corporate_ew": int((corporate & ew & labelled).sum()),
            "seasoned_12_36_corporate_ew_labelled": int(
                (corporate & ew & labelled & seasoned).sum()
            ),
            "exact_unique_matched_eligible": int(
                (
                    corporate
                    & ew
                    & labelled
                    & seasoned
                    & exact_unique
                    & has_company
                ).sum()
            ),
        }
    )

    eligible = merged.loc[
        corporate & ew & labelled & seasoned & exact_unique & has_company
    ].copy()
    eligible["matched_company_number"] = (
        eligible["matched_company_number"].astype(str).str.strip()
    )
    eligible["label"] = (eligible["JudgmentStatus"] == "Satisfied").astype(int)
    eligible = eligible.sort_values(
        ["JudgmentDate", "matched_company_number", "ID"], kind="mergesort"
    )
    eligible = eligible.drop_duplicates("matched_company_number", keep="first").copy()
    funnel["earliest_eligible_unique_companies"] = int(len(eligible))
    if eligible.empty:
        raise ModelDataError("no rows remain in the primary modelling cohort")

    incorporation_col = _first_present(merged.columns, _INCORPORATION_ALIASES)
    if incorporation_col is None:
        eligible["_incorporation_date"] = pd.NaT
    else:
        eligible["_incorporation_date"] = pd.to_datetime(
            eligible[incorporation_col], errors="coerce", dayfirst=True
        )
    age_days = (eligible["JudgmentDate"] - eligible["_incorporation_date"]).dt.days
    eligible["company_age_at_judgment_years"] = age_days.where(age_days >= 0) / 365.25
    eligible["company_age_at_judgment_missing"] = eligible[
        "company_age_at_judgment_years"
    ].isna().astype(int)

    eligible["judgment_amount_missing"] = eligible["_amount_numeric"].isna().astype(int)
    eligible["log1p_judgment_amount"] = np.log1p(
        eligible["_amount_numeric"].fillna(0.0).clip(lower=0.0)
    )

    history_filter = (
        corporate
        & ew
        & exact_unique
        & has_company
        & merged["JudgmentDate"].notna()
    )
    history = merged.loc[
        history_filter,
        ["matched_company_number", "JudgmentDate", "_amount_numeric"],
    ]
    _add_prior_judgment_features(eligible, history, settings.prior_history_months)
    _add_snapshot_features(eligible, obs)

    eligible = assign_chronological_splits(eligible)
    split_counts = _split_counts(eligible)
    _guard_splits(eligible)

    warnings: list[str] = []
    if incorporation_col is None:
        warnings.append("incorporation date absent; company age is entirely missing")
    if eligible["_amount_numeric"].notna().sum() == 0:
        warnings.append("judgment amount absent; amount features contain missing/zero only")
    missing_snapshot = [
        name for name in SNAPSHOT_ADDITIONAL_FEATURES if eligible[name].notna().sum() == 0
    ]
    if missing_snapshot:
        warnings.append("snapshot features entirely missing: " + ", ".join(missing_snapshot))

    # Keep only the IDs, dates, labels, splits and model features used below.
    keep = [
        "ID",
        "matched_company_number",
        "JudgmentDate",
        "label",
        "split",
        *dict.fromkeys(PROSPECTIVE_FEATURES + SNAPSHOT_ADDITIONAL_FEATURES),
    ]
    return PreparedCohort(
        frame=eligible.loc[:, keep].reset_index(drop=True),
        observation_date=obs,
        feature_families=dict(FEATURE_FAMILIES),
        funnel=funnel,
        split_counts=split_counts,
        warnings=warnings,
    )


def assign_chronological_splits(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign deterministic 60/15/10/15 splits without dividing a date."""

    _require_columns(
        frame,
        ("matched_company_number", "JudgmentDate", "ID"),
        "cohort",
    )
    if frame["matched_company_number"].duplicated().any():
        raise ModelDataError("chronological primary split requires one row per company")
    ordered = frame.sort_values(
        ["JudgmentDate", "matched_company_number", "ID"], kind="mergesort"
    ).copy()
    if ordered["JudgmentDate"].isna().any():
        raise ModelDataError("chronological split cannot use missing judgment dates")

    # Keep each judgment date in one split, so the shares can be approximate.
    group_sizes = ordered.groupby("JudgmentDate", sort=True).size()
    group_count = len(group_sizes)
    if group_count < len(SPLIT_ORDER):
        # Let tiny cohorts reach the clearer empty-split check below.
        date_labels = np.array(SPLIT_ORDER[:group_count], dtype=object)
    else:
        cumulative = group_sizes.cumsum().to_numpy()
        targets = np.asarray(
            [sum(SPLIT_FRACTIONS[: index + 1]) * len(ordered) for index in range(3)]
        )
        boundaries: list[int] = []
        previous = 0
        for index, target in enumerate(targets):
            remaining_splits = len(SPLIT_ORDER) - index - 1
            candidates = range(previous + 1, group_count - remaining_splits + 1)
            boundary = min(
                candidates,
                key=lambda position: (
                    abs(float(cumulative[position - 1]) - target),
                    position,
                ),
            )
            boundaries.append(boundary)
            previous = boundary
        date_labels = np.empty(group_count, dtype=object)
        starts = (0, *boundaries)
        ends = (*boundaries, group_count)
        for split, start, end in zip(SPLIT_ORDER, starts, ends):
            date_labels[start:end] = split
    ordered["split"] = ordered["JudgmentDate"].map(
        dict(zip(group_sizes.index, date_labels))
    )
    return ordered


# Fit and test the four models

def fit_evaluate_models(
    cohort: PreparedCohort,
    settings: Settings,
) -> ModelEvaluation:
    """Fit logistic and LightGBM for both sets of company facts.

    Validation chooses the preferred model before the test rows are opened.
    Both models are then refitted, calibrated and measured on those same rows.
    """

    lgb_module = _require_lightgbm()
    data = cohort.frame
    _guard_splits(data)
    parts = {name: data.loc[data["split"] == name].copy() for name in SPLIT_ORDER}
    if any(part.empty for part in parts.values()):
        empty = [name for name, part in parts.items() if part.empty]
        raise ModelDataError(f"empty modelling split(s): {empty}")
    y_train = parts["train"]["label"].astype(int).to_numpy()
    if np.unique(y_train).size < 2:
        raise ModelDataError("training split contains only one outcome class")
    training_prevalence = float(y_train.mean())

    runs: dict[str, ModelRun] = {}
    champions: dict[str, str] = {}
    for family, features in cohort.feature_families.items():
        validation_candidates: dict[str, tuple[FittedNumericModel, dict[str, Any]]] = {}
        logistic_initial = _fit_logistic(
            family,
            features,
            parts["train"],
            settings.locked_seed,
        )
        validation_candidates["logistic"] = (
            logistic_initial,
            evaluate_predictions(
                parts["validation"]["label"],
                logistic_initial.predict_proba(parts["validation"]),
            ),
        )
        lgb_initial = _fit_lightgbm(
            family,
            features,
            parts["train"],
            settings.locked_seed,
            lgb_module,
            validation=parts["validation"],
        )
        validation_candidates["lightgbm"] = (
            lgb_initial,
            evaluate_predictions(
                parts["validation"]["label"],
                lgb_initial.predict_proba(parts["validation"]),
            ),
        )
        champion = choose_champion(
            validation_candidates["logistic"][1],
            validation_candidates["lightgbm"][1],
        )
        champions[family] = champion

        refit_rows = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
        fitted = {
            "logistic": _fit_logistic(family, features, refit_rows, settings.locked_seed),
            "lightgbm": _fit_lightgbm(
                family,
                features,
                refit_rows,
                settings.locked_seed,
                lgb_module,
                fixed_iterations=lgb_initial.best_iteration,
            ),
        }
        for algorithm, model in fitted.items():
            calibration_raw = model.predict_proba(parts["calibration"])
            calibration = fit_calibrator(
                parts["calibration"]["label"].astype(int).to_numpy(),
                calibration_raw,
                settings,
                seed=settings.locked_seed,
            )
            raw_test = model.predict_proba(parts["test"])
            calibrated_test = calibration.predict(raw_test)
            y_test = parts["test"]["label"].astype(int).to_numpy()
            raw_metrics = evaluate_predictions(y_test, raw_test, training_prevalence)
            calibrated_metrics = evaluate_predictions(
                y_test, calibrated_test, training_prevalence
            )
            intervals = bootstrap_metrics(
                y_test,
                calibrated_test,
                settings.bootstrap_replicates,
                settings.locked_seed,
                training_prevalence=training_prevalence,
            )
            key = f"{family}.{algorithm}"
            runs[key] = ModelRun(
                family=family,
                algorithm=algorithm,
                validation_metrics=validation_candidates[algorithm][1],
                test_metrics_raw=raw_metrics,
                test_metrics_calibrated=calibrated_metrics,
                bootstrap_intervals=intervals,
                reliability_bins=reliability_table(y_test, calibrated_test),
                feature_effects=_feature_effects(model),
                calibration=calibration,
                model=model,
            )

    primary_key = f"prospective.{champions['prospective']}"
    acceptance = assess_acceptance(
        runs[primary_key],
        cohort,
        settings,
        training_prevalence=training_prevalence,
    )
    return ModelEvaluation(
        cohort=cohort,
        runs=runs,
        champions=champions,
        primary_acceptance=acceptance,
        training_prevalence=training_prevalence,
    )


def choose_champion(
    logistic_validation_metrics: Mapping[str, Any],
    lightgbm_validation_metrics: Mapping[str, Any],
) -> str:
    """Return LightGBM only for a >=1% Brier gain without an AUC loss.

    This conservative, predeclared rule makes logistic the simplicity default.
    Missing/non-evaluable validation metrics also resolve to logistic.
    """

    log_brier = _finite_or_none(logistic_validation_metrics.get("brier"))
    lgb_brier = _finite_or_none(lightgbm_validation_metrics.get("brier"))
    log_auc = _finite_or_none(logistic_validation_metrics.get("roc_auc"))
    lgb_auc = _finite_or_none(lightgbm_validation_metrics.get("roc_auc"))
    if None in (log_brier, lgb_brier, log_auc, lgb_auc):
        return "logistic"
    if lgb_brier <= 0.99 * log_brier and lgb_auc >= log_auc:
        return "lightgbm"
    return "logistic"


def fit_calibrator(
    labels: Sequence[int],
    probabilities: Sequence[float],
    settings: Settings,
    *,
    seed: int,
) -> CalibrationResult:
    """Fit isotonic, Platt, or return an explicit underpowered result."""

    y = np.asarray(labels, dtype=int)
    p = _clip_probabilities(np.asarray(probabilities, dtype=float))
    if len(y) != len(p):
        raise ValueError("calibration labels and probabilities differ in length")
    n_positive = int(y.sum())
    n_negative = int(len(y) - n_positive)
    each = min(n_positive, n_negative)
    if each >= settings.isotonic_each_class:
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(p, y)
        return CalibrationResult("isotonic", n_positive, n_negative, model)
    if each >= settings.min_calibration_each_class:
        model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2_000,
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(_logit(p).reshape(-1, 1), y)
        return CalibrationResult("platt", n_positive, n_negative, model)
    return CalibrationResult("underpowered", n_positive, n_negative, None)


def evaluate_predictions(
    labels: Sequence[int] | pd.Series,
    probabilities: Sequence[float],
    training_prevalence: float | None = None,
) -> dict[str, Any]:
    """Compute aggregate discrimination, probability, and top-band metrics."""

    y = np.asarray(labels, dtype=int)
    p = _clip_probabilities(np.asarray(probabilities, dtype=float))
    if len(y) != len(p):
        raise ValueError("labels and probabilities differ in length")
    if len(y) == 0:
        raise ValueError("cannot evaluate an empty prediction set")
    positive = int(y.sum())
    negative = int(len(y) - positive)
    prevalence = float(y.mean())
    two_classes = positive > 0 and negative > 0
    auc = float(roc_auc_score(y, p)) if two_classes else None
    ap = float(average_precision_score(y, p)) if positive > 0 else None
    brier = float(brier_score_loss(y, p))
    ll = float(log_loss(y, p, labels=[0, 1]))
    intercept, slope = _calibration_intercept_slope(y, p)
    top = _top_fraction_metrics(y, p, 0.10)
    result: dict[str, Any] = {
        "n": int(len(y)),
        "n_positive": positive,
        "n_negative": negative,
        "base_rate": prevalence,
        "mean_prediction": float(p.mean()),
        "roc_auc": auc,
        "average_precision": ap,
        "brier": brier,
        "log_loss": ll,
        "calibration_gap": abs(float(p.mean()) - prevalence),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "top_decile": top,
    }
    if training_prevalence is not None:
        q = float(np.clip(training_prevalence, 1e-6, 1 - 1e-6))
        null = np.full(len(y), q, dtype=float)
        null_brier = float(brier_score_loss(y, null))
        null_log_loss = float(log_loss(y, null, labels=[0, 1]))
        result.update(
            {
                "null_training_prevalence": q,
                "null_brier": null_brier,
                "brier_improvement_vs_null": null_brier - brier,
                "null_log_loss": null_log_loss,
                "log_loss_improvement_vs_null": null_log_loss - ll,
            }
        )
    return result


def bootstrap_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    replicates: int,
    seed: int,
    *,
    training_prevalence: float | None = None,
) -> dict[str, dict[str, float | None]]:
    """Return outcome-stratified row-bootstrap percentile intervals."""

    y = np.asarray(labels, dtype=int)
    p = _clip_probabilities(np.asarray(probabilities, dtype=float))
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    names = (
        "roc_auc",
        "average_precision",
        "brier",
        "log_loss",
        "brier_improvement_vs_null",
        "calibration_gap",
        "top_decile_lift",
    )
    if len(pos) == 0 or len(neg) == 0:
        return {name: {"lower": None, "upper": None} for name in names}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(int(replicates)):
        idx = np.concatenate(
            (
                rng.choice(pos, size=len(pos), replace=True),
                rng.choice(neg, size=len(neg), replace=True),
            )
        )
        rng.shuffle(idx)
        sample_y = y[idx]
        sample_p = p[idx]
        sample_base = float(sample_y.mean())
        top = _top_fraction_metrics(sample_y, sample_p, 0.10)
        brier = float(brier_score_loss(sample_y, sample_p))
        if training_prevalence is None:
            brier_lift = None
        else:
            null = np.full(
                len(sample_y),
                float(np.clip(training_prevalence, 1e-6, 1 - 1e-6)),
            )
            brier_lift = float(brier_score_loss(sample_y, null) - brier)
        values = {
            "roc_auc": float(roc_auc_score(sample_y, sample_p)),
            "average_precision": float(average_precision_score(sample_y, sample_p)),
            "brier": brier,
            "log_loss": float(log_loss(sample_y, sample_p, labels=[0, 1])),
            "brier_improvement_vs_null": brier_lift,
            "calibration_gap": abs(float(sample_p.mean()) - sample_base),
            "top_decile_lift": top["lift"],
        }
        for name, value in values.items():
            if value is not None and np.isfinite(value):
                samples[name].append(float(value))
    out: dict[str, dict[str, float | None]] = {}
    for name in names:
        values = samples[name]
        if values:
            lower, upper = np.quantile(values, (0.025, 0.975))
            out[name] = {"lower": float(lower), "upper": float(upper)}
        else:
            out[name] = {"lower": None, "upper": None}
    return out


def reliability_table(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    bins: int = 10,
) -> list[dict[str, Any]]:
    """Return up to ``bins`` deterministic equal-count reliability groups."""

    y = np.asarray(labels, dtype=int)
    p = _clip_probabilities(np.asarray(probabilities, dtype=float))
    if len(y) != len(p):
        raise ValueError("labels and probabilities differ in length")
    if len(y) == 0:
        raise ValueError("cannot build reliability bins for an empty prediction set")
    if bins < 1:
        raise ValueError("bins must be positive")
    order = np.argsort(p, kind="mergesort")
    groups = np.array_split(order, min(bins, len(y)))
    rows: list[dict[str, Any]] = []
    for position, indices in enumerate(groups, start=1):
        if len(indices) == 0:
            continue
        rows.append(
            {
                "bin": position,
                "rows": int(len(indices)),
                "positive": int(y[indices].sum()),
                "negative": int(len(indices) - y[indices].sum()),
                "mean_prediction": float(p[indices].mean()),
                "observed_rate": float(y[indices].mean()),
                "min_prediction": float(p[indices].min()),
                "max_prediction": float(p[indices].max()),
            }
        )
    return rows


def assess_acceptance(
    run: ModelRun,
    cohort: PreparedCohort,
    settings: Settings,
    *,
    training_prevalence: float,
) -> dict[str, Any]:
    """Apply the locked primary acceptance guards to the prospective champion."""

    metrics = run.test_metrics_calibrated
    reasons: list[str] = []
    if run.family != "prospective":
        reasons.append("snapshot_exploratory_only")
    test_rows = _finite_or_none(metrics.get("n"))
    positive = _finite_or_none(metrics.get("n_positive"))
    negative = _finite_or_none(metrics.get("n_negative"))
    if (
        test_rows is None
        or test_rows < 0
        or not test_rows.is_integer()
        or test_rows < settings.min_test_rows
    ):
        reasons.append("test_rows_below_minimum")
    invalid_class_counts = (
        positive is None
        or negative is None
        or positive < 0
        or negative < 0
        or not positive.is_integer()
        or not negative.is_integer()
        or test_rows is None
        or positive + negative != test_rows
    )
    if invalid_class_counts or min(positive, negative) < settings.min_test_each_class:
        reasons.append("test_class_count_below_minimum")
    if not run.calibration.powered:
        reasons.append("calibration_underpowered")
    auc = _finite_or_none(metrics.get("roc_auc"))
    if auc is None or not 0.0 <= auc <= 1.0 or auc < settings.auc_floor:
        reasons.append("auc_below_floor")
    auc_lower = _finite_or_none(
        run.bootstrap_intervals.get("roc_auc", {}).get("lower")
    )
    if auc_lower is None or not 0.0 <= auc_lower <= 1.0 or auc_lower <= 0.50:
        reasons.append("auc_lower_ci_not_above_chance")
    calibration_gap = _finite_or_none(metrics.get("calibration_gap"))
    if (
        calibration_gap is None
        or calibration_gap < 0.0
        or calibration_gap > settings.max_calibration_gap
    ):
        reasons.append("calibration_gap_above_maximum")
    slope = _finite_or_none(metrics.get("calibration_slope"))
    if slope is None or not (
        settings.min_calibration_slope <= slope <= settings.max_calibration_slope
    ):
        reasons.append("calibration_slope_outside_range")
    brier_improvement = _finite_or_none(
        metrics.get("brier_improvement_vs_null")
    )
    if (
        brier_improvement is None
        or not -1.0 <= brier_improvement <= 1.0
        or brier_improvement <= 0
    ):
        reasons.append("brier_not_better_than_training_prevalence")
    return {
        "status": "pass" if not reasons else "fail",
        "passed": not reasons,
        "family": run.family,
        "algorithm": run.algorithm,
        "reasons": reasons,
        "guards": {
            "test_rows_min": settings.min_test_rows,
            "test_each_class_min": settings.min_test_each_class,
            "auc_floor": settings.auc_floor,
            "max_calibration_gap": settings.max_calibration_gap,
            "calibration_slope_range": [
                settings.min_calibration_slope,
                settings.max_calibration_slope,
            ],
            "training_prevalence": training_prevalence,
            "point_in_time_family_only": True,
        },
        "cohort_test_counts": cohort.split_counts.get("test", {}),
    }


# Save the model files

def write_model_artifacts(
    evaluation: ModelEvaluation,
    outdir: str | Path,
) -> dict[str, str]:
    """Write stable aggregate/model JSON and a text summary; never pickle."""

    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    evaluation_path = destination / "model_evaluation.json"
    _write_stable_json(evaluation_path, evaluation.to_public_dict())
    written["evaluation"] = str(evaluation_path)

    for key, run in sorted(evaluation.runs.items()):
        safe_key = key.replace(".", "_")
        model_path = destination / f"model_{safe_key}.json"
        model_payload = run.model.to_model_dict()
        model_payload["calibration"] = run.calibration.to_public_dict()
        model_payload["role"] = (
            "primary_candidate" if run.family == "prospective" else "exploratory_only"
        )
        _write_stable_json(model_path, model_payload)
        written[f"model_{safe_key}"] = str(model_path)

    summary_path = destination / "MODEL_SUMMARY.txt"
    acceptance = evaluation.primary_acceptance
    lines = [
        "RT TWO-MODEL EVALUATION",
        "=======================",
        f"Observation date: {evaluation.cohort.observation_date.date().isoformat()}",
        f"Primary champion: {evaluation.champions.get('prospective', 'n/a')}",
        f"Exploratory snapshot champion: {evaluation.champions.get('snapshot_exploratory', 'n/a')}",
        f"Primary acceptance: {acceptance['status'].upper()}",
        "Acceptance reasons: "
        + (", ".join(acceptance["reasons"]) if acceptance["reasons"] else "none"),
        "",
        "Split counts (rows / satisfied / unsatisfied):",
    ]
    for split in SPLIT_ORDER:
        counts = evaluation.cohort.split_counts.get(split, {})
        lines.append(
            f"  {split:<11} {counts.get('rows', 0)} / "
            f"{counts.get('positive', 0)} / {counts.get('negative', 0)}"
        )
    lines.extend(
        [
            "",
            "The snapshot feature family is retrospective/exploratory and cannot pass acceptance.",
            "No row-level identifiers or predictions are contained in these artefacts.",
        ]
    )
    _write_stable_text(summary_path, "\n".join(lines) + "\n")
    written["summary"] = str(summary_path)
    return written


# Build the model inputs

def _add_prior_judgment_features(
    targets: pd.DataFrame,
    history: pd.DataFrame,
    months: int,
) -> None:
    """Add strictly-prior event features without ever reading prior statuses."""

    history_companies = (
        history["matched_company_number"]
        .astype(str)
        .str.strip()
        .to_numpy(copy=False)
    )
    history_dates = pd.to_datetime(
        history["JudgmentDate"], errors="coerce"
    ).to_numpy(dtype="datetime64[ns]")
    history_amounts = (
        pd.to_numeric(history["_amount_numeric"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=float)
    )
    order = np.lexsort((history_dates, history_companies))
    history_companies = history_companies[order]
    history_dates = history_dates[order]
    history_amounts = history_amounts[order]

    target_count = len(targets)
    counts = np.zeros(target_count, dtype=np.int64)
    values = np.zeros(target_count, dtype=float)
    recencies = np.full(target_count, np.nan, dtype=float)
    if len(history_companies):
        group_starts = np.flatnonzero(
            np.r_[True, history_companies[1:] != history_companies[:-1]]
        )
        group_ends = np.r_[group_starts[1:], len(history_companies)]
        unique_companies = history_companies[group_starts]

        target_companies = (
            targets["matched_company_number"]
            .astype(str)
            .str.strip()
            .to_numpy(copy=False)
        )
        target_dates = pd.to_datetime(
            targets["JudgmentDate"], errors="coerce"
        ).to_numpy(dtype="datetime64[ns]")
        lower_dates = (
            pd.DatetimeIndex(target_dates) - pd.DateOffset(months=months)
        ).to_numpy(dtype="datetime64[ns]")
        group_positions = np.searchsorted(unique_companies, target_companies)
        has_history = group_positions < len(unique_companies)
        possible = np.flatnonzero(has_history)
        has_history[possible] = (
            unique_companies[group_positions[possible]]
            == target_companies[possible]
        )

        for target_position in np.flatnonzero(has_history):
            group_position = group_positions[target_position]
            start = group_starts[group_position]
            end = group_ends[group_position]
            dates = history_dates[start:end]
            left = int(
                np.searchsorted(
                    dates, lower_dates[target_position], side="left"
                )
            )
            right = int(
                np.searchsorted(
                    dates, target_dates[target_position], side="left"
                )
            )  # strictly before the target judgment
            count = right - left
            counts[target_position] = count
            if count:
                values[target_position] = float(
                    history_amounts[start + left : start + right].sum()
                )
                recencies[target_position] = float(
                    (target_dates[target_position] - dates[right - 1])
                    / np.timedelta64(1, "D")
                )

    targets["prior_judgment_count_24m"] = counts
    targets["prior_judgment_value_24m"] = values
    targets["days_since_prior_judgment_24m"] = recencies
    targets["no_prior_judgment_24m"] = (
        targets["prior_judgment_count_24m"] == 0
    ).astype(int)


def _add_snapshot_features(targets: pd.DataFrame, observation_date: pd.Timestamp) -> None:
    for output, aliases in _SNAPSHOT_ALIASES.items():
        source = _first_present(targets.columns, aliases)
        if source is not None:
            targets[output] = pd.to_numeric(targets[source], errors="coerce")
        else:
            targets[output] = np.nan

    # Build the five snapshot features here, after matching.
    raw_n_charges = pd.to_numeric(
        targets.get("Mortgages.NumMortCharges", pd.Series(np.nan, index=targets.index)),
        errors="coerce",
    )
    if targets["snapshot_n_charges"].notna().sum() == 0:
        targets["snapshot_n_charges"] = raw_n_charges
    if targets["snapshot_any_charges"].notna().sum() == 0:
        targets["snapshot_any_charges"] = (raw_n_charges > 0).where(
            raw_n_charges.notna()
        ).astype(float)

    raw_satisfied = pd.to_numeric(
        targets.get("Mortgages.NumMortSatisfied", pd.Series(np.nan, index=targets.index)),
        errors="coerce",
    )
    if targets["snapshot_pct_charges_satisfied"].notna().sum() == 0:
        denominator = raw_n_charges.where(raw_n_charges > 0)
        ratio = (raw_satisfied / denominator).clip(lower=0.0, upper=1.0)
        ratio = ratio.where(raw_n_charges.ne(0), 0.0)
        targets["snapshot_pct_charges_satisfied"] = ratio

    if targets["snapshot_accounts_overdue"].notna().sum() == 0:
        due_raw = targets.get("Accounts.NextDueDate")
        if due_raw is not None:
            due = pd.to_datetime(due_raw, errors="coerce", dayfirst=True)
            targets["snapshot_accounts_overdue"] = (
                (due < observation_date).where(due.notna()).astype(float)
            )

    if targets["snapshot_company_status_active"].notna().sum() == 0:
        status_raw = targets.get("CompanyStatus")
        if status_raw is not None:
            status = status_raw.astype("string").str.strip().str.casefold()
            targets["snapshot_company_status_active"] = (
                status.eq("active").where(status.notna()).astype(float)
            )


def _select_model_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Keep only the current matcher fields needed by modelling."""

    if "match_tier" in matches.columns:
        raise ModelDataError("legacy matches.match_tier is not supported")
    if "tier" not in matches.columns:
        raise ModelDataError("matches is missing required column: tier")
    wanted = [
        "ID",
        "matched_company_number",
        "tier",
        *(
            column
            for column in _MODEL_MATCH_OPTIONAL_COLUMNS
            if column in matches.columns
        ),
    ]
    present = list(dict.fromkeys(column for column in wanted if column in matches.columns))
    return matches.loc[:, present].copy()


def _fit_logistic(
    family: str,
    features: tuple[str, ...],
    train: pd.DataFrame,
    seed: int,
) -> FittedNumericModel:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    matrix = imputer.fit_transform(train.loc[:, features])
    scaler = StandardScaler()
    matrix = scaler.fit_transform(matrix)
    estimator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2_000,
        random_state=seed,
    )
    estimator.fit(matrix, train["label"].astype(int).to_numpy())
    return FittedNumericModel(
        family=family,
        algorithm="logistic",
        feature_names=features,
        imputer=imputer,
        scaler=scaler,
        estimator=estimator,
    )


def _fit_lightgbm(
    family: str,
    features: tuple[str, ...],
    train: pd.DataFrame,
    seed: int,
    module: Any,
    *,
    validation: pd.DataFrame | None = None,
    fixed_iterations: int | None = None,
) -> FittedNumericModel:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    matrix = imputer.fit_transform(train.loc[:, features])
    iterations = int(fixed_iterations or 500)
    estimator = module.LGBMClassifier(
        objective="binary",
        n_estimators=iterations,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=5,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    fit_kwargs: dict[str, Any] = {}
    if validation is not None and fixed_iterations is None:
        validation_matrix = imputer.transform(validation.loc[:, features])
        fit_kwargs["eval_X"] = validation_matrix
        fit_kwargs["eval_y"] = validation["label"].astype(int).to_numpy()
        fit_kwargs["eval_metric"] = "binary_logloss"
        fit_kwargs["callbacks"] = [module.early_stopping(50, verbose=False)]
    estimator.fit(matrix, train["label"].astype(int).to_numpy(), **fit_kwargs)
    best_iteration = getattr(estimator, "best_iteration_", None)
    if not best_iteration:
        best_iteration = iterations
    return FittedNumericModel(
        family=family,
        algorithm="lightgbm",
        feature_names=features,
        imputer=imputer,
        estimator=estimator,
        best_iteration=int(best_iteration),
    )


def _feature_effects(model: FittedNumericModel) -> list[dict[str, Any]]:
    """Return aggregate, named effects without exposing fitted row data."""

    if model.algorithm == "logistic":
        values = np.asarray(model.estimator.coef_[0], dtype=float)
        effect_type = "standardized_log_odds_coefficient"
    else:
        booster = model.estimator.booster_
        if hasattr(booster, "feature_importance"):
            values = np.asarray(
                booster.feature_importance(importance_type="gain"), dtype=float
            )
        else:  # small test doubles need not implement the full Booster surface
            values = np.asarray(
                getattr(
                    model.estimator,
                    "feature_importances_",
                    np.zeros(len(model.feature_names)),
                ),
                dtype=float,
            )
        effect_type = "lightgbm_gain"
    if len(values) != len(model.feature_names):
        raise RuntimeError("model effect count does not match declared feature names")
    total = float(np.abs(values).sum())
    return [
        {
            "feature": feature,
            "effect_type": effect_type,
            "value": float(value),
            "absolute_share": float(abs(value) / total) if total else 0.0,
            "direction": (
                "positive" if value > 0 else "negative" if value < 0 else "zero"
            ) if model.algorithm == "logistic" else "not_directional",
        }
        for feature, value in zip(model.feature_names, values)
    ]


def _require_lightgbm() -> Any:
    try:
        return import_module("lightgbm")
    except (ImportError, OSError) as exc:
        raise LightGBMUnavailableError(
            "LightGBM is required for the declared two-model RT rerun. "
            "Install the pinned lightgbm package and its native runtime "
            "(for example libomp on macOS); no estimator fallback is permitted."
        ) from exc


def _split_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for split in SPLIT_ORDER:
        part = frame.loc[frame["split"] == split]
        positive = int(part["label"].sum())
        out[split] = {
            "rows": int(len(part)),
            "unique_companies": int(part["matched_company_number"].nunique()),
            "positive": positive,
            "negative": int(len(part) - positive),
        }
    return out


def _guard_splits(frame: pd.DataFrame) -> None:
    unknown = set(frame["split"].dropna().astype(str)) - set(SPLIT_ORDER)
    if unknown:
        raise ModelDataError(f"unknown split labels: {sorted(unknown)}")
    memberships = frame.groupby("matched_company_number")["split"].nunique()
    if (memberships > 1).any():
        raise ModelDataError("one or more companies cross modelling splits")
    ordered_max: pd.Timestamp | None = None
    for split in SPLIT_ORDER:
        dates = frame.loc[frame["split"] == split, "JudgmentDate"]
        if dates.empty:
            continue
        current_min = dates.min()
        if ordered_max is not None and current_min < ordered_max:
            raise ModelDataError("modelling splits are not chronological")
        ordered_max = dates.max()


def _top_fraction_metrics(y: np.ndarray, p: np.ndarray, fraction: float) -> dict[str, Any]:
    count = max(1, int(math.ceil(len(y) * fraction)))
    order = np.argsort(-p, kind="mergesort")[:count]
    rate = float(y[order].mean())
    base = float(y.mean())
    lift = rate / base if base > 0 else None
    recall = float(y[order].sum() / y.sum()) if y.sum() > 0 else None
    return {
        "fraction": fraction,
        "selected": count,
        "selected_positive": int(y[order].sum()),
        "selected_negative": int(count - y[order].sum()),
        "positive_rate": rate,
        "lift": lift,
        "recall": recall,
    }


def _calibration_intercept_slope(
    y: np.ndarray, p: np.ndarray
) -> tuple[float | None, float | None]:
    if np.unique(y).size < 2:
        return None, None
    try:
        model = LogisticRegression(
            C=1_000_000.0,
            solver="lbfgs",
            max_iter=2_000,
        )
        with warnings.catch_warnings():
            # Separation can make the slope non-finite; the check below then fails.
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(_logit(p).reshape(-1, 1), y)
        return float(model.intercept_[0]), float(model.coef_[0, 0])
    except Exception:
        return None, None


def _clip_probabilities(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 1e-6, 1 - 1e-6)


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = _clip_probabilities(values)
    return np.log(clipped / (1.0 - clipped))


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ModelDataError(f"{name} missing required column(s): {missing}")


def _require_unique(frame: pd.DataFrame, column: str, name: str) -> None:
    if frame[column].isna().any():
        raise ModelDataError(f"{name}.{column} contains missing values")
    if frame[column].duplicated().any():
        raise ModelDataError(f"{name}.{column} must be unique")


def _as_timestamp(value: str | pd.Timestamp, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise ModelDataError(f"{label} is not a valid date") from exc
    if pd.isna(timestamp):
        raise ModelDataError(f"{label} is not a valid date")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.normalize()


def _first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _write_stable_json(path: Path, payload: Any) -> None:
    text = json.dumps(
        _json_safe(payload),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    _write_stable_text(path, text)


def _write_stable_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is pd.NA:
        return None
    return value


__all__ = [
    "CalibrationResult",
    "FEATURE_FAMILIES",
    "LightGBMUnavailableError",
    "ModelDataError",
    "ModelEvaluation",
    "ModelRun",
    "PROSPECTIVE_FEATURES",
    "PreparedCohort",
    "SNAPSHOT_ADDITIONAL_FEATURES",
    "assign_chronological_splits",
    "assess_acceptance",
    "bootstrap_metrics",
    "choose_champion",
    "evaluate_predictions",
    "fit_calibrator",
    "fit_evaluate_models",
    "reliability_table",
    "prepare_model_cohort",
    "write_model_artifacts",
]
