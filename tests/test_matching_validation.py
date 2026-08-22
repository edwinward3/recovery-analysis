"""Tests for outcome-blind, two-arm linkage validation."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from recovery.config import Settings
from recovery.matching import (
    accepted_validation_sample,
    summarize_linkage_validation,
    unmatched_validation_sample,
    validate_linkage_adjudications,
)


def _sampling_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted_n = 1_004
    unmatched_n = 20
    identifiers = [f"A-{position:04d}" for position in range(accepted_n)] + [
        f"U-{position:04d}" for position in range(unmatched_n)
    ]
    years = [2022 + position % 2 for position in range(accepted_n)] + [
        2022 + position % 2 for position in range(unmatched_n)
    ]
    judgments = pd.DataFrame(
        {
            "ID": identifiers,
            "JudgmentDate": [pd.Timestamp(year, 1, 1) for year in years],
            "JudgmentStatus": ["Satisfied"] * len(identifiers),
            "Defendant Address": [f"{position} Example Street" for position in range(len(identifiers))],
        }
    )
    tiers = ["exact_unique"] * accepted_n + ["unmatched"] * unmatched_n
    matches = pd.DataFrame(
        {
            "ID": identifiers,
            "tier": tiers,
            "reason": ["unique_exact_name_postcode_agrees"] * accepted_n
            + [
                "no_date_valid_unique_exact_name" if (position // 2) % 2 else "missing_name"
                for position in range(unmatched_n)
            ],
            "matched_company_number": [f"{position:08d}" for position in range(accepted_n)]
            + [""] * unmatched_n,
            "matched_company_name": [f"MATCH {position} LIMITED" for position in range(accepted_n)]
            + [""] * unmatched_n,
            "matched_name": [f"MATCH {position} LIMITED" for position in range(accepted_n)]
            + [""] * unmatched_n,
            "matched_name_kind": ["current"] * accepted_n + [""] * unmatched_n,
            "matched_on": ["company_name"] * accepted_n + [""] * unmatched_n,
            "matched_company_postcode": ["AA1 1AA"] * accepted_n + [""] * unmatched_n,
            "postcode_agrees": [True] * accepted_n + [False] * unmatched_n,
            "source_company_name": [f"SOURCE {position} LIMITED" for position in range(len(identifiers))],
            "source_trading_name": [""] * len(identifiers),
            "source_postcode": ["AA1 1AA"] * len(identifiers),
        }
    )
    return judgments, matches


def test_separate_validation_samples_are_deterministic_weighted_and_outcome_blind() -> None:
    judgments, matches = _sampling_frames()
    settings = Settings()

    accepted_first = accepted_validation_sample(judgments, matches, settings, seed=71)
    accepted_second = accepted_validation_sample(judgments, matches, settings, seed=71)
    unmatched_first = unmatched_validation_sample(
        judgments, matches, 12, seed=72
    )
    unmatched_second = unmatched_validation_sample(
        judgments, matches, 12, seed=72
    )

    pd.testing.assert_frame_equal(accepted_first, accepted_second)
    pd.testing.assert_frame_equal(unmatched_first, unmatched_second)
    assert len(accepted_first) == 1_000
    assert len(unmatched_first) == 12
    assert set(unmatched_first["sampling_stratum"]) == {
        "reason=missing_name|judgment_year=2022",
        "reason=missing_name|judgment_year=2023",
        "reason=no_date_valid_unique_exact_name|judgment_year=2022",
        "reason=no_date_valid_unique_exact_name|judgment_year=2023",
    }
    assert unmatched_first.groupby("sampling_stratum").apply(
        lambda group: math.isclose(
            float(group["sampling_weight"].sum()),
            float(group["stratum_population_n"].iloc[0]),
        ),
        include_groups=False,
    ).all()
    for sample in (accepted_first, unmatched_first):
        assert "JudgmentStatus" not in sample
        assert "CompanyStatus" not in sample
        assert sample["reviewer_1_label"].eq("").all()
        assert sample["adjudicated_label"].eq("").all()
        assert (sample["inclusion_probability"] * sample["sampling_weight"]).map(
            lambda value: math.isclose(float(value), 1.0)
        ).all()


def test_unmatched_sample_uses_a_census_when_pool_is_smaller_than_target() -> None:
    judgments, matches = _sampling_frames()

    sampled = unmatched_validation_sample(judgments, matches, 1_000, seed=73)

    assert len(sampled) == 20
    assert sampled["inclusion_probability"].eq(1.0).all()
    assert sampled["sampling_weight"].eq(1.0).all()


def _completed_adjudications() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    accepted_labels = ["correct_match", "correct_match", "correct_match", "incorrect_match"]
    for position, label in enumerate(accepted_labels):
        rows.append(
            {
                "ID": f"A-{position}",
                "validation_arm": "accepted",
                "sampling_stratum": "tier=exact_unique",
                "stratum_population_n": 100,
                "stratum_sample_n": 4,
                "inclusion_probability": 0.04,
                "sampling_weight": 25.0,
                "matched_company_number": f"A{position}",
                "reviewer_1_label": label,
                "reviewer_1_company_number": "",
                "reviewer_2_label": label,
                "reviewer_2_company_number": "",
                "adjudicated_label": label,
                "adjudicated_company_number": "",
                "adjudication_notes": "",
            }
        )
    missed_by_stratum = {"reason=one|judgment_year=2022": 1, "reason=two|judgment_year=2023": 2}
    unmatched_position = 0
    for stratum, missed_n in missed_by_stratum.items():
        for within_stratum in range(4):
            missed = within_stratum < missed_n
            final_label = "missed_match" if missed else "true_unmatched"
            company = f"M{unmatched_position}" if missed else ""
            reviewer_2_label = final_label
            reviewer_2_company = company
            if unmatched_position == 0:
                reviewer_2_label = "true_unmatched"
                reviewer_2_company = ""
            rows.append(
                {
                    "ID": f"U-{unmatched_position}",
                    "validation_arm": "unmatched",
                    "sampling_stratum": stratum,
                    "stratum_population_n": 50,
                    "stratum_sample_n": 4,
                    "inclusion_probability": 0.08,
                    "sampling_weight": 12.5,
                    "matched_company_number": "",
                    "reviewer_1_label": final_label,
                    "reviewer_1_company_number": company,
                    "reviewer_2_label": reviewer_2_label,
                    "reviewer_2_company_number": reviewer_2_company,
                    "adjudicated_label": final_label,
                    "adjudicated_company_number": company,
                    "adjudication_notes": "resolved" if unmatched_position == 0 else "",
                }
            )
            unmatched_position += 1
    return pd.DataFrame(rows)


def test_completed_adjudications_produce_weighted_accuracy_and_gated_recall() -> None:
    adjudications = _completed_adjudications()

    withheld = summarize_linkage_validation(adjudications)
    withheld_estimates = withheld["estimates"].set_index("measure")
    assert withheld_estimates.loc["accepted_match_precision", "estimate"] == pytest.approx(0.75)
    assert withheld_estimates.loc["unmatched_missed_link_prevalence", "estimate"] == pytest.approx(0.375)
    assert withheld_estimates.loc["linkage_recall", "status"] == "not_estimated"
    assert pd.isna(withheld_estimates.loc["linkage_recall", "estimate"])

    supported = summarize_linkage_validation(
        adjudications, recall_denominator_supported=True
    )
    estimates = supported["estimates"].set_index("measure")
    assert estimates.loc["linkage_recall", "estimate"] == pytest.approx(2 / 3)
    assert 0 <= estimates.loc["linkage_recall", "lower_ci"] <= 2 / 3
    assert 2 / 3 <= estimates.loc["linkage_recall", "upper_ci"] <= 1
    agreement = supported["reviewer_agreement"].set_index("validation_arm")
    assert agreement.loc["overall", "label_agreement"] == pytest.approx(11 / 12)
    assert agreement.loc["overall", "decision_agreement"] == pytest.approx(11 / 12)


def test_adjudication_validation_fails_on_incomplete_invalid_or_leaked_data() -> None:
    completed = _completed_adjudications()

    incomplete = completed.copy()
    incomplete.loc[0, "reviewer_1_label"] = ""
    with pytest.raises(ValueError, match="incomplete linkage adjudication"):
        validate_linkage_adjudications(incomplete)

    invalid = completed.copy()
    invalid.loc[0, "reviewer_1_label"] = "maybe"
    with pytest.raises(ValueError, match="invalid reviewer_1_label"):
        validate_linkage_adjudications(invalid)

    missing_company = completed.copy()
    missed_row = missing_company["adjudicated_label"].eq("missed_match").idxmax()
    missing_company.loc[missed_row, "adjudicated_company_number"] = ""
    with pytest.raises(ValueError, match="company_number is required"):
        validate_linkage_adjudications(missing_company)

    leaked = completed.assign(JudgmentStatus="Satisfied")
    with pytest.raises(ValueError, match="must not contain outcome/status"):
        validate_linkage_adjudications(leaked)

    bad_weight = completed.copy()
    bad_weight.loc[0, "sampling_weight"] = 99.0
    with pytest.raises(ValueError, match="invalid or inconsistent"):
        validate_linkage_adjudications(bad_weight)
