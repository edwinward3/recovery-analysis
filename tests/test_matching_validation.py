"""Tests for the two review samples without judgment outcomes."""

from __future__ import annotations

import pandas as pd
import pytest

from recovery.config import Settings
from recovery.matching import (
    _escape_spreadsheet_formulas,
    accepted_validation_sample,
    unmatched_validation_sample,
)


def test_review_text_is_inert_when_opened_in_a_spreadsheet() -> None:
    raw = pd.DataFrame(
        {
            "source_company_name": ["=1+1", "+cmd", "SAFE LIMITED"],
            "number": [1, 2, 3],
        }
    )

    safe = _escape_spreadsheet_formulas(raw)

    assert safe["source_company_name"].tolist() == [
        "'=1+1",
        "'+cmd",
        "SAFE LIMITED",
    ]
    assert safe["number"].tolist() == [1, 2, 3]


def _sampling_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted_n = 1_004
    unmatched_n = 20
    identifiers = [f"A-{position:04d}" for position in range(accepted_n)] + [
        f"U-{position:04d}" for position in range(unmatched_n)
    ]
    judgments = pd.DataFrame(
        {
            "ID": identifiers,
            "JudgmentDate": [
                pd.Timestamp(2022 + position % 2, 1, 1)
                for position in range(len(identifiers))
            ],
            "JudgmentStatus": ["Satisfied"] * len(identifiers),
            "Defendant Address": [
                f"{position} Example Street" for position in range(len(identifiers))
            ],
        }
    )
    matches = pd.DataFrame(
        {
            "ID": identifiers,
            "tier": ["exact_unique"] * accepted_n + ["unmatched"] * unmatched_n,
            "reason": ["unique_exact_name_postcode_agrees"] * accepted_n
            + ["missing_name", "no_date_valid_unique_exact_name"] * 10,
            "matched_company_number": [
                f"{position:08d}" for position in range(accepted_n)
            ]
            + [""] * unmatched_n,
            "matched_company_name": [
                f"MATCH {position} LIMITED" for position in range(accepted_n)
            ]
            + [""] * unmatched_n,
            "matched_name": [
                f"MATCH {position} LIMITED" for position in range(accepted_n)
            ]
            + [""] * unmatched_n,
            "matched_name_kind": ["current"] * accepted_n + [""] * unmatched_n,
            "matched_on": ["company_name"] * accepted_n + [""] * unmatched_n,
            "matched_company_postcode": ["AA1 1AA"] * accepted_n
            + [""] * unmatched_n,
            "postcode_agrees": [True] * accepted_n + [False] * unmatched_n,
            "source_company_name": [
                f"SOURCE {position} LIMITED" for position in range(len(identifiers))
            ],
            "source_trading_name": [""] * len(identifiers),
            "source_postcode": ["AA1 1AA"] * len(identifiers),
            "CompanyStatus": ["Active"] * len(identifiers),
        }
    )
    return judgments, matches


def test_samples_are_seeded_reproducible_and_outcome_blind() -> None:
    judgments, matches = _sampling_frames()
    settings = Settings()

    accepted = accepted_validation_sample(judgments, matches, settings, seed=71)
    accepted_again = accepted_validation_sample(judgments, matches, settings, seed=71)
    unmatched = unmatched_validation_sample(judgments, matches, 12, seed=72)
    unmatched_again = unmatched_validation_sample(judgments, matches, 12, seed=72)
    unmatched_other_seed = unmatched_validation_sample(
        judgments, matches, 12, seed=73
    )

    pd.testing.assert_frame_equal(accepted, accepted_again)
    pd.testing.assert_frame_equal(unmatched, unmatched_again)
    assert accepted.shape[0] == 1_000
    assert unmatched.shape[0] == 12
    assert unmatched["ID"].tolist() != unmatched_other_seed["ID"].tolist()

    removed = {
        "JudgmentStatus",
        "CompanyStatus",
        "validation_arm",
        "sampling_design",
        "sampling_stratum",
        "sampling_weight",
        "inclusion_probability",
        "reviewer_1_label",
        "reviewer_2_label",
        "adjudicated_label",
    }
    for sample in (accepted, unmatched):
        assert removed.isdisjoint(sample.columns)
        assert "source_address" in sample.columns


def test_unmatched_sample_is_a_census_when_pool_is_smaller_than_target() -> None:
    judgments, matches = _sampling_frames()

    sampled = unmatched_validation_sample(judgments, matches, 1_000, seed=73)

    assert sampled.shape[0] == 20
    assert set(sampled["ID"]) == {f"U-{position:04d}" for position in range(20)}


@pytest.mark.parametrize("accepted_n", [0, 12])
def test_accepted_sample_is_a_census_when_pool_is_smaller_than_target(
    accepted_n: int,
) -> None:
    judgments, matches = _sampling_frames()
    matches = matches.loc[
        matches["ID"].isin({f"A-{position:04d}" for position in range(accepted_n)})
    ].copy()

    sampled = accepted_validation_sample(judgments, matches, Settings(), seed=71)
    sampled_again = accepted_validation_sample(judgments, matches, Settings(), seed=71)

    pd.testing.assert_frame_equal(sampled, sampled_again)
    assert sampled.shape[0] == accepted_n
    assert set(sampled["ID"]) == {
        f"A-{position:04d}" for position in range(accepted_n)
    }
    assert "JudgmentStatus" not in sampled
    assert "CompanyStatus" not in sampled


def test_accepted_sample_allows_a_structured_empty_csv() -> None:
    judgments, matches = _sampling_frames()
    matches = matches.loc[matches["tier"].eq("unmatched")].copy()

    sampled = accepted_validation_sample(judgments, matches, Settings(), seed=71)

    assert sampled.empty
    assert {
        "ID",
        "source_company_name",
        "source_postcode",
        "matched_company_number",
        "tier",
        "match_method",
        "JudgmentDate",
        "source_address",
    }.issubset(sampled.columns)
    assert sampled.to_csv(index=False).strip()


def test_unmatched_sample_allows_an_empty_census() -> None:
    judgments, matches = _sampling_frames()
    matches = matches.loc[matches["tier"].eq("exact_unique")].copy()

    sampled = unmatched_validation_sample(judgments, matches, 1_000, seed=73)
    sampled_again = unmatched_validation_sample(judgments, matches, 1_000, seed=73)

    pd.testing.assert_frame_equal(sampled, sampled_again)
    assert sampled.empty
    assert {
        "ID",
        "tier",
        "reason",
        "source_company_name",
        "source_trading_name",
        "source_postcode",
        "JudgmentDate",
        "source_address",
        "match_method",
    }.issubset(sampled.columns)
    assert "JudgmentStatus" not in sampled
    assert "CompanyStatus" not in sampled
