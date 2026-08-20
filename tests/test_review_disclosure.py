"""Test manual-review quality gates and the aggregate-only egress boundary.

Inputs are planted synthetic identifiers and review decisions. Outputs are
temporary aggregate files used to prove that confidential rows cannot egress.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from recovery.config import Settings
from recovery.disclosure import (
    DisclosureViolation,
    scan_identifiers,
    stage_egress,
    suppress_small_cells,
    validate_egress,
)
from recovery.review import (
    ReviewFormatError,
    parse_completed_review,
    wilson_interval,
    write_review_aggregates,
)


def _completed_review(settings: Settings, *, auto_incorrect: int = 0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    allocation = {
        "auto": settings.sample_auto,
        "review": settings.sample_review,
        "fallback_review": settings.sample_fallback,
    }
    row_id = 0
    for tier, count in allocation.items():
        for position in range(count):
            decision = "incorrect" if tier == "auto" and position < auto_incorrect else "correct"
            rows.append(
                {
                    "review_row_id": f"R{row_id:04d}",
                    "review_tier": tier,
                    "review_decision": decision,
                    "source_company_name": f"RT PRIVATE ROW {row_id} LIMITED",
                }
            )
            row_id += 1
    return pd.DataFrame(rows)


class ReviewTests(unittest.TestCase):
    def test_completed_review_wilson_stats_and_gate(self) -> None:
        settings = Settings()
        result = parse_completed_review(_completed_review(settings), settings)
        self.assertTrue(result.gate_passed)
        self.assertEqual(result.total_rows, 1_000)
        auto = result.stats.loc[result.stats["tier"].eq("auto")].iloc[0]
        expected_low, expected_high = wilson_interval(500, 500)
        self.assertAlmostEqual(float(auto["wilson_lower_95"]), expected_low)
        self.assertAlmostEqual(float(auto["wilson_upper_95"]), expected_high)
        self.assertGreater(float(auto["wilson_lower_95"]), settings.match_precision_lower_ci_floor)

    def test_uncertain_decision_counts_as_incorrect(self) -> None:
        settings = Settings()
        review = _completed_review(settings)
        review.loc[0, "review_decision"] = " Uncertain "
        result = parse_completed_review(review, settings)
        self.assertTrue(result.gate_passed)
        self.assertEqual(result.uncertain_rows, 1)
        auto = result.stats.loc[result.stats["tier"].eq("auto")].iloc[0]
        self.assertEqual(int(auto["n_uncertain"]), 1)
        self.assertAlmostEqual(float(auto["observed_precision"]), 499 / 500)

    def test_auto_observed_precision_gate_is_inclusive(self) -> None:
        settings = Settings()
        passing = parse_completed_review(
            _completed_review(settings, auto_incorrect=10), settings
        )
        failing = parse_completed_review(
            _completed_review(settings, auto_incorrect=11), settings
        )
        self.assertTrue(passing.gate_passed)
        self.assertFalse(failing.gate_passed)

    def test_auto_lower_wilson_bound_must_be_strictly_above_floor(self) -> None:
        settings = replace(
            Settings(), sample_auto=50, sample_review=750, sample_fallback=200
        )
        result = parse_completed_review(_completed_review(settings), settings)
        self.assertFalse(result.gate_passed)
        self.assertTrue(any("Wilson lower bound" in reason for reason in result.gate_reasons))

    def test_incomplete_or_invalid_review_is_rejected(self) -> None:
        settings = Settings()
        with self.assertRaises(ReviewFormatError):
            parse_completed_review(_completed_review(settings).iloc[:-1], settings)
        invalid = _completed_review(settings)
        invalid.loc[0, "review_decision"] = "probably"
        with self.assertRaises(ReviewFormatError):
            parse_completed_review(invalid, settings)

    def test_legacy_fallback_value_is_normalized(self) -> None:
        settings = Settings()
        review = _completed_review(settings)
        review.loc[review["review_tier"].eq("fallback_review"), "review_tier"] = "fallback"
        result = parse_completed_review(review, settings)
        self.assertIn("fallback_review", result.stats["tier"].tolist())

    def test_redistributed_sample_allocation_is_accepted(self) -> None:
        settings = Settings()
        review = _completed_review(settings)
        fallback = review["review_tier"].eq("fallback_review")
        review.loc[fallback, "review_tier"] = "review"
        review["sample_allocation"] = review["review_tier"].map(
            {"auto": 500, "review": 500, "fallback_review": 0}
        )
        result = parse_completed_review(review, settings)
        counts = result.stats.set_index("tier")["n_reviewed"].to_dict()
        self.assertEqual(counts, {"auto": 500, "review": 500, "fallback_review": 0})

    def test_written_review_outputs_are_aggregate_only(self) -> None:
        settings = Settings()
        result = parse_completed_review(_completed_review(settings), settings)
        with TemporaryDirectory() as temporary:
            csv_path, txt_path = write_review_aggregates(result, temporary)
            combined = csv_path.read_text() + txt_path.read_text()
            self.assertNotIn("RT PRIVATE ROW", combined)
            self.assertFalse(scan_identifiers(temporary))

    def test_small_nonzero_review_outcome_suppresses_tier_detail(self) -> None:
        settings = Settings()
        result = parse_completed_review(
            _completed_review(settings, auto_incorrect=1), settings
        )
        with TemporaryDirectory() as temporary:
            csv_path, _ = write_review_aggregates(result, temporary)
            public = pd.read_csv(csv_path)
            self.assertNotIn("auto", public["tier"].tolist())

    def test_rare_nonauto_outcome_is_not_revealed_in_text(self) -> None:
        settings = Settings()
        review = _completed_review(settings)
        first_review = review.index[review["review_tier"].eq("review")][0]
        review.loc[first_review, "review_decision"] = "uncertain"
        result = parse_completed_review(review, settings)
        with TemporaryDirectory() as temporary:
            csv_path, txt_path = write_review_aggregates(result, temporary)
            public = pd.read_csv(csv_path)
            text = txt_path.read_text(encoding="utf-8")
            self.assertNotIn("review", public["tier"].tolist())
            self.assertIn("Uncertain decisions: <10", text)
            self.assertNotIn("Uncertain decision present", text)
            self.assertNotIn("0.996667", text)


class DisclosureTests(unittest.TestCase):
    def test_small_cells_are_removed(self) -> None:
        frame = pd.DataFrame(
            {"tier": ["large", "small"], "n": [12, 4], "auc": [0.72, 0.99]}
        )
        cleaned, removed = suppress_small_cells(frame, count_columns="n", min_cell_n=10)
        self.assertEqual(removed, 1)
        self.assertEqual(cleaned["tier"].tolist(), ["large"])

    def test_zero_cells_are_safe_but_small_nonzero_subcounts_are_removed(self) -> None:
        frame = pd.DataFrame(
            {
                "group": ["zero", "small", "large"],
                "rows": [100, 100, 100],
                "positive": [0, 1, 10],
            }
        )
        cleaned, removed = suppress_small_cells(
            frame,
            count_columns=("rows", "positive"),
            min_cell_n=10,
        )
        self.assertEqual(removed, 1)
        self.assertEqual(cleaned["group"].tolist(), ["zero", "large"])

    def test_staging_uses_allowlist_and_suppresses_small_cells(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate"
            source.mkdir()
            pd.DataFrame(
                {"tier": ["large", "small"], "n": [12, 4], "auc": [0.72, 0.99]}
            ).to_csv(source / "E4.csv", index=False)
            (source / "not_allowed.txt").write_text("not staged\n", encoding="utf-8")
            report = stage_egress(
                source,
                root / "egress",
                allowlist={"E4.csv": "n"},
                min_cell_n=10,
            )
            self.assertTrue(report.passed)
            self.assertEqual(report.suppressed_rows, (("E4.csv", 1),))
            staged = pd.read_csv(root / "egress" / "E4.csv")
            self.assertEqual(staged["tier"].tolist(), ["large"])
            self.assertFalse((root / "egress" / "not_allowed.txt").exists())

    def test_internal_identifiers_are_ignored_but_egress_leak_fails(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            internal = root / "rt_internal"
            internal.mkdir()
            planted = pd.DataFrame(
                {
                    "company_name": ["EDWIN WIDGETS LIMITED"],
                    "postcode": ["SW1A 1AA"],
                    "company_number": ["12345678"],
                }
            )
            planted.to_csv(internal / "RT_INTERNAL_match_pairs_1000.csv", index=False)
            self.assertFalse(scan_identifiers(root))

            egress = root / "egress"
            egress.mkdir()
            planted.to_csv(egress / "leaked_pairs.csv", index=False)
            findings = scan_identifiers(root)
            kinds = {finding.kind for finding in findings}
            self.assertIn("identifier_column", kinds)
            self.assertIn("postcode_value", kinds)
            self.assertIn("company_name_value", kinds)
            self.assertIn("company_number_value", kinds)
            with self.assertRaises(DisclosureViolation):
                validate_egress(egress)

    def test_lettered_company_number_and_known_name_are_detected(self) -> None:
        with TemporaryDirectory() as temporary:
            egress = Path(temporary) / "egress"
            egress.mkdir()
            (egress / "bad.txt").write_text(
                "Pair: North Star Trading; company no SC123456\n", encoding="utf-8"
            )
            with self.assertRaises(DisclosureViolation) as caught:
                validate_egress(egress, known_identifiers=["North Star Trading"])
            kinds = {finding.kind for finding in caught.exception.report.findings}
            self.assertIn("company_number_value", kinds)
            self.assertIn("known_identifier_value", kinds)

    def test_rt_internal_path_cannot_be_allowlisted(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate"
            (source / "rt_internal").mkdir(parents=True)
            (source / "rt_internal" / "pairs.csv").write_text("n\n1000\n")
            with self.assertRaises(ValueError):
                stage_egress(
                    source,
                    root / "egress",
                    allowlist={"rt_internal/pairs.csv": "n"},
                )


if __name__ == "__main__":
    unittest.main()
