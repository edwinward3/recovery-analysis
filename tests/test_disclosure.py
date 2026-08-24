"""Test the final output checks with planted names and small counts."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from recovery.disclosure import (
    DisclosureViolation,
    scan_identifiers,
    stage_egress,
    suppress_small_cells,
    validate_egress,
)


class DisclosureTests(unittest.TestCase):
    def test_small_counts_and_associated_estimates_are_blanked(self) -> None:
        frame = pd.DataFrame(
            {"tier": ["large", "small"], "n": [12, 4], "auc": [0.72, 0.99]}
        )
        cleaned, removed = suppress_small_cells(frame, count_columns="n", min_cell_n=10)
        self.assertEqual(removed, 2)
        self.assertEqual(cleaned["tier"].tolist(), ["large", "small"])
        self.assertTrue(cleaned["n"].isna().all())
        self.assertTrue(cleaned["auc"].isna().all())

    def test_zero_cells_remain_visible_and_are_not_complementary_cells(self) -> None:
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
        self.assertEqual(removed, 2)
        self.assertEqual(cleaned.loc[0, "rows"], 100)
        self.assertTrue(pd.isna(cleaned.loc[1, "rows"]))
        self.assertEqual(cleaned.loc[2, "rows"], 100)
        self.assertEqual(cleaned.loc[0, "positive"], 0)
        self.assertTrue(cleaned.loc[[1, 2], "positive"].isna().all())

    def test_missing_count_cells_remain_missing(self) -> None:
        frame = pd.DataFrame({"group": ["a", "b"], "rows": [pd.NA, 20]})

        cleaned, suppressed = suppress_small_cells(frame, count_columns="rows")

        self.assertTrue(pd.isna(cleaned.loc[0, "rows"]))
        self.assertEqual(cleaned.loc[1, "rows"], 20)
        self.assertEqual(suppressed, 0)

    def test_grouped_total_gets_one_complementary_suppression(self) -> None:
        frame = pd.DataFrame(
            {
                "stratum": ["A", "A", "A", "B", "B", "B"],
                "band": ["small", "next", "large", "one", "two", "safe"],
                "n": [4, 12, 20, 2, 5, 11],
                "estimate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "standard_error": [0.01] * 6,
                "lower_ci": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "upper_ci": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            }
        )

        cleaned, suppressed = suppress_small_cells(
            frame,
            count_columns="n",
            min_cell_n=10,
            group_columns="stratum",
        )

        self.assertEqual(suppressed, 4)
        # A has one primary cell, so its smallest safe cell (12) is complementary.
        self.assertTrue(cleaned.loc[[0, 1], "n"].isna().all())
        self.assertEqual(cleaned.loc[2, "n"], 20)
        # B already has two primary suppressions; its safe cell need not be hidden.
        self.assertTrue(cleaned.loc[[3, 4], "n"].isna().all())
        self.assertEqual(cleaned.loc[5, "n"], 11)
        for column in ("estimate", "standard_error", "lower_ci", "upper_ci"):
            self.assertTrue(cleaned.loc[[0, 1, 3, 4], column].isna().all())
            self.assertFalse(pd.isna(cleaned.loc[2, column]))
            self.assertFalse(pd.isna(cleaned.loc[5, column]))

    def test_curve_hides_event_parts_and_later_cumulative_estimates(self) -> None:
        frame = pd.DataFrame(
            {
                "month": [0, 1, 2],
                "at_risk": [100, 95, 80],
                "satisfaction_events": [0, 4, 12],
                "cancellation_events": [0, 0, 1],
                "censored": [0, 1, 2],
                "satisfaction_cif": [0.0, 0.04, 0.16],
                "cancellation_cif": [0.0, 0.0, 0.01],
                "event_free_survival": [1.0, 0.96, 0.83],
            }
        )

        cleaned, _ = suppress_small_cells(
            frame,
            count_columns=(
                "at_risk",
                "satisfaction_events",
                "cancellation_events",
                "censored",
            ),
        )

        for column in (
            "satisfaction_events",
            "cancellation_events",
            "censored",
        ):
            self.assertTrue(pd.isna(cleaned.loc[1, column]))
        for column in (
            "satisfaction_cif",
            "cancellation_cif",
            "event_free_survival",
        ):
            self.assertTrue(cleaned.loc[1:, column].isna().all())
        self.assertEqual(cleaned.loc[0, "at_risk"], 100)
        self.assertTrue(cleaned.loc[1:, "at_risk"].isna().all())

    def test_split_row_hides_counts_that_reconstruct_a_small_event_cell(self) -> None:
        frame = pd.DataFrame(
            {
                "split": ["train", "validation", "calibration", "final_test"],
                "rows": [100, 100, 100, 100],
                "events": [5, 20, 20, 20],
                "non_events": [95, 80, 80, 80],
                "cancellations": [10, 10, 10, 10],
            }
        )

        cleaned, _ = suppress_small_cells(
            frame,
            count_columns=("rows", "events", "non_events", "cancellations"),
        )

        self.assertTrue(
            cleaned.loc[0, ["rows", "events", "non_events", "cancellations"]]
            .isna()
            .all()
        )

    def test_cohort_flow_row_hides_rows_when_company_count_is_small(self) -> None:
        frame = pd.DataFrame(
            {
                "stage": ["exact_linked_start", "excluded", "final_cohort"],
                "rows": [100, 20, 80],
                "companies": [80, 5, 75],
            }
        )

        cleaned, _ = suppress_small_cells(
            frame,
            count_columns=("rows", "companies"),
        )

        self.assertTrue(cleaned.loc[1, ["rows", "companies"]].isna().all())

    def test_pipeline_group_columns_and_shares_are_inferred(self) -> None:
        frame = pd.DataFrame(
            {
                "dimension": ["status", "status", "status"],
                "value": ["rare", "next", "common"],
                "rows": [3, 12, 85],
                "share": [0.03, 0.12, 0.85],
            }
        )

        cleaned, suppressed = suppress_small_cells(frame, count_columns="rows")

        self.assertEqual(suppressed, 2)
        self.assertTrue(cleaned.loc[[0, 1], "rows"].isna().all())
        self.assertTrue(cleaned.loc[[0, 1], "share"].isna().all())
        self.assertEqual(cleaned.loc[2, "rows"], 85)
        self.assertEqual(cleaned.loc[2, "share"], 0.85)

    def test_registration_dimensions_get_independent_complementary_suppression(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "dimension": [
                    "validity",
                    "validity",
                    "validity",
                    "calendar_day_delay",
                    "calendar_day_delay",
                    "calendar_day_delay",
                    "calendar_day_delay",
                    "working_day_delay",
                    "working_day_delay",
                    "working_day_delay",
                    "registration_day",
                    "registration_day",
                    "registration_day",
                ],
                "measure": [
                    "all_records",
                    "valid_for_delay_calculation",
                    "excluded_date_anomaly",
                    "valid_records",
                    "same_day",
                    "next_day",
                    "later",
                    "valid_records",
                    "within_one_working_day",
                    "more_than_one_working_day",
                    "valid_records",
                    "working_day",
                    "non_working_day",
                ],
                "rows": [100, 100, 0, 100, 5, 80, 15, 100, 95, 5, 100, 100, 0],
                "share": [
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    0.05,
                    0.80,
                    0.15,
                    1.0,
                    0.95,
                    0.05,
                    1.0,
                    1.0,
                    0.0,
                ],
            }
        )

        cleaned, suppressed = suppress_small_cells(
            frame,
            count_columns="rows",
            min_cell_n=10,
        )
        indexed = cleaned.set_index(["dimension", "measure"])

        self.assertEqual(suppressed, 4)
        for key in (
            ("calendar_day_delay", "same_day"),
            ("calendar_day_delay", "later"),
            ("working_day_delay", "within_one_working_day"),
            ("working_day_delay", "more_than_one_working_day"),
        ):
            self.assertTrue(pd.isna(indexed.loc[key, "rows"]))
            self.assertTrue(pd.isna(indexed.loc[key, "share"]))
        self.assertEqual(indexed.loc[("calendar_day_delay", "next_day"), "rows"], 80)
        self.assertEqual(indexed.loc[("validity", "all_records"), "rows"], 100)
        self.assertEqual(indexed.loc[("registration_day", "non_working_day"), "rows"], 0)

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
            self.assertEqual(report.suppressed_rows, (("E4.csv", 2),))
            staged = pd.read_csv(root / "egress" / "E4.csv")
            self.assertEqual(staged["tier"].tolist(), ["large", "small"])
            self.assertTrue(staged["n"].isna().all())
            self.assertTrue(staged["auc"].isna().all())
            self.assertFalse((root / "egress" / "not_allowed.txt").exists())

    def test_named_working_file_is_ignored_but_a_report_leak_fails(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            working = root / "working_files"
            working.mkdir()
            planted = pd.DataFrame(
                {
                    "company_name": ["EDWIN WIDGETS LIMITED"],
                    "postcode": ["SW1A 1AA"],
                    "company_number": ["12345678"],
                }
            )
            planted.to_csv(working / "matching_pairs_1000.csv", index=False)
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

    def test_named_working_files_cannot_be_copied_as_reports(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate"
            (source / "working_files").mkdir(parents=True)
            (source / "working_files" / "pairs.csv").write_text("n\n1000\n")
            with self.assertRaises(ValueError):
                stage_egress(
                    source,
                    root / "egress",
                    allowlist={"working_files/pairs.csv": "n"},
                )


if __name__ == "__main__":
    unittest.main()
