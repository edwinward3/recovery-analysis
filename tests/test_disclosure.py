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
