"""Test the two run stages on synthetic data."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import pandas as pd

from recovery.disclosure import DisclosureViolation
from recovery.run import MATCH_FILENAME, PAIR_FILENAME, RunFailure, analyze
from recovery.reporting import write_e5 as real_write_e5
from recovery.synthetic import make_synthetic_bundle, write_bundle


ROOT = Path(__file__).resolve().parents[1]


class RunTests(unittest.TestCase):
    def test_run_1_matches_every_row_without_fitting_models(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = make_synthetic_bundle(1_600, include_prior_rows=False)
            judgments, companies, _ = write_bundle(bundle, root / "inputs", excel=False)
            with (
                patch(
                    "recovery.run.prepare_model_cohort",
                    side_effect=AssertionError("Run 1 must not prepare a model cohort"),
                ),
                patch(
                    "recovery.run.fit_evaluate_models",
                    side_effect=AssertionError("Run 1 must not fit a model"),
                ),
            ):
                paths = analyze(
                    stage="diagnostic",
                    judgments_path=judgments,
                    companies_house_path=companies,
                    observation_date=bundle.observation_date,
                    settings_path=ROOT / "settings.toml",
                    output_base=root / "outputs",
                    run_id="matching_only",
                )

            matches = pd.read_csv(paths.working / MATCH_FILENAME)
            coverage = pd.read_csv(paths.results / "E2_match_coverage.csv")
            self.assertEqual(len(matches), len(bundle.judgments))
            self.assertEqual(int(coverage["rows"].sum()), len(bundle.judgments))
            self.assertFalse((paths.results / "E3_model_comparison.csv").exists())
            self.assertFalse((paths.working / "model_rows.csv").exists())

            pairs = pd.read_csv(paths.working / PAIR_FILENAME)
            self.assertEqual(len(pairs), 1_000)
            self.assertIn("source_company_name", pairs)
            self.assertIn("matched_company_name", pairs)
            self.assertNotIn("review_decision", pairs)
            self.assertNotIn("review_notes", pairs)

            summary = (paths.results / "SUMMARY.txt").read_text(encoding="utf-8")
            self.assertIn("RUN 1 MATCHING SUMMARY", summary)
            self.assertIn("No satisfaction model was trained", summary)
            manifest = json.loads(
                (paths.results / "E5_run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertIsNone(manifest["model_only_acceptance"])
            self.assertNotIn("review_binding", manifest)

    def test_short_names_are_not_used_as_disclosure_needles(self) -> None:
        from recovery.run import _bounded_known_identifiers

        sample = pd.DataFrame(
            {
                "source_company_name": ["A"],
                "source_trading_name": ["AB"],
                "matched_company_name": ["DISTINCTIVE SYNTHETIC NAME LIMITED"],
            }
        )
        identifiers = _bounded_known_identifiers(sample)
        self.assertNotIn("A", identifiers)
        self.assertIn("DISTINCTIVE SYNTHETIC NAME LIMITED", identifiers)

    def test_run_2_requires_an_explicit_extract_date(self) -> None:
        with self.assertRaisesRegex(RunFailure, "explicit RT extract date"):
            analyze(
                stage="locked",
                judgments_path="not opened.csv",
                companies_house_path="not opened.zip",
                observation_date=None,
                settings_path=ROOT / "settings.toml",
                output_base="not created",
            )

    def test_unsafe_final_report_is_not_copied(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = make_synthetic_bundle(1_600, include_prior_rows=False)
            judgments, companies, _ = write_bundle(bundle, root / "inputs", excel=False)

            def write_unsafe_e5(*args: object, **kwargs: object) -> object:
                written = real_write_e5(*args, **kwargs)
                manifest_path = Path(args[0]) / "E5_run_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["unsafe_test_value"] = "PRIVATE FIXTURE LIMITED"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                return written

            with (
                patch("recovery.run.write_e5", side_effect=write_unsafe_e5),
                self.assertRaises(DisclosureViolation),
            ):
                analyze(
                    stage="diagnostic",
                    judgments_path=judgments,
                    companies_house_path=companies,
                    observation_date=bundle.observation_date,
                    settings_path=ROOT / "settings.toml",
                    output_base=root / "outputs",
                    run_id="unsafe",
                )
            reports = root / "outputs" / "diagnostic_unsafe" / "results"
            self.assertEqual(list(reports.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
