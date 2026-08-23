"""Tests for the Registry Trust data check."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
import zipfile
from unittest.mock import call, patch

import pandas as pd

from recovery.disclosure import DisclosureViolation
from recovery.matching import (
    ACCEPTED_LINKAGE_VALIDATION_FILENAME,
    UNMATCHED_LINKAGE_VALIDATION_FILENAME,
)
from recovery.run import (
    RunFailure,
    analyze,
    main,
    package_results,
)
from recovery.reporting import write_e5 as real_write_e5
from recovery.synthetic import make_synthetic_bundle, write_bundle


ROOT = Path(__file__).resolve().parents[1]


class RunTests(unittest.TestCase):
    def test_command_prints_only_the_zip_to_send(self) -> None:
        root = Path("outputs/run_finished").resolve()
        archive = Path("outputs/SEND_TO_EDWIN_finished.zip").resolve()
        paths = SimpleNamespace(root=root)
        arguments = [
            "analyze",
            "--judgments",
            "rt.xlsx",
            "--companies-house",
            "BasicCompanyData-2026-08-01.zip",
            "--observation-date",
            "2026-08-01",
            "--companies-house-date",
            "2026-08-01",
        ]
        with (
            patch("recovery.run.analyze", return_value=paths),
            patch("recovery.run.package_results", return_value=archive),
            patch("builtins.print") as printer,
        ):
            status = main(arguments)

        self.assertEqual(status, 0)
        self.assertEqual(
            printer.call_args_list,
            [
                call("RUN COMPLETE"),
                call(f"SEND THIS FILE TO EDWIN: {archive}"),
            ],
        )

    def test_run_matches_every_row_and_writes_expected_files(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = make_synthetic_bundle(1_600, include_prior_rows=False)
            judgments, companies, _ = write_bundle(bundle, root / "inputs", excel=False)
            matched_rows: list[int] = []
            paths = analyze(
                judgments_path=judgments,
                companies_house_path=companies,
                observation_date=bundle.observation_date,
                companies_house_date=bundle.observation_date,
                settings_path=ROOT / "settings.toml",
                output_base=root / "outputs",
                run_id="matching_only",
                _match_validator=lambda frame: matched_rows.append(len(frame)),
            )

            coverage = pd.read_csv(paths.results / "E2_match_coverage.csv")
            self.assertEqual(matched_rows, [len(bundle.judgments)])
            self.assertEqual(int(coverage["rows"].sum()), len(bundle.judgments))
            self.assertFalse((paths.working / "matching_table.csv.gz").exists())
            self.assertFalse((paths.root / ".aggregate_staging").exists())

            accepted = pd.read_csv(
                paths.working / ACCEPTED_LINKAGE_VALIDATION_FILENAME
            )
            self.assertEqual(len(accepted), 1_000)
            self.assertIn("source_company_name", accepted)
            self.assertIn("matched_company_name", accepted)
            self.assertEqual(set(accepted["tier"]), {"exact_unique"})
            self.assertNotIn("score", accepted)
            self.assertNotIn("margin", accepted)
            self.assertEqual(
                {path.name for path in paths.working.iterdir()},
                {
                    ACCEPTED_LINKAGE_VALIDATION_FILENAME,
                    UNMATCHED_LINKAGE_VALIDATION_FILENAME,
                },
            )
            unmatched = pd.read_csv(
                paths.working / UNMATCHED_LINKAGE_VALIDATION_FILENAME
            )
            self.assertGreater(len(unmatched), 0)
            self.assertEqual(set(unmatched["tier"]), {"unmatched"})

            summary = (paths.results / "SUMMARY.txt").read_text(encoding="utf-8")
            self.assertIn("RT MATCHING CHECK", summary)
            self.assertIn("Date Inserted (RT registration date)", summary)
            self.assertIn("Satisfaction Date", summary)
            self.assertIn("absent; 0 filled", summary)
            self.assertIn("Stock or historical extract", summary)
            self.assertIn("Corporate E&W rows", summary)
            self.assertNotIn("Full-dataset", summary)
            audit = pd.read_csv(paths.results / "E1_data_audit.csv", dtype="string")
            audit = audit.set_index("dimension")
            self.assertIn("Date Inserted (literal) distinct values", audit.index)
            self.assertIn("Date Inserted (literal) minimum", audit.index)
            self.assertIn("Date Inserted (literal) maximum", audit.index)
            manifest = json.loads(
                (paths.results / "E5_run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], 6)
            self.assertEqual(
                manifest["matching_rule"],
                "unique_date_valid_exact_normalized_name_v1",
            )

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

    def test_completed_run_is_packaged_without_inputs(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            root = output / "run_known"
            results = root / "results"
            working = root / "working_files"
            results.mkdir(parents=True)
            working.mkdir()
            (results / "SUMMARY.txt").write_text("complete\n", encoding="utf-8")
            (working / "review.csv").write_text("id\n1\n", encoding="utf-8")
            paths = SimpleNamespace(root=root)

            archive = package_results(paths)

            self.assertEqual(archive.name, "SEND_TO_EDWIN_known.zip")
            with zipfile.ZipFile(archive) as package:
                self.assertEqual(
                    {entry.filename for entry in package.infolist() if not entry.is_dir()},
                    {
                        "run_known/results/SUMMARY.txt",
                        "run_known/working_files/review.csv",
                    },
                )

    def test_empty_review_sample_does_not_stop_the_run(self) -> None:
        for empty_arm in ("accepted", "unmatched"):
            with self.subTest(empty_arm=empty_arm), TemporaryDirectory() as temporary:
                root = Path(temporary)
                bundle = make_synthetic_bundle(100, include_prior_rows=False)
                bundle.judgments["Defendant Trading Name"] = ""
                if empty_arm == "accepted":
                    bundle.judgments["Defendant Company Name"] = [
                        f"UNLINKED SYNTHETIC DEFENDANT {position:04d}"
                        for position in range(len(bundle.judgments))
                    ]
                else:
                    bundle.companies_house["PreviousName_1.CompanyName"] = ""
                    bundle.companies_house["PreviousName_1.CONDATE"] = ""
                    names_by_number = bundle.companies_house.set_index(
                        "CompanyNumber"
                    )["CompanyName"]
                    bundle.judgments["Defendant Company Name"] = bundle.ground_truth[
                        "expected_company_number"
                    ].map(names_by_number)
                judgments, companies, _ = write_bundle(
                    bundle, root / "inputs", excel=False
                )

                paths = analyze(
                    judgments_path=judgments,
                    companies_house_path=companies,
                    observation_date=bundle.observation_date,
                    companies_house_date=bundle.observation_date,
                    settings_path=ROOT / "settings.toml",
                    output_base=root / "outputs",
                    run_id=empty_arm,
                )

                accepted = pd.read_csv(
                    paths.working / ACCEPTED_LINKAGE_VALIDATION_FILENAME
                )
                unmatched = pd.read_csv(
                    paths.working / UNMATCHED_LINKAGE_VALIDATION_FILENAME
                )
                self.assertEqual(len(accepted if empty_arm == "accepted" else unmatched), 0)
                self.assertEqual(
                    len(unmatched if empty_arm == "accepted" else accepted), 100
                )

    def test_undated_companies_house_filename_stops_before_output(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = make_synthetic_bundle(1_600, include_prior_rows=False)
            judgments, dated_companies, _ = write_bundle(
                bundle, root / "inputs", excel=False
            )
            companies = dated_companies.with_name("companies-house.zip")
            dated_companies.rename(companies)
            output_base = root / "outputs"
            with self.assertRaisesRegex(RunFailure, "filename must contain"):
                analyze(
                    judgments_path=judgments,
                    companies_house_path=companies,
                    observation_date=bundle.observation_date,
                    companies_house_date=bundle.observation_date,
                    settings_path=ROOT / "settings.toml",
                    output_base=output_base,
                )
            self.assertFalse(output_base.exists())

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
                    judgments_path=judgments,
                    companies_house_path=companies,
                    observation_date=bundle.observation_date,
                    companies_house_date=bundle.observation_date,
                    settings_path=ROOT / "settings.toml",
                    output_base=root / "outputs",
                    run_id="unsafe",
                )
            self.assertFalse((root / "outputs" / "run_unsafe").exists())


if __name__ == "__main__":
    unittest.main()
