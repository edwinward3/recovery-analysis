"""Tests for the Registry Trust data check."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import hashlib
import json
import sys
import unittest
import zipfile
from unittest.mock import Mock, call, patch

import pandas as pd

from recovery.disclosure import DisclosureViolation
from recovery.matching import (
    ACCEPTED_LINKAGE_VALIDATION_FILENAME,
    UNMATCHED_LINKAGE_VALIDATION_FILENAME,
)
from recovery.run import (
    RunFailure,
    _companies_house_filename_date,
    _elapsed_updates,
    _format_elapsed,
    analyze,
    main,
    package_results,
)
from recovery.reporting import write_e5 as real_write_e5
from recovery.synthetic import make_synthetic_bundle, write_bundle


ROOT = Path(__file__).resolve().parents[1]


class RunTests(unittest.TestCase):
    def test_command_prints_elapsed_time_and_zip_to_send(self) -> None:
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
                call("Elapsed time: 00:00:00"),
                call(f"SEND THIS FILE TO EDWIN: {archive}"),
            ],
        )

    def test_elapsed_time_format_and_heartbeat(self) -> None:
        stop = Mock()
        stop.wait.side_effect = [False, True]

        with (
            patch("recovery.run.time.monotonic", return_value=3662.9),
            patch("builtins.print") as printer,
        ):
            _elapsed_updates(stop, 1.0)

        self.assertEqual(_format_elapsed(-1), "00:00:00")
        self.assertEqual(_format_elapsed(3661.9), "01:01:01")
        stop.wait.assert_has_calls([call(300), call(300)])
        printer.assert_called_once_with(
            "Still running. Elapsed time: 01:01:01", flush=True
        )

    def test_elapsed_thread_stops_when_run_fails(self) -> None:
        arguments = [
            "analyze",
            "--judgments",
            "rt.xlsx",
            "--companies-house",
            "BasicCompanyData-2026-08-01.zip",
            "--observation-date",
            "2026-08-01",
        ]
        with (
            patch("recovery.run.analyze", side_effect=RunFailure("test failure")),
            patch("recovery.run.Thread") as thread_type,
            patch("builtins.print") as printer,
        ):
            status = main(arguments)

        self.assertEqual(status, 2)
        thread_type.return_value.start.assert_called_once_with()
        thread_type.return_value.join.assert_called_once_with(1)
        self.assertEqual(
            printer.call_args_list,
            [
                call("STOP: RunFailure: test failure", file=sys.stderr),
                call("Elapsed time: 00:00:00", file=sys.stderr),
            ],
        )

    def test_run_continues_if_elapsed_thread_cannot_start(self) -> None:
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
        ]
        with (
            patch("recovery.run.analyze", return_value=paths),
            patch("recovery.run.package_results", return_value=archive),
            patch("recovery.run.Thread.start", side_effect=RuntimeError),
            patch("recovery.run.time.monotonic", side_effect=[10.0, 11.0]),
            patch("builtins.print") as printer,
        ):
            status = main(arguments)

        self.assertEqual(status, 0)
        self.assertEqual(
            printer.call_args_list,
            [
                call("RUN COMPLETE"),
                call("Elapsed time: 00:00:01"),
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
            self.assertIn("RT DATA CHECK", summary)
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
            self.assertEqual(manifest["schema_version"], 8)
            self.assertEqual(
                manifest["companies_house_filename_date"],
                pd.Timestamp(bundle.observation_date).date().isoformat(),
            )
            self.assertEqual(
                manifest["matching_rule"],
                "unique_date_valid_exact_normalized_name_v1",
            )
            self.assertEqual(manifest["package_versions"]["scikit-learn"], "1.9.0")
            self.assertEqual(manifest["package_versions"]["lightgbm"], "4.7.0")
            self.assertEqual(
                manifest["academic_design"]["prediction_bootstrap_replicates"],
                500,
            )

    def test_complete_event_dates_run_longitudinal_analysis_and_prediction(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = make_synthetic_bundle(
                8_000,
                include_prior_rows=False,
                include_event_dates=True,
            )
            judgments, companies, _ = write_bundle(
                bundle, root / "inputs", excel=False
            )

            paths = analyze(
                judgments_path=judgments,
                companies_house_path=companies,
                observation_date=bundle.observation_date,
                settings_path=ROOT / "settings.toml",
                output_base=root / "outputs",
                run_id="longitudinal",
            )

            outcome_gate = pd.read_csv(paths.results / "E3_outcome_gate.csv")
            prediction_gate = pd.read_csv(paths.results / "E4_prediction_gate.csv")
            cohort_flow = pd.read_csv(paths.results / "E4_cohort_flow.csv")
            self.assertEqual(outcome_gate.loc[0, "selected_analysis"], "longitudinal")
            self.assertTrue(prediction_gate["status"].eq("pass").all())
            self.assertIn("primary_estimand", set(prediction_gate["check"]))
            self.assertIn("cancellation_treatment", set(prediction_gate["check"]))
            self.assertIn("final_analysed_cohort", set(cohort_flow["stage"]))
            for name in (
                "E3_cumulative_incidence.csv",
                "E3_fixed_horizon.csv",
                "E4_cohort_flow.csv",
                "E4_model_performance.csv",
                "E4_paired_improvement.csv",
                "E4_calibration_curve.csv",
                "E4_ranking.csv",
            ):
                self.assertTrue((paths.results / name).is_file(), name)
            archive = package_results(paths)
            with zipfile.ZipFile(archive) as package:
                names = {entry.filename for entry in package.infolist()}
            self.assertFalse(any("working_files" in name for name in names))
            self.assertFalse(any("linkage_validation" in name for name in names))

    def test_conflicting_event_dates_select_cross_sectional_fallback(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = make_synthetic_bundle(
                200,
                include_prior_rows=False,
                include_event_dates=True,
            )
            affected = bundle.judgments["JudgmentStatus"].eq("Satisfied")
            bundle.judgments.loc[affected.idxmax(), "Satisfaction Date"] = ""
            judgments, companies, _ = write_bundle(
                bundle, root / "inputs", excel=False
            )

            paths = analyze(
                judgments_path=judgments,
                companies_house_path=companies,
                observation_date=bundle.observation_date,
                settings_path=ROOT / "settings.toml",
                output_base=root / "outputs",
                run_id="fallback",
            )

            outcome = pd.read_csv(paths.results / "E3_outcome_gate.csv")
            prediction = pd.read_csv(paths.results / "E4_prediction_gate.csv")
            self.assertEqual(outcome.loc[0, "selected_analysis"], "cross_sectional")
            self.assertTrue((paths.results / "E3_status_at_extract.csv").is_file())
            self.assertFalse((paths.results / "E3_cumulative_incidence.csv").exists())
            self.assertEqual(prediction.loc[0, "status"], "fail")

    def test_unparseable_event_date_selects_cross_sectional_fallback(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = make_synthetic_bundle(
                200,
                include_prior_rows=False,
                include_event_dates=True,
            )
            affected = bundle.judgments["JudgmentStatus"].eq("Satisfied").idxmax()
            bundle.judgments.loc[affected, "Satisfaction Date"] = "not a date"
            judgments, companies, _ = write_bundle(
                bundle, root / "inputs", excel=False
            )

            paths = analyze(
                judgments_path=judgments,
                companies_house_path=companies,
                observation_date=bundle.observation_date,
                settings_path=ROOT / "settings.toml",
                output_base=root / "outputs",
                run_id="unparseable_event",
            )

            outcome = pd.read_csv(paths.results / "E3_outcome_gate.csv")
            self.assertEqual(outcome.loc[0, "selected_analysis"], "cross_sectional")
            self.assertTrue((paths.results / "E3_status_at_extract.csv").is_file())
            self.assertFalse((paths.results / "E3_cumulative_incidence.csv").exists())

    def test_unknown_outcome_header_blocks_outcomes_without_stopping_run(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = make_synthetic_bundle(200, include_prior_rows=False)
            bundle.judgments["Payment Confirmed At"] = ""
            judgments, companies, _ = write_bundle(
                bundle, root / "inputs", excel=False
            )

            paths = analyze(
                judgments_path=judgments,
                companies_house_path=companies,
                observation_date=bundle.observation_date,
                settings_path=ROOT / "settings.toml",
                output_base=root / "outputs",
                run_id="unknown_outcome_header",
            )

            outcome = pd.read_csv(paths.results / "E3_outcome_gate.csv")
            audit = pd.read_csv(paths.results / "E1_data_audit.csv")
            self.assertEqual(outcome.loc[0, "selected_analysis"], "blocked")
            self.assertFalse((paths.results / "E3_status_at_extract.csv").exists())
            self.assertFalse((paths.results / "E3_cumulative_incidence.csv").exists())
            self.assertIn(
                "Payment Confirmed At",
                set(
                    audit.loc[
                        audit["dimension"].eq(
                            "outcome_or_history_header_not_recognised"
                        ),
                        "value",
                    ]
                ),
            )

    def test_outcome_issue_outside_corporate_ew_does_not_block_longitudinal(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = make_synthetic_bundle(
                200,
                include_prior_rows=False,
                include_event_dates=True,
            )
            affected = bundle.judgments["JudgmentStatus"].eq("Satisfied").idxmax()
            bundle.judgments.loc[affected, "DefendantType"] = "Consumer"
            bundle.judgments.loc[affected, "Satisfaction Date"] = ""
            judgments, companies, _ = write_bundle(
                bundle, root / "inputs", excel=False
            )

            paths = analyze(
                judgments_path=judgments,
                companies_house_path=companies,
                observation_date=bundle.observation_date,
                settings_path=ROOT / "settings.toml",
                output_base=root / "outputs",
                run_id="target_only",
            )

            outcome = pd.read_csv(paths.results / "E3_outcome_gate.csv")
            self.assertEqual(outcome.loc[0, "selected_analysis"], "longitudinal")

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
            summary = results / "SUMMARY.txt"
            summary.write_text("complete\n", encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "artifact_name": summary.name,
                        "bytes": summary.stat().st_size,
                        "sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                        "rows": pd.NA,
                    }
                ]
            ).to_csv(results / "E5_artifact_manifest.csv", index=False)
            (working / "review.csv").write_text("id\n1\n", encoding="utf-8")
            paths = SimpleNamespace(root=root, results=results)

            archive = package_results(paths)

            self.assertEqual(archive.name, "SEND_TO_EDWIN_known.zip")
            with zipfile.ZipFile(archive) as package:
                self.assertEqual(
                    {entry.filename for entry in package.infolist() if not entry.is_dir()},
                    {
                        "run_known/results/E5_artifact_manifest.csv",
                        "run_known/results/SUMMARY.txt",
                    },
                )
            checksum = archive.with_suffix(".zip.sha256")
            self.assertEqual(
                checksum.read_text(encoding="ascii").split(),
                [hashlib.sha256(archive.read_bytes()).hexdigest(), archive.name],
            )

    def test_zip_write_failure_leaves_no_send_package(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = self._minimal_completed_run(Path(temporary))
            archive = paths.root.parent / "SEND_TO_EDWIN_known.zip"

            with (
                patch.object(zipfile.ZipFile, "write", side_effect=OSError("disk full")),
                self.assertRaisesRegex(OSError, "disk full"),
            ):
                package_results(paths)

            self.assertFalse(archive.exists())
            self.assertFalse(archive.with_suffix(".zip.sha256").exists())

    def test_checksum_write_failure_leaves_no_send_package(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = self._minimal_completed_run(Path(temporary))
            archive = paths.root.parent / "SEND_TO_EDWIN_known.zip"
            real_write_text = Path.write_text

            def fail_checksum(path: Path, *args: object, **kwargs: object) -> int:
                if path.name.endswith(".zip.sha256"):
                    raise OSError("checksum disk full")
                return real_write_text(path, *args, **kwargs)

            with (
                patch.object(Path, "write_text", autospec=True, side_effect=fail_checksum),
                self.assertRaisesRegex(OSError, "checksum disk full"),
            ):
                package_results(paths)

            self.assertFalse(archive.exists())
            self.assertFalse(archive.with_suffix(".zip.sha256").exists())

    def test_archive_publish_failure_removes_published_checksum(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = self._minimal_completed_run(Path(temporary))
            archive = paths.root.parent / "SEND_TO_EDWIN_known.zip"
            real_replace = Path.replace

            def fail_archive_publish(path: Path, target: Path) -> Path:
                if Path(target).suffix == ".zip":
                    raise OSError("archive publish failed")
                return real_replace(path, target)

            with (
                patch.object(
                    Path,
                    "replace",
                    autospec=True,
                    side_effect=fail_archive_publish,
                ),
                self.assertRaisesRegex(OSError, "archive publish failed"),
            ):
                package_results(paths)

            self.assertFalse(archive.exists())
            self.assertFalse(archive.with_suffix(".zip.sha256").exists())

    @staticmethod
    def _minimal_completed_run(output: Path) -> SimpleNamespace:
        root = output / "run_known"
        results = root / "results"
        results.mkdir(parents=True)
        summary = results / "SUMMARY.txt"
        summary.write_text("complete\n", encoding="utf-8")
        pd.DataFrame(
            [
                {
                    "artifact_name": summary.name,
                    "bytes": summary.stat().st_size,
                    "sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                    "rows": pd.NA,
                }
            ]
        ).to_csv(results / "E5_artifact_manifest.csv", index=False)
        return SimpleNamespace(root=root, results=results)

    def test_changed_result_is_not_packaged(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "run_known"
            results = root / "results"
            results.mkdir(parents=True)
            summary = results / "SUMMARY.txt"
            summary.write_text("complete\n", encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "artifact_name": summary.name,
                        "bytes": summary.stat().st_size,
                        "sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                        "rows": pd.NA,
                    }
                ]
            ).to_csv(results / "E5_artifact_manifest.csv", index=False)
            summary.write_text("changed\n", encoding="utf-8")

            with self.assertRaisesRegex(RunFailure, "artifact check failed"):
                package_results(SimpleNamespace(root=root, results=results))

            self.assertFalse((root.parent / "SEND_TO_EDWIN_known.zip").exists())

    def test_unlisted_nested_file_is_not_packaged(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = self._minimal_completed_run(Path(temporary))
            nested = paths.results / "nested"
            nested.mkdir()
            (nested / "private.csv").write_text(
                "company_name\nPRIVATE COMPANY LIMITED\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(
                RunFailure, "results folder contains an unlisted file or folder"
            ):
                package_results(paths)

            archive = paths.root.parent / "SEND_TO_EDWIN_known.zip"
            self.assertFalse(archive.exists())
            self.assertFalse(archive.with_suffix(".zip.sha256").exists())

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

    def test_companies_house_filename_date_is_optional(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = make_synthetic_bundle(1_600, include_prior_rows=False)
            judgments, dated_companies, _ = write_bundle(
                bundle, root / "inputs", excel=False
            )
            companies = dated_companies.with_name("companies-house.zip")
            dated_companies.rename(companies)
            output_base = root / "outputs"
            paths = analyze(
                judgments_path=judgments,
                companies_house_path=companies,
                observation_date=bundle.observation_date,
                settings_path=ROOT / "settings.toml",
                output_base=output_base,
            )
            summary = (paths.results / "SUMMARY.txt").read_text(encoding="utf-8")
            manifest = json.loads(
                (paths.results / "E5_run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertIn("Companies House filename date  unknown", summary)
            self.assertIsNone(manifest["companies_house_filename_date"])

    def test_companies_house_filename_date_is_inferred_only_when_clear(self) -> None:
        self.assertEqual(
            _companies_house_filename_date(Path("BasicCompanyData-2026-08-01.zip")),
            pd.Timestamp("2026-08-01"),
        )
        for name in (
            "BasicCompanyData.zip",
            "BasicCompanyData-2026-99-99.zip",
            "BasicCompanyData-2026-07-01-to-2026-08-01.zip",
        ):
            with self.subTest(name=name):
                self.assertIsNone(_companies_house_filename_date(Path(name)))

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
                    settings_path=ROOT / "settings.toml",
                    output_base=root / "outputs",
                    run_id="unsafe",
                )
            self.assertFalse((root / "outputs" / "run_unsafe").exists())


if __name__ == "__main__":
    unittest.main()
