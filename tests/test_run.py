"""Test review provenance and final-status orchestration without confidential data.

Inputs are synthetic 1,000-row review tables and temporary settings. Outputs
are aggregate-only temporary files. No network or shell operations occur.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import pandas as pd

from recovery.disclosure import DisclosureViolation, validate_egress
from recovery.run import (
    MATCH_FILENAME,
    RunFailure,
    STATE_FILENAME,
    _analysis_allowlist,
    _bounded_known_identifiers,
    _json_fingerprint,
    analyze,
    main,
    review_completed_sample,
    review_sample_digest,
)
from recovery.config import load_settings
from recovery.reporting import source_fingerprint, write_e5 as real_write_e5
from recovery.synthetic import make_synthetic_bundle, write_bundle


ROOT = Path(__file__).resolve().parents[1]


def _review_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    allocation = {"auto": 500, "review": 300, "fallback_review": 200}
    number = 0
    for tier, count in allocation.items():
        for _ in range(count):
            rows.append(
                {
                    "review_row_id": f"R-{number:04d}",
                    "review_tier": tier,
                    "tier": tier,
                    "sample_allocation": count,
                    "sample_seed": 20260619,
                    "sampling_design": "equal_probability_systematic_stratified_v1",
                    "source_company_name": f"SYNTHETIC SOURCE {number:04d}",
                    "matched_company_number": f"{10_000_000 + number:08d}",
                    "JudgmentDate": pd.Timestamp("2024-01-02"),
                    "IncorporationDate": pd.Timestamp("2010-01-01"),
                    "Accounts.NextDueDate": pd.Timestamp("2025-06-30"),
                    "Mortgages.NumMortCharges": 0,
                    "score": "0.95",
                    "review_decision": "correct",
                    "review_notes": "",
                }
            )
            number += 1
    return pd.DataFrame(rows)


def _write_review_case(root: Path, *, stage: str = "locked") -> Path:
    settings = ROOT / "settings.toml"
    configured = load_settings(settings)
    frame = _review_frame()
    seed = (
        configured.diagnostic_seed if stage == "diagnostic" else configured.locked_seed
    )
    frame["sample_seed"] = seed
    run_root = root / f"{stage}_fixture"
    internal = run_root / "rt_internal"
    egress = run_root / "egress_candidate"
    internal.mkdir(parents=True)
    egress.mkdir()
    review_file = internal / "RT_INTERNAL_match_pairs_1000.csv"
    frame.to_csv(review_file, index=False, encoding="utf-8-sig")
    baseline = pd.read_csv(review_file, dtype="string", keep_default_na=False)
    model_hashes: dict[str, str] = {}
    model_only: dict[str, object] | None = None
    if stage == "locked":
        models = internal / "models"
        models.mkdir()
        model_file = models / "model_evaluation.json"
        model_file.write_text("{}\n", encoding="utf-8")
        model_hashes[model_file.name] = sha256(model_file.read_bytes()).hexdigest()
        model_only = {
            "status": "pass",
            "passed": True,
            "reasons": [],
            "family": "prospective",
            "algorithm": "logistic",
        }
    code_fingerprint = source_fingerprint(ROOT / "recovery", settings)
    state = {
        "schema_version": 1,
        "data_classification": "RT INTERNAL - DO NOT EGRESS",
        "run_id": run_root.name,
        "stage": stage,
        "observation_date": "2026-06-01",
        "sample_seed": seed,
        "sample_rows": 1_000,
        "sample_allocation": {"auto": 500, "review": 300, "fallback_review": 200},
        "immutable_columns": sorted(
            set(baseline.columns) - {"review_decision", "review_notes"}
        ),
        "sample_digest": review_sample_digest(baseline),
        "sampling_design": "equal_probability_systematic_stratified_v1",
        "settings_fingerprint": sha256(settings.read_bytes()).hexdigest(),
        "code_fingerprint": code_fingerprint,
        "rt_analysis_fingerprint": "1" * 64,
        "ch_analysis_fingerprint": "2" * 64,
        "model_artifact_fingerprints": model_hashes,
        "model_only_acceptance": model_only,
    }
    (internal / STATE_FILENAME).write_text(
        json.dumps(state), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "run_id": run_root.name,
        "stage": stage,
        "observation_date": "2026-06-01",
        "sample_seed": seed,
        "fingerprints": {
            "rt_analysis_content": "1" * 64,
            "ch_analysis_content": "2" * 64,
            "code_and_settings": code_fingerprint,
            "settings_file": sha256(settings.read_bytes()).hexdigest(),
        },
        "model_only_acceptance": state["model_only_acceptance"],
        "review_binding": {
            "review_state_sha256": _json_fingerprint(state),
            "model_artifact_sha256": model_hashes,
        },
        "disclosure": {"status": "pass"},
    }
    (egress / "E5_run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    for filename in _analysis_allowlist(stage):
        if filename == "E5_run_manifest.json":
            continue
        path = egress / filename
        if path.suffix == ".csv":
            path.write_text("metric,rows\nfixture,1000\n", encoding="utf-8")
        else:
            path.write_text("aggregate fixture\n", encoding="utf-8")
    return review_file


class ReviewOrchestrationTests(unittest.TestCase):
    def test_diagnostic_run_matches_every_row_and_never_calls_models(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = make_synthetic_bundle(1_600, include_prior_rows=False)
            judgments, companies, _ = write_bundle(
                bundle,
                root / "inputs",
                excel=False,
            )
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

            matches = pd.read_csv(paths.internal / MATCH_FILENAME)
            self.assertEqual(len(matches), len(bundle.judgments))
            coverage = pd.read_csv(paths.egress / "E2_match_coverage.csv")
            self.assertEqual(int(coverage["rows"].sum()), len(bundle.judgments))
            self.assertFalse((paths.egress / "E3_model_comparison.csv").exists())
            self.assertFalse((paths.internal / "RT_INTERNAL_split_membership.csv").exists())
            summary = (paths.egress / "SUMMARY.txt").read_text(encoding="utf-8")
            self.assertIn("RUN 1 MATCHING SUMMARY", summary)
            self.assertIn("No satisfaction model was trained", summary)
            state = json.loads(
                (paths.internal / STATE_FILENAME).read_text(encoding="utf-8")
            )
            self.assertIsNone(state["model_only_acceptance"])
            self.assertEqual(state["model_artifact_fingerprints"], {})

            pair_file = paths.internal / "RT_INTERNAL_match_pairs_1000.csv"
            pairs = pd.read_csv(pair_file, dtype="string", keep_default_na=False)
            pairs["review_decision"] = "correct"
            pairs.to_csv(pair_file, index=False, encoding="utf-8-sig")
            review_output = root / "diagnostic review"
            review = review_completed_sample(
                review_file=pair_file,
                settings_path=ROOT / "settings.toml",
                output_dir=review_output,
            )
            self.assertTrue(review.match_review.gate_passed)
            self.assertFalse(review.combined_passed)
            self.assertTrue(
                (review_output / "MATCH_REVIEW_STATUS.txt").is_file()
            )
            self.assertFalse((review_output / "FINAL_STATUS.txt").exists())

    def test_short_names_are_not_used_as_substring_disclosure_needles(self) -> None:
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

    def test_digest_is_stable_across_csv_typing_and_order(self) -> None:
        frame = _review_frame()
        expected = review_sample_digest(frame)
        shuffled = frame.sample(frac=1, random_state=7)[list(reversed(frame.columns))]
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "review.csv"
            shuffled.to_csv(path, index=False, encoding="utf-8-sig")
            loaded = pd.read_csv(path, dtype="string", keep_default_na=False)
        self.assertEqual(review_sample_digest(loaded), expected)

    def test_date_digest_equates_iso_and_uk_but_rejects_day_month_swap(self) -> None:
        frame = pd.DataFrame(
            {
                "review_row_id": ["R-0001"],
                "JudgmentDate": ["2024-01-02"],
                "review_decision": ["correct"],
                "review_notes": [""],
            }
        )
        uk_equivalent = frame.copy()
        uk_equivalent["JudgmentDate"] = "02/01/2024"
        swapped = frame.copy()
        swapped["JudgmentDate"] = "01/02/2024"

        expected = review_sample_digest(frame)
        self.assertEqual(review_sample_digest(uk_equivalent), expected)
        self.assertNotEqual(review_sample_digest(swapped), expected)

    def test_locked_review_combines_match_and_saved_model_gates(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            review_file = _write_review_case(root)
            output = root / "aggregate review"
            with patch(
                "recovery.run.fit_evaluate_models",
                side_effect=AssertionError("review must not refit"),
            ):
                result = review_completed_sample(
                    review_file=review_file,
                    settings_path=ROOT / "settings.toml",
                    output_dir=output,
                )
            self.assertTrue(result.match_review.gate_passed)
            self.assertTrue(result.combined_passed)
            self.assertTrue((output / "E2_review_quality.csv").is_file())
            self.assertTrue((output / "MATCH_REVIEW_STATUS.json").is_file())
            self.assertTrue((output / "FINAL_STATUS.json").is_file())
            validate_egress(output)

    def test_xlsx_round_trip_preserves_review_provenance(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_file = _write_review_case(root)
            frame = pd.read_csv(csv_file, dtype="string", keep_default_na=False)
            # Exercise real Excel date cells, not only ISO strings stored in an
            # XLSX container. The ambiguous 2 January date must remain Jan 2.
            frame["JudgmentDate"] = pd.to_datetime(
                frame["JudgmentDate"], format="%Y-%m-%d"
            )
            xlsx_file = csv_file.with_suffix(".xlsx")
            frame.to_excel(xlsx_file, index=False)
            result = review_completed_sample(
                review_file=xlsx_file,
                settings_path=ROOT / "settings.toml",
                output_dir=root / "xlsx review result",
            )
            self.assertTrue(result.match_review.gate_passed)

    def test_diagnostic_review_reports_matching_without_model_status(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            review_file = _write_review_case(root, stage="diagnostic")
            result = review_completed_sample(
                review_file=review_file,
                settings_path=ROOT / "settings.toml",
                output_dir=root / "review result",
            )
            self.assertTrue(result.match_review.gate_passed)
            self.assertFalse(result.combined_passed)
            self.assertIn(
                "satisfaction_model_deferred",
                result.combined_reasons,
            )
            output = root / "review result"
            self.assertTrue((output / "MATCH_REVIEW_STATUS.json").is_file())
            self.assertFalse((output / "FINAL_STATUS.json").exists())

    def test_review_rejects_changed_analysis_egress(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            review_file = _write_review_case(root, stage="diagnostic")
            egress = review_file.parent.parent / "egress_candidate"
            unexpected = egress / "unexpected" / "safe.txt"
            unexpected.parent.mkdir()
            unexpected.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(
                RunFailure, "egress files differ"
            ):
                review_completed_sample(
                    review_file=review_file,
                    settings_path=ROOT / "settings.toml",
                    output_dir=root / "review result",
                )

    def test_locked_run_requires_explicit_extract_date(self) -> None:
        with self.assertRaisesRegex(RunFailure, "explicit RT extract date"):
            analyze(
                stage="locked",
                judgments_path="not opened.csv",
                companies_house_path="not opened.zip",
                observation_date=None,
                settings_path=ROOT / "settings.toml",
                output_base="not created",
            )

    def test_unsafe_final_e5_leaves_egress_empty(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = make_synthetic_bundle(1_600, include_prior_rows=False)
            judgments, companies, _ = write_bundle(
                bundle,
                root / "inputs",
                excel=False,
            )

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
                    run_id="atomic_failure",
                )
            egress = root / "outputs" / "diagnostic_atomic_failure" / "egress_candidate"
            self.assertEqual(list(egress.iterdir()), [])

    def test_immutable_sample_edit_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            review_file = _write_review_case(root)
            changed = pd.read_csv(review_file, dtype="string", keep_default_na=False)
            changed.loc[0, "source_company_name"] = "CHANGED SOURCE"
            changed.to_csv(review_file, index=False, encoding="utf-8-sig")
            with self.assertRaises(RunFailure):
                review_completed_sample(
                    review_file=review_file,
                    settings_path=ROOT / "settings.toml",
                    output_dir=root / "review result",
                )

    def test_review_state_stage_tampering_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            review_file = _write_review_case(root, stage="diagnostic")
            state_file = review_file.parent / STATE_FILENAME
            state = json.loads(state_file.read_text(encoding="utf-8"))
            state["stage"] = "locked"
            state_file.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaises(RunFailure):
                review_completed_sample(
                    review_file=review_file,
                    settings_path=ROOT / "settings.toml",
                    output_dir=root / "tampered result",
                )

    def test_review_cli_exit_code_tracks_the_auto_match_gate(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            review_file = _write_review_case(root)
            frame = pd.read_csv(review_file, dtype="string", keep_default_na=False)
            auto = frame.index[frame["review_tier"].eq("auto")][:11]
            frame.loc[auto, "review_decision"] = "incorrect"
            frame.to_csv(review_file, index=False, encoding="utf-8-sig")
            code = main(
                [
                    "review",
                    "--review-file",
                    str(review_file),
                    "--settings",
                    str(ROOT / "settings.toml"),
                    "--output-dir",
                    str(root / "failed gate"),
                ]
            )
            self.assertEqual(code, 3)

    def test_final_status_suppresses_small_auto_outcome_detail(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            review_file = _write_review_case(root)
            frame = pd.read_csv(review_file, dtype="string", keep_default_na=False)
            first_auto = frame.index[frame["review_tier"].eq("auto")][0]
            frame.loc[first_auto, "review_decision"] = "incorrect"
            frame.to_csv(review_file, index=False, encoding="utf-8-sig")
            output = root / "suppressed review result"

            result = review_completed_sample(
                review_file=review_file,
                settings_path=ROOT / "settings.toml",
                output_dir=output,
            )

            self.assertTrue(result.match_review.gate_passed)
            status = json.loads(
                (output / "FINAL_STATUS.json").read_text(encoding="utf-8")
            )
            self.assertTrue(status["auto_quality_detail_suppressed"])
            self.assertIsNone(status["review_decision_fingerprint"])
            self.assertIsNone(status["auto_observed_precision"])
            self.assertIsNone(status["auto_wilson_lower_95"])
            self.assertIsNone(status["auto_wilson_upper_95"])
            text = (output / "FINAL_STATUS.txt").read_text(encoding="utf-8")
            self.assertIn("Auto quality detail suppressed: yes", text)
            self.assertNotIn("0.998000", text)
            self.assertIn("SUPPRESSED (small nonzero outcome cell)", text)
            validate_egress(output)


if __name__ == "__main__":
    unittest.main()
