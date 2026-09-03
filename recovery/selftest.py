"""Test matching and output generation using fake data."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence
import json
import sys

import pandas as pd

from .disclosure import validate_egress
from .matching import (
    ACCEPTED_LINKAGE_VALIDATION_FILENAME,
    UNMATCHED_LINKAGE_VALIDATION_FILENAME,
)
from .run import analyze
from .synthetic import make_synthetic_bundle, write_bundle


_MATCHING_EGRESS = {
    "SUMMARY.txt",
    "E1_data_audit.csv",
    "E1_data_funnel.csv",
    "E1_registration_gate.csv",
    "E1_registration_counts.csv",
    "E1_registration_statistics.csv",
    "E2_match_coverage.csv",
    "E2_unmatched_reasons.csv",
    "E2_match_methods.csv",
    "E2_linkage_profile.csv",
    "E2_linkage_checks.csv",
    "E2_population_comparison.csv",
    "E2_validation_sampling.csv",
    "E3_outcome_gate.csv",
    "E3_status_at_extract.csv",
    "E4_prediction_gate.csv",
    "E5_artifact_manifest.csv",
    "E5_output_dictionary.csv",
    "E5_run_log.csv",
    "E5_run_manifest.json",
}

_EXPECTED_TIERS = {
    "clean": "exact_unique",
    "punctuation": "exact_unique",
    "same_postcode_wrong_name": "unmatched",
    "trading": "exact_unique",
    "former": "exact_unique",
    "postcode_drift": "exact_unique",
    "same_postcode_ambiguous": "unmatched",
    "unmatched": "unmatched",
}


# Options and fake input files

def _parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run the synthetic recovery-analysis test")
    parser.add_argument(
        "--write-inputs",
        metavar="DIRECTORY",
        help="write synthetic inputs for a launcher or scale test and stop",
    )
    parser.add_argument("--n-companies", type=int, default=1_600)
    parser.add_argument("--format", choices=("xlsx", "csv"), default="xlsx")
    parser.add_argument(
        "--no-prior-rows",
        action="store_true",
        help="omit earlier history rows so n-companies equals judgment rows",
    )
    parser.add_argument(
        "--event-dates",
        action="store_true",
        help="include complete synthetic satisfaction and cancellation dates",
    )
    parser.add_argument("--settings", default="settings.toml")
    return parser


def _write_inputs(args: object) -> int:
    bundle = make_synthetic_bundle(
        args.n_companies,
        include_prior_rows=not args.no_prior_rows,
        include_event_dates=args.event_dates,
    )
    paths = write_bundle(
        bundle,
        args.write_inputs,
        excel=args.format == "xlsx",
    )
    for path in paths:
        print(path)
    print(
        f"SYNTHETIC INPUTS: PASS ({len(bundle.judgments):,} judgment rows; "
        f"{args.n_companies:,} target companies)"
    )
    return 0


# Check the known matches and output files

def _assert_match_truth(matches: pd.DataFrame, truth_path: Path) -> None:
    """Check every planted target while the match table is still in memory."""

    truth = pd.read_csv(
        truth_path,
        dtype={"ID": "string", "expected_company_number": "string"},
    )
    checked = truth.merge(matches, on="ID", how="left", validate="one_to_one")
    if checked["tier"].isna().any():
        raise AssertionError("one or more planted target judgments were not matched")

    matched = checked["tier"].eq("exact_unique")
    wrong = checked.loc[
        matched
        & checked["matched_company_number"].ne(checked["expected_company_number"])
    ]
    if not wrong.empty:
        raise AssertionError(f"{len(wrong)} exact matches have the wrong identity")

    observed = (
        checked.groupby("corruption_class", observed=True)["tier"]
        .agg(lambda values: tuple(sorted(set(values))))
        .to_dict()
    )
    expected = {key: (value,) for key, value in _EXPECTED_TIERS.items()}
    if observed != expected:
        raise AssertionError(
            f"corruption-class tiers differ from ground truth: {observed!r}"
        )


def _assert_accepted_sample_truth(run_root: Path, truth_path: Path) -> None:
    """Check identities in the 1,000-row matched review sample."""

    accepted = pd.read_csv(
        run_root / "working_files" / ACCEPTED_LINKAGE_VALIDATION_FILENAME,
        dtype={"ID": "string", "matched_company_number": "string"},
    )
    truth = pd.read_csv(
        truth_path,
        dtype={"ID": "string", "expected_company_number": "string"},
    )
    checked = accepted.merge(truth, on="ID", how="left", validate="one_to_one")
    if checked["expected_company_number"].isna().any():
        raise AssertionError("one or more sampled IDs are absent from synthetic truth")
    wrong = checked["matched_company_number"].ne(checked["expected_company_number"])
    if wrong.any():
        raise AssertionError(f"{int(wrong.sum())} sampled matches have the wrong identity")


def _assert_outputs(run_root: Path) -> None:
    results = run_root / "results"
    actual = {path.name for path in results.iterdir() if path.is_file()}
    missing = sorted(_MATCHING_EGRESS - actual)
    if missing:
        raise AssertionError(f"missing aggregate outputs: {missing}")
    unexpected = sorted(actual - _MATCHING_EGRESS)
    if unexpected:
        raise AssertionError(f"unexpected outputs: {unexpected}")
    validate_egress(results)

    manifest = json.loads(
        (results / "E5_run_manifest.json").read_text(encoding="utf-8")
    )
    expected_working = set(manifest["artifact_manifest"]["working_files"])
    for relative in expected_working:
        if not (run_root / "working_files" / relative).is_file():
            raise AssertionError(f"manifest lists a missing working file: {relative}")
    actual_working = {
        path.relative_to(run_root / "working_files").as_posix()
        for path in (run_root / "working_files").rglob("*")
        if path.is_file()
    }
    if actual_working != expected_working:
        raise AssertionError(
            f"working files differ from the retained manifest: {actual_working!r}"
        )
    if (run_root / ".aggregate_staging").exists():
        raise AssertionError("aggregate staging was retained after a successful run")

    accepted = pd.read_csv(
        run_root / "working_files" / ACCEPTED_LINKAGE_VALIDATION_FILENAME
    )
    if len(accepted) != 1_000:
        raise AssertionError(
            f"accepted-link validation file has {len(accepted)} rows, expected 1,000"
        )
    if set(accepted["tier"]) != {"exact_unique"}:
        raise AssertionError("accepted-link validation file contains a non-exact match")
    unmatched = pd.read_csv(
        run_root / "working_files" / UNMATCHED_LINKAGE_VALIDATION_FILENAME
    )
    if unmatched.empty or set(unmatched["tier"]) != {"unmatched"}:
        raise AssertionError("unmatched validation file is empty or contaminated")

    coverage = pd.read_csv(results / "E2_match_coverage.csv")
    funnel = pd.read_csv(results / "E1_data_funnel.csv").set_index("stage")
    total_stage = (
        "matching_decisions" if "matching_decisions" in funnel.index else "judgments_read"
    )
    matching_rows = int(funnel.loc[total_stage, "rows"])
    if int(coverage["rows"].sum()) != matching_rows:
        raise AssertionError("E2 coverage does not account for every matching decision")

def _run_full(args: object) -> int:
    if args.n_companies < 1_600:
        raise ValueError("the full self-test requires at least 1,600 companies")
    bundle = make_synthetic_bundle(
        args.n_companies,
        include_prior_rows=not args.no_prior_rows,
        include_event_dates=args.event_dates,
    )
    with TemporaryDirectory(prefix="recovery-selftest-") as temporary:
        root = Path(temporary)
        judgments, companies, truth = write_bundle(
            bundle,
            root / "inputs !",
            excel=args.format == "xlsx",
        )
        run = analyze(
            judgments_path=judgments,
            companies_house_path=companies,
            observation_date=bundle.observation_date,
            settings_path=args.settings,
            output_base=root / "outputs !",
            run_id="selftest",
            _match_validator=lambda matches: _assert_match_truth(matches, truth),
        )
        _assert_outputs(run.root)
    print(
        "SELF-TEST: PASS. Matching, review samples and output checks passed."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.write_inputs:
            return _write_inputs(args)
        return _run_full(args)
    except Exception as exc:
        print(f"SELF-TEST: FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
