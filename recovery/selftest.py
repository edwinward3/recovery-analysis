"""Self-test. Runs matching and the model on fake data before any RT file is opened.

It checks the known matches, the four model fits, the pair files and the final
output check. ``--write-inputs`` saves the fake files for the Windows tests.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence
import json
import sys

import pandas as pd

from .disclosure import validate_egress
from .run import PAIR_FILENAME, analyze
from .synthetic import make_synthetic_bundle, write_bundle


_MATCHING_EGRESS = {
    "SUMMARY.txt",
    "E1_data_audit.csv",
    "E1_data_funnel.csv",
    "E2_match_coverage.csv",
    "E2_unmatched_reasons.csv",
    "E2_match_methods.csv",
    "E2_match_by_defendant_type.csv",
    "E2_match_by_judgment_vintage.csv",
    "E2_incorporation_guards.csv",
    "E5_run_log.csv",
    "E5_run_manifest.json",
}

_MODEL_EGRESS = {
    "E3_model_comparison.csv",
    "E3_calibration.csv",
    "E3_feature_effects.csv",
    "E3_split_counts.csv",
    "E4_sensitivities.csv",
    "E4_lift.csv",
    "E4_limitations.txt",
}

_EXPECTED_TIERS = {
    "clean": "auto",
    "punctuation": "auto",
    "review_name": "review",
    "trading": "auto",
    "former": "auto",
    "postcode_drift": "fallback_review",
    "ambiguous": "review",
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
    parser.add_argument("--settings", default="settings.toml")
    return parser


def _write_inputs(args: object) -> int:
    bundle = make_synthetic_bundle(
        args.n_companies,
        include_prior_rows=not args.no_prior_rows,
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

    proposed = checked["tier"].ne("unmatched")
    wrong = checked.loc[
        proposed
        & checked["matched_company_number"].ne(checked["expected_company_number"])
    ]
    if not wrong.empty:
        raise AssertionError(f"{len(wrong)} proposed matches have the wrong identity")

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


def _assert_pair_truth(run_root: Path, truth_path: Path) -> None:
    """Check the identities in the retained 1,000-pair scale-test sample."""

    pairs = pd.read_csv(
        run_root / "working_files" / PAIR_FILENAME,
        dtype={"ID": "string", "matched_company_number": "string"},
    )
    truth = pd.read_csv(
        truth_path,
        dtype={"ID": "string", "expected_company_number": "string"},
    )
    checked = pairs.merge(truth, on="ID", how="left", validate="one_to_one")
    if checked["expected_company_number"].isna().any():
        raise AssertionError("one or more sampled IDs are absent from synthetic truth")
    wrong = checked["matched_company_number"].ne(checked["expected_company_number"])
    if wrong.any():
        raise AssertionError(f"{int(wrong.sum())} sampled matches have the wrong identity")


def _assert_outputs(run_root: Path, *, stage: str) -> None:
    results = run_root / "results"
    actual = {path.name for path in results.iterdir() if path.is_file()}
    expected = _MATCHING_EGRESS | (_MODEL_EGRESS if stage == "locked" else set())
    missing = sorted(expected - actual)
    if missing:
        raise AssertionError(f"missing aggregate outputs: {missing}")
    unexpected_models = sorted(_MODEL_EGRESS & actual) if stage == "diagnostic" else []
    if unexpected_models:
        raise AssertionError(
            f"matching-only diagnostic wrote model outputs: {unexpected_models}"
        )
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

    pairs = pd.read_csv(run_root / "working_files" / PAIR_FILENAME)
    if len(pairs) != 1_000:
        raise AssertionError(f"match-example file has {len(pairs)} rows, expected 1,000")
    allocation = pairs["tier"].value_counts().to_dict()
    expected_allocation = {"auto": 500, "review": 300, "fallback_review": 200}
    if allocation != expected_allocation:
        raise AssertionError(f"pair allocation differs: {allocation!r}")

    coverage = pd.read_csv(results / "E2_match_coverage.csv")
    funnel = pd.read_csv(results / "E1_data_funnel.csv").set_index("stage")
    total_stage = (
        "matching_decisions" if "matching_decisions" in funnel.index else "judgments_read"
    )
    matching_rows = int(funnel.loc[total_stage, "rows"])
    if int(coverage["rows"].sum()) != matching_rows:
        raise AssertionError("E2 coverage does not account for every matching decision")

    if stage == "locked":
        comparison = pd.read_csv(results / "E3_model_comparison.csv")
        if set(comparison["model"]) != {
            "prospective.logistic",
            "prospective.lightgbm",
            "snapshot_exploratory.logistic",
            "snapshot_exploratory.lightgbm",
        }:
            raise AssertionError("the four declared model fits were not all reported")
        split_rows = pd.read_csv(results / "E3_split_counts.csv")
        if set(split_rows["split"]) != {
            "train",
            "validation",
            "calibration",
            "test",
        }:
            raise AssertionError("the four declared partitions were not all reported")
    else:
        summary = (results / "SUMMARY.txt").read_text(encoding="utf-8")
        if "No satisfaction model was trained or assessed" not in summary:
            raise AssertionError("diagnostic summary does not state that modelling was skipped")

# Run both stages

def _run_full(args: object) -> int:
    if args.n_companies < 1_600:
        raise ValueError("the full self-test requires at least 1,600 companies")
    bundle = make_synthetic_bundle(
        args.n_companies,
        include_prior_rows=not args.no_prior_rows,
    )
    with TemporaryDirectory(prefix="recovery-selftest-") as temporary:
        root = Path(temporary)
        judgments, companies, truth = write_bundle(
            bundle,
            root / "inputs !",
            excel=args.format == "xlsx",
        )
        diagnostic = analyze(
            stage="diagnostic",
            judgments_path=judgments,
            companies_house_path=companies,
            observation_date=bundle.observation_date,
            settings_path=args.settings,
            output_base=root / "outputs !",
            run_id="selftest_diagnostic",
            _match_validator=lambda matches: _assert_match_truth(matches, truth),
        )
        _assert_outputs(diagnostic.root, stage="diagnostic")
        locked = analyze(
            stage="locked",
            judgments_path=judgments,
            companies_house_path=companies,
            observation_date=bundle.observation_date,
            settings_path=args.settings,
            output_base=root / "outputs !",
            run_id="selftest_locked",
            _match_validator=lambda matches: _assert_match_truth(matches, truth),
        )
        _assert_outputs(locked.root, stage="locked")
    print(
        "SELF-TEST: PASS. Matching-only Run 1, locked Run 2, planted match "
        "identities, four model fits and the disclosure boundary all passed."
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
