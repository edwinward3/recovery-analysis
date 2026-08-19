"""Exercise the real offline pipeline using deterministic synthetic data only.

Inputs are generated fake RT and Companies House records. Outputs are either a
requested synthetic input bundle or a temporary diagnostic run that is deleted
on exit. The test checks planted match identities, corruption-class behaviour,
model/report production and the disclosure boundary. No network or shell calls
are made by this module and no confidential data is read.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence
import sys

import pandas as pd

from .disclosure import validate_egress
from .run import MATCH_FILENAME, PAIR_FILENAME, analyze
from .synthetic import make_synthetic_bundle, write_bundle


_EXPECTED_EGRESS = {
    "SUMMARY.txt",
    "E1_data_audit.csv",
    "E1_data_funnel.csv",
    "E2_match_coverage.csv",
    "E2_unmatched_reasons.csv",
    "E2_match_methods.csv",
    "E2_match_by_defendant_type.csv",
    "E2_match_by_judgment_vintage.csv",
    "E2_incorporation_guards.csv",
    "E3_model_comparison.csv",
    "E3_calibration.csv",
    "E3_feature_effects.csv",
    "E3_split_counts.csv",
    "E4_sensitivities.csv",
    "E4_lift.csv",
    "E4_limitations.txt",
    "E5_run_log.csv",
    "E5_run_manifest.json",
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


def _assert_match_truth(run_root: Path, truth_path: Path) -> None:
    matches = pd.read_csv(
        run_root / "rt_internal" / MATCH_FILENAME,
        dtype={"ID": "string", "matched_company_number": "string"},
    )
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


def _assert_outputs(run_root: Path, truth_path: Path) -> None:
    egress = run_root / "egress_candidate"
    actual = {path.name for path in egress.iterdir() if path.is_file()}
    missing = sorted(_EXPECTED_EGRESS - actual)
    if missing:
        raise AssertionError(f"missing aggregate outputs: {missing}")
    validate_egress(egress)

    pairs = pd.read_csv(run_root / "rt_internal" / PAIR_FILENAME)
    if len(pairs) != 1_000:
        raise AssertionError(f"match-review file has {len(pairs)} rows, expected 1,000")
    allocation = pairs["review_tier"].value_counts().to_dict()
    expected_allocation = {"auto": 500, "review": 300, "fallback_review": 200}
    if allocation != expected_allocation:
        raise AssertionError(f"review allocation differs: {allocation!r}")

    comparison = pd.read_csv(egress / "E3_model_comparison.csv")
    if set(comparison["model"]) != {
        "prospective.logistic",
        "prospective.lightgbm",
        "snapshot_exploratory.logistic",
        "snapshot_exploratory.lightgbm",
    }:
        raise AssertionError("the four declared model fits were not all reported")
    split_rows = pd.read_csv(egress / "E3_split_counts.csv")
    if set(split_rows["split"]) != {"train", "validation", "calibration", "test"}:
        raise AssertionError("the four declared partitions were not all reported")

    _assert_match_truth(run_root, truth_path)


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
        paths = analyze(
            stage="diagnostic",
            judgments_path=judgments,
            companies_house_path=companies,
            observation_date=bundle.observation_date,
            settings_path=args.settings,
            output_base=root / "outputs !",
            run_id="selftest",
        )
        _assert_outputs(paths.root, truth)
    print(
        "SELF-TEST: PASS. Planted match identities, four model fits, E1-E5 "
        "and the disclosure boundary all passed."
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
