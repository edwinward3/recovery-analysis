"""Step 2. Match the judgments to Companies House and write the match report."""

import argparse
import os
import sys
from pathlib import Path

# engine.py is in this folder; this line lets the "import engine" below find it
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine


def main() -> int:
    ap = argparse.ArgumentParser(description="Match judgments to Companies House.")
    ap.add_argument("--fold", required=True, help="audited judgment extract (.csv/.xlsx)")
    ap.add_argument("--ch", required=True, help="Companies House file (.zip or .csv)")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    # read the two input files: judgments, then Companies House
    fold = engine.read_extract(args.fold)
    try:
        index = engine.load_ch_index(args.ch)
    except ValueError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2

    # match the judgments to companies (reuses the saved match if these two files were matched before)
    matched = engine.get_matched(fold, index, args.fold, args.ch)
    report = engine.match_report(fold, matched)

    # write the two match reports into the outputs folder
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report["tier_counts"].to_csv(outdir / "match_counts.csv", index=False)
    report["unmatched_diagnosis"].to_csv(outdir / "unmatched_reasons.csv", index=False)

    n_auto = int((matched["match_tier"] == "auto").sum())
    n_review = int((matched["match_tier"] == "review").sum())
    n_pop = int(matched["in_population"].sum())
    print(f"in-population judgments: {n_pop}; auto: {n_auto}; review: {n_review}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
