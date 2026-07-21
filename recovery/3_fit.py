"""Step 3. Work out the facts about each company, train the model, write the model files."""

import argparse
import os
import sys

# engine.py is in this folder; this line lets the "import engine" below find it
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine


def main() -> int:
    ap = argparse.ArgumentParser(description="Probability-of-payment fit.")
    ap.add_argument("--fold", required=True, help="audited judgment extract (.csv/.xlsx)")
    ap.add_argument("--ch", required=True, help="Companies House file (.zip or .csv)")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    # same inputs and match as step 2; step 2's saved match is reused, not worked out again.
    fold = engine.read_extract(args.fold)
    try:
        index = engine.load_ch_index(args.ch)
    except ValueError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2

    matched = engine.get_matched(fold, index, args.fold, args.ch)
    # turn the matches into the facts about each company, then into what the model learns from
    feats = engine.build_features(matched, fold, index)
    labelled = engine.build_labelled(fold)
    if labelled["counts"]["seasoned_primary"] < 2 * 50:
        # too few rows to trust the score; say so rather than print a flattering number
        print("thin primary population, AUC may be unstable", file=sys.stderr)

    # train the model and write the model files (how accurate it is) into the outputs folder
    result = engine.fit_pfull(feats, labelled["primary"])
    engine.write_model_files(result, labelled, args.outdir)

    if result.evaluable:
        print(f"AUC (out-of-time holdout): {result.auc_oot:.4f}")
    else:
        print("AUC: not computable (holdout had a single class or was empty)")
    print(
        f"fit rows: {labelled['counts']['seasoned_primary']}; "
        f"cancelled group: {labelled['counts']['cancelled_stratum']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
