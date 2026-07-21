"""Step 4. Write SUMMARY.txt and the breakdown and log files, then check nothing identifying leaves."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

# engine.py is in this folder; this line lets the "import engine" below find it
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine


def main() -> int:
    ap = argparse.ArgumentParser(description="Results summary and disclosure check.")
    ap.add_argument("--fold", required=True)
    ap.add_argument("--ch", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    fold = engine.read_extract(args.fold)
    try:
        index = engine.load_ch_index(args.ch)
    except ValueError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2

    matched = engine.get_matched(fold, index, args.fold, args.ch)
    feats = engine.build_features(matched, fold, index)
    labelled = engine.build_labelled(fold)
    result = engine.fit_pfull(feats, labelled["primary"])

    counts = labelled["counts"]
    primary = labelled["primary"]
    n_auto = int((matched["match_tier"] == "auto").sum())
    n_review = int((matched["match_tier"] == "review").sum())
    n_pop = int(matched["in_population"].sum())
    in_pop_matched = int(
        (matched["in_population"] & matched["match_tier"].isin(["auto", "review"])).sum()
    )
    rate = (100.0 * in_pop_matched / n_pop) if n_pop else 0.0
    n_sat = int(primary["p_full"].sum()) if len(primary) else 0
    n_unsat = int(len(primary)) - n_sat
    amounts = pd.to_numeric(fold["Amount"], errors="coerce")
    has_amount = bool(amounts.notna().any())
    median_amt = float(amounts.median()) if has_amount else 0.0
    total_amt = float(amounts.fillna(0).sum()) if has_amount else 0.0

    ho = result.holdout
    if result.evaluable and not ho.empty:
        y = ho["y_true"].astype(float)
        p = ho["proba"].astype(float)
        base_rate = f"{float(y.mean()):.3f}"
        brier = f"{float(((p - y) ** 2).mean()):.4f}"
        pr_auc = f"{float(average_precision_score(y, p)):.3f}" if y.nunique() > 1 else "n/a"
        cal = result.calibration
        cal_err = (
            f"{float((cal['bin_mean_pred'] - cal['bin_frac_pos']).abs().mean()):.3f}"
            if not cal.empty
            else "n/a"
        )
        auc = f"{result.auc_oot:.3f}"
        mean_pfull = f"{result.mean_pred_pfull:.3f}"
        below_floor = "yes" if result.below_floor else "no"
    else:
        base_rate = brier = pr_auc = cal_err = auc = mean_pfull = below_floor = "n/a"

    fi = result.feature_importance or {}
    fi_total = sum(fi.values()) or 1.0
    fi_sorted = sorted(fi.items(), key=lambda kv: -kv[1])
    fi_rows = [(f"  {k}", f"{v / fi_total:.3f}") for k, v in fi_sorted]

    rows = [
        ("Judgments read", len(fold)),
        ("In scope", n_pop),
        ("Matched", in_pop_matched),
        ("Match rate", f"{rate:.0f}%"),
        ("  Auto", n_auto),
        ("  Review", n_review),
        ("Trainable", counts["seasoned_primary"]),
        ("  Satisfied", n_sat),
        ("  Unsatisfied", n_unsat),
        ("Cancelled (held out)", counts["cancelled_stratum"]),
        ("Scotland (held out)", counts["scotland_stratum"]),
        ("", ""),
        ("Model: probability of full payment", ""),
        ("  AUC", auc),
        ("  PR-AUC", pr_auc),
        ("  Brier", brier),
        ("  Calibration error", cal_err),
        ("  Base rate P(full)", base_rate),
        ("  Below floor (AUC<0.70)", below_floor),
        ("  Mean predicted P(full)", mean_pfull),
        ("  Train rows", result.n_train),
        ("  Test rows", result.n_holdout),
        ("  Fitted with", result.backend),
        ("", ""),
        ("Feature weights (share of total)", ""),
        *fi_rows,
        ("", ""),
        ("Median judgment (£)", f"{median_amt:,.0f}" if has_amount else "not provided"),
        ("Total judgment value (£)", f"{total_amt:,.0f}" if has_amount else "not provided"),
        ("", ""),
        ("Caveats:", ""),
        (f"  Features are measured at the snapshot date ({engine.SNAPSHOT_DATE}), not the", ""),
        ("  judgment date, so the AUC is an optimistic upper bound.", ""),
        ("  AUC below 0.70 is a minimum-usability floor, not an acceptance bar.", ""),
    ]
    lines = ["SUMMARY", "======="]
    if not result.evaluable or in_pop_matched == 0:
        lines += [
            "",
            "*** CHECK THIS RUN: the model was not evaluable or nothing matched -",
            "*** the numbers below are NOT usable. See unmatched_reasons.csv. ***",
        ]
    for name, val in rows:
        lines.append("" if name == "" else f"{name:<32}{val}")
    # write the headline summary into the outputs folder
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # the detailed breakdowns and the run log. the next round of work is based on these,
    # so they are kept; the one-page SUMMARY.txt is not enough on its own.
    engine.write_breakdown_files(engine.breakdown_report(result.holdout, feats, fold, labelled), outdir)
    engine.write_run_log(
        [
            {"stage": "audit", "rows": len(fold), "note": "judgments read"},
            {"stage": "match", "rows": in_pop_matched, "note": f"{rate:.0f}% of in-scope matched"},
            {"stage": "fit", "rows": result.n_train, "note": f"fitted with {result.backend}"},
        ],
        outdir,
    )

    # the disclosure gate: refuse the run if any output could identify a person or company,
    # and blank counts under 10. the last check before results leave the machine.
    disc = engine.apply_disclosure(args.outdir)
    if disc["violations"]:
        print("DISCLOSURE GATE FAILED: do not egress:", file=sys.stderr)
        for v in disc["violations"]:
            print(f"  {v}", file=sys.stderr)
        return 3

    # all the output files passed the check
    print(f"disclosure: zero violations; results are in {outdir}")
    if not result.evaluable or in_pop_matched == 0:
        print(
            "CHECK THIS RUN: the model could not be evaluated or nothing matched. The numbers "
            "above are not usable. Keep unmatched_reasons.csv to see why.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
