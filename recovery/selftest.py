"""Self-test. Runs the whole thing on fake data, no real file needed. Run this first.

It checks the packages are installed, builds a tiny fake Companies House file and judgment file,
runs every step into a temporary folder, and confirms the disclosure gate is clean. Prints
SELF-TEST: PASS only if all of that worked. No internet, no other programs, no real data.
"""

import os
import sys
import tempfile
from pathlib import Path

# engine.py is in this folder; this line lets the "import engine" below find it
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    print(f"python: {sys.version.split()[0]}  platform: {sys.platform}")
    try:
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import rapidfuzz  # noqa: F401
        import scipy  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as exc:
        print(f"SELF-TEST: FAIL, missing dependency: {exc}", file=sys.stderr)
        return 1

    import engine

    try:
        # 60 fake companies and 800 fake judgments
        names = [f"COMPANY {i} LIMITED" for i in range(60)]
        ch = engine.generate_ch_bulk(names, seed=0)
        fold, _ = engine.generate_fold(
            n_rows=800, ch_names=names, seed=0, ch_bulk=ch, plant_signal=True
        )
        # temp folder, auto-deleted on exit
        with tempfile.TemporaryDirectory() as d:
            outdir = Path(d) / "outputs"
            outdir.mkdir()
            ch_path = Path(d) / "ch.csv"
            ch.to_csv(ch_path, index=False)
            fold_path = Path(d) / "fold.csv"
            # write UK dates (day/month/year) like a real file; the copy in memory keeps its
            # date columns for the steps below.
            fold_to_write = fold.copy()
            for col in ("Date Inserted", "JudgmentDate"):
                fold_to_write[col] = fold_to_write[col].dt.strftime("%d/%m/%Y")
            fold_to_write.to_csv(fold_path, index=False)

            engine.run_audit(fold_path, outdir)
            index = engine.load_ch_index(ch_path)
            matched = engine.match_judgments(fold, index)
            engine.match_report(fold, matched)["tier_counts"].to_csv(
                outdir / "match_counts.csv", index=False
            )
            feats = engine.build_features(matched, fold, index)
            labelled = engine.build_labelled(fold)
            result = engine.fit_pfull(feats, labelled["primary"])
            engine.write_model_files(result, labelled, outdir)
            engine.write_breakdown_files(engine.breakdown_report(result.holdout, feats, fold, labelled), outdir)
            engine.write_run_log([{"stage": "self-test", "rows": len(fold), "note": "ok"}], outdir)
            disc = engine.apply_disclosure(outdir)
            if disc["violations"]:
                print(f"SELF-TEST: FAIL, disclosure violations: {disc['violations']}", file=sys.stderr)
                return 1
            print(f"backend: {result.backend}  AUC: {result.auc_oot:.4f}")
    except Exception as exc:  # noqa: BLE001 (a self-test must report ANY failure, not crash silently)
        print(f"SELF-TEST: FAIL, {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("SELF-TEST: PASS. Whole chain ran, disclosure gate clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
