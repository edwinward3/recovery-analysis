"""Step 1. Read the judgment file, check the columns, write the count files."""

import argparse
import os
import sys

# engine.py is in this folder; this line lets the "import engine" below find it
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit the judgment extract.")
    ap.add_argument("--input", required=True, help="judgment extract (.xlsx or .csv)")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    # open the judgment file, check the columns, and write the count files into the outputs folder
    try:
        result = engine.run_audit(args.input, args.outdir)
    except ValueError as exc:
        # a missing required column, or an unreadable file, stops here with a plain message
        print(f"STOP: {exc}", file=sys.stderr)
        return 2

    print(f"rows: {result.n_rows}")
    print(f"Date Inserted constant: {result.date_inserted_constant}")
    # flag any unexpected status / type / jurisdiction value
    for col, d in result.value_sets.items():
        if d["unseen"]:
            print(f"UNSEEN {col}: {d['unseen']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
