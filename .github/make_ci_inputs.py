"""For the automated test only: writes a small fake judgment file and Companies House file so
RUN.bat can be run start to finish. Not used on real data."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "recovery"))
import engine

names = [f"COMPANY {i} LIMITED" for i in range(60)]
ch = engine.generate_ch_bulk(names, seed=0)
fold, _ = engine.generate_fold(n_rows=800, ch_names=names, seed=0, ch_bulk=ch, plant_signal=True)

ch.to_csv("ch.csv", index=False)
for col in ("Date Inserted", "JudgmentDate"):
    fold[col] = fold[col].dt.strftime("%d/%m/%Y")
fold.to_csv("fold.csv", index=False)
print(f"wrote ch.csv ({len(ch)} companies) and fold.csv ({len(fold)} judgments)")
