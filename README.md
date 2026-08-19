# Recovery analysis

Links Registry Trust commercial judgments to Companies House and tests whether
information known at judgment can distinguish records later shown as
`Satisfied` rather than `Unsatisfied`.

This is an academic analysis. Raw records and named matches stay inside RT, and
nothing is authorised for external use unless RT approves it.

## Run it

1. Check that 64-bit Python 3.13 or 3.14 is installed and added to `PATH`.
2. Double-click `RUN.bat`.
3. Choose Run 1 or Run 2, then drag in the RT extract and Companies House file.
4. Enter the date on which the fresh RT extract was produced.
5. Open the newest folder beneath `outputs` when the run finishes.

On first use, `RUN.bat` downloads the fixed package versions in
`requirements.lock`. That is the only internet use: the RT and Companies House
files are processed locally and are never uploaded. If the RT machine cannot
access PyPI, the same launcher can use a separately supplied `wheels` folder.

Use the free Companies House `BasicCompanyDataAsOneFile` download and leave it
zipped: <https://download.companieshouse.gov.uk/en_output.html>.

## The two runs

**Run 1 — diagnostic.** The code reports why records did or did not match and
creates 1,000 proposed pairs for RT to check. Those checks tell us whether any
matching rule genuinely needs changing.

**Run 2 — locked.** After the matching rules are frozen, the code rematches from
scratch, creates a different 1,000-pair sample and produces the final model
results.

For each sample, RT enters `correct`, `incorrect` or `uncertain` in the
`review_decision` column, then double-clicks `CHECK_MATCH_REVIEW.bat` and selects
the completed file. `uncertain` is treated as incorrect. The final result cannot
pass unless the locked automatic matches pass the review gate.

## RT judgment file

CSV or XLSX, with these required columns (case and order do not matter):

`ID`, `Date Inserted`, `JudgmentDate`, `JudgmentStatus`, `DefendantType`,
`Jurisdiction`, `Defendant Company Name`, `Defendant_Postcode`.

`Amount`, `Defendant Trading Name` and `Defendant Address` are optional.

`Date Inserted` is used only to audit registration delay. Judgment age is
measured from `JudgmentDate` to the date of the fresh RT extract.

## What it does

- Matches by normalised postcode and company/trading name, including valid
  former Companies House names.
- Keeps automatic, review, different-postcode fallback and unmatched results
  separate. Match rate is coverage; the 1,000-pair check measures accuracy.
- Uses automatic matches for the primary E&W Corporate analysis and keeps the
  other populations as separate diagnostics.
- Fits a judgment-time model and a clearly marked exploratory model using
  present-day Companies House fields.
- Reports the full row funnel, matching results, model results, sensitivities
  and run record as E1–E5.

## Outputs

Each run creates two separate folders:

- `egress_candidate` contains only aggregate, disclosure-checked reports,
  including the one-page `SUMMARY.txt` and E1–E5 files.
- `rt_internal` contains named match pairs, the match table, partitions and
  fitted-model files. It must remain inside RT.

RT should review every egress candidate before returning it.

## Reading and testing the code

Each source file begins with a short explanation of its purpose and data
sensitivity. `data.py`, `matching.py` and `models.py` perform the analysis;
`reporting.py` and `disclosure.py` check the outputs; `review.py` checks the
completed match sample; and `run.py` coordinates the complete process.

The analysis package makes no network or shell calls. To test it without real
data, run `python -m recovery.selftest`.
