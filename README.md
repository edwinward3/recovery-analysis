# Recovery analysis

Run 1 links the full Registry Trust extract to Companies House and shows how
well the matching works. Run 2 is the later satisfaction analysis; do not use
it until RT and Edwin have agreed what that analysis should mean.

Raw records and named matches stay inside RT. Nothing is authorised for
external use unless RT approves it.

## Run 1: check the matching

1. On GitHub, select **Code**, then **Download ZIP**, and extract the folder.
2. Check that 64-bit Python 3.13 or 3.14 is installed and added to `PATH`.
3. Double-click `RUN.bat`, which starts Run 1.
4. Drag in the full RT judgment CSV/XLSX and Companies House CSV/ZIP.
5. Enter the date on which the fresh RT extract was produced.
6. Open the newest folder beneath `outputs` when the run finishes.

On first use, `RUN.bat` downloads the fixed package versions in
`requirements.lock`. That is the only internet use; the data files stay local
and are never uploaded. If PyPI is blocked, ask for the optional `wheels`
folder and use the same launcher.

Use the free Companies House `BasicCompanyDataAsOneFile` download and leave it
zipped: <https://download.companieshouse.gov.uk/en_output.html>.

Run 1 uses the full extract for matching and does not train a satisfaction
model. It reports results by defendant type, because Companies House covers
companies and a single all-row match rate can be misleading. It shows coverage,
unmatched reasons and 1,000 proposed pairs for RT to check.

Match rate measures coverage. The pair review measures whether the links are
actually correct.

## Check the 1,000 pairs

Open `RT_INTERNAL_match_pairs_1000.csv` in the run's `rt_internal` folder. For
every row, enter exactly `correct`, `incorrect` or `uncertain` in
`review_decision`. Do not change the sampled rows, filename, allocation columns
or companion files. `uncertain` is treated as not confirmed.

Save the file in its original folder. Double-click `CHECK_MATCH_REVIEW.bat`,
drag in the completed file and choose where to put the aggregate result.

The named pairs remain RT-internal. If the review shows a genuine matching
problem, the rule can be corrected and Run 1 repeated. Matching rules should
not be weakened merely to raise the headline percentage.

## Run 2: deferred

The satisfaction code is included so it can be tested in advance, but the
double-click launcher currently starts Run 1 only. Run 2 will be enabled after
RT and Edwin agree what `Satisfied` and `Unsatisfied` mean, the observation
date, the eligible age range and the intended population. It will then rematch
from scratch, draw a new review sample and run the locked models.

## RT judgment file

Required columns (case and order do not matter):

`ID`, `Date Inserted`, `JudgmentDate`, `JudgmentStatus`, `DefendantType`,
`Jurisdiction`, `Defendant Company Name`, `Defendant_Postcode`.

`Amount`, `Defendant Trading Name` and `Defendant Address` are optional.
`Date Inserted` is the register-entry date, not necessarily the date on which
the status was observed.

## Outputs

- `egress_candidate` contains aggregate, disclosure-checked reports for RT to
  review, including `SUMMARY.txt`.
- `rt_internal` contains named matches and other confidential working files.
  It must remain inside RT.

Nothing is sent automatically. RT should approve every egress candidate before
returning or using it.

## The files

- `data.py` reads and checks the input files.
- `matching.py` links defendants to companies and selects review pairs.
- `review.py` checks RT's completed pair review.
- `models.py` contains the deferred Run 2 satisfaction analysis.
- `reporting.py` writes reports; `disclosure.py` checks what may leave RT.
- `run.py` coordinates the process; `selftest.py` runs it on fake data.

Each source file starts with its purpose and data sensitivity. The analysis
code makes no network or shell calls. To test it without real RT data, run
`python -m recovery.selftest`.
