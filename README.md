# Recovery analysis

Matches a court judgment file to Companies House data and estimates how likely each judgment was
paid in full.

## Run it

1. Check that Python 3.13 or 3.14 is installed and added to PATH.
2. Double-click `RUN.bat`.
3. Drag in your judgment file, then the Companies House file, when prompted.
4. Wait about 45 minutes.
5. Open `outputs\SUMMARY.txt`.

The Companies House file is the free "BasicCompanyDataAsOneFile" download from
https://download.companieshouse.gov.uk/. Leave it zipped.

## Your judgment file

A .csv or .xlsx with these columns (upper/lower case and order don't matter):

`ID`, `Date Inserted`, `JudgmentDate`, `JudgmentStatus`, `DefendantType`, `Jurisdiction`,
`Defendant Company Name`, `Defendant_Postcode`.

Dates as DD/MM/YYYY. `Amount`, `Defendant Trading Name` and `Defendant Address` are optional.

## Reading it before you run it

Every file has a short line at the top saying what it does, and the code is commented throughout.
It never uses the internet and never runs another program, and it checks the outputs before finishing,
blanking anything that could identify a person or company.

To watch it run on fake data, with no real file: `python recovery\selftest.py`.

## The files

Everything is in `recovery/`:

- `1_audit.py` reads your judgment file and counts what is in it.
- `2_match.py` matches the judgments to Companies House. This is the slow step.
- `3_fit.py` builds the company features and fits the model.
- `4_results.py` writes the summary and diagnostics, then runs the disclosure check.
- `selftest.py` runs the whole thing on fake data.
- `engine.py` holds the shared code the four steps call.
