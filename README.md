# Recovery analysis

Matches a court judgment file to Companies House data and reports the matching coverage.
The normal Run 1 does not fit the payment model.

## Run it

1. Check that Python 3.13 or 3.14 is installed and added to PATH.
2. Double-click `RUN.bat`.
3. Drag in your judgment file, then the Companies House file, when prompted.
4. Wait about 45 minutes.
5. Open `SUMMARY.txt` from the location shown when it finishes.

The Companies House file is the free "BasicCompanyDataAsOneFile" download from
https://download.companieshouse.gov.uk/. Leave it zipped.

It also creates a separate file containing 1,000 matching pairs.

## Your judgment file

A .csv or .xlsx with these columns (upper/lower case and order don't matter):

`ID`, `Date Inserted`, `JudgmentDate`, `JudgmentStatus`, `DefendantType`, `Jurisdiction`,
`Defendant Company Name`, `Defendant_Postcode`.

Dates as DD/MM/YYYY. `Amount`, `Defendant Trading Name` and `Defendant Address` are optional.
`Date Inserted` is the date RT put the judgment on the register.

## Reading it before you run it

Every file has a short line at the top saying what it does, and the code is commented throughout.
The analysis itself never uses the internet. On the first run, `RUN.bat` may use it to install
the required Python packages; neither data file is uploaded.

To watch it run on fake data, with no real file: `python -m recovery.selftest`.

## The files

Everything is in `recovery/`:

- `config.py` holds the fixed numbers used by the matching and model.
- `data.py` reads the two files and checks their columns and dates.
- `matching.py` matches the judgments and makes the 1,000-pair example file.
- `models.py` contains the later payment-model code. It is not used in the first run.
- `reporting.py` writes the summary and detail files; `disclosure.py` runs the final check.
- `run.py` holds the shared code the steps call.
- `selftest.py` runs the whole thing on fake data made by `synthetic.py`.
