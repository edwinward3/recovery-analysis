# Recovery analysis

Offline linkage and research code for Registry Trust judgment records and the
Companies House live-company bulk snapshot.

## Diagnostic run

1. Install 64-bit Python 3.13 or 3.14 and add it to PATH.
2. Double-click `RUN.bat` and choose Run 1.
3. Supply the RT file, Companies House file, RT extract date, and Companies House
   snapshot date when prompted.

The diagnostic produces aggregate audit reports plus two outcome-blind review
files: 1,000 accepted exact links and a probability sample of up to 1,000
unmatched records. Both must be independently double-reviewed before development.

Run 2 stays unavailable in the double-click launcher. Development and the
single-use locked release require the supervised procedure in `STUDY_DESIGN.md`.
No key is needed for setup, diagnostic matching, or development; the release
custodian keeps the later approval key outside this repository.

## RT input

A CSV or XLSX containing:

`ID`, `Date Inserted`, `JudgmentDate`, `JudgmentStatus`, `DefendantType`,
`Jurisdiction`, `Defendant Company Name`, `Defendant_Postcode`.

`Amount`, `Defendant Trading Name`, and `Defendant Address` are optional. Known
satisfaction, cancellation, status-effective, and snapshot dates are preserved.
Unrecognised outcome/history fields stop the run instead of being discarded.

The Companies House input is the dated, zipped Basic Company Data snapshot from
https://download.companieshouse.gov.uk/. It contains live companies only.

## Safeguards

- The supplied no-event-date schema is analysed only as status at extract date.
- If event dates or historical snapshots appear, modelling stops for redesign.
- Development masks test outcomes and test class counts.
- The locked test requires a manifest-bound, one-use approval.
- Public reports are aggregate and disclosure checked; fitted weights remain local.
- The AUC 0.70 rule is internal, not a publication criterion.

Run the fake-data check with `python -m recovery.selftest`.

The full population, estimand, linkage protocol, model protocol, interpretation
rules, and release protocol are in `STUDY_DESIGN.md`. Claim and reviewer registers
are in `CLAIM_EVIDENCE_REGISTER.md` and `REVIEWER_OBJECTIONS.md`.
