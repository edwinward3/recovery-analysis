# Registry Trust data check

You need:

- the complete RT CSV or XLSX; do not remove any columns;
- the latest Companies House [company data as one file](https://download.companieshouse.gov.uk/en_output.html) — download `BasicCompanyDataAsOneFile-YYYY-MM-DD.zip` and leave it zipped;
- 64-bit Python 3.13 or 3.14 for Windows. If needed, get it from [python.org](https://www.python.org/downloads/windows/) and select **Add Python to PATH** during installation.

## Run

1. [Download this program](https://github.com/edwinward3/recovery-analysis/archive/refs/heads/main.zip) and select **Extract All**.
2. Open the extracted folder and double-click `RUN.bat`.
3. Drag in the complete RT file when asked.
4. Drag in the Companies House ZIP when asked.
5. Enter the RT extract date, then the date shown in the Companies House filename.
6. Wait for **RUN COMPLETE** and send Edwin the ZIP file marked **SEND_TO_EDWIN**.

Use the RT export exactly as produced. If it contains Satisfaction Date, Cancellation Date, Cancellation Reason, Status Effective Date or Snapshot Date, leave those columns in place. The program does not upload either source file. Keep the internet connected on the first run while it installs the required packages.

If the program shows **STOP**, send Edwin the complete message. Do not alter the data to get around the check.
