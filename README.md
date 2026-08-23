# Registry Trust data check

The program runs on the local computer. It does not upload RT data.

## What you need

- 64-bit Python 3.13 or 3.14, with **Add Python to PATH** selected
- the full RT CSV or XLSX, with no columns removed or renamed
- the dated Companies House Basic Company Data CSV or ZIP
- the RT extract date and Companies House file date

## Run it

1. [Download the program](https://github.com/edwinward3/recovery-analysis/archive/refs/heads/main.zip) and select **Extract All**.
2. Open the extracted folder and double-click `RUN.bat`.
3. Drag the RT file into the black window and press Enter.
4. Drag the Companies House file into the window and press Enter.
5. Enter the two requested dates as `YYYY-MM-DD`.
6. Wait for **RUN COMPLETE**.

The first run may download the required Python packages. It does not send either
data file over the internet.

When the run finishes, it shows **SEND THIS FOLDER TO EDWIN**. Zip that folder
and send the ZIP to Edwin.

If the program displays **STOP**, send Edwin the complete message. Do not alter
the data or remove columns to work around it.
