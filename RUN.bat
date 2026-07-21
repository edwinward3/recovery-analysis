@echo off
setlocal disabledelayedexpansion
REM Work in the folder this script sits in, so the Python setup, recovery\ and outputs are all created and read here.
cd /d "%~dp0"
title Recovery analysis

REM Sets up Python once, self-tests on fake data, then runs the four steps on your two files.
REM Special handling of "!" is turned off, so a dragged-in path that contains "!" is not damaged.

set "JUD=%~1"
set "CH=%~2"

REM .setup-ok is written only after the install succeeds, so a failed install is retried.
if exist ".venv\.setup-ok" goto ready

where python >nul 2>nul || (echo Python is not installed. Install Python 3.13 or 3.14 from python.org, tick "Add python.exe to PATH", then run this again. & echo. & pause & exit /b 1)
python -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3,13),(3,14)) else 1)" 2>nul || (echo Wrong Python version. Install Python 3.13 or 3.14 from python.org, tick "Add python.exe to PATH", then run this again. & echo. & pause & exit /b 1)

echo Setting up (first run only)...
if not exist ".venv\Scripts\python.exe" python -m venv .venv || goto setupfail
call ".venv\Scripts\activate.bat" || goto setupfail

if exist "wheels" (
  echo Installing the bundled packages. No internet needed.
  python -m pip install --no-index --find-links wheels -r requirements.lock -q || goto setupfail
) else (
  echo Installing packages from PyPI. This is the only step that uses the internet.
  python -m pip install -r requirements.txt -q || goto setupfail
)
echo ok> ".venv\.setup-ok"
goto selftest

:setupfail
echo.
echo Setup failed. See the message above.
echo If this machine has no internet, or blocks it, ask us for the offline package bundle.
echo Nothing has left the machine.
echo.
if "%RECOVERY_NO_PAUSE%"=="" pause
exit /b 1

:ready
call ".venv\Scripts\activate.bat"

:selftest
echo Running the self-test on fake data...
python recovery\selftest.py || goto stop

:files
echo.
if not "%~1"=="" goto dequote
set /p "JUD=Drag the court-judgment file here and press Enter: "
set /p "CH=Drag the Companies House file here and press Enter: "
:dequote
REM A dragged-in path arrives wrapped in double quotes; remove them so the path can be checked and opened.
set "JUD=%JUD:"=%"
set "CH=%CH:"=%"

if not exist "%JUD%" (echo Cannot find the judgment file: %JUD% & goto stop)
if not exist "%CH%" (echo Cannot find the Companies House file: %CH% & goto stop)

echo.
echo Reading judgments...
python recovery\1_audit.py --input "%JUD%" --outdir outputs || goto stop
echo Matching to Companies House. This takes about 45-55 minutes...
python recovery\2_match.py --fold "%JUD%" --ch "%CH%" --outdir outputs || goto stop
echo Fitting model...
python recovery\3_fit.py --fold "%JUD%" --ch "%CH%" --outdir outputs || goto stop
echo Writing outputs...
python recovery\4_results.py --fold "%JUD%" --ch "%CH%" --outdir outputs || goto stop

echo.
echo Done. Results are in the outputs folder. Open outputs\SUMMARY.txt.
echo.
if "%RECOVERY_NO_PAUSE%"=="" pause
exit /b 0

:stop
echo.
echo Stopped before finishing. Nothing has left the machine.
echo.
REM Always pause on an error so the message stays on screen.
if "%RECOVERY_NO_PAUSE%"=="" pause
exit /b 1
