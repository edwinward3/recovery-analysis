@rem Sets up Python, checks the setup with fake data, then runs matching.
@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
title Registry Trust data check

set "SETUP_ONLY="
if /i "%~1"=="--setup-only" set "SETUP_ONLY=1"

if not exist "requirements.lock" goto missing_lock
where python >nul 2>nul
if errorlevel 1 goto bad_python
python -c "import platform,struct,sys; ok=platform.python_implementation()=='CPython' and sys.version_info[:2] in ((3,13),(3,14)) and struct.calcsize('P')==8 and platform.machine().lower() in ('amd64','x86_64'); raise SystemExit(0 if ok else 1)" >nul 2>nul
if errorlevel 1 goto bad_python

set "PY_TAG="
for /f "delims=" %%V in ('python -c "import sys; print('py'+str(sys.version_info.major)+str(sys.version_info.minor))"') do set "PY_TAG=%%V"
if not defined PY_TAG goto bad_python
set "SETUP_KEY="
for /f "delims=" %%K in ('python -c "import hashlib,sys; print('.'.join(map(str,sys.version_info[:3]))+'-'+hashlib.sha256(open('requirements.lock','rb').read()).hexdigest())"') do set "SETUP_KEY=%%K"
if not defined SETUP_KEY goto setup_failed

set "INSTALLED_KEY="
if not exist ".venv\.setup-key" goto install
for /f "usebackq delims=" %%K in (".venv\.setup-key") do set "INSTALLED_KEY=%%K"
if not exist ".venv\Scripts\python.exe" goto install
if "%INSTALLED_KEY%"=="%SETUP_KEY%" goto environment_ready

:install
echo Preparing the Python environment...
if exist ".venv" rmdir /s /q ".venv"
python -m venv ".venv"
if errorlevel 1 goto setup_failed
if exist "wheels\%PY_TAG%\*.whl" goto install_offline
echo Downloading the required packages from PyPI. No RT data is read or sent.
".venv\Scripts\python.exe" -m pip install --require-hashes --only-binary=:all: -r "requirements.lock"
if errorlevel 1 goto setup_failed
goto install_done

:install_offline
echo Installing the required packages from the offline folder. No internet is needed.
".venv\Scripts\python.exe" -m pip install --no-index --find-links "wheels\%PY_TAG%" --require-hashes --only-binary=:all: -r "requirements.lock"
if errorlevel 1 goto setup_failed

:install_done
> ".venv\.setup-key" echo %SETUP_KEY%

:environment_ready
if defined SETUP_ONLY exit /b 0

echo Checking the setup with fake data...
".venv\Scripts\python.exe" -m recovery.selftest
if errorlevel 1 goto run_failed

set "STAGE=%~1"
set "JUDGMENTS=%~2"
set "COMPANIES=%~3"
set "OBSERVATION=%~4"
set "CH_DATE=%~5"
set "OUTPUT_BASE=%~6"
set "INTERACTIVE="

if defined STAGE goto have_stage
set "INTERACTIVE=1"
set "STAGE=diagnostic"

:have_stage
set "STAGE=%STAGE:"=%"
if "%STAGE%"=="1" set "STAGE=diagnostic"
if /i "%STAGE%"=="diagnostic" goto stage_ok
goto modelling_disabled

:stage_ok
if defined JUDGMENTS goto have_judgments
set "INTERACTIVE=1"
set /p "JUDGMENTS=Drag the RT judgment CSV/XLSX here and press Enter: "

:have_judgments
if defined COMPANIES goto have_companies
set "INTERACTIVE=1"
set /p "COMPANIES=Drag the Companies House CSV/ZIP here and press Enter: "

:have_companies
set "JUDGMENTS=%JUDGMENTS:"=%"
set "COMPANIES=%COMPANIES:"=%"
if not exist "%JUDGMENTS%" goto missing_judgments
if not exist "%COMPANIES%" goto missing_companies

if not defined INTERACTIVE goto arguments_ready
if not defined OBSERVATION set /p "OBSERVATION=RT extract date YYYY-MM-DD (required): "
if not defined CH_DATE set /p "CH_DATE=Companies House file date YYYY-MM-DD (required): "

:arguments_ready
if not defined OBSERVATION goto missing_observation
if not defined CH_DATE goto missing_ch_date
if not defined OUTPUT_BASE set "OUTPUT_BASE=outputs"

echo.
if /i "%STAGE%"=="diagnostic" echo Checking the files and matching companies...
".venv\Scripts\python.exe" -m recovery.run analyze --stage "%STAGE%" --judgments "%JUDGMENTS%" --companies-house "%COMPANIES%" --observation-date "%OBSERVATION%" --companies-house-date "%CH_DATE%" --settings "settings.toml" --output-base "%OUTPUT_BASE%"
if errorlevel 1 goto run_failed

:run_succeeded
echo.
echo Done. Send the folder shown above to Edwin.
if "%RECOVERY_NO_PAUSE%"=="" pause
exit /b 0

:bad_python
echo.
echo STOP: 64-bit Python 3.13 or 3.14 must be installed and first on PATH.
goto stop

:missing_lock
echo.
echo STOP: requirements.lock is missing. Download the complete reviewed repository again.
goto stop

:setup_failed
echo.
echo STOP: Python setup failed. See the message above.
echo If PyPI is blocked on this machine, use the optional offline package folder.
echo No RT data was read or sent during setup.
goto stop

:modelling_disabled
echo.
echo STOP: Run 2 is not available until the data and study design have been checked.
goto stop

:missing_judgments
echo.
echo STOP: Cannot find the RT judgment file: "%JUDGMENTS%"
goto stop

:missing_companies
echo.
echo STOP: Cannot find the Companies House file: "%COMPANIES%"
goto stop

:missing_observation
echo.
echo STOP: The RT extract date is required in YYYY-MM-DD format.
goto stop

:missing_ch_date
echo.
echo STOP: The Companies House file date is required in YYYY-MM-DD format.
goto stop

:run_failed
echo.
echo STOP: The run failed and any incomplete output was removed.
echo Do not use or share output from this failed run.

:stop
if "%RECOVERY_NO_PAUSE%"=="" pause
exit /b 1
