@rem Sets up Python, checks the code on fake data, then runs the selected stage locally.
@rem Run 1 matches the full RT extract only. Run 2 adds the satisfaction models and
@rem remains available for testing but is not offered interactively until RT agrees it.
@rem Raw names and matches stay in rt_internal; only checked aggregate reports go
@rem to egress_candidate.
@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
title Registry Trust recovery analysis

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
echo Preparing the fixed Python environment...
if exist ".venv" rmdir /s /q ".venv"
python -m venv ".venv"
if errorlevel 1 goto setup_failed
if exist "wheels\%PY_TAG%\*.whl" goto install_offline
echo Downloading the fixed packages from PyPI. No RT data is read or sent during setup.
".venv\Scripts\python.exe" -m pip install --require-hashes --only-binary=:all: -r "requirements.lock"
if errorlevel 1 goto setup_failed
goto install_done

:install_offline
echo Installing the fixed packages from the local wheel folder. No internet is needed.
".venv\Scripts\python.exe" -m pip install --no-index --find-links "wheels\%PY_TAG%" --require-hashes --only-binary=:all: -r "requirements.lock"
if errorlevel 1 goto setup_failed

:install_done
> ".venv\.setup-key" echo %SETUP_KEY%

:environment_ready
if defined SETUP_ONLY exit /b 0

echo Running the self-test on fake data...
".venv\Scripts\python.exe" -m recovery.selftest
if errorlevel 1 goto run_failed

set "STAGE=%~1"
set "JUDGMENTS=%~2"
set "COMPANIES=%~3"
set "OBSERVATION=%~4"
set "OUTPUT_BASE=%~5"
set "INTERACTIVE="

if defined STAGE goto have_stage
set "INTERACTIVE=1"
set "STAGE=diagnostic"
echo Run 1 selected: full-data matching only.

:have_stage
set "STAGE=%STAGE:"=%"
if "%STAGE%"=="1" set "STAGE=diagnostic"
if "%STAGE%"=="2" set "STAGE=locked"
if /i "%STAGE%"=="diagnostic" goto stage_ok
if /i "%STAGE%"=="locked" goto stage_ok
goto bad_stage

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
if defined OBSERVATION goto arguments_ready
set /p "OBSERVATION=RT extract date YYYY-MM-DD (blank = today): "

:arguments_ready
set "OBSERVATION=%OBSERVATION:"=%"
set "OUTPUT_BASE=%OUTPUT_BASE:"=%"
if not defined OUTPUT_BASE set "OUTPUT_BASE=outputs"

echo.
if /i "%STAGE%"=="diagnostic" echo Running Run 1: full-data matching only. No satisfaction model will be trained.
if /i "%STAGE%"=="locked" echo Running Run 2: locked matching plus the agreed satisfaction models.
echo Raw and RT-internal files stay on this machine.
if defined OBSERVATION goto run_with_date
".venv\Scripts\python.exe" -m recovery.run analyze --stage "%STAGE%" --judgments "%JUDGMENTS%" --companies-house "%COMPANIES%" --settings "settings.toml" --output-base "%OUTPUT_BASE%"
if errorlevel 1 goto run_failed
goto run_succeeded

:run_with_date
".venv\Scripts\python.exe" -m recovery.run analyze --stage "%STAGE%" --judgments "%JUDGMENTS%" --companies-house "%COMPANIES%" --observation-date "%OBSERVATION%" --settings "settings.toml" --output-base "%OUTPUT_BASE%"
if errorlevel 1 goto run_failed

:run_succeeded
echo.
echo Done. Open the newest run folder beneath "%OUTPUT_BASE%".
if /i "%STAGE%"=="diagnostic" echo Complete the 1,000-pair file in rt_internal before changing the matcher.
echo RT must review egress_candidate before anything leaves this machine.
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
echo If PyPI is blocked on this machine, ask for the optional reviewed wheel folder.
echo No RT data was read or sent during setup.
goto stop

:bad_stage
echo.
echo STOP: Choose 1 for full-data matching or 2 for the agreed satisfaction/model run.
goto stop

:missing_judgments
echo.
echo STOP: Cannot find the RT judgment file: "%JUDGMENTS%"
goto stop

:missing_companies
echo.
echo STOP: Cannot find the Companies House file: "%COMPANIES%"
goto stop

:run_failed
echo.
echo STOP: The run did not complete. No output is cleared to leave RT.

:stop
if "%RECOVERY_NO_PAUSE%"=="" pause
exit /b 1
