@rem Checks RT's answers in a completed 1,000-pair file and writes aggregate results only.
@rem A Run 1 pass confirms sampled automatic-match quality, not every match or a model.
@rem The named pair file stays inside RT; setup may contact PyPI before it is read.
@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
title Registry Trust match-review check

call RUN.bat --setup-only
if errorlevel 1 goto setup_failed

set "REVIEW_FILE=%~1"
set "OUTPUT_DIR=%~2"
if defined REVIEW_FILE goto have_review
set /p "REVIEW_FILE=Drag the completed 1,000-pair review CSV/XLSX here and press Enter: "

:have_review
set "REVIEW_FILE=%REVIEW_FILE:"=%"
if not exist "%REVIEW_FILE%" goto missing_review
if defined OUTPUT_DIR goto have_output
set /p "OUTPUT_DIR=Aggregate review-output folder (outputs\review): "
if not defined OUTPUT_DIR set "OUTPUT_DIR=outputs\review"

:have_output
set "OUTPUT_DIR=%OUTPUT_DIR:"=%"
".venv\Scripts\python.exe" -m recovery.run review --review-file "%REVIEW_FILE%" --settings "settings.toml" --output-dir "%OUTPUT_DIR%"
if errorlevel 1 goto review_failed

echo.
echo Match-review gate passed. Aggregate results are in "%OUTPUT_DIR%".
echo Open MATCH_REVIEW_STATUS.txt there for the matching result.
echo The completed pair file remains RT-internal and must not leave.
if "%RECOVERY_NO_PAUSE%"=="" pause
exit /b 0

:setup_failed
echo.
echo STOP: The fixed Python environment could not be prepared.
goto stop

:missing_review
echo.
echo STOP: Cannot find the completed review file: "%REVIEW_FILE%"
goto stop

:review_failed
echo.
echo STOP: Match-review validation failed or the precision gate did not pass.
echo Inspect the aggregate result; the completed pair file must remain inside RT.

:stop
if "%RECOVERY_NO_PAUSE%"=="" pause
exit /b 1
