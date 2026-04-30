@echo off
REM Cellscope install script for Windows.
REM Creates two conda envs (`cellpose` + `cellpose4`) and verifies imports.
REM Run from a "Anaconda Prompt" (or any shell where `conda` is on PATH).
REM
REM Usage:
REM   install.bat
REM
REM Requirements: Miniconda or Anaconda installed.
REM   https://docs.conda.io/en/latest/miniconda.html

setlocal

echo === Cellscope install (Windows) ===
echo.

REM Step 1: verify conda is available
where conda >nul 2>nul
if errorlevel 1 (
  echo ERROR: `conda` not found on PATH.
  echo Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html
  echo Then run this script from an "Anaconda Prompt".
  exit /b 1
)

echo [1/4] conda found.
echo.

REM Step 2: create the main `cellpose` env (with GUI + CP3 models)
echo [2/4] Creating `cellpose` env (CP3 + GUI)...
echo This takes 5-10 minutes on the first run.
call conda env create -f environment.yml
if errorlevel 1 (
  echo.
  echo NOTE: env may already exist. To rebuild it, run:
  echo   conda env remove -n cellpose ^&^& conda env create -f environment.yml
  echo Continuing anyway...
)
echo.

REM Step 3: create the `cellpose4` env (cpsam ViT)
echo [3/4] Creating `cellpose4` env (cpsam ViT)...
call conda env create -f environment-cellpose4.yml
if errorlevel 1 (
  echo.
  echo NOTE: env may already exist. To rebuild it, run:
  echo   conda env remove -n cellpose4 ^&^& conda env create -f environment-cellpose4.yml
  echo Continuing anyway...
)
echo.

REM Step 4: verify both envs load the right cellpose version
echo [4/4] Verifying envs...
call conda run -n cellpose  python -c "import cellpose; print('cellpose env: cellpose', cellpose.version)"
if errorlevel 1 (
  echo ERROR: `cellpose` env failed to import cellpose.
  exit /b 1
)
call conda run -n cellpose4 python -c "import cellpose; print('cellpose4 env: cellpose', cellpose.version)"
if errorlevel 1 (
  echo ERROR: `cellpose4` env failed to import cellpose.
  exit /b 1
)
echo.

echo === Install complete ===
echo.
echo Next step: download the cpsam_dic model (1.1 GB) by running:
echo   conda run -n cellpose python download_models.py
echo.
echo Then launch the GUI:
echo   conda activate cellpose
echo   python main_focused.py
echo.

endlocal
