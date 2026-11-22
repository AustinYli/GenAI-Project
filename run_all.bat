@echo off
setlocal
rem  pass --nopause to run without waiting for a key press
set NOPAUSE=%~1
cd /d "%~dp0"
set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" (
  echo Run setup.bat first.
  if /i not "%NOPAUSE%"=="--nopause" pause
  exit /b 1
)
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
if not defined HF_HOME if exist "%~dp0cache\hf\hub" set "HF_HOME=%~dp0cache\hf"
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set FAILED=0
if not exist outputs\logs mkdir outputs\logs
echo [%date% %time%] run_all started >> outputs\logs\status.txt
echo Each step writes its log to outputs\logs\. Finished steps are skipped if you run this again.
echo.

rem  name                 script and arguments
call :step 01_replicate        replicate\main.py
call :step 02_idea1_word       idea1_residual\run.py --scorer word
call :step 03_idea2_prepare    idea2_arrows\prepare_data.py
call :step 04_idea2_embed      idea2_arrows\embed_data.py
call :step 05_idea2_train      idea2_arrows\train.py
call :step 06_idea2_eval       idea2_arrows\eval_multihop.py
call :step 07_idea2_eval_mask  idea2_arrows\eval_multihop.py --mask_outlets
call :step 08_idea1_llm        idea1_residual\run.py --scorer llm_inplace
call :step 09_idea1_probe      idea1_residual\run.py --scorer llm_probe
call :step 10_idea1_mlm        idea1_residual\run.py --scorer mlm
call :step 11_replicate_rerank replicate\main.py --rerank

if "%FAILED%"=="1" (
  echo [%date% %time%] FINISHED WITH ERRORS >> outputs\logs\status.txt
  echo Some steps failed. See outputs\logs\status.txt for details.
  if /i not "%NOPAUSE%"=="--nopause" pause
  exit /b 1
)
echo [%date% %time%] ALL DONE >> outputs\logs\status.txt
echo.
echo All steps finished. Results are in outputs\
if /i not "%NOPAUSE%"=="--nopause" pause
exit /b 0

:step
set NAME=%~1
if exist outputs\logs\%NAME%.done (
  echo [skip] %NAME% already done
  exit /b 0
)
echo [%time%] %NAME% ...
echo [%date% %time%] START %NAME% >> outputs\logs\status.txt
"%PY%" %2 %3 %4 %5 %6 > outputs\logs\%NAME%.log 2>&1
if errorlevel 1 (
  set FAILED=1
  echo [%date% %time%] FAILED %NAME% >> outputs\logs\status.txt
  echo     failed - see outputs\logs\%NAME%.log
) else (
  echo [%date% %time%] DONE %NAME% >> outputs\logs\status.txt
  echo done > outputs\logs\%NAME%.done
)
exit /b 0
