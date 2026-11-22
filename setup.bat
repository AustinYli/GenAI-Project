@echo off
setlocal
rem  pass --nopause to run without waiting for a key press
set NOPAUSE=%~1
cd /d "%~dp0"
if not exist outputs\logs mkdir outputs\logs
set LOG=outputs\logs\00_setup.log
echo Setting up the Python environment. Details go to %LOG%
echo [%date% %time%] setup started > %LOG%

rem --- find a Python 3 (Anaconda's, else the one on PATH)
set BASEPY=%USERPROFILE%\anaconda3\python.exe
if not exist "%BASEPY%" set BASEPY=python

rem --- project-local virtual environment that can reuse Anaconda's packages
if not exist .venv\Scripts\python.exe (
  echo Creating .venv ...
  "%BASEPY%" -m venv --system-site-packages .venv >> %LOG% 2>&1
)
set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" (
  echo Could not create the environment. See %LOG%
  if /i not "%NOPAUSE%"=="--nopause" pause
  exit /b 1
)
"%PY%" -m pip install --upgrade pip >> %LOG% 2>&1 || goto failed

rem --- PyTorch with CUDA, only if a CUDA build isn't already there
"%PY%" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
  echo Installing PyTorch with CUDA, about 2.5 GB, this takes a while ...
  "%PY%" -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128 >> %LOG% 2>&1 || goto failed
)

echo Installing the other packages ...
"%PY%" -m pip install --no-cache-dir -r requirements.txt >> %LOG% 2>&1 || goto failed
"%PY%" -m spacy download en_core_web_sm >> %LOG% 2>&1 || goto failed

echo Checking ...
"%PY%" -c "import sys,torch; available=torch.cuda.is_available(); print('torch',torch.__version__,'| CUDA available:',available,'|',torch.cuda.get_device_name(0) if available else 'no GPU'); sys.exit(0 if available else 1)" >> %LOG% 2>&1 || goto failed
"%PY%" -c "import transformers,datasets,spacy;spacy.load('en_core_web_sm');print('transformers',transformers.__version__,'datasets',datasets.__version__,'spacy OK')" >> %LOG% 2>&1 || goto failed
"%PY%" tests\test_selection.py >> %LOG% 2>&1 || goto failed
echo [%date% %time%] setup finished >> %LOG%
findstr /C:"CUDA available" /C:"spacy OK" /C:"ok test" /C:"Error" %LOG%
echo.
echo Setup finished. If it says "CUDA available: True" above, double-click run_all.bat
if /i not "%NOPAUSE%"=="--nopause" pause
exit /b 0

:failed
echo [%date% %time%] setup failed >> %LOG%
echo Setup failed. See %LOG% for details.
if /i not "%NOPAUSE%"=="--nopause" pause
exit /b 1
