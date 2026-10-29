@echo off
setlocal
cd /d "%~dp0"

set "MPLCONFIGDIR=%CD%\.cache\matplotlib"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "VENV_PY=.venv\Scripts\python.exe"

if exist "%VENV_PY%" goto :check_version

where py >nul 2>nul
if errorlevel 1 goto :use_python
py -3 -c "import sys; raise SystemExit(sys.version_info < (3, 10))"
if errorlevel 1 goto :python_error
echo [Shaktimaan] Creating a local Python environment...
py -3 -m venv .venv
if errorlevel 1 goto :venv_error
goto :check_version

:use_python
where python >nul 2>nul
if errorlevel 1 goto :python_error
python -c "import sys; raise SystemExit(sys.version_info < (3, 10))"
if errorlevel 1 goto :python_error
echo [Shaktimaan] Creating a local Python environment...
python -m venv .venv
if errorlevel 1 goto :venv_error

:check_version
"%VENV_PY%" -c "import sys; raise SystemExit(sys.version_info < (3, 10))"
if errorlevel 1 goto :python_error
fc /b requirements.txt ".venv\.vanish-requirements" >nul 2>nul
if errorlevel 1 goto :install
"%VENV_PY%" -c "import cv2, mediapipe, numpy, pyvirtualcam, sounddevice, soundfile" >nul 2>nul
if errorlevel 1 goto :install
goto :run

:install
echo [Shaktimaan] Installing dependencies. Internet is required for this setup.
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :install_error
copy /y requirements.txt ".venv\.vanish-requirements" >nul
if errorlevel 1 goto :venv_error

:run
echo [Shaktimaan] Starting webcam. Step fully out of view and wait for BACKGROUND READY.
"%VENV_PY%" -m shaktiman %*
if errorlevel 1 goto :app_error
exit /b 0

:python_error
echo Install Python 3.10 or newer from python.org and enable its launcher/PATH option.
echo If this folder already has an older .venv, rename it and run this launcher again.
goto :error

:venv_error
echo Could not create or update .venv. Check that this folder is writable and Python includes venv.
goto :error

:install_error
echo Dependency setup failed. Check the error above and your internet connection, then rerun this launcher.
goto :error

:app_error
echo The webcam app stopped with an error. Read the message above; see README.md for help.

:error
echo.
pause
exit /b 1
