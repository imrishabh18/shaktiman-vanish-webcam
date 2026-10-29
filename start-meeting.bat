@echo off
call "%~dp0start.bat" --virtual-camera --virtual-backend obs --width 1280 --height 720 --fps 30 %*
exit /b %errorlevel%
