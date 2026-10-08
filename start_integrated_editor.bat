@echo off
setlocal
cd /d "%~dp0"
if not defined SAM_API_URL set "SAM_API_URL=http://127.0.0.1:8010/sam"
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" mask_editor_app.py %*
) else (
    python mask_editor_app.py %*
)
if errorlevel 1 pause
