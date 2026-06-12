@echo off
REM ============================================================
REM  VaultRAG v2.0 - Windows Launcher
REM ============================================================
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [INFO] No .venv found. Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
)

set PYTHONPATH=%cd%

echo.
echo  ============================================
echo   VaultRAG v2.0 - Secure Offline RAG Server
echo   http://127.0.0.1:8000
echo  ============================================
echo.

python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
pause
