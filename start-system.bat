@echo off
echo ========================================
echo   Football Analytics AI System
echo ========================================
echo.

REM Activate virtual environment
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo WARNING: Virtual environment not found
    echo Run setup.bat first
    pause
    exit /b 1
)

echo.
echo [1/2] Initializing database...
python -m app.database init
if errorlevel 1 (
    echo WARNING: Database initialization may have issues
)

echo.
echo [2/2] Starting servers...
echo.
echo Starting FastAPI backend on http://localhost:8000
echo Starting Streamlit dashboard on http://localhost:8501
echo.
echo Press Ctrl+C in each window to stop
echo.

REM Start API in new window
start "Football Analytics API" cmd /k "call venv\Scripts\activate.bat && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

REM Wait for API to start
timeout /t 3 /nobreak > nul

REM Start Streamlit in new window
start "Football Analytics Dashboard" cmd /k "call venv\Scripts\activate.bat && streamlit run dashboard/app.py --server.port 8501"

echo.
echo ========================================
echo   System Started!
echo ========================================
echo.
echo API Docs: http://localhost:8000/docs
echo Dashboard: http://localhost:8501
echo.
echo Close this window to keep the servers running,
echo or close the server windows to stop them.
echo.
pause
