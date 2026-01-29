@echo off
echo ========================================
echo   Football Analytics AI System Setup
echo ========================================
echo.

REM Check Python
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [3/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo WARNING: Some dependencies may have failed to install
)

echo.
echo [4/4] Copying environment file...
if not exist .env (
    copy .env.example .env
    echo Created .env file - please configure your settings
) else (
    echo .env file already exists
)

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Edit .env file with your configuration
echo   2. Run 'start-system.bat' to start the application
echo.
pause
