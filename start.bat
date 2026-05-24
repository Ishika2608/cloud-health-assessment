@echo off
echo ================================================
echo   CloudPulse - Cloud Health Monitor Setup
echo ================================================
echo.

:: Check for Python using 'py' launcher (standard on Windows)
py --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python from https://python.org
    echo Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

echo [1/3] Installing Python dependencies...
py -m pip install flask flask-cors scikit-learn numpy pandas
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install dependencies.
    echo Try running this file as Administrator.
    echo.
    pause
    exit /b 1
)

echo.
echo [2/3] Dependencies installed successfully!
echo.
echo [3/3] Starting Flask backend on http://localhost:5000
echo       Open frontend\index.html in your browser AFTER the server starts.
echo.
echo Press Ctrl+C to stop the server.
echo ================================================
echo.

cd backend
py app.py

echo.
echo Server stopped. Press any key to close.
pause
