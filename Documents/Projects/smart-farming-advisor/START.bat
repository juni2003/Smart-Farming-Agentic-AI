@echo off
echo.
echo ====================================================================
echo   SMART FARMING ADVISOR - QUICK START
echo ====================================================================
echo.
echo This will start both backend and frontend servers
echo.
echo Backend (Flask): http://localhost:5000
echo Frontend (Next.js): http://localhost:3000
echo.
echo Press Ctrl+C in each window to stop the servers
echo.
pause

echo.
echo Starting Backend Server...
start "Smart Farming - Backend" cmd /k "cd /d %~dp0 && python app.py"

timeout /t 3 /nobreak >nul

echo Starting Frontend Server...
start "Smart Farming - Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ====================================================================
echo   SERVERS STARTED!
echo ====================================================================
echo.
echo Backend:  http://localhost:5000
echo Frontend: http://localhost:3000
echo.
echo Open http://localhost:3000 in your browser
echo.
echo Two terminal windows have been opened.
echo Close them to stop the servers.
echo.
pause
