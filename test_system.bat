@echo off
echo ====================================================================
echo SMART FARMING ADVISOR - SYSTEM TEST
echo ====================================================================
echo.

echo [1/4] Testing Python environment...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)
echo.

echo [2/4] Testing backend imports...
python -c "import config; print('Config OK')"
if %errorlevel% neq 0 (
    echo ERROR: Failed to import config
    pause
    exit /b 1
)
echo.

echo [3/4] Checking models directory...
if exist "models\crop_model.pkl" (
    echo   - Crop model: FOUND
) else (
    echo   - Crop model: MISSING
)

if exist "models\disease_model_resnet50.pth" (
    echo   - Disease model: FOUND
) else (
    echo   - Disease model: MISSING
)

if exist "models\faq_vector_store.index" (
    echo   - RAG vector store: FOUND
) else (
    echo   - RAG vector store: MISSING
)
echo.

echo [4/4] System check complete!
echo.
echo To start the backend server, run: python app.py
echo To start the frontend, run: cd frontend && npm run dev
echo.
pause
