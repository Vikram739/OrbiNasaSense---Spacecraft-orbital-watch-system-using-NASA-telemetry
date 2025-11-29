@echo off
REM OrbiNasaSense - Complete Setup and Run Script
REM This script will train a model and launch the Streamlit UI

echo.
echo ========================================================================
echo                   OrbiNasaSense - NASA Anomaly Detection
echo ========================================================================
echo.

REM Step 1: Train the model
echo [1/2] Training model on channel P-1...
echo This will take 2-5 minutes depending on your computer.
echo.

C:\Users\Mortal\AppData\Local\Python\pythoncore-3.14-64\python.exe train.py --channel P-1 --window-size 50 --epochs 30 --batch-size 32

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Training failed!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo Training complete! Model saved in models/ folder.
echo ========================================================================
echo.

REM Step 2: Launch Streamlit
echo [2/2] Launching Streamlit UI...
echo.
echo Your browser will open automatically.
echo Press Ctrl+C in this window to stop the server.
echo.

C:\Users\Mortal\AppData\Local\Python\pythoncore-3.14-64\python.exe -m streamlit run app.py

pause
