@echo off
REM Quick Launch - Just open the Streamlit UI (model must be trained first)

echo.
echo Starting OrbiNasaSense Dashboard...
echo.
echo Your browser will open at http://localhost:8501
echo.
echo To stop the server, close this window.
echo.

start "" "C:\Users\Mortal\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m streamlit run app.py

pause
