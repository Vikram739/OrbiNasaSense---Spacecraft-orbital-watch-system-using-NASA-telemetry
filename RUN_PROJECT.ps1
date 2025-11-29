# Quick Training Script - Just run this in PowerShell!

Write-Host "`n========================================================================" -ForegroundColor Cyan
Write-Host "              OrbiNasaSense - NASA Anomaly Detection" -ForegroundColor Cyan
Write-Host "========================================================================`n" -ForegroundColor Cyan

Write-Host "[1/2] Training model on channel P-1..." -ForegroundColor Yellow
Write-Host "This will take 2-5 minutes.`n" -ForegroundColor White

C:/Users/Mortal/AppData/Local/Python/pythoncore-3.14-64/python.exe train.py --channel P-1 --window-size 50 --epochs 30 --batch-size 32

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n========================================================================" -ForegroundColor Green
    Write-Host "Training complete! Model saved successfully." -ForegroundColor Green
    Write-Host "========================================================================`n" -ForegroundColor Green
    
    Write-Host "[2/2] Launching Streamlit UI...`n" -ForegroundColor Yellow
    Write-Host "Your browser will open automatically." -ForegroundColor White
    Write-Host "Press Ctrl+C to stop the server.`n" -ForegroundColor White
    
    C:/Users/Mortal/AppData/Local/Python/pythoncore-3.14-64/python.exe -m streamlit run app.py
} else {
    Write-Host "`nERROR: Training failed!" -ForegroundColor Red
    Write-Host "Please check the error messages above.`n" -ForegroundColor Red
}
