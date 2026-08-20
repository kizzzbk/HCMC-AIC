$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent

# Set environment variables
$env:PYTHONPATH = "src"
$env:PYTHONIOENCODING = "utf-8"

# Start the uvicorn server
Write-Host "Starting Vector Search & Temporal Navigation Server..." -ForegroundColor Green
Write-Host "Access Dashboard at: http://localhost:8000" -ForegroundColor Cyan
python -m uvicorn shotlab.server:app --host 0.0.0.0 --port 8000
