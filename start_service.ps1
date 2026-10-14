$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Project Python environment not found at $python. Create .venv and install requirements first."
}

if ([string]::IsNullOrWhiteSpace($env:MLFLOW_TRACKING_URI)) {
    $env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
}

if ([string]::IsNullOrWhiteSpace($env:MODEL_URI)) {
    $env:MODEL_URI = "models:/weather-rain-classifier@champion"
}

Push-Location $projectRoot
try {
    & $python -m uvicorn src.service:app --host 127.0.0.1 --port 8000
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
