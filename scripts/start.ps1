$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
$projectPython = Join-Path (Get-Location) '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Create .venv and install backend dependencies first. See README.md.'
}
$env:PYTHONPATH = 'backend'
$env:DATA_DIR = 'data'
$env:STATS_DIR = 'backend/data'
& $projectPython -m uvicorn app.main:app --host 127.0.0.1 --port 8180
