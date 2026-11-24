param([switch]$Demo)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    throw 'Run the setup commands in README.md first.'
}
if (-not (Test-Path -LiteralPath 'artifacts\baseline.joblib')) {
    & .\.venv\Scripts\python.exe train_baseline.py
    if ($LASTEXITCODE -ne 0) { throw 'Model training failed.' }
}
if ($Demo) {
    & .\.venv\Scripts\python.exe prepare_demo.py
    if ($LASTEXITCODE -ne 0) { throw 'Demo preparation failed.' }
}
$previousDemo = $env:MOODJOURNAL_DEMO
try {
    $env:MOODJOURNAL_DEMO = if ($Demo) { '1' } else { '0' }
    $port = if ($Demo) { '8502' } else { '8501' }
    & .\.venv\Scripts\python.exe -m streamlit run app.py --server.port $port
} finally {
    $env:MOODJOURNAL_DEMO = $previousDemo
}
