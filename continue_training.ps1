$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath 'artifacts\distilbert_continued\training_checkpoint.pt') {
    & $pythonPath -u continue_transformer.py --resume
} else {
    & $pythonPath -u continue_transformer.py
}
if ($LASTEXITCODE -ne 0) { throw 'Training stopped. Re-run this script to restore its last full checkpoint.' }
if (-not (Test-Path -LiteralPath 'artifacts\distilbert_continued\metadata.json')) {
    throw 'Training has not completed evaluation.'
}
& $pythonPath -u evaluate_models.py
if ($LASTEXITCODE -ne 0) { throw 'Model comparison failed.' }
& $pythonPath -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Verification failed.' }
