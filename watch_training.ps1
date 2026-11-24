param([switch]$Once)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$Host.UI.RawUI.WindowTitle = 'MoodJournal - Training progress'
$logPath = Join-Path $PSScriptRoot 'local_data\continuation.log'
$errorPath = Join-Path $PSScriptRoot 'local_data\continuation-error.log'
$progressPath = Join-Path $PSScriptRoot 'artifacts\distilbert_continued\progress.json'
$runnerId = [int](Get-Content -LiteralPath 'local_data\continuation.pid')
$runner = Get-Process -Id $runnerId -ErrorAction SilentlyContinue
$runnerStarted = if ($runner) { $runner.StartTime } else { $null }
Write-Host 'MOODJOURNAL - LOCAL TRAINING' -ForegroundColor Cyan
Write-Host 'This window only displays progress. Closing it does not stop training.'
Write-Host 'Training, evaluation and tests run locally without chat/model calls.'
Write-Host ''
if (Test-Path -LiteralPath $progressPath) {
    $snapshot = Get-Content -LiteralPath $progressPath -Raw | ConvertFrom-Json
    Write-Host "Phase: $($snapshot.phase) | Additional epoch: $($snapshot.additional_epoch) | Step: $($snapshot.global_step)"
    Write-Host "Best validation macro-F1: $($snapshot.best_validation_macro_f1)"
}
$seen = 0
$seenErrors = 0
$first = $true
do {
    $lines = @(Get-Content -LiteralPath $logPath -ErrorAction SilentlyContinue)
    $errors = @(Get-Content -LiteralPath $errorPath -ErrorAction SilentlyContinue)
    if ($first) {
        $seen = [Math]::Max(0, $lines.Count - 8)
        $seenErrors = $errors.Count
        $first = $false
    }
    if ($lines.Count -lt $seen) { $seen = 0 }
    for ($i = $seen; $i -lt $lines.Count; $i++) { Write-Host $lines[$i] }
    $seen = $lines.Count
    if ($errors.Count -lt $seenErrors) { $seenErrors = 0 }
    for ($i = $seenErrors; $i -lt $errors.Count; $i++) { Write-Host $errors[$i] -ForegroundColor Yellow }
    $seenErrors = $errors.Count
    $running = Get-Process -Id $runnerId -ErrorAction SilentlyContinue
    if (-not $running -or $running.StartTime -ne $runnerStarted) {
        $testPassed = ($errors | Where-Object { $_.Trim() -eq 'OK' }).Count -gt 0
        $metadataExists = Test-Path -LiteralPath 'artifacts\distilbert_continued\metadata.json'
        if ($metadataExists -and $testPassed) {
            Write-Host "`nCOMPLETE - training, evaluation and tests finished." -ForegroundColor Green
            $report = Get-Content 'artifacts\distilbert_continued_report.json' -Raw | ConvertFrom-Json
            Write-Host "Test accuracy: $([Math]::Round(100 * $report.test.accuracy, 2))%"
            Write-Host "Test macro-F1: $([Math]::Round($report.test.macro_f1, 4))"
            Write-Host 'Reports: artifacts\distilbert_continued_report.json and artifacts\model_comparison.json'
            Write-Host 'Return to the chat and say: review the completed training results.'
        } else {
            Write-Host "`nSTOPPED - review local_data\continuation-error.log." -ForegroundColor Yellow
            Write-Host 'Run .\continue_training.ps1 to resume from the last full checkpoint.'
            $errors | Select-Object -Last 12 | ForEach-Object { Write-Host $_ }
        }
        break
    }
    if ($Once) { break }
    Start-Sleep -Seconds 2
} while ($true)
