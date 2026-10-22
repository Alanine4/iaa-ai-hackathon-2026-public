<#
.SYNOPSIS
Rebuilds a Part 1 submission zip from the raw data, one stage after another.

.DESCRIPTION
Stages, in order (see README.md, "Reproducing a submission"):
  1. scripts\build_target.py             developed target per training row (output\target_<line>.parquet)
  2. scripts\oof_predictions.py <line>   out-of-fold folds, blend weights, refit and 2025 predictions, for motor, buildings, contents
  3. scripts\blend.py <models> <name> --weighted   per-line weighted blend, writes work\<name>.csv and work\<name>.zip
  4. pytest                              test suite (skipped with -SkipTests)

Every stage runs with the venv interpreter in isolated, unbuffered mode (-I -u). The script stops at the first
stage that returns a non-zero exit code. A full run takes about two hours.

Note: blend_weights_<line>.json is written by oof_predictions.py with the searched weights. The submitted
S07 file used equal weights for motor because the searched motor weights failed the "better than equal on both
folds" rule (C021); that rule is applied by hand, not by this script. See README.md.

.PARAMETER Name
Output name. The blend is written to work\<Name>.csv and work\<Name>.zip.

.PARAMETER Models
Comma-separated model list passed to oof_predictions.py (--models) and blend.py.

.PARAMETER SkipTests
Skip the pytest stage.

.EXAMPLE
.\reproduce.ps1
.\reproduce.ps1 -Name sub_check -SkipTests
#>
[CmdletBinding()]
param(
    [string]$Name = 'sub_repro',
    [string]$Models = 'glm,gam,lgbm,xgb,cat',
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Set-Location $root

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Venv interpreter not found at $python. Run: python -m venv .venv ; .venv\Scripts\python.exe -m pip install -r requirements.txt"
}

$lines = @('motor', 'buildings', 'contents')
$runStart = Get-Date

function Write-Log([string]$Message) {
    Write-Host ("[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message)
}

function Invoke-Stage([string]$Label, [string[]]$PythonArgs) {
    $start = Get-Date
    Write-Log "START $Label"
    & $python -I -u @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Stage '$Label' failed with exit code $LASTEXITCODE after $([math]::Round(((Get-Date) - $start).TotalMinutes, 1)) min."
    }
    Write-Log ("END   {0} ({1:N1} min)" -f $Label, ((Get-Date) - $start).TotalMinutes)
}

Write-Log "Reproduction run: Name=$Name Models=$Models SkipTests=$($SkipTests.IsPresent)"

Invoke-Stage 'build_target' @('scripts\build_target.py')

foreach ($line in $lines) {
    Invoke-Stage "oof_predictions $line" @('scripts\oof_predictions.py', $line, '--models', $Models)
}

Invoke-Stage "blend $Name" @('scripts\blend.py', $Models, $Name, '--weighted')

if (-not $SkipTests) {
    Invoke-Stage 'pytest' @('-m', 'pytest', '-q')
}

$zip = Join-Path $root "work\$Name.zip"
if (-not (Test-Path -LiteralPath $zip)) {
    throw "Expected output $zip was not produced."
}
Write-Log ("Done in {0:N1} min." -f ((Get-Date) - $runStart).TotalMinutes)
Write-Host "Submission zip: $zip"
