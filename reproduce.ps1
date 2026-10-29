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

.PARAMETER Part2
After the Part 1 stages, also rebuild the Part 2 submission P2-S13 (work\p2_c047.zip) with the chain in the Part 2
block at the end of this script. Off by default; it adds about 7 to 8 hours (estimates in that block).

.EXAMPLE
.\reproduce.ps1
.\reproduce.ps1 -Name sub_check -SkipTests
.\reproduce.ps1 -Part2
#>
[CmdletBinding()]
param(
    [string]$Name = 'sub_repro',
    [string]$Models = 'glm,gam,lgbm,xgb,cat',
    [switch]$SkipTests,
    [switch]$Part2
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

# ----------------------------------------------------------------------------------------------------------------
# Part 2 (emerging market): rebuilds P2-S13, work\p2_c047.zip. Runs only with -Part2, after the Part 1 stages above,
# because the pooled runs read the Part 1 target tables output\target_<line>.parquet written by stage 'build_target'.
# See README.md, "Reproducing the Part 2 submission". Stages run one after another, so only one data process is alive
# at a time; plan for about 4.5 GB of memory at peak. Durations are marked "measured" (taken from CHANGELOG C045, C047
# and the blend result files) or "estimate" (not timed). The whole block is roughly 7 to 8 hours.
# ----------------------------------------------------------------------------------------------------------------
if ($Part2) {
    $part2Start = Get-Date
    $p2Models = 'glm,lgbm,xgb,cat'

    Invoke-Stage 'p2 build_target emerging' @('scripts\build_target.py', '--market', 'emerging')   # minutes, not timed

    # Full pooled runs (every developed row added at the given weight). Per-line weights: motor 0.3, buildings 1.0,
    # contents 0.3 (C040). The weight-1.0 runs of motor and contents are blend candidates for the searches below.
    Invoke-Stage 'p2 oof motor w0.3' @('scripts\oof_predictions.py', 'motor', '--market', 'emerging', '--train-set', 'pooled', '--developed-weight', '0.3', '--models', $p2Models)   # estimate 1.5 to 2 h
    Invoke-Stage 'p2 oof motor w1.0' @('scripts\oof_predictions.py', 'motor', '--market', 'emerging', '--train-set', 'pooled', '--developed-weight', '1.0', '--models', $p2Models)   # estimate 1.5 to 2 h
    Invoke-Stage 'p2 oof buildings w1.0' @('scripts\oof_predictions.py', 'buildings', '--market', 'emerging', '--train-set', 'pooled', '--developed-weight', '1.0', '--models', $p2Models)   # estimate 1 h
    Invoke-Stage 'p2 oof contents w0.3' @('scripts\oof_predictions.py', 'contents', '--market', 'emerging', '--train-set', 'pooled', '--developed-weight', '0.3', '--models', $p2Models)   # about 1 h (57 to 58 min measured on the comparable C045 runs)
    Invoke-Stage 'p2 oof contents w1.0' @('scripts\oof_predictions.py', 'contents', '--market', 'emerging', '--train-set', 'pooled', '--developed-weight', '1.0', '--models', $p2Models)   # about 1 h (same size as the w0.3 run)

    # Motor run on the new-business rows of both markets only (folder pooled_w0.3_nbrows, C047).
    Invoke-Stage 'p2 oof motor w0.3 nbrows' @('scripts\oof_predictions.py', 'motor', '--market', 'emerging', '--train-set', 'pooled', '--developed-weight', '0.3', '--models', $p2Models, '--rows', 'nb')   # 41 min measured (C047)

    # Blend searches (C044, C047), then the write step that builds work\p2_c047.csv and work\p2_c047.zip.
    Invoke-Stage 'p2 blend_runs contents two-run' @('scripts\blend_runs.py', 'contents', '--runs', 'pooled_w0.3,pooled_w1', '--gate', 'testmix', '--out', 'xrun_c044_contents_2run')   # about 4 min (257 s measured)
    Invoke-Stage 'p2 blend_runs motor new-business segment' @('scripts\blend_runs.py', 'motor', '--runs', 'pooled_w0.3,pooled_w1,pooled_w0.3_nbrows', '--segment', 'new_business', '--gate', 'testmix', '--baseline', 'P2-S12', '--step', '0.1', '--out', 'xrun_c047_motor')   # about 2 min (128 s measured)
    Invoke-Stage 'p2 blend_runs write p2_c047' @('scripts\blend_runs.py', 'motor,buildings,contents', '--write-from', 'motor=xrun_c047_motor,buildings=baseline,contents=xrun_c044_contents_2run', '--trend', 'mixed-full', '--name', 'p2_c047')   # a few minutes, not timed

    $p2Zip = Join-Path $root 'work\p2_c047.zip'
    if (-not (Test-Path -LiteralPath $p2Zip)) {
        throw "Expected output $p2Zip was not produced."
    }
    Write-Log ("Part 2 done in {0:N1} min." -f ((Get-Date) - $part2Start).TotalMinutes)
    Write-Host "Part 2 submission zip: $p2Zip"
}
