# =====================================================================================
# Retrain every run whose reported checkpoint is NOT the one the prespecified early-stopping rule
# (patience 10) would select.
#
# WHY: the grid first executed with --patience 30 (the training driver's default at the time)
# while config.yaml declares early_stopping_patience: 10. Most runs select the SAME checkpoint
# under either patience and are left untouched - retraining them would only re-roll the
# stochastic realisation. Only the non-equivalent ones are retrained; ten runs were.
#
# THE WORK LIST COMES FROM THE GATE, NOT FROM A HARDCODED ARRAY. verify_grid_consistency.py
# recomputes the criterion from each run's stored val-loss trajectory and writes the offenders to
# JSON; this script retrains exactly those. So the script is idempotent and resumable: run it again
# after any interruption and it picks up precisely what is still wrong.
#
# TRANSIENT FAILURES: one run died at startup with 0xC0000409 (native crash in the DataLoader worker
# spawn, empty stderr) and succeeded on a plain retry. So each run gets ONE retry, loudly logged.
# A second consecutive failure aborts - retries must not mask a real fault.
#
# src/train.py trains the FULL epoch budget (so the final-epoch checkpoint is also kept, as for
# every run) and RESTRICTS the reported checkpoint to the patience-10 window. No --patience /
# --max-epochs flags are passed: both are inherited from config.yaml.
#
# NOTE: ASCII ONLY. Non-ASCII punctuation silently breaks the PowerShell parser.
#
#   powershell -ExecutionPolicy Bypass -File scripts\retrain_deviating_checkpoints.ps1
# =====================================================================================
param(
  [string] $Python = "C:\ProgramData\miniconda3\envs\nodules\python.exe",
  [string] $Root   = "E:\NODULES",
  [string] $Config = "config.yaml"
)
Set-Location $Root
$stamp  = Get-Date -Format "yyyyMMdd-HHmmss"
$logDir = Join-Path $Root "outputs\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$master = Join-Path $logDir "retrain_$stamp.log"
function Write-Log { param([string]$m)
  $l = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
  Write-Output $l; Add-Content -Path $master -Value $l -Encoding UTF8 }

function Invoke-Run {
  param($ds, $su, $arm, $rep, $fold, $arch, $logPath)
  $pyArgs = @("-u","-m","src.train","--config",$Config,"--dataset",$ds,
              "--arch",$arch,"--arm",$arm,"--sample-unit",$su,"--rep","$rep","--fold","$fold",
              "--seed","42")
  $p = Start-Process -FilePath $Python -ArgumentList $pyArgs -Wait -NoNewWindow -PassThru `
        -RedirectStandardOutput $logPath -RedirectStandardError ($logPath + ".err")
  return $p.ExitCode
}

# ---- derive the work list from the gate (a transient file, kept with the logs) ----
$offenders = Join-Path $logDir "gate_offenders.json"
# A list left by an earlier invocation must never be mistaken for this gate's answer.
if (Test-Path $offenders) { Remove-Item $offenders -Force }
Start-Process -FilePath $Python -ArgumentList @("scripts\verify_grid_consistency.py","--config",$Config,"--json",$offenders) -Wait -NoNewWindow | Out-Null
if (-not (Test-Path $offenders)) { Write-Log "FATAL: gate produced no offender list."; exit 3 }
$work = (Get-Content $offenders -Raw | ConvertFrom-Json).bad_checkpoint
if ($work.Count -eq 0) { Write-Log "Nothing to do: every reported checkpoint is already patience-10-equivalent."; exit 0 }

Write-Log "=== retrain START | $($work.Count) runs (from gate) | patience/max-epochs INHERITED from $Config ==="
$i = 0
foreach ($w in $work) {
  $i++
  # The gate inspects every cohort (lidc_binary and lidc_binary_ge3) and records each offender's
  # dataset; retraining must use that dataset, or a valid run of the other cohort is overwritten.
  if (-not $w.dataset) { Write-Log "FATAL: offender without a dataset field: $($w.run)"; exit 3 }
  $name = "$($w.dataset)_$($w.su)_$($w.arm)_rep$($w.rep)_fold$($w.fold)_$($w.arch)"
  $runLog = Join-Path $logDir ($name + "_retrain.log")
  Write-Log "[$i/$($work.Count)] START $name (selected ep $($w.selected_epoch) -> protocol ep $($w.protocol_epoch))"
  $t0 = Get-Date
  $code = Invoke-Run $w.dataset $w.su $w.arm $w.rep $w.fold $w.arch $runLog
  if ($code -ne 0) {
    Write-Log "  TRANSIENT? $name exit=$code - retrying ONCE"
    $code = Invoke-Run $w.dataset $w.su $w.arm $w.rep $w.fold $w.arch $runLog
  }
  $secs = [int]((Get-Date) - $t0).TotalSeconds
  if ($code -ne 0) {
    Write-Log "FAIL  $name exit=$code ${secs}s (failed twice - not transient)"
    if (Test-Path ($runLog + ".err")) { Get-Content ($runLog + ".err") -Tail 15 | ForEach-Object { Write-Log "    $_" } }
    Write-Log "ABORTING: refusing to continue with a partially corrected grid."
    exit 2
  }
  Write-Log "OK    $name ${secs}s"
}

Write-Log "=== retrain DONE - running the consistency gate ==="
$g = Start-Process -FilePath $Python -ArgumentList @("scripts\verify_grid_consistency.py","--config",$Config) -Wait -NoNewWindow -PassThru
if ($g.ExitCode -ne 0) {
  Write-Log "GATE FAILED (exit $($g.ExitCode)): do NOT report numbers yet."
  exit 1
}
Write-Log "GATE PASSED: every reported checkpoint is patience-10-equivalent. Recompute the results."
exit 0
