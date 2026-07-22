<#
  run_probe740.ps1 — 740-cohort convergence probe (the gate before the grid, D34 safeguard #3).

  Run from YOUR OWN terminal AFTER preprocessing the full cohort and generating the 740 splits:
      cd E:\NODULES
      powershell -ExecutionPolicy Bypass -File scripts\run_probe740.ps1

  WHY. The canonical 30-epoch ceiling was set on the 250-patient subset, where the RANDOM arm
  already ran to best_epoch 22-27 (6/10 never triggered patience within 30). At 740 the training
  set is ~3x larger, so 30 epochs may truncate the random arm mid-improvement. This probes the
  BINDING case — the random arm of both architectures — with an EXTENDED ceiling (60) and
  patience 10, and reads where the best-val-loss checkpoint actually lands.

  VERDICT:
    * both best_epoch < 30  -> the 30-epoch config scales unchanged; launch the grid as-is.
    * any best_epoch >= 30  -> AMEND the ceiling BEFORE the grid: set train.max_epochs in
      config.yaml to the new value and pass -MaxEpochs <new> to run_grid.ps1, so the WHOLE grid
      runs under one ceiling (never half at 30, half at 60).

  The probe trains at --max-epochs 60, so its artifacts carry max_epochs=60 and MUST NOT be folded
  into the grid (which runs at 30, or the amended ceiling). This script DELETES the probe artifacts
  after reading the verdict, so the grid retrains those two runs cleanly and
  verify_grid_consistency stays green. Nothing else is touched.
#>
param(
  [string] $Config = "config.yaml",
  [string] $Archs  = "densenet121,efficientnet_b0",
  [string] $Arm    = "random",              # the binding arm (highest best_epoch)
  [int]    $Ceiling = 60,
  [int]    $Patience = 10,
  [int]    $Seed   = 42,
  [string] $Python = "C:\ProgramData\miniconda3\envs\nodules\python.exe",
  [string] $Root   = "E:\NODULES"
)

Set-Location $Root
$stamp  = Get-Date -Format "yyyyMMdd-HHmmss"
$logDir = Join-Path $Root "outputs\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$master = Join-Path $logDir "probe740_$stamp.log"
function Write-Log { param([string]$m)
  $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
  Write-Output $line; Add-Content -Path $master -Value $line -Encoding UTF8 }

$ArchList = @($Archs -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
Write-Log "=== probe740 START | arm=$Arm ceiling=$Ceiling patience=$Patience ==="

# coverage must be at grid scale, else the probe would run on the 250 subset
$cov = Start-Process -FilePath $Python -ArgumentList @("scripts\verify_coverage.py","--config",$Config,"--phase","grid") -Wait -NoNewWindow -PassThru
if ($cov.ExitCode -ne 0) { Write-Log "ABORT: coverage --phase grid FAILED. Preprocess the full cohort first."; exit 1 }

$O = Join-Path $Root "outputs"
$probeRuns = @()
$verdict = @()
foreach ($arch in $ArchList) {
  $tag = "lidc_binary_slice_" + $Arm + "_rep0_fold0"
  $run = $tag + "_" + $arch + "_none_seed" + $Seed
  $sp  = Join-Path $O ("splits\" + $tag + "_test.csv")
  if (-not (Test-Path $sp)) { Write-Log "ABORT: split $sp missing. Generate the 740 splits first (src.splits, no --limit-patients)."; exit 1 }

  $runLog = Join-Path $logDir ($run + "_probe.log")
  Write-Log "PROBE $run (max-epochs $Ceiling, patience $Patience) ..."
  $t0 = Get-Date
  $pyArgs = @("-u","-m","src.train","--config",$Config,"--dataset","lidc_binary","--arch",$arch,
              "--arm",$Arm,"--sample-unit","slice","--rep","0","--fold","0","--seed","$Seed",
              "--max-epochs","$Ceiling","--patience","$Patience")
  $r = Start-Process -FilePath $Python -ArgumentList $pyArgs -Wait -NoNewWindow -PassThru -RedirectStandardOutput $runLog -RedirectStandardError ($runLog + ".err")
  $secs = [int]((Get-Date) - $t0).TotalSeconds
  if ($r.ExitCode -ne 0) {
    Write-Log "FAIL $run exit=$($r.ExitCode) ${secs}s"
    if (Test-Path ($runLog + ".err")) { Get-Content ($runLog + ".err") -Tail 15 | ForEach-Object { Write-Log "    $_" } }
    exit 2
  }
  $hp = Join-Path $O ("history\" + $run + ".json")
  $h  = Get-Content $hp -Raw | ConvertFrom-Json
  $be = [int]$h.best_epoch; $es = $h.early_stop_would_fire_at
  Write-Log ("  {0}: best_epoch {1} · early-stop would fire at {2} · epochs_run {3} · {4}s" -f $arch, $be, $es, $h.epochs_run, $secs)
  $verdict += [pscustomobject]@{ arch = $arch; best_epoch = $be }
  $probeRuns += $run
}

# ---- verdict ----
$maxBe = ($verdict | Measure-Object -Property best_epoch -Maximum).Maximum
Write-Log "--- VERDICT (arm=$Arm) ---"
$verdict | ForEach-Object { Write-Log ("  {0}: best_epoch {1}" -f $_.arch, $_.best_epoch) }
if ($maxBe -lt 30) {
  Write-Log "OK: every probed best_epoch < 30 -> the 30-epoch canonical config SCALES. Launch the grid as-is (-MaxEpochs 30)."
} else {
  Write-Log "AMEND: a best_epoch >= 30 (max $maxBe) -> the 30-epoch ceiling TRUNCATES the random arm at 740."
  Write-Log "       BEFORE the grid: set train.max_epochs in config.yaml to a higher value (e.g. $Ceiling) and"
  Write-Log "       pass -MaxEpochs <new> to run_grid.ps1 so the WHOLE grid runs under one ceiling."
}

# ---- clean up probe artifacts (they are at ceiling $Ceiling; the grid must retrain them) ----
Write-Log "cleaning probe artifacts (so the grid retrains them under the final ceiling) ..."
foreach ($run in $probeRuns) {
  foreach ($rn in @($run, ($run + "_final"))) {
    foreach ($sub in @("probs\$rn.npz","history\$rn.json","models\$rn.pt","metrics\$rn.json")) {
      $p = Join-Path $O $sub
      if (Test-Path $p) { Remove-Item $p -Force }
    }
  }
  Write-Log "  removed probe artifacts for $run"
}
Write-Log "=== probe740 DONE ==="
exit 0
