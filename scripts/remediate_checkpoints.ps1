<#
  remediate_checkpoints.ps1 - repair the 2 runs whose CANONICAL checkpoint violated the
  pre-registered early-stopping protocol (DECISIONS D22).

  Run from YOUR OWN terminal (GPU job - a Claude-launched process dies with the client):

      cd E:\NODULES
      powershell -ExecutionPolicy Bypass -File scripts\remediate_checkpoints.ps1

  WHY
    run_grid.ps1 passes -Patience 30 on purpose: the dual-checkpoint design (D22 ii) needs the
    loop to reach epoch 30 so the `_final` max-memorization PROBE exists. Selecting best_val_loss
    from that 30-epoch pass equals what patience-10 early stopping would have chosen - but ONLY
    when the argmin val_loss occurs before patience would have fired. In 2 of 20 runs it did not,
    and BOTH are in the `random` arm, so the deviation biases the reported gap in favour of the
    project's own hypothesis. scripts/check_checkpoint_fidelity.py detects and lists them.

  WHAT THIS DOES, per affected run
    1. BACKS UP all 6 artifacts (canonical + _final; probs/history/models) - nothing is destroyed.
    2. Retrains with --patience 10, i.e. the protocol AS PRE-REGISTERED. The loop now really
       stops, so the saved best_loss checkpoint is the early-stopped selection by construction.
    3. RESTORES the backed-up `_final` artifacts. This matters: the retrained run's `_final` is
       its early-stop epoch, NOT the epoch-30 maximum-memorization state the PROBE is defined as.

  CONSEQUENCE, stated plainly: for these 2 runs the CANONICAL number and the PROBE number come
  from two different training passes (patience-10 pass and 30-epoch pass respectively). That is
  unavoidable - the two checkpoints have contradictory stopping requirements and can only share a
  pass when the argmin falls early. It must be DISCLOSED in the manuscript, not hidden.

  Re-running is safe: already-compliant runs are detected and skipped.
#>
param(
  [string] $Config    = "config.yaml",
  [int]    $Patience  = 10,
  [int]    $MaxEpochs = 30,
  [int]    $Seed      = 42,
  [string] $Python    = "C:\ProgramData\miniconda3\envs\nodules\python.exe",
  [string] $Root      = "E:\NODULES"
)

Set-Location $Root
$stamp  = Get-Date -Format "yyyyMMdd-HHmmss"
$logDir = Join-Path $Root "outputs\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$master = Join-Path $logDir "remediate_$stamp.log"
$backup = Join-Path $Root "outputs\_pre_remediation\$stamp"

function Write-Log {
  param([string]$Message)
  $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
  Write-Output $line
  Add-Content -Path $master -Value $line -Encoding UTF8
}

# The runs to repair, as identified by scripts/check_checkpoint_fidelity.py.
$targets = @(
  @{ arch = "densenet121";     arm = "random"; fold = 3 },
  @{ arch = "efficientnet_b0"; arm = "random"; fold = 4 }
)

Write-Log "=== remediate_checkpoints START | patience=$Patience ==="
Write-Log "master log: $master"
Write-Log "backup dir: $backup"

# ---------------- BEFORE: record the deviation we are repairing ----------------
Write-Log "[1/4] pre-remediation fidelity audit ..."
$preOut = Join-Path $logDir "fidelity_pre_$stamp.log"
$fidArgs = @("scripts\check_checkpoint_fidelity.py", "--config", $Config, "--patience", "$Patience")
$pre = Start-Process -FilePath $Python -ArgumentList $fidArgs -Wait -NoNewWindow -PassThru -RedirectStandardOutput $preOut -RedirectStandardError ($preOut + ".err")
if (Test-Path $preOut) { Get-Content $preOut | Select-Object -Last 8 | ForEach-Object { Write-Log "    $_" } }
if ($pre.ExitCode -eq 0) {
  Write-Log "[1/4] nothing to repair - every canonical checkpoint already matches. EXITING."
  exit 0
}
Write-Log "[1/4] deviation confirmed (exit $($pre.ExitCode)) - proceeding."

New-Item -ItemType Directory -Path $backup -Force | Out-Null
$O = Join-Path $Root "outputs"

foreach ($t in $targets) {
  $run = "lidc_binary_slice_" + $t.arm + "_rep0_fold" + $t.fold + "_" + $t.arch + "_none_seed" + $Seed
  Write-Log "--- $run ---"

  # ---------------- 2. BACK UP everything before touching it ----------------
  $files = @(
    @{ src = (Join-Path $O "probs\$run.npz");           name = "$run.npz" },
    @{ src = (Join-Path $O "history\$run.json");        name = "$run.json" },
    @{ src = (Join-Path $O "models\$run.pt");           name = "$run.pt" },
    @{ src = (Join-Path $O "probs\${run}_final.npz");   name = "${run}_final.npz" },
    @{ src = (Join-Path $O "history\${run}_final.json"); name = "${run}_final.json" },
    @{ src = (Join-Path $O "models\${run}_final.pt");   name = "${run}_final.pt" }
  )
  foreach ($f in $files) {
    if (-not (Test-Path $f.src)) { Write-Log "  ABORT: expected artifact missing: $($f.src)"; exit 1 }
    Copy-Item -Path $f.src -Destination (Join-Path $backup $f.name) -Force
  }
  Write-Log "  backed up 6 artifacts -> $backup"

  # ---------------- 3. RETRAIN under the pre-registered protocol ----------------
  $runLog = Join-Path $logDir ($run + "_remediated.log")
  Write-Log "  retraining with --patience $Patience ..."
  $t0 = Get-Date
  $pyArgs = @("-u", "-m", "src.train", "--config", $Config, "--dataset", "lidc_binary",
              "--arch", $t.arch, "--arm", $t.arm, "--sample-unit", "slice", "--rep", "0",
              "--fold", "$($t.fold)", "--seed", "$Seed", "--max-epochs", "$MaxEpochs",
              "--patience", "$Patience")
  $r = Start-Process -FilePath $Python -ArgumentList $pyArgs -Wait -NoNewWindow -PassThru -RedirectStandardOutput $runLog -RedirectStandardError ($runLog + ".err")
  $secs = [int]((Get-Date) - $t0).TotalSeconds
  if ($r.ExitCode -ne 0) {
    Write-Log "  FAIL exit=$($r.ExitCode) ${secs}s"
    if (Test-Path ($runLog + ".err")) { Get-Content ($runLog + ".err") -Tail 15 | ForEach-Object { Write-Log "      $_" } }
    Write-Log "  ABORTING. Backups are intact in $backup - restore from there if needed."
    exit 2
  }
  if (Test-Path $runLog) {
    Select-String -Path $runLog -Pattern "CANONICAL|PROBE|best_epoch|early stop" | ForEach-Object { Write-Log ("      " + $_.Line.Trim()) }
  }
  Write-Log "  retrained OK in ${secs}s"

  # ---------------- 4. RESTORE the epoch-30 PROBE ----------------
  # The retrained `_final` is the early-stop epoch, not the max-memorization state the PROBE
  # is defined as. Put the original epoch-30 probe back.
  Copy-Item -Path (Join-Path $backup "${run}_final.npz")  -Destination (Join-Path $O "probs\${run}_final.npz")   -Force
  Copy-Item -Path (Join-Path $backup "${run}_final.json") -Destination (Join-Path $O "history\${run}_final.json") -Force
  Copy-Item -Path (Join-Path $backup "${run}_final.pt")   -Destination (Join-Path $O "models\${run}_final.pt")    -Force
  Write-Log "  restored epoch-30 PROBE artifacts (canonical now from the patience-$Patience pass)"

  # content-validate both checkpoints
  foreach ($rn in @($run, ($run + "_final"))) {
    $v = Start-Process -FilePath $Python -ArgumentList @("scripts\run_is_valid.py", $rn) -Wait -NoNewWindow -PassThru
    if ($v.ExitCode -ne 0) { Write-Log "  FAIL: $rn did not pass run_is_valid after remediation. ABORTING."; exit 3 }
  }
  Write-Log "  both checkpoints content-validated"
}

# ---------------- AFTER: prove the deviation is gone ----------------
Write-Log "[4/4] post-remediation fidelity audit ..."
$postOut = Join-Path $Root "outputs\metrics\checkpoint_fidelity.txt"
$post = Start-Process -FilePath $Python -ArgumentList $fidArgs -Wait -NoNewWindow -PassThru -RedirectStandardOutput $postOut -RedirectStandardError ($postOut + ".err")
if (Test-Path $postOut) { Get-Content $postOut | Select-Object -Last 6 | ForEach-Object { Write-Log "    $_" } }
if ($post.ExitCode -ne 0) {
  Write-Log "=== FAILED: canonical checkpoints STILL deviate from the protocol. Do not report these numbers. ==="
  exit 4
}

Write-Log "=== remediate_checkpoints DONE - all 20 canonical checkpoints now protocol-compliant ==="
Write-Log "NEXT: re-run the gap analysis so the reported numbers come from the repaired checkpoints:"
Write-Log "  $Python scripts\analyze_gap.py --arch densenet121     --probe"
Write-Log "  $Python scripts\analyze_gap.py --arch efficientnet_b0 --probe"
exit 0
