<#
  run_grid.ps1 - hardened, resumable training driver (DECISIONS D23).

  Runs OUTSIDE Claude Code so it survives client crashes. Launch from YOUR OWN terminal:

      cd E:\NODULES
      powershell -ExecutionPolicy Bypass -File scripts\run_grid.ps1 -Phase subset

  FOUR GATES, in order, abort on failure:
    1. CONFIG CONTRACT - scripts/verify_config_contract.py. Every config.yaml key must have a
                        declared, true relationship to the code (consumed / hardcoded /
                        documentation / deferred). A key no code reads is a false claim, and at
                        grid scale it would be inherited silently by hundreds of runs (D27).
    2. COVERAGE       - scripts/verify_coverage.py --phase <subset|grid>, before any training.
    3. ATOMIC WRITES  - enforced inside src/train.py (.tmp + os.replace for .npz/.json/.pt).
    4. VALIDATED SKIP - scripts/run_is_valid.py decides "already done" by CONTENT
                        (loads, finite, in-range, len(y_prob) == len(test split)),
                        never by mere file existence. Catches stale/truncated artifacts.

  Every run is timestamped into outputs/logs/ so the whole thing is auditable without
  watching the terminal. Re-running resumes cleanly: valid runs are skipped, missing or
  invalid ones are retrained.

  -Phase subset : the 250-patient pre-registered experiment.
  -Phase grid   : REQUIRES 100% coverage of the full 740-patient cohort.
#>
param(
  [string]   $Config    = "config.yaml",
  [ValidateSet("subset","grid")]
  [string]   $Phase     = "subset",
  # NOTE: comma-separated STRINGS, not arrays. `powershell -File script.ps1 -Folds 0,1`
  # silently bound only ONE element (measured 2026-07-19) - a mis-parsed grid would have
  # produced a partial measurement that looked complete. Strings + explicit split are safe.
  [string]   $Archs        = "densenet121,efficientnet_b0",
  [string]   $Arms         = "patient,random",
  [string]   $Folds        = "0,1,2,3,4",
  # Grid axes (D34: S2 is THE experiment). Reps parameterize the repeats (D5): S2 = 3 reps x 5
  # folds = n>=10 for the paired test. SampleUnits = the 2x2 granularity axis (slice + nodule).
  # Seeds per rep come from config.repetition.seed_list; --rep selects which seed src/splits used.
  [string]   $Reps         = "0",
  [string]   $SampleUnits  = "slice",
  # 0 = INHERIT from config.train.* (D48). Pass a value only to deliberately override, and expect
  # verify_grid_consistency to reject the grid if that override contradicts the config.
  [int]      $MaxEpochs = 0,
  [int]      $Patience  = 0,
  [int]      $Seed      = 42,
  # Dataset tag = the split-file prefix. "lidc_binary" is the principal cohort; "lidc_binary_ge3"
  # is the >=3-annotator sensitivity cohort (D37), whose splits src/splits.py writes under that
  # distinct prefix precisely so its runs can NEVER be averaged in with the principal ones.
  # Was hardcoded in three places until 2026-07-26 - the same defect class as the patience hardcode
  # (D48): a value the script asserts rather than receives.
  [string]   $Dataset   = "lidc_binary",
  [string]   $Python    = "C:\ProgramData\miniconda3\envs\nodules\python.exe",
  [string]   $Root      = "E:\NODULES"
)

Set-Location $Root
$stamp  = Get-Date -Format "yyyyMMdd-HHmmss"
$logDir = Join-Path $Root "outputs\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$master = Join-Path $logDir "grid_$stamp.log"

function Write-Log {
  param([string]$Message)
  $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
  Write-Output $line
  # The master log is a CONVENIENCE, not an artifact: the run artifacts and the per-run logs are
  # written elsewhere and are what the gates read. A reader holding the file open (e.g. tailing it
  # with `Get-Content -Wait`, which takes a lock in PS 5.1) must never be able to interrupt a
  # multi-hour grid. Retry briefly, then give up SILENTLY -- console output above is the real record.
  for ($i = 0; $i -lt 3; $i++) {
    try { Add-Content -Path $master -Value $line -Encoding UTF8 -ErrorAction Stop; break }
    catch { Start-Sleep -Milliseconds 120 }
  }
}

$ArchList = @($Archs -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
$ArmList  = @($Arms  -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
$FoldList = @($Folds -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" } | ForEach-Object { [int]$_ })
$RepList  = @($Reps  -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" } | ForEach-Object { [int]$_ })
$SampleList = @($SampleUnits -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })

Write-Log "=== run_grid START | phase=$Phase ==="
Write-Log "master log: $master"

# Expand and LOG the exact run list before doing anything. A mis-parsed parameter must be
# visible here, not discovered later as a silently partial measurement.
$plan = @()
foreach ($su in $SampleList) { foreach ($a in $ArchList) { foreach ($m in $ArmList) { foreach ($r in $RepList) { foreach ($f in $FoldList) {
  $plan += ($Dataset + "_" + $su + "_" + $m + "_rep" + $r + "_fold" + $f + "_" + $a + "_none_seed" + $Seed)
} } } } }
Write-Log ("PLAN: {0} runs | sample_units=[{1}] archs=[{2}] arms=[{3}] reps=[{4}] folds=[{5}]" -f $plan.Count, ($SampleList -join ' '), ($ArchList -join ' '), ($ArmList -join ' '), ($RepList -join ' '), ($FoldList -join ' '))
foreach ($r in $plan) { Write-Log "  planned: $r" }
if ($plan.Count -eq 0) { Write-Log "PLAN is empty - parameter parsing failed. ABORTING."; exit 4 }

# ---------------- GATE 1: CONFIG-CODE CONTRACT (cheapest, fails fastest) ----------------
Write-Log "[GATE 1/4] config-code contract ..."
$ccOut = Join-Path $logDir "contract_$stamp.log"
$g0 = Start-Process -FilePath $Python -ArgumentList @("scripts\verify_config_contract.py", "--config", $Config) `
      -Wait -NoNewWindow -PassThru -RedirectStandardOutput $ccOut -RedirectStandardError ($ccOut + ".err")
if (Test-Path $ccOut) { Get-Content $ccOut | ForEach-Object { Write-Log "    $_" } }
if ($g0.ExitCode -ne 0) {
  Write-Log "[GATE 1/4] FAILED - config.yaml and the code disagree. ABORTING (no training performed)."
  exit 5
}
Write-Log "[GATE 1/4] PASSED"

# ---------------- GATE 2: COVERAGE (abort before touching the GPU) ----------------
Write-Log "[GATE 2/4] coverage assertion (--phase $Phase) ..."
$covOut = Join-Path $logDir "coverage_$stamp.log"
$covErr = $covOut + ".err"
$covArgs = @("scripts\verify_coverage.py", "--config", $Config, "--phase", $Phase)
$g1 = Start-Process -FilePath $Python -ArgumentList $covArgs -Wait -NoNewWindow -PassThru -RedirectStandardOutput $covOut -RedirectStandardError $covErr
if (Test-Path $covOut) { Get-Content $covOut | ForEach-Object { Write-Log "    $_" } }
if ($g1.ExitCode -ne 0) {
  Write-Log "[GATE 2/4] FAILED - coverage gate refused. ABORTING (no training performed)."
  exit 1
}
Write-Log "[GATE 2/4] PASSED"
Write-Log "[GATE 3/4] atomic writes: enforced inside src/train.py (.tmp + os.replace)"

# ---------------- run loop with GATE 4 (content-validated skip) ----------------
$done = 0
$skipped = 0
$total = $plan.Count

foreach ($su in $SampleList) {
 foreach ($arch in $ArchList) {
  foreach ($arm in $ArmList) {
   foreach ($rep in $RepList) {
    foreach ($fold in $FoldList) {
      $run = $Dataset + "_" + $su + "_" + $arm + "_rep" + $rep + "_fold" + $fold + "_" + $arch + "_none_seed" + $Seed

      $skipArgs = @("scripts\run_is_valid.py", $run)
      $v = Start-Process -FilePath $Python -ArgumentList $skipArgs -Wait -NoNewWindow -PassThru
      if ($v.ExitCode -eq 0) {
        Write-Log "SKIP  $run (artifacts valid)"
        $skipped = $skipped + 1
        continue
      }

      $runLog = Join-Path $logDir ($run + ".log")
      $runErr = $runLog + ".err"
      Write-Log "START $run"
      $t0 = Get-Date

      # D48: DO NOT hardcode epoch budget / patience here. 0 = inherit config.train.* (src/train.py
      # falls back to the config when the flag is absent). Hardcoding them is how --max-epochs 8,
      # channels_last and --patience 30 each silently overrode the DECLARED protocol while every
      # run still agreed with every other run and the old gate passed.
      $pyArgs = @("-u", "-m", "src.train", "--config", $Config, "--dataset", $Dataset,
                  "--arch", $arch, "--arm", $arm, "--sample-unit", "$su", "--rep", "$rep",
                  "--fold", "$fold", "--seed", "$Seed")
      if ($MaxEpochs -gt 0) { $pyArgs += @("--max-epochs", "$MaxEpochs") }
      if ($Patience  -gt 0) { $pyArgs += @("--patience",  "$Patience")  }
      $r = Start-Process -FilePath $Python -ArgumentList $pyArgs -Wait -NoNewWindow -PassThru -RedirectStandardOutput $runLog -RedirectStandardError $runErr
      $secs = [int]((Get-Date) - $t0).TotalSeconds

      if ($r.ExitCode -ne 0) {
        Write-Log "FAIL  $run exit=$($r.ExitCode) ${secs}s"
        if (Test-Path $runErr) { Get-Content $runErr -Tail 15 | ForEach-Object { Write-Log "    $_" } }
        Write-Log "ABORTING: a run failed; refusing to continue with a possibly broken pipeline."
        exit 2
      }

      $v2 = Start-Process -FilePath $Python -ArgumentList $skipArgs -Wait -NoNewWindow -PassThru
      if ($v2.ExitCode -ne 0) {
        Write-Log "FAIL  $run produced INVALID artifacts (post-run integrity check). ABORTING."
        exit 3
      }

      if (Test-Path $runLog) {
        Select-String -Path $runLog -Pattern "CANONICAL|PROBE|best_epoch" | ForEach-Object { Write-Log ("    " + $_.Line.Trim()) }
      }
      Write-Log "OK    $run ${secs}s (artifacts validated)"
      $done = $done + 1
    }
   }
  }
 }
}

# ---------------- FINAL GATE: the grid must be ONE experiment (D34 internal consistency) ----------------
Write-Log "[FINAL] grid internal-consistency (single config/max_epochs/patience across all runs) ..."
$gcOut = Join-Path $logDir "grid_consistency_$stamp.log"
$gc = Start-Process -FilePath $Python -ArgumentList @("scripts\verify_grid_consistency.py", "--config", $Config, "--require-stamp") `
      -Wait -NoNewWindow -PassThru -RedirectStandardOutput $gcOut -RedirectStandardError ($gcOut + ".err")
if (Test-Path $gcOut) { Get-Content $gcOut | ForEach-Object { Write-Log "    $_" } }
if ($gc.ExitCode -ne 0) {
  Write-Log "[FINAL] FAILED - the grid spans more than one experiment (config/epochs/patience). Do NOT report the average."
  exit 6
}
Write-Log "[FINAL] PASSED - grid is a single experiment."

Write-Log "=== run_grid DONE | trained=$done | skipped=$skipped | total=$total ==="
exit 0
