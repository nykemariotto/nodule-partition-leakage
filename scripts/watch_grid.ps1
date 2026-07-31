# =====================================================================================
# watch_grid.ps1 - live progress for a running grid. READ-ONLY.
#
#   powershell -ExecutionPolicy Bypass -File E:\NODULES\scripts\watch_grid.ps1
#   powershell -ExecutionPolicy Bypass -File E:\NODULES\scripts\watch_grid.ps1 -Dataset lidc_binary
#
# Run it in a SECOND window while run_grid.ps1 works in the first. It touches nothing the grid
# uses: it reads artifact timestamps and the per-run log that train.py already writes. Editing
# run_grid.ps1 or src/train.py mid-grid is not an option -- a relaunch after a crash would then
# pick up a different driver, and half the grid would run under one version and half under another.
#
# ETA is weighted PER ARCHITECTURE, because DenseNet-121 costs roughly twice EfficientNet-B0 per
# run; a flat mean over completed runs would read far too optimistic while the DenseNet half is
# still going, and far too pessimistic afterwards.
#
# NOTE: ASCII only, and no `&&` / ternary / null-coalescing. Windows PowerShell 5.1.
# =====================================================================================
param(
  [string] $Root     = (Split-Path $PSScriptRoot -Parent),
  [string] $Dataset  = "lidc_binary_ge3",
  [int]    $Total     = 60,
  [int]    $MaxEpochs = 30,      # config.yaml training.max_epochs; used only to draw the bar
  [int]    $Every     = 20       # seconds between refreshes
)

$probs = Join-Path $Root "outputs\probs"
$logs  = Join-Path $Root "outputs\logs"

function Fmt([double] $minutes) {
  if ($minutes -le 0 -or [double]::IsNaN($minutes)) { return "-" }
  $h = [math]::Floor($minutes / 60); $m = [math]::Round($minutes - ($h * 60))
  if ($h -gt 0) { return ("{0}h {1:00}m" -f $h, $m) }
  return ("{0}m" -f $m)
}

function ArchOf([string] $name) {
  if ($name -match "densenet121") { return "densenet121" }
  if ($name -match "efficientnet_b0") { return "efficientnet_b0" }
  return "other"
}

# Fallback cost per run, in minutes: measured medians on the principal cohort (DenseNet 54,
# EfficientNet 29) scaled by 0.66 for this cohort's smaller training set. Used only until an
# architecture has finished a run of its own, after which the observed mean takes over.
$fallback = @{ "densenet121" = 36.0; "efficientnet_b0" = 19.0 }

while ($true) {
  Clear-Host
  $done = @(Get-ChildItem $probs -Filter "$Dataset*_seed42.npz" -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -notlike "*_final.npz" } | Sort-Object LastWriteTime)

  $grid = Get-ChildItem $logs -Filter "grid_*.log" -ErrorAction SilentlyContinue |
          Sort-Object LastWriteTime | Select-Object -Last 1
  $t0 = $null
  if ($grid) { $t0 = $grid.CreationTime }
  if ($done.Count -gt 0 -and $t0 -eq $null) { $t0 = $done[0].LastWriteTime }

  Write-Host ""
  Write-Host ("  {0}   {1} / {2} runs" -f $Dataset, $done.Count, $Total) -ForegroundColor Cyan
  Write-Host ("  " + ("-" * 66))

  # ---- per-architecture accounting -------------------------------------------------------
  $meanBy = @{}
  foreach ($a in @("densenet121", "efficientnet_b0")) {
    $mine = @($done | Where-Object { (ArchOf $_.Name) -eq $a })
    $durs = @()
    for ($i = 1; $i -lt $mine.Count; $i++) {
      $d = ($mine[$i].LastWriteTime - $mine[$i - 1].LastWriteTime).TotalMinutes
      if ($d -gt 0 -and $d -lt 180) { $durs += $d }
    }
    if ($durs.Count -ge 2) {
      $sorted = $durs | Sort-Object
      $meanBy[$a] = $sorted[[math]::Floor($sorted.Count / 2)]      # median, robust to a stall
    } else {
      $meanBy[$a] = $fallback[$a]
    }
    $per = ($Total / 2)
    $obs = "estimated"
    if ($durs.Count -ge 2) { $obs = "observed" }
    Write-Host ("  {0,-16} {1,3} / {2,-3} done    {3,7} per run  ({4})" -f `
                $a, $mine.Count, $per, (Fmt $meanBy[$a]), $obs)
  }

  # ---- elapsed and ETA -------------------------------------------------------------------
  $elapsed = 0.0
  if ($t0) { $elapsed = ((Get-Date) - $t0).TotalMinutes }
  $left = 0.0
  foreach ($a in @("densenet121", "efficientnet_b0")) {
    $remaining = ($Total / 2) - @($done | Where-Object { (ArchOf $_.Name) -eq $a }).Count
    if ($remaining -gt 0) { $left += $remaining * $meanBy[$a] }
  }
  Write-Host ("  " + ("-" * 66))
  Write-Host ("  elapsed   {0,-12} remaining  {1,-12} ETA {2}" -f `
              (Fmt $elapsed), (Fmt $left), (Get-Date).AddMinutes($left).ToString("ddd HH:mm"))

  # ---- the run in flight -----------------------------------------------------------------
  $doneNames = @{}
  foreach ($d in $done) { $doneNames[$d.BaseName] = $true }
  $cur = Get-ChildItem $logs -Filter "$Dataset*.log" -ErrorAction SilentlyContinue |
         Where-Object { -not $doneNames.ContainsKey($_.BaseName) } |
         Sort-Object LastWriteTime | Select-Object -Last 1

  Write-Host ""
  if ($cur) {
    $tail = @(Get-Content $cur.FullName -Tail 40 -ErrorAction SilentlyContinue)
    $epLines = @($tail | Where-Object { $_ -match "^\s*epoch\s+\d+:" })
    $runMin = ((Get-Date) - $cur.CreationTime).TotalMinutes
    Write-Host ("  in flight   {0}" -f $cur.BaseName) -ForegroundColor Yellow
    if ($epLines.Count -gt 0) {
      $last = $epLines[-1]
      $null = $last -match "^\s*epoch\s+(\d+):"
      $ep = [int]$Matches[1]
      $perEp = 0.0
      if ($ep -gt 0) { $perEp = $runMin / $ep }
      $epLeft = $MaxEpochs - $ep
      if ($epLeft -lt 0) { $epLeft = 0 }
      Write-Host ("              epoch {0}/{1} - {2} elapsed - {3:N1} min/epoch - about {4} left" -f `
                  $ep, $MaxEpochs, (Fmt $runMin), $perEp, (Fmt ($epLeft * $perEp)))
      Write-Host ("              {0}" -f $last.Trim())
    } else {
      Write-Host ("              starting up - {0} elapsed, no epoch logged yet" -f (Fmt $runMin))
    }
  } else {
    Write-Host "  in flight   (none - between runs, or the grid has finished)"
  }

  Write-Host ""
  Write-Host ("  refreshing every {0}s - Ctrl+C to stop watching (the grid keeps running)" -f $Every) -ForegroundColor DarkGray
  Start-Sleep -Seconds $Every
}
