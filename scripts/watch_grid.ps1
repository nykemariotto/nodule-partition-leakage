# =====================================================================================
# watch_grid.ps1 - live progress for a running grid. READ-ONLY.
#
#   powershell -ExecutionPolicy Bypass -File E:\NODULES\scripts\watch_grid.ps1
#
# Run it in a SECOND window while run_grid.ps1 works in the first. It touches nothing the grid
# uses: it reads the grid's own log and the artifact timestamps. Editing run_grid.ps1 or
# src/train.py mid-grid is not an option -- a relaunch after a crash would then pick up a different
# driver, and half the grid would run under one version and half under another.
#
# IT READS THE PLAN FROM THE LOG, and does not glob for artifacts. The first version globbed
# "$Dataset*_seed42.npz" and hardcoded the two CNN architectures, which broke in three ways the
# moment a third architecture and a second cohort existed:
#   * "lidc_binary*" also matches "lidc_binary_ge3*", because one dataset name is a PREFIX of the
#     other, so the sensitivity cohort's 60 runs were counted as part of the principal grid;
#   * vit_small did not appear at all, having not been in the hardcoded list;
#   * per-architecture totals came from Total/2, which printed "112,5 done" -- half a run.
# run_grid.ps1 writes one "planned: <tag>" line per run before it starts. That is the exact set,
# with no assumption about how many architectures, arms, cohorts or sample units exist, so this
# version stays correct for any grid.
#
# ETA is weighted PER ARCHITECTURE, because per-run cost differs by a factor of three or more
# between them; a flat mean would read badly wrong while one architecture's block is still going.
#
# NOTE: ASCII only, and no `&&` / ternary / null-coalescing. Windows PowerShell 5.1.
# =====================================================================================
param(
  [string] $Root      = (Split-Path $PSScriptRoot -Parent),
  [string] $GridLog   = "",     # defaults to the newest outputs/logs/grid_*.log
  [int]    $MaxEpochs = 30,     # config.yaml training.max_epochs; only used to draw the bar
  [int]    $Every     = 20
)

$probs = Join-Path $Root "outputs\probs"
$logs  = Join-Path $Root "outputs\logs"

function Fmt([double] $minutes) {
  if ($minutes -le 0 -or [double]::IsNaN($minutes)) { return "-" }
  $h = [math]::Floor($minutes / 60); $m = [math]::Round($minutes - ($h * 60))
  if ($h -gt 0) { return ("{0}h {1:00}m" -f $h, $m) }
  return ("{0}m" -f $m)
}

# tag = {dataset}_{sample_unit}_{arm}_rep{R}_fold{K}_{arch}_none_seed{N}
function ArchOf([string] $tag) {
  if ($tag -match "_fold\d+_(.+)_none_seed\d+$") { return $Matches[1] }
  return "unknown"
}

# Fallback cost per run, minutes, until an architecture has completed runs of its own in THIS grid.
# Measured medians on the principal cohort at slice level; the nodule sample unit costs about a
# fifth of that, which the observed value picks up as soon as there is one.
$fallback = @{ "densenet121" = 54.0; "efficientnet_b0" = 29.0; "swin_tiny" = 74.0; "vit_small" = 21.0 }

while ($true) {
  Clear-Host

  if ($GridLog -ne "") { $grid = Get-Item $GridLog -ErrorAction SilentlyContinue }
  else { $grid = Get-ChildItem $logs -Filter "grid_*.log" -ErrorAction SilentlyContinue |
                 Sort-Object LastWriteTime | Select-Object -Last 1 }
  if (-not $grid) { Write-Host "  no grid log found in $logs"; Start-Sleep -Seconds $Every; continue }

  $planned = @(Get-Content $grid.FullName -ErrorAction SilentlyContinue |
               Where-Object { $_ -match "planned:\s+(\S+)\s*$" } |
               ForEach-Object { if ($_ -match "planned:\s+(\S+)\s*$") { $Matches[1] } })
  if ($planned.Count -eq 0) { Write-Host "  grid log has no plan yet"; Start-Sleep -Seconds $Every; continue }

  $doneItems = @()
  foreach ($t in $planned) {
    $f = Join-Path $probs ($t + ".npz")
    if (Test-Path $f) { $doneItems += (Get-Item $f) }
  }
  $doneNames = @{}
  foreach ($d in $doneItems) { $doneNames[$d.BaseName] = $true }
  $doneItems = @($doneItems | Sort-Object LastWriteTime)

  Write-Host ""
  Write-Host ("  {0}   {1} / {2} runs" -f $grid.Name, $doneItems.Count, $planned.Count) -ForegroundColor Cyan
  Write-Host ("  " + ("-" * 68))

  $archs = @($planned | ForEach-Object { ArchOf $_ } | Sort-Object -Unique)
  $meanBy = @{}
  foreach ($a in $archs) {
    $mine = @($doneItems | Where-Object { (ArchOf $_.BaseName) -eq $a })
    $durs = @()
    for ($i = 1; $i -lt $mine.Count; $i++) {
      $d = ($mine[$i].LastWriteTime - $mine[$i - 1].LastWriteTime).TotalMinutes
      if ($d -gt 0 -and $d -lt 240) { $durs += $d }
    }
    $obs = "estimated"
    if ($durs.Count -ge 2) {
      $sorted = $durs | Sort-Object
      $meanBy[$a] = $sorted[[math]::Floor($sorted.Count / 2)]     # median: one stall must not skew it
      $obs = "observed"
    } elseif ($fallback.ContainsKey($a)) { $meanBy[$a] = $fallback[$a] }
    else { $meanBy[$a] = 30.0 }
    $tot = @($planned | Where-Object { (ArchOf $_) -eq $a }).Count
    Write-Host ("  {0,-18} {1,3} / {2,-3} done   {3,8} per run  ({4})" -f `
                $a, $mine.Count, $tot, (Fmt $meanBy[$a]), $obs)
  }

  $elapsed = ((Get-Date) - $grid.CreationTime).TotalMinutes
  $left = 0.0
  foreach ($a in $archs) {
    $rem = @($planned | Where-Object { (ArchOf $_) -eq $a }).Count -
           @($doneItems | Where-Object { (ArchOf $_.BaseName) -eq $a }).Count
    if ($rem -gt 0) { $left += $rem * $meanBy[$a] }
  }
  Write-Host ("  " + ("-" * 68))
  Write-Host ("  elapsed   {0,-12} remaining  {1,-12} ETA {2}" -f `
              (Fmt $elapsed), (Fmt $left), (Get-Date).AddMinutes($left).ToString("ddd HH:mm"))

  $curTag = @($planned | Where-Object { -not $doneNames.ContainsKey($_) }) | Select-Object -First 1
  Write-Host ""
  if ($curTag) {
    $cl = Join-Path $logs ($curTag + ".log")
    if (Test-Path $cl) {
      $ci = Get-Item $cl
      $tail = @(Get-Content $cl -Tail 40 -ErrorAction SilentlyContinue)
      $epLines = @($tail | Where-Object { $_ -match "^\s*epoch\s+\d+:" })
      $runMin = ((Get-Date) - $ci.CreationTime).TotalMinutes
      Write-Host ("  in flight   {0}" -f $curTag) -ForegroundColor Yellow
      if ($epLines.Count -gt 0) {
        $last = $epLines[-1]
        $null = $last -match "^\s*epoch\s+(\d+):"
        $ep = [int]$Matches[1]
        $perEp = 0.0
        if ($ep -gt 0) { $perEp = $runMin / $ep }
        $epLeft = $MaxEpochs - $ep
        if ($epLeft -lt 0) { $epLeft = 0 }
        Write-Host ("              epoch {0}/{1} - {2} elapsed - {3:N2} min/epoch - about {4} left" -f `
                    $ep, $MaxEpochs, (Fmt $runMin), $perEp, (Fmt ($epLeft * $perEp)))
        Write-Host ("              {0}" -f $last.Trim())
      } else {
        Write-Host ("              starting up - {0} elapsed, no epoch logged yet" -f (Fmt $runMin))
      }
    } else {
      Write-Host ("  in flight   {0}  (log not open yet)" -f $curTag) -ForegroundColor Yellow
    }
  } else {
    Write-Host "  in flight   (none - every planned run has its artifact; the grid should be done)"
  }

  Write-Host ""
  Write-Host ("  refreshing every {0}s - Ctrl+C stops watching, the grid keeps running" -f $Every) -ForegroundColor DarkGray
  Start-Sleep -Seconds $Every
}
