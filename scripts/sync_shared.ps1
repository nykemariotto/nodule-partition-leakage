# =====================================================================================
# sync_shared.ps1 - push the current deliverables to the Google Drive folder shared with
# the advisor. Run after any change to the manuscript or the response document.
#
#   powershell -ExecutionPolicy Bypass -File scripts\sync_shared.ps1
#
# WHAT IT DOES AND DOES NOT DO
#   * COPIES ONLY the reviewer-facing deliverables. Working files (DECISIONS.md, journal.md,
#     code, outputs) stay in the repository -- the shared folder is for review, not for state.
#   * Files are DATE-STAMPED, never overwritten in place. The advisor may have annotated an
#     earlier version, and silently replacing it would destroy his comments. Old versions
#     accumulate; that is deliberate.
#   * Nothing is ever deleted from the shared folder.
#
# NOTE: ASCII only. Non-ASCII punctuation has broken the PowerShell parser in this repo before.
# =====================================================================================
# -Only sends a subset. The deliverables do not always advance together: after an edit to the
# manuscript SOURCE the compiled PDF is one step behind, and shipping it next to an updated response
# document would show the advisor prose he has already commented on. Use -Only response in that
# window, then sync everything once the PDF has been recompiled.
param(
  [string] $Root   = "E:\NODULES",
  [string] $Shared = "G:\Drives compartilhados\Laboratorio de Fisica Medica - IBB\Projetos\Projeto - Nodulos pulmonares\Ressubmissao",
  [ValidateSet("all", "response", "manuscript", "highlighted")]
  [string] $Only   = "all"
)

# The real path carries accents; resolve it rather than hardcoding a possibly-wrong literal.
if (-not (Test-Path $Shared)) {
  $base = "G:\Drives compartilhados"
  $cand = Get-ChildItem -Path $base -Recurse -Directory -Depth 4 -ErrorAction SilentlyContinue |
          Where-Object { $_.Name -match "Ressubmiss" } | Select-Object -First 1
  if ($cand) { $Shared = $cand.FullName }
}
if (-not (Test-Path $Shared)) {
  Write-Output "FATAL: shared folder not found. Is Google Drive mounted?"
  exit 1
}
Write-Output "shared folder: $Shared"

$stamp = Get-Date -Format "yyyy-MM-dd"
$src   = Join-Path $Root "paper\2_resubmission"

# source file -> destination name (the {0} is replaced by the date stamp)
$items = @(
  @{ Key = "response";    From = "Response-to-Reviewers.docx";           To = "Reviewers response ({0}).docx" },
  @{ Key = "manuscript";  From = "overleaf_upload.pdf";                  To = "Manuscript_current ({0}).pdf" },
  @{ Key = "highlighted"; From = "Manuscript_highlighted-changes.pdf";   To = "Manuscript_highlighted-changes ({0}).pdf" }
)
if ($Only -ne "all") { $items = $items | Where-Object { $_.Key -eq $Only } }
Write-Output ("sending: {0}" -f (($items | ForEach-Object { $_.Key }) -join ", "))

foreach ($it in $items) {
  $f = Join-Path $src $it.From
  if (-not (Test-Path $f)) { Write-Output ("  SKIP  {0} (not found)" -f $it.From); continue }
  $dest = Join-Path $Shared ($it.To -f $stamp)
  if (Test-Path $dest) {
    # same date, already synced today: only replace if the source is genuinely newer
    if ((Get-Item $f).LastWriteTime -le (Get-Item $dest).LastWriteTime) {
      Write-Output ("  same  {0}" -f (Split-Path $dest -Leaf)); continue
    }
  }
  Copy-Item $f $dest -Force
  Write-Output ("  sent  {0}" -f (Split-Path $dest -Leaf))
}

Write-Output ""
Write-Output "Shared folder now contains:"
Get-ChildItem $Shared -File | Where-Object { $_.Name -notlike "~*" } |
  Sort-Object LastWriteTime |
  ForEach-Object { Write-Output ("  {0,-46} {1,8:N0} KB  {2}" -f $_.Name, ($_.Length/1KB), $_.LastWriteTime.ToString("yyyy-MM-dd HH:mm")) }
