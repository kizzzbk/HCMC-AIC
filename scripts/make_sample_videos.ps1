param([int]$Count = 10)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "conda_helpers.ps1")
if ($Count -lt 1 -or $Count -gt 20) { throw "Count phải nằm trong [1, 20]." }
$root = Split-Path $PSScriptRoot -Parent
$condaExe = Find-CondaExe
Assert-ExternalTool "ffmpeg" $condaExe
& $condaExe run --no-capture-output -n shotlab-student python (Join-Path $PSScriptRoot "make_sample_videos.py") `
  --output (Join-Path $root "data\videos") --count $Count
if ($LASTEXITCODE -ne 0) { throw "Tạo video mẫu thất bại." }
