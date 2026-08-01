param([string]$Methods = "pyscenedetect", [string]$Primary = "pyscenedetect", [string]$Device = "auto")
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "conda_helpers.ps1")
$root = Split-Path $PSScriptRoot -Parent
$condaExe = Find-CondaExe
$methodArgs = $Methods.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries)
& $condaExe run --no-capture-output -n shotlab-student shotlab-student `
  --input (Join-Path $root "data\videos") --output (Join-Path $root "output") `
  --methods $methodArgs --primary $Primary --device $Device
if ($LASTEXITCODE -ne 0) { throw "Pipeline thất bại hoặc còn TODO." }
