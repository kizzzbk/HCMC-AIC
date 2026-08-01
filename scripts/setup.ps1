$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "conda_helpers.ps1")
$root = Split-Path $PSScriptRoot -Parent
$condaExe = Find-CondaExe

& $condaExe env update --name shotlab-student --file (Join-Path $root "environment.yml") --prune
if ($LASTEXITCODE -ne 0) { & $condaExe env create --file (Join-Path $root "environment.yml") }
if ($LASTEXITCODE -ne 0) { throw "Không thể tạo environment." }

Assert-ExternalTool "ffmpeg" $condaExe
Assert-ExternalTool "ffprobe" $condaExe

& $condaExe run --no-capture-output -n shotlab-student python -m pip install -e $root
if ($LASTEXITCODE -ne 0) { throw "Không thể cài starter package." }