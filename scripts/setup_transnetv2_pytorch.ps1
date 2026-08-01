$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "conda_helpers.ps1")
$condaExe = Find-CondaExe
& $condaExe run --no-capture-output -n shotlab-student python -m pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "Không thể cài PyTorch CPU." }
& $condaExe run --no-capture-output -n shotlab-student python -m pip install "transnetv2-pytorch==1.0.5"
if ($LASTEXITCODE -ne 0) { throw "Không thể cài TransNetV2 PyTorch." }
