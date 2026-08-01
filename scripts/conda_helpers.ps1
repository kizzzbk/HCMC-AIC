function Find-CondaExe {
    $candidates = @(
        (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
        "C:\ProgramData\miniconda3\Scripts\conda.exe", "C:\ProgramData\anaconda3\Scripts\conda.exe"
    )
    $fromPath = Get-Command conda -ErrorAction SilentlyContinue
    if ($fromPath) { return $fromPath.Source }
    foreach ($candidate in $candidates) { if (Test-Path -LiteralPath $candidate) { return $candidate } }
    throw "Không tìm thấy conda.exe."
}

function Assert-ExternalTool([string]$Name, $CondaExe) {
    if (Get-Command $Name -ErrorAction SilentlyContinue) {
        return
    }
    if ($CondaExe) {
        & $CondaExe run -n shotlab-student --no-capture-output cmd /c "$Name -version" > $null 2>&1
        if ($LASTEXITCODE -eq 0) {
            return
        }
    }
    throw "Không tìm thấy $Name trong PATH hoặc conda environment 'shotlab-student'."
}
