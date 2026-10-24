[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)][int]$Port = 8501,
    [switch]$Setup,
    [switch]$CheckOnly,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath $projectPython)) {
        Write-Host "Creating the project's isolated Python environment..."
        & $Python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Install Python 3.12, or pass -Python with its executable path." }
        $Setup = $true
    }
    if ($Setup) {
        & $projectPython -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt
        if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
    }
    & $projectPython -m scripts.doctor
    if ($LASTEXITCODE -ne 0) { throw "Application checks failed. Run this script with -Setup to install the combined dependencies." }
    if (-not $CheckOnly) {
        Write-Host "Opening MedInspect at http://localhost:$Port. Press Ctrl+C to stop."
        & $projectPython -m streamlit run app.py --server.address 127.0.0.1 --server.port $Port
        if ($LASTEXITCODE -ne 0) { throw "Streamlit could not start. Try another -Port if this one is in use." }
    }
} finally {
    Pop-Location
}
