[CmdletBinding()]
param(
    [switch]$SkipTorch,
    [switch]$SkipDependencies
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$TempDir = Join-Path $Root ".tmp"
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
$env:TEMP = $TempDir
$env:TMP = $TempDir

if (-not (Test-Path (Join-Path $Root ".venv\Scripts\Activate.ps1"))) {
    python -m venv (Join-Path $Root ".venv")
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}

& $VenvPython -m ensurepip --upgrade --default-pip
if ($LASTEXITCODE -ne 0) { throw "ensurepip failed" }
& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed" }
if (-not $SkipTorch) {
    & $VenvPython -m pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed" }
}
if (-not $SkipDependencies) {
    & $VenvPython -m pip install -r (Join-Path $Root "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "Dependencies installation failed" }
    & $VenvPython -m pip install -e $Root
    if ($LASTEXITCODE -ne 0) { throw "Editable install failed" }
}
& $VenvPython (Join-Path $PSScriptRoot "check_environment.py") --strict
if ($LASTEXITCODE -ne 0) { throw "Environment check failed" }

