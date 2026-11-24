
# AI Eco Lab setup script
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

# Discover Python through the Windows launcher or PATH
$PythonLauncher = Get-Command py -ErrorAction SilentlyContinue

if ($PythonLauncher) {
    $Python = $PythonLauncher.Source
    $PythonArgs = @("-3")
}
else {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue

    if (-not $PythonCommand) {
        throw "Python 3 was not found. Install Python and ensure it is available through the Windows Python launcher or PATH."
    }

    $Python = $PythonCommand.Source
    $PythonArgs = @()
}

Set-Location $ProjectRoot

$VersionOutput = & $Python @PythonArgs --version 2>&1

if ($LASTEXITCODE -ne 0 -or $VersionOutput -notmatch 'Python (\d+)\.(\d+)') {
    throw "Unable to determine Python version."
}

$PythonMajor = [int]$Matches[1]
$PythonMinor = [int]$Matches[2]

if ($PythonMajor -ne 3 -or $PythonMinor -lt 10) {
    throw "Python 3.10 or newer is required. Found: $VersionOutput"
}

Write-Host "Using $VersionOutput"

# Create the project virtual environment
& $Python @PythonArgs -m venv venv

if ($LASTEXITCODE -ne 0) {
    throw "Failed to create the virtual environment."
}

$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"

# Install project dependencies
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip."
}

& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install project requirements."
}

# Run the test suite
& $VenvPython -m pytest tests/ -v
if ($LASTEXITCODE -ne 0) {
    throw "Some tests failed."
}

Write-Host "Setup complete."
