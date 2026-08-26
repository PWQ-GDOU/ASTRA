[CmdletBinding()]
param(
    [string]$ExperimentOutput,
    [string]$Output,
    [string]$BindHost = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 8090,
    [switch]$NoServe,
    [string]$Python = "D:\tools\python_3_11_6\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

if (-not $ExperimentOutput) {
    $ExperimentOutput = Join-Path $ProjectRoot "outputs\femto_ims_to_comsol_outerloo_v5_exploratory_fitonly_innerloo_full5x120_20260825"
}
if (-not $Output) {
    $Output = Join-Path $ProjectRoot "outputs\engineering_demo\femto_ims_to_comsol_v5"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

$arguments = @(
    "scripts/run_engineering_demo.py",
    "--experiment-output", $ExperimentOutput,
    "--output", $Output,
    "--host", $BindHost,
    "--port", $Port
)
if ($NoServe) {
    $arguments += "--no-serve"
}

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
