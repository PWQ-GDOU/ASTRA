[CmdletBinding()]
param(
    [ValidateSet("preflight", "smoke", "full", "all")]
    [string]$Phase = "all",
    [string]$RunId = ("femto-ims-to-comsol-" + [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")),
    [string]$Device = "cpu",
    [double]$TimeoutHours = 8,
    [int]$MaxRetries = 1
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepositoryRoot
$OutputRoot = "/workspace/outputs/reproducibility/$RunId"

& docker compose run --rm --build reproduce python scripts/reproduce_femto_ims_to_comsol.py `
    --phase $Phase --data-root /workspace/data --output-root $OutputRoot `
    --device $Device --timeout-hours $TimeoutHours --max-retries $MaxRetries
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
