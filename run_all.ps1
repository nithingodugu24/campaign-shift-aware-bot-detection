param([switch]$SkipDownloads)
$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Create .venv and install requirements using README.md before running this script.'
}
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Path 'logs' -Force | Out-Null
function Invoke-ProjectPython {
    param([string[]]$Arguments)
    & $projectPython @Arguments 2>&1 | Tee-Object -FilePath 'logs\reproduce.log' -Append
    if ($LASTEXITCODE -ne 0) { throw "Project command failed: $Arguments" }
}
Invoke-ProjectPython -Arguments @('-m','pytest','-q')
if (-not $SkipDownloads) {
    Invoke-ProjectPython -Arguments @('-m','scripts.prepare_public_data')
    Invoke-ProjectPython -Arguments @('-m','scripts.prepare_snapshot_targets')
}
Invoke-ProjectPython -Arguments @('-m','scripts.prepare_collection_pool')
Invoke-ProjectPython -Arguments @('-m','scripts.prepare_synthetic')
Invoke-ProjectPython -Arguments @('-m','src.run_experiments','--data','data/processed/cresci2017','--target','data/processed/gilani-2017','--config','configs/pilot.yaml','--output','results/public')
Invoke-ProjectPython -Arguments @('-m','src.run_experiments','--data','data/processed/collection_pool','--target','data/processed/cresci-rtbust-2019','--config','configs/collection.yaml','--output','results/collection')
Invoke-ProjectPython -Arguments @('-m','src.run_experiments','--data','data/processed/synthetic_source','--target','data/processed/synthetic_target','--config','configs/pilot.yaml','--output','results/synthetic')
Invoke-ProjectPython -Arguments @('-m','scripts.analyze_results','results/public','results/collection','results/synthetic')
Invoke-ProjectPython -Arguments @('-m','scripts.graph_sensitivity')
Invoke-ProjectPython -Arguments @('-m','scripts.validate_evidence')
Invoke-ProjectPython -Arguments @('-m','scripts.build_report_content')
Invoke-ProjectPython -Arguments @('-m','scripts.organize_experiments')
