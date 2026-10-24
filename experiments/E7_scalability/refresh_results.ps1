$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Join-Path $PSScriptRoot '..\..')
& '.\.venv\Scripts\python.exe' -m scripts.organize_experiments --experiment E7
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
