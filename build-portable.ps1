$ErrorActionPreference = "Stop"

cargo build --release --locked

$outputDirectory = Join-Path $PSScriptRoot "dist"
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$executable = Join-Path $PSScriptRoot "target\release\foto-cleaner.exe"
$destination = Join-Path $outputDirectory "Foto-cleaner.exe"
Copy-Item -LiteralPath $executable -Destination $destination -Force

Write-Host "Executabil creat: dist\Foto-cleaner.exe"
