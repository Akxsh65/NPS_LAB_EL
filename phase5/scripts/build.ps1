# Build manuscript PDF (MiKTeX on Windows)
$ErrorActionPreference = "Stop"
$manuscript = Join-Path $PSScriptRoot "..\manuscript"
Set-Location $manuscript

$bin = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64"
$pdflatex = if (Test-Path "$bin\pdflatex.exe") { "$bin\pdflatex.exe" } else { "pdflatex" }
$bibtex = if (Test-Path "$bin\bibtex.exe") { "$bin\bibtex.exe" } else { "bibtex" }

foreach ($run in 1..2) {
    & $pdflatex -interaction=nonstopmode main.tex | Out-Null
}
& $bibtex main | Out-Null
foreach ($run in 1..2) {
    & $pdflatex -interaction=nonstopmode main.tex | Out-Null
}

if (Test-Path main.pdf) {
    Write-Host "Built: $(Resolve-Path main.pdf)"
} else {
    Write-Host "Build failed - check main.log"
    exit 1
}
