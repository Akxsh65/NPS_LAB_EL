# Start presentation: inference API + static file server
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

Write-Host "Repo: $Root"
Write-Host ""
Write-Host "Starting inference API on http://127.0.0.1:8765 ..."
$api = Start-Process -FilePath "python" -ArgumentList "presentation/api_server.py" -PassThru -WindowStyle Normal

Start-Sleep -Seconds 15

Write-Host "Starting static server on http://localhost:8080/presentation/ ..."
Write-Host "Press Ctrl+C to stop the static server (close the API window separately)."
Write-Host ""

try {
    python -m http.server 8080
} finally {
    if (-not $api.HasExited) {
        Write-Host "Stopping API (pid $($api.Id))..."
        Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
    }
}
