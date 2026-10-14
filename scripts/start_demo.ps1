$ErrorActionPreference = 'Stop'

function Get-PortProcess {
    param([int]$Port)
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Ignore
    if ($conn) {
        $proc = Get-Process -Id $conn[0].OwningProcess -ErrorAction Ignore
        return $proc
    }
    return $null
}

$proc8001 = Get-PortProcess 8001
$proc8002 = Get-PortProcess 8002

if ($proc8001) {
    Write-Host "Port 8001 is already in use by process $($proc8001.Name) (PID: $($proc8001.Id)). Please stop it first."
    exit 1
}
if ($proc8002) {
    Write-Host "Port 8002 is already in use by process $($proc8002.Name) (PID: $($proc8002.Id)). Please stop it first."
    exit 1
}

Write-Host "Starting IntentGuard on port 8001..."
$igCmd = "python -m intentguard.api"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle = 'IntentGuard (8001)'; $igCmd" -WorkingDirectory "$PSScriptRoot\..\intentguard"

Write-Host "Starting Dashboard on port 8002..."
$dbCmd = "python demo_server.py"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle = 'Dashboard (8002)'; $dbCmd" -WorkingDirectory "$PSScriptRoot\..\agent"

Write-Host "Waiting for services to start (up to 60s)..."
$timeout = 60
$stopwatch = [Diagnostics.Stopwatch]::StartNew()
$igUp = $false
$dbUp = $false

while ($stopwatch.Elapsed.TotalSeconds -lt $timeout -and (-not $igUp -or -not $dbUp)) {
    if (-not $igUp) {
        try {
            $res = Invoke-WebRequest -Uri "http://127.0.0.1:8001/health" -UseBasicParsing -ErrorAction Ignore
            if ($res.StatusCode -eq 200) { $igUp = $true }
        } catch { }
    }
    if (-not $dbUp) {
        try {
            $res = Invoke-WebRequest -Uri "http://127.0.0.1:8002/api/health" -UseBasicParsing -ErrorAction Ignore
            if ($res.StatusCode -eq 200) { $dbUp = $true }
        } catch { }
    }
    Start-Sleep -Milliseconds 500
}

Write-Host ""
Write-Host "=== SERVICE STATUS ==="

if ($igUp) { Write-Host "IntentGuard (8001): ok" } else { Write-Host "IntentGuard (8001): down (timeout)" }
if ($dbUp) { Write-Host "Dashboard (8002): ok" } else { Write-Host "Dashboard (8002): down (timeout)" }

try {
    $res = Invoke-WebRequest -Uri "http://localhost:11434/" -UseBasicParsing -ErrorAction Stop
    Write-Host "Ollama (11434): ok"
} catch {
    Write-Host "Ollama (11434): down ($($_.Exception.Message))"
}

try {
    $res = Invoke-WebRequest -Uri "http://localhost:8000/api/schema/" -UseBasicParsing -ErrorAction Stop
    Write-Host "Django backend (8000): ok"
} catch {
    Write-Host "Django backend (8000): down ($($_.Exception.Message))"
}

Write-Host ""
Write-Host "Note: Ollama and Docker (for the backend) are started separately."
Write-Host "A down backend only disables sending; dry runs still work."

