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
    Write-Host "Process listening on 8001: $($proc8001.Name) (PID: $($proc8001.Id))"
    Stop-Process -Id $proc8001.Id -Force
    Write-Host "Stopped $($proc8001.Name)."
} else {
    Write-Host "No process listening on 8001."
}

if ($proc8002) {
    Write-Host "Process listening on 8002: $($proc8002.Name) (PID: $($proc8002.Id))"
    Stop-Process -Id $proc8002.Id -Force
    Write-Host "Stopped $($proc8002.Name)."
} else {
    Write-Host "No process listening on 8002."
}
