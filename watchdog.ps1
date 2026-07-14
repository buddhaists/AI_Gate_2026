# ============================================================
#  AI Camera Gate - LPR Engine Watchdog  v2
#  watchdog.ps1
#
#  偵測方式：
#    1. 優先檢查 port 8081 是否有 LISTENING 狀態
#    2. 同時確認有 python.exe 在運作
#  若服務停止，自動重啟 lpr_engine.py
#  每 30 秒執行一次檢查。
# ============================================================

$EnginePath    = "d:\AntiGravity\ai camera-gate"
$EngineScript  = "lpr_engine.py"
$PythonExe     = "C:\Users\username\AppData\Local\Programs\Python\Python312\python.exe"
$LogFile       = "D:\AntiGravity\lpr_data\watchdog.log"
$CheckInterval = 30   # seconds

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $Message"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

function Is-EngineRunning {
    # Method 1: port 8081 LISTENING
    $portUp = netstat -ano 2>$null | Select-String "0\.0\.0\.0:8081\s+0\.0\.0\.0:0\s+LISTENING"
    if ($portUp) { return $true }

    # Method 2: any python.exe process present (fallback - engine still loading)
    $pyProcs = Get-Process python -ErrorAction SilentlyContinue
    if ($pyProcs -and $pyProcs.Count -gt 0) { return $true }

    return $false
}

Write-Log "=== LPR Engine Watchdog v2 started ==="

$restartCount = 0

while ($true) {
    if (Is-EngineRunning) {
        # Engine OK - silent pass (log only every 10 minutes for auditing)
        $now = Get-Date
        if ($now.Minute % 10 -eq 0 -and $now.Second -lt $CheckInterval) {
            Write-Log "Heartbeat: engine running normally."
        }
    } else {
        $restartCount++
        Write-Log "WARNING: LPR engine not detected (port 8081 down, no python.exe)! Restarting... (#$restartCount)"
        try {
            # Kill any orphan python processes that might be stuck
            Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

            Start-Sleep -Seconds 2

            Start-Process -FilePath $PythonExe `
                          -ArgumentList $EngineScript `
                          -WorkingDirectory $EnginePath `
                          -WindowStyle Hidden

            Write-Log "Engine restart #$restartCount triggered. Waiting 60s for startup..."
            Start-Sleep -Seconds 60   # Give extra time for model loading before next check

        } catch {
            Write-Log "ERROR: Failed to restart engine: $_"
        }
    }

    Start-Sleep -Seconds $CheckInterval
}
