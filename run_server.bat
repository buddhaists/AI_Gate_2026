@echo off
title AI LPR Gate Monitor Server (Python 3.12)
echo ===================================================
echo   AI Camera Gate Monitor - Engine Starting...
echo   Python: 3.12
echo ===================================================
cd /d "D:\AntiGravity\ai camera-gate"
"C:\Users\username\AppData\Local\Programs\Python\Python312\python.exe" -u lpr_engine.py
pause
