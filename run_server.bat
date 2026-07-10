@echo off
title AI LPR Gate Monitor Server
echo ===================================================
echo   AI 智慧學校大門車牌監控系統 啟動中...
echo ===================================================
cd /d "D:\AntiGravity\ai camera-gate"
python -u lpr_engine.py
pause
