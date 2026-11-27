@echo off
chcp 65001 >nul
title 植物叶片病害智能诊断系统
cd /d "%~dp0"

echo ============================================
echo   植物叶片病害智能诊断系统
echo   正在启动，请稍候...
echo ============================================
echo.

python "%~dp0app\web_agent.py"

if errorlevel 1 (
    echo.
    echo 启动失败。请确认已安装依赖：
    echo   pip install -r requirements.txt
    pause
)

