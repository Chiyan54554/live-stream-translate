@echo off
REM filepath: c:\Users\chiyan\Desktop\live-stream-translate\start.bat
REM 🎙️ Live Stream Translator - Windows 啟動腳本

setlocal enabledelayedexpansion

echo.
echo ╔═══════════════════════════════════════════╗
echo ║     🎙️ Live Stream Translator             ║
echo ║     日文直播即時翻譯系統                   ║
echo ╚═══════════════════════════════════════════╝
echo.

REM 檢查參數
if "%1"=="--stop" goto :stop
if "%1"=="--logs" goto :logs
if "%1"=="--clean" goto :clean
if "%1"=="--build" goto :build

:start
echo 🔍 檢查 Docker...
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ 錯誤: Docker 未啟動
    echo 請先啟動 Docker Desktop
    pause
    exit /b 1
)
echo ✅ Docker 已就緒

echo.
echo 🔍 檢查 NVIDIA GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ⚠️ 警告: 未偵測到 NVIDIA GPU
) else (
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2^>nul') do (
        echo ✅ 偵測到 GPU: %%i
    )
)

echo.
echo 🚀 啟動服務...
docker-compose up -d

echo.
echo ✅ 服務已啟動！
echo.
echo 📋 服務狀態:
docker-compose ps
echo.
echo 🌐 開啟瀏覽器訪問: http://localhost:8080
echo.
echo 💡 提示:
echo    • 首次啟動需下載模型（約 10-15GB），請耐心等待
echo    • 查看日誌: start.bat --logs
echo    • 停止服務: start.bat --stop
echo.
pause
goto :eof

:build
echo 📦 強制重建映像...
docker-compose up --build -d
echo ✅ 完成
pause
goto :eof

:stop
echo 🛑 停止服務...
docker-compose down
echo ✅ 服務已停止
pause
goto :eof

:logs
echo 📋 顯示即時日誌 (Ctrl+C 退出)...
docker-compose logs -f
goto :eof

:clean
echo ⚠️ 警告: 這將刪除所有容器、映像和快取資料
set /p confirm=確定要繼續嗎? (y/N): 
if /i "%confirm%"=="y" (
    echo 🧹 清除所有資源...
    docker-compose down -v --rmi all
    echo ✅ 清除完成
) else (
    echo 取消操作
)
pause
goto :eof