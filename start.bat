@echo off
REM filepath: c:\Users\chiyan\Desktop\live-stream-translate\start.bat
REM 🎙️ Live Stream Translator - Windows 啟動腳本
REM 用法: start.bat [選項]
REM   選項:
 REM     --build    強制重建映像
REM     --stop     停止所有服務
REM     --restart  重啟所有服務
REM     --status   查看服務狀態
REM     --logs     查看即時日誌
REM     --logs-p   僅查看 processor 日誌
REM     --logs-o   僅查看 ollama 日誌
REM     --health   檢查服務健康狀態
REM     --clean    清除所有容器和映像
REM     --help     顯示幫助訊息

setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

REM 切換到腳本所在目錄
cd /d "%~dp0"

call :show_banner

REM 檢查參數
if "%1"=="" goto :start
if "%1"=="--stop" goto :stop
if "%1"=="--restart" goto :restart
if "%1"=="--status" goto :status
if "%1"=="--logs" goto :logs
if "%1"=="--logs-p" goto :logs_processor
if "%1"=="--logs-o" goto :logs_ollama
if "%1"=="--health" goto :health
if "%1"=="--clean" goto :clean
if "%1"=="--build" goto :build
if "%1"=="--help" goto :help
if "%1"=="-h" goto :help

echo ❌ 未知選項: %1
echo 使用 start.bat --help 查看可用選項
pause
goto :eof

:show_banner
echo.
echo ╔═══════════════════════════════════════════╗
echo ║     🎙️ Live Stream Translator             ║
echo ║     日文直播即時翻譯系統                   ║
echo ╚═══════════════════════════════════════════╝
echo.
goto :eof

:check_docker
echo 🔍 檢查 Docker...
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ 錯誤: Docker 未啟動
    echo    請先啟動 Docker Desktop
    exit /b 1
)
echo ✅ Docker 已就緒
goto :eof

:check_gpu
echo.
echo 🔍 檢查 NVIDIA GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ⚠️ 警告: 未偵測到 NVIDIA GPU，將使用 CPU 模式（較慢）
) else (
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2^>nul') do (
        set "GPU_NAME=%%i"
    )
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2^>nul') do (
        set "GPU_MEM=%%i"
    )
    echo ✅ 偵測到 GPU: !GPU_NAME! ^(!GPU_MEM! MB^)
    if !GPU_MEM! LSS 8000 (
        echo ⚠️ 警告: VRAM 不足 8GB，建議使用較小的模型
    )
)
goto :eof

:check_nvidia_docker
echo.
echo 🔍 檢查 NVIDIA Container Toolkit...
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ⚠️ 警告: NVIDIA Container Toolkit 未安裝或未正確配置
    echo    如需 GPU 加速，請參考: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
) else (
    echo ✅ NVIDIA Container Toolkit 已就緒
)
goto :eof

:start
call :check_docker
if errorlevel 1 (
    pause
    goto :eof
)
call :check_gpu
call :check_nvidia_docker

echo.
echo 🚀 啟動服務...
docker-compose up -d
if errorlevel 1 (
    echo ❌ 啟動失敗，請檢查日誌: start.bat --logs
    pause
    goto :eof
)

echo.
echo ✅ 服務已啟動！
call :show_status
echo.
echo 🌐 開啟瀏覽器訪問: http://localhost:8080
echo.
echo 💡 提示:
echo    • 首次啟動需下載模型（約 10-15GB），請耐心等待
echo    • 查看服務狀態: start.bat --status
echo    • 查看即時日誌: start.bat --logs
echo    • 停止服務: start.bat --stop
echo    • 顯示幫助: start.bat --help
echo.
pause
goto :eof

:build
call :check_docker
if errorlevel 1 (
    pause
    goto :eof
)
echo.
echo 📦 強制重建映像...
docker-compose build --no-cache
if errorlevel 1 (
    echo ❌ 建置失敗
    pause
    goto :eof
)
echo.
echo 🚀 啟動服務...
docker-compose up -d
echo.
echo ✅ 重建完成並已啟動
call :show_status
pause
goto :eof

:stop
echo 🛑 停止服務...
docker-compose down
echo.
echo ✅ 服務已停止
pause
goto :eof

:restart
echo 🔄 重啟服務...
docker-compose restart
if errorlevel 1 (
    echo ❌ 重啟失敗
    pause
    goto :eof
)
echo.
echo ✅ 服務已重啟
call :show_status
pause
goto :eof

:status
call :show_status
pause
goto :eof

:show_status
echo.
echo 📋 服務狀態:
docker-compose ps
goto :eof

:logs
echo 📋 顯示即時日誌 (Ctrl+C 退出)...
docker-compose logs -f --tail=100
goto :eof

:logs_processor
echo 📋 顯示 Processor 日誌 (Ctrl+C 退出)...
docker-compose logs -f --tail=100 processor
goto :eof

:logs_ollama
echo 📋 顯示 Ollama 日誌 (Ctrl+C 退出)...
docker-compose logs -f --tail=100 ollama
goto :eof

:health
echo 🏥 檢查服務健康狀態...
echo.

REM 檢查 Redis
for /f "tokens=*" %%i in ('docker inspect --format="{{.State.Health.Status}}" redis_pubsub 2^>nul') do set "REDIS_HEALTH=%%i"
if "!REDIS_HEALTH!"=="healthy" (
    echo ✅ Redis: 健康
) else if "!REDIS_HEALTH!"=="" (
    echo ❌ Redis: 未運行
) else (
    echo ⚠️ Redis: !REDIS_HEALTH!
)

REM 檢查 Ollama
for /f "tokens=*" %%i in ('docker inspect --format="{{.State.Health.Status}}" ollama_llm 2^>nul') do set "OLLAMA_HEALTH=%%i"
if "!OLLAMA_HEALTH!"=="healthy" (
    echo ✅ Ollama: 健康 ^(模型已就緒^)
) else if "!OLLAMA_HEALTH!"=="starting" (
    echo ⏳ Ollama: 啟動中 ^(可能正在下載模型...^)
) else if "!OLLAMA_HEALTH!"=="" (
    echo ❌ Ollama: 未運行
) else (
    echo ⚠️ Ollama: !OLLAMA_HEALTH!
)

REM 檢查 Processor
for /f "tokens=*" %%i in ('docker inspect --format="{{.State.Status}}" python_processor 2^>nul') do set "PROC_STATUS=%%i"
if "!PROC_STATUS!"=="running" (
    echo ✅ Processor: 運行中
) else if "!PROC_STATUS!"=="" (
    echo ❌ Processor: 未運行
) else (
    echo ⚠️ Processor: !PROC_STATUS!
)

REM 檢查 Server
for /f "tokens=*" %%i in ('docker inspect --format="{{.State.Status}}" node_server 2^>nul') do set "SERVER_STATUS=%%i"
if "!SERVER_STATUS!"=="running" (
    echo ✅ Server: 運行中
) else if "!SERVER_STATUS!"=="" (
    echo ❌ Server: 未運行
) else (
    echo ⚠️ Server: !SERVER_STATUS!
)

echo.
pause
goto :eof

:clean
echo ⚠️ 警告: 這將刪除所有容器、映像和快取資料
echo    （不會刪除已下載的模型）
set /p confirm=確定要繼續嗎? (y/N): 
if /i "%confirm%"=="y" (
    echo.
    echo 🧹 清除容器和映像...
    docker-compose down -v --rmi all
    echo.
    echo ✅ 清除完成
    echo.
    echo 💡 如需刪除下載的模型，請手動執行:
    echo    docker volume rm live-stream-translate_ollama_models
    echo    docker volume rm live-stream-translate_huggingface_cache
) else (
    echo 取消操作
)
pause
goto :eof

:help
echo 用法: start.bat [選項]
echo.
echo 選項:
echo   （無參數）   啟動所有服務
echo   --build     強制重建映像後啟動
echo   --stop      停止所有服務
echo   --restart   重啟所有服務
echo   --status    查看服務狀態
echo   --logs      查看所有服務即時日誌
echo   --logs-p    僅查看 Processor 日誌
echo   --logs-o    僅查看 Ollama 日誌
echo   --health    檢查服務健康狀態
echo   --clean     清除所有容器和映像
echo   --help, -h  顯示此幫助訊息
echo.
echo 範例:
echo   start.bat           啟動服務
echo   start.bat --logs    查看日誌
echo   start.bat --stop    停止服務
echo.
pause
goto :eof