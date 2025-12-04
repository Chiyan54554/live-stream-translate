#!/bin/bash
# filepath: c:\Users\chiyan\Desktop\live-stream-translate\start.sh
# 🎙️ Live Stream Translator - 啟動腳本
# 用法: ./start.sh [選項]
#   選項:
#     --build     強制重建映像
#     --stop      停止所有服務
#     --restart   重啟所有服務
#     --status    查看服務狀態
#     --logs      查看即時日誌
#     --logs-p    僅查看 processor 日誌
#     --logs-o    僅查看 ollama 日誌
#     --health    檢查服務健康狀態
#     --clean     清除所有容器和映像
#     --help      顯示幫助訊息

set -e

# 切換到腳本所在目錄
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 顯示標題
show_banner() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════╗"
    echo "║     🎙️ Live Stream Translator             ║"
    echo "║     日文直播即時翻譯系統                   ║"
    echo "╚═══════════════════════════════════════════╝"
    echo -e "${NC}"
}

# 檢查 Docker
check_docker() {
    echo -e "${YELLOW}🔍 檢查 Docker...${NC}"
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ 錯誤: Docker 未安裝${NC}"
        echo "   請先安裝 Docker: https://www.docker.com/products/docker-desktop"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        echo -e "${RED}❌ 錯誤: Docker 未啟動${NC}"
        echo "   請啟動 Docker Desktop 或 Docker 服務"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Docker 已就緒${NC}"
}

# 檢查 NVIDIA GPU
check_gpu() {
    echo -e "${YELLOW}🔍 檢查 NVIDIA GPU...${NC}"
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -n1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)
        
        if [ -n "$GPU_NAME" ]; then
            echo -e "${GREEN}✅ 偵測到 GPU: ${GPU_NAME} (${GPU_MEM} MB)${NC}"
            
            if [ "$GPU_MEM" -lt 8000 ] 2>/dev/null; then
                echo -e "${YELLOW}⚠️ 警告: VRAM 不足 8GB，建議使用較小的模型${NC}"
            fi
        else
            echo -e "${YELLOW}⚠️ 警告: 未偵測到 NVIDIA GPU，將使用 CPU 模式（較慢）${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️ 警告: 未偵測到 NVIDIA GPU，將使用 CPU 模式（較慢）${NC}"
    fi
}

# 檢查 NVIDIA Container Toolkit
check_nvidia_docker() {
    echo -e "${YELLOW}🔍 檢查 NVIDIA Container Toolkit...${NC}"
    
    if docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✅ NVIDIA Container Toolkit 已就緒${NC}"
    else
        echo -e "${YELLOW}⚠️ 警告: NVIDIA Container Toolkit 未安裝或未正確配置${NC}"
        echo "   如需 GPU 加速，請參考: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi
}

# 顯示服務狀態
show_status() {
    echo ""
    echo -e "${BLUE}📋 服務狀態:${NC}"
    docker-compose ps
}

# 啟動服務
start_services() {
    echo ""
    echo -e "${YELLOW}🚀 啟動服務...${NC}"
    
    if ! docker-compose up -d; then
        echo -e "${RED}❌ 啟動失敗，請檢查日誌: ./start.sh --logs${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}✅ 服務已啟動！${NC}"
    show_status
    echo ""
    echo -e "${GREEN}🌐 開啟瀏覽器訪問: ${BOLD}http://localhost:8080${NC}"
    echo ""
    echo -e "${YELLOW}💡 提示:${NC}"
    echo "   • 首次啟動需下載模型（約 10-15GB），請耐心等待"
    echo "   • 查看服務狀態: ./start.sh --status"
    echo "   • 查看即時日誌: ./start.sh --logs"
    echo "   • 停止服務: ./start.sh --stop"
    echo "   • 顯示幫助: ./start.sh --help"
}

# 強制重建並啟動
build_services() {
    echo ""
    echo -e "${BLUE}📦 強制重建映像...${NC}"
    
    if ! docker-compose build --no-cache; then
        echo -e "${RED}❌ 建置失敗${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${YELLOW}🚀 啟動服務...${NC}"
    docker-compose up -d
    
    echo ""
    echo -e "${GREEN}✅ 重建完成並已啟動${NC}"
    show_status
}

# 停止服務
stop_services() {
    echo -e "${YELLOW}🛑 停止服務...${NC}"
    docker-compose down
    echo ""
    echo -e "${GREEN}✅ 服務已停止${NC}"
}

# 重啟服務
restart_services() {
    echo -e "${YELLOW}🔄 重啟服務...${NC}"
    
    if ! docker-compose restart; then
        echo -e "${RED}❌ 重啟失敗${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}✅ 服務已重啟${NC}"
    show_status
}

# 查看日誌
show_logs() {
    echo -e "${YELLOW}📋 顯示即時日誌 (Ctrl+C 退出)...${NC}"
    docker-compose logs -f --tail=100
}

# 查看 Processor 日誌
show_logs_processor() {
    echo -e "${YELLOW}📋 顯示 Processor 日誌 (Ctrl+C 退出)...${NC}"
    docker-compose logs -f --tail=100 processor
}

# 查看 Ollama 日誌
show_logs_ollama() {
    echo -e "${YELLOW}📋 顯示 Ollama 日誌 (Ctrl+C 退出)...${NC}"
    docker-compose logs -f --tail=100 ollama
}

# 檢查健康狀態
check_health() {
    echo -e "${CYAN}🏥 檢查服務健康狀態...${NC}"
    echo ""
    
    # 檢查 Redis
    REDIS_HEALTH=$(docker inspect --format='{{.State.Health.Status}}' redis_pubsub 2>/dev/null || echo "not_running")
    case $REDIS_HEALTH in
        "healthy")
            echo -e "${GREEN}✅ Redis: 健康${NC}"
            ;;
        "not_running")
            echo -e "${RED}❌ Redis: 未運行${NC}"
            ;;
        *)
            echo -e "${YELLOW}⚠️ Redis: $REDIS_HEALTH${NC}"
            ;;
    esac
    
    # 檢查 Ollama
    OLLAMA_HEALTH=$(docker inspect --format='{{.State.Health.Status}}' ollama_llm 2>/dev/null || echo "not_running")
    case $OLLAMA_HEALTH in
        "healthy")
            echo -e "${GREEN}✅ Ollama: 健康 (模型已就緒)${NC}"
            ;;
        "starting")
            echo -e "${YELLOW}⏳ Ollama: 啟動中 (可能正在下載模型...)${NC}"
            ;;
        "not_running")
            echo -e "${RED}❌ Ollama: 未運行${NC}"
            ;;
        *)
            echo -e "${YELLOW}⚠️ Ollama: $OLLAMA_HEALTH${NC}"
            ;;
    esac
    
    # 檢查 Processor
    PROC_STATUS=$(docker inspect --format='{{.State.Status}}' python_processor 2>/dev/null || echo "not_running")
    case $PROC_STATUS in
        "running")
            echo -e "${GREEN}✅ Processor: 運行中${NC}"
            ;;
        "not_running")
            echo -e "${RED}❌ Processor: 未運行${NC}"
            ;;
        *)
            echo -e "${YELLOW}⚠️ Processor: $PROC_STATUS${NC}"
            ;;
    esac
    
    # 檢查 Server
    SERVER_STATUS=$(docker inspect --format='{{.State.Status}}' node_server 2>/dev/null || echo "not_running")
    case $SERVER_STATUS in
        "running")
            echo -e "${GREEN}✅ Server: 運行中${NC}"
            ;;
        "not_running")
            echo -e "${RED}❌ Server: 未運行${NC}"
            ;;
        *)
            echo -e "${YELLOW}⚠️ Server: $SERVER_STATUS${NC}"
            ;;
    esac
    
    echo ""
}

# 清除所有
clean_all() {
    echo -e "${RED}⚠️ 警告: 這將刪除所有容器、映像和快取資料${NC}"
    echo "   （不會刪除已下載的模型）"
    read -p "確定要繼續嗎? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo -e "${YELLOW}🧹 清除容器和映像...${NC}"
        docker-compose down -v --rmi all
        echo ""
        echo -e "${GREEN}✅ 清除完成${NC}"
        echo ""
        echo -e "${YELLOW}💡 如需刪除下載的模型，請手動執行:${NC}"
        echo "   docker volume rm live-stream-translate_ollama_models"
        echo "   docker volume rm live-stream-translate_huggingface_cache"
    else
        echo "取消操作"
    fi
}

# 顯示幫助
show_help() {
    echo -e "${BOLD}用法:${NC} ./start.sh [選項]"
    echo ""
    echo -e "${BOLD}選項:${NC}"
    echo "  （無參數）    啟動所有服務"
    echo "  --build      強制重建映像後啟動"
    echo "  --stop       停止所有服務"
    echo "  --restart    重啟所有服務"
    echo "  --status     查看服務狀態"
    echo "  --logs       查看所有服務即時日誌"
    echo "  --logs-p     僅查看 Processor 日誌"
    echo "  --logs-o     僅查看 Ollama 日誌"
    echo "  --health     檢查服務健康狀態"
    echo "  --clean      清除所有容器和映像"
    echo "  --help, -h   顯示此幫助訊息"
    echo ""
    echo -e "${BOLD}範例:${NC}"
    echo "  ./start.sh           啟動服務"
    echo "  ./start.sh --logs    查看日誌"
    echo "  ./start.sh --stop    停止服務"
    echo ""
}

# 主程式
main() {
    show_banner
    
    case "$1" in
        --stop)
            stop_services
            ;;
        --restart)
            restart_services
            ;;
        --status)
            show_status
            ;;
        --logs)
            show_logs
            ;;
        --logs-p)
            show_logs_processor
            ;;
        --logs-o)
            show_logs_ollama
            ;;
        --health)
            check_health
            ;;
        --clean)
            clean_all
            ;;
        --build)
            check_docker
            check_gpu
            check_nvidia_docker
            build_services
            ;;
        --help|-h)
            show_help
            ;;
        "")
            check_docker
            check_gpu
            check_nvidia_docker
            start_services
            ;;
        *)
            echo -e "${RED}❌ 未知選項: $1${NC}"
            echo "使用 ./start.sh --help 查看可用選項"
            exit 1
            ;;
    esac
}

main "$@"