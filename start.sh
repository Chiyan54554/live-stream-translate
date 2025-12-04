#!/bin/bash
# filepath: c:\Users\chiyan\Desktop\live-stream-translate\start.sh
# 🎙️ Live Stream Translator - 啟動腳本
# 用法: ./start.sh [選項]
#   選項:
#     --build    強制重建映像
#     --stop     停止所有服務
#     --logs     查看即時日誌
#     --clean    清除所有容器和映像

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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
        echo "請先安裝 Docker Desktop: https://www.docker.com/products/docker-desktop"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        echo -e "${RED}❌ 錯誤: Docker 未啟動${NC}"
        echo "請啟動 Docker Desktop"
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
        echo -e "${GREEN}✅ 偵測到 GPU: ${GPU_NAME} (${GPU_MEM} MB)${NC}"
        
        if [ "$GPU_MEM" -lt 8000 ]; then
            echo -e "${YELLOW}⚠️ 警告: VRAM 不足 8GB，建議使用較小的模型${NC}"
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
        echo "   請參考: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi
}

# 啟動服務
start_services() {
    echo ""
    echo -e "${YELLOW}🚀 啟動服務...${NC}"
    
    if [ "$1" == "--build" ]; then
        echo -e "${BLUE}📦 強制重建映像...${NC}"
        docker-compose up --build -d
    else
        docker-compose up -d
    fi
    
    echo ""
    echo -e "${GREEN}✅ 服務已啟動！${NC}"
    echo ""
    echo -e "${BLUE}📋 服務狀態:${NC}"
    docker-compose ps
    echo ""
    echo -e "${GREEN}🌐 開啟瀏覽器訪問: http://localhost:8080${NC}"
    echo ""
    echo -e "${YELLOW}💡 提示:${NC}"
    echo "   • 首次啟動需下載模型（約 10-15GB），請耐心等待"
    echo "   • 查看日誌: ./start.sh --logs"
    echo "   • 停止服務: ./start.sh --stop"
}

# 停止服務
stop_services() {
    echo -e "${YELLOW}🛑 停止服務...${NC}"
    docker-compose down
    echo -e "${GREEN}✅ 服務已停止${NC}"
}

# 查看日誌
show_logs() {
    echo -e "${YELLOW}📋 顯示即時日誌 (Ctrl+C 退出)...${NC}"
    docker-compose logs -f
}

# 清除所有
clean_all() {
    echo -e "${RED}⚠️ 警告: 這將刪除所有容器、映像和快取資料${NC}"
    read -p "確定要繼續嗎? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}🧹 清除所有資源...${NC}"
        docker-compose down -v --rmi all
        echo -e "${GREEN}✅ 清除完成${NC}"
    else
        echo "取消操作"
    fi
}

# 主程式
main() {
    show_banner
    
    case "$1" in
        --stop)
            stop_services
            ;;
        --logs)
            show_logs
            ;;
        --clean)
            clean_all
            ;;
        *)
            check_docker
            check_gpu
            check_nvidia_docker
            start_services "$1"
            ;;
    esac
}

main "$@"