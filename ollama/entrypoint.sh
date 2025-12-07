#!/bin/bash
# Ollama 啟動腳本
# 🚀 優化版：服務和模型載入並行

# 要使用的模型（可透過環境變數覆蓋）
MODEL_NAME="${OLLAMA_MODEL:-qwen3:8b}"
WARMUP_ENABLED="${OLLAMA_WARMUP:-1}"
WARMUP_PROMPT="${OLLAMA_WARMUP_PROMPT:-你好}"
WARMUP_MAX_TOKENS="${OLLAMA_WARMUP_MAX_TOKENS:-8}"
WARMUP_RETRIES="${OLLAMA_WARMUP_RETRIES:-6}"
WARMUP_DELAY="${OLLAMA_WARMUP_DELAY:-5}"

echo "🚀 啟動 Ollama 服務 (安靜模式，詳細日誌寫入 /var/log/ollama.log)..."

# 背景啟動 Ollama 服務（stdout/stderr 轉存到檔案，減少終端噪音）
mkdir -p /var/log
ollama serve > /var/log/ollama.log 2>&1 &
OLLAMA_PID=$!

# 簡短等待服務就緒 (3秒)
sleep 3

# 背景執行模型檢查和載入（不阻塞健康檢查）
(
    # 等待 API 就緒
    for i in {1..15}; do
        if ollama list > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    
    # 檢查模型是否已存在
    echo "🔍 檢查模型 $MODEL_NAME..."
    if ollama list 2>/dev/null | grep -q "$(echo $MODEL_NAME | cut -d: -f1)"; then
        echo "✅ 模型 $MODEL_NAME 已存在"
    else
        echo "📥 下載模型 $MODEL_NAME..."
        ollama pull "$MODEL_NAME"
        [ $? -eq 0 ] && echo "✅ 模型下載完成" || echo "❌ 模型下載失敗"
    fi

    if [ "$WARMUP_ENABLED" = "1" ]; then
        echo "🔥 預熱模型 $MODEL_NAME（載入權重以降低首次延遲）..."
        success=0
        for i in $(seq 1 "$WARMUP_RETRIES"); do
            ollama run "$MODEL_NAME" -p "$WARMUP_PROMPT" --keepalive 5m --quiet --options "{\"num_predict\":$WARMUP_MAX_TOKENS}" >/dev/null 2>&1 && success=1 && break
            echo "⚠️ 預熱嘗試 $i 失敗，等待 ${WARMUP_DELAY}s 重試..."
            sleep "$WARMUP_DELAY"
        done
        if [ "$success" -eq 1 ]; then
            echo "✅ 預熱完成"
        else
            echo "⚠️ 預熱失敗（稍後請求時再載入）"
        fi
    fi
    
    echo "📋 已安裝模型: $(ollama list 2>/dev/null | grep -v NAME | awk '{print $1}' | tr '\n' ' ')"
) &

echo "🎯 Ollama 服務運行中 (模型背景載入)..."

# 等待 Ollama 進程
wait $OLLAMA_PID
