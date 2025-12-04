#!/bin/bash
# Ollama 啟動腳本
# 🎯 自動下載指定模型（如果還沒下載的話）

# 要使用的模型（可透過環境變數覆蓋）
MODEL_NAME="${OLLAMA_MODEL:-qwen3:8b}"

echo "🚀 啟動 Ollama 服務..."

# 背景啟動 Ollama 服務
ollama serve &
OLLAMA_PID=$!

# 等待 Ollama 服務啟動
echo "⏳ 等待 Ollama 服務就緒..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama 服務已就緒"
        break
    fi
    sleep 1
done

# 檢查模型是否已存在
echo "🔍 檢查模型 $MODEL_NAME 是否已下載..."
if ollama list | grep -q "$(echo $MODEL_NAME | cut -d: -f1)"; then
    echo "✅ 模型 $MODEL_NAME 已存在，跳過下載"
else
    echo "📥 下載模型 $MODEL_NAME..."
    ollama pull "$MODEL_NAME"
    if [ $? -eq 0 ]; then
        echo "✅ 模型 $MODEL_NAME 下載完成"
    else
        echo "❌ 模型下載失敗"
    fi
fi

echo "🎯 Ollama 服務運行中..."
echo "📋 已安裝的模型:"
ollama list

# 等待 Ollama 進程
wait $OLLAMA_PID
