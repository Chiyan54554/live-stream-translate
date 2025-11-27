import sys
import json
import time
import numpy as np
import redis
import os
import base64

# --- 配置參數 ---
SAMPLE_RATE = 16000 
BYTES_PER_SAMPLE = 2 
SOURCE_LANG = "中文"
TARGET_LANG = "英文"

# Redis 配置 (從環境變量讀取，供 Docker Compose 使用)
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
AUDIO_CHANNEL = "audio_feed"           # 📢 訂閱音頻的頻道
TRANSLATION_CHANNEL = "translation_feed" # 👂 發佈翻譯結果的頻道

# ----------------------------------------------------
# ⚠️ 實際項目中的 ASR 和 MT 替換點
# ----------------------------------------------------

def mock_asr(audio_data_b64):
    current_time_ms = int(time.time() * 1000)
    # 模擬根據當前時間生成轉錄文本
    return f"直播語音片段：歡迎收看我們的實時翻譯示範，時間戳 {current_time_ms}"

def mock_translate(text):
    if not text:
        return ""
    # 從中文片段中提取時間戳，模擬翻譯
    return f"Live voice snippet: Welcome to watch our real-time translation demonstration, timestamp {text.split(' ')[-1]}"

# ----------------------------------------------------
# 核心處理函數：從 Redis 接收數據，處理，再發佈到 Redis
# ----------------------------------------------------

def process_audio_chunk(audio_data_b64, r):
    """
    接收 Base64 音頻數據，執行 ASR 和 MT，並發佈 JSON 結果到 Redis。
    """
    transcribed_text = mock_asr(audio_data_b64)
    translated_text = mock_translate(transcribed_text)
    
    duration_seconds = 0.128 # Mock duration
    timestamp = time.strftime("%H:%M:%S")
    
    result = {
        "timestamp": timestamp,
        "source_lang": SOURCE_LANG,
        "target_lang": TARGET_LANG,
        "duration_s": f"{duration_seconds:.3f}",
        "transcription": transcribed_text,
        "translation": translated_text
    }
    
    try:
        # ensure_ascii=False 確保中文能被正確編碼，避免亂碼
        json_output = json.dumps(result, ensure_ascii=False)
        r.publish(TRANSLATION_CHANNEL, json_output) # 發佈到翻譯結果頻道
    except Exception as e:
        print(f"致命錯誤：Python 發佈翻譯結果到 Redis 失敗: {e}", file=sys.stderr, flush=True)


def main():
    """
    主循環：訂閱 Redis 音頻頻道。
    """
    # 1. 初始化 Redis 客戶端
    try:
        # 容器內連接
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0) 
        r.ping()
        print(f"Python 成功連接到 Redis ({REDIS_HOST}:{REDIS_PORT})。", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"致命錯誤：Python 無法連接到 Redis: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    # 2. 設置 Redis 訂閱
    p = r.pubsub()
    p.subscribe(AUDIO_CHANNEL)
    print(f"Python 成功訂閱 Redis 頻道: {AUDIO_CHANNEL}。", file=sys.stderr, flush=True)

    # 3. 主循環：從 Redis 訂閱中讀取音頻數據
    for message in p.listen():
        if message['type'] == 'message':
            # message['data'] 是 Node.js 發佈的 Base64 音頻數據 (bytes)
            # 必須解碼成字串，Python 才能處理 (雖然是 Mock)
            audio_chunk_b64 = message['data'].decode('utf-8') 
            process_audio_chunk(audio_chunk_b64, r)
        elif message['type'] == 'subscribe':
             # 成功訂閱的通知
             print(f"已成功訂閱 {message['channel'].decode('utf-8')}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()