import sys
import json
import time
import numpy as np
import redis
import os
import base64
import io

# 引入 PyTorch 以檢查 CUDA 可用性，以及 Whisper 和 googletrans
try:
    import torch 
    import whisper 
    from googletrans import Translator
except ImportError:
    print("錯誤：運行此腳本需要安裝 'openai-whisper', 'torch', 'numpy', 'redis', 和 'googletrans'。", file=sys.stderr, flush=True)
    sys.exit(1)


# --- 配置參數 ---
SAMPLE_RATE = 16000           # FFmpeg 應該輸出 16kHz
BYTES_PER_SAMPLE = 2          # 16-bit PCM
SOURCE_LANG_CODE = "zh"       # Whisper/Googletrans 源語言 (中文)
TARGET_LANG_CODE = "ja"       # Whisper/Googletrans 目標語言 (日文)

# 從環境變數讀取模型名稱，默認使用 'tiny'
ASR_MODEL_NAME = os.getenv('ASR_MODEL_NAME', 'tiny') 

# 確定要使用的設備：如果 CUDA 可用，則使用 GPU，否則使用 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 全局資源
asr_model = None
translator = None

# ----------------------------------------------------
# 資源初始化與 ASR/MT 函數
# ----------------------------------------------------

def init_global_resources():
    """載入 Whisper 模型和初始化翻譯器。"""
    global asr_model, translator
    
    print(f"Whisper 將使用的設備: {DEVICE}", file=sys.stderr, flush=True)

    # 1. 初始化翻譯器
    translator = Translator()

    # 2. 載入 Whisper 模型
    try:
        print(f"正在載入 Whisper ASR 模型: {ASR_MODEL_NAME}...", file=sys.stderr, flush=True)
        
        # 關鍵修改: 將模型載入到確定的 DEVICE 上
        asr_model = whisper.load_model(ASR_MODEL_NAME, device=DEVICE)
        
        print(f"Whisper 模型載入成功並已移動到 {DEVICE} 上。", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"致命錯誤：Whisper 模型載入失敗，請檢查 PyTorch 和 GPU 依賴項: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


def whisper_asr(audio_data_b64: str) -> str:
    """
    使用 Whisper 模型將 Base64 音訊數據轉錄為文本。
    """
    if asr_model is None:
        return "錯誤: Whisper 模型尚未載入。"

    try:
        # 1. 解碼 Base64 數據為原始 PCM 數據 (bytes)
        raw_audio_bytes = base64.b64decode(audio_data_b64)
        
        # 2. 將原始 16-bit PCM 數據轉換為 Whisper 所需的 float32 Numpy 陣列
        audio_array = np.frombuffer(raw_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 3. 關鍵修改: 將 Numpy 陣列轉換為 PyTorch Tensor 並移動到正確的 DEVICE
        audio_tensor = torch.from_numpy(audio_array).to(DEVICE)

        # 4. 使用 Whisper 轉錄 (直接傳遞 Tensor)
        result = asr_model.transcribe(
            audio_tensor, # 使用 PyTorch Tensor
            language=SOURCE_LANG_CODE,
            # 啟用 fp16 加速 GPU 上的推論
            fp16=True if DEVICE == "cuda" else False, 
            verbose=False 
        )
        return result["text"].strip()

    except Exception as e:
        print(f"Whisper ASR 處理失敗: {e}", file=sys.stderr, flush=True)
        return "Whisper_ASR_FAILURE"


def google_mt(text: str) -> str:
    """
    使用 googletrans 進行機器翻譯。
    """
    if not text or translator is None:
        return ""
    try:
        translation = translator.translate(
            text, 
            src=SOURCE_LANG_CODE, 
            dest=TARGET_LANG_CODE
        )
        return translation.text
    except Exception as e:
        print(f"翻譯失敗 (googletrans error): {e}", file=sys.stderr, flush=True)
        return f"MT_FAILURE: {text}"

# ----------------------------------------------------
# 核心處理函數：從 Redis 接收數據，處理，再發佈到 Redis
# ----------------------------------------------------

def process_audio_chunk(audio_data_b64, r):
    # 執行實際的 Whisper ASR
    transcribed_text = whisper_asr(audio_data_b64)
    
    # 執行實際翻譯
    translated_text = google_mt(transcribed_text)
    
    duration_seconds = 0.128 
    timestamp = time.strftime("%H:%M:%S")
    
    result = {
        "timestamp": timestamp,
        "source_lang": SOURCE_LANG_CODE,
        "target_lang": TARGET_LANG_CODE,
        "duration_s": f"{duration_seconds:.3f}",
        "transcription": transcribed_text,
        "translation": translated_text
    }
    
    try:
        json_output = json.dumps(result, ensure_ascii=False)
        r.publish(TRANSLATION_CHANNEL, json_output) # 發佈到翻譯結果頻道
    except Exception as e:
        print(f"致命錯誤：Python 發佈翻譯結果到 Redis 失敗: {e}", file=sys.stderr, flush=True)


def main():
    """
    主循環：訂閱 Redis 音頻頻道，並初始化全局資源。
    """
    # 載入 Whisper 模型
    init_global_resources() 

    # 1. 初始化 Redis 客戶端
    try:
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
            audio_chunk_b64 = message['data'].decode('utf-8') 
            process_audio_chunk(audio_chunk_b64, r)
        elif message['type'] == 'subscribe':
             print(f"已成功訂閱 {message['channel'].decode('utf-8')}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()