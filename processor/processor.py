import sys
import json
import time
from datetime import datetime, timezone, timedelta
from contextlib import redirect_stdout
import numpy as np
import redis
import os
import base64
import io
import re
from contextlib import redirect_stdout

# 引入 PyTorch 以檢查 CUDA 可用性，以及 Whisper 和 googletrans
try:
    import torch 
    import whisper 
    from deep_translator import GoogleTranslator
except ImportError:
    print("錯誤：運行此腳本需要安裝 'openai-whisper', 'torch', 'numpy', 'redis', 和 'deep_translator'。", file=sys.stderr, flush=True)
    sys.exit(1)


# --- 配置參數 ---
SAMPLE_RATE = 16000           # FFmpeg 應該輸出 16kHz
BYTES_PER_SAMPLE = 2          # 16-bit PCM
SOURCE_LANG_CODE = "ja"       # Whisper/Googletrans 源語言 (日文)
TARGET_LANG_CODE = "zh-TW"       # Whisper/Googletrans 目標語言 (中文)

# Redis 配置 (從環境變量讀取，供 Docker Compose 使用)
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))

AUDIO_CHANNEL = "audio_feed"           # 📢 訂閱音頻的頻道
TRANSLATION_CHANNEL = "translation_feed" # 👂 發佈翻譯結果的頻道

# 從環境變數讀取模型名稱，默認使用 'medium'
ASR_MODEL_NAME = os.getenv('ASR_MODEL_NAME', 'medium') 

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
    try:
        # 🌟 修正點 3：使用 Deep Translator 實例化，並預先指定源語言和目標語言
        translator = GoogleTranslator(source=SOURCE_LANG_CODE, target=TARGET_LANG_CODE)
        print("翻譯引擎 (Deep Translator/Google) 初始化成功。", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"翻譯引擎初始化失敗: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

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
        # ... (音訊處理部分保持不變) ...
        raw_audio_bytes = base64.b64decode(audio_data_b64)
        audio_array = np.frombuffer(raw_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_array).to(DEVICE)

        # 4. 使用 Whisper 轉錄 (直接傳遞 Tensor)
        # ====================================================================
        # 【修正】使用 redirect_stdout 重新導向輸出到空設備 (os.devnull)，以消除進度條。
        # ====================================================================
        # 3. 【原始 Whisper 轉錄】
        with io.StringIO() as f, redirect_stdout(f):
            result = asr_model.transcribe(
                audio_tensor,
                language=SOURCE_LANG_CODE,
                fp16=True if DEVICE == "cuda" else False,
                
                beam_size=5,     # 啟用 Beam Search，提升準確度（建議值為 5）
                patience=1.0,    # 鼓勵模型等待更完整的語句結束

                # 保持 Initial Prompt 協助抗幻覺 (引導對話)
                initial_prompt="会話中です。",

                # ==========================================================
                # 核心修正：應用最完整的結束語 Token 抑制列表
                # 專門針對: 「最後までご視聴ありがとうございました」
                suppress_tokens=[-1, 50363, 50362, 50361, 50360, 50359, 
                                 32205, 21840, 1023, 1970, 310, 28, 13], 
                
                # 保持靜音門檻 (抑制 [音訊標籤])
                no_speech_threshold=0.7, 
                logprob_threshold=-0.4 
                # ==========================================================
            )
        
        return result["text"].strip()

    except Exception as e:
        # ⚠️ 這裡使用 sys.stderr 輸出錯誤，不會被重定向靜音
        print(f"Whisper ASR 處理失敗: {e}", file=sys.stderr, flush=True)
        return "Whisper_ASR_FAILURE"

def google_mt(text: str) -> str:
    """
    使用 Deep Translator 呼叫 Google 翻譯進行機器翻譯。
    """
    if not text or translator is None:
        return ""
    try:
        # 🌟 修正點 4：呼叫實例的 translate 方法
        translation = translator.translate(text)
        # Deep Translator 返回的是純文字，無需 .text
        return translation 
    except Exception as e:
        print(f"翻譯失敗 (Deep Translator error): {e}", file=sys.stderr, flush=True)
        return f"MT_FAILURE: {text}"

# ----------------------------------------------------
# 核心處理函數：從 Redis 接收數據，處理，再發佈到 Redis
# ----------------------------------------------------

def process_audio_chunk(audio_data_b64, r):
    # 執行實際的 Whisper ASR
    transcribed_text = whisper_asr(audio_data_b64)

    # 【關鍵修改：檢查轉錄文本】
    # 如果轉錄文本為空字串，則直接返回，不進行翻譯和發佈
    if not transcribed_text:
        return
    
    text = transcribed_text.strip()

    # === 【新增修正：強制日文/常用字符過濾】 ===
    # 目的：移除韓文、俄文、德文 (非拉丁字母) 等亂碼，只保留日文、英文、數字和常用符號。

    # 允許的字符範圍 (日文假名/漢字/平假名/片假名、常用標點、數字、基本拉丁字母)
    # \u3040-\u309F: 平假名; \u30A0-\u30FF: 片假名; \u4E00-\u9FFF: 漢字; 
    # \uFF00-\uFFEF: 全形符號; \u0020-\u007E: 基本拉丁字母 (英文, 數字, 標點)
    japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uFF00-\uFFEF\u0020-\u007E]+')

    # 僅保留匹配日文/英文/數字/符號的連續區塊
    filtered_segments = japanese_pattern.findall(text)

    # 將所有通過過濾的區塊重新連接成單一句子
    cleaned_text = "".join(filtered_segments).strip()

    # 更新用於後續流程的文本
    transcribed_text = cleaned_text 
    text = cleaned_text

    if not text:
        print("警告: ASR 文本經過字符過濾後變為空字串，已跳過。", file=sys.stderr, flush=True)
        return

    # -----------------------------------------------------------------
    # 【新增修正：過濾重複的結束語】
    # 目的：防止 Whisper 在靜音或低音量時幻覺出結束語並重複輸出。
    # -----------------------------------------------------------------
    unwanted_phrases = [
        "[音声なし]",
        "ご視聴ありがとうございました。",
        "ご視聴ありがとうございました",
        "最後までご視聴ありがとうございました。",
        "最後までご視聴ありがとうございました",
        "ご視聴ありがとうございました。", # 確保包含各種標點符號的變體
        "[音声なし]",  # 靜音標記
        "(幕の開ける音)",
        "(拍手)",
        "(笑い)",
        "(ため息)",
        "19}",         # 您的範例中的極短噪音
        "19",          # 預防沒有大括號
        "}",
    ]

    # 標準化處理：移除日文句號「。」和頓號「、」，並移除多餘空格
    normalized_text = transcribed_text.strip().replace("。", "").replace("、", "") 
    
    # 檢查轉錄文本是否包含在不想發佈的短語列表中
    is_unwanted = False
    
    # 檢查是否包含在不想要的標記中
    if any(marker in text for marker in unwanted_phrases):
        is_unwanted = True
    
    # 檢查是否為極短且無意義的文字 (例如，少於 3 個非數字、非符號的字符)
    # 這裡我們只檢查長度，確保不發佈單個數字或符號
    if len(text) < 3 and not any(c.isalpha() for c in text):
        is_unwanted = True

    if is_unwanted:
        print(f"警告: 偵測到並過濾了事件標記或噪音文本: {transcribed_text}", file=sys.stderr, flush=True)
        return # 偵測到噪音/標記，跳過翻譯和發佈
    
    # 如果轉錄文本為空字串，則直接返回
    if not text:
        return
    
    # if re.search(r'[a-zA-Z]', text) or re.search(r'[а-яА-Я]', text): 
    #     print(f"警告: 偵測到外文或亂碼（ASR 幻覺），已過濾: {text}", file=sys.stderr, flush=True)
    #     return # 偵測到外文/亂碼，跳過翻譯和發佈
    
    # 執行實際翻譯
    translated_text = google_mt(transcribed_text)
    
    duration_seconds = 0.128 

    # 🌟 關鍵修正：確保時間戳記為當地時間 (UTC+8 / 台北時間)
    # 建立時區偏移量 (台灣為 UTC+8)
    tz = timezone(timedelta(hours=8))
    # 取得當前 UTC 時間並轉換為指定的時區
    current_time_cst = datetime.now(tz)
    # 格式化輸出
    timestamp = current_time_cst.strftime("%H:%M:%S")
    
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