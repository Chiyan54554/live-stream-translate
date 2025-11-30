import sys
import json
import time
from datetime import datetime, timezone, timedelta
import numpy as np
import redis
import os
import base64
import re
from concurrent.futures import ThreadPoolExecutor

# 🌟 設定環境變數
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 🌟 設定 cuDNN 路徑
def setup_cudnn_path():
    possible_paths = [
        "/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib",
        "/opt/conda/lib/python3.12/site-packages/nvidia/cudnn/lib",
        "/usr/local/cuda/lib64",
    ]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    new_paths = [p for p in possible_paths if os.path.exists(p)]
    if new_paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths) + ":" + existing

setup_cudnn_path()

# 引入依賴
try:
    import torch
    print(f"PyTorch 版本: {torch.__version__}", file=sys.stderr, flush=True)
    print(f"CUDA 可用: {torch.cuda.is_available()}", file=sys.stderr, flush=True)
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}", file=sys.stderr, flush=True)
        print(f"GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr, flush=True)
    
    from faster_whisper import WhisperModel
    from deep_translator import GoogleTranslator
except ImportError as e:
    print(f"錯誤：缺少依賴套件: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

# --- 配置參數 ---
SAMPLE_RATE = 16000           # FFmpeg 應該輸出 16kHz
BYTES_PER_SAMPLE = 2          # 16-bit PCM
SOURCE_LANG_CODE = "ja"       # 源語言 (日文)
TARGET_LANG_CODE = "zh-TW"    # 目標語言 (中文)

# 🌟 優化：調整緩衝配置以平衡準確率和速度
BUFFER_DURATION_S = 3.0       # 縮短至 3 秒，加快回應速度
OVERLAP_DURATION_S = 0.5      # 保留 0.5 秒重疊，避免語句切斷
MIN_AUDIO_ENERGY = 0.005      # 降低能量閾值，捕捉更多語音

# Redis 配置
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))

AUDIO_CHANNEL = "audio_feed"
TRANSLATION_CHANNEL = "translation_feed"

# 🌟 faster-whisper 支援的模型: tiny, base, small, medium, large-v2, large-v3
ASR_MODEL_NAME = os.getenv('ASR_MODEL_NAME', 'medium')

# 修正：faster-whisper 使用的快取目錄
MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', '/root/.cache/huggingface/hub')

# 🌟 修改：預設嘗試 CUDA，但準備降級
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# 全局資源
asr_model = None
translator = None
audio_buffer = b''
overlap_buffer = b''  # 🌟 新增：重疊緩衝區

# 🌟 新增：執行緒池用於並行翻譯
executor = ThreadPoolExecutor(max_workers=2)

# 🌟 新增：上一次轉錄結果，用於去重
last_transcription = ""

# ----------------------------------------------------
# 資源初始化
# ----------------------------------------------------

def init_global_resources():
    """載入 faster-whisper 模型和初始化翻譯器。"""
    global asr_model, translator, DEVICE, COMPUTE_TYPE
    
    print(f"="*50, file=sys.stderr, flush=True)
    print(f"初始設備: {DEVICE}, 計算類型: {COMPUTE_TYPE}", file=sys.stderr, flush=True)
    print(f"模型名稱: {ASR_MODEL_NAME}", file=sys.stderr, flush=True)
    print(f"快取目錄: {MODEL_CACHE_DIR}", file=sys.stderr, flush=True)
    print(f"="*50, file=sys.stderr, flush=True)

    # 1. 初始化翻譯器
    try:
        translator = GoogleTranslator(source=SOURCE_LANG_CODE, target=TARGET_LANG_CODE)
        print("✅ 翻譯引擎初始化成功。", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"❌ 翻譯引擎初始化失敗: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    # 2. 載入模型 (帶有自動降級)
    def try_load_model(device, compute_type):
        try:
            print(f"🔄 嘗試 {device}/{compute_type}...", file=sys.stderr, flush=True)
            model = WhisperModel(
                ASR_MODEL_NAME,
                device=device,
                compute_type=compute_type,
                download_root=MODEL_CACHE_DIR,
                local_files_only=False,
                cpu_threads=4,
                num_workers=1,
            )
            # 🌟 測試模型是否真的能運作
            test_audio = np.zeros(16000, dtype=np.float32)
            list(model.transcribe(test_audio, language="ja"))
            return model
        except Exception as e:
            print(f"⚠️ {device}/{compute_type} 失敗: {e}", file=sys.stderr, flush=True)
            return None

    print(f"正在載入 faster-whisper 模型...", file=sys.stderr, flush=True)
    start_time = time.time()
    
    # 🌟 嘗試順序
    attempts = [
        ("cuda", "float16"),
        ("cuda", "int8_float16"),
        ("cuda", "int8"),
        ("cpu", "int8"),
        ("cpu", "float32"),
    ]
    
    for device, compute_type in attempts:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        asr_model = try_load_model(device, compute_type)
        if asr_model is not None:
            DEVICE = device
            COMPUTE_TYPE = compute_type
            break
    
    if asr_model is None:
        print("❌ 所有載入嘗試均失敗", file=sys.stderr, flush=True)
        sys.exit(1)
    
    elapsed = time.time() - start_time
    print(f"✅ 模型載入成功！設備: {DEVICE}, 類型: {COMPUTE_TYPE}, 耗時: {elapsed:.2f}s", file=sys.stderr, flush=True)

def check_voice_activity(audio_array: np.ndarray) -> bool:
    """簡單的語音活動偵測 (VAD)。"""
    rms_energy = np.sqrt(np.mean(audio_array ** 2))
    return rms_energy > MIN_AUDIO_ENERGY

def whisper_asr(audio_array: np.ndarray) -> str:
    """使用 faster-whisper 進行語音辨識。"""
    if asr_model is None:
        return ""

    try:
        # 🌟 檢查語音活動
        if not check_voice_activity(audio_array):
            return ""
        
        # 🌟 faster-whisper 直接接受 numpy array
        segments, info = asr_model.transcribe(
            audio_array,
            language=SOURCE_LANG_CODE,
            beam_size=5,
            best_of=5,
            patience=1.5,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=False,  # 🌟 關閉以避免錯誤累積
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            initial_prompt="これは日本語の会話です。",
            vad_filter=True,  # 🌟 啟用內建 VAD 過濾
            vad_parameters=dict(
                min_silence_duration_ms=500,  # 最小靜音時長
                speech_pad_ms=200,            # 語音前後填充
            ),
        )
        
        # 收集所有片段的文字
        text_parts = [segment.text for segment in segments]
        return "".join(text_parts).strip()

    except Exception as e:
        print(f"ASR 處理失敗: {e}", file=sys.stderr, flush=True)
        return ""

def google_mt(text: str) -> str:
    """使用 Deep Translator 進行翻譯。"""
    if not text or translator is None:
        return ""
    try:
        return translator.translate(text)
    except Exception as e:
        print(f"翻譯失敗: {e}", file=sys.stderr, flush=True)
        return f"MT_FAILURE: {text}"

def filter_text(text: str) -> str:
    """過濾無效文字。"""
    if not text:
        return ""
    
    # 日文字符過濾
    japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uFF00-\uFFEF\u0020-\u007E]+')
    filtered_segments = japanese_pattern.findall(text)
    cleaned_text = "".join(filtered_segments).strip()
    
    # 不想要的短語
    unwanted_phrases = [
        "[音声なし]", "ご視聴ありがとうございました", "最後までご視聴ありがとうございました",
        "(幕の開ける音)", "(拍手)", "(笑い)", "(ため息)", "字幕",
    ]
    
    for phrase in unwanted_phrases:
        if phrase in cleaned_text:
            return ""
    
    # 過短的文字
    if len(cleaned_text) < 2:
        return ""
    
    return cleaned_text

def remove_duplicate(current: str, previous: str) -> str:
    """移除與上一次轉錄重複的部分。"""
    if not previous or not current:
        return current
    
    # 檢查是否完全重複
    if current == previous:
        return ""
    
    # 檢查是否為前一次的子字串
    if current in previous:
        return ""
    
    # 檢查重疊部分並移除
    for i in range(min(len(previous), len(current)), 0, -1):
        if previous[-i:] == current[:i]:
            return current[i:]
    
    return current

# ----------------------------------------------------
# 核心處理函數
# ----------------------------------------------------

def process_audio_chunk(audio_data_b64: str, r):
    """處理音訊塊，使用滑動視窗機制。"""
    global audio_buffer, overlap_buffer, last_transcription
    
    # 解碼音訊
    raw_audio_bytes = base64.b64decode(audio_data_b64)
    
    # 🌟 合併重疊緩衝區和新數據
    audio_buffer = overlap_buffer + audio_buffer + raw_audio_bytes
    
    # 計算目標大小
    target_buffer_size = int(BUFFER_DURATION_S * SAMPLE_RATE * BYTES_PER_SAMPLE)
    overlap_size = int(OVERLAP_DURATION_S * SAMPLE_RATE * BYTES_PER_SAMPLE)
    
    if len(audio_buffer) < target_buffer_size:
        return
    
    # 取出處理的音訊
    audio_to_process = audio_buffer[:target_buffer_size]
    
    # 🌟 保留重疊部分供下次使用
    overlap_buffer = audio_buffer[target_buffer_size - overlap_size:target_buffer_size]
    audio_buffer = audio_buffer[target_buffer_size:]
    
    # 轉換為 numpy array
    audio_array = np.frombuffer(audio_to_process, dtype=np.int16).astype(np.float32) / 32768.0
    
    # ASR 轉錄
    transcribed_text = whisper_asr(audio_array)
    
    # 過濾文字
    transcribed_text = filter_text(transcribed_text)
    if not transcribed_text:
        return
    
    # 🌟 去除重複
    transcribed_text = remove_duplicate(transcribed_text, last_transcription)
    if not transcribed_text:
        return
    
    last_transcription = transcribed_text
    
    # 🌟 並行執行翻譯
    future = executor.submit(google_mt, transcribed_text)
    translated_text = future.result(timeout=5)
    
    # 時間戳
    tz = timezone(timedelta(hours=8))
    timestamp = datetime.now(tz).strftime("%H:%M:%S")
    
    result = {
        "timestamp": timestamp,
        "source_lang": SOURCE_LANG_CODE,
        "target_lang": TARGET_LANG_CODE,
        "duration_s": f"{BUFFER_DURATION_S:.3f}",
        "transcription": transcribed_text,
        "translation": translated_text
    }
    
    try:
        json_output = json.dumps(result, ensure_ascii=False)
        r.publish(TRANSLATION_CHANNEL, json_output)
    except Exception as e:
        print(f"發佈失敗: {e}", file=sys.stderr, flush=True)


def main():
    """主循環。"""
    init_global_resources()

    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        r.ping()
        print(f"Python 成功連接到 Redis ({REDIS_HOST}:{REDIS_PORT})。", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"致命錯誤：無法連接到 Redis: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    p = r.pubsub()
    p.subscribe(AUDIO_CHANNEL)
    print(f"Python 成功訂閱 Redis 頻道: {AUDIO_CHANNEL}。", file=sys.stderr, flush=True)

    for message in p.listen():
        if message['type'] == 'message':
            audio_chunk_b64 = message['data'].decode('utf-8')
            process_audio_chunk(audio_chunk_b64, r)
        elif message['type'] == 'subscribe':
            print(f"已成功訂閱 {message['channel'].decode('utf-8')}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()