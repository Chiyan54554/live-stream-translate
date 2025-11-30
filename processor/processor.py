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
from collections import deque

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 🌟 確保 cuDNN 路徑正確（在 import torch 之前）
try:
    import nvidia.cudnn
    cudnn_lib = os.path.join(nvidia.cudnn.__path__[0], "lib")
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if cudnn_lib not in current_ld:
        os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib}:{current_ld}"
    print(f"✅ cuDNN 路徑已設定: {cudnn_lib}", file=sys.stderr, flush=True)
except ImportError:
    print("⚠️ nvidia-cudnn 未安裝", file=sys.stderr, flush=True)

try:
    import torch
    print(f"PyTorch: {torch.__version__}", file=sys.stderr, flush=True)
    print(f"CUDA 可用: {torch.cuda.is_available()}", file=sys.stderr, flush=True)
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}", file=sys.stderr, flush=True)
        print(f"GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr, flush=True)
    
    from faster_whisper import WhisperModel
    from deep_translator import GoogleTranslator
except ImportError as e:
    print(f"錯誤：{e}", file=sys.stderr, flush=True)
    sys.exit(1)

# --- 配置參數 ---
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
SOURCE_LANG_CODE = "ja"
TARGET_LANG_CODE = "zh-TW"

BUFFER_DURATION_S = 5.0
OVERLAP_DURATION_S = 1.5
MIN_AUDIO_ENERGY = 0.006

REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
AUDIO_CHANNEL = "audio_feed"
TRANSLATION_CHANNEL = "translation_feed"

ASR_MODEL_NAME = os.getenv('ASR_MODEL_NAME', 'large-v3')
MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', '/root/.cache/huggingface/hub')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

asr_model = None
translator = None
audio_buffer = b''
overlap_buffer = b''
last_transcription = ""
context_history = deque(maxlen=5)
executor = ThreadPoolExecutor(max_workers=2)

def init_global_resources():
    global asr_model, translator, DEVICE, COMPUTE_TYPE
    
    print(f"="*50, file=sys.stderr, flush=True)
    print(f"🎯 設備: {DEVICE}, 計算類型: {COMPUTE_TYPE}", file=sys.stderr, flush=True)
    print(f"🎯 模型: {ASR_MODEL_NAME}", file=sys.stderr, flush=True)
    print(f"="*50, file=sys.stderr, flush=True)

    try:
        translator = GoogleTranslator(source=SOURCE_LANG_CODE, target=TARGET_LANG_CODE)
        print("✅ 翻譯引擎就緒", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"❌ 翻譯引擎失敗: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    def try_load_model(device, compute_type):
        try:
            print(f"🔄 載入: {device}/{compute_type}...", file=sys.stderr, flush=True)
            model = WhisperModel(
                ASR_MODEL_NAME,
                device=device,
                compute_type=compute_type,
                download_root=MODEL_CACHE_DIR,
                cpu_threads=os.cpu_count() or 4,
                num_workers=2,
            )
            # 預熱測試
            list(model.transcribe(np.zeros(16000, dtype=np.float32), language="ja"))
            return model
        except Exception as e:
            print(f"⚠️ {device}/{compute_type} 失敗: {e}", file=sys.stderr, flush=True)
            return None

    start = time.time()
    for device, ctype in [("cuda", "float16"), ("cuda", "int8_float16"), ("cpu", "int8")]:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        asr_model = try_load_model(device, ctype)
        if asr_model:
            DEVICE, COMPUTE_TYPE = device, ctype
            break
    
    if not asr_model:
        print("❌ 模型載入失敗", file=sys.stderr, flush=True)
        sys.exit(1)
    
    status = "🚀 GPU" if DEVICE == "cuda" else "⚠️ CPU"
    print(f"✅ {status} 模式: {DEVICE}/{COMPUTE_TYPE}, {time.time()-start:.1f}s", file=sys.stderr, flush=True)

def check_voice_activity(audio_array: np.ndarray) -> bool:
    """簡單的語音活動偵測 (VAD)。"""
    rms = np.sqrt(np.mean(audio_array ** 2))
    return rms > MIN_AUDIO_ENERGY

def get_context_prompt() -> str:
    """生成上下文提示"""
    if not context_history:
        return "これは日本語の会話です。"
    recent = "。".join(list(context_history)[-3:])
    return f"これは日本語の会話です。{recent}"

def whisper_asr(audio_array: np.ndarray) -> str:
    """使用 faster-whisper 進行語音辨識。"""
    if asr_model is None or not check_voice_activity(audio_array):
        return ""

    try:
        segments, info = asr_model.transcribe(
            audio_array,
            language=SOURCE_LANG_CODE,
            
            # 🌟 提升準確度的參數
            beam_size=8,              # 增加搜索寬度
            best_of=8,                # 多候選選擇
            patience=2.0,             # 增加耐心值
            
            temperature=[0.0, 0.2, 0.4],  # 多溫度回退
            compression_ratio_threshold=2.4,
            
            condition_on_previous_text=True,  # 利用上下文
            no_speech_threshold=0.5,
            log_prob_threshold=-0.8,
            
            initial_prompt=get_context_prompt(),
            
            # 🌟 VAD 優化
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.4,
                min_speech_duration_ms=200,
                min_silence_duration_ms=400,
                speech_pad_ms=250,
            ),
            
            word_timestamps=True,
        )
        
        text_parts = []
        for seg in segments:
            # 過濾低置信度片段
            if seg.avg_logprob > -0.9 and seg.no_speech_prob < 0.6:
                text_parts.append(seg.text)
        
        result = "".join(text_parts).strip()
        
        # 更新上下文
        if result and len(result) > 3:
            context_history.append(result)
        
        return result

    except Exception as e:
        print(f"ASR 錯誤: {e}", file=sys.stderr, flush=True)
        return ""

def google_mt(text: str) -> str:
    """使用 Deep Translator 進行翻譯。"""
    if not text or not translator:
        return ""
    try:
        return translator.translate(text)
    except Exception as e:
        print(f"翻譯錯誤: {e}", file=sys.stderr, flush=True)
        return ""

def filter_text(text: str) -> str:
    """過濾無效文字。"""
    if not text:
        return ""
    
    # 日文字符過濾
    pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uFF00-\uFFEF\u0020-\u007E]+')
    cleaned = "".join(pattern.findall(text)).strip()
    
    # 🌟 擴展過濾列表
    unwanted = [
        "[音声なし]", "ご視聴ありがとう", "最後までご視聴",
        "(拍手)", "(笑い)", "(ため息)", "字幕",
        "チャンネル登録", "高評価", "MBSニュース",
        "提供は", "ご覧いただき",
    ]
    
    for phrase in unwanted:
        if phrase in cleaned:
            return ""
    
    # 檢查重複字符（幻覺特徵）
    if len(cleaned) > 4:
        char_count = max(cleaned.count(c) for c in set(cleaned))
        if char_count > len(cleaned) * 0.5:
            return ""
    
    return cleaned if len(cleaned) >= 2 else ""

def remove_duplicate(current: str, previous: str) -> str:
    """移除與上一次轉錄重複的部分。"""
    if not previous or not current:
        return current
    if current == previous or current in previous:
        return ""
    
    # 🌟 改進重疊檢測
    if previous in current:
        idx = current.find(previous)
        if idx == 0:
            return current[len(previous):].strip()
    
    for i in range(min(len(previous), len(current)), 0, -1):
        if previous[-i:] == current[:i]:
            return current[i:].strip()
    
    return current

# ----------------------------------------------------
# 核心處理函數
# ----------------------------------------------------

def process_audio_chunk(audio_data_b64: str, r):
    """處理音訊塊，使用滑動視窗機制。"""
    global audio_buffer, overlap_buffer, last_transcription
    
    # 解碼音訊
    raw_bytes = base64.b64decode(audio_data_b64)
    audio_buffer = overlap_buffer + audio_buffer + raw_bytes
    
    # 計算目標大小
    target_size = int(BUFFER_DURATION_S * SAMPLE_RATE * BYTES_PER_SAMPLE)
    overlap_size = int(OVERLAP_DURATION_S * SAMPLE_RATE * BYTES_PER_SAMPLE)
    
    if len(audio_buffer) < target_size:
        return
    
    # 取出處理的音訊
    audio_to_process = audio_buffer[:target_size]
    # 🌟 保留重疊部分供下次使用
    overlap_buffer = audio_buffer[target_size - overlap_size:target_size]
    audio_buffer = audio_buffer[target_size:]
    
    # 轉換為 numpy array
    audio_array = np.frombuffer(audio_to_process, dtype=np.int16).astype(np.float32) / 32768.0
    
    # ASR 轉錄
    text = whisper_asr(audio_array)
    # 過濾文字
    text = filter_text(text)
    if not text:
        return
    
    # 🌟 去除重複
    text = remove_duplicate(text, last_transcription)
    if not text:
        return
    
    last_transcription = text
    
    # 🌟 並行執行翻譯
    future = executor.submit(google_mt, text)
    translation = future.result(timeout=5)
    
    # 時間戳
    tz = timezone(timedelta(hours=8))
    result = {
        "timestamp": datetime.now(tz).strftime("%H:%M:%S"),
        "source_lang": SOURCE_LANG_CODE,
        "target_lang": TARGET_LANG_CODE,
        "duration_s": f"{BUFFER_DURATION_S:.3f}",
        "transcription": text,
        "translation": translation
    }
    
    try:
        r.publish(TRANSLATION_CHANNEL, json.dumps(result, ensure_ascii=False))
    except Exception as e:
        print(f"發佈錯誤: {e}", file=sys.stderr, flush=True)

def main():
    """主循環。"""
    init_global_resources()

    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        r.ping()
        print(f"✅ Redis 連線成功", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"❌ Redis 連線失敗: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    p = r.pubsub()
    p.subscribe(AUDIO_CHANNEL)
    print(f"✅ 已訂閱: {AUDIO_CHANNEL}", file=sys.stderr, flush=True)

    for msg in p.listen():
        if msg['type'] == 'message':
            process_audio_chunk(msg['data'].decode('utf-8'), r)

if __name__ == "__main__":
    main()