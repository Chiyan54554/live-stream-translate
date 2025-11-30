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

# 🚀 延遲優化：縮短緩衝區 (5s -> 3s)，重疊時間 (1.5s -> 1s)
BUFFER_DURATION_S = 3.0
OVERLAP_DURATION_S = 1.0
MIN_AUDIO_ENERGY = 0.005  # 略微降低門檻，避免漏掉輕聲

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
last_transcriptions = deque(maxlen=3)  # 🎯 記錄最近 3 次轉錄用於去重
context_history = deque(maxlen=8)      # 🎯 增加上下文長度 (5 -> 8)
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
    """生成上下文提示 - 針對直播優化"""
    # 🎯 更精確的場景描述，幫助 Whisper 理解語境
    base_prompt = "これは日本語のライブ配信です。配信者がリスナーと会話しています。"
    
    if not context_history:
        return base_prompt
    
    # 取最近 4 句作為上下文（不要太長以免誤導）
    recent = "。".join(list(context_history)[-4:])
    return f"{base_prompt} {recent}"

def whisper_asr(audio_array: np.ndarray) -> str:
    """使用 faster-whisper 進行語音辨識。"""
    if asr_model is None or not check_voice_activity(audio_array):
        return ""

    try:
        segments, info = asr_model.transcribe(
            audio_array,
            language=SOURCE_LANG_CODE,
            
            # 🎯 準確度優化 (不增加延遲)
            beam_size=5,              # 維持速度
            best_of=5,                # 🎯 增加候選數量 (3 -> 5)，提升準確度
            patience=1.8,             # 🎯 略微增加耐心值 (1.5 -> 1.8)
            
            temperature=[0.0, 0.15, 0.3],  # 🎯 更細緻的溫度回退
            compression_ratio_threshold=2.2,  # 🎯 更嚴格的壓縮比 (過濾重複)
            
            condition_on_previous_text=True,  # 保持上下文
            no_speech_threshold=0.6,   # 🎯 提高靜音門檻 (0.5 -> 0.6)
            log_prob_threshold=-0.7,   # 🎯 更嚴格的置信度 (-0.8 -> -0.7)
            
            initial_prompt=get_context_prompt(),
            
            # 🎯 VAD 優化：平衡響應與準確度
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.4,            # 🎯 稍微提高門檻 (減少噪音)
                min_speech_duration_ms=180,  # 🎯 略微增加最小語音長度
                min_silence_duration_ms=350, # 🎯 適度增加靜音判定
                speech_pad_ms=220,
            ),
            
            word_timestamps=False,    # 維持關閉以保持速度
        )
        
        text_parts = []
        for seg in segments:
            # 🎯 更嚴格的置信度過濾
            if seg.avg_logprob > -0.7 and seg.no_speech_prob < 0.5:
                text_parts.append(seg.text)
            elif seg.avg_logprob > -0.85 and seg.no_speech_prob < 0.3:
                # 🎯 次優但高確定性的片段也接受
                text_parts.append(seg.text)
        
        result = "".join(text_parts).strip()
        
        # 🎯 更新上下文 (只保留有意義的內容)
        if result and len(result) >= 4:
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
    
    # 🎯 擴展幻覺過濾列表 (針對直播場景)
    unwanted = [
        # 常見幻覺
        "[音声なし]", "ご視聴ありがとう", "最後までご視聴",
        "(拍手)", "(笑い)", "(ため息)", "字幕",
        "チャンネル登録", "高評価", "MBSニュース",
        "提供は", "ご覧いただき", "ありがとうございました",
        # 🎯 新增：更多幻覺模式
        "お疲れ様でした", "また会いましょう", "バイバイ",
        "次回も", "チャンネル", "登録", "お願いします",
        "♪", "BGM", "音楽", "エンディング",
        "テロップ", "ナレーション", "アナウンス",
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
    
    # 🎯 檢查是否與最近的任何一次轉錄重複
    for old in last_transcriptions:
        if current == old or current in old:
            return ""
    
    # 🎯 改進重疊檢測
    if previous in current:
        idx = current.find(previous)
        if idx == 0:
            return current[len(previous):].strip()
    
    # 🎯 更智能的後綴-前綴重疊檢測
    max_overlap = min(len(previous), len(current), 20)  # 限制檢測長度
    for i in range(max_overlap, 2, -1):  # 至少 3 個字符才算重疊
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
    
    # 🎯 去除重複 (使用歷史記錄)
    text = remove_duplicate(text, last_transcription)
    if not text:
        return
    
    # 🎯 更新歷史記錄
    last_transcription = text
    last_transcriptions.append(text)
    
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