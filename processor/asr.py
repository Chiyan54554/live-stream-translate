"""
ASR 模組 - 語音辨識
🚀 優化版：預建立參數字典、減少重複運算
"""
import os
import sys
import time
import traceback
import numpy as np
from collections import Counter

# 控制資訊級日誌：LOG_VERBOSE=1 時輸出；預設靜音
LOG_VERBOSE = os.getenv("LOG_VERBOSE", "0") == "1"
def info(msg):
    if LOG_VERBOSE:
        print(msg, file=sys.stderr, flush=True)

from config import (
    ASR_MODEL_NAME, MODEL_CACHE_DIR, USE_KOTOBA_PIPELINE,
    SAMPLE_RATE, SOURCE_LANG_CODE, MIN_AUDIO_ENERGY,
    USE_VAD, VAD_THRESHOLD, SUPPRESS_SILENCE, ONLY_VOICE_FREQ,
    AVG_PROB_THRESHOLD, MAX_INSTANT_WORDS,
    USE_GOOGLE_STT, GOOGLE_STT_MODEL, GOOGLE_STT_MAX_ALTERNATIVES,
    GOOGLE_STT_ENABLE_PUNCTUATION, GOOGLE_STT_FAIL_LIMIT, GOOGLE_STT_BACKOFF_MS
)

# === 全域變數 ===
asr_model = None
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
USING_KOTOBA_PIPELINE = False
TRANSFORMERS_AVAILABLE = False
google_speech_client = None
GOOGLE_STT_ENABLED = USE_GOOGLE_STT
GOOGLE_STT_FAILS = 0

# ============================================================
# 🚀 預建立 ASR 參數字典（避免每次呼叫重新建立）
# ============================================================

# Kotoba Pipeline 音訊輸入模板（每次只需更新 raw）
_KOTOBA_AUDIO_TEMPLATE = {
    "sampling_rate": SAMPLE_RATE
}

# Kotoba Pipeline 生成參數（不變）
_KOTOBA_GENERATE_KWARGS = {
    "language": "ja",
    "task": "transcribe",
    "num_beams": 5,
    "do_sample": False,
    "repetition_penalty": 1.3,
    "no_repeat_ngram_size": 4,
    "length_penalty": 1.0,
    "max_new_tokens": 440,
}

# Kotoba Pipeline 呼叫參數（不含音訊）
_KOTOBA_PIPELINE_KWARGS = {
    "chunk_length_s": 30,
    "stride_length_s": [4, 2],
    "batch_size": 1,
    "return_timestamps": True,
    "ignore_warning": True,
    "generate_kwargs": _KOTOBA_GENERATE_KWARGS,
}

# stable-ts 轉錄參數（不變）
_STABLE_TS_KWARGS = {
    "language": SOURCE_LANG_CODE,
    "beam_size": 5,
    "best_of": 5,
    "patience": 1.2,
    "temperature": [0.0, 0.2],
    "compression_ratio_threshold": 2.0,
    "condition_on_previous_text": False,
    "no_speech_threshold": 0.5,
    "log_prob_threshold": AVG_PROB_THRESHOLD,
    "initial_prompt": "",
    "word_timestamps": True,
    "vad": USE_VAD,
    "vad_threshold": VAD_THRESHOLD,
    "suppress_silence": SUPPRESS_SILENCE,
    "suppress_word_ts": True,
    "min_word_dur": 0.1,
    "nonspeech_error": 0.3,
    "only_voice_freq": ONLY_VOICE_FREQ,
    "regroup": True,
}

# 置信度閾值（預計算）
_CONFIDENCE_THRESHOLDS = (
    (-0.4, 0.3, 0),   # (avg_prob_min, no_speech_max, min_text_len)
    (-0.7, 0.4, 3),
    (-1.0, 0.15, 5),
)


def setup_environment():
    """設定環境變數和 CUDA"""
    global DEVICE, COMPUTE_TYPE, TRANSFORMERS_AVAILABLE

    # 雲端 STT 不需 CUDA/Torch
    if GOOGLE_STT_ENABLED:
        info("🎯 Google Speech-to-Text 已啟用")
        return
    
    # 確保 cuDNN 路徑正確
    try:
        import nvidia.cudnn
        cudnn_lib = os.path.join(nvidia.cudnn.__path__[0], "lib")
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        if cudnn_lib not in current_ld:
            os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib}:{current_ld}"
        info(f"✅ cuDNN 路徑已設定: {cudnn_lib}")
    except ImportError:
        info("⚠️ nvidia-cudnn 未安裝")
    
    import torch
    info(f"PyTorch: {torch.__version__}")
    info(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        info(f"CUDA 版本: {torch.version.cuda}")
        info(f"GPU: {torch.cuda.get_device_name(0)}")
        DEVICE = "cuda"
        COMPUTE_TYPE = "float16"
    
    import stable_whisper
    info(f"✅ stable-ts 版本: {stable_whisper.__version__}")
    
    try:
        from transformers import pipeline as hf_pipeline
        TRANSFORMERS_AVAILABLE = True
        info("✅ Transformers pipeline 可用")
    except ImportError:
        info("⚠️ Transformers 未安裝，將使用 faster-whisper")


def init_asr_model():
    """初始化 ASR 模型"""
    global asr_model, DEVICE, COMPUTE_TYPE, USING_KOTOBA_PIPELINE, google_speech_client

    if GOOGLE_STT_ENABLED:
        try:
            from google.cloud import speech_v1p1beta1 as speech
        except ImportError:
            print("❌ 缺少 google-cloud-speech 套件，請安裝後重試", file=sys.stderr, flush=True)
            sys.exit(1)

        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if cred_path and not os.path.exists(cred_path):
            print(f"❌ GOOGLE_APPLICATION_CREDENTIALS 路徑不存在: {cred_path}", file=sys.stderr, flush=True)
            sys.exit(1)

        try:
            google_speech_client = speech.SpeechClient()
            info(f"✅ Google Speech-to-Text 客戶端已就緒 (model={GOOGLE_STT_MODEL})")
            return
        except Exception as e:
            print(f"❌ 初始化 Google Speech-to-Text 失敗: {e}", file=sys.stderr, flush=True)
            traceback.print_exc()
            sys.exit(1)
    
    import torch
    import stable_whisper

    start = time.time()
    
    # 根據模型類型選擇載入方式
    if USE_KOTOBA_PIPELINE:
        if not TRANSFORMERS_AVAILABLE:
            info(f"⚠️ 使用 Kotoba 需要 Transformers，但未安裝")
            info(f"🔄 自動切換到 large-v3 (faster-whisper)...")
        else:
            try:
                from transformers import pipeline as hf_pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
                
                model_version = "v2.2" if "v2.2" in ASR_MODEL_NAME else "v2.1"
                info(f"🔄 使用 Transformers Pipeline 載入 Kotoba-Whisper {model_version}...")
                
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                
                # 只使用 model_kwargs，不在 pipeline 中重複設定 torch_dtype
                model_kwargs = {
                    "attn_implementation": "sdpa",
                    "low_cpu_mem_usage": True,
                } if torch.cuda.is_available() else {
                    "low_cpu_mem_usage": True,
                }
                
                asr_model = hf_pipeline(
                    "automatic-speech-recognition",
                    model=ASR_MODEL_NAME,
                    torch_dtype=torch_dtype,
                    device=device,
                    model_kwargs=model_kwargs,
                    batch_size=1,
                    trust_remote_code=True,
                )
                
                DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
                COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "float32"
                USING_KOTOBA_PIPELINE = True
                
                info(f"✅ Kotoba-Whisper {model_version} 已就緒 (Transformers)")
                info(f"✅ 🚀 GPU 模式: {DEVICE}/{COMPUTE_TYPE}, {time.time()-start:.1f}s")
                return
                
            except Exception as e:
                info(f"⚠️ Kotoba Pipeline 載入失敗: {e}")
                info(f"🔄 退回使用 large-v3 (faster-whisper)...")
                traceback.print_exc()
    
    # 標準 faster-whisper + stable-ts
    USING_KOTOBA_PIPELINE = False
    fallback_model = "large-v3" if USE_KOTOBA_PIPELINE else ASR_MODEL_NAME
    
    def try_load_model(device, compute_type):
        try:
            info(f"🔄 使用 stable-ts 載入 {fallback_model}: {device}/{compute_type}...")
            
            model = stable_whisper.load_faster_whisper(
                fallback_model,
                device=device,
                compute_type=compute_type,
                download_root=MODEL_CACHE_DIR,
                cpu_threads=os.cpu_count() or 4,
                num_workers=2,
            )
            
            # 移除預熱步驟以加速載入（首次推理會稍慢但可接受）
            return model
        except Exception as e:
            info(f"⚠️ {device}/{compute_type} 失敗: {e}")
            traceback.print_exc()
            return None

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
    info(f"✅ {status} 模式 ({fallback_model}): {DEVICE}/{COMPUTE_TYPE}, {time.time()-start:.1f}s")
    info(f"✅ stable-ts 模型已就緒")


def check_voice_activity(audio_array: np.ndarray) -> bool:
    """簡單的語音活動偵測 (VAD)"""
    rms = np.sqrt(np.mean(audio_array ** 2))
    return rms > MIN_AUDIO_ENERGY


def google_stt_transcribe(audio_array: np.ndarray) -> str:
    """使用 Google Speech-to-Text 轉寫線性 PCM"""
    global google_speech_client, GOOGLE_STT_FAILS, GOOGLE_STT_ENABLED
    try:
        from google.cloud import speech_v1p1beta1 as speech

        pcm16 = np.clip(audio_array, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        audio_bytes = pcm16.tobytes()

        config_kwargs = dict(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=SOURCE_LANG_CODE,
            max_alternatives=GOOGLE_STT_MAX_ALTERNATIVES,
            enable_automatic_punctuation=GOOGLE_STT_ENABLE_PUNCTUATION,
        )

        if GOOGLE_STT_MODEL:
            # 只有指定時才帶入 model，避免不支援語言時 400 錯誤
            config_kwargs["model"] = GOOGLE_STT_MODEL
            info(f"🔧 Google STT 使用 model={GOOGLE_STT_MODEL}")

        config = speech.RecognitionConfig(**config_kwargs)

        audio = speech.RecognitionAudio(content=audio_bytes)

        response = google_speech_client.recognize(
            config=config,
            audio=audio,
            timeout=15,
        )

        # 成功則重置失敗計數
        GOOGLE_STT_FAILS = 0

        for result in response.results:
            if not result.alternatives:
                continue
            transcript = result.alternatives[0].transcript.strip()
            if transcript:
                return transcript
        return ""

    except Exception as e:
        GOOGLE_STT_FAILS += 1
        print(f"ASR 錯誤 (Google STT): {e}", file=sys.stderr, flush=True)
        # 失敗退避，避免打爆 API
        backoff_ms = min(GOOGLE_STT_BACKOFF_MS * GOOGLE_STT_FAILS, 3000)
        if backoff_ms > 0:
            time.sleep(backoff_ms / 1000.0)
        if GOOGLE_STT_FAILS >= GOOGLE_STT_FAIL_LIMIT:
            GOOGLE_STT_ENABLED = False
            print(f"⚠️ Google STT 連續失敗 {GOOGLE_STT_FAILS} 次，切換回本地 ASR", file=sys.stderr, flush=True)
        traceback.print_exc()
        return ""


def whisper_asr(audio_array: np.ndarray) -> str:
    """使用 ASR 進行語音辨識"""
    if GOOGLE_STT_ENABLED:
        if google_speech_client is None or not check_voice_activity(audio_array):
            return ""
        return google_stt_transcribe(audio_array)

    if asr_model is None or not check_voice_activity(audio_array):
        return ""

    try:
        # Kotoba-Whisper (Transformers Pipeline)
        if USING_KOTOBA_PIPELINE:
            # 使用預建立的參數字典
            audio_input = {"raw": audio_array, **_KOTOBA_AUDIO_TEMPLATE}
            
            result = asr_model(audio_input, **_KOTOBA_PIPELINE_KWARGS)
            
            # 提取文字（優化判斷）
            if isinstance(result, dict):
                return result.get("text", "").strip()
            return str(result).strip()
        
        # 標準 faster-whisper + stable-ts（使用預建立參數）
        result = asr_model.transcribe(audio_array, **_STABLE_TS_KWARGS)
        
        if hasattr(result, 'remove_repetition'):
            result.remove_repetition(max_words=1, verbose=False)
        
        # 過濾低置信度片段
        text_parts = []
        if hasattr(result, 'segments'):
            for seg in result.segments:
                seg_text = seg.text if hasattr(seg, 'text') else str(seg)
                avg_prob = getattr(seg, 'avg_logprob', -0.5)
                no_speech = getattr(seg, 'no_speech_prob', 0.5)
                
                # 幻覺偵測：瞬時詞
                if hasattr(seg, 'words') and seg.words:
                    instant_words = sum(1 for w in seg.words if hasattr(w, 'duration') and w.duration < 0.05)
                    instant_ratio = instant_words / len(seg.words) if seg.words else 0
                    if instant_ratio > MAX_INSTANT_WORDS:
                        print(f"⚠️ 跳過瞬時詞過多片段: {seg_text[:30]}...", file=sys.stderr, flush=True)
                        continue
                
                # 幻覺偵測：單詞重複
                if hasattr(seg, 'words') and seg.words and len(seg.words) >= 4:
                    word_texts = [w.word.strip() for w in seg.words if hasattr(w, 'word')]
                    if word_texts:
                        word_counts = Counter(word_texts)
                        max_word_count = max(word_counts.values())
                        if max_word_count > len(word_texts) * 0.4:
                            print(f"⚠️ 跳過單詞重複片段: {seg_text[:30]}...", file=sys.stderr, flush=True)
                            continue
                
                # 分級置信度過濾（使用預定義閾值）
                seg_text_len = len(seg_text.strip())
                for prob_min, speech_max, min_len in _CONFIDENCE_THRESHOLDS:
                    if avg_prob > prob_min and no_speech < speech_max and seg_text_len >= min_len:
                        text_parts.append(seg_text)
                        break
        else:
            text_parts = [result.text if hasattr(result, 'text') else str(result)]
        
        text = "".join(text_parts).strip()
        return text

    except Exception as e:
        print(f"ASR 錯誤: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return ""
