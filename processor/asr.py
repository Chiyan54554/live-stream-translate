"""
ASR 模組 - 語音辨識
"""
import os
import sys
import time
import numpy as np
from collections import Counter

from config import (
    ASR_MODEL_NAME, MODEL_CACHE_DIR, USE_KOTOBA_PIPELINE,
    SAMPLE_RATE, SOURCE_LANG_CODE, MIN_AUDIO_ENERGY,
    USE_VAD, VAD_THRESHOLD, SUPPRESS_SILENCE, ONLY_VOICE_FREQ,
    AVG_PROB_THRESHOLD, MAX_INSTANT_WORDS
)

# === 全域變數 ===
asr_model = None
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
USING_KOTOBA_PIPELINE = False
TRANSFORMERS_AVAILABLE = False


def setup_environment():
    """設定環境變數和 CUDA"""
    global DEVICE, COMPUTE_TYPE, TRANSFORMERS_AVAILABLE
    
    # 確保 cuDNN 路徑正確
    try:
        import nvidia.cudnn
        cudnn_lib = os.path.join(nvidia.cudnn.__path__[0], "lib")
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        if cudnn_lib not in current_ld:
            os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib}:{current_ld}"
        print(f"✅ cuDNN 路徑已設定: {cudnn_lib}", file=sys.stderr, flush=True)
    except ImportError:
        print("⚠️ nvidia-cudnn 未安裝", file=sys.stderr, flush=True)
    
    import torch
    print(f"PyTorch: {torch.__version__}", file=sys.stderr, flush=True)
    print(f"CUDA 可用: {torch.cuda.is_available()}", file=sys.stderr, flush=True)
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}", file=sys.stderr, flush=True)
        print(f"GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr, flush=True)
        DEVICE = "cuda"
        COMPUTE_TYPE = "float16"
    
    import stable_whisper
    print(f"✅ stable-ts 版本: {stable_whisper.__version__}", file=sys.stderr, flush=True)
    
    try:
        from transformers import pipeline as hf_pipeline
        TRANSFORMERS_AVAILABLE = True
        print("✅ Transformers pipeline 可用", file=sys.stderr, flush=True)
    except ImportError:
        print("⚠️ Transformers 未安裝，將使用 faster-whisper", file=sys.stderr, flush=True)


def init_asr_model():
    """初始化 ASR 模型"""
    global asr_model, DEVICE, COMPUTE_TYPE, USING_KOTOBA_PIPELINE
    
    import torch
    import stable_whisper
    import requests
    import threading
    from config import LLM_API_URL, LLM_MODEL
    
    # 並行測試 LLM 連線並預熱模型
    def test_llm_async():
        import time as _time
        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"🔄 等待 LLM 模型載入... ({attempt + 1}/{max_retries})", file=sys.stderr, flush=True)
                test_resp = requests.post(
                    LLM_API_URL,
                    json={"model": LLM_MODEL, "prompt": "測試", "stream": False, "think": False},
                    timeout=60  # 首次載入需要較長時間
                )
                if test_resp.status_code == 200:
                    print(f"✅ LLM 翻譯引擎就緒 ({LLM_MODEL})", file=sys.stderr, flush=True)
                    return
                else:
                    print(f"⚠️ LLM 回應異常: {test_resp.status_code}", file=sys.stderr, flush=True)
            except requests.exceptions.Timeout:
                print(f"⚠️ LLM 載入中，等待...", file=sys.stderr, flush=True)
                _time.sleep(2)
            except requests.exceptions.ConnectionError:
                print(f"⚠️ Ollama 尚未就緒，等待...", file=sys.stderr, flush=True)
                _time.sleep(2)
            except Exception as e:
                print(f"⚠️ LLM 測試失敗: {e}", file=sys.stderr, flush=True)
                _time.sleep(2)
        print(f"⚠️ LLM 預熱失敗，翻譯可能延遲", file=sys.stderr, flush=True)
    
    # 啟動 LLM 測試（非阻塞）
    llm_thread = threading.Thread(target=test_llm_async, daemon=True)
    llm_thread.start()

    start = time.time()
    
    # 根據模型類型選擇載入方式
    if USE_KOTOBA_PIPELINE:
        if not TRANSFORMERS_AVAILABLE:
            print(f"⚠️ 使用 Kotoba 需要 Transformers，但未安裝", file=sys.stderr, flush=True)
            print(f"🔄 自動切換到 large-v3 (faster-whisper)...", file=sys.stderr, flush=True)
        else:
            try:
                from transformers import pipeline as hf_pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
                
                model_version = "v2.2" if "v2.2" in ASR_MODEL_NAME else "v2.1"
                print(f"🔄 使用 Transformers Pipeline 載入 Kotoba-Whisper {model_version}...", file=sys.stderr, flush=True)
                
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
                
                print(f"✅ Kotoba-Whisper {model_version} 已就緒 (Transformers)", file=sys.stderr, flush=True)
                print(f"✅ 🚀 GPU 模式: {DEVICE}/{COMPUTE_TYPE}, {time.time()-start:.1f}s", file=sys.stderr, flush=True)
                
                # 等待 LLM 測試完成
                llm_thread.join(timeout=5)
                return
                
            except Exception as e:
                print(f"⚠️ Kotoba Pipeline 載入失敗: {e}", file=sys.stderr, flush=True)
                print(f"🔄 退回使用 large-v3 (faster-whisper)...", file=sys.stderr, flush=True)
                import traceback
                traceback.print_exc()
    
    # 標準 faster-whisper + stable-ts
    USING_KOTOBA_PIPELINE = False
    fallback_model = "large-v3" if USE_KOTOBA_PIPELINE else ASR_MODEL_NAME
    
    def try_load_model(device, compute_type):
        try:
            print(f"🔄 使用 stable-ts 載入 {fallback_model}: {device}/{compute_type}...", file=sys.stderr, flush=True)
            
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
            print(f"⚠️ {device}/{compute_type} 失敗: {e}", file=sys.stderr, flush=True)
            import traceback
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
    print(f"✅ {status} 模式 ({fallback_model}): {DEVICE}/{COMPUTE_TYPE}, {time.time()-start:.1f}s", file=sys.stderr, flush=True)
    print(f"✅ stable-ts 模型已就緒", file=sys.stderr, flush=True)
    
    # 等待 LLM 測試完成
    llm_thread.join(timeout=5)


def check_voice_activity(audio_array: np.ndarray) -> bool:
    """簡單的語音活動偵測 (VAD)"""
    rms = np.sqrt(np.mean(audio_array ** 2))
    return rms > MIN_AUDIO_ENERGY


def whisper_asr(audio_array: np.ndarray) -> str:
    """使用 ASR 進行語音辨識"""
    if asr_model is None or not check_voice_activity(audio_array):
        return ""

    try:
        # Kotoba-Whisper (Transformers Pipeline)
        if USING_KOTOBA_PIPELINE:
            audio_input = {
                "raw": audio_array,
                "sampling_rate": SAMPLE_RATE
            }
            
            result = asr_model(
                audio_input,
                chunk_length_s=30,
                stride_length_s=[4, 2],  # 左右 stride 確保連貫
                batch_size=1,
                return_timestamps=True,  # 使用 segment-level timestamps (穩定)
                ignore_warning=True,
                generate_kwargs={
                    "language": "ja",
                    "task": "transcribe",
                    "num_beams": 5,
                    "do_sample": False,
                    "repetition_penalty": 1.3,
                    "no_repeat_ngram_size": 4,
                    "length_penalty": 1.0,
                    "max_new_tokens": 440,
                },
            )
            
            # 提取文字
            text = ""
            if isinstance(result, dict):
                text = result.get("text", "")
            else:
                text = str(result)
            
            return text.strip()
        
        # 標準 faster-whisper + stable-ts
        result = asr_model.transcribe(
            audio_array,
            language=SOURCE_LANG_CODE,
            beam_size=5,
            best_of=5,
            patience=1.2,
            temperature=[0.0, 0.2],
            compression_ratio_threshold=2.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.5,
            log_prob_threshold=AVG_PROB_THRESHOLD,
            initial_prompt="",
            word_timestamps=True,
            vad=USE_VAD,
            vad_threshold=VAD_THRESHOLD,
            suppress_silence=SUPPRESS_SILENCE,
            suppress_word_ts=True,
            min_word_dur=0.1,
            nonspeech_error=0.3,
            only_voice_freq=ONLY_VOICE_FREQ,
            regroup=True,
        )
        
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
                
                # 分級置信度過濾
                if avg_prob > -0.4 and no_speech < 0.3:
                    text_parts.append(seg_text)
                elif avg_prob > -0.7 and no_speech < 0.4 and len(seg_text.strip()) >= 3:
                    text_parts.append(seg_text)
                elif avg_prob > -1.0 and no_speech < 0.15 and len(seg_text.strip()) >= 5:
                    text_parts.append(seg_text)
        else:
            text_parts = [result.text if hasattr(result, 'text') else str(result)]
        
        text = "".join(text_parts).strip()
        return text

    except Exception as e:
        print(f"ASR 錯誤: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        return ""
