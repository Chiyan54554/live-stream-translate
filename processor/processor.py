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
    
    # 🎯 使用 stable-ts 整合 faster-whisper
    import stable_whisper
    from deep_translator import GoogleTranslator
    
    print(f"✅ stable-ts 版本: {stable_whisper.__version__}", file=sys.stderr, flush=True)
    
except ImportError as e:
    print(f"錯誤：{e}", file=sys.stderr, flush=True)
    sys.exit(1)

# --- 配置參數 ---
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
SOURCE_LANG_CODE = "ja"
TARGET_LANG_CODE = "zh-TW"

# 🎯 準確率優化：平衡緩衝與延遲
BUFFER_DURATION_S = 4.0       # 🎯 4 秒緩衝，平衡準確度與延遲
OVERLAP_DURATION_S = 0.8      # 🎯 適度重疊確保連貫性
MIN_AUDIO_ENERGY = 0.006      # 🎯 適中的能量門檻

REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
AUDIO_CHANNEL = "audio_feed"
TRANSLATION_CHANNEL = "translation_feed"

# 🎯 使用 large-v3-turbo：更新的模型，幻覺更少
ASR_MODEL_NAME = os.getenv('ASR_MODEL_NAME', 'large-v3-turbo')
MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', '/root/.cache/huggingface/hub')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# 🎯 stable-ts 與 VAD 相關設定
USE_STABLE_TS = True                    # 啟用 stable-ts
USE_VAD = True                          # 啟用 Silero VAD
VAD_THRESHOLD = 0.35                    # VAD 語音偵測閾值
SUPPRESS_SILENCE = True                 # 靜音抑制
HALLUCINATION_SILENCE_TH = 2.0          # 🎯 幻覺靜音閾值（秒）
AVG_PROB_THRESHOLD = -0.8               # 平均置信度閾值
MAX_INSTANT_WORDS = 0.35                # 🎯 降低閾值，更積極過濾幻覺
ONLY_VOICE_FREQ = False                 # 是否只保留語音頻率 (200-5000 Hz)

asr_model = None
translator = None
audio_buffer = b''
overlap_buffer = b''         # 🎯 恢復重疊緩衝區
last_transcription = ""
last_full_sentence = ""       # 🎯 新增：記錄上一個完整句子
pending_text = ""             # 🎯 新增：待處理的不完整文字
last_publish_time = 0
recent_texts = deque(maxlen=10)
context_history = deque(maxlen=8)  # 🎯 增加上下文長度
executor = ThreadPoolExecutor(max_workers=2)

MIN_PUBLISH_INTERVAL = 0.8    # 🎯 縮短最小間隔
SIMILARITY_THRESHOLD = 0.7    # 🎯 提高相似度閾值

def init_global_resources():
    global asr_model, translator, DEVICE, COMPUTE_TYPE
    
    print(f"="*50, file=sys.stderr, flush=True)
    print(f"🎯 設備: {DEVICE}, 計算類型: {COMPUTE_TYPE}", file=sys.stderr, flush=True)
    print(f"🎯 模型: {ASR_MODEL_NAME}", file=sys.stderr, flush=True)
    print(f"🎯 stable-ts: {USE_STABLE_TS}, VAD: {USE_VAD}", file=sys.stderr, flush=True)
    print(f"="*50, file=sys.stderr, flush=True)

    try:
        translator = GoogleTranslator(source=SOURCE_LANG_CODE, target=TARGET_LANG_CODE)
        print("✅ 翻譯引擎就緒", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"❌ 翻譯引擎失敗: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    def try_load_model(device, compute_type):
        try:
            print(f"🔄 使用 stable-ts 載入: {device}/{compute_type}...", file=sys.stderr, flush=True)
            
            # 🎯 使用 stable-ts 的 load_faster_whisper
            model = stable_whisper.load_faster_whisper(
                ASR_MODEL_NAME,
                device=device,
                compute_type=compute_type,
                download_root=MODEL_CACHE_DIR,
                cpu_threads=os.cpu_count() or 4,
                num_workers=2,
            )
            
            # 預熱測試
            warmup_audio = np.zeros(16000, dtype=np.float32)
            _ = model.transcribe(
                warmup_audio,
                language="ja",
                vad=False,  # 預熱時關閉 VAD 加速
                suppress_silence=False,
            )
            
            return model
        except Exception as e:
            print(f"⚠️ {device}/{compute_type} 失敗: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc()
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
    print(f"✅ stable-ts 模型已就緒", file=sys.stderr, flush=True)

def check_voice_activity(audio_array: np.ndarray) -> bool:
    """簡單的語音活動偵測 (VAD)。"""
    rms = np.sqrt(np.mean(audio_array ** 2))
    return rms > MIN_AUDIO_ENERGY

def get_context_prompt() -> str:
    """生成上下文提示 - 針對直播優化"""
    # 🎯 移除 initial_prompt - 它可能被 Whisper 當成轉錄輸出
    # 返回空字串以避免幻覺
    return ""

def whisper_asr(audio_array: np.ndarray) -> str:
    """使用 stable-ts + faster-whisper 進行語音辨識 - 完整整合版"""
    if asr_model is None or not check_voice_activity(audio_array):
        return ""

    try:
        # 🎯 使用 stable-ts 的 transcribe 方法
        # 這會自動整合 VAD、靜音抑制、重複移除等功能
        result = asr_model.transcribe(
            audio_array,
            language=SOURCE_LANG_CODE,
            
            # === 基本 Whisper 參數 ===
            beam_size=5,
            best_of=5,
            patience=1.2,
            temperature=[0.0, 0.2],
            compression_ratio_threshold=2.0,
            condition_on_previous_text=False,  # 🎯 關閉避免錯誤累積
            no_speech_threshold=0.5,
            log_prob_threshold=AVG_PROB_THRESHOLD,
            initial_prompt=get_context_prompt(),
            word_timestamps=True,  # 🎯 啟用詞級時間戳以支援去重複
            
            # === stable-ts VAD 與靜音抑制 ===
            vad=USE_VAD,                      # 🎯 使用 Silero VAD
            vad_threshold=VAD_THRESHOLD,      # 🎯 VAD 閾值
            suppress_silence=SUPPRESS_SILENCE, # 🎯 靜音抑制
            suppress_word_ts=True,            # 🎯 抑制靜音時的時間戳
            
            # === 額外的 stable-ts 參數 ===
            min_word_dur=0.1,                 # 最短詞持續時間
            nonspeech_error=0.3,              # 非語音誤差容忍度
            only_voice_freq=ONLY_VOICE_FREQ,  # 只保留語音頻率範圍
            
            regroup=True,  # 🎯 自動重新分組片段
        )
        
        # 🎯 stable-ts 的核心功能：移除重複
        if hasattr(result, 'remove_repetition'):
            result.remove_repetition(max_words=1, verbose=False)
        
        # 🎯 過濾低置信度片段
        text_parts = []
        if hasattr(result, 'segments'):
            for seg in result.segments:
                # 取得片段屬性
                seg_text = seg.text if hasattr(seg, 'text') else str(seg)
                avg_prob = getattr(seg, 'avg_logprob', -0.5)
                no_speech = getattr(seg, 'no_speech_prob', 0.5)
                
                # 🎯 幻覺偵測：檢查是否有過多瞬時詞
                if hasattr(seg, 'words') and seg.words:
                    instant_words = sum(1 for w in seg.words if hasattr(w, 'duration') and w.duration < 0.05)
                    instant_ratio = instant_words / len(seg.words) if seg.words else 0
                    if instant_ratio > MAX_INSTANT_WORDS:
                        print(f"⚠️ 跳過瞬時詞過多片段: {seg_text[:30]}...", file=sys.stderr, flush=True)
                        continue
                
                # 🎯 額外幻覺偵測：檢查單詞重複
                if hasattr(seg, 'words') and seg.words and len(seg.words) >= 4:
                    word_texts = [w.word.strip() for w in seg.words if hasattr(w, 'word')]
                    if word_texts:
                        from collections import Counter
                        word_counts = Counter(word_texts)
                        max_word_count = max(word_counts.values())
                        # 如果某個詞出現超過 40% 的次數，視為幻覺
                        if max_word_count > len(word_texts) * 0.4:
                            print(f"⚠️ 跳過單詞重複片段: {seg_text[:30]}...", file=sys.stderr, flush=True)
                            continue
                
                # 🎯 分級置信度過濾
                if avg_prob > -0.4 and no_speech < 0.3:
                    text_parts.append(seg_text)
                elif avg_prob > -0.7 and no_speech < 0.4 and len(seg_text.strip()) >= 3:
                    text_parts.append(seg_text)
                elif avg_prob > -1.0 and no_speech < 0.15 and len(seg_text.strip()) >= 5:
                    text_parts.append(seg_text)
        else:
            # fallback: 直接取得文字
            text_parts = [result.text if hasattr(result, 'text') else str(result)]
        
        text = "".join(text_parts).strip()
        return text

    except Exception as e:
        print(f"ASR 錯誤: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        return ""

def google_mt(text: str) -> str:
    """使用 Deep Translator 進行翻譯，並過濾重複。"""
    if not text or not translator:
        return ""
    try:
        translated = translator.translate(text)
        
        # 🎯 過濾翻譯後的重複內容
        if translated:
            translated = filter_translated_repetition(translated)
        
        return translated
    except Exception as e:
        print(f"翻譯錯誤: {e}", file=sys.stderr, flush=True)
        return ""

def filter_translated_repetition(text: str) -> str:
    """過濾翻譯後的重複內容 - 加強版"""
    if not text or len(text) < 4:
        return text
    
    original_text = text
    
    # 🎯 方法 0: 偵測空格分隔的完全相同片段 (如：不在乎的基德先生 不在乎的基德先生)
    if ' ' in text:
        space_parts = [p.strip() for p in text.split(' ') if p.strip()]
        if len(space_parts) >= 2:
            # 檢查連續重複
            unique_space = []
            for p in space_parts:
                if not unique_space or p != unique_space[-1]:
                    # 也檢查相似度
                    is_dup = False
                    for u in unique_space:
                        if p == u or calculate_similarity(p, u) > 0.7:
                            is_dup = True
                            break
                    if not is_dup:
                        unique_space.append(p)
            
            if len(unique_space) < len(space_parts):
                result = ' '.join(unique_space)
                print(f"🔧 去除空格重複: {original_text[:40]} -> {result[:40]}", file=sys.stderr, flush=True)
                text = result
                original_text = result
    
    # 🎯 方法 1: 偵測連續重複的子字串 (如：我可以走了嗎？我可以走了嗎？)
    cleaned = remove_repeated_substrings(text)
    if cleaned != text:
        print(f"🔧 去除重複子字串: {original_text[:40]} -> {cleaned[:40]}", file=sys.stderr, flush=True)
        return cleaned
    
    # 🎯 方法 2: 按標點分割並去重
    separators = ['，', '。', '！', '？']
    for sep in separators:
        if sep in text and text.count(sep) >= 1:
            parts = [p.strip() for p in text.split(sep) if p.strip()]
            if len(parts) >= 2:
                # 檢查是否有重複或高度相似
                unique = []
                for p in parts:
                    is_dup = False
                    for u in unique:
                        # 🎯 降低相似度閾值，更積極去重
                        if p == u or calculate_similarity(p, u) > 0.6:
                            is_dup = True
                            break
                    if not is_dup:
                        unique.append(p)
                
                if len(unique) < len(parts):
                    result = sep.join(unique)
                    if sep in ['。', '！', '？']:
                        result = result + sep if not result.endswith(sep) else result
                    print(f"🔧 去除翻譯重複: {original_text[:40]} -> {result[:40]}", file=sys.stderr, flush=True)
                    return result
    
    return text

def remove_repeated_substrings(text: str) -> str:
    """移除連續重複的子字串 - 保留不重複的前綴"""
    if len(text) < 8:
        return text
    
    # 🎯 方法 1: 按句尾標點分割，找完整的重複句子
    sentence_endings = ['。', '！', '？', '!', '?']
    for ending in sentence_endings:
        if ending in text:
            # 按句尾分割
            parts = []
            current = ""
            for char in text:
                current += char
                if char == ending:
                    if current.strip():
                        parts.append(current.strip())
                    current = ""
            if current.strip():
                parts.append(current.strip())
            
            if len(parts) >= 2:
                # 🎯 保留所有不重複的句子
                unique = []
                seen = set()
                for p in parts:
                    if p not in seen:
                        unique.append(p)
                        seen.add(p)
                
                # 只有當確實有重複被移除時才返回
                if len(unique) < len(parts):
                    return ''.join(unique)
    
    # 🎯 方法 2: 偵測連續重複的子字串模式
    # 優先嘗試較長的模式 (從長到短)
    for pattern_len in range(min(30, len(text) // 2), 4, -1):
        for start in range(len(text) - pattern_len * 2 + 1):
            pattern = text[start:start + pattern_len]
            
            # 跳過純標點或空白
            if all(c in '，。！？ 、,.!? ' for c in pattern):
                continue
            
            # 🎯 確保 pattern 以標點結尾
            has_ending = any(pattern.endswith(e) for e in ['。', '！', '？', '，', '!', '?', ','])
            if not has_ending:
                continue
            
            # 計算連續出現次數
            count = 0
            pos = 0
            first_idx = -1
            while True:
                idx = text.find(pattern, pos)
                if idx == -1:
                    break
                if first_idx == -1:
                    first_idx = idx
                count += 1
                pos = idx + len(pattern)
            
            # 如果模式連續出現 2 次以上
            if count >= 2 and len(pattern) * count > len(text) * 0.5:
                # 🎯 保留重複前的內容 + 一次重複模式
                prefix = text[:first_idx].strip() if first_idx > 0 else ""
                result = pattern.strip()
                if prefix:
                    return prefix + result
                return result
    
    return text

def filter_text(text: str) -> str:
    """過濾無效文字，去除重複後保留有效內容繼續處理。"""
    if not text:
        return ""
    
    # 日文字符過濾
    pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uFF00-\uFFEF\u0020-\u007E]+')
    cleaned = "".join(pattern.findall(text)).strip()
    
    if not cleaned:
        return ""
    
    # 🎯 幻覺過濾列表 (這些是完全無意義的，直接過濾)
    unwanted = [
        # === Whisper 常見幻覺 ===
        "[音声なし]", "ご視聴ありがとう", "最後までご視聴",
        "(拍手)", "(笑い)", "(ため息)", "字幕",
        "チャンネル登録", "高評価", "MBSニュース",
        "提供は", "ご覧いただき", "ありがとうございました",
        "お疲れ様でした", "また会いましょう", "バイバイ",
        "次回も", "チャンネル", "登録", "お願いします",
        "♪", "BGM", "音楽", "エンディング",
        "テロップ", "ナレーション", "アナウンス",
        # === Initial Prompt 被誤輸出的內容 ===
        "話し言葉", "カジュアルな表現", "ネットスラング",
        "VTuber配信", "配信者とリスナー", "日本語の",
        # === 其他常見幻覺模式 ===
        "翻訳", "字幕提供", "自動生成", "機械翻訳",
        "続きは", "詳しくは", "リンクは",
        "概要欄", "説明欄", "コメント欄",
    ]
    
    for phrase in unwanted:
        if phrase in cleaned:
            return ""
    
    # 🎯 去除重複字符後保留有效內容
    if detect_character_repetition(cleaned):
        deduped = remove_source_repetition(cleaned)
        if deduped and len(deduped) >= 2:
            print(f"🔄 去除源文重複: {cleaned[:30]}... -> {deduped[:30]}", file=sys.stderr, flush=True)
            cleaned = deduped
        else:
            print(f"⚠️ 過濾純重複: {cleaned[:30]}...", file=sys.stderr, flush=True)
            return ""
    
    # 🎯 去除重複詞組後保留有效內容
    if detect_phrase_repetition(cleaned):
        deduped = remove_source_repetition(cleaned)
        if deduped and len(deduped) >= 2:
            print(f"🔄 去除源文重複: {cleaned[:30]}... -> {deduped[:30]}", file=sys.stderr, flush=True)
            cleaned = deduped
        else:
            print(f"⚠️ 過濾純重複: {cleaned[:30]}...", file=sys.stderr, flush=True)
            return ""
    
    return cleaned if len(cleaned) >= 2 else ""

def remove_source_repetition(text: str) -> str:
    """從日文源文中去除重複，保留有意義的內容"""
    if not text or len(text) < 4:
        return text
    
    original = text
    
    # 🎯 方法 1: 按空格分割去重
    if ' ' in text:
        parts = text.split(' ')
        unique = []
        seen = set()
        for p in parts:
            p = p.strip()
            if p and p not in seen:
                unique.append(p)
                seen.add(p)
        if len(unique) < len(parts):
            text = ' '.join(unique)
    
    # 🎯 方法 2: 尋找重複模式並只保留一次
    for pattern_len in range(2, min(30, len(text) // 2 + 1)):
        for start in range(min(5, len(text) - pattern_len * 2)):
            pattern = text[start:start + pattern_len]
            
            # 跳過純標點或空白
            if all(c in '、，。！？　 ・ー' for c in pattern):
                continue
            
            # 計算連續出現次數
            count = text.count(pattern)
            
            if count >= 3 and len(pattern) * count > len(text) * 0.4:
                # 找到重複模式，保留一次 + 前後內容
                first_idx = text.find(pattern)
                last_idx = text.rfind(pattern)
                
                prefix = text[:first_idx].strip() if first_idx > 0 else ""
                suffix = text[last_idx + len(pattern):].strip() if last_idx + len(pattern) < len(text) else ""
                
                result = prefix + pattern + suffix
                result = result.strip()
                
                if result and len(result) >= 2:
                    return result
    
    # 🎯 方法 3: 如果整個文字只是單一模式重複
    for pattern_len in range(2, min(20, len(text) // 3 + 1)):
        pattern = text[:pattern_len]
        if all(c in '、，。！？　 ・ー' for c in pattern):
            continue
        
        # 檢查是否整個文字都是這個模式的重複
        repeated = pattern * (len(text) // len(pattern) + 1)
        if text in repeated or repeated.startswith(text):
            return pattern.strip()
    
    return text

def detect_character_repetition(text: str) -> bool:
    """偵測異常的字符重複 (幻覺特徵) - 優化版"""
    if len(text) < 6:
        return False
    
    # 🎯 排除常見的合法重複
    valid_patterns = ['ww', 'ーー', '...', '！！', '？？', '〜〜']
    temp_text = text
    for vp in valid_patterns:
        temp_text = temp_text.replace(vp, '')
    
    if len(temp_text) < 4:
        return False
    
    # 🎯 計算每個字符出現的比例（排除空格和標點）
    content_chars = [c for c in temp_text if c not in ' 　、。！？，']
    if len(content_chars) < 4:
        return False
    
    from collections import Counter
    char_counts = Counter(content_chars)
    max_count = max(char_counts.values())
    
    # 單字符佔比超過 35%
    if max_count > len(content_chars) * 0.35:
        return True
    
    # 🎯 偵測連續重複模式 (如 ABCABCABC)
    for pattern_len in range(2, min(15, len(text) // 3 + 1)):
        for start in range(min(3, len(text) - pattern_len * 3)):
            pattern = text[start:start + pattern_len]
            # 跳過純標點
            if all(c in '、，。！？　 ・' for c in pattern):
                continue
            if pattern * 3 in text:
                return True
    
    return False

def detect_phrase_repetition(text: str) -> bool:
    """偵測重複的詞組 - 加強版"""
    # 🎯 方法 1: 偵測連續重複的子字串
    for pattern_len in range(2, min(20, len(text) // 2 + 1)):
        for start in range(len(text) - pattern_len * 2 + 1):
            pattern = text[start:start + pattern_len]
            
            # 跳過純標點
            if all(c in '、，。！？　 ' for c in pattern):
                continue
            
            # 檢查連續重複
            if pattern * 3 in text:
                return True
    
    # 🎯 方法 2: 按標點分割檢查
    separators = ['、', '，', '。', ' ']
    for sep in separators:
        if sep in text:
            parts = [p.strip() for p in text.split(sep) if p.strip() and len(p.strip()) >= 2]
            if len(parts) >= 3:
                # 檢查連續相同
                consecutive = 1
                for i in range(1, len(parts)):
                    if parts[i] == parts[i-1]:
                        consecutive += 1
                        if consecutive >= 2:  # 🎯 降低到 2 次就視為重複
                            return True
                    else:
                        consecutive = 1
                
                # 檢查總重複率
                from collections import Counter
                counts = Counter(parts)
                for part, count in counts.items():
                    if count >= 2 and count >= len(parts) * 0.4:  # 🎯 降低門檻
                        return True
    
    return False

def remove_duplicate(current: str, previous: str) -> str:
    """移除與上一次轉錄重複的部分。"""
    if not previous or not current:
        return current
    if current == previous or current in previous:
        return ""
    
    # 🎯 檢查是否與最近的任何一次轉錄重複
    for old in recent_texts:
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

def calculate_similarity(s1: str, s2: str) -> float:
    """計算兩個字串的相似度 (0-1) - 使用多種算法"""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    
    # 🎯 方法 1: 子字串檢測
    if s1 in s2 or s2 in s1:
        shorter = min(len(s1), len(s2))
        longer = max(len(s1), len(s2))
        return shorter / longer
    
    # 🎯 方法 2: N-gram 相似度 (更準確)
    def get_ngrams(s, n=2):
        return set(s[i:i+n] for i in range(len(s)-n+1)) if len(s) >= n else {s}
    
    ngrams1 = get_ngrams(s1, 2)
    ngrams2 = get_ngrams(s2, 2)
    
    if not ngrams1 or not ngrams2:
        # fallback to character set
        set1 = set(s1)
        set2 = set(s2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    
    return intersection / union if union > 0 else 0.0

def is_duplicate_or_overlap(text: str) -> bool:
    """檢查文字是否與最近發布的內容重複或高度重疊"""
    global recent_texts, last_transcription
    
    if not text:
        return True
    
    # 檢查是否完全重複
    if text == last_transcription:
        return True
    
    # 檢查是否為子字串
    if text in last_transcription or last_transcription in text:
        # 如果新文字是舊文字的子字串，跳過
        if text in last_transcription:
            return True
        # 如果舊文字是新文字的子字串，計算新增部分
        # 不視為重複，稍後會處理
    
    # 檢查與最近文字的相似度
    for recent in recent_texts:
        similarity = calculate_similarity(text, recent)
        if similarity > SIMILARITY_THRESHOLD:
            return True
    
    return False

def extract_new_content(current: str, previous: str) -> str:
    """提取新內容，移除與前一次重疊的部分"""
    if not previous or not current:
        return current
    
    if current == previous:
        return ""
    
    # 如果前一次是當前的子字串，提取新增部分
    if previous in current:
        idx = current.find(previous)
        if idx == 0:
            # 前綴重複，取後面的新內容
            return current[len(previous):].strip()
        elif idx + len(previous) == len(current):
            # 後綴重複，取前面的新內容
            return current[:idx].strip()
    
    # 檢查前綴重疊
    for i in range(min(len(previous), len(current)), 0, -1):
        if previous[-i:] == current[:i]:
            new_part = current[i:].strip()
            # 只有當新部分有意義時才返回
            if len(new_part) >= 2:
                return new_part
            return ""
    
    # 檢查後綴重疊
    for i in range(min(len(previous), len(current)), 0, -1):
        if previous[:i] == current[-i:]:
            new_part = current[:-i].strip()
            if len(new_part) >= 2:
                return new_part
            return ""
    
    return current

# 🎯 新增：句尾偵測函數
def is_sentence_complete(text: str) -> bool:
    """檢查文字是否為完整句子"""
    if not text:
        return False
    
    # 日文句尾標記
    sentence_endings = [
        '。', '！', '？', '、',  # 日文標點
        'ね', 'よ', 'よね', 'わ', 'か',  # 語氣詞
        'です', 'ます', 'た', 'だ',  # 動詞結尾
        'い', 'いよ', 'いね',  # 形容詞結尾
        '...', '…',  # 省略號
    ]
    
    text = text.strip()
    for ending in sentence_endings:
        if text.endswith(ending):
            return True
    
    # 如果文字超過 15 個字符，可能是完整句子
    if len(text) >= 15:
        return True
    
    return False

# 🎯 新增：合併不完整的句子
def merge_incomplete_sentence(pending: str, new_text: str) -> tuple:
    """合併不完整的句子，返回 (完整句子, 剩餘待處理)"""
    if not pending:
        combined = new_text
    else:
        combined = pending + new_text
    
    if is_sentence_complete(combined):
        return combined, ""
    else:
        return "", combined

# ----------------------------------------------------
# 核心處理函數
# ----------------------------------------------------

def process_audio_chunk(audio_data_b64: str, r):
    """處理音訊塊，使用滑動視窗機制"""
    global audio_buffer, overlap_buffer, last_transcription, last_publish_time
    global recent_texts, pending_text, last_full_sentence
    
    # 解碼音訊
    raw_bytes = base64.b64decode(audio_data_b64)
    
    # 🎯 恢復重疊機制：將重疊緩衝 + 新數據累積
    audio_buffer = overlap_buffer + audio_buffer + raw_bytes
    
    # 計算目標大小
    target_size = int(BUFFER_DURATION_S * SAMPLE_RATE * BYTES_PER_SAMPLE)
    overlap_size = int(OVERLAP_DURATION_S * SAMPLE_RATE * BYTES_PER_SAMPLE)
    
    if len(audio_buffer) < target_size:
        return
    
    # 取出處理的音訊
    audio_to_process = audio_buffer[:target_size]
    
    # 🎯 保留重疊部分供下次使用
    overlap_buffer = audio_buffer[target_size - overlap_size:target_size]
    audio_buffer = audio_buffer[target_size:]
    
    # 轉換為 numpy array
    audio_array = np.frombuffer(audio_to_process, dtype=np.int16).astype(np.float32) / 32768.0
    
    # ASR 轉錄
    text = whisper_asr(audio_array)
    text = filter_text(text)
    
    if not text:
        return
    
    # 🎯 檢查是否與最近內容重複
    if is_duplicate_or_overlap(text):
        return
    
    # 🎯 提取新內容
    text = extract_new_content(text, last_transcription)
    if not text or len(text) < 2:
        return
    
    # 🎯 句子完整性處理
    complete_sentence, pending_text = merge_incomplete_sentence(pending_text, text)
    
    # 如果沒有完整句子，等待更多資料
    if not complete_sentence:
        # 但如果待處理文字太長，強制發布
        if len(pending_text) >= 30:
            complete_sentence = pending_text
            pending_text = ""
        else:
            return
    
    # 檢查發布間隔
    current_time = time.time()
    if current_time - last_publish_time < MIN_PUBLISH_INTERVAL:
        # 間隔太短，將內容加入待處理
        pending_text = complete_sentence + pending_text
        return
    
    # 更新狀態
    last_transcription = complete_sentence
    last_full_sentence = complete_sentence
    last_publish_time = current_time
    recent_texts.append(complete_sentence)
    context_history.append(complete_sentence)
    
    # 翻譯
    translation = executor.submit(google_mt, complete_sentence).result(timeout=5)
    
    # 發布結果
    tz = timezone(timedelta(hours=8))
    result = {
        "timestamp": datetime.now(tz).strftime("%H:%M:%S"),
        "source_lang": SOURCE_LANG_CODE,
        "target_lang": TARGET_LANG_CODE,
        "duration_s": f"{BUFFER_DURATION_S:.3f}",
        "transcription": complete_sentence,
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
    print(f"🎯 stable-ts 整合模式已啟用", file=sys.stderr, flush=True)
    print(f"🎯 VAD: {USE_VAD}, 靜音抑制: {SUPPRESS_SILENCE}", file=sys.stderr, flush=True)

    for msg in p.listen():
        if msg['type'] == 'message':
            process_audio_chunk(msg['data'].decode('utf-8'), r)

if __name__ == "__main__":
    main()