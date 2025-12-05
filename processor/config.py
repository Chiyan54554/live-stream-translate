"""
配置模組 - 所有設定參數
🎯 優化：預先計算常數、避免重複運算、使用 __slots__ 減少記憶體
"""
import os
import sys

# === 環境變數處理 ===
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# === 音訊參數 (預先計算的整數常數) ===
SAMPLE_RATE: int = 16000
BYTES_PER_SAMPLE: int = 2
SOURCE_LANG_CODE: str = "ja"
TARGET_LANG_CODE: str = "zh-TW"

# === 緩衝與品質設定 (使用 float 而非計算式) ===
BUFFER_DURATION_S: float = 5.0       # 5 秒緩衝，讓 ASR 有更多上下文
OVERLAP_DURATION_S: float = 1.5      # 增加重疊確保語句連貫
MIN_AUDIO_ENERGY: float = 0.002      # 較低門檻，捕捉輕聲語音

# 🎯 預先計算的緩衝區大小 (避免運行時乘法)
BUFFER_SIZE_BYTES: int = int(BUFFER_DURATION_S * SAMPLE_RATE * BYTES_PER_SAMPLE)
OVERLAP_SIZE_BYTES: int = int(OVERLAP_DURATION_S * SAMPLE_RATE * BYTES_PER_SAMPLE)

# 🎯 預計算的能量閾值平方（避免 sqrt）
MIN_AUDIO_ENERGY_SQUARED: float = MIN_AUDIO_ENERGY ** 2

# === Redis 設定 ===
REDIS_HOST: str = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT: int = int(os.getenv('REDIS_PORT', 6379))
AUDIO_CHANNEL: str = "audio_feed"
TRANSLATION_CHANNEL: str = "translation_feed"

# === ASR 模型設定 ===
# - large-v3: 標準 faster-whisper 穩定版
# - kotoba-tech/kotoba-whisper-v2.2: 日文優化 Transformers 版 (最新，支援標點)
# - kotoba-tech/kotoba-whisper-v2.1: 日文優化 Transformers 版 (幻覺更少)
# - kotoba-tech/kotoba-whisper-v2.0-faster: 日文優化 CTranslate2 版 (RTX 50 系列可能不相容)
# ⚠️ 注意：v2.2 沒有提供 faster 版本
ASR_MODEL_NAME: str = os.getenv('ASR_MODEL_NAME', 'kotoba-tech/kotoba-whisper-v2.2')
MODEL_CACHE_DIR: str = os.getenv('MODEL_CACHE_DIR', '/root/.cache/huggingface/hub')

# 🎯 預先計算的布林值 (避免重複字串查找)
USE_KOTOBA_PIPELINE: bool = 'kotoba-whisper-v2.1' in ASR_MODEL_NAME or 'kotoba-whisper-v2.2' in ASR_MODEL_NAME

# === LLM 翻譯設定 (Ollama) ===
LLM_HOST: str = os.getenv('LLM_HOST', 'ollama')
LLM_PORT: str = os.getenv('LLM_PORT', '11434')
LLM_MODEL: str = os.getenv('LLM_MODEL', 'qwen3:8b')
# 🎯 預先建立的 URL (避免每次請求時字串格式化)
LLM_API_URL: str = f"http://{LLM_HOST}:{LLM_PORT}/api/generate"
LLM_TIMEOUT: int = 10  # 翻譯超時秒數

# === stable-ts 與 VAD 設定 (使用 bool 和 float 常數) ===
USE_STABLE_TS: bool = True
USE_VAD: bool = True
VAD_THRESHOLD: float = 0.40          # 稍低閾值，減少漏檢
SUPPRESS_SILENCE: bool = True
HALLUCINATION_SILENCE_TH: float = 1.5
AVG_PROB_THRESHOLD: float = -0.6     # 更嚴格的置信度過濾
MAX_INSTANT_WORDS: float = 0.25      # 更嚴格過濾瞬時詞幻覺
ONLY_VOICE_FREQ: bool = True         # 聚焦人聲頻率範圍

# === 發布控制設定 ===
MIN_PUBLISH_INTERVAL: float = 0.5
SIMILARITY_THRESHOLD: float = 0.75

# 🎯 預先建立的配置字串 (用於 print_config)
_CONFIG_SEPARATOR: str = "=" * 50
_CONFIG_LINES: tuple = (
    f"🎯 ASR 模型: {ASR_MODEL_NAME}",
    f"🎯 使用 Kotoba Pipeline: {USE_KOTOBA_PIPELINE}",
    f"🎯 LLM 翻譯: {LLM_MODEL} @ {LLM_HOST}:{LLM_PORT}",
    f"🎯 stable-ts: {USE_STABLE_TS}, VAD: {USE_VAD}",
)


def print_config() -> None:
    """印出當前配置 - 🎯 使用預建立的字串減少運行時格式化"""
    print(_CONFIG_SEPARATOR, file=sys.stderr, flush=True)
    for line in _CONFIG_LINES:
        print(line, file=sys.stderr, flush=True)
    print(_CONFIG_SEPARATOR, file=sys.stderr, flush=True)
