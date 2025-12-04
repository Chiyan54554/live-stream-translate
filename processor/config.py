"""
配置模組 - 所有設定參數
"""
import os
import sys

# === 環境變數處理 ===
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# === 音訊參數 ===
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
SOURCE_LANG_CODE = "ja"
TARGET_LANG_CODE = "zh-TW"

# === 緩衝與品質設定 ===
BUFFER_DURATION_S = 3.0       # 3 秒緩衝，提升 ASR 品質
OVERLAP_DURATION_S = 0.5      # 適度重疊確保連貫性
MIN_AUDIO_ENERGY = 0.005      # 稍低門檻，捕捉更多語音

# === Redis 設定 ===
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
AUDIO_CHANNEL = "audio_feed"
TRANSLATION_CHANNEL = "translation_feed"

# === ASR 模型設定 ===
# - large-v3: 標準 faster-whisper 穩定版
# - kotoba-tech/kotoba-whisper-v2.2: 日文優化 Transformers 版 (最新，支援標點)
# - kotoba-tech/kotoba-whisper-v2.1: 日文優化 Transformers 版 (幻覺更少)
# - kotoba-tech/kotoba-whisper-v2.0-faster: 日文優化 CTranslate2 版 (RTX 50 系列可能不相容)
# ⚠️ 注意：v2.2 沒有提供 faster 版本
ASR_MODEL_NAME = os.getenv('ASR_MODEL_NAME', 'kotoba-tech/kotoba-whisper-v2.2')
MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', '/root/.cache/huggingface/hub')

# 自動判斷模型類型
USE_KOTOBA_PIPELINE = 'kotoba-whisper-v2.1' in ASR_MODEL_NAME or 'kotoba-whisper-v2.2' in ASR_MODEL_NAME

# === LLM 翻譯設定 (Ollama) ===
LLM_HOST = os.getenv('LLM_HOST', 'ollama')
LLM_PORT = os.getenv('LLM_PORT', '11434')
LLM_MODEL = os.getenv('LLM_MODEL', 'qwen3:8b')
LLM_API_URL = f"http://{LLM_HOST}:{LLM_PORT}/api/generate"
LLM_TIMEOUT = 10  # 翻譯超時秒數

# === stable-ts 與 VAD 設定 ===
USE_STABLE_TS = True
USE_VAD = True
VAD_THRESHOLD = 0.45
SUPPRESS_SILENCE = True
HALLUCINATION_SILENCE_TH = 1.5
AVG_PROB_THRESHOLD = -0.7
MAX_INSTANT_WORDS = 0.30
ONLY_VOICE_FREQ = False

# === 發布控制設定 ===
MIN_PUBLISH_INTERVAL = 0.5
SIMILARITY_THRESHOLD = 0.75


def print_config():
    """印出當前配置"""
    print(f"="*50, file=sys.stderr, flush=True)
    print(f"🎯 ASR 模型: {ASR_MODEL_NAME}", file=sys.stderr, flush=True)
    print(f"🎯 使用 Kotoba Pipeline: {USE_KOTOBA_PIPELINE}", file=sys.stderr, flush=True)
    print(f"🎯 LLM 翻譯: {LLM_MODEL} @ {LLM_HOST}:{LLM_PORT}", file=sys.stderr, flush=True)
    print(f"🎯 stable-ts: {USE_STABLE_TS}, VAD: {USE_VAD}", file=sys.stderr, flush=True)
    print(f"="*50, file=sys.stderr, flush=True)
