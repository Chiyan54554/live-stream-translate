"""
Live Stream Translate - 日文直播即時翻譯處理器
主程式入口
🚀 優化版：預計算常數、減少重複運算
"""
import sys
import json
import time
import asyncio
import base64
from datetime import datetime, timezone, timedelta
from collections import deque

import numpy as np
import redis.asyncio as aioredis
import aiohttp

from config import (
    REDIS_HOST, REDIS_PORT, AUDIO_CHANNEL, TRANSLATION_CHANNEL,
    SAMPLE_RATE, BYTES_PER_SAMPLE, SOURCE_LANG_CODE, TARGET_LANG_CODE,
    BUFFER_DURATION_S, OVERLAP_DURATION_S,
    MIN_PUBLISH_INTERVAL, SIMILARITY_THRESHOLD,
    USE_VAD, SUPPRESS_SILENCE,
    print_config
)
from asr import setup_environment, init_asr_model, whisper_asr
from translator import llm_translate, warmup_llm
from text_utils import (
    filter_text, calculate_similarity,
    extract_new_content, merge_incomplete_sentence
)


# ============================================================
# 🚀 預計算常數（避免每次處理重新計算）
# ============================================================

# 音訊緩衝區大小（單位：bytes）
TARGET_BUFFER_SIZE = int(BUFFER_DURATION_S * SAMPLE_RATE * BYTES_PER_SAMPLE)
OVERLAP_BUFFER_SIZE = int(OVERLAP_DURATION_S * SAMPLE_RATE * BYTES_PER_SAMPLE)

# 預建立時區物件（避免每次建立）
TZ_TAIPEI = timezone(timedelta(hours=8))

# 預格式化 duration 字串
DURATION_STR = f"{BUFFER_DURATION_S:.3f}"


# === 全域狀態 ===
audio_buffer = b''
overlap_buffer = b''
last_transcription = ""
last_full_sentence = ""
pending_text = ""
last_publish_time = 0
recent_texts = deque(maxlen=15)
context_history = deque(maxlen=8)
pending_translation_task = None
aio_session: aiohttp.ClientSession = None


def is_duplicate_or_overlap(text: str) -> bool:
    """檢查文字是否與最近發布的內容重複或高度重疊 - 優化版"""
    global recent_texts, last_transcription
    
    # 提前返回：空字串或完全相同
    if not text or text == last_transcription:
        return True
    
    # 子字串檢查（先檢查較短的）
    text_len = len(text)
    last_len = len(last_transcription)
    
    if text_len <= last_len:
        if text in last_transcription:
            return True
    elif last_transcription in text:
        pass  # 新文字包含舊文字，可能是擴展，不算重複
    
    # 使用 any() 提前終止
    return any(
        calculate_similarity(text, recent) > SIMILARITY_THRESHOLD
        for recent in recent_texts
    )


async def process_audio_chunk(audio_data_b64: str, r):
    """處理音訊塊，使用滑動視窗機制 + 並行翻譯"""
    global audio_buffer, overlap_buffer, last_transcription, last_publish_time
    global recent_texts, pending_text, last_full_sentence, pending_translation_task
    global aio_session
    
    # 先檢查上一個翻譯任務是否完成
    if pending_translation_task is not None:
        if pending_translation_task.done():
            try:
                result = pending_translation_task.result()
                if result:
                    await r.publish(TRANSLATION_CHANNEL, json.dumps(result, ensure_ascii=False))
            except Exception as e:
                print(f"翻譯任務錯誤: {e}", file=sys.stderr, flush=True)
            pending_translation_task = None
    
    # 解碼音訊
    raw_bytes = base64.b64decode(audio_data_b64)
    
    # 恢復重疊機制
    audio_buffer = overlap_buffer + audio_buffer + raw_bytes
    
    # 使用預計算的常數
    if len(audio_buffer) < TARGET_BUFFER_SIZE:
        return
    
    # 取出處理的音訊
    audio_to_process = audio_buffer[:TARGET_BUFFER_SIZE]
    
    # 保留重疊部分
    overlap_buffer = audio_buffer[TARGET_BUFFER_SIZE - OVERLAP_BUFFER_SIZE:TARGET_BUFFER_SIZE]
    audio_buffer = audio_buffer[TARGET_BUFFER_SIZE:]
    
    # 轉換為 numpy array
    audio_array = np.frombuffer(audio_to_process, dtype=np.int16).astype(np.float32) / 32768.0
    
    # ASR 轉錄
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, whisper_asr, audio_array)
    text = filter_text(text)
    
    if not text:
        return
    
    # 檢查是否與最近內容重複
    if is_duplicate_or_overlap(text):
        return
    
    # 提取新內容
    text = extract_new_content(text, last_transcription)
    if not text or len(text) < 2:
        return
    
    # 句子完整性處理
    complete_sentence, pending_text = merge_incomplete_sentence(pending_text, text)
    
    # 如果沒有完整句子，等待更多資料
    if not complete_sentence:
        if len(pending_text) >= 30:
            complete_sentence = pending_text
            pending_text = ""
        else:
            return
    
    # 檢查發布間隔
    current_time = time.time()
    if current_time - last_publish_time < MIN_PUBLISH_INTERVAL:
        pending_text = complete_sentence + pending_text
        return
    
    # 更新狀態
    last_transcription = complete_sentence
    last_full_sentence = complete_sentence
    last_publish_time = current_time
    recent_texts.append(complete_sentence)
    context_history.append(complete_sentence)
    
    # 並行翻譯
    async def translate_and_prepare_result(text_to_translate: str):
        """翻譯並準備結果 - 優化版"""
        translation = await llm_translate(text_to_translate, aio_session)
        
        # 如果翻譯為空，返回 None 不發布
        if not translation or not translation.strip():
            return None
        
        # 使用預建立的時區和常數
        return {
            "timestamp": datetime.now(TZ_TAIPEI).strftime("%H:%M:%S"),
            "source_lang": SOURCE_LANG_CODE,
            "target_lang": TARGET_LANG_CODE,
            "duration_s": DURATION_STR,
            "transcription": text_to_translate,
            "translation": translation
        }
    
    # 如果有正在進行的翻譯，等待它完成
    if pending_translation_task is not None and not pending_translation_task.done():
        try:
            result = await pending_translation_task
            if result:
                await r.publish(TRANSLATION_CHANNEL, json.dumps(result, ensure_ascii=False))
        except Exception as e:
            print(f"翻譯任務錯誤: {e}", file=sys.stderr, flush=True)
    
    # 啟動新的翻譯任務
    pending_translation_task = asyncio.create_task(translate_and_prepare_result(complete_sentence))


async def main():
    """主循環"""
    global aio_session
    
    import concurrent.futures
    
    # 設定環境
    setup_environment()
    
    # 印出配置
    print_config()
    
    # 建立異步 HTTP session（提前建立）
    aio_session = aiohttp.ClientSession()
    
    # 建立執行緒池
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    
    # 並行初始化：ASR 模型載入 + Redis 連線
    loop = asyncio.get_event_loop()
    
    # 在背景執行 ASR 初始化
    asr_future = loop.run_in_executor(executor, init_asr_model)
    
    # 同時連接 Redis
    try:
        r = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        await r.ping()
        print(f"✅ Redis 連線成功 (異步模式)", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"❌ Redis 連線失敗: {e}", file=sys.stderr, flush=True)
        await aio_session.close()
        sys.exit(1)
    
    # 等待 ASR 初始化完成
    await asr_future
    executor.shutdown(wait=False)
    
    # 🚀 背景預熱 LLM（不阻塞主流程）
    asyncio.create_task(warmup_llm())

    p = r.pubsub()
    await p.subscribe(AUDIO_CHANNEL)
    print(f"✅ 已訂閱: {AUDIO_CHANNEL}", file=sys.stderr, flush=True)
    print(f"🎯 stable-ts 整合模式已啟用 (異步)", file=sys.stderr, flush=True)
    print(f"🎯 VAD: {USE_VAD}, 靜音抑制: {SUPPRESS_SILENCE}", file=sys.stderr, flush=True)

    try:
        # 異步讀取訊息
        async for msg in p.listen():
            if msg['type'] == 'message':
                data = msg['data']
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                await process_audio_chunk(data, r)
    except asyncio.CancelledError:
        print(f"🛑 收到取消信號", file=sys.stderr, flush=True)
    finally:
        # 清理資源
        await p.unsubscribe(AUDIO_CHANNEL)
        await r.close()
        await aio_session.close()
        print(f"✅ 資源已清理", file=sys.stderr, flush=True)


def run():
    """程式入口點"""
    asyncio.run(main())


if __name__ == "__main__":
    run()
