"""
翻譯模組 - LLM 翻譯與文字轉換
🚀 優化版：預編譯正則、高效資料結構、減少重複運算
"""
import re
import sys
import asyncio
import aiohttp
import time
import os

from config import (
    LLM_API_URL,
    LLM_MODEL,
    LLM_TIMEOUT,
    USE_CLOUD_TRANSLATION,
    CLOUD_TRANSLATE_PROJECT_ID,
    CLOUD_TRANSLATE_LOCATION,
    CLOUD_TRANSLATE_TIMEOUT,
    TARGET_LANG_CODE,
    SOURCE_LANG_CODE,
)

# 🚀 全域狀態：追蹤 LLM 是否就緒
_llm_ready = False
_llm_warmup_done = False
_translate_client = None
_translate_parent = None
_cloud_disabled = False  # 避免重複 403 造成刷屏

try:
    from google.cloud import translate
except Exception:
    translate = None

from text_utils import (
    remove_inline_repetition, 
    filter_translated_repetition, 
    clean_gibberish_from_translation,
    RE_REPEATED_WORDS,
    RE_STUTTERING,
    RE_J_PREFIX_HALLUCINATION
)

# ============================================================
# 🚀 預編譯正則表達式（模組載入時只編譯一次）
# ============================================================

RE_ROMAJI = re.compile(r'^[a-z\s\-\']+$', re.IGNORECASE)
RE_RUSSIAN = re.compile(r'[а-яА-ЯёЁ]+')
RE_NON_TARGET_LANG = re.compile(r'[\u0600-\u06FF\u0590-\u05FF\u0E00-\u0E7F\u0900-\u097F\uAC00-\uD7AF]+')
RE_BOPOMOFO = re.compile(r'[\u3100-\u312F]+')
RE_TRAILING_DIGITS = re.compile(r'[\s]*[0-9]+[\s]*$')
RE_HIRAGANA_KATAKANA = re.compile(r'[\u3040-\u309F\u30A0-\u30FF]')
RE_CHINESE_CHARS = re.compile(r'[\u4E00-\u9FFF]')
RE_CONSECUTIVE_KANA = re.compile(r'[\u3040-\u309F\u30A0-\u30FF]{3,}')
RE_KANJI_KANA_PUNCT = re.compile(r'[\u4E00-\u9FFF][\u3040-\u309F\u30A0-\u30FF]+[？！。]?')
RE_PURE_ENGLISH = re.compile(r'^[a-zA-Z_\s]+$')
RE_ENGLISH_WORD = re.compile(r'\b[a-zA-Z_]{4,}\b')
RE_MULTI_SPACE = re.compile(r'\s+')
RE_MARKDOWN_BOLD = re.compile(r'\*\*(.+?)\*\*')
RE_MARKDOWN_UNDERLINE = re.compile(r'__(.+?)__')
RE_MARKDOWN_CODE = re.compile(r'`(.+?)`')

# 🎯 低品質翻譯過濾（語意不通順模式）
RE_NONSENSE_PATTERN = re.compile(r'(.{1,2})\1{4,}')  # 連續重複 1-2 字 5 次以上
RE_INCOMPLETE_ENDING = re.compile(r'[的在是了和要]$')  # 不完整結尾

# 符號清理（預編譯列表）
RE_SYMBOL_CLEANUP = (
    (re.compile(r'[,\s]*[}\]]\s*'), ''),
    (re.compile(r'[:\s]*[)\]>]+\s*[?\s]*$'), ''),
    (re.compile(r'^[,\s]*[{\[]\s*'), ''),
    (re.compile(r'[!?]*["\';)]+\s*$'), ''),
    (re.compile(r'["\';(]+\s*[!?]*\s*$'), ''),
    (re.compile(r'\s*[!]{2,}["\');\s]*$'), ''),
    (re.compile(r'的["\'\s.。，,]+$'), '的'),
    (re.compile(r'你這[.\s]*$'), '你這傢伙'),
    (re.compile(r'[.\s]+$'), ''),
    (re.compile(r'^[-=_*#]+\s*'), ''),
    (re.compile(r'\s*[-=_*#]+$'), ''),
)

# ============================================================
# 🚀 高效資料結構（frozenset O(1) 查找）
# ============================================================

ALLOWED_ENGLISH_UPPER = frozenset({
    'K', 'KO', 'OK', 'COMBO', 'GAUGE', 'GUARD', 'ATTACK', 'WIN',
    'LOSE', 'HP', 'MP', 'SP', 'BGM', 'NG', 'GG', 'VS', 'DLC',
    'ONLINE', 'OFFLINE', 'S', 'A', 'B', 'C', 'D'
})

PREFIXES_TO_REMOVE = (
    '翻譯：', '翻譯:', '中文：', '中文:', '答：', '答:',
    '繁體中文：', '繁體中文:', '譯文：', '譯文:', '回答：', '回答:'
)

QUOTE_PAIRS = (
    ('"', '"'), ('「', '」'), ('『', '』'), ("'", "'"),
)

# ============================================================
# 🚀 轉換表載入與預處理
# ============================================================

def _load_mapping(filename: str, description: str) -> dict:
    """從檔案載入映射表"""
    mapping = {}
    txt_path = os.path.join(os.path.dirname(__file__), 'mappings', filename)
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        key, value = parts[0].strip(), parts[1].strip()
                        if key and value:
                            mapping[key] = value
        print(f"✅ 載入{description}: {len(mapping)} 組", file=sys.stderr, flush=True)
    except FileNotFoundError:
        print(f"⚠️ 找不到{description}: {txt_path}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"⚠️ 載入{description}失敗: {e}", file=sys.stderr, flush=True)
    return mapping

# === OpenCC 簡繁轉換器 ===
try:
    import opencc
    OPENCC_CONVERTER = opencc.OpenCC('s2twp')
    print(f"✅ OpenCC 簡繁轉換器已載入 (s2twp)", file=sys.stderr, flush=True)
except ImportError:
    OPENCC_CONVERTER = None
    print(f"⚠️ OpenCC 未安裝，將使用備用 txt 字典", file=sys.stderr, flush=True)

# 載入並預排序轉換表（只排序一次）
_s2t_raw = _load_mapping('simplified_to_traditional.txt', '簡繁轉換表') if not OPENCC_CONVERTER else {}
_c2t_raw = _load_mapping('china_to_taiwan.txt', '中台用語表')

SIMPLIFIED_TO_TRADITIONAL_SORTED = tuple(
    sorted(_s2t_raw.items(), key=lambda x: len(x[0]), reverse=True)
) if _s2t_raw else ()

CHINA_TO_TAIWAN_SORTED = tuple(
    sorted(_c2t_raw.items(), key=lambda x: len(x[0]), reverse=True)
)


async def warmup_llm():
    """🚀 非阻塞 LLM 預熱 - 背景等待 Ollama 就緒"""
    global _llm_ready, _llm_warmup_done

    if _llm_warmup_done:
        return _llm_ready

    if USE_CLOUD_TRANSLATION:
        _llm_ready = True
        _llm_warmup_done = True
        print("🌐 使用 Cloud Translation，跳過 LLM 預熱", file=sys.stderr, flush=True)
        return True

    async def _ensure_llm_model(session: aiohttp.ClientSession) -> bool:
        """檢查模型是否存在；不存在時觸發拉取以避免 404"""
        try:
            async with session.post(
                f"{LLM_API_URL.replace('/api/generate','')}/api/show",
                json={"model": LLM_MODEL},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    return True
                if resp.status != 404:
                    print(f"⚠️ 模型檢查失敗: HTTP {resp.status}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"⚠️ 模型檢查錯誤: {e}", file=sys.stderr, flush=True)
            # 繼續嘗試拉取
        # 嘗試拉取模型
        print(f"🔄 自動拉取模型 {LLM_MODEL} ...", file=sys.stderr, flush=True)
        try:
            async with session.post(
                f"{LLM_API_URL.replace('/api/generate','')}/api/pull",
                json={"model": LLM_MODEL, "stream": False},
                timeout=aiohttp.ClientTimeout(total=900)
            ) as resp:
                if resp.status == 200:
                    print(f"✅ 已拉取模型 {LLM_MODEL}", file=sys.stderr, flush=True)
                    return True
                else:
                    print(f"⚠️ 拉取模型失敗: HTTP {resp.status}", file=sys.stderr, flush=True)
                    return False
        except Exception as e:
            print(f"⚠️ 拉取模型錯誤: {e}", file=sys.stderr, flush=True)
            return False

    print("🔄 背景等待 Ollama 模型載入...", file=sys.stderr, flush=True)
    start_time = time.time()
    max_wait = 300  # 最多等待 5 分鐘（首次載入模型到 GPU 需要時間）
    
    async with aiohttp.ClientSession() as session:
        model_ready = await _ensure_llm_model(session)
        if not model_ready:
            print("⚠️ 模型不存在且拉取失敗，請檢查模型名稱或網路", file=sys.stderr, flush=True)
            _llm_warmup_done = True
            _llm_ready = False
            return False

        while time.time() - start_time < max_wait:
            try:
                # 🚀 首次請求需要 60 秒以上（模型載入到 GPU）
                async with session.post(
                    LLM_API_URL,
                    json={
                        "model": LLM_MODEL,
                        "prompt": "你好",
                        "stream": False,
                        "think": False,
                        "options": {"num_predict": 5}
                    },
                    timeout=aiohttp.ClientTimeout(total=120)  # 120 秒超時
                ) as response:
                    if response.status == 200:
                        elapsed = time.time() - start_time
                        print(f"✅ LLM 就緒！(載入耗時 {elapsed:.1f}s)", file=sys.stderr, flush=True)
                        _llm_ready = True
                        _llm_warmup_done = True
                        return True
                    elif response.status == 499:
                        # 499 = 請求被取消（模型正在載入）
                        print(f"⏳ 模型載入中... ({time.time() - start_time:.0f}s)", file=sys.stderr, flush=True)
                        await asyncio.sleep(5)
                    else:
                        print(f"⚠️ LLM 回應: {response.status}", file=sys.stderr, flush=True)
                        await asyncio.sleep(5)
            except asyncio.TimeoutError:
                print(f"⏳ 等待模型載入... ({time.time() - start_time:.0f}s)", file=sys.stderr, flush=True)
                await asyncio.sleep(3)
            except aiohttp.ClientError:
                # Ollama 服務還沒啟動
                await asyncio.sleep(3)
            except Exception as e:
                print(f"⚠️ 預熱錯誤: {e}", file=sys.stderr, flush=True)
                await asyncio.sleep(5)
        
        print("⚠️ LLM 預熱超時，翻譯功能可能受影響", file=sys.stderr, flush=True)
        _llm_warmup_done = True
        # 即使超時也標記為就緒，讓翻譯可以嘗試
        _llm_ready = True
        return False


def is_llm_ready() -> bool:
    """檢查 LLM 是否已就緒"""
    return _llm_ready


# ============================================================
# 🚀 優化版清理函數
# ============================================================

def _clean_english_word(match) -> str:
    """清理英文詞（O(1) frozenset 查找）"""
    word = match.group(0)
    if word.upper() in ALLOWED_ENGLISH_UPPER or len(word) <= 2:
        return word
    return ''


def clean_llm_output(text: str) -> str:
    """清理 LLM 輸出 - 優化版（預編譯正則 + 高效資料結構）"""
    if not text:
        return ""
    
    text_stripped = text.strip()
    
    # 1. 過濾羅馬拼音
    if RE_ROMAJI.match(text_stripped) and len(text) > 10:
        print(f"⚠️ 過濾羅馬拼音: {text[:40]}", file=sys.stderr, flush=True)
        return ""
    
    # 2. 移除非目標語言字符
    if RE_RUSSIAN.search(text):
        text = RE_RUSSIAN.sub('', text)
        print(f"⚠️ 移除俄文字符", file=sys.stderr, flush=True)
    
    if RE_NON_TARGET_LANG.search(text):
        text = RE_NON_TARGET_LANG.sub('', text)
        print(f"⚠️ 移除非目標語言字符", file=sys.stderr, flush=True)
    
    if RE_BOPOMOFO.search(text):
        text = RE_BOPOMOFO.sub('', text)
        print(f"⚠️ 移除注音符號", file=sys.stderr, flush=True)
    
    text = RE_TRAILING_DIGITS.sub('', text)
    
    # 3. 日文假名過濾
    hiragana_katakana = len(RE_HIRAGANA_KATAKANA.findall(text))
    chinese_chars = len(RE_CHINESE_CHARS.findall(text))
    
    if hiragana_katakana > chinese_chars and hiragana_katakana > 5:
        print(f"⚠️ 過濾未翻譯日文: {text[:40]}", file=sys.stderr, flush=True)
        return ""
    
    def _clean_kana_fragment(match):
        fragment = match.group(0)
        if len(fragment) <= 2:
            return fragment
        print(f"⚠️ 移除日文片段: {fragment}", file=sys.stderr, flush=True)
        return ''
    
    text = RE_CONSECUTIVE_KANA.sub(_clean_kana_fragment, text)
    
    def _clean_kanji_kana(match):
        m_text = match.group(0)
        if len(RE_HIRAGANA_KATAKANA.findall(m_text)) >= 2:
            return ''
        return m_text
    
    text = RE_KANJI_KANA_PUNCT.sub(_clean_kanji_kana, text)
    
    # 4. 過濾純英文
    if RE_PURE_ENGLISH.match(text.strip()) and len(text) > 5:
        print(f"⚠️ 過濾純英文: {text[:40]}", file=sys.stderr, flush=True)
        return ""
    
    # 5. 移除前綴（tuple 迭代）
    for prefix in PREFIXES_TO_REMOVE:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    
    # 6. 移除引號包裹
    if len(text) >= 2:
        for open_q, close_q in QUOTE_PAIRS:
            if text[0] == open_q and text[-1] == close_q:
                text = text[1:-1].strip()
                break
    
    # 7. 批次符號清理
    for pattern, replacement in RE_SYMBOL_CLEANUP:
        text = pattern.sub(replacement, text)
    
    # 8. 移除 Markdown
    text = RE_MARKDOWN_BOLD.sub(r'\1', text)
    text = RE_MARKDOWN_UNDERLINE.sub(r'\1', text)
    text = RE_MARKDOWN_CODE.sub(r'\1', text)
    
    # 9. 移除異常英文
    text = RE_ENGLISH_WORD.sub(_clean_english_word, text)
    
    # 10. 清理連續重複
    text = remove_inline_repetition(text)
    
    # 11. 簡體轉繁體
    if OPENCC_CONVERTER:
        try:
            text = OPENCC_CONVERTER.convert(text)
        except Exception as e:
            print(f"⚠️ OpenCC 轉換失敗: {e}", file=sys.stderr, flush=True)
            for simp, trad in SIMPLIFIED_TO_TRADITIONAL_SORTED:
                text = text.replace(simp, trad)
    elif SIMPLIFIED_TO_TRADITIONAL_SORTED:
        for simp, trad in SIMPLIFIED_TO_TRADITIONAL_SORTED:
            text = text.replace(simp, trad)
    
    # 12. 中國用語 → 台灣用語（預排序 tuple）
    for china, taiwan in CHINA_TO_TAIWAN_SORTED:
        text = text.replace(china, taiwan)
    
    # 13. 🎯 過濾低品質翻譯
    # 過濾：是想要是想要要回來想要呢大隻的
    if RE_REPEATED_WORDS.search(text):
        print(f"⚠️ 過濾低品質翻譯（重複詞）: {text[:40]}", file=sys.stderr, flush=True)
        return ""
    
    # 過濾：快快魔加丁要不要要不要啦
    if RE_NONSENSE_PATTERN.search(text):
        match = RE_NONSENSE_PATTERN.search(text)
        pattern = match.group(1)
        # 嘗試修復：只保留一次
        fixed = RE_NONSENSE_PATTERN.sub(pattern, text)
        if fixed != text and len(fixed) >= 4:
            print(f"🔧 修復重複模式: {text[:40]} -> {fixed[:40]}", file=sys.stderr, flush=True)
            text = fixed
        else:
            print(f"⚠️ 過濾無意義重複: {text[:40]}", file=sys.stderr, flush=True)
            return ""
    
    # 過濾不完整的句子（但不過濾太短的）
    if len(text) >= 8 and RE_INCOMPLETE_ENDING.search(text):
        # 嘗試移除不完整結尾
        cleaned = RE_INCOMPLETE_ENDING.sub('', text).strip()
        if len(cleaned) >= 4:
            print(f"🔧 移除不完整結尾: {text[:40]} -> {cleaned[:40]}", file=sys.stderr, flush=True)
            text = cleaned
    
    # 14. 移除多餘空格
    text = RE_MULTI_SPACE.sub(' ', text).strip()
    
    return text


# ============================================================
# 🌐 Cloud Translation (Google)
# ============================================================

def _get_translate_client():
    """建立或回傳共用的 Cloud Translation client"""
    global _translate_client, _translate_parent
    if _translate_client:
        return _translate_client
    if not translate:
        print("⚠️ 未安裝 google-cloud-translate，無法使用 Cloud Translation", file=sys.stderr, flush=True)
        return None
    if not CLOUD_TRANSLATE_PROJECT_ID:
        print("⚠️ 未設定 CLOUD_TRANSLATE_PROJECT_ID，無法使用 Cloud Translation", file=sys.stderr, flush=True)
        return None
    if not re.match(r"^[a-z][a-z0-9-]*$", CLOUD_TRANSLATE_PROJECT_ID):
        print(f"⚠️ CLOUD_TRANSLATE_PROJECT_ID 格式無效: {CLOUD_TRANSLATE_PROJECT_ID}", file=sys.stderr, flush=True)
        return None
    try:
        _translate_client = translate.TranslationServiceClient()
        _translate_parent = f"projects/{CLOUD_TRANSLATE_PROJECT_ID}/locations/{CLOUD_TRANSLATE_LOCATION}"
    except Exception as e:
        print(f"⚠️ 建立 Cloud Translation client 失敗: {e}", file=sys.stderr, flush=True)
        _translate_client = None
    return _translate_client


def _cloud_translate_sync(text: str) -> str:
    global _cloud_disabled
    if _cloud_disabled:
        return ""
    client = _get_translate_client()
    if client is None:
        return ""
    if not _translate_parent:
        return ""
    try:
        response = client.translate_text(
            request={
                "parent": _translate_parent,
                "contents": [text],
                "mime_type": "text/plain",
                "source_language_code": SOURCE_LANG_CODE,
                "target_language_code": TARGET_LANG_CODE,
            },
            timeout=CLOUD_TRANSLATE_TIMEOUT,
        )
        if response.translations:
            return response.translations[0].translated_text
    except Exception as e:
        msg = str(e)
        print(f"⚠️ Cloud Translation 失敗: {msg}", file=sys.stderr, flush=True)
        if "cloudtranslate.generalModels.predict" in msg or "403" in msg:
            print("⚠️ 偵測到權限不足，暫停 Cloud Translation，請為 service account 加上 Cloud Translation API User 角色", file=sys.stderr, flush=True)
            _cloud_disabled = True
    return ""


async def _translate_with_cloud(text: str) -> str:
    if not text:
        return ""
    loop = asyncio.get_event_loop()
    translated = await loop.run_in_executor(None, _cloud_translate_sync, text)
    if translated:
        translated = clean_llm_output(translated)
    if translated:
        translated = filter_translated_repetition(translated)
    if translated:
        translated = clean_gibberish_from_translation(translated)
    return translated


# ============================================================
# 🚀 LLM 翻譯（預建立模板）
# ============================================================

_PROMPT_TEMPLATE = """你是專業的日文遊戲直播即時翻譯員。請將以下日文準確翻譯成繁體中文（台灣用語）。

翻譯規則：
1. 只輸出翻譯結果，不要解釋或註解
2. 保持口語化、自然的語氣
3. 人名音譯：用常見中文譯法（如ヒロ→阿廣、タケシ→阿武、さん→桑/先生）
4. 遊戲術語：使用台灣玩家慣用譯法
5. 片假名外來語：翻成中文意思，不要音譯
6. 語氣詞保留自然感（如：啊、呢、啦、欸）
7. 聽不清或無意義的輸入，回覆空白

日文：{text}
中文："""

_REQUEST_OPTIONS = {
    "temperature": 0.2,
    "top_p": 0.85,
    "top_k": 30,
    "num_predict": 256,
    "repeat_penalty": 1.15,
    "stop": ["\n\n", "日文：", "日文原文", "中文：", "翻譯："]
}


async def llm_translate(text: str, session: aiohttp.ClientSession) -> str:
    """使用 Ollama Qwen3 LLM 進行日文到繁體中文翻譯"""
    if not text:
        return ""

    if USE_CLOUD_TRANSLATION and not _cloud_disabled:
        translated = await _translate_with_cloud(text)
        if translated:
            return translated
        print("⚠️ Cloud Translation 失敗，改用 Ollama 備援", file=sys.stderr, flush=True)
    
    if not _llm_ready and not _llm_warmup_done:
        return ""
    
    prompt = _PROMPT_TEMPLATE.format(text=text)
    request_body = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": _REQUEST_OPTIONS
    }
    
    max_retries = 2
    retry_timeout = LLM_TIMEOUT
    
    for attempt in range(max_retries + 1):
        try:
            current_timeout = retry_timeout * 3 if attempt == 0 else retry_timeout
            
            async with session.post(
                LLM_API_URL,
                json=request_body,
                timeout=aiohttp.ClientTimeout(total=current_timeout)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    translated = result.get('response', '').strip()
                    translated = clean_llm_output(translated)
                    if translated:
                        translated = filter_translated_repetition(translated)
                    if translated:
                        translated = clean_gibberish_from_translation(translated)
                    return translated
                else:
                    print(f"LLM 翻譯失敗: HTTP {response.status}", file=sys.stderr, flush=True)
                    if attempt < max_retries:
                        await asyncio.sleep(0.5)
                        continue
                    return ""
                    
        except asyncio.TimeoutError:
            if attempt < max_retries:
                print(f"LLM 超時，重試 ({attempt + 1}/{max_retries})...", file=sys.stderr, flush=True)
                await asyncio.sleep(0.5)
                continue
            print(f"LLM 翻譯超時 ({LLM_TIMEOUT}s)", file=sys.stderr, flush=True)
            return ""
        except aiohttp.ClientError as e:
            if attempt < max_retries:
                print(f"LLM 連線失敗，重試 ({attempt + 1}/{max_retries})...", file=sys.stderr, flush=True)
                await asyncio.sleep(1)
                continue
            print(f"無法連接 LLM 服務: {e}", file=sys.stderr, flush=True)
            return ""
        except Exception as e:
            print(f"LLM 翻譯錯誤: {e}", file=sys.stderr, flush=True)
            return ""
    
    return ""
