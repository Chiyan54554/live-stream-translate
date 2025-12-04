"""
翻譯模組 - LLM 翻譯與文字轉換
"""
import re
import sys
import asyncio
import aiohttp

from config import LLM_API_URL, LLM_MODEL, LLM_TIMEOUT
from text_utils import remove_inline_repetition, filter_translated_repetition

# === OpenCC 簡繁轉換器 ===
try:
    import opencc
    OPENCC_CONVERTER = opencc.OpenCC('s2twp')
    print(f"✅ OpenCC 簡繁轉換器已載入 (s2twp)", file=sys.stderr, flush=True)
except ImportError:
    OPENCC_CONVERTER = None
    print(f"⚠️ OpenCC 未安裝，將使用備用 txt 字典", file=sys.stderr, flush=True)


def load_simplified_to_traditional() -> dict:
    """從外部 txt 檔案載入簡繁轉換表（備用）"""
    import os
    mapping = {}
    txt_path = os.path.join(os.path.dirname(__file__), 'mappings', 'simplified_to_traditional.txt')
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        simp, trad = parts[0].strip(), parts[1].strip()
                        if simp and trad:
                            mapping[simp] = trad
        if not OPENCC_CONVERTER:
            print(f"✅ 載入備用簡繁轉換表: {len(mapping)} 組", file=sys.stderr, flush=True)
    except FileNotFoundError:
        if not OPENCC_CONVERTER:
            print(f"⚠️ 找不到簡繁轉換表: {txt_path}", file=sys.stderr, flush=True)
    except Exception as e:
        if not OPENCC_CONVERTER:
            print(f"⚠️ 載入簡繁轉換表失敗: {e}", file=sys.stderr, flush=True)
    
    return mapping


def load_china_to_taiwan() -> dict:
    """從外部 txt 檔案載入中國用語轉台灣用語表"""
    import os
    mapping = {}
    txt_path = os.path.join(os.path.dirname(__file__), 'mappings', 'china_to_taiwan.txt')
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        china, taiwan = parts[0].strip(), parts[1].strip()
                        if china and taiwan:
                            mapping[china] = taiwan
        print(f"✅ 載入中台用語表: {len(mapping)} 組", file=sys.stderr, flush=True)
    except FileNotFoundError:
        print(f"⚠️ 找不到中台用語表: {txt_path}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"⚠️ 載入中台用語表失敗: {e}", file=sys.stderr, flush=True)
    
    return mapping


# 全域轉換表
SIMPLIFIED_TO_TRADITIONAL = load_simplified_to_traditional()
CHINA_TO_TAIWAN = load_china_to_taiwan()


def clean_llm_output(text: str) -> str:
    """清理 LLM 輸出的各種問題"""
    if not text:
        return ""
    
    # 1. 過濾羅馬拼音
    romaji_pattern = re.compile(r'^[a-z\s\-\']+$', re.IGNORECASE)
    if romaji_pattern.match(text.strip()) and len(text) > 10:
        print(f"⚠️ 過濾羅馬拼音: {text[:40]}", file=sys.stderr, flush=True)
        return ""
    
    # 2. 移除俄文字符
    if re.search(r'[а-яА-ЯёЁ]', text):
        text = re.sub(r'[а-яА-ЯёЁ]+', '', text)
        print(f"⚠️ 移除俄文字符", file=sys.stderr, flush=True)
    
    # 3. 過濾未翻譯日文
    hiragana_katakana = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF]', text))
    chinese_chars = len(re.findall(r'[\u4E00-\u9FFF]', text))
    if hiragana_katakana > chinese_chars and hiragana_katakana > 5:
        print(f"⚠️ 過濾未翻譯日文: {text[:40]}", file=sys.stderr, flush=True)
        return ""
    
    # 4. 過濾純英文
    if re.match(r'^[a-zA-Z_\s]+$', text.strip()) and len(text) > 5:
        print(f"⚠️ 過濾純英文: {text[:40]}", file=sys.stderr, flush=True)
        return ""
    
    # 移除常見前綴
    prefixes = ['翻譯：', '翻譯:', '中文：', '中文:', '答：', '答:', 
                '繁體中文：', '繁體中文:', '譯文：', '譯文:', '回答：', '回答:']
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    
    # 移除引號包裹
    if len(text) >= 2:
        if (text[0] == '"' and text[-1] == '"') or \
           (text[0] == '「' and text[-1] == '」') or \
           (text[0] == '『' and text[-1] == '』') or \
           (text[0] == "'" and text[-1] == "'"):
            text = text[1:-1].strip()
    
    # 移除奇怪的符號組合
    text = re.sub(r'[,\s]*[}\]]\s*', '', text)
    text = re.sub(r'[:\s]*[)\]>]+\s*[?\s]*$', '', text)
    text = re.sub(r'^[,\s]*[{\[]\s*', '', text)
    text = re.sub(r'[!?]*["\';)]+\s*$', '', text)
    text = re.sub(r'["\';(]+\s*[!?]*\s*$', '', text)
    text = re.sub(r'\s*[!]{2,}["\');\s]*$', '', text)
    text = re.sub(r'的["\'\s.。，,]+$', '的', text)
    text = re.sub(r'你這[.\s]*$', '你這傢伙', text)
    text = re.sub(r'[.\s]+$', '', text)
    
    # 移除開頭結尾的特殊符號
    text = re.sub(r'^[-=_*#]+\s*', '', text)
    text = re.sub(r'\s*[-=_*#]+$', '', text)
    
    # 移除 markdown 格式
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    
    # 移除句中異常的英文片段
    allowed_english = ['K', 'KO', 'OK', 'Combo', 'Gauge', 'Guard', 'Attack', 'Win', 
                       'Lose', 'HP', 'MP', 'SP', 'BGM', 'NG', 'GG', 'VS', 'DLC',
                       'Online', 'Offline', 'S', 'A', 'B', 'C', 'D']
    
    def clean_english(match):
        word = match.group(0)
        if word.upper() in [w.upper() for w in allowed_english] or len(word) <= 2:
            return word
        return ''
    
    text = re.sub(r'\b[a-zA-Z_]{4,}\b', clean_english, text)
    
    # 清理連續重複
    text = remove_inline_repetition(text)
    
    # 簡體轉繁體
    if OPENCC_CONVERTER:
        try:
            text = OPENCC_CONVERTER.convert(text)
        except Exception as e:
            print(f"⚠️ OpenCC 轉換失敗: {e}", file=sys.stderr, flush=True)
            sorted_mappings = sorted(SIMPLIFIED_TO_TRADITIONAL.items(), key=lambda x: len(x[0]), reverse=True)
            for simp, trad in sorted_mappings:
                text = text.replace(simp, trad)
    else:
        sorted_mappings = sorted(SIMPLIFIED_TO_TRADITIONAL.items(), key=lambda x: len(x[0]), reverse=True)
        for simp, trad in sorted_mappings:
            text = text.replace(simp, trad)
    
    # 中國用語 → 台灣用語
    for china, taiwan in CHINA_TO_TAIWAN.items():
        text = text.replace(china, taiwan)
    
    # 移除多餘空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


async def llm_translate(text: str, session: aiohttp.ClientSession) -> str:
    """使用 Ollama Qwen3 LLM 進行日文到繁體中文翻譯"""
    if not text:
        return ""
    
    prompt = f"""你是專業的日文即時直播翻譯員。將以下日文翻譯成自然流暢的繁體中文（台灣用語）。

規則：
- 只輸出翻譯結果
- 保持口語化語氣
- 人名音譯保留日文發音
- 無意義輸入回覆空白

日文：{text}
翻譯："""
    
    try:
        async with session.post(
            LLM_API_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 200,
                    "repeat_penalty": 1.1,
                    "stop": ["\n\n", "日文：", "日文原文", "翻譯："]
                }
            },
            timeout=aiohttp.ClientTimeout(total=LLM_TIMEOUT)
        ) as response:
            if response.status == 200:
                result = await response.json()
                translated = result.get('response', '').strip()
                
                translated = clean_llm_output(translated)
                
                if translated:
                    translated = filter_translated_repetition(translated)
                
                return translated
            else:
                print(f"LLM 翻譯失敗: HTTP {response.status}", file=sys.stderr, flush=True)
                return ""
                
    except asyncio.TimeoutError:
        print(f"LLM 翻譯超時 ({LLM_TIMEOUT}s)", file=sys.stderr, flush=True)
        return ""
    except aiohttp.ClientError as e:
        print(f"無法連接 LLM 服務: {e}", file=sys.stderr, flush=True)
        return ""
    except Exception as e:
        print(f"LLM 翻譯錯誤: {e}", file=sys.stderr, flush=True)
        return ""
