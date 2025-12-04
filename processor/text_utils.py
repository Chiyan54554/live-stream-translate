"""
文字處理模組 - 過濾、去重、清理
🚀 優化版：預編譯正則、高效資料結構、減少重複運算
"""
import re
import sys
from collections import Counter
from functools import lru_cache

# ============================================================
# 🚀 預編譯正則表達式（模組載入時只編譯一次）
# ============================================================

RE_JAPANESE_CHARS = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uFF00-\uFFEF\u0020-\u007E]+')
RE_PUNCTUATION_CLEANUP = re.compile(r'[，。！？]{2,}')
RE_GIBBERISH_PUNCT = re.compile(r'[，。！？、～~\s]+')

# 標點符號集合（O(1) 查找）
PUNCTUATION_SET = frozenset('，。！？、 ～~')
PUNCTUATION_ONLY_SET = frozenset('、，。！？　 ・')
SENTENCE_ENDINGS_SET = frozenset(('。', '！', '？', '!', '?'))

# 幻覺過濾列表（frozenset O(1) 查找）
UNWANTED_PHRASES = frozenset({
    "[音声なし]", "ご視聴ありがとう", "最後までご視聴",
    "(拍手)", "(笑い)", "(ため息)", "字幕",
    "チャンネル登録", "高評価", "MBSニュース",
    "提供は", "ご覧いただき", "ありがとうございました",
    "お疲れ様でした", "また会いましょう", "バイバイ",
    "次回も", "チャンネル", "登録", "お願いします",
    "♪", "BGM", "音楽", "エンディング",
    "テロップ", "ナレーション", "アナウンス",
    "話し言葉", "カジュアルな表現", "ネットスラング",
    "VTuber配信", "配信者とリスナー", "日本語の",
    "翻訳", "字幕提供", "自動生成", "機械翻訳",
    "続きは", "詳しくは", "リンクは",
    "概要欄", "説明欄", "コメント欄",
})

# 🎯 ASR 幻覺模式（正則匹配）
RE_ASR_HALLUCINATION = re.compile(
    r'^[JKLMNOPQRSTUVWXYZＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ][\u4e00-\u9fff]+'
    r'[JKLMNOPQRSTUVWXYZＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ]'  # J愛心J 模式
)
RE_BROKEN_PATTERN = re.compile(r'([^\s]{1,3}[JKL][^\s]{1,3}){2,}')  # 重複 J/K/L 模式

# 🎯 過濾「J+中文」開頭的翻譯（ASR 常見幻覺）
RE_J_PREFIX_HALLUCINATION = re.compile(r'^[JKLMN]\s*[\u4e00-\u9fff]')

# 🎯 翻譯品質檢測（重複詞模式）
RE_REPEATED_WORDS = re.compile(r'(要不要|想要|是不是|有沒有|可不可以){3,}')
RE_STUTTERING = re.compile(r'(.{2,4})\1{2,}')  # 連續重複 2-4 字 3次以上

# 句尾結束詞（tuple 較快迭代）
SENTENCE_ENDINGS = (
    '。', '！', '？', '、',
    'ね', 'よ', 'よね', 'わ', 'か',
    'です', 'ます', 'た', 'だ',
    'い', 'いよ', 'いね',
    '...', '…',
)

# 同義詞群組（預編譯正則）
SIMILAR_GROUPS_PATTERNS = tuple(
    (re.compile('(' + '|'.join(re.escape(w) for w in group) + ')'), group[0])
    for group in (
        ('右邊', '右側', '右面'),
        ('左邊', '左側', '左面'),
        ('上面', '上邊', '上方'),
        ('下面', '下邊', '下方'),
        ('前面', '前邊', '前方'),
        ('後面', '後邊', '後方'),
        ('這邊', '這裡', '這兒'),
        ('那邊', '那裡', '那兒'),
    )
)

# 有效重複模式（用於 detect_character_repetition）
VALID_REPEAT_PATTERNS = frozenset({'ww', 'ーー', '...', '！！', '？？', '〜〜'})

# 常見有意義的重複（不應被過濾）
COMMON_VALID_REPEATS = frozenset({'哈', '呵', '嘿', '嗯', '啊', '欸', '喔', '噢', '耶', '唉', '嘻', '笑'})

# 音譯常用字（frozenset O(1) 查找）
TRANSLITERATION_CHARS = frozenset(
    '巴托斯拉達馬卡帕塔瓦薩納拉莫諾洛羅波索佐多科戈伊尼里基米希'
    '克德特爾布格恩姆師夫吉斯安列文茲許勒蒂娜雅該赫阿卡'
    '梅泰克尼德吉雷戈菲米森尼克维克卡州'
)

# 排除的常用字（frozenset O(1) 查找）
COMMON_CHARS_EXCLUDE = frozenset('的是了在有我你他她它們這那不也就都而且但只要如果因為所以還可以很太真好壞')


@lru_cache(maxsize=256)
def _get_bigrams(s: str) -> frozenset:
    """取得字串的 bigram 集合（快取結果）"""
    if len(s) < 2:
        return frozenset({s})
    return frozenset(s[i:i+2] for i in range(len(s)-1))


def calculate_similarity(s1: str, s2: str) -> float:
    """計算兩個字串的相似度 (0-1) - 優化版"""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    
    len1, len2 = len(s1), len(s2)
    
    # 方法 1: 子字串檢測（先檢查較短的）
    if len1 <= len2:
        if s1 in s2:
            return len1 / len2
    else:
        if s2 in s1:
            return len2 / len1
    
    # 方法 2: Bigram Jaccard 相似度（使用快取）
    bigrams1 = _get_bigrams(s1)
    bigrams2 = _get_bigrams(s2)
    
    intersection = len(bigrams1 & bigrams2)
    union = len(bigrams1 | bigrams2)
    
    return intersection / union if union > 0 else 0.0


def remove_inline_repetition(text: str) -> str:
    """移除句中連續重複的片段 - 優化版（預編譯正則）"""
    if not text or len(text) < 8:
        return text
    
    original = text
    
    # 方法 0: 偵測連續相似詞（使用預編譯正則）
    for pattern, replacement in SIMILAR_GROUPS_PATTERNS:
        matches = pattern.findall(text)
        if len(matches) >= 3:
            # 移除連續出現的同義詞（保留第一個）
            result = pattern.sub(lambda m, c=[0]: (c.__setitem__(0, c[0]+1), m.group(0) if c[0] == 1 else '')[1], text)
            if result != text:
                print(f"🔧 移除同義詞重複: {text[:40]} -> {result[:40]}", file=sys.stderr, flush=True)
                text = result
                original = result
    
    # 方法 1: 偵測完全相同的連續重複（優化：從大到小，找到即返回）
    text_len = len(text)
    max_pattern_len = min(25, text_len // 2)
    
    for pattern_len in range(max_pattern_len, 3, -1):
        search_range = text_len - pattern_len * 2 + 1
        for start in range(search_range):
            pattern = text[start:start + pattern_len]
            
            # 使用 frozenset 快速檢查（O(1)）
            if all(c in PUNCTUATION_SET for c in pattern):
                continue
            
            repeat_pos = start + pattern_len
            if text[repeat_pos:repeat_pos + pattern_len] == pattern:
                # 計算重複次數
                count = 2
                check_pos = repeat_pos + pattern_len
                while check_pos + pattern_len <= text_len and text[check_pos:check_pos + pattern_len] == pattern:
                    count += 1
                    check_pos += pattern_len
                
                # 直接拼接結果
                result = (text[:start] + pattern + text[start + pattern_len * count:]).strip()
                
                if result != original:
                    print(f"🔧 移除行內重複: {original[:40]} -> {result[:40]}", file=sys.stderr, flush=True)
                    return remove_inline_repetition(result)
    
    # 方法 2: 偵測非連續重複（優化：提前終止）
    max_phrase_len = min(15, text_len // 3)
    for phrase_len in range(3, max_phrase_len):
        search_limit = text_len - phrase_len
        for start in range(search_limit):
            phrase = text[start:start + phrase_len]
            
            # 使用 frozenset O(1) 檢查
            if all(c in PUNCTUATION_SET for c in phrase):
                continue
            
            count = text.count(phrase)
            if count >= 3:
                first_idx = text.find(phrase)
                result = text[:first_idx + phrase_len] + text[first_idx + phrase_len:].replace(phrase, '')
                result = RE_PUNCTUATION_CLEANUP.sub('。', result).strip()
                
                if result != original and len(result) >= 4:
                    print(f"🔧 移除散落重複: {original[:40]} -> {result[:40]}", file=sys.stderr, flush=True)
                    return result
    
    return text


def remove_repeated_substrings(text: str) -> str:
    """移除連續重複的子字串 - 保留不重複的前綴"""
    if len(text) < 8:
        return text
    
    # 按句尾標點分割
    sentence_endings = ['。', '！', '？', '!', '?']
    for ending in sentence_endings:
        if ending in text:
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
                unique = []
                seen = set()
                for p in parts:
                    if p not in seen:
                        unique.append(p)
                        seen.add(p)
                
                if len(unique) < len(parts):
                    return ''.join(unique)
    
    # 偵測連續重複的子字串模式
    for pattern_len in range(min(30, len(text) // 2), 4, -1):
        for start in range(len(text) - pattern_len * 2 + 1):
            pattern = text[start:start + pattern_len]
            
            if all(c in '，。！？ 、,.!? ' for c in pattern):
                continue
            
            has_ending = any(pattern.endswith(e) for e in ['。', '！', '？', '，', '!', '?', ','])
            if not has_ending:
                continue
            
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
            
            if count >= 2 and len(pattern) * count > len(text) * 0.5:
                prefix = text[:first_idx].strip() if first_idx > 0 else ""
                result = pattern.strip()
                if prefix:
                    return prefix + result
                return result
    
    return text


def filter_translated_repetition(text: str) -> str:
    """過濾翻譯後的重複內容 - 加強版"""
    if not text or len(text) < 4:
        return text
    
    original_text = text
    
    # 🎯 過濾明顯的翻譯品質問題（要不要要不要、想要想要想要）
    if RE_REPEATED_WORDS.search(text):
        print(f"⚠️ 過濾重複詞翻譯: {text[:40]}", file=sys.stderr, flush=True)
        return ""
    
    # 🎯 過濾連續結巴模式（XX XX XX）
    stuttering_match = RE_STUTTERING.search(text)
    if stuttering_match:
        # 嘗試修復：只保留一次重複
        pattern = stuttering_match.group(1)
        fixed = RE_STUTTERING.sub(pattern, text)
        if fixed != text:
            print(f"🔧 修復結巴翻譯: {text[:40]} -> {fixed[:40]}", file=sys.stderr, flush=True)
            text = fixed
            original_text = fixed
    
    # 🎯 過濾「J+中文」開頭的幻覺翻譯
    if RE_J_PREFIX_HALLUCINATION.match(text):
        print(f"⚠️ 過濾 J 前綴幻覺: {text[:40]}", file=sys.stderr, flush=True)
        return ""
    
    # 先用 remove_inline_repetition 處理
    text = remove_inline_repetition(text)
    if text != original_text:
        original_text = text
    
    # 偵測空格分隔的完全相同片段
    if ' ' in text:
        space_parts = [p.strip() for p in text.split(' ') if p.strip()]
        if len(space_parts) >= 2:
            unique_space = []
            for p in space_parts:
                if not unique_space or p != unique_space[-1]:
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
    
    # 偵測連續重複的子字串
    cleaned = remove_repeated_substrings(text)
    if cleaned != text:
        print(f"🔧 去除重複子字串: {original_text[:40]} -> {cleaned[:40]}", file=sys.stderr, flush=True)
        return cleaned
    
    # 按標點分割並去重
    separators = ['，', '。', '！', '？']
    for sep in separators:
        if sep in text and text.count(sep) >= 1:
            parts = [p.strip() for p in text.split(sep) if p.strip()]
            if len(parts) >= 2:
                unique = []
                for p in parts:
                    is_dup = False
                    for u in unique:
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


def detect_character_repetition(text: str) -> bool:
    """偵測異常的字符重複 - 優化版（frozenset + 提前終止）"""
    text_len = len(text)
    if text_len < 6:
        return False
    
    # 移除有效模式（使用 frozenset 檢查）
    temp_text = text
    for vp in VALID_REPEAT_PATTERNS:
        if vp in temp_text:
            temp_text = temp_text.replace(vp, '')
    
    if len(temp_text) < 4:
        return False
    
    # 使用 frozenset O(1) 過濾
    content_chars = [c for c in temp_text if c not in PUNCTUATION_ONLY_SET]
    content_len = len(content_chars)
    if content_len < 4:
        return False
    
    # Counter 統計
    char_counts = Counter(content_chars)
    max_count = max(char_counts.values())
    threshold = content_len * 0.35
    
    if max_count > threshold:
        return True
    
    # 模式重複檢測（優化迴圈範圍）
    max_pattern = min(15, text_len // 3 + 1)
    for pattern_len in range(2, max_pattern):
        max_start = min(3, text_len - pattern_len * 3)
        for start in range(max_start):
            pattern = text[start:start + pattern_len]
            # 使用 frozenset O(1) 檢查
            if all(c in PUNCTUATION_ONLY_SET for c in pattern):
                continue
            if pattern * 3 in text:
                return True
    
    return False


def detect_phrase_repetition(text: str) -> bool:
    """偵測重複的詞組"""
    for pattern_len in range(2, min(20, len(text) // 2 + 1)):
        for start in range(len(text) - pattern_len * 2 + 1):
            pattern = text[start:start + pattern_len]
            if all(c in '、，。！？　 ' for c in pattern):
                continue
            if pattern * 3 in text:
                return True
    
    separators = ['、', '，', '。', ' ']
    for sep in separators:
        if sep in text:
            parts = [p.strip() for p in text.split(sep) if p.strip() and len(p.strip()) >= 2]
            if len(parts) >= 3:
                consecutive = 1
                for i in range(1, len(parts)):
                    if parts[i] == parts[i-1]:
                        consecutive += 1
                        if consecutive >= 2:
                            return True
                    else:
                        consecutive = 1
                
                counts = Counter(parts)
                for part, count in counts.items():
                    if count >= 2 and count >= len(parts) * 0.4:
                        return True
    
    return False


def remove_source_repetition(text: str) -> str:
    """從日文源文中去除重複，保留有意義的內容"""
    if not text or len(text) < 4:
        return text
    
    # 按空格分割去重
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
    
    # 尋找重複模式並只保留一次
    for pattern_len in range(2, min(30, len(text) // 2 + 1)):
        for start in range(min(5, len(text) - pattern_len * 2)):
            pattern = text[start:start + pattern_len]
            
            if all(c in '、，。！？　 ・ー' for c in pattern):
                continue
            
            count = text.count(pattern)
            
            if count >= 3 and len(pattern) * count > len(text) * 0.4:
                first_idx = text.find(pattern)
                last_idx = text.rfind(pattern)
                
                prefix = text[:first_idx].strip() if first_idx > 0 else ""
                suffix = text[last_idx + len(pattern):].strip() if last_idx + len(pattern) < len(text) else ""
                
                result = prefix + pattern + suffix
                result = result.strip()
                
                if result and len(result) >= 2:
                    return result
    
    # 如果整個文字只是單一模式重複
    for pattern_len in range(2, min(20, len(text) // 3 + 1)):
        pattern = text[:pattern_len]
        if all(c in '、，。！？　 ・ー' for c in pattern):
            continue
        
        repeated = pattern * (len(text) // len(pattern) + 1)
        if text in repeated or repeated.startswith(text):
            return pattern.strip()
    
    return text


def filter_text(text: str) -> str:
    """過濾無效文字 - 優化版（預編譯正則 + frozenset）"""
    if not text:
        return ""
    
    # 日文字符過濾（使用預編譯正則）
    cleaned = "".join(RE_JAPANESE_CHARS.findall(text)).strip()
    
    if not cleaned:
        return ""
    
    # 🎯 ASR 幻覺過濾（J愛心J 等模式）
    if RE_ASR_HALLUCINATION.search(cleaned) or RE_BROKEN_PATTERN.search(cleaned):
        print(f"⚠️ 過濾 ASR 幻覺: {cleaned[:40]}", file=sys.stderr, flush=True)
        return ""
    
    # 幻覺過濾（使用 frozenset O(1) 查找）
    for phrase in UNWANTED_PHRASES:
        if phrase in cleaned:
            return ""
    
    # 去除重複字符
    if detect_character_repetition(cleaned):
        deduped = remove_source_repetition(cleaned)
        if deduped and len(deduped) >= 2:
            print(f"🔄 去除源文重複: {cleaned[:30]}... -> {deduped[:30]}", file=sys.stderr, flush=True)
            cleaned = deduped
        else:
            print(f"⚠️ 過濾純重複: {cleaned[:30]}...", file=sys.stderr, flush=True)
            return ""
    
    # 去除重複詞組
    if detect_phrase_repetition(cleaned):
        deduped = remove_source_repetition(cleaned)
        if deduped and len(deduped) >= 2:
            print(f"🔄 去除源文重複: {cleaned[:30]}... -> {deduped[:30]}", file=sys.stderr, flush=True)
            cleaned = deduped
        else:
            print(f"⚠️ 過濾純重複: {cleaned[:30]}...", file=sys.stderr, flush=True)
            return ""
    
    return cleaned if len(cleaned) >= 2 else ""


def is_sentence_complete(text: str) -> bool:
    """檢查文字是否為完整句子 - 優化版（tuple 迭代）"""
    if not text:
        return False
    
    text = text.strip()
    text_len = len(text)
    
    # 長度足夠即視為完整
    if text_len >= 15:
        return True
    
    # 使用預定義 tuple（比 list 快）
    for ending in SENTENCE_ENDINGS:
        if text.endswith(ending):
            return True
    
    return False


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


def extract_new_content(current: str, previous: str) -> str:
    """提取新內容，移除與前一次重疊的部分"""
    if not previous or not current:
        return current
    
    if current == previous:
        return ""
    
    if previous in current:
        idx = current.find(previous)
        if idx == 0:
            return current[len(previous):].strip()
        elif idx + len(previous) == len(current):
            return current[:idx].strip()
    
    for i in range(min(len(previous), len(current)), 0, -1):
        if previous[-i:] == current[:i]:
            new_part = current[i:].strip()
            if len(new_part) >= 2:
                return new_part
            return ""
    
    for i in range(min(len(previous), len(current)), 0, -1):
        if previous[:i] == current[-i:]:
            new_part = current[:-i].strip()
            if len(new_part) >= 2:
                return new_part
            return ""
    
    return current


def detect_gibberish_transliteration(text: str) -> bool:
    """偵測無意義的音譯串 - 優化版（預編譯正則 + frozenset）"""
    if not text or len(text) < 8:
        return False
    
    # 移除標點符號（使用預編譯正則）
    clean_text = RE_GIBBERISH_PUNCT.sub('', text)
    clean_len = len(clean_text)
    if clean_len < 6:
        return False
    
    # 1. 相同音節重複檢測
    for syllable_len in range(2, 5):
        max_start = clean_len - syllable_len * 3
        for start in range(max_start):
            syllable = clean_text[start:start + syllable_len]
            repeated = syllable * 3
            if repeated in clean_text:
                # 使用 frozenset O(1) 檢查
                if not any(syllable.startswith(c) for c in COMMON_VALID_REPEATS):
                    return True
    
    # 2. 音譯比例檢測（使用 frozenset O(1) 查找）
    text_chars = [c for c in clean_text if c not in COMMON_CHARS_EXCLUDE]
    if text_chars:
        # 使用生成器避免建立中間列表
        transliteration_count = sum(1 for c in text_chars if c in TRANSLITERATION_CHARS)
        transliteration_ratio = transliteration_count / len(text_chars)
        if transliteration_ratio > 0.5 and clean_len > 12:
            return True
    
    return False


def clean_gibberish_from_translation(text: str) -> str:
    """清理翻譯結果中的無意義音譯串"""
    if not text:
        return text
    
    # 先檢查整句是否為無意義音譯
    if detect_gibberish_transliteration(text):
        print(f"⚠️ 過濾無意義音譯: {text[:40]}", file=sys.stderr, flush=True)
        return ""
    
    # 分句處理，移除無意義的部分
    separators = ['，', '。', '！', '？']
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            cleaned_parts = []
            for part in parts:
                part = part.strip()
                if part and not detect_gibberish_transliteration(part):
                    cleaned_parts.append(part)
            
            if len(cleaned_parts) < len([p for p in parts if p.strip()]):
                result = sep.join(cleaned_parts)
                if sep in ['。', '！', '？'] and result and not result.endswith(sep):
                    result += sep
                if result != text:
                    print(f"🔧 移除部分無意義音譯: {text[:40]} -> {result[:40]}", file=sys.stderr, flush=True)
                return result
    
    return text
