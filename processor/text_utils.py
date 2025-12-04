"""
文字處理模組 - 過濾、去重、清理
"""
import re
import sys
from collections import Counter


def calculate_similarity(s1: str, s2: str) -> float:
    """計算兩個字串的相似度 (0-1) - 使用多種算法"""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    
    # 方法 1: 子字串檢測
    if s1 in s2 or s2 in s1:
        shorter = min(len(s1), len(s2))
        longer = max(len(s1), len(s2))
        return shorter / longer
    
    # 方法 2: N-gram 相似度
    def get_ngrams(s, n=2):
        return set(s[i:i+n] for i in range(len(s)-n+1)) if len(s) >= n else {s}
    
    ngrams1 = get_ngrams(s1, 2)
    ngrams2 = get_ngrams(s2, 2)
    
    if not ngrams1 or not ngrams2:
        set1 = set(s1)
        set2 = set(s2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    
    return intersection / union if union > 0 else 0.0


def remove_inline_repetition(text: str) -> str:
    """移除句中連續重複的片段（如：這代碼不錯這代碼不錯）"""
    if not text or len(text) < 8:
        return text
    
    original = text
    
    # 方法 1: 偵測完全相同的連續重複
    for pattern_len in range(min(25, len(text) // 2), 3, -1):
        for start in range(len(text) - pattern_len * 2 + 1):
            pattern = text[start:start + pattern_len]
            
            if all(c in '，。！？、 ～~' for c in pattern):
                continue
            
            repeat_pos = start + pattern_len
            if text[repeat_pos:repeat_pos + pattern_len] == pattern:
                count = 2
                check_pos = repeat_pos + pattern_len
                while text[check_pos:check_pos + pattern_len] == pattern:
                    count += 1
                    check_pos += pattern_len
                
                prefix = text[:start]
                suffix = text[start + pattern_len * count:]
                result = (prefix + pattern + suffix).strip()
                
                if result != original:
                    print(f"🔧 移除行內重複: {original[:40]} -> {result[:40]}", file=sys.stderr, flush=True)
                    return remove_inline_repetition(result)
    
    # 方法 2: 偵測非連續重複
    for phrase_len in range(3, min(15, len(text) // 3)):
        for start in range(len(text) - phrase_len):
            phrase = text[start:start + phrase_len]
            if all(c in '，。！？、 ～~' for c in phrase):
                continue
            
            count = text.count(phrase)
            if count >= 3:
                first_idx = text.find(phrase)
                result = text[:first_idx + phrase_len]
                remaining = text[first_idx + phrase_len:]
                remaining = remaining.replace(phrase, '')
                result = (result + remaining).strip()
                result = re.sub(r'[，。！？]{2,}', '。', result)
                
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
    """偵測異常的字符重複 (幻覺特徵)"""
    if len(text) < 6:
        return False
    
    valid_patterns = ['ww', 'ーー', '...', '！！', '？？', '〜〜']
    temp_text = text
    for vp in valid_patterns:
        temp_text = temp_text.replace(vp, '')
    
    if len(temp_text) < 4:
        return False
    
    content_chars = [c for c in temp_text if c not in ' 　、。！？，']
    if len(content_chars) < 4:
        return False
    
    char_counts = Counter(content_chars)
    max_count = max(char_counts.values())
    
    if max_count > len(content_chars) * 0.35:
        return True
    
    for pattern_len in range(2, min(15, len(text) // 3 + 1)):
        for start in range(min(3, len(text) - pattern_len * 3)):
            pattern = text[start:start + pattern_len]
            if all(c in '、，。！？　 ・' for c in pattern):
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
    """過濾無效文字，去除重複後保留有效內容繼續處理"""
    if not text:
        return ""
    
    # 日文字符過濾
    pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uFF00-\uFFEF\u0020-\u007E]+')
    cleaned = "".join(pattern.findall(text)).strip()
    
    if not cleaned:
        return ""
    
    # 幻覺過濾列表
    unwanted = [
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
    ]
    
    for phrase in unwanted:
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
    """檢查文字是否為完整句子"""
    if not text:
        return False
    
    sentence_endings = [
        '。', '！', '？', '、',
        'ね', 'よ', 'よね', 'わ', 'か',
        'です', 'ます', 'た', 'だ',
        'い', 'いよ', 'いね',
        '...', '…',
    ]
    
    text = text.strip()
    for ending in sentence_endings:
        if text.endswith(ending):
            return True
    
    if len(text) >= 15:
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
