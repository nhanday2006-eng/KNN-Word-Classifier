"""
commonsense_lexicon.py
Từ điển Cặp từ Đời thường (Everyday Commonsense Lexicon) & Bộ Chuẩn hóa Từ gốc (Normalizer)
Giải quyết các cặp từ giao tiếp hàng ngày bị lệch từ loại (POS), bất đối xứng hoặc nối gián tiếp trong WordNet.
"""

import nltk
from nltk.corpus import wordnet as wn

# =============================================================================
# 1. BẢNG QUY ĐỔI SO SÁNH HƠN / SO SÁNH NHẤT (COMPARATIVE / SUPERLATIVE)
# =============================================================================
IRREGULAR_COMPARATIVES = {
    # Bất quy tắc
    "better": "good",
    "best": "good",
    "worse": "bad",
    "worst": "bad",
    "farther": "far",
    "further": "far",
    "farthest": "far",
    "furthest": "far",
    "less": "little",
    "least": "little",
    "more": "much",
    "most": "much",

    # Các từ so sánh hơn và so sánh nhất thông dụng
    "bigger": "big",
    "biggest": "big",
    "smaller": "small",
    "smallest": "small",
    "older": "old",
    "oldest": "old",
    "younger": "young",
    "youngest": "young",
    "faster": "fast",
    "fastest": "fast",
    "slower": "slow",
    "slowest": "slow",
    "taller": "tall",
    "tallest": "tall",
    "shorter": "short",
    "shortest": "short",
    "higher": "high",
    "highest": "high",
    "lower": "low",
    "lowest": "low",
    "heavier": "heavy",
    "heaviest": "heavy",
    "lighter": "light",
    "lightest": "light",
    "cleaner": "clean",
    "cleanest": "clean",
    "dirtier": "dirty",
    "dirtiest": "dirty",
    "easier": "easy",
    "easiest": "easy",
    "harder": "hard",
    "hardest": "hard",
    "earlier": "early",
    "earliest": "early",
    "later": "late",
    "latest": "late",
    "richer": "rich",
    "richest": "rich",
    "poorer": "poor",
    "poorest": "poor",
    "stronger": "strong",
    "strongest": "strong",
    "weaker": "weak",
    "weakest": "weak",
    "hotter": "hot",
    "hottest": "hot",
    "colder": "cold",
    "coldest": "cold",
    "brighter": "bright",
    "brightest": "bright",
    "darker": "dark",
    "darkest": "dark",
    "sweeter": "sweet",
    "sweetest": "sweet",
    "cheaper": "cheap",
    "cheapest": "cheap",
    "wider": "wide",
    "widest": "wide",
    "narrower": "narrow",
    "narrowest": "narrow",
    "thicker": "thick",
    "thickest": "thick",
    "thinner": "thin",
    "thinnest": "thin",
    "deeper": "deep",
    "deepest": "deep",
    "shallower": "shallow",
    "shallowest": "shallow",
    "quieter": "quiet",
    "quietest": "quiet",
    "louder": "loud",
    "loudest": "loud",
    "safer": "safe",
    "safest": "safe",
    "simpler": "simple",
    "simplest": "simple"
}

# =============================================================================
# 2. CHÍNH TẢ ANH - ANH (UK) vs ANH - MỸ (US)
# =============================================================================
UK_TO_US_SPELLING = {
    "colour": "color",
    "centre": "center",
    "favourite": "favorite",
    "grey": "gray",
    "behaviour": "behavior",
    "analyse": "analyze",
    "traveller": "traveler",
    "theatre": "theater",
    "organise": "organize",
    "defence": "defense",
    "licence": "license",
    "programme": "program",
    "flavour": "flavor",
    "humour": "humor",
    "honour": "honor",
    "neighbour": "neighbor",
    "catalogue": "catalog",
    "dialogue": "dialog",
    "apologise": "apologize",
    "realise": "realize"
}

UK_US_PAIRS = set()
for uk, us in UK_TO_US_SPELLING.items():
    UK_US_PAIRS.add(tuple(sorted([uk, us])))


def normalize_spelling(word: str) -> str:
    """Quy đổi từ viết theo chính tả UK sang US nếu có."""
    w = word.lower().strip()
    return UK_TO_US_SPELLING.get(w, w)


def is_spelling_variant(w1: str, w2: str) -> bool:
    """Kiểm tra hai từ có phải là biến thể chính tả UK/US hay không."""
    w1 = w1.lower().strip()
    w2 = w2.lower().strip()
    if w1 == w2:
        return True
    if tuple(sorted([w1, w2])) in UK_US_PAIRS:
        return True
    if normalize_spelling(w1) == normalize_spelling(w2):
        return True
    return False


def normalize_base_word(word: str) -> str:
    """
    Quy đổi từ về dạng gốc (positive degree / base form):
    - better -> good, worse -> bad
    - bigger -> big, smaller -> small
    - faster -> fast, slower -> slow
    - taller -> tall, shorter -> short
    """
    w = word.lower().strip()
    # 1. Tra cứu bảng bất quy tắc & phổ biến
    if w in IRREGULAR_COMPARATIVES:
        return IRREGULAR_COMPARATIVES[w]

    # 2. Tra cứu WordNet morphy
    try:
        m = wn.morphy(w, wn.ADJ)
        if m and m != w and len(m) >= 2:
            return m
    except Exception:
        pass

    # 3. Quy tắc cắt đuôi -est / -er
    if w.endswith("est") and len(w) >= 5:
        if w.endswith("iest"):
            cand = w[:-4] + "y"
            if wn.synsets(cand):
                return cand
        base = w[:-3]
        if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "aeiouls":
            cand = base[:-1]
            if wn.synsets(cand):
                return cand
        if wn.synsets(base):
            return base

    if w.endswith("er") and len(w) >= 4:
        if w.endswith("ier"):
            cand = w[:-3] + "y"
            if wn.synsets(cand):
                return cand
        base = w[:-2]
        if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "aeiouls":
            cand = base[:-1]
            if wn.synsets(cand):
                return cand
        if wn.synsets(base):
            return base

    return w


# =============================================================================
# 3. TỪ ĐIỂN CẶP TỪ TRÁI NGHĨA ĐỜI THƯỜNG (COMMONSENSE ANTONYMS)
# =============================================================================
RAW_COMMONSENSE_ANTONYMS = [
    # Yêu cầu đề bài Hạng mục 2
    ("fun", "boring"), ("fun", "dull"), ("exciting", "boring"), ("interesting", "boring"),
    ("smart", "stupid"), ("smart", "dumb"), ("clever", "stupid"), ("clever", "dumb"),
    ("funny", "serious"), ("brave", "cowardly"), ("brave", "scared"), ("brave", "afraid"),
    ("calm", "nervous"), ("calm", "anxious"), ("calm", "stressed"),
    ("quiet", "noisy"), ("quiet", "loud"),
    ("cheap", "expensive"), ("clean", "messy"), ("clean", "dirty"),
    ("fake", "real"), ("fake", "genuine"),
    ("lazy", "hardworking"), ("lazy", "diligent"),
    ("polite", "rude"), ("safe", "dangerous"),
    ("easy", "hard"), ("easy", "difficult"),
    ("hungry", "full"), ("free", "busy"),
    ("sick", "healthy"), ("ill", "healthy"),
    ("tired", "energetic"), ("tired", "rested"),
    ("excited", "bored"), ("relaxed", "tense"),
    ("afraid", "confident"), ("awake", "asleep"),
    ("alive", "dead"), ("married", "single"),
    ("guilty", "innocent"), ("public", "private"),
    ("ancient", "modern"), ("fresh", "stale"),
    ("sharp", "blunt"), ("sweet", "sour"), ("sweet", "bitter"),
    ("deep", "shallow"), ("wide", "narrow"),
    ("thick", "thin"), ("tight", "loose"),
    ("wet", "dry"), ("empty", "full"),
    ("heavy", "light"), ("raw", "cooked"),
    ("rich", "poor"), ("rich", "broke"),
    ("cool", "lame"), ("dad", "mom"),
    ("buy", "sell"), ("borrow", "lend"),
    ("win", "lose"), ("hire", "fire"),

    # Các cặp kinh điển cơ bản để hỗ trợ so sánh hơn (better-worse, taller-shorter, ...)
    ("good", "bad"), ("tall", "short"), ("big", "small"), ("large", "little"),
    ("fast", "slow"), ("happy", "sad"), ("hot", "cold"), ("high", "low"),
    ("old", "young"), ("strong", "weak"), ("bright", "dark"), ("near", "far"),
    ("rich", "poor"), ("hard", "soft"), ("early", "late"), ("true", "false"),
    ("open", "close"), ("start", "finish"), ("love", "hate"), ("give", "take")
]

COMMONSENSE_ANTONYMS = {
    tuple(sorted([normalize_base_word(w1), normalize_base_word(w2)]))
    for w1, w2 in RAW_COMMONSENSE_ANTONYMS
}

# =============================================================================
# 4. TỪ ĐIỂN CẶP TỪ ĐỒNG NGHĨA ĐỜI THƯỜNG (COMMONSENSE SYNONYMS)
# =============================================================================
RAW_COMMONSENSE_SYNONYMS = [
    # Yêu cầu đề bài Hạng mục 2
    ("fun", "enjoyable"), ("fun", "entertaining"), ("fun", "amusing"),
    ("funny", "hilarious"), ("tired", "exhausted"),
    ("scared", "frightened"), ("scared", "afraid"),
    ("angry", "mad"), ("angry", "furious"),
    ("smart", "clever"), ("smart", "bright"), ("smart", "intelligent"),
    ("cool", "awesome"), ("great", "awesome"),
    ("sick", "ill"), ("gift", "present"),
    ("job", "work"), ("house", "home"),
    ("stone", "rock"), ("trash", "garbage"),
    ("fast", "quick"), ("kid", "child"),
    ("hard", "difficult"),

    # Các cặp cốt lõi bổ sung
    ("happy", "joyful"), ("happy", "glad"), ("big", "large"), ("small", "tiny"),
    ("buy", "purchase"), ("start", "begin"), ("rich", "wealthy"),
    ("tall", "high"), ("tall", "tallish"), ("hot", "warm"), ("cold", "chilly")
]

COMMONSENSE_SYNONYMS = {
    tuple(sorted([normalize_base_word(w1), normalize_base_word(w2)]))
    for w1, w2 in RAW_COMMONSENSE_SYNONYMS
}


def is_commonsense_antonym(w1: str, w2: str) -> bool:
    """Kiểm tra cặp từ có nằm trong kho trái nghĩa đời thường hay không."""
    n1 = normalize_base_word(w1)
    n2 = normalize_base_word(w2)
    pair = tuple(sorted([n1, n2]))
    if pair in COMMONSENSE_ANTONYMS:
        return True

    s1 = normalize_spelling(n1)
    s2 = normalize_spelling(n2)
    return tuple(sorted([s1, s2])) in COMMONSENSE_ANTONYMS


def is_commonsense_synonym(w1: str, w2: str) -> bool:
    """Kiểm tra cặp từ có nằm trong kho đồng nghĩa đời thường hay không."""
    if is_spelling_variant(w1, w2):
        return True

    n1 = normalize_base_word(w1)
    n2 = normalize_base_word(w2)
    pair = tuple(sorted([n1, n2]))
    if pair in COMMONSENSE_SYNONYMS:
        return True

    s1 = normalize_spelling(n1)
    s2 = normalize_spelling(n2)
    return tuple(sorted([s1, s2])) in COMMONSENSE_SYNONYMS


def get_commonsense_associations(word: str) -> dict:
    """
    Trả về tất cả các từ đồng nghĩa và trái nghĩa đời thường liên kết với word.
    Dùng cho Tab 2 (Word Explorer).
    """
    w_norm = normalize_base_word(word)
    w_spell = normalize_spelling(w_norm)

    syns = set()
    ants = set()

    # Kiểm tra biến thể chính tả
    for uk, us in UK_US_PAIRS:
        if w_norm == uk:
            syns.add(us)
        elif w_norm == us:
            syns.add(uk)

    # Quét antonyms
    for p1, p2 in COMMONSENSE_ANTONYMS:
        if w_norm == p1 or w_spell == p1:
            ants.add(p2)
        elif w_norm == p2 or w_spell == p2:
            ants.add(p1)

    # Quét synonyms
    for p1, p2 in COMMONSENSE_SYNONYMS:
        if w_norm == p1 or w_spell == p1:
            syns.add(p2)
        elif w_norm == p2 or w_spell == p2:
            syns.add(p1)

    return {"synonyms": syns, "antonyms": ants}
