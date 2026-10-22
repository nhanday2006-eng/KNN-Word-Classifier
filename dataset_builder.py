"""
dataset_builder.py
Trích xuất tự động và cân bằng 3000 cặp từ tiếng Anh từ WordNet và kho từ vựng cốt lõi:
- 1000 cặp Trái nghĩa (Antonyms) - Nhãn 0
- 1000 cặp Đồng nghĩa (Synonyms) - Nhãn 1
- 1000 cặp Không liên quan (Unrelated) - Nhãn 2

Tích hợp danh sách các cặp từ thông dụng cốt lõi và từ điển đời thường (Everyday Commonsense Lexicon),
bao gồm cả biến thể chính tả UK/US, và ưu tiên các synset phổ biến đầu tiên để tránh các nghĩa bóng hiếm gặp.
"""

import os
import random
import sys
import time
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn

from commonsense_lexicon import (
    COMMONSENSE_ANTONYMS,
    COMMONSENSE_SYNONYMS,
    UK_US_PAIRS,
    is_commonsense_antonym,
    is_commonsense_synonym
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

LABEL_NAMES = {
    0: "Antonym",
    1: "Synonym",
    2: "Unrelated"
}

# 1. Danh sách cặp từ trái nghĩa cốt lõi phổ biến trong tiếng Anh
CORE_ANTONYMS = [
    ("tall", "short"), ("hot", "cold"), ("big", "small"), ("large", "little"),
    ("fast", "slow"), ("happy", "sad"), ("good", "bad"), ("rich", "poor"),
    ("strong", "weak"), ("bright", "dark"), ("high", "low"), ("old", "young"),
    ("easy", "difficult"), ("clean", "dirty"), ("heavy", "light"), ("wide", "narrow"),
    ("thick", "thin"), ("hard", "soft"), ("dry", "wet"), ("deep", "shallow"),
    ("loud", "quiet"), ("sweet", "sour"), ("smooth", "rough"), ("safe", "dangerous"),
    ("alive", "dead"), ("true", "false"), ("empty", "full"), ("tight", "loose"),
    ("early", "late"), ("near", "far"), ("sharp", "dull"), ("brave", "cowardly"),
    ("cruel", "kind"), ("cheap", "expensive"), ("simple", "complex"), ("increase", "decrease"),
    ("accept", "refuse"), ("enter", "exit"), ("win", "lose"), ("push", "pull"),
    ("open", "close"), ("start", "finish"), ("love", "hate"), ("buy", "sell"),
    ("give", "take"), ("careful", "careless"), ("hopeful", "hopeless"), ("useful", "useless"),
    ("happy", "unhappy"), ("able", "unable"), ("visible", "invisible"), ("legal", "illegal"),
    ("possible", "impossible"), ("regular", "irregular"), ("patient", "impatient"),
    ("polite", "impolite"), ("agree", "disagree"), ("honest", "dishonest"),
    ("appear", "disappear"), ("connect", "disconnect"), ("cover", "uncover"),
    ("active", "passive"), ("brave", "timid"), ("clever", "foolish"), ("healthy", "sick"),
    ("calm", "stormy"), ("friendly", "hostile"), ("guilty", "innocent"), ("proud", "humble"),
    ("wise", "foolish"), ("sharp", "blunt"), ("fresh", "stale"), ("wild", "tame"),
    ("public", "private"), ("modern", "ancient"), ("permanent", "temporary"),
    ("normal", "abnormal"), ("sane", "insane"), ("stable", "unstable")
]

# 2. Danh sách cặp từ đồng nghĩa cốt lõi phổ biến
CORE_SYNONYMS = [
    ("tall", "high"), ("tall", "tallish"), ("short", "little"), ("short", "brief"),
    ("hot", "warm"), ("cold", "chilly"), ("big", "large"), ("huge", "giant"),
    ("small", "tiny"), ("small", "miniature"), ("fast", "quick"), ("fast", "speedy"),
    ("happy", "glad"), ("happy", "joyful"), ("happy", "cheerful"), ("sad", "unhappy"),
    ("sad", "sorrowful"), ("good", "fine"), ("good", "great"), ("bad", "terrible"),
    ("bad", "awful"), ("bad", "dreadful"), ("rich", "wealthy"), ("rich", "affluent"),
    ("poor", "needy"), ("poor", "penniless"), ("strong", "powerful"), ("strong", "mighty"),
    ("weak", "frail"), ("weak", "feeble"), ("bright", "shining"), ("bright", "radiant"),
    ("dark", "dim"), ("dark", "gloomy"), ("high", "elevated"), ("high", "lofty"),
    ("low", "depressed"), ("old", "aged"), ("old", "ancient"), ("young", "youthful"),
    ("easy", "simple"), ("easy", "effortless"), ("difficult", "hard"), ("difficult", "tough"),
    ("clean", "pure"), ("clean", "spotless"), ("dirty", "filthy"), ("dirty", "grimy"),
    ("brave", "courageous"), ("brave", "valiant"), ("smart", "clever"), ("smart", "bright"),
    ("smart", "intelligent"), ("begin", "start"), ("end", "finish"), ("buy", "purchase"),
    ("stop", "halt"), ("help", "assist"), ("speak", "talk"), ("answer", "reply"),
    ("calm", "peaceful"), ("careful", "cautious"), ("angry", "furious"), ("tired", "exhausted"),
    ("tasty", "delicious"), ("strange", "odd"), ("correct", "accurate"), ("false", "untrue"),
    ("silent", "quiet"), ("near", "close"), ("leave", "depart"), ("create", "make"),
    ("hide", "conceal"), ("show", "display"), ("protect", "defend"), ("rich", "loaded"),
    ("famous", "renowned"), ("kind", "generous"), ("brave", "fearless"), ("neat", "tidy")
]

# 3. Danh sách cặp từ không liên quan cốt lõi (Bao gồm cả các cặp từ cùng nhóm sự vật / Coordinate Terms)
CORE_UNRELATED = [
    # Nhóm kiểm thử quan trọng (Coordinate Terms & Unrelated)
    ("tall", "apple"), ("hot", "computer"), ("apple", "banana"), ("table", "chair"),
    ("dog", "cat"), ("car", "bicycle"), ("pizza", "music"),

    # Các cặp cùng nhóm thực thể (Co-hyponyms nhưng không đồng nghĩa)
    ("shirt", "pants"), ("coffee", "tea"), ("pen", "pencil"), ("hammer", "wrench"),
    ("bus", "train"), ("orange", "lemon"), ("fork", "spoon"), ("sun", "moon"),
    ("gold", "silver"), ("red", "blue"), ("lion", "tiger"), ("horse", "cow"),
    ("guitar", "piano"), ("doctor", "nurse"), ("ocean", "lake"), ("potato", "tomato"),
    ("knife", "fork"), ("football", "basketball"), ("newspaper", "magazine"),

    # Các cặp hoàn toàn xa nhau
    ("apple", "car"), ("doctor", "hospital"), ("table", "sky"), ("water", "guitar"),
    ("music", "window"), ("river", "clock"), ("chair", "cloud"), ("keyboard", "tree"),
    ("coffee", "mountain"), ("shoe", "cloud"), ("window", "cat"), ("banana", "keyboard"),
    ("elephant", "pencil"), ("plane", "soup"), ("lamp", "forest"), ("stone", "cloud"),
    ("bread", "bicycle"), ("bottle", "star"), ("watch", "flower"), ("camera", "river")
]


def ensure_nltk_resources():
    """Tự động kiểm tra và tải tài nguyên WordNet."""
    for pkg in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def is_valid_word(w: str) -> bool:
    """
    [KIỂM TRA HỢP LỆ]
    Từ hợp lệ chỉ gồm chữ cái tiếng Anh (a-z), độ dài từ 2 đến 20 ký tự,
    không chứa số, dấu cách hay ký tự đặc biệt.
    """
    return bool(w and w.isalpha() and 2 <= len(w) <= 20)


def extract_antonyms_wordnet(target_count=1000, initial_set=None):
    """
    [TRÍCH XUẤT TỪ TRÁI NGHĨA - NHÃN 0]
    Duyệt qua các Synset trong WordNet, tìm các cặp Lemma có quan hệ antonyms() trực tiếp.
    Ví dụ: lemma 'tall' -> antonyms() -> lemma 'short'.
    """
    antonyms = set(initial_set or [])
    for syn in wn.all_synsets():
        # Ưu tiên các synset chính
        for lem in syn.lemmas():
            w1 = lem.name().lower().strip()
            if not is_valid_word(w1):
                continue
            for ant in lem.antonyms():
                w2 = ant.name().lower().strip()
                if is_valid_word(w2) and w1 != w2:
                    pair = tuple(sorted([w1, w2]))
                    antonyms.add(pair)
                    if len(antonyms) >= target_count:
                        return antonyms
    return antonyms


def extract_synonyms_wordnet(target_count=1000, exclude_pairs=None, initial_set=None):
    """
    [TRÍCH XUẤT TỪ ĐỒNG NGHĨA - NHÃN 1]
    Duyệt qua các Synset trong WordNet. Mỗi Synset đại diện cho một khái niệm ngữ nghĩa cụ thể.
    Các từ (lemmas) nằm chung trong cùng 1 Synset chính là các từ đồng nghĩa của nhau.
    Ví dụ: synset 'big.a.01' chứa cả 'big' và 'large' -> cặp ('big', 'large').
    """
    if exclude_pairs is None:
        exclude_pairs = set()
    synonyms = set(initial_set or [])

    for syn in wn.all_synsets():
        valid_lemmas = [l.name().lower().strip() for l in syn.lemmas() if is_valid_word(l.name().lower().strip())]
        unique_lemmas = sorted(list(set(valid_lemmas)))
        if len(unique_lemmas) >= 2:
            for i in range(len(unique_lemmas)):
                for j in range(i + 1, len(unique_lemmas)):
                    pair = tuple(sorted([unique_lemmas[i], unique_lemmas[j]]))
                    if pair not in exclude_pairs and pair not in synonyms:
                        synonyms.add(pair)
                        if len(synonyms) >= target_count:
                            return synonyms
    return synonyms


def extract_unrelated_wordnet(vocab_list, target_count=1000, exclude_pairs=None, initial_set=None):
    """
    [TRÍCH XUẤT TỪ KHÔNG LIÊN QUAN - NHÃN 2]
    Trích xuất cặp từ không liên quan ngữ nghĩa từ kho từ vựng.
    Bao gồm cả các cặp từ cùng nhóm sự vật (Coordinate terms như apple-banana, dog-cat),
    nhưng tuyệt đối kiểm tra không được chung Synset và không được có quan hệ Antonyms/Similar_to.
    Điều này giúp mô hình k-NN học được ranh giới giữa Đồng nghĩa thật và Cùng nhóm danh mục.
    """
    if exclude_pairs is None:
        exclude_pairs = set()
    unrelated = set(initial_set or [])
    attempts = 0
    max_attempts = target_count * 35

    while len(unrelated) < target_count and attempts < max_attempts:
        attempts += 1
        w1, w2 = random.sample(vocab_list, 2)
        pair = tuple(sorted([w1, w2]))
        if pair in exclude_pairs or pair in unrelated:
            continue
        # Tuyệt đối không nằm trong commonsense lexicon
        if is_commonsense_antonym(w1, w2) or is_commonsense_synonym(w1, w2):
            continue

        all_s1 = set(wn.synsets(w1))
        all_s2 = set(wn.synsets(w2))
        # Tuyệt đối không được chung synset
        if all_s1 & all_s2:
            continue

        # Không được có quan hệ antonyms trực tiếp
        has_ant = False
        for s in all_s1:
            for lem in s.lemmas():
                for ant in lem.antonyms():
                    if ant.name().lower() == w2:
                        has_ant = True
                        break
            if has_ant:
                break
        if has_ant:
            continue

        # Không được có quan hệ similar_tos của tính từ
        sim1 = {st for s in all_s1 for st in s.similar_tos()}
        if all_s2 & sim1:
            continue

        unrelated.add(pair)

    return unrelated


def build_or_load_dataset(filepath="data/dataset.csv", n_per_class=1000, random_seed=42):
    """
    [HÀM ĐIỀU PHỐI DỮ LIỆU CHÍNH]
    Tạo hoặc nạp dataset 3000 mẫu cân bằng (1000 Antonyms, 1000 Synonyms, 1000 Unrelated).
    Nếu file CSV đã có sẵn trên đĩa thì đọc lên ngay để tiết kiệm thời gian khởi động.
    """
    ensure_nltk_resources()

    print(f"[Dataset] Đang xây dựng dataset cải tiến ({n_per_class} cặp/nhãn)...")
    random.seed(random_seed)
    t0 = time.time()

    # Khởi tạo tập cốt lõi kết hợp từ điển đời thường & chính tả UK/US
    core_ant_set = {tuple(sorted([w1.lower(), w2.lower()])) for w1, w2 in CORE_ANTONYMS} | set(COMMONSENSE_ANTONYMS)
    core_syn_set = {tuple(sorted([w1.lower(), w2.lower()])) for w1, w2 in CORE_SYNONYMS} | set(COMMONSENSE_SYNONYMS) | set(UK_US_PAIRS)
    core_unrel_set = {tuple(sorted([w1.lower(), w2.lower()])) for w1, w2 in CORE_UNRELATED}

    # 1. Trích xuất Antonyms
    ant_pairs = extract_antonyms_wordnet(target_count=n_per_class, initial_set=core_ant_set)
    print(f"  + Trích xuất {len(ant_pairs)} cặp Antonyms (Label 0)")

    # 2. Trích xuất Synonyms
    syn_pairs = extract_synonyms_wordnet(target_count=n_per_class, exclude_pairs=ant_pairs, initial_set=core_syn_set)
    print(f"  + Trích xuất {len(syn_pairs)} cặp Synonyms (Label 1)")

    # 3. Trích xuất Unrelated
    known_pairs = ant_pairs | syn_pairs
    vocab = set()
    for w1, w2 in known_pairs:
        vocab.add(w1)
        vocab.add(w2)
    vocab_list = list(vocab)

    unrelated_pairs = extract_unrelated_wordnet(vocab_list, target_count=n_per_class, exclude_pairs=known_pairs, initial_set=core_unrel_set)
    print(f"  + Trích xuất {len(unrelated_pairs)} cặp Unrelated (Label 2)")

    # 4. Gộp thành DataFrame
    rows = []
    for w1, w2 in list(ant_pairs)[:n_per_class]:
        rows.append({"word1": w1, "word2": w2, "label": 0, "label_name": LABEL_NAMES[0]})
    for w1, w2 in list(syn_pairs)[:n_per_class]:
        rows.append({"word1": w1, "word2": w2, "label": 1, "label_name": LABEL_NAMES[1]})
    for w1, w2 in list(unrelated_pairs)[:n_per_class]:
        rows.append({"word1": w1, "word2": w2, "label": 2, "label_name": LABEL_NAMES[2]})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False, encoding="utf-8")
    print(f"[Dataset] Hoàn tất lưu {len(df)} mẫu vào '{filepath}' trong {time.time() - t0:.2f}s!")
    return df


if __name__ == "__main__":
    df = build_or_load_dataset()
    print("\nPhân bố nhãn:")
    print(df["label_name"].value_counts())
    print("\n5 mẫu đầu tiên:")
    print(df.head())
    print("\nKiểm tra tall-short:", df[(df['word1'] == 'short') & (df['word2'] == 'tall') | (df['word1'] == 'tall') & (df['word2'] == 'short')])
