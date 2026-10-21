"""
feature_extractor.py
Trích xuất vector đặc trưng 12 chiều (12-dimensional feature vector) cô đọng, chất lượng cao
cho thuật toán k-Nearest Neighbors (k-NN), loại bỏ hoàn toàn hiện tượng Curse of Dimensionality:

1.  lemma_overlap: Tỷ lệ synset dùng chung giữa 2 từ trong WordNet.
2.  similar_to_score: Mức độ liên kết qua quan hệ similar_tos() và also_sees() của tính từ.
3.  antonym_graph_score: Điểm đối lập ngữ nghĩa trên đồ thị (direct + indirect cluster antonyms).
4.  shared_attribute_or_hypernym: Jaccard similarity của hypernyms (danh/động từ) + attributes (tính từ).
5.  max_path_sim: Max path_similarity (có chuẩn hóa tương đương cho tính từ chia sẻ attribute/cluster).
6.  max_wup_sim: Max wup_similarity (có chuẩn hóa tương đương cho tính từ).
7.  same_lexname: Tỷ lệ các synset có cùng lĩnh vực từ điển (synset.lexname()).
8.  same_pos: Tỷ lệ có cùng từ loại (Part-of-Speech).
9.  gloss_cosine_sim: Độ tương đồng Cosine TF-IDF giữa văn bản định nghĩa + ví dụ của 2 từ.
10. svd_cosine_sim: Độ tương đồng Cosine giữa 2 vector SVD 50 chiều.
11. svd_euclidean_dist: Khoảng cách Euclidean giữa 2 vector SVD 50 chiều.
12. morph_opposition_flag: Cờ phát hiện tiền tố/hậu tố đối lập hình thái (un-, in-, dis-, -less vs -ful).
"""

import sys
import numpy as np
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.corpus import wordnet as wn

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

NEGATION_PREFIXES = (
    "un", "in", "im", "ir", "il", "dis", "non", "ab", "de", "anti", "mis", "counter", "a"
)


def check_morphological_opposition(w1: str, w2: str) -> float:
    """Kiểm tra tiền tố hoặc hậu tố đối lập hình thái học."""
    w1 = w1.lower().strip()
    w2 = w2.lower().strip()
    if len(w1) >= len(w2):
        w_long, w_short = w1, w2
    else:
        w_long, w_short = w2, w1

    # 1. Tiền tố đối lập
    for p in NEGATION_PREFIXES:
        if w_long.startswith(p):
            rem = w_long[len(p):]
            if rem == w_short or (len(rem) >= 3 and SequenceMatcher(None, rem, w_short).ratio() >= 0.85):
                return 1.0

    # 2. Hậu tố -ful vs -less
    if (w1.endswith("ful") and w2.endswith("less")) or (w1.endswith("less") and w2.endswith("ful")):
        b1 = w1[:-3] if w1.endswith("ful") else w1[:-4]
        b2 = w2[:-3] if w2.endswith("ful") else w2[:-4]
        if b1 == b2:
            return 1.0
    elif w_long.endswith("less") and w_long[:-4] == w_short:
        return 1.0

    return 0.0


class WordPairFeatureExtractor:
    """Bộ trích xuất 12 đặc trưng ngữ nghĩa & hình thái tối ưu hóa cho k-NN."""

    def __init__(self, n_components: int = 50, max_tfidf_features: int = 6000):
        self.n_components = n_components
        self.max_tfidf_features = max_tfidf_features
        self.tfidf = None
        self.svd = None
        self.vector_cache = {}
        self.info_cache = {}

    def _get_gloss_text(self, word: str) -> str:
        """Lấy văn bản định nghĩa và ví dụ từ các synset phổ biến của từ."""
        syns = wn.synsets(word.lower().strip())[:4]
        if syns:
            parts = []
            for s in syns:
                parts.append(s.definition())
                for ex in s.examples()[:2]:
                    parts.append(ex)
            return " ".join(parts)
        return word.lower().strip()

    def fit(self, words_iterable):
        """Huấn luyện LSA (TF-IDF + SVD 50 chiều) trên kho định nghĩa của WordNet."""
        unique_words = sorted(list(set(w.lower().strip() for w in words_iterable if w)))
        corpus = [self._get_gloss_text(w) for w in unique_words]

        # Bổ sung một số định nghĩa mẫu từ WordNet để làm giàu từ vựng
        sample_synsets = list(wn.all_synsets())[:2000]
        extra_docs = [s.definition() + " " + " ".join(s.examples()) for s in sample_synsets]
        corpus.extend(extra_docs)

        self.tfidf = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            stop_words="english",
            min_df=1
        )
        tfidf_matrix = self.tfidf.fit_transform(corpus)

        n_comp = min(self.n_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        if n_comp < 2:
            n_comp = 2
        self.n_components = n_comp

        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.svd.fit(tfidf_matrix)

        # Lưu cache vector SVD cho các từ huấn luyện
        for w in unique_words:
            self.get_word_vector(w)

        return self

    def get_word_vector(self, word: str) -> np.ndarray:
        """Lấy vector ngữ nghĩa 50 chiều đã chuẩn hóa L2 của một từ."""
        w = word.lower().strip()
        if w in self.vector_cache:
            return self.vector_cache[w]

        if self.tfidf is None or self.svd is None:
            return np.zeros(self.n_components, dtype=np.float32)

        gloss = self._get_gloss_text(w)
        tf = self.tfidf.transform([gloss])
        sv = self.svd.transform(tf)[0]
        norm = np.linalg.norm(sv)
        if norm > 0:
            sv = sv / norm
        else:
            sv = np.zeros(self.n_components, dtype=np.float32)

        self.vector_cache[w] = sv
        return sv

    def get_word_info(self, word: str) -> dict:
        """Trích xuất và lưu cache thông tin đồ thị WordNet của từ."""
        w = word.lower().strip()
        if w in self.info_cache:
            return self.info_cache[w]

        syns = wn.synsets(w)
        # Ưu tiên các synset phổ biến đầu tiên
        primary_syns = syns[:4]

        # 1. Lemmas
        lemmas = set(l.name().lower().replace("_", " ") for s in syns for l in s.lemmas())

        # 2. Similar_tos & Also_sees
        similar_syns = set(st for s in syns for st in s.similar_tos()) | set(asyn for s in syns for asyn in s.also_sees())
        similar_lemmas = set(l.name().lower().replace("_", " ") for s in similar_syns for l in s.lemmas())

        # 3. Direct antonyms
        antonyms = set()
        for s in syns:
            for l in s.lemmas():
                for a in l.antonyms():
                    antonyms.add(a.name().lower().replace("_", " "))

        # 4. Concepts (Hypernyms 2-hop + Attributes trên toàn bộ synsets)
        concepts = set()
        for s in syns:
            for a in s.attributes():
                concepts.add(a.name())
            for h in s.hypernyms():
                concepts.add(h.name())
                for hh in h.hypernyms():
                    concepts.add(hh.name())

        # 5. Lexnames & POS
        lexnames = set(s.lexname() for s in primary_syns)
        poses = set('a' if s.pos() in ('a', 's') else s.pos() for s in primary_syns)

        info = {
            "synset_names": [s.name() for s in syns],
            "primary_synset_names": [s.name() for s in primary_syns],
            "lemmas": lemmas,
            "similar_synset_names": [s.name() for s in similar_syns],
            "similar_lemmas": similar_lemmas,
            "antonyms": antonyms,
            "concepts": concepts,
            "lexnames": lexnames,
            "poses": poses
        }
        self.info_cache[w] = info
        return info

    def extract_pair_features(self, w1: str, w2: str) -> np.ndarray:
        """
        Trích xuất đúng 12 đặc trưng số cốt lõi cho một cặp từ (w1, w2).
        Đảm bảo không bị Curse of Dimensionality và xử lý hoàn hảo tính từ.
        """
        w1 = w1.lower().strip()
        w2 = w2.lower().strip()

        info1 = self.get_word_info(w1)
        info2 = self.get_word_info(w2)

        s1_set = set(info1["synset_names"])
        s2_set = set(info2["synset_names"])

        # 1. lemma_overlap: Tỷ lệ chia sẻ synset
        u_syns = s1_set | s2_set
        lemma_overlap = float(len(s1_set & s2_set) / len(u_syns)) if u_syns else 0.0

        # 2. similar_to_score: Mức độ liên kết qua similar_tos và also_sees
        if w2 in info1["similar_lemmas"] or w1 in info2["similar_lemmas"]:
            similar_to_score = 1.0
        else:
            sim_u = set(info1["similar_synset_names"]) | set(info2["similar_synset_names"])
            sim_i = set(info1["similar_synset_names"]) & set(info2["similar_synset_names"])
            similar_to_score = float(len(sim_i) / len(sim_u)) if sim_u else 0.0

        # 3. antonym_graph_score: Quan hệ đối lập trên đồ thị (Direct + Cluster)
        antonym_graph_score = 0.0
        # Direct lemma antonym
        if w2 in info1["antonyms"] or w1 in info2["antonyms"]:
            antonym_graph_score = 1.0
        elif (info1["antonyms"] & info2["lemmas"]) or (info2["antonyms"] & info1["lemmas"]):
            antonym_graph_score = 0.95
        elif (info1["antonyms"] & info2["similar_lemmas"]) or (info2["antonyms"] & info1["similar_lemmas"]):
            antonym_graph_score = 0.85

        # 4. shared_attribute_or_hypernym: Jaccard similarity hypernyms & attributes
        c_u = info1["concepts"] | info2["concepts"]
        c_i = info1["concepts"] & info2["concepts"]
        shared_attr_hyp = float(len(c_i) / len(c_u)) if c_u else 0.0

        # 5. max_path_sim & 6. max_wup_sim:
        max_path = 0.0
        max_wup = 0.0
        raw_s1 = [wn.synset(name) for name in info1["primary_synset_names"][:3]]
        raw_s2 = [wn.synset(name) for name in info2["primary_synset_names"][:3]]

        for x in raw_s1:
            for y in raw_s2:
                try:
                    p = x.path_similarity(y)
                    if p is not None and p > max_path:
                        max_path = float(p)
                    w = x.wup_similarity(y)
                    if w is not None and w > max_wup:
                        max_wup = float(w)
                except Exception:
                    pass

        # Chuẩn hóa mức tương đồng cho tính từ (khi path_similarity mặc định trả về None/thấp)
        if ('a' in info1["poses"] and 'a' in info2["poses"]) or max_path <= 0.10:
            if similar_to_score > 0.0:
                max_path = max(max_path, 0.50)
                max_wup = max(max_wup, 0.70)
            elif antonym_graph_score > 0.0:
                max_path = max(max_path, 0.33)
                max_wup = max(max_wup, 0.50)
            elif shared_attr_hyp > 0.0:
                max_path = max(max_path, 0.25)
                max_wup = max(max_wup, 0.40)
            elif (info1["lexnames"] & info2["lexnames"]) and (info1["poses"] & info2["poses"]):
                max_path = max(max_path, 0.12)
                max_wup = max(max_wup, 0.25)

        # 7. same_lexname: Cùng lĩnh vực từ điển
        lex_u = info1["lexnames"] | info2["lexnames"]
        same_lexname = float(len(info1["lexnames"] & info2["lexnames"]) / len(lex_u)) if lex_u else 0.0

        # 8. same_pos: Cùng từ loại
        pos_u = info1["poses"] | info2["poses"]
        same_pos = float(len(info1["poses"] & info2["poses"]) / len(pos_u)) if pos_u else 0.0

        # 9. gloss_cosine_sim: Độ tương đồng văn bản định nghĩa TF-IDF
        if self.tfidf is not None:
            g1 = self._get_gloss_text(w1)
            g2 = self._get_gloss_text(w2)
            tf1 = self.tfidf.transform([g1])
            tf2 = self.tfidf.transform([g2])
            n1 = np.linalg.norm(tf1.data)
            n2 = np.linalg.norm(tf2.data)
            if n1 > 0 and n2 > 0:
                gloss_cosine_sim = float(tf1.dot(tf2.T).toarray()[0, 0] / (n1 * n2))
            else:
                gloss_cosine_sim = 0.0
        else:
            gloss_cosine_sim = 0.0

        # 10. svd_cosine_sim & 11. svd_euclidean_dist:
        v1 = self.get_word_vector(w1)
        v2 = self.get_word_vector(w2)
        svd_cosine_sim = float(np.dot(v1, v2))
        svd_euclidean_dist = float(np.linalg.norm(v1 - v2))

        # 12. morph_opposition_flag: Cờ đối lập hình thái
        morph_opposition_flag = check_morphological_opposition(w1, w2)

        # Vector 12 chiều chính xác
        features = np.array([
            lemma_overlap,          # 1
            similar_to_score,       # 2
            antonym_graph_score,    # 3
            shared_attr_hyp,        # 4
            max_path,               # 5
            max_wup,                # 6
            same_lexname,           # 7
            same_pos,               # 8
            gloss_cosine_sim,       # 9
            svd_cosine_sim,         # 10
            svd_euclidean_dist,     # 11
            morph_opposition_flag   # 12
        ], dtype=np.float32)

        return features

    def transform(self, pairs: list) -> np.ndarray:
        """Trích xuất ma trận đặc trưng (N, 12) cho danh sách các cặp từ."""
        feature_list = []
        for p in pairs:
            w1, w2 = p[0], p[1]
            feature_list.append(self.extract_pair_features(w1, w2))
        return np.array(feature_list, dtype=np.float32)

    def get_feature_names(self) -> list:
        """Trả về danh sách 12 tên đặc trưng mô tả."""
        return [
            "lemma_overlap",
            "similar_to_score",
            "antonym_graph_score",
            "shared_attribute_or_hypernym",
            "max_path_sim",
            "max_wup_sim",
            "same_lexname",
            "same_pos",
            "gloss_cosine_sim",
            "svd_cosine_sim",
            "svd_euclidean_dist",
            "morph_opposition_flag"
        ]


if __name__ == "__main__":
    extractor = WordPairFeatureExtractor(n_components=50)
    words = ["tall", "short", "high", "tallish", "apple", "hot", "cold", "happy", "sad"]
    extractor.fit(words)

    test_pairs = [
        ("tall", "short"),
        ("tall", "high"),
        ("tall", "tallish"),
        ("tall", "apple"),
        ("hot", "cold"),
        ("happy", "sad")
    ]
    feats = extractor.transform(test_pairs)
    print(f"Kích thước ma trận đặc trưng: {feats.shape}")
    print(f"Số lượng đặc trưng: {len(extractor.get_feature_names())}")
    names = extractor.get_feature_names()
    for (w1, w2), row in zip(test_pairs, feats):
        print(f"\n({w1}, {w2}):")
        for n, val in zip(names, row):
            print(f"  {n:28}: {val:.4f}")
