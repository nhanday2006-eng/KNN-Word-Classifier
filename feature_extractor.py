"""
feature_extractor.py
Trích xuất vector đặc trưng 12 chiều (12-dimensional feature vector) cô đọng, chất lượng cao
cho thuật toán k-Nearest Neighbors (k-NN), loại bỏ hoàn toàn hiện tượng Curse of Dimensionality.

Tích hợp:
1. Chuẩn hóa so sánh hơn / so sánh nhất (better -> good, worse -> bad).
2. Chuẩn hóa chính tả Anh - Anh (UK) vs Anh - Mỹ (US) (color - colour -> 100% Synonym).
3. Mở rộng họ từ phái sinh (derivationally_related_forms) & đồ thị bắc cầu 2 bước (2-Hop Antonyms/Synonyms).
4. Tích hợp Từ điển Cặp từ Đời thường (Everyday Commonsense Lexicon).
"""

import sys
import numpy as np
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.corpus import wordnet as wn

from commonsense_lexicon import (
    normalize_base_word,
    normalize_spelling,
    is_spelling_variant,
    is_commonsense_antonym,
    is_commonsense_synonym,
    get_commonsense_associations
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

NEGATION_PREFIXES = (
    "un", "in", "im", "ir", "il", "dis", "non", "ab", "de", "anti", "mis", "counter", "a"
)


def check_morphological_opposition(w1: str, w2: str) -> float:
    """
    [ĐẶC TRƯNG HÌNH THÁI HỌC - MORPHOLOGY]
    Kiểm tra xem hai từ có quan hệ đối lập qua tiền tố phủ định (un-, in-, im-, dis-,...)
    hoặc hậu tố (-ful vs -less) hay không.
    Trả về 1.0 nếu phát hiện quan hệ đối lập hình thái (ví dụ legal - illegal, visible - invisible),
    ngược lại trả về 0.0.
    """
    w1 = normalize_base_word(w1.lower().strip())
    w2 = normalize_base_word(w2.lower().strip())
    if len(w1) >= len(w2):
        w_long, w_short = w1, w2
    else:
        w_long, w_short = w2, w1

    # 1. Kiểm tra tiền tố đối lập (Ví dụ: illegal = il + legal)
    for p in NEGATION_PREFIXES:
        if w_long.startswith(p):
            rem = w_long[len(p):]
            if rem == w_short or (len(rem) >= 3 and SequenceMatcher(None, rem, w_short).ratio() >= 0.85):
                return 1.0

    # 2. Kiểm tra cặp hậu tố đối lập -ful vs -less (Ví dụ: useful vs useless)
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
        w = normalize_spelling(normalize_base_word(word.lower().strip()))
        syns = wn.synsets(w)[:4]
        if not syns and w != word:
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
        w = normalize_spelling(normalize_base_word(word.lower().strip()))
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
        """
        Trích xuất và lưu cache thông tin đồ thị WordNet của từ,
        bao gồm mở rộng họ từ phái sinh (derivationally_related_forms)
        và quan hệ bắc cầu 2 bước (2-Hop Antonyms/Synonyms).
        """
        orig_w = word.lower().strip()
        norm_w = normalize_base_word(orig_w)
        std_w = normalize_spelling(norm_w)

        cache_key = f"{orig_w}:{std_w}"
        if cache_key in self.info_cache:
            return self.info_cache[cache_key]

        # Gom synsets từ từ gốc, từ chuẩn hóa và biến thể
        syns_dict = {}
        for query_word in [orig_w, norm_w, std_w]:
            for s in wn.synsets(query_word):
                syns_dict[s.name()] = s

        # Thêm biến thể hình thái morphy (verb, noun, adj)
        for pos in (wn.VERB, wn.NOUN, wn.ADJ):
            try:
                m = wn.morphy(orig_w, pos)
                if m and m not in (orig_w, norm_w, std_w):
                    for s in wn.synsets(m):
                        syns_dict[s.name()] = s
            except Exception:
                pass

        # Xử lý cắt hậu tố phổ biến: -ing, -ed, -ly, -ness (ví dụ: boring -> bore)
        for suffix in ("ing", "ed", "ly", "ness"):
            if orig_w.endswith(suffix) and len(orig_w) > len(suffix) + 2:
                stem = orig_w[:-len(suffix)]
                for pos in (wn.VERB, wn.NOUN, wn.ADJ):
                    try:
                        m = wn.morphy(stem, pos)
                        if m:
                            for s in wn.synsets(m):
                                syns_dict[s.name()] = s
                    except Exception:
                        pass
                if stem.endswith("e") or orig_w.endswith("ing"):
                    stem_e = stem if stem.endswith("e") else stem + "e"
                    for s in wn.synsets(stem_e):
                        syns_dict[s.name()] = s

        all_syns = list(syns_dict.values())
        primary_syns = all_syns[:5] if all_syns else []

        # 1. Lemmas
        lemmas = set(l.name().lower().replace("_", " ") for s in all_syns for l in s.lemmas())
        lemmas.add(orig_w)
        lemmas.add(norm_w)
        lemmas.add(std_w)

        # 2. Similar_tos & Also_sees
        similar_syns = set(st for s in all_syns for st in s.similar_tos()) | set(asyn for s in all_syns for asyn in s.also_sees())
        similar_lemmas = set(l.name().lower().replace("_", " ") for s in similar_syns for l in s.lemmas())

        # 3. Direct antonyms
        direct_antonyms = set()
        for s in all_syns:
            for l in s.lemmas():
                for a in l.antonyms():
                    direct_antonyms.add(a.name().lower().replace("_", " "))

        # 4. Derivationally Related Forms (Họ từ phái sinh) & Derivational Antonyms
        derivational_lemmas = set()
        derivational_antonyms = set()
        for s in all_syns:
            for l in s.lemmas():
                for drf in l.derivationally_related_forms():
                    drf_name = drf.name().lower().replace("_", " ")
                    derivational_lemmas.add(drf_name)
                    for a in drf.antonyms():
                        derivational_antonyms.add(a.name().lower().replace("_", " "))
                    # Cũng gom synset của drf
                    drf_syn = drf.synset()
                    for sim in drf_syn.similar_tos():
                        for sl in sim.lemmas():
                            derivational_lemmas.add(sl.name().lower().replace("_", " "))

        # 5. Cluster Antonyms (2-Hop: Antonyms của similar_tos và also_sees)
        cluster_antonyms = set()
        for s in similar_syns:
            for l in s.lemmas():
                for a in l.antonyms():
                    cluster_antonyms.add(a.name().lower().replace("_", " "))

        # 6. Commonsense associations (Từ điển đời thường)
        cs = get_commonsense_associations(orig_w)
        cs_syns = cs["synonyms"]
        cs_ants = cs["antonyms"]

        # 7. Concepts (Hypernyms 2-hop + Attributes trên toàn bộ synsets)
        concepts = set()
        for s in all_syns:
            for a in s.attributes():
                concepts.add(a.name())
            for h in s.hypernyms():
                concepts.add(h.name())
                for hh in h.hypernyms():
                    concepts.add(hh.name())

        # 8. Lexnames & POS
        lexnames = set(s.lexname() for s in primary_syns)
        poses = set('a' if s.pos() in ('a', 's') else s.pos() for s in primary_syns)

        info = {
            "synset_names": [s.name() for s in all_syns],
            "primary_synset_names": [s.name() for s in primary_syns],
            "lemmas": lemmas,
            "similar_synset_names": [s.name() for s in similar_syns],
            "similar_lemmas": similar_lemmas,
            "antonyms": direct_antonyms,
            "derivational_lemmas": derivational_lemmas,
            "derivational_antonyms": derivational_antonyms,
            "cluster_antonyms": cluster_antonyms,
            "cs_synonyms": cs_syns,
            "cs_antonyms": cs_ants,
            "concepts": concepts,
            "lexnames": lexnames,
            "poses": poses
        }
        self.info_cache[cache_key] = info
        return info

    def extract_pair_features(self, w1: str, w2: str) -> np.ndarray:
        """
        Trích xuất đúng 12 đặc trưng số cốt lõi cho một cặp từ (w1, w2).
        Đảm bảo không bị Curse of Dimensionality và xử lý hoàn hảo tính từ,
        so sánh hơn, chính tả UK/US, và cặp từ đời thường.
        """
        w1_orig = w1.lower().strip()
        w2_orig = w2.lower().strip()

        # =====================================================================
        # KIỂM TRA CHÍNH TẢ ANH - ANH (UK) vs ANH - MỸ (US)
        # =====================================================================
        if is_spelling_variant(w1_orig, w2_orig):
            # Cặp từ chính tả là Đồng nghĩa 100%
            return np.array([
                1.0,  # 1. lemma_overlap
                1.0,  # 2. similar_to_score
                0.0,  # 3. antonym_graph_score
                1.0,  # 4. shared_attr_hyp
                1.0,  # 5. max_path
                1.0,  # 6. max_wup
                1.0,  # 7. same_lexname
                1.0,  # 8. same_pos
                1.0,  # 9. gloss_cosine_sim
                1.0,  # 10. svd_cosine_sim
                0.0,  # 11. svd_euclidean_dist
                0.0   # 12. morph_opposition_flag
            ], dtype=np.float32)

        # =====================================================================
        # CHUẨN HÓA TỪ GỐC (SO SÁNH HƠN / SO SÁNH NHẤT)
        # =====================================================================
        w1_norm = normalize_base_word(w1_orig)
        w2_norm = normalize_base_word(w2_orig)

        # Kiểm tra kho từ điển đời thường
        is_cs_ant = is_commonsense_antonym(w1_orig, w2_orig) or is_commonsense_antonym(w1_norm, w2_norm)
        is_cs_syn = is_commonsense_synonym(w1_orig, w2_orig) or is_commonsense_synonym(w1_norm, w2_norm)

        info1 = self.get_word_info(w1_orig)
        info2 = self.get_word_info(w2_orig)

        s1_set = set(info1["synset_names"])
        s2_set = set(info2["synset_names"])

        # 1. lemma_overlap: Tỷ lệ chia sẻ synset
        u_syns = s1_set | s2_set
        lemma_overlap = float(len(s1_set & s2_set) / len(u_syns)) if u_syns else 0.0
        if w1_norm == w2_norm:
            lemma_overlap = 1.0

        # 2. similar_to_score: Mức độ liên kết qua similar_tos, also_sees và họ từ phái sinh
        if w2_orig in info1["similar_lemmas"] or w1_orig in info2["similar_lemmas"]:
            similar_to_score = 1.0
        elif w2_norm in info1["similar_lemmas"] or w1_norm in info2["similar_lemmas"]:
            similar_to_score = 1.0
        elif w2_orig in info1["cs_synonyms"] or w1_orig in info2["cs_synonyms"]:
            similar_to_score = 1.0
        elif info1["derivational_lemmas"] & info2["lemmas"] or info2["derivational_lemmas"] & info1["lemmas"]:
            similar_to_score = 0.85
        else:
            sim_u = set(info1["similar_synset_names"]) | set(info2["similar_synset_names"])
            sim_i = set(info1["similar_synset_names"]) & set(info2["similar_synset_names"])
            similar_to_score = float(len(sim_i) / len(sim_u)) if sim_u else 0.0

        # 3. antonym_graph_score: Quan hệ đối lập trên đồ thị (Direct + Derivational + Cluster 2-Hop)
        antonym_graph_score = 0.0
        if is_cs_ant:
            antonym_graph_score = 1.0
        elif w2_orig in info1["antonyms"] or w1_orig in info2["antonyms"]:
            antonym_graph_score = 1.0
        elif w2_norm in info1["antonyms"] or w1_norm in info2["antonyms"]:
            antonym_graph_score = 1.0
        elif w2_orig in info1["cs_antonyms"] or w1_orig in info2["cs_antonyms"]:
            antonym_graph_score = 1.0
        elif (info1["antonyms"] & info2["lemmas"]) or (info2["antonyms"] & info1["lemmas"]):
            antonym_graph_score = 0.95
        elif (info1["antonyms"] & info2["similar_lemmas"]) or (info2["antonyms"] & info1["similar_lemmas"]):
            antonym_graph_score = 0.90
        elif (info1["cluster_antonyms"] & info2["lemmas"]) or (info2["cluster_antonyms"] & info1["lemmas"]):
            antonym_graph_score = 0.90
        elif (info1["derivational_antonyms"] & info2["lemmas"]) or (info2["derivational_antonyms"] & info1["lemmas"]):
            antonym_graph_score = 0.85
        elif (info1["derivational_antonyms"] & info2["similar_lemmas"]) or (info2["derivational_antonyms"] & info1["similar_lemmas"]):
            antonym_graph_score = 0.80

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
            g1 = self._get_gloss_text(w1_orig)
            g2 = self._get_gloss_text(w2_orig)
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
        v1 = self.get_word_vector(w1_orig)
        v2 = self.get_word_vector(w2_orig)
        svd_cosine_sim = float(np.dot(v1, v2))
        svd_euclidean_dist = float(np.linalg.norm(v1 - v2))

        # 12. morph_opposition_flag: Cờ đối lập hình thái
        morph_opposition_flag = check_morphological_opposition(w1_orig, w2_orig)

        # =====================================================================
        # HIỆU CHỈNH TRỌNG SỐ CHO COMMONSENSE LEXICON
        # =====================================================================
        if is_cs_ant:
            antonym_graph_score = 1.0
            similar_to_score = 0.0
            lemma_overlap = 0.0
            shared_attr_hyp = max(shared_attr_hyp, 0.35)
            max_path = max(max_path, 0.33)
            max_wup = max(max_wup, 0.50)
            same_pos = max(same_pos, 0.50)
            svd_cosine_sim = max(svd_cosine_sim, 0.50)
            svd_euclidean_dist = min(svd_euclidean_dist, 1.0)
        elif is_cs_syn:
            antonym_graph_score = 0.0
            morph_opposition_flag = 0.0
            similar_to_score = 1.0
            lemma_overlap = max(lemma_overlap, 0.50)
            shared_attr_hyp = max(shared_attr_hyp, 0.40)
            max_path = max(max_path, 0.50)
            max_wup = max(max_wup, 0.70)
            same_pos = max(same_pos, 0.50)
            gloss_cosine_sim = max(gloss_cosine_sim, 0.30)
            svd_cosine_sim = max(svd_cosine_sim, 0.70)
            svd_euclidean_dist = min(svd_euclidean_dist, 0.60)

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
        return [
            "lemma_overlap",
            "similar_to_score",
            "antonym_graph_score",
            "shared_attr_hyp",
            "max_path",
            "max_wup",
            "same_lexname",
            "same_pos",
            "gloss_cosine_sim",
            "svd_cosine_sim",
            "svd_euclidean_dist",
            "morph_opposition_flag"
        ]
