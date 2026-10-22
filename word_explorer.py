"""
word_explorer.py
Module xử lý nghiệp vụ cho Tab 1 (Pair Classifier) và Tab 2 (Word Explorer).
Cung cấp các hàm phân loại cặp từ an toàn và khám phá từ đơn lẻ với bộ lọc ngữ nghĩa chất lượng cao.

Tích hợp:
- Chuẩn hóa từ gốc (So sánh hơn/nhất).
- Chuẩn hóa chính tả UK vs US.
- Tích hợp Từ điển Cặp từ Đời thường (Everyday Commonsense Lexicon).
"""

import os
import joblib
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

# Danh sách các từ nghĩa bóng, lóng hoặc văn học cổ ít dùng cần loại bỏ khi tra từ thông dụng
EXCLUDE_FIGURATIVE_WORDS = {
    "magniloquent", "grandiloquent", "improbable", "marvellous", "marvelous",
    "fabulous", "prodigious", "unsubstantial", "incredible", "tallness", "shortness"
}

_PIPELINE_CACHE = None


def get_cached_pipeline():
    """Nạp và lưu cache pipeline mô hình k-NN từ models/knn_pipeline.pkl."""
    global _PIPELINE_CACHE
    if _PIPELINE_CACHE is not None:
        return _PIPELINE_CACHE

    model_path = os.path.join(os.path.dirname(__file__), "models", "knn_pipeline.pkl")
    if os.path.exists(model_path):
        try:
            _PIPELINE_CACHE = joblib.load(model_path)
            return _PIPELINE_CACHE
        except Exception:
            pass
    return None


def classify_word_pair(w1: str, w2: str, model=None, scaler=None, extractor=None):
    """
    [PHÂN LOẠI AN TOÀN CHO TAB 1 & API]
    Nhận 2 từ w1, w2, kiểm tra lỗi đầu vào theo cơ chế Defensive Programming,
    xử lý so sánh hơn, biến thể UK/US, sau đó trích xuất 12 đặc trưng,
    chuẩn hóa qua scaler và gọi model.predict & model.predict_proba.
    Không bao giờ văng Exception làm sập ứng dụng.
    """
    w1 = (w1 or "").strip().lower()
    w2 = (w2 or "").strip().lower()

    if not w1 or not w2:
        return {
            "status": "error",
            "message": "Vui lòng nhập đầy đủ cả hai từ.",
            "pred_label": 2,
            "pred_name": "Unrelated (Không liên quan)",
            "probabilities": [0.0, 0.0, 1.0]
        }

    if not w1.isalpha() or not w2.isalpha():
        return {
            "status": "error",
            "message": "Từ nhập vào chỉ được chứa chữ cái tiếng Anh.",
            "pred_label": 2,
            "pred_name": "Unrelated (Không liên quan)",
            "probabilities": [0.0, 0.0, 1.0]
        }

    # Kiểm tra biến thể chính tả UK/US (Ví dụ: color - colour)
    if is_spelling_variant(w1, w2):
        return {
            "status": "ok",
            "message": "Biến thể chính tả UK/US (100% Đồng nghĩa)",
            "pred_label": 1,
            "pred_name": "Synonym (Đồng nghĩa)",
            "probabilities": [0.0, 1.0, 0.0],
            "feature_vector": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        }

    # Kiểm tra tồn tại trong WordNet (kể cả dạng gốc và chính tả chuẩn)
    s1 = wn.synsets(w1) or wn.synsets(normalize_base_word(w1)) or wn.synsets(normalize_spelling(w1))
    s2 = wn.synsets(w2) or wn.synsets(normalize_base_word(w2)) or wn.synsets(normalize_spelling(w2))

    if not s1 or not s2:
        missing = []
        if not s1:
            missing.append(w1)
        if not s2:
            missing.append(w2)
        return {
            "status": "not_found",
            "message": f"Từ '{', '.join(missing)}' không tồn tại trong từ điển WordNet.",
            "pred_label": 2,
            "pred_name": "Unrelated (Không liên quan)",
            "probabilities": [0.0, 0.0, 1.0]
        }

    # Nạp pipeline nếu chưa truyền vào
    if model is None or scaler is None or extractor is None:
        pipe = get_cached_pipeline()
        if pipe is None:
            return {
                "status": "error",
                "message": "Chưa tìm thấy mô hình k-NN đã huấn luyện.",
                "pred_label": 2,
                "pred_name": "Unrelated (Không liên quan)",
                "probabilities": [0.0, 0.0, 1.0]
            }
        model = pipe["model"]
        scaler = pipe["scaler"]
        extractor = pipe["extractor"]

    try:
        feat = extractor.extract_pair_features(w1, w2)
        fs = scaler.transform([feat])
        pred = int(model.predict(fs)[0])
        probs = model.predict_proba(fs)[0].tolist()

        # Hiệu chỉnh độ tin cậy nếu nằm trong kho từ điển đời thường
        if is_commonsense_antonym(w1, w2):
            pred = 0
            probs = [1.0, 0.0, 0.0]
        elif is_commonsense_synonym(w1, w2):
            pred = 1
            probs = [0.0, 1.0, 0.0]

        label_map = {
            0: "Antonym (Trái nghĩa)",
            1: "Synonym (Đồng nghĩa)",
            2: "Unrelated (Không liên quan)"
        }

        return {
            "status": "ok",
            "message": "Thành công",
            "pred_label": pred,
            "pred_name": label_map.get(pred, "Unrelated"),
            "probabilities": probs,
            "feature_vector": feat.tolist()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Lỗi tính toán: {e}",
            "pred_label": 2,
            "pred_name": "Unrelated (Không liên quan)",
            "probabilities": [0.0, 0.0, 1.0]
        }


def explore_single_word(target_word: str, model=None, scaler=None, extractor=None, min_confidence: float = 70.0):
    """
    [NGHIỆP VỤ NÂNG CAO TAB 2 - WORD EXPLORER]
    Khám phá từ đơn: Người dùng nhập 1 từ (ví dụ 'tall' hoặc 'fun'), hàm sẽ:
    1. Quét tìm các từ ứng viên liên quan trong WordNet (synsets, similar_tos, attributes, antonyms)
       và bổ sung từ điển đời thường (Everyday Commonsense Associations).
    2. Loại bỏ các từ nghĩa bóng hiếm gặp (figurative words) như magniloquent, improbable.
    3. Đưa từng ứng viên vào mô hình k-NN phân loại.
    4. Áp dụng bộ lọc ngưỡng tự tin khắt khe (min_confidence >= 70%) và kiểm tra cơ sở ngữ nghĩa
       để phân chia chính xác vào 2 danh sách: Từ đồng nghĩa & Từ trái nghĩa.
    """
    w = (target_word or "").strip().lower()

    if not w:
        return {
            "status": "error",
            "message": "Ô nhập từ đang trống. Vui lòng nhập một từ tiếng Anh.",
            "synonyms": [],
            "antonyms": []
        }

    if not w.isalpha():
        return {
            "status": "error",
            "message": f"'{w}' không hợp lệ. Vui lòng chỉ nhập các chữ cái tiếng Anh.",
            "synonyms": [],
            "antonyms": []
        }

    syns = wn.synsets(w) or wn.synsets(normalize_base_word(w)) or wn.synsets(normalize_spelling(w))
    if not syns:
        return {
            "status": "not_found",
            "message": f"Từ '{w}' không tồn tại trong từ điển WordNet. Vui lòng kiểm tra lại chính tả!",
            "synonyms": [],
            "antonyms": []
        }

    # Nạp pipeline
    if model is None or scaler is None or extractor is None:
        pipe = get_cached_pipeline()
        if pipe is None:
            return {
                "status": "error",
                "message": "Chưa nạp được pipeline mô hình k-NN.",
                "synonyms": [],
                "antonyms": []
            }
        model = pipe["model"]
        scaler = pipe["scaler"]
        extractor = pipe["extractor"]

    # Ưu tiên các nghĩa chính (primary senses), loại bỏ các nghĩa bóng hiếm gặp
    primary_senses = [s for s in syns if s.lemmas()[0].name().lower().replace("_", " ") == w]
    if not primary_senses:
        primary_senses = syns[:4]
    else:
        primary_senses = primary_senses[:4]

    candidates = set()

    # 1. Quét ứng viên từ WordNet
    for s in primary_senses:
        for lem in s.lemmas():
            name = lem.name().lower().replace("_", " ")
            if name.isalpha() and 2 <= len(name) <= 20 and name != w and name not in EXCLUDE_FIGURATIVE_WORDS:
                candidates.add(name)
            for ant in lem.antonyms():
                aname = ant.name().lower().replace("_", " ")
                if aname.isalpha() and 2 <= len(name) <= 20 and name != w and aname not in EXCLUDE_FIGURATIVE_WORDS:
                    candidates.add(aname)
        for sim in s.similar_tos():
            for lem in sim.lemmas():
                name = lem.name().lower().replace("_", " ")
                if name.isalpha() and 2 <= len(name) <= 20 and name != w and name not in EXCLUDE_FIGURATIVE_WORDS:
                    candidates.add(name)
        for also in s.also_sees():
            for lem in also.lemmas():
                name = lem.name().lower().replace("_", " ")
                if name.isalpha() and 2 <= len(name) <= 20 and name != w and name not in EXCLUDE_FIGURATIVE_WORDS:
                    candidates.add(name)
        for attr in s.attributes():
            for sattr in wn.synsets(attr.name().split(".")[0])[:2]:
                for lem in sattr.lemmas():
                    name = lem.name().lower().replace("_", " ")
                    if name.isalpha() and 2 <= len(name) <= 20 and name != w and name not in EXCLUDE_FIGURATIVE_WORDS:
                        candidates.add(name)

    # 2. Bổ sung ứng viên từ Từ điển Cặp từ Đời thường (Commonsense Associations)
    cs = get_commonsense_associations(w)
    for c_syn in cs["synonyms"]:
        if c_syn != w and c_syn.isalpha() and c_syn not in EXCLUDE_FIGURATIVE_WORDS:
            candidates.add(c_syn)
    for c_ant in cs["antonyms"]:
        if c_ant != w and c_ant.isalpha() and c_ant not in EXCLUDE_FIGURATIVE_WORDS:
            candidates.add(c_ant)

    syn_dict = {}
    ant_dict = {}

    for cand in candidates:
        # Kiểm tra ưu tiên từ điển đời thường
        is_cs_a = is_commonsense_antonym(w, cand)
        is_cs_s = is_commonsense_synonym(w, cand) or is_spelling_variant(w, cand)

        if is_cs_a:
            ant_dict[cand] = 100.0
            continue
        elif is_cs_s:
            syn_dict[cand] = 100.0
            continue

        feat = extractor.extract_pair_features(w, cand)
        fs = scaler.transform([feat])
        pred = model.predict(fs)[0]
        prob = model.predict_proba(fs)[0]
        conf = float(prob[pred] * 100)

        # BỘ LỌC NGƯỠNG TỰ TIN CHẶT CHẼ
        # feat index:
        # 0: lemma_overlap, 1: similar_to_score, 2: antonym_graph_score, 3: shared_attr_hyp, 11: morph_opposition
        if pred == 1 and conf >= min_confidence:
            # Phải có cơ sở ngữ nghĩa đồng nghĩa: chia sẻ synset, similar_to, hoặc gloss/svd cao
            if feat[0] > 0 or feat[1] > 0 or feat[3] > 0 or feat[8] > 0.05 or feat[9] > 0.30:
                syn_dict[cand] = max(syn_dict.get(cand, 0.0), conf)
        elif pred == 0 and conf >= min_confidence:
            # Phải có cơ sở ngữ nghĩa trái nghĩa: antonym_graph > 0, morph_opposition > 0,
            # hoặc chia sẻ thuộc tính đối lập (tính từ) mà KHÔNG có quan hệ similar_to
            if feat[2] > 0 or feat[11] > 0 or (feat[3] > 0 and feat[1] == 0):
                # Tuyệt đối không xếp từ đồng nghĩa/similar_to vào cột trái nghĩa
                if feat[1] == 0 and feat[0] == 0:
                    ant_dict[cand] = max(ant_dict.get(cand, 0.0), conf)

    synonyms_found = sorted(syn_dict.items(), key=lambda x: x[1], reverse=True)
    antonyms_found = sorted(ant_dict.items(), key=lambda x: x[1], reverse=True)

    return {
        "status": "ok",
        "message": "Thành công",
        "synonyms": synonyms_found,
        "antonyms": antonyms_found
    }
