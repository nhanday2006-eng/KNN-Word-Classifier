"""
word_explorer.py
Module chứa logic nghiệp vụ dùng chung cho cả giao diện GUI (app_gui.py)
và kịch bản kiểm thử tự động (qa_tester.py).
Đảm bảo kết quả kiểm thử và kết quả trên phần mềm đồng nhất 100%.
"""

import os
import sys
import joblib
import numpy as np
from nltk.corpus import wordnet as wn

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Các từ nghĩa bóng hiếm gặp cần loại bỏ khỏi danh sách gợi ý chính
EXCLUDE_FIGURATIVE_WORDS = {
    "magniloquent", "grandiloquent", "improbable", "marvelous", "marvellous",
    "rhetorical", "unplumbed", "unsounded"
}

_PIPELINE_CACHE = None


def get_cached_pipeline():
    """Tải và lưu đệm pipeline mô hình k-NN để tái sử dụng nhanh chóng."""
    global _PIPELINE_CACHE
    if _PIPELINE_CACHE is not None:
        return _PIPELINE_CACHE

    model_path = os.path.join(os.path.dirname(__file__), "models", "knn_pipeline.pkl")
    if os.path.exists(model_path):
        _PIPELINE_CACHE = joblib.load(model_path)
        return _PIPELINE_CACHE
    return None


def classify_word_pair(w1: str, w2: str, model=None, scaler=None, extractor=None):
    """
    Phân loại cặp từ (w1, w2) an toàn, xử lý triệt để các trường hợp ngoại lệ (Edge Cases).
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

    # Kiểm tra tồn tại trong WordNet
    s1 = wn.synsets(w1)
    s2 = wn.synsets(w2)
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
    Hàm xử lý nghiệp vụ Tab 2 (Word Explorer).
    Quét từ điển WordNet cho 1 từ đơn, loại bỏ nghĩa bóng, áp dụng ngưỡng tin cậy >= 70%
    và phân bổ chính xác vào 2 cột: Đồng nghĩa & Trái nghĩa.
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

    syns = wn.synsets(w)
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
        primary_senses = syns[:3]
    else:
        primary_senses = primary_senses[:3]

    candidates = set()
    for s in primary_senses:
        for lem in s.lemmas():
            name = lem.name().lower().replace("_", " ")
            if name.isalpha() and 2 <= len(name) <= 20 and name != w and name not in EXCLUDE_FIGURATIVE_WORDS:
                candidates.add(name)
            for ant in lem.antonyms():
                aname = ant.name().lower().replace("_", " ")
                if aname.isalpha() and 2 <= len(aname) <= 20 and aname != w and aname not in EXCLUDE_FIGURATIVE_WORDS:
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

    synonyms_found = []
    antonyms_found = []

    for cand in candidates:
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
                synonyms_found.append((cand, conf))
        elif pred == 0 and conf >= min_confidence:
            # Phải có cơ sở ngữ nghĩa trái nghĩa: antonym_graph > 0, morph_opposition > 0,
            # hoặc chia sẻ thuộc tính đối lập (tính từ) mà KHÔNG có quan hệ similar_to
            if feat[2] > 0 or feat[11] > 0 or (feat[3] > 0 and feat[1] == 0):
                # Tuyệt đối không xếp từ đồng nghĩa/similar_to vào cột trái nghĩa
                if feat[1] == 0 and feat[0] == 0:
                    antonyms_found.append((cand, conf))

    synonyms_found.sort(key=lambda x: x[1], reverse=True)
    antonyms_found.sort(key=lambda x: x[1], reverse=True)

    return {
        "status": "ok",
        "message": "Thành công",
        "synonyms": synonyms_found,
        "antonyms": antonyms_found
    }
