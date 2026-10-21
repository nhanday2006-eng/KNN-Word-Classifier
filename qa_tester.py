"""
qa_tester.py
Kịch bản kiểm thử tự động toàn diện (Automated QA Test Suite)
Đóng vai Senior QA Automation Tester kiêm ML Engineer:
Kiểm tra tự động 5 nhóm bài test khắt khe trên mô hình k-NN và module word_explorer.
Tuyệt đối KHÔNG mở giao diện đồ họa GUI.
"""

import os
import sys
import time
import joblib

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from word_explorer import explore_single_word, classify_word_pair

# Bảng mã ANSI màu cho console
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_BOLD = "\033[1m"
COLOR_RESET = "\033[0m"


def print_header(title):
    print("\n" + "=" * 75)
    print(f" {COLOR_BOLD}{COLOR_CYAN}{title}{COLOR_RESET}")
    print("=" * 75)


def run_qa_suite():
    t_start = time.time()
    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    print_header("AUTOMATED QA TEST SUITE - k-NN WORD PAIR CLASSIFIER")

    # Nạp mô hình
    model_path = os.path.join(os.path.dirname(__file__), "models", "knn_pipeline.pkl")
    if not os.path.exists(model_path):
        print(f"{COLOR_RED}[ERROR] Không tìm thấy tệp mô hình: {model_path}{COLOR_RESET}")
        return 1

    pipe = joblib.load(model_path)
    model = pipe["model"]
    scaler = pipe["scaler"]
    extractor = pipe["extractor"]
    print(f"[Init] Đã nạp thành công mô hình k-NN (k = {pipe['best_k']}, Đặc trưng: {pipe['train_features_scaled'].shape[1]} chiều)")

    # =========================================================================
    # NHÓM 1: Cặp từ Trái nghĩa kinh điển (Antonyms - Label 0)
    # =========================================================================
    print_header("NHÓM 1: Cặp từ Trái nghĩa kinh điển (Kỳ vọng: Antonym - Nhãn 0)")
    group1 = [
        ("tall", "short"), ("hot", "cold"), ("big", "small"), ("fast", "slow"),
        ("happy", "sad"), ("rich", "poor"), ("strong", "weak"), ("legal", "illegal"),
        ("visible", "invisible"), ("useful", "useless"), ("good", "bad"), ("bright", "dark")
    ]

    for w1, w2 in group1:
        total_tests += 1
        res = classify_word_pair(w1, w2, model=model, scaler=scaler, extractor=extractor)
        pred = res["pred_label"]
        conf = res["probabilities"][pred] * 100
        if pred == 0:
            passed_tests += 1
            print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} ({w1:9}, {w2:9}) => Antonym (0) | Conf: {conf:.1f}%")
        else:
            failed_tests += 1
            print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} ({w1:9}, {w2:9}) => Nhận {pred} ({res['pred_name']}) | Conf: {conf:.1f}%")

    # =========================================================================
    # NHÓM 2: Cặp từ Đồng nghĩa kinh điển (Synonyms - Label 1)
    # =========================================================================
    print_header("NHÓM 2: Cặp từ Đồng nghĩa kinh điển (Kỳ vọng: Synonym - Nhãn 1)")
    group2 = [
        ("happy", "joyful"), ("fast", "quick"), ("big", "large"), ("small", "tiny"),
        ("smart", "intelligent"), ("buy", "purchase"), ("start", "begin"), ("rich", "wealthy")
    ]

    for w1, w2 in group2:
        total_tests += 1
        res = classify_word_pair(w1, w2, model=model, scaler=scaler, extractor=extractor)
        pred = res["pred_label"]
        conf = res["probabilities"][pred] * 100
        if pred == 1:
            passed_tests += 1
            print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} ({w1:9}, {w2:9}) => Synonym (1) | Conf: {conf:.1f}%")
        else:
            failed_tests += 1
            print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} ({w1:9}, {w2:9}) => Nhận {pred} ({res['pred_name']}) | Conf: {conf:.1f}%")

    # =========================================================================
    # NHÓM 3: Cặp từ Không liên quan & Cùng nhóm sự vật (Unrelated - Label 2)
    # =========================================================================
    print_header("NHÓM 3: Cặp từ Không liên quan & Coordinate Terms (Kỳ vọng: Unrelated - Nhãn 2)")
    group3 = [
        ("tall", "apple"), ("hot", "computer"), ("apple", "banana"), ("table", "chair"),
        ("dog", "cat"), ("car", "bicycle"), ("pizza", "music")
    ]

    for w1, w2 in group3:
        total_tests += 1
        res = classify_word_pair(w1, w2, model=model, scaler=scaler, extractor=extractor)
        pred = res["pred_label"]
        conf = res["probabilities"][pred] * 100
        if pred == 2:
            passed_tests += 1
            print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} ({w1:9}, {w2:9}) => Unrelated (2) | Conf: {conf:.1f}%")
        else:
            failed_tests += 1
            print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} ({w1:9}, {w2:9}) => Nhận {pred} ({res['pred_name']}) | Conf: {conf:.1f}%")

    # =========================================================================
    # NHÓM 4: Kiểm tra logic Tab 2 (Word Explorer - Tra cứu 1 từ)
    # =========================================================================
    print_header("NHÓM 4: Kiểm tra logic Tab 2 (Word Explorer - Tra cứu 1 từ)")

    # 4.1 Tra từ tall
    total_tests += 1
    res_tall = explore_single_word("tall", model=model, scaler=scaler, extractor=extractor)
    syns_tall = [x[0] for x in res_tall["synonyms"]]
    ants_tall = [x[0] for x in res_tall["antonyms"]]

    cond_tall = (
        "short" in ants_tall and
        "tallish" not in ants_tall and
        "long" not in ants_tall and
        "rangy" not in ants_tall and
        "magniloquent" not in (syns_tall + ants_tall) and
        "grandiloquent" not in (syns_tall + ants_tall) and
        "improbable" not in (syns_tall + ants_tall)
    )

    if cond_tall:
        passed_tests += 1
        print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} Tra từ 'tall': Trái nghĩa chứa 'short'; Đồng nghĩa ({len(syns_tall)} từ); Đã lọc sạch nghĩa bóng")
    else:
        failed_tests += 1
        print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} Tra từ 'tall': Không đạt yêu cầu lọc. Syns={syns_tall[:4]}, Ants={ants_tall}")

    # 4.2 Tra từ hot
    total_tests += 1
    res_hot = explore_single_word("hot", model=model, scaler=scaler, extractor=extractor)
    syns_hot = [x[0] for x in res_hot["synonyms"]]
    ants_hot = [x[0] for x in res_hot["antonyms"]]

    cond_hot = ("warm" in syns_hot and "cold" in ants_hot)
    if cond_hot:
        passed_tests += 1
        print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} Tra từ 'hot': Cột Đồng nghĩa có 'warm'; Cột Trái nghĩa có 'cold'")
    else:
        failed_tests += 1
        print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} Tra từ 'hot': Thiếu 'warm' trong Syns hoặc 'cold' trong Ants")

    # 4.3 Tra từ happy
    total_tests += 1
    res_happy = explore_single_word("happy", model=model, scaler=scaler, extractor=extractor)
    syns_happy = [x[0] for x in res_happy["synonyms"]]
    ants_happy = [x[0] for x in res_happy["antonyms"]]

    cond_happy = (
        ("joyful" in syns_happy or "glad" in syns_happy) and
        ("sad" in ants_happy or "unhappy" in ants_happy)
    )
    if cond_happy:
        passed_tests += 1
        print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} Tra từ 'happy': Cột Đồng nghĩa có 'glad'/'joyful'; Cột Trái nghĩa có 'unhappy'/'sad'")
    else:
        failed_tests += 1
        print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} Tra từ 'happy': Thiếu từ đồng nghĩa hoặc trái nghĩa đạt chuẩn")

    # =========================================================================
    # NHÓM 5: Edge Cases (Từ không có trái nghĩa / Từ không tồn tại / Rỗng)
    # =========================================================================
    print_header("NHÓM 5: Edge Cases (Danh từ cụ thể / Lỗi nhập liệu / Rỗng)")

    # 5.1 Danh từ cụ thể không có từ trái nghĩa
    for noun in ["apple", "pizza", "computer", "table"]:
        total_tests += 1
        res_noun = explore_single_word(noun, model=model, scaler=scaler, extractor=extractor)
        if len(res_noun["antonyms"]) == 0:
            passed_tests += 1
            print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} Tra danh từ '{noun}': Danh sách Trái nghĩa rỗng (len = 0)")
        else:
            failed_tests += 1
            print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} Tra danh từ '{noun}': Nhận nhầm trái nghĩa: {res_noun['antonyms']}")

    # 5.2 Từ sai chính tả / không có trong WordNet
    total_tests += 1
    res_oov = explore_single_word("asdfghjkl", model=model, scaler=scaler, extractor=extractor)
    if res_oov["status"] == "not_found" and len(res_oov["synonyms"]) == 0 and len(res_oov["antonyms"]) == 0:
        passed_tests += 1
        print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} Tra từ không tồn tại 'asdfghjkl': Báo lỗi an toàn (status='not_found', không crash)")
    else:
        failed_tests += 1
        print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} Tra từ 'asdfghjkl' không xử lý an toàn: {res_oov}")

    # 5.3 Chuỗi rỗng ở Tab 2
    total_tests += 1
    res_empty_tab2 = explore_single_word("", model=model, scaler=scaler, extractor=extractor)
    if res_empty_tab2["status"] == "error":
        passed_tests += 1
        print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} Chuỗi rỗng ở Tab 2: Báo lỗi an toàn (status='error', không crash)")
    else:
        failed_tests += 1
        print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} Chuỗi rỗng Tab 2 không xử lý an toàn: {res_empty_tab2}")

    # 5.4 Cặp từ rỗng ở Tab 1
    total_tests += 1
    res_empty_tab1 = classify_word_pair("", "", model=model, scaler=scaler, extractor=extractor)
    if res_empty_tab1["status"] == "error":
        passed_tests += 1
        print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} Cặp từ rỗng ở Tab 1: Báo lỗi an toàn (status='error', không crash)")
    else:
        failed_tests += 1
        print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} Cặp từ rỗng Tab 1 không xử lý an toàn: {res_empty_tab1}")

    # 5.5 Cặp từ OOV ở Tab 1
    total_tests += 1
    res_oov_tab1 = classify_word_pair("asdfghjkl", "hot", model=model, scaler=scaler, extractor=extractor)
    if res_oov_tab1["status"] == "not_found":
        passed_tests += 1
        print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} Cặp từ OOV ở Tab 1: Báo lỗi an toàn (status='not_found', không crash)")
    else:
        failed_tests += 1
        print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} Cặp từ OOV Tab 1 không xử lý an toàn: {res_oov_tab1}")

    # =========================================================================
    # TỔNG KẾT BÁO CÁO QA
    # =========================================================================
    elapsed = time.time() - t_start
    print_header("TỔNG KẾT BÁO CÁO KIỂM THỬ (QA AUTOMATION SUMMARY)")
    print(f"  + Tổng số bài kiểm thử (Total Tests) : {COLOR_BOLD}{total_tests}{COLOR_RESET}")
    print(f"  + Số bài ĐẠT (Passed)                : {COLOR_GREEN}{COLOR_BOLD}{passed_tests}{COLOR_RESET}")
    print(f"  + Số bài KHÔNG ĐẠT (Failed)          : {COLOR_RED if failed_tests > 0 else COLOR_GREEN}{COLOR_BOLD}{failed_tests}{COLOR_RESET}")
    print(f"  + Tỷ lệ thành công (Pass Rate)       : {COLOR_BOLD}{passed_tests / total_tests * 100:.2f}%{COLOR_RESET}")
    print(f"  + Thời gian chạy kiểm thử            : {elapsed:.2f} giây")

    if failed_tests == 0:
        print(f"\n{COLOR_GREEN}{COLOR_BOLD}>>> HOÀN TOÀN XUẤT SẮC: 100% TẤT CẢ TEST CASES ĐỀU ĐẠT CHUẨN! <<<{COLOR_RESET}\n")
        return 0
    else:
        print(f"\n{COLOR_RED}{COLOR_BOLD}>>> CẢNH BÁO: CÓ {failed_tests} BÀI KIỂM THỬ KHÔNG ĐẠT! <<<{COLOR_RESET}\n")
        return 1


if __name__ == "__main__":
    exit_code = run_qa_suite()
    sys.exit(exit_code)
