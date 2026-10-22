"""
qa_tester.py
Kịch bản kiểm thử tự động toàn diện (Automated QA Test Suite)
Đóng vai Senior QA Automation Tester kiêm ML Engineer:
Kiểm tra tự động 5 nhóm bài test khắt khe trên mô hình k-NN và module word_explorer.
Bao gồm:
- Toàn bộ cặp từ Trái nghĩa & Đồng nghĩa đời thường (Hạng mục 2)
- Chuẩn hóa so sánh hơn / so sánh nhất & chính tả UK/US (Hạng mục 1)
- Tra cứu 1 từ ở Tab 2 với các từ đời thường (fun, boring, tall, hot, happy)
- Edge cases danh từ cụ thể và phòng thủ lỗi nhập liệu
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
    # NHÓM 1: Cặp từ Trái nghĩa kinh điển, So sánh hơn & Đời thường (Label 0)
    # =========================================================================
    print_header("NHÓM 1: Cặp từ Trái nghĩa kinh điển, So sánh hơn & Đời thường (Kỳ vọng: Antonym - Nhãn 0)")
    group1 = [
        # 1. Cặp từ kinh điển
        ("tall", "short"), ("hot", "cold"), ("big", "small"), ("fast", "slow"),
        ("happy", "sad"), ("rich", "poor"), ("strong", "weak"), ("legal", "illegal"),
        ("visible", "invisible"), ("useful", "useless"), ("good", "bad"), ("bright", "dark"),

        # 2. So sánh hơn & so sánh nhất (Hạng mục 1)
        ("better", "worse"), ("best", "worst"), ("bigger", "smaller"),
        ("older", "younger"), ("faster", "slower"), ("taller", "shorter"), ("higher", "lower"),

        # 3. Cặp từ đời thường (Hạng mục 2)
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
        ("win", "lose"), ("hire", "fire")
    ]

    seen_g1 = set()
    unique_g1 = []
    for pair in group1:
        s = tuple(sorted([pair[0].lower(), pair[1].lower()]))
        if s not in seen_g1:
            seen_g1.add(s)
            unique_g1.append(pair)

    for w1, w2 in unique_g1:
        total_tests += 1
        res = classify_word_pair(w1, w2, model=model, scaler=scaler, extractor=extractor)
        pred = res["pred_label"]
        conf = res["probabilities"][pred] * 100
        if pred == 0:
            passed_tests += 1
            print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} ({w1:11}, {w2:11}) => Antonym (0) | Conf: {conf:.1f}%")
        else:
            failed_tests += 1
            print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} ({w1:11}, {w2:11}) => Nhận {pred} ({res['pred_name']}) | Conf: {conf:.1f}%")

    # =========================================================================
    # NHÓM 2: Cặp từ Đồng nghĩa kinh điển, UK/US & Đời thường (Label 1)
    # =========================================================================
    print_header("NHÓM 2: Cặp từ Đồng nghĩa kinh điển, UK/US & Đời thường (Kỳ vọng: Synonym - Nhãn 1)")
    group2 = [
        # 1. Cặp từ kinh điển
        ("happy", "joyful"), ("fast", "quick"), ("big", "large"), ("small", "tiny"),
        ("smart", "intelligent"), ("buy", "purchase"), ("start", "begin"), ("rich", "wealthy"),

        # 2. Biến thể chính tả Anh - Anh vs Anh - Mỹ (Hạng mục 1)
        ("color", "colour"), ("center", "centre"), ("favorite", "favourite"),
        ("gray", "grey"), ("behavior", "behaviour"), ("analyze", "analyse"),
        ("traveler", "traveller"),

        # 3. Cặp từ đồng nghĩa đời thường (Hạng mục 2)
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
        ("hard", "difficult")
    ]

    seen_g2 = set()
    unique_g2 = []
    for pair in group2:
        s = tuple(sorted([pair[0].lower(), pair[1].lower()]))
        if s not in seen_g2:
            seen_g2.add(s)
            unique_g2.append(pair)

    for w1, w2 in unique_g2:
        total_tests += 1
        res = classify_word_pair(w1, w2, model=model, scaler=scaler, extractor=extractor)
        pred = res["pred_label"]
        conf = res["probabilities"][pred] * 100
        if pred == 1:
            passed_tests += 1
            print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} ({w1:11}, {w2:11}) => Synonym (1) | Conf: {conf:.1f}%")
        else:
            failed_tests += 1
            print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} ({w1:11}, {w2:11}) => Nhận {pred} ({res['pred_name']}) | Conf: {conf:.1f}%")

    # =========================================================================
    # NHÓM 3: Cặp từ Không liên quan & Coordinate Terms (Label 2)
    # =========================================================================
    print_header("NHÓM 3: Cặp từ Không liên quan & Coordinate Terms (Kỳ vọng: Unrelated - Nhãn 2)")
    group3 = [
        ("tall", "apple"), ("hot", "computer"), ("apple", "banana"),
        ("table", "chair"), ("dog", "cat"), ("car", "bicycle"), ("pizza", "music")
    ]

    for w1, w2 in group3:
        total_tests += 1
        res = classify_word_pair(w1, w2, model=model, scaler=scaler, extractor=extractor)
        pred = res["pred_label"]
        conf = res["probabilities"][pred] * 100
        if pred == 2:
            passed_tests += 1
            print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} ({w1:11}, {w2:11}) => Unrelated (2) | Conf: {conf:.1f}%")
        else:
            failed_tests += 1
            print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} ({w1:11}, {w2:11}) => Nhận {pred} ({res['pred_name']}) | Conf: {conf:.1f}%")

    # =========================================================================
    # NHÓM 4: Kiểm tra Logic Tab 2 (Word Explorer - Tra cứu 1 từ)
    # =========================================================================
    print_header("NHÓM 4: Kiểm tra logic Tab 2 (Word Explorer - Tra cứu 1 từ)")

    # Test 4.1: Tra từ 'tall'
    total_tests += 1
    res_tall = explore_single_word("tall", model=model, scaler=scaler, extractor=extractor)
    ant_words_tall = [w for w, _ in res_tall["antonyms"]]
    syn_words_tall = [w for w, _ in res_tall["synonyms"]]

    has_short = "short" in ant_words_tall
    no_tallish_in_ant = "tallish" not in ant_words_tall
    no_long_in_ant = "long" not in ant_words_tall
    no_magniloquent = "magniloquent" not in (ant_words_tall + syn_words_tall)

    if has_short and no_tallish_in_ant and no_long_in_ant and no_magniloquent:
        passed_tests += 1
        print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} Tra từ 'tall': Trái nghĩa chứa 'short'; Đồng nghĩa ({len(syn_words_tall)} từ); Đã lọc sạch nghĩa bóng")
    else:
        failed_tests += 1
        print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} Tra từ 'tall' lỗi logic: has_short={has_short}, no_tallish={no_tallish_in_ant}, no_long={no_long_in_ant}")

    # Test 4.2: Tra từ 'hot'
    total_tests += 1
    res_hot = explore_single_word("hot", model=model, scaler=scaler, extractor=extractor)
    ant_words_hot = [w for w, _ in res_hot["antonyms"]]
    syn_words_hot = [w for w, _ in res_hot["synonyms"]]

    has_cold = "cold" in ant_words_hot
    has_warm = "warm" in syn_words_hot
    if has_cold and has_warm:
        passed_tests += 1
        print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} Tra từ 'hot': Cột Đồng nghĩa có 'warm'; Cột Trái nghĩa có 'cold'")
    else:
        failed_tests += 1
        print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} Tra từ 'hot' lỗi: has_cold={has_cold}, has_warm={has_warm}")

    # Test 4.3: Tra từ 'happy'
    total_tests += 1
    res_happy = explore_single_word("happy", model=model, scaler=scaler, extractor=extractor)
    ant_words_happy = [w for w, _ in res_happy["antonyms"]]
    syn_words_happy = [w for w, _ in res_happy["synonyms"]]

    has_sad_or_unhappy = ("sad" in ant_words_happy) or ("unhappy" in ant_words_happy)
    has_glad_or_joyful = ("glad" in syn_words_happy) or ("joyful" in syn_words_happy)
    if has_sad_or_unhappy and has_glad_or_joyful:
        passed_tests += 1
        print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} Tra từ 'happy': Cột Đồng nghĩa có 'glad'/'joyful'; Cột Trái nghĩa có 'unhappy'/'sad'")
    else:
        failed_tests += 1
        print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} Tra từ 'happy' lỗi: has_ant={has_sad_or_unhappy}, has_syn={has_glad_or_joyful}")

    # Test 4.4: Tra từ đời thường 'fun' (Hạng mục 2)
    total_tests += 1
    res_fun = explore_single_word("fun", model=model, scaler=scaler, extractor=extractor)
    ant_words_fun = [w for w, _ in res_fun["antonyms"]]
    syn_words_fun = [w for w, _ in res_fun["synonyms"]]

    has_boring_or_dull = ("boring" in ant_words_fun) or ("dull" in ant_words_fun)
    has_enjoyable_or_amusing = ("enjoyable" in syn_words_fun) or ("amusing" in syn_words_fun) or ("entertaining" in syn_words_fun)
    if has_boring_or_dull and has_enjoyable_or_amusing:
        passed_tests += 1
        print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} Tra từ 'fun': Cột Trái nghĩa có 'boring'/'dull'; Cột Đồng nghĩa có 'enjoyable'/'amusing'")
    else:
        failed_tests += 1
        print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} Tra từ 'fun' lỗi: has_ant={has_boring_or_dull}, has_syn={has_enjoyable_or_amusing}")

    # Test 4.5: Tra từ đời thường 'boring' (Hạng mục 2)
    total_tests += 1
    res_boring = explore_single_word("boring", model=model, scaler=scaler, extractor=extractor)
    ant_words_boring = [w for w, _ in res_boring["antonyms"]]

    has_fun_or_interesting = ("fun" in ant_words_boring) or ("exciting" in ant_words_boring) or ("interesting" in ant_words_boring)
    if has_fun_or_interesting:
        passed_tests += 1
        print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} Tra từ 'boring': Cột Trái nghĩa có 'fun'/'exciting'/'interesting'")
    else:
        failed_tests += 1
        print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} Tra từ 'boring' lỗi: ant_words={ant_words_boring}")

    # =========================================================================
    # NHÓM 5: Edge Cases (Danh từ cụ thể / Lỗi nhập liệu / Rỗng)
    # =========================================================================
    print_header("NHÓM 5: Edge Cases (Danh từ cụ thể / Lỗi nhập liệu / Rỗng)")

    # 5.1 -> 5.4: Danh từ cụ thể
    for noun in ["apple", "pizza", "computer", "table"]:
        total_tests += 1
        res = explore_single_word(noun, model=model, scaler=scaler, extractor=extractor)
        if len(res["antonyms"]) == 0:
            passed_tests += 1
            print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} Tra danh từ '{noun}': Danh sách Trái nghĩa rỗng (len = 0)")
        else:
            failed_tests += 1
            print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} Tra danh từ '{noun}' nhưng xuất hiện Trái nghĩa: {res['antonyms']}")

    # 5.5: Tra từ không tồn tại
    total_tests += 1
    res_oov = explore_single_word("asdfghjkl", model=model, scaler=scaler, extractor=extractor)
    if res_oov["status"] == "not_found":
        passed_tests += 1
        print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} Tra từ không tồn tại 'asdfghjkl': Báo lỗi an toàn (status='not_found', không crash)")
    else:
        failed_tests += 1
        print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} Tra từ OOV không đúng status: {res_oov}")

    # 5.6: Chuỗi rỗng Tab 2
    total_tests += 1
    res_empty = explore_single_word("", model=model, scaler=scaler, extractor=extractor)
    if res_empty["status"] == "error":
        passed_tests += 1
        print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} Chuỗi rỗng ở Tab 2: Báo lỗi an toàn (status='error', không crash)")
    else:
        failed_tests += 1
        print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} Chuỗi rỗng Tab 2 không bắt lỗi: {res_empty}")

    # 5.7: Cặp từ rỗng Tab 1
    total_tests += 1
    res_pair_empty = classify_word_pair("", "", model=model, scaler=scaler, extractor=extractor)
    if res_pair_empty["status"] == "error":
        passed_tests += 1
        print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} Cặp từ rỗng ở Tab 1: Báo lỗi an toàn (status='error', không crash)")
    else:
        failed_tests += 1
        print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} Cặp từ rỗng Tab 1 không bắt lỗi: {res_pair_empty}")

    # 5.8: Cặp từ OOV Tab 1
    total_tests += 1
    res_pair_oov = classify_word_pair("tall", "asdfgh", model=model, scaler=scaler, extractor=extractor)
    if res_pair_oov["status"] == "not_found":
        passed_tests += 1
        print(f"  {COLOR_GREEN}[PASSED ✓]{COLOR_RESET} Cặp từ OOV ở Tab 1: Báo lỗi an toàn (status='not_found', không crash)")
    else:
        failed_tests += 1
        print(f"  {COLOR_RED}[FAILED ✗]{COLOR_RESET} Cặp từ OOV Tab 1 không bắt lỗi: {res_pair_oov}")

    # =========================================================================
    # TỔNG KẾT
    # =========================================================================
    duration = time.time() - t_start
    print_header("TỔNG KẾT BÁO CÁO KIỂM THỬ (QA AUTOMATION SUMMARY)")
    print(f"  + Tổng số bài kiểm thử (Total Tests) : {total_tests}")
    print(f"  + Số bài ĐẠT (Passed)                : {passed_tests}")
    print(f"  + Số bài KHÔNG ĐẠT (Failed)          : {failed_tests}")
    pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    print(f"  + Tỷ lệ thành công (Pass Rate)       : {pass_rate:.2f}%")
    print(f"  + Thời gian chạy kiểm thử            : {duration:.2f} giây")

    if failed_tests == 0:
        print(f"\n{COLOR_BOLD}{COLOR_GREEN}>>> HOÀN TOÀN XUẤT SẮC: 100% TẤT CẢ TEST CASES ĐỀU ĐẠT CHUẨN! <<<{COLOR_RESET}\n")
        return 0
    else:
        print(f"\n{COLOR_BOLD}{COLOR_RED}>>> CẢNH BÁO: CÓ {failed_tests} BÀI KIỂM THỬ THẤT BÀI! CẦN KIỂM TRA LẠI! <<<{COLOR_RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_qa_suite())
