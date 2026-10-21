"""
train_and_test.py
Script CLI tự động:
1. Nạp/Tạo dataset 3000 cặp từ cân bằng (1000 Synonyms, 1000 Antonyms, 1000 Unrelated).
2. Trích xuất đúng 12 đặc trưng số cốt lõi (12-dim feature vector) bằng WordPairFeatureExtractor.
3. Huấn luyện k-NN, quét k từ 1 đến 21 để tìm k tối ưu.
4. In bảng classification_report chi tiết ra console.
5. Xuất 2 biểu đồ chất lượng cao:
   - plots/k_optimization.png (Đồ thị tối ưu hóa tham số k)
   - plots/confusion_matrix.png (Ma trận nhầm lẫn chuẩn trực quan)
6. Đóng gói pipeline hoàn chỉnh lưu vào models/knn_pipeline.pkl.
7. Chạy Unit Test kiểm tra chặt chẽ:
   - ("tall", "short") -> Antonym (0)
   - ("tall", "high")  -> Synonym (1)
   - ("tall", "apple") -> Unrelated (2)
"""

import os
import sys
import time
import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Chặn mở cửa sổ GUI khi vẽ đồ thị
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

from dataset_builder import build_or_load_dataset, LABEL_NAMES
from feature_extractor import WordPairFeatureExtractor

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def plot_k_optimization(k_values, accuracies, f1_scores, best_k, best_score, save_path="plots/k_optimization.png"):
    """Vẽ biểu đồ tối ưu hóa tham số k theo phong cách GitHub Dark hiện đại."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    ax.plot(k_values, accuracies, marker="o", color="#58a6ff", linewidth=2.2, markersize=6, label="Accuracy (Độ chính xác)")
    ax.plot(k_values, f1_scores, marker="s", color="#3fb950", linewidth=2.2, markersize=6, linestyle="--", label="Macro F1-Score")

    # Đánh dấu điểm tối ưu
    best_acc = accuracies[k_values.index(best_k)]
    ax.scatter([best_k], [best_acc], color="#f85149", s=140, zorder=5, edgecolors="#ffffff", linewidth=2)
    ax.annotate(
        f"Điểm tối ưu k={best_k}\nAcc: {best_acc * 100:.2f}%\nF1: {best_score * 100:.2f}%",
        xy=(best_k, best_acc),
        xytext=(best_k + 1.2, best_acc - 0.035 if best_acc > 0.88 else best_acc + 0.015),
        arrowprops=dict(facecolor="#f85149", shrink=0.08, width=1.5, headwidth=7),
        fontsize=10,
        fontweight="bold",
        color="#ffffff",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#21262d", edgecolor="#f85149", alpha=0.9)
    )

    ax.set_title("TỐI ƯU HÓA THAM SỐ k TRONG THUẬT TOÁN k-NN (12 ĐẶC TRƯNG)\n(Hyperparameter Tuning on 12-Dimensional Feature Space)",
                 fontsize=12, fontweight="bold", color="#e6edf3", pad=15)
    ax.set_xlabel("Số lượng láng giềng k (Number of Neighbors)", fontsize=11, color="#c9d1d9", labelpad=10)
    ax.set_ylabel("Điểm số Đánh giá (Score)", fontsize=11, color="#c9d1d9", labelpad=10)
    ax.set_xticks(k_values)
    ax.set_ylim([min(min(accuracies), min(f1_scores)) - 0.05, 1.02])
    ax.grid(True, linestyle=":", alpha=0.3, color="#8b949e")

    legend = ax.legend(frameon=True, facecolor="#21262d", edgecolor="#30363d", fontsize=10)
    for text in legend.get_texts():
        text.set_color("#c9d1d9")

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[Plot] Đã xuất biểu đồ tối ưu k: {save_path}")


def plot_confusion_matrix(y_true, y_pred, target_names, save_path="plots/confusion_matrix.png"):
    """Vẽ ma trận nhầm lẫn (Confusion Matrix) với tỷ lệ phần trăm và số lượng."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
        linewidths=1.5,
        linecolor="#0d1117"
    )

    ax.set_title("MA TRẬN NHẦM LẪN (CONFUSION MATRIX) - k-NN 12-DIMENSIONAL",
                 fontsize=12, fontweight="bold", color="#e6edf3", pad=15)
    ax.set_xlabel("Nhãn Dự đoán (Predicted Label)", fontsize=10, color="#c9d1d9", labelpad=10)
    ax.set_ylabel("Nhãn Thực tế (Actual Ground Truth)", fontsize=10, color="#c9d1d9", labelpad=10)

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[Plot] Đã xuất ma trận nhầm lẫn: {save_path}")


def main():
    total_start = time.time()
    print("=" * 70)
    print("  HUẤN LUYỆN k-NN VỚI 12 ĐẶC TRƯNG NGỮ NGHĨA & HÌNH THÁI CÔ ĐỌNG")
    print("  Đề tài: Synonyms (1) vs Antonyms (0) vs Unrelated (2)")
    print("=" * 70)

    # 1. Tải / Xây dựng dataset
    df = build_or_load_dataset(filepath="data/dataset.csv", n_per_class=1000, random_seed=42)
    pairs = list(zip(df["word1"].astype(str), df["word2"].astype(str)))
    y = df["label"].values

    # 2. Huấn luyện bộ trích xuất 12 đặc trưng
    print("\n[Trích xuất đặc trưng] Đang huấn luyện LSA và trích xuất đúng 12 đặc trưng cốt lõi...")
    t_feat = time.time()

    all_words = set(df["word1"].astype(str)).union(set(df["word2"].astype(str)))
    extractor = WordPairFeatureExtractor(n_components=50, max_tfidf_features=6000)
    extractor.fit(all_words)

    X = extractor.transform(pairs)
    print(f"[Trích xuất đặc trưng] Hoàn thành trong {time.time() - t_feat:.2f}s! Kích thước ma trận: {X.shape}")

    # 3. Chia tập Train/Test (80% / 20% Stratified)
    train_indices, test_indices = train_test_split(
        np.arange(len(y)),
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    train_pairs = [pairs[i] for i in train_indices]
    test_pairs = [pairs[i] for i in test_indices]

    # Chuẩn hóa 12 đặc trưng
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Quét tìm k tối ưu (k từ 1 đến 21)
    print("\n[Huấn luyện k-NN] Đang quét k từ 1 đến 21 để tìm k tối ưu...")
    k_range = list(range(1, 22, 2))  # Số lẻ: 1, 3, 5, ..., 21
    accuracies = []
    f1_scores = []

    best_k = 1
    best_f1 = -1.0
    best_model = None

    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean")
        clf.fit(X_train_scaled, y_train)
        pred = clf.predict(X_test_scaled)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")

        accuracies.append(acc)
        f1_scores.append(f1)
        print(f"  + k = {k:2d} | Accuracy = {acc * 100:.2f}% | Macro F1 = {f1 * 100:.2f}%")

        if f1 > best_f1:
            best_f1 = f1
            best_k = k
            best_model = clf

    print(f"\n=> GIÁ TRỊ k TỐI ƯU NHẤT: k = {best_k} (Macro F1 = {best_f1 * 100:.2f}%)")

    # 5. Đánh giá chi tiết mô hình tối ưu
    y_pred = best_model.predict(X_test_scaled)
    target_names = [f"{LABEL_NAMES[i]} ({i})" for i in range(3)]
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)

    acc_final = accuracy_score(y_test, y_pred)
    prec_final = precision_score(y_test, y_pred, average="macro")
    rec_final = recall_score(y_test, y_pred, average="macro")
    f1_final = f1_score(y_test, y_pred, average="macro")

    print("\n" + "=" * 70)
    print("                     BẢNG BÁO CÁO ĐÁNH GIÁ CHI TIẾT")
    print("=" * 70)
    print(report)
    print(f"Tổng kết: Accuracy = {acc_final * 100:.2f}% | Precision = {prec_final * 100:.2f}% | Recall = {rec_final * 100:.2f}% | F1 = {f1_final * 100:.2f}%")
    print("=" * 70)

    # 6. Xuất 2 biểu đồ báo cáo
    os.makedirs("plots", exist_ok=True)
    plot_k_optimization(k_range, accuracies, f1_scores, best_k, best_f1, save_path="plots/k_optimization.png")
    plot_confusion_matrix(y_test, y_pred, ["Antonym", "Synonym", "Unrelated"], save_path="plots/confusion_matrix.png")

    # 7. Lưu pipeline mô hình
    os.makedirs("models", exist_ok=True)
    pipeline_data = {
        "model": best_model,
        "scaler": scaler,
        "extractor": extractor,
        "train_pairs": train_pairs,
        "train_labels": y_train,
        "train_features_scaled": X_train_scaled,
        "test_features_scaled": X_test_scaled,
        "test_labels": y_test,
        "best_k": best_k,
        "k_range": k_range,
        "accuracies": accuracies,
        "f1_scores": f1_scores,
        "metrics": {
            "accuracy": acc_final,
            "precision": prec_final,
            "recall": rec_final,
            "f1": f1_final,
            "report_str": report
        },
        "label_names": {
            0: "Antonym (Trái nghĩa)",
            1: "Synonym (Đồng nghĩa)",
            2: "Unrelated (Không liên quan)"
        }
    }

    model_path = "models/knn_pipeline.pkl"
    joblib.dump(pipeline_data, model_path, compress=3)
    print(f"\n[Model] Đã lưu pipeline mô hình k-NN 12 chiều vào: {model_path}")
    print(f"[Model] Kích thước file lưu: {os.path.getsize(model_path) / 1024:.1f} KB")

    # 8. UNIT TEST KIỂM TRA CHẶT CHẼ CÁC TRƯỜNG HỢP THEO YÊU CẦU
    print("\n" + "=" * 70)
    print("                  KIỂM TRA CHẶT CHẼ CÁC CẶP TỪ THEO YÊU CẦU")
    print("=" * 70)

    unit_tests = [
        ("tall", "short", 0, "Antonym"),
        ("tall", "high", 1, "Synonym"),
        ("tall", "tallish", 1, "Synonym"),
        ("tall", "apple", 2, "Unrelated"),
        ("hot", "cold", 0, "Antonym"),
        ("happy", "glad", 1, "Synonym"),
        ("apple", "car", 2, "Unrelated")
    ]

    all_passed = True
    for w1, w2, exp_lbl, exp_name in unit_tests:
        f_val = extractor.extract_pair_features(w1, w2)
        f_scaled = scaler.transform([f_val])
        pred = best_model.predict(f_scaled)[0]
        prob = best_model.predict_proba(f_scaled)[0]
        passed = (pred == exp_lbl)
        if not passed:
            all_passed = False
        status_str = "PASSED ✓" if passed else "FAILED ✗"
        print(f"  [{status_str}] ({w1:7}, {w2:7}) => Dự đoán: {LABEL_NAMES[pred]:10} (Kỳ vọng: {exp_name:10}) | Conf: {prob[pred]*100:.1f}%")

    if all_passed:
        print("\n>>> TẤT CẢ UNIT TESTS ĐỀU ĐẠT CHUẨN 100%! <<<")
    else:
        print("\n>>> CẢNH BÁO: CÓ UNIT TEST CHƯA ĐẠT! <<<")

    # Dọn dẹp __pycache__
    for root, dirs, files in os.walk("."):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)

    print(f"\n>>> HOÀN THÀNH XUẤT SẮC TRONG: {time.time() - total_start:.2f} GIÂY! <<<")


if __name__ == "__main__":
    main()
