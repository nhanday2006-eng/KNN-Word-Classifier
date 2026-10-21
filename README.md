# 🧠 k-NN Word Pair Semantic Classifier (Desktop GUI)

> **Ứng dụng Desktop hoàn chỉnh sử dụng thuật toán k-Nearest Neighbors (k-NN) để phân loại quan hệ cặp từ tiếng Anh thành Đồng nghĩa (Synonyms), Trái nghĩa (Antonyms) và Không liên quan (Unrelated).**

Giao diện Desktop hiện đại được phát triển bằng thư viện **`customtkinter`** theo phong cách **GitHub Dark Mode**, kết hợp trích xuất 12 đặc trưng số cốt lõi từ ngữ nghĩa LSA và cấu trúc đồ thị WordNet.

---

## 🌟 Nâng Cấp Kỹ Thuật Đột Phá: Không Gian 12 Đặc Trưng Cô Đọng (12-D)

Để giải quyết triệt để vấn đề phân loại Tính từ (Adjectives) như `tall` - `short` (do tính từ trong WordNet không có cây phân cấp `hypernyms()`) và khắc phục hiện tượng **Curse of Dimensionality**, mô hình sử dụng đúng **12 đặc trưng số cốt lõi**:

1. `lemma_overlap`: Tỷ lệ synset dùng chung giữa 2 từ trong WordNet.
2. `similar_to_score`: Mức độ liên kết qua quan hệ `similar_tos()` và `also_sees()` của cụm tính từ (Adjective Bipolar Clusters).
3. `antonym_graph_score`: Điểm đối lập ngữ nghĩa trên đồ thị (bao gồm cả direct lemma antonyms và indirect cluster antonyms).
4. `shared_attribute_or_hypernym`: Jaccard similarity của `hypernyms()` (danh/động từ) KẾT HỢP với `attributes()` (tính từ — ví dụ `tall` và `short` cùng chia sẻ thuộc tính `stature.n.02`).
5. `max_path_sim`: Max `path_similarity` (có chuẩn hóa tương đương khi phân tích tính từ).
6. `max_wup_sim`: Max `wup_similarity` (chuẩn hóa tương đương cho cụm tính từ).
7. `same_lexname`: Tỷ lệ các synset có cùng lĩnh vực từ điển (`synset.lexname()`).
8. `same_pos`: Tỷ lệ có cùng từ loại (Part-of-Speech).
9. `gloss_cosine_sim`: Độ tương đồng Cosine TF-IDF giữa văn bản định nghĩa + ví dụ của 2 từ.
10. `svd_cosine_sim`: Độ tương đồng Cosine giữa 2 vector SVD 50 chiều.
11. `svd_euclidean_dist`: Khoảng cách Euclidean giữa 2 vector SVD 50 chiều.
12. `morph_opposition_flag`: Cờ phát hiện tiền tố/hậu tố đối lập hình thái (`un-`, `in-`, `dis-`, `-less` vs `-ful`).

---

## 📊 Kết Quả Huấn Luyện & Đánh Giá

Mô hình được huấn luyện và kiểm thử trên tập dữ liệu 3,000 cặp từ cân bằng (1,000 Synonyms, 1,000 Antonyms, 1,000 Unrelated) chia theo tỷ lệ Stratified 80/20:

| Chỉ số Đánh giá | Điểm số đạt được |
| :--- | :---: |
| **Độ chính xác tổng thể (Accuracy)** | **98.67%** |
| **Macro F1-Score** | **98.66%** |
| **Macro Precision** | **98.69%** |
| **Macro Recall** | **98.67%** |
| **QA Automation Suite (38 tests)** | **100.00% PASSED** |

### Biểu đồ Báo cáo tự động xuất trong thư mục `plots/`:
1. `plots/k_optimization.png`: Khảo sát đường cong Accuracy và F1-Score khi thay đổi $k$ từ 1 đến 21, làm nổi bật điểm tối ưu.
2. `plots/confusion_matrix.png`: Ma trận nhầm lẫn chuẩn (Confusion Matrix) với số lượng mẫu và phần trăm phân bổ rõ ràng.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```
knn_word_pair_classifier/
├── data/
│   └── dataset.csv             # 3,000 cặp từ trích xuất từ WordNet và kho từ vựng cốt lõi
├── models/
│   └── knn_pipeline.pkl        # Pipeline k-NN 12 chiều, Scaler và Vectorizer đã huấn luyện
├── plots/
│   ├── k_optimization.png      # Biểu đồ tối ưu hóa tham số k
│   └── confusion_matrix.png    # Ma trận nhầm lẫn trực quan
├── dataset_builder.py          # Script trích xuất tự động 3,000 cặp từ (kèm coordinate terms)
├── feature_extractor.py        # Module trích xuất 12 đặc trưng ngữ nghĩa & hình thái cô đọng
├── word_explorer.py            # Module nghiệp vụ Tab 2 & bộ lọc ngưỡng tự tin >= 70%
├── train_and_test.py           # CLI script huấn luyện k-NN và xuất biểu đồ báo cáo
├── qa_tester.py                # Bộ kịch bản kiểm thử tự động 5 nhóm bài test khắt khe
├── app_gui.py                  # Ứng dụng Desktop GUI bằng CustomTkinter
├── Start_Tool.bat              # Script khởi động 1-click trên Windows
├── start_tool.sh               # Script khởi động trên Linux/macOS
├── requirements.txt            # Danh sách thư viện Python cần thiết
├── .gitignore                  # Cấu hình bỏ qua tệp tạm
└── README.md                   # Tài liệu hướng dẫn sử dụng
```

---

## 🛠️ Hướng Dẫn Cài Đặt & Sử Dụng

### 1. Cài đặt các thư viện cần thiết

Mở Terminal / PowerShell và chạy lệnh:

```bash
pip install -r requirements.txt
```

---

### 2. Chạy Kiểm thử Tự động (Automated QA Test Suite)

Chạy script kiểm thử tự động 5 nhóm bài test (38 test cases):

```bash
python qa_tester.py
```

*Kịch bản sẽ tự động đánh giá:*
- **Nhóm 1:** Cặp từ Trái nghĩa kinh điển (`tall-short`, `hot-cold`, `happy-sad`,...).
- **Nhóm 2:** Cặp từ Đồng nghĩa kinh điển (`happy-joyful`, `fast-quick`, `buy-purchase`,...).
- **Nhóm 3:** Cặp từ Không liên quan & Coordinate Terms (`tall-apple`, `apple-banana`, `table-chair`, `dog-cat`, `car-bicycle`,...).
- **Nhóm 4:** Logic Tab 2 Word Explorer (`tall` có `short` ở cột trái nghĩa; `tallish`, `long`, `rangy` ở cột đồng nghĩa; lọc sạch nghĩa bóng).
- **Nhóm 5:** Edge Cases (Danh từ cụ thể `apple`, `table`, `pizza` có danh sách trái nghĩa rỗng; xử lý an toàn từ không tồn tại và chuỗi rỗng).

---

### 3. Huấn luyện mô hình (CLI)

Để huấn luyện lại mô hình và xuất lại biểu đồ:

```bash
python train_and_test.py
```

---

### 4. Mở phần mềm Desktop GUI

**Cách 1:** Nhấp đúp chuột vào file `Start_Tool.bat` (trên Windows).

**Cách 2:** Chạy lệnh Terminal:

```bash
python app_gui.py
```

---

## 💡 Hướng Dẫn Sử Dụng Giao Diện Desktop

### 1. Tab "Phân loại Cặp từ (Pair Classifier)":
- Nhập 2 từ vào ô **Word 1** và **Word 2** (hoặc bấm các nút thử cặp mẫu nhanh như `tall - short`, `tall - high`, `hot - cold`, `apple - car`).
- Nhấn **"PHÂN LOẠI (RUN k-NN)"** (hoặc nhấn phím `Enter`).
- Quan sát thẻ kết quả đổi màu (Xanh: Đồng nghĩa, Đỏ: Trái nghĩa, Xám: Không liên quan).
- Xem bảng **k Hàng xóm gần nhất** ở bên phải để thấy các cặp từ tương tự trong tập huấn luyện đã bỏ phiếu như thế nào.

### 2. Tab "Tra cứu 1 Từ (Word Explorer)" - Bộ lọc Ngưỡng Tự tin >= 70%:
- Nhập một từ tiếng Anh đơn lẻ (ví dụ: `tall`, `happy`, `fast`, `strong`, `cold`, `apple`).
- Nhấn **"QUÉT TỪ LIÊN QUAN"**.
- Ứng dụng ưu tiên quét các nghĩa chính của từ (`synsets[:3]`), lọc bỏ các nghĩa bóng hiếm gặp (như "tall tale" = `improbable`), và áp dụng **ngưỡng tự tin chặt chẽ $\ge 70\%$** có cơ sở ngữ nghĩa rõ ràng:
  - Các từ đồng nghĩa (`high`, `tallish`, `long`, `rangy`,...) nằm chuẩn xác ở **Cột Từ Đồng Nghĩa**.
  - Từ trái nghĩa (`short`) nằm chuẩn xác ở **Cột Từ Trái Nghĩa**.
- Nếu nhập danh từ cụ thể không có từ trái nghĩa (như `apple`, `table`, `pizza`), ứng dụng hiển thị thông báo lịch sự: *"Không có từ trái nghĩa phù hợp cho từ này (đặc biệt đối với danh từ chỉ thực thể/sự vật cụ thể)."*
- Nếu nhập từ không có trong WordNet hoặc chuỗi rỗng, hiển thị cảnh báo an toàn.

### 3. Tab "Mô hình & Benchmark (Model Benchmark)":
- Theo dõi 4 thẻ chỉ số đánh giá: **Accuracy, Macro F1, Precision, Recall**.
- Điều chỉnh thanh trượt **$k$** (từ 1 đến 21), đổi metric khoảng cách (`euclidean`, `manhattan`, `cosine`), nhấn **"HUẤN LUYỆN LẠI"** để cập nhật mô hình ngay lập tức ($< 0.03$s).
- Xem trực tiếp ảnh biểu đồ tối ưu hóa $k$ và ma trận nhầm lẫn ngay trong ứng dụng, hoặc nhấn **"Mở Thư mục Plots"** để lấy ảnh chèn vào báo cáo.
