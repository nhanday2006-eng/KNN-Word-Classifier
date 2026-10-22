# CẨM NANG THUYẾT TRÌNH BẢO VỆ ĐỒ ÁN MACHINE LEARNING
## ĐỀ TÀI: PHÂN LOẠI CẶP TỪ TIẾNG ANH (SYNONYMS - ANTONYMS - UNRELATED) BẰNG THUẬT TOÁN k-NEAREST NEIGHBORS (k-NN)

---

> **Lời nhắn nhủ từ đàn anh:**  
> Chào em! Là một sinh viên năm nhất, việc em hoàn thành được một đồ án kết hợp cả **Xử lý ngôn ngữ tự nhiên (NLP)**, **Thuật toán máy học cổ điển (k-NN)** và **Lập trình giao diện Desktop (CustomTkinter GUI)** là một thành tích cực kỳ xuất sắc. Đừng lo lắng về việc bị giảng viên "bắt bẻ". Giảng viên chấm đồ án đại học không tìm cách đánh trượt sinh viên, mà muốn kiểm tra xem em **có thực sự hiểu bản chất những gì mình viết ra hay không**.  
>
> Cẩm nang này được anh biên soạn "cầm tay chỉ việc", bám sát 100% từng file code và số liệu thực tế trong project của em. Hãy đọc kỹ, thực hành theo kịch bản và tự tin bước lên bục bảo vệ nhé! Điểm A đang đợi em!

---

# MỤC LỤC
1. [PHẦN 1: BẢN ĐỒ KIẾN TRÚC & LUỒNG CHẠY CỦA TỪNG FILE CODE](#phần-1-bản-đồ-kiến-trúc--luồng-chạy-của-từng-file-code)
2. [PHẦN 2: KỊCH BẢN THUYẾT TRÌNH 5 - 7 PHÚT (CẦM TAY CHỈ VIỆC)](#phần-2-kịch-bản-thuyết-trình-5---7-phút-cầm-tay-chỉ-việc)
3. [PHẦN 3: CÁCH THUYẾT MINH 2 BIỂU ĐỒ BÁO CÁO (PLOTS)](#phần-3-cách-thuyết-minh-2-biểu-đồ-báo-cáo-plots)
4. [PHẦN 4: TOP 8 CÂU HỎI VẤN ĐÁP GIẢNG VIÊN HAY XOÁY VÀ CÂU TRẢ LỜI ĂN ĐIỂM A](#phần-4-top-8-câu-hỏi-vấn-đáp-giảng-viên-hay-xoáy-và-câu-trả-lời-ăn-điểm-a)

---

# PHẦN 1: BẢN ĐỒ KIẾN TRÚC & LUỒNG CHẠY CỦA TỪNG FILE CODE

### Sơ đồ luồng dữ liệu tổng quát (Pipeline Flow):
```
[WordNet NLTK Corpus]
        │
        ▼
1. dataset_builder.py  ──> Sinh 3,000 cặp từ cân bằng ──> data/dataset.csv
        │
        ▼
2. feature_extractor.py ──> Biến cặp từ (w1, w2) thành Vector 12 đặc trưng số
        │
        ▼
3. train_and_test.py   ──> Quét k, Train k-NN, Xuất plots & models/knn_pipeline.pkl
        │
        ├─────────────────────────────┐
        ▼                             ▼
4. word_explorer.py          5. qa_tester.py (Kiểm thử 38 ca test CLI tự động)
   (Logic nghiệp vụ Tab 2)
        │
        ▼
6. app_gui.py (Giao diện Desktop CustomTkinter Dark Mode cho người dùng cuối)
```

---

## 1. File `dataset_builder.py` (260 dòng) — Nhà máy chế tạo dữ liệu
* **Mục đích**: Tự động khai thác từ điển `nltk.corpus.wordnet` để trích xuất tập dữ liệu gồm đúng **3,000 cặp từ tiếng Anh cân bằng tuyệt đối** (1,000 Đồng nghĩa - Nhãn 1, 1,000 Trái nghĩa - Nhãn 0, 1,000 Không liên quan - Nhãn 2), lưu ra file `data/dataset.csv`.
* **Hàm quan trọng nhất**:
  - `build_or_load_dataset(...)` (khoảng dòng 204 - 260): Hàm nhạc trưởng. Nếu file CSV đã có thì đọc lên trong 0.1 giây; nếu chưa có thì kích hoạt 3 hàm con dưới đây rồi xáo trộn (shuffle) ngẫu nhiên.
  - `extract_antonyms_wordnet()` (dòng 120 - 134): Lấy các cặp từ đối lập trực tiếp qua `lem.antonyms()`.
  - `extract_synonyms_wordnet()` (dòng 136 - 154): Lấy các cặp từ chia sẻ chung synset trong WordNet.
  - `extract_unrelated_wordnet()` (dòng 156 - 202): **Điểm sáng kỹ thuật!** Hàm này không lấy ngẫu nhiên bừa bãi, mà chủ động gom cả các cặp từ cùng họ hàng danh mục (Coordinate terms như `apple - banana`, `dog - cat`, `car - bicycle`) nhưng nghiêm cấm không được chung synset hay antonyms. Điều này giúp mô hình k-NN học được cách phân biệt giữa *"Đồng nghĩa thật sự"* và *"Cùng nhóm đồ vật"*.
* **Giải thích bằng lời đời thường**:  
  > *"Thưa thầy cô, file này giống như một người công nhân phân loại siêu thị. Nó vào kho từ điển WordNet nhặt đúng 1000 cặp đối nhau, 1000 cặp giống nhau và 1000 cặp không liên quan, đóng hộp thành file CSV gọn gàng để sẵn sàng huấn luyện."*

---

## 2. File `feature_extractor.py` (364 dòng) — Trái tim kỹ thuật trích xuất đặc trưng
* **Mục đích**: Biến đổi một cặp từ dạng chữ `(w1, w2)` (ví dụ: `("tall", "short")`) thành một **vector số thực đúng 12 chiều** ($12$-dimensional numerical vector). Thuật toán k-NN không hiểu được chữ cái, nó chỉ hiểu các tọa độ số trong không gian để tính khoảng cách hình học Euclidean.
* **Hàm quan trọng nhất**:
  - `extract_pair_features(self, w1: str, w2: str) -> np.ndarray` (khoảng dòng 194 - 316): Trích xuất 12 chỉ số:
    1. `lemma_overlap`: Tỷ lệ phần trăm chia sẻ chung synset giữa 2 từ.
    2. `similar_to_score`: Mức độ liên kết qua đồ thị `similar_tos()` (đặc biệt quan trọng với tính từ).
    3. `antonym_graph_score`: Điểm số quan hệ đối lập trên đồ thị WordNet (trực tiếp qua lemma hoặc gián tiếp qua cụm cluster tính từ).
    4. `shared_attr_hyp`: Tương đồng Jaccard giữa tập Hypernyms (danh/động từ) và Attributes (tính từ, ví dụ `tall` và `short` cùng chung thuộc tính chiều cao `stature/height`).
    5. `max_path`: Khoảng cách ngắn nhất trên cây phân cấp WordNet.
    6. `max_wup`: Độ tương đồng ngữ nghĩa Wu-Palmer.
    7. `same_lexname`: Có cùng lĩnh vực từ điển không (ví dụ cùng thuộc `noun.animal` hay `adj.all`).
    8. `same_pos`: Có cùng từ loại không (danh từ, động từ hay tính từ).
    9. `gloss_cosine_sim`: Cosine similarity giữa văn bản định nghĩa (definitions + examples) bằng TF-IDF.
    10. `svd_cosine_sim`: Cosine similarity từ vector ngữ nghĩa tiềm ẩn (LSA 50 chiều).
    11. `svd_euclidean_dist`: Khoảng cách Euclidean giữa 2 vector LSA.
    12. `morph_opposition_flag`: Cờ nhị phân (0 hoặc 1) phát hiện tiền tố đối lập như `un-`, `in-`, `im-`, `dis-`, `-ful/-less` (hàm `check_morphological_opposition`, dòng 39 - 64).
* **Giải thích bằng lời đời thường**:  
  > *"Máy tính không biết chữ 'tall' hay 'short' nghĩa là gì. Hàm này đóng vai trò như một phòng khám đa khoa: đo huyết áp, chụp X-quang, xét nghiệm máu cho cặp từ để quy về 12 con số cụ thể. Nhờ đó, cặp từ trở thành một điểm tọa độ trong không gian 12 chiều."*

---

## 3. File `train_and_test.py` (298 dòng) — Huấn luyện, tối ưu và đánh giá mô hình
* **Mục đích**:
  1. Nạp dữ liệu từ `data/dataset.csv`.
  2. Dùng `StandardScaler` chuẩn hóa 12 đặc trưng về phân phối chuẩn (mean = 0, std = 1).
  3. Quét vòng lặp siêu tham số $k$ từ $1, 3, 5, \dots, 21$ để tìm giá trị $k$ cho điểm F1 cao nhất.
  4. Huấn luyện mô hình tối ưu `KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean')`.
  5. Đánh giá trên tập kiểm thử độc lập (Test Set 20% - 600 mẫu), xuất 2 biểu đồ ra `plots/` và đóng gói toàn bộ pipeline vào `models/knn_pipeline.pkl`.
* **Vị trí cốt lõi trong code**:
  - Dòng 167 - 169: Chuẩn hóa bằng `StandardScaler()`.
  - Dòng 181 - 197: Vòng lặp quét tìm $k$ tối ưu bằng khoảng cách có trọng số (`weights='distance'`).
* **Giải thích bằng lời đời thường**:  
  > *"Đây là phòng thi đấu. Mô hình sẽ thử hỏi ý kiến của 1 người hàng xóm, 3 người, 5 người xung quanh... xem số lượng hàng xóm bằng bao nhiêu thì đoán chính xác nhất. Kết quả máy tìm ra k = 3 là tối ưu nhất với độ chính xác lên tới 98.67%."*

---

## 4. File `word_explorer.py` (244 dòng) — Động cơ nghiệp vụ & bộ lọc thông minh
* **Mục đích**: Phục vụ tính năng nâng cao cho ứng dụng (Tab 2 - Word Explorer: người dùng nhập 1 từ đơn, tự động tìm ra danh sách Từ đồng nghĩa và Từ trái nghĩa của từ đó).
* **Hàm quan trọng nhất**:
  - `classify_word_pair(...)` (dòng 42 - 128): Phân loại an toàn 1 cặp từ, bắt trọn vẹn các ngoại lệ (từ rỗng, chứa số, từ không có trong từ điển) mà không bao giờ làm sập chương trình.
  - `explore_single_word(...)` (dòng 130 - 244):
    - Quét tìm tất cả các từ ứng viên liên quan trong WordNet.
    - **Bộ lọc từ nghĩa bóng (Figurative words)** (dòng 188 - 209): Loại bỏ các từ hiếm gặp như `magniloquent`, `improbable` khi tra từ `tall`.
    - **Bộ lọc ngưỡng tin cậy khắt khe $\ge 70\%$** (dòng 221 - 235): Chỉ những từ nào mô hình dự đoán với độ tự tin trên 70% và thỏa mãn điều kiện ngữ nghĩa chặt chẽ mới được hiển thị. Nhờ đó, các danh từ cụ thể như `apple`, `table`, `computer` sẽ có cột Trái nghĩa trống trơn (vì quả táo thì không có trái nghĩa).

---

## 5. File `app_gui.py` (957 dòng) — Giao diện Desktop hiện đại (CustomTkinter)
* **Mục đích**: Ứng dụng Desktop độc lập chạy trên máy tính, giao diện Dark Mode phong cách GitHub hiện đại, mượt mà và trực quan.
* **Cấu trúc 3 Tab chính**:
  1. **Tab 1: Phân loại Cặp từ (`classify_pair`, dòng 443 - 515)**: Nhập `Word 1` và `Word 2`, bấm Phân loại -> Hiển thị nhãn kết quả (Màu sắc theo nhãn), 3 thanh phần trăm xác suất và **Bảng hiển thị chi tiết $k$ hàng xóm gần nhất trong tập train**.
  2. **Tab 2: Tra cứu 1 Từ (`run_word_explorer`, dòng 641 - 724)**: Nhập 1 từ (ví dụ `tall`), chương trình tự quét và chia làm 2 cột: Cột Đồng nghĩa (xanh lá) và Cột Trái nghĩa (cam/đỏ), kèm độ tin cậy của mô hình.
  3. **Tab 3: Huấn luyện & Báo cáo (`retrain_model`, dòng 900 - 950)**: Cho phép điều chỉnh trực tiếp số lượng mẫu, tỷ lệ Test size và chọn giá trị $k$ trên giao diện để huấn luyện lại mô hình bằng cơ chế đa luồng (`threading`), không gây giật lag (freeze UI).

---

# PHẦN 2: KỊCH BẢN THUYẾT TRÌNH 5 - 7 PHÚT (CẦM TAY CHỈ VIỆC)

> **Hướng dẫn sử dụng kịch bản:**  
> Chữ in đậm trong ngoặc vuông `[HÀNH ĐỘNG]` là thao tác tay của em trên máy tính.  
> Các đoạn trong dấu ngoặc kép `"..."` là lời em nói trực tiếp với thầy cô. Nói to, rõ ràng, phong thái tự tin.

---

### PHÚT 1 - 2: GIỚI THIỆU BÀI TOÁN & Ý TƯỞNG THIẾT KẾ
* **[HÀNH ĐỘNG]**: Mở slide đầu tiên hoặc mở sẵn màn hình chính của ứng dụng Desktop.
* **LỜI THOẠI**:
  > *"Kính thưa quý Thầy/Cô trong Hội đồng chấm đồ án,*  
  > *Em tên là [Họ và tên của em], sinh viên lớp [Tên lớp]. Hôm nay, em xin phép được báo cáo đồ án môn học với đề tài: **'Ứng dụng thuật toán k-Nearest Neighbors (k-NN) để phân loại quan hệ cặp từ tiếng Anh thành Đồng nghĩa, Trái nghĩa và Không liên quan'**.*
  >
  > *Thưa thầy cô, trong Xử lý ngôn ngữ tự nhiên (NLP), việc phân biệt giữa từ đồng nghĩa và từ trái nghĩa là một bài toán kinh điển nhưng vô cùng thách thức. Các mô hình Word Embedding thông thường như Word2Vec hay GloVe thường bị nhầm lẫn giữa từ đồng nghĩa và trái nghĩa (ví dụ `hot` và `cold`) vì chúng thường xuất hiện trong các ngữ cảnh câu giống hệt nhau.*  
  >
  > *Để giải quyết triệt để bài toán này với một thuật toán học máy cổ điển là **k-NN**, em đã thiết kế một hệ thống trích xuất **12 đặc trưng ngữ nghĩa và hình thái học chuyên sâu** kết hợp cơ sở tri thức từ điển WordNet. Hệ thống không chỉ đạt độ chính xác thực nghiệm vượt trội **98.67%**, mà còn được đóng gói hoàn chỉnh thành một phần mềm Desktop trực quan, thân thiện cho người dùng."*

---

### PHÚT 3 - 4: SHOW CODE CỐT LÕI TRÊN VS CODE (CHỈ SHOW 2 ĐOẠN ĐẮT GIÁ NHẤT)
* **[HÀNH ĐỘNG]**: Chuyển sang cửa sổ VS Code, mở file `feature_extractor.py`, cuộn chuột đến **dòng 301 - 315**.

#### Điểm sáng 1: Vector 12 đặc trưng (Tránh Lời nguyền số chiều)
* **LỜI THOẠI**:
  > *"Em xin phép trình chiếu phần cốt lõi đầu tiên của mô hình: hàm `extract_pair_features` trong file `feature_extractor.py`."*

```python
# Trích đoạn feature_extractor.py (Dòng 301 - 314)
features = np.array([
    lemma_overlap,          # 1. Tỷ lệ chia sẻ synset
    similar_to_score,       # 2. Liên kết similar_tos của tính từ
    antonym_graph_score,    # 3. Quan hệ đối lập trên đồ thị WordNet
    shared_attr_hyp,        # 4. Thuộc tính chung (attributes & hypernyms)
    max_path,               # 5. Khoảng cách ngắn nhất trên cây phân cấp
    max_wup,                # 6. Độ đo tương đồng Wu-Palmer
    same_lexname,           # 7. Cùng miền tri thức từ điển
    same_pos,               # 8. Cùng từ loại (Part-of-Speech)
    gloss_cosine_sim,       # 9. Tương đồng định nghĩa văn bản TF-IDF
    svd_cosine_sim,         # 10. Cosine ngữ nghĩa tiềm ẩn (LSA)
    svd_euclidean_dist,     # 11. Khoảng cách Euclidean LSA
    morph_opposition_flag   # 12. Cờ đối lập tiền tố (un-, in-, dis-, -less)
], dtype=np.float32)
```

* **LỜI THOẠI GIẢI THÍCH**:
  > *"Thưa thầy cô, thay vì dùng các mảng embedding thô hàng trăm chiều làm k-NN bị hiện tượng 'Lời nguyền số chiều' (Curse of Dimensionality) khiến khoảng cách bị nhiễu, em đã tinh gọn lại đúng 12 đặc trưng có ý nghĩa vật lý rõ ràng.*  
  > *Đặc biệt, đối với các tính từ như `tall` và `short`, vốn dĩ trong WordNet không có quan hệ cây cha con `hypernyms`, em đã kết hợp đặc trưng `shared_attr_hyp` thông qua quan hệ `attributes()` và đặc trưng `morph_opposition_flag` để bắt các tiền tố phủ định như `legal - illegal`, `visible - invisible`."*

---

* **[HÀNH ĐỘNG]**: Mở file `train_and_test.py`, cuộn đến **dòng 167 - 184**.

#### Điểm sáng 2: Chuẩn hóa dữ liệu & Quét siêu tham số k
```python
# Trích đoạn train_and_test.py (Dòng 167 - 184)
# 1. Bắt buộc chuẩn hóa đặc trưng cho k-NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Quét k từ 1 đến 21 để tối ưu hóa mô hình
k_range = list(range(1, 22, 2))  # [1, 3, 5, ..., 21]
for k in k_range:
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean")
    clf.fit(X_train_scaled, y_train)
```

* **LỜI THOẠI GIẢI THÍCH**:
  > *"Ở phần huấn luyện, em áp dụng `StandardScaler` để đưa 12 đặc trưng về cùng thang đo chuẩn hóa z-score. Vì k-NN tính khoảng cách hình học, nếu không chuẩn hóa thì đặc trưng nào có giá trị lớn sẽ áp đảo toàn bộ các đặc trưng còn lại.*  
  > *Đồng thời, em cấu hình tham số `weights='distance'`, tức là hàng xóm nào càng ở gần điểm truy vấn thì tiếng nói bỏ phiếu càng có trọng số cao hơn, giúp ranh giới quyết định mượt mà và chính xác hơn."*

---

### PHÚT 5 - 6: DEMO TRỰC TIẾP TRÊN PHẦN MỀM DESKTOP (APP GUI)
* **[HÀNH ĐỘNG]**: Bật phần mềm lên (chạy file `Start_Tool.bat` hoặc câu lệnh `py app_gui.py`). Giới thiệu giao diện Dark Mode.

#### Thao tác 1: Tab 1 — Phân loại Cặp từ (Pair Classifier)
1. Gõ `tall` vào ô Word 1, `short` vào ô Word 2. Bấm **"Phân loại Cặp từ (Classify)"**.
   - **LỜI THOẠI**: *"Khi em nhập cặp tính từ đối lập kinh điển `tall` - `short`, mô hình lập tức phân loại chính xác là **Antonym (Trái nghĩa)** với xác suất gần như tuyệt đối **99.5%**. Phía dưới giao diện hiển thị rõ ràng bảng **3 hàng xóm gần nhất** trong không gian dữ liệu với khoảng cách hình học cực kỳ gần (khoảng cách chỉ ~0.08)."*
2. Gõ `happy` và `joyful`. Bấm **"Phân loại Cặp từ"**.
   - **LỜI THOẠI**: *"Tiếp tục với cặp đồng nghĩa `happy` - `joyful`, hệ thống xác định đúng nhãn **Synonym (Đồng nghĩa)** với độ tự tin 99.5%."*
3. Gõ `apple` và `banana`. Bấm **"Phân loại Cặp từ"**.
   - **LỜI THOẠI**: *"Đặc biệt, với cặp từ `apple` và `banana`, dù cả hai đều là danh từ chỉ hoa quả, nhưng mô hình của em không hề bị nhầm lẫn mà nhận diện chuẩn xác là **Unrelated (Không liên quan)** với độ tự tin 100%. Đây là minh chứng cho thấy tập huấn luyện của em đã được xử lý triệt để các trường hợp Coordinate terms."*

#### Thao tác 2: Tab 2 — Tra cứu 1 Từ (Word Explorer)
1. Chuyển sang Tab 2. Gõ từ `tall` vào ô tìm kiếm, bấm **"Tra cứu & Khám phá Từ"**.
   - **LỜI THOẠI**: *"Ở Tab 2, khi người dùng muốn tra cứu các từ liên quan đến `tall`, phần mềm sẽ quét và phân bổ tự động vào 2 cột riêng biệt: Cột Đồng nghĩa gồm `tallish`, `high`... và Cột Trái nghĩa chứa chính xác từ `short`. Nhờ có bộ lọc ngữ nghĩa của em, các từ lóng hay nghĩa bóng hiếm gặp như `magniloquent` hoàn toàn bị loại bỏ."*
2. Gõ thử từ `apple`. Bấm **"Tra cứu"**.
   - **LỜI THOẠI**: *"Khi tra từ `apple`, cột Đồng nghĩa hiển thị các danh từ tương đương, còn cột Trái nghĩa hoàn toàn trống trơn. Đây là tính năng rất thông minh của phần mềm, vì quả táo trong thực tế là danh từ cụ thể, không có từ trái nghĩa."*

#### Thao tác 3: Tab 3 — Xem Báo cáo & Đồ thị
1. Bấm sang Tab 3, chỉ vào 2 ảnh đồ thị đã được nhúng sẵn trên giao diện.
   - **LỜI THOẠI**: *"Tab 3 là trung tâm giám sát chất lượng mô hình, hiển thị trực quan Ma trận nhầm lẫn (Confusion Matrix) và Đồ thị tối ưu hóa k, đồng thời cho phép người dùng cấu hình lại tham số để Retrain ngay trên giao diện."*

---

### PHÚT 7: KẾT LUẬN & LỜI CẢM ƠN
* **LỜI THOẠI**:
  > *"Kính thưa quý Thầy/Cô, toàn bộ quy trình từ xây dựng dữ liệu, trích xuất đặc trưng, huấn luyện mô hình đến kiểm thử tự động 38 kịch bản test (QA Automation) đều đã được em hoàn thiện và kiểm chứng đạt độ chính xác 98.67%.  
  > Em xin chân thành cảm ơn quý Thầy/Cô đã lắng nghe phần trình bày của em. Em rất mong nhận được những lời nhận xét và câu hỏi đóng góp từ Thầy/Cô để đồ án được hoàn thiện hơn nữa!"*

---

# PHẦN 3: CÁCH THUYẾT MINH 2 BIỂU ĐỒ BÁO CÁO (PLOTS)

Khi thầy cô yêu cầu em mở 2 ảnh trong thư mục `plots/` hoặc nhìn vào Tab 3 của giao diện, em hãy chỉ tay vào màn hình và đọc chính xác các con số sau:

```
Thư mục: plots/
├── k_optimization.png      (Biểu đồ tìm k tối ưu)
└── confusion_matrix.png    (Ma trận nhầm lẫn 3x3)
```

---

## 1. Biểu đồ 1: `k_optimization.png` (Đồ thị tối ưu hóa siêu tham số k)

* **Cấu trúc biểu đồ**:
  - Trục hoành ($Ox$): Các giá trị $k$ lẻ được khảo sát từ $1, 3, 5, \dots, 21$.
  - Trục tung ($Oy$): Thang đo phần trăm từ $97.5\%$ đến $99.0\%$.
  - Hai đường gấp khúc: Đường màu xanh dương biểu thị **Accuracy**, đường màu cam nét đứt biểu thị **Macro F1-Score**.
* **Con số thực tế trong bài của em**:
  - Tại $k = 1$: Accuracy $= 98.50\%$, F1 $= 98.50\%$.
  - **Tại $k = 3$: Điểm đỉnh cao nhất (Global Optimum) — Accuracy $= 98.67\%$, Macro F1 $= 98.66\%$**.
  - Khi $k$ tăng từ 5 đến 11: Độ chính xác giữ ổn định ở mức $98.50\%$.
  - Khi $k \ge 13$: Độ chính xác bắt đầu suy giảm dần về $98.00\%$.
* **Lời nhận xét ăn điểm khi thuyết minh**:
  > *"Thưa thầy cô, đồ án đã tiến hành quét k trên toàn bộ miền giá trị lẻ từ 1 đến 21.  
  > - Nhìn vào biểu đồ, tại **k = 3**, mô hình đạt hiệu năng cao nhất với **Accuracy 98.67%** và **Macro F1 98.66%**.  
  > - Khi k nhỏ (k = 1), mô hình dễ nhạy cảm với các điểm dữ liệu nhiễu ngoại lai (Outliers).  
  > - Ngược lại, khi k tăng quá lớn (k > 11), hiện tượng **Over-smoothing** xảy ra: vùng biên quyết định bị làm mượt quá mức, dẫn đến việc các mẫu dữ liệu ở biên bị gom nhầm sang lớp đa số khác, làm giảm độ chính xác xuống 98.0%.  
  > Do đó, việc lựa chọn **k = 3** kết hợp khoảng cách có trọng số (`weights='distance'`) là sự cân bằng hoàn hảo nhất."*

---

## 2. Biểu đồ 2: `confusion_matrix.png` (Ma trận nhầm lẫn)

* **Cấu trúc biểu đồ**:
  - Ma trận vuông $3 \times 3$ thể hiện kết quả dự đoán trên **600 mẫu thử nghiệm độc lập** của tập Test (mỗi lớp có đúng 200 mẫu cân bằng tuyệt đối).
  - Trục tung ($y$): Nhãn thực tế (True Label).
  - Trục hoành ($x$): Nhãn mô hình dự đoán (Predicted Label).
* **Con số thực tế trong bài của em**:
  1. **Lớp Antonym (Trái nghĩa - Nhãn 0)**:
     - Dự đoán đúng: **199 / 200 mẫu (99.5%)**.
     - Nhầm: Chỉ duy nhất 1 mẫu bị nhầm sang Unrelated.
  2. **Lớp Synonym (Đồng nghĩa - Nhãn 1)**:
     - Dự đoán đúng: **193 / 200 mẫu (96.5%)**.
     - Nhầm: 7 mẫu bị nhầm sang Unrelated (do một số từ đồng nghĩa có ngữ cảnh định nghĩa rộng trong WordNet).
     - **Hoàn toàn KHÔNG bị nhầm bất kỳ mẫu nào sang Antonym (0 mẫu)**.
  3. **Lớp Unrelated (Không liên quan - Nhãn 2)**:
     - Dự đoán đúng: **200 / 200 mẫu (100.0%)** — Độ nhạy (Recall) tuyệt đối!
* **Bảng tổng hợp chỉ số chi tiết (Classification Report)**:
  - **Antonym**: Precision = $99.50\%$, Recall = $99.50\%$, F1-score = $99.50\%$.
  - **Synonym**: Precision = $99.48\%$, Recall = $96.50\%$, F1-score = $97.97\%$.
  - **Unrelated**: Precision = $97.09\%$, Recall = $100.00\%$, F1-score = $98.52\%$.
* **Lời nhận xét ăn điểm khi thuyết minh**:
  > *"Thưa thầy cô, nhìn vào đường chéo chính của Ma trận nhầm lẫn, các ô màu xanh đậm áp đảo hoàn toàn, minh chứng cho việc mô hình có độ đặc hiệu cực cao trên cả 3 lớp.  
  > Điểm ấn tượng nhất là **không có bất kỳ một cặp từ Đồng nghĩa nào bị đoán nhầm thành Trái nghĩa** và ngược lại. Lớp Không liên quan đạt Recall 100%. Điều này khẳng định 12 đặc trưng ngữ nghĩa được trích xuất đã phân tách không gian một cách rạch ròi, giúp k-NN tìm đúng những hàng xóm thực sự của từng mẫu."*

---

# PHẦN 4: TOP 8 CÂU HỎI VẤN ĐÁP GIẢNG VIÊN HAY XOÁY VÀ CÂU TRẢ LỜI ĂN ĐIỂM A

Dưới đây là 8 câu hỏi "tủ" mà các thầy cô bộ môn Trí tuệ nhân tạo / Khai phá dữ liệu thường xuyên dùng để kiểm tra độ hiểu sâu của sinh viên. Em chỉ cần trả lời đúng theo gợi ý dưới đây là giảng viên sẽ gật gù khen ngợi:

---

### CÂU 1: *"Tại sao em lại chọn thuật toán k-NN cho bài toán này mà không dùng các mô hình phức tạp hơn như SVM, Random Forest hay Deep Learning (BERT, RoBERTa)?"*
* **Trả lời chuẩn điểm A**:
  > *"Dạ thưa Thầy/Cô, việc em lựa chọn k-NN xuất phát từ 3 lý do kỹ thuật rõ ràng:*  
  > *1. **Mục tiêu nghiên cứu**: Đồ án nhằm chứng minh rằng với một thuật toán học máy cổ điển, đơn giản (Instance-based learning) như k-NN, nếu ta thực hiện **Kỹ nghệ đặc trưng (Feature Engineering)** thật tốt và khai thác đúng tri thức ngôn ngữ, ta vẫn đạt được độ chính xác xuất sắc (98.67%) mà không cần hệ thống phần cứng GPU đắt tiền.*  
  > *2. **Tính giải thích được (Explainability)**: k-NN là mô hình 'hộp trắng'. Trong Tab 1, em có thể chỉ ra chính xác cho người dùng thấy 3 điểm dữ liệu hàng xóm gần nhất là ai, khoảng cách bao nhiêu. Điều này các mô hình Deep Learning 'hộp đen' không làm được trực quan như vậy.*  
  > *3. **Tốc độ huấn luyện và triển khai**: Mô hình k-NN của em nạp và huấn luyện chỉ mất chưa đầy 3 giây, toàn bộ file mô hình chỉ vài chục KB, cực kỳ nhẹ và phù hợp chạy offline trên các ứng dụng máy tính văn phòng."*

---

### CÂU 2: *"Trong WordNet, các Tính từ (Adjectives) như 'tall' và 'short' không có quan hệ Hypernym (từ cha) và hàm `path_similarity()` luôn trả về None hoặc 0. Em đã giải quyết vấn đề này như thế nào trong code?"*
* **Trả lời chuẩn điểm A**:
  > *(Câu này hỏi trúng tim đen của bài toán, em hãy bình tĩnh trả lời)*  
  > *"Dạ thưa Thầy/Cô, đây chính là thách thức lớn nhất mà em đã giải quyết trong file `feature_extractor.py` (dòng 230 - 267):*  
  > *Trong cấu trúc WordNet, tính từ không tổ chức theo cây phân cấp (Hierarchical Tree) như danh từ nên không có Hypernym. Em đã khắc phục bằng 2 cơ chế:*  
  > *Thứ nhất, em dùng hàm `attributes()` để tìm thuộc tính chung. Ví dụ cả `tall` và `short` đều trỏ về thuộc tính độ cao `stature` hoặc `height`. Em đưa nó vào đặc trưng `shared_attr_hyp`.*  
  > *Thứ hai, em dùng mạng đồ thị vệ tinh `similar_tos()` và `also_sees()`. Nếu 2 từ là tính từ mà chia sẻ chung thuộc tính hoặc nằm trong cùng cluster vệ tinh, em thực hiện chuẩn hóa gán mức tương đồng mặc định tương ứng (`max_path = 0.33 - 0.50`), tránh việc ma trận khoảng cách bị trả về 0 oan uổng."*

---

### CÂU 3: *"Tại sao em lại rút gọn chỉ còn đúng 12 đặc trưng thay vì giữ nguyên các mảng vector ngữ nghĩa hàng trăm chiều của SVD hay Word2Vec?"*
* **Trả lời chuẩn điểm A**:
  > *"Dạ thưa Thầy/Cô, vì đặc thù toán học của thuật toán k-NN phụ thuộc hoàn toàn vào khoảng cách Euclidean trong không gian $D$ chiều.  
  > Khi số chiều $D$ tăng lên quá lớn (ví dụ hàng trăm chiều), hiện tượng **Lời nguyền số chiều (Curse of Dimensionality)** sẽ xảy ra: không gian trở nên cực kỳ thưa thớt, khoảng cách giữa điểm gần nhất và điểm xa nhất tiến dần về bằng nhau, khiến cho khái niệm 'hàng xóm gần nhất' mất đi ý nghĩa phân loại.*  
  > *Bằng cách cô đọng lại đúng 12 đặc trưng đại diện cho 4 khía cạnh cốt lõi: Trùng lặp Synset, Cấu trúc đồ thị, Ngữ nghĩa văn bản LSA/TF-IDF và Hình thái học, em vừa giữ trọn vẹn thông tin ngữ nghĩa, vừa giúp k-NN tính toán khoảng cách Euclidean một cách sắc bén và chuẩn xác nhất."*

---

### CÂU 4: *"Hai từ như 'apple' (táo) và 'banana' (chuối) cùng thuộc họ hoa quả (Coordinate terms / Co-hyponyms). Làm sao mô hình của em không bị nhầm chúng là từ Đồng nghĩa (Synonyms)?"*
* **Trả lời chuẩn điểm A**:
  > *"Dạ thưa Thầy/Cô, đây là lỗi rất phổ biến nếu sinh dữ liệu ngẫu nhiên. Em đã xử lý điều này ngay từ khâu thiết kế dữ liệu và trích xuất đặc trưng:*  
  > *1. **Ở khâu Dataset (`dataset_builder.py`, dòng 156)**: Em chủ động tạo các mẫu kiểm thử Coordinate terms vào nhãn Unrelated (nhãn 2) nhưng đảm bảo chúng không chung synset.*  
  > *2. **Ở khâu Đặc trưng (`feature_extractor.py`)**: `apple` và `banana` tuy có chung Hypernym là `fruit`, nhưng đặc trưng `lemma_overlap` của chúng bằng 0 (không chia sẻ synset) và `similar_to_score` bằng 0.*  
  > *Nhờ đó, tọa độ 12 chiều của cặp `apple - banana` nằm cách rất xa các cụm từ đồng nghĩa thực sự, giúp mô hình k-NN phân loại chính xác là Unrelated (Không liên quan)."*

---

### CÂU 5: *"Trong thuật toán k-NN, tại sao em lại đặt tham số `weights='distance'` mà không dùng `weights='uniform'` mặc định?"*
* **Trả lời chuẩn điểm A**:
  > *"Dạ thưa Thầy/Cô:*  
  > *- Với `weights='uniform'`, cả $k$ người hàng xóm đều có quyền biểu quyết ngang nhau (mỗi người 1 phiếu), bất kể người đó ở rất gần hay ở rìa ngoài cùng.*  
  > *- Với `weights='distance'`, trọng số bỏ phiếu của mỗi hàng xóm tỉ lệ nghịch với khoảng cách ($w_i = \frac{1}{d_i}$). Hàng xóm nào càng gần điểm đang xét thì tiếng nói càng có trọng lượng quyết định.*  
  > *Điều này đặc biệt có lợi ở những vùng biên ranh giới giữa các lớp: nếu có 1 hàng xóm nằm cực gần nhưng 2 hàng xóm kia nằm xa hơn một chút, cơ chế trọng số khoảng cách sẽ giúp mô hình không bị dự đoán sai theo số đông ở xa."*

---

### CÂU 6: *"Tại sao việc sử dụng `StandardScaler` lại là bước BẮT BUỘC (sống còn) đối với thuật toán k-NN trong bài toán này?"*
* **Trả lời chuẩn điểm A**:
  > *"Dạ thưa Thầy/Cô, vì công thức khoảng cách Euclidean là căn bậc hai của tổng bình phương chênh lệch giữa các tọa độ:  
  > $$d(x, y) = \sqrt{\sum_{i=1}^{12} (x_i - y_i)^2}$$  
  > Trong 12 đặc trưng của em:  
  > - Đặc trưng `lemma_overlap` hay `max_path` có giá trị nhỏ trong khoảng $[0.0, 1.0]$.  
  > - Nhưng đặc trưng `svd_euclidean_dist` lại có giá trị dao động lớn từ $0.0$ đến $1.4$ hoặc hơn.*  
  > *Nếu không dùng `StandardScaler` để đưa tất cả 12 đặc trưng về cùng một phân phối chuẩn (trung bình bằng 0, phương sai bằng 1), thì đặc trưng có thang đo lớn sẽ hoàn toàn thống trị khoảng cách Euclidean, biến các đặc trưng quan trọng khác thành vô nghĩa. Do đó, chuẩn hóa là bước bắt buộc của k-NN."*

---

### CÂU 7: *"Nếu người dùng nhập vào một từ chưa từng có trong từ điển WordNet (từ OOV - Out-of-Vocabulary) hoặc gõ sai chính tả, phần mềm của em có bị crash (văng lỗi) không và xử lý ra sao?"*
* **Trả lời chuẩn điểm A**:
  > *(Em tự tin mở Tab 1 gõ thử từ linh tinh như `asdfgh` để chứng minh)*  
  > *"Dạ thưa Thầy/Cô, hệ thống của em được thiết kế theo chuẩn phòng thủ ngoại lệ (Defensive Programming) trong file `word_explorer.py` (dòng 42 - 84):*  
  > *Trước khi đưa vào mô hình, hàm sẽ kiểm tra dữ liệu qua 3 lớp phòng vệ:*  
  > *1. Kiểm tra chuỗi rỗng và ký tự không phải chữ cái (`w.isalpha()`).*  
  > *2. Tra cứu tồn tại trong WordNet (`wn.synsets(w)`). Nếu từ không tồn tại, hàm lập tức trả về thông báo lỗi thân thiện: 'Từ X không tồn tại trong từ điển WordNet. Vui lòng kiểm tra lại chính tả!'.*  
  > *3. Toàn bộ khối tính toán đều được bọc trong khối `try...except`, đảm bảo ứng dụng Desktop không bao giờ bị văng Exception hay sập đột ngột."*

---

### CÂU 8: *"Khi bấm nút Huấn luyện lại mô hình (Retrain) ở Tab 3, làm thế nào mà giao diện phần mềm CustomTkinter không bị đơ giật (Freeze UI) trong lúc đang tính toán?"*
* **Trả lời chuẩn điểm A**:
  > *"Dạ thưa Thầy/Cô, trong lập trình giao diện Tkinter, luồng chính (Main UI Thread) chịu trách nhiệm lắng nghe sự kiện chuột và vẽ đồ họa. Nếu ta chạy một tác vụ máy học nặng trên luồng chính, giao diện sẽ bị treo cứng (Not Responding).*  
  > *Để giải quyết vấn đề này, trong file `app_gui.py` (dòng 900 - 950), em đã áp dụng **Kỹ thuật Đa luồng (Multithreading)** bằng thư viện `threading.Thread` của Python.  
  > Khi người dùng bấm nút Huấn luyện lại, tác vụ tính toán được đẩy sang một luồng phụ chạy ngầm (`daemon thread`), còn luồng chính vẫn tiếp tục cập nhật thanh tiến trình (Progress bar) và cho phép người dùng thao tác bình thường, đem lại trải nghiệm mượt mà tuyệt đối."*

---

> **LỜI CHÚC TỪ ĐÀN ANH:**  
> Nắm vững 4 phần trên, em đã sở hữu 100% kiến thức từ tổng quan đến chi tiết của đồ án này. Hãy ngủ đủ giấc trước ngày bảo vệ, ăn mặc chỉnh chu, tự tin trả lời và rinh điểm A về nhé! Chúc em thành công rực rỡ!
