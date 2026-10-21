"""
app_gui.py
Ứng dụng Desktop GUI hoàn chỉnh bằng CustomTkinter (Phong cách Dark Mode GitHub):
Phân loại cặp từ tiếng Anh thành Đồng nghĩa (Synonyms), Trái nghĩa (Antonyms), Không liên quan (Unrelated).
Sử dụng mô hình k-Nearest Neighbors (k-NN) trên không gian 12 đặc trưng cô đọng, chất lượng cao.

Kích thước cửa sổ: 1000x680.
Bao gồm 3 tab chức năng:
- Tab 1: Phân loại Cặp từ (Pair Classifier) + Bảng k Hàng xóm gần nhất
- Tab 2: Tra cứu 1 Từ (Word Explorer) + Tự động phân loại 2 cột Đồng nghĩa / Trái nghĩa với bộ lọc >= 70%
- Tab 3: Huấn luyện & Benchmark Mô hình + Tùy chỉnh k/Metric siêu tốc + Trình xem đồ thị
"""

import os
import sys
import subprocess
import joblib
import numpy as np
from PIL import Image

import customtkinter as ctk
from nltk.corpus import wordnet as wn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset_builder import build_or_load_dataset
from feature_extractor import WordPairFeatureExtractor
from word_explorer import explore_single_word, classify_word_pair

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Cài đặt giao diện Dark Mode hiện đại phong cách GitHub
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Bảng mã màu chuyên nghiệp
COLOR_BG_DARK = "#0d1117"
COLOR_SIDEBAR = "#161b22"
COLOR_CARD = "#21262d"
COLOR_CARD_BORDER = "#30363d"
COLOR_ACCENT_BLUE = "#1f6feb"
COLOR_ACCENT_HOVER = "#388bfd"
COLOR_TEXT_MAIN = "#f0f6fc"
COLOR_TEXT_MUTED = "#8b949e"

COLOR_SYNONYM = "#238636"
COLOR_SYNONYM_LIGHT = "#3fb950"
COLOR_ANTONYM = "#da3633"
COLOR_ANTONYM_LIGHT = "#f85149"
COLOR_UNRELATED = "#30363d"
COLOR_UNRELATED_LIGHT = "#8b949e"


class KNNWordPairApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("k-NN Word Pair Classifier | 12-Dimensional Semantic Desktop AI")
        self.geometry("1000x680")
        self.minsize(980, 650)
        self.configure(fg_color=COLOR_BG_DARK)

        # Nạp Pipeline mô hình 12 chiều
        self.pipeline_data = self.load_or_init_pipeline()
        self.model = self.pipeline_data["model"]
        self.scaler = self.pipeline_data["scaler"]
        self.extractor = self.pipeline_data["extractor"]
        self.train_pairs = self.pipeline_data["train_pairs"]
        self.train_labels = self.pipeline_data["train_labels"]
        self.train_features_scaled = self.pipeline_data["train_features_scaled"]
        self.test_features_scaled = self.pipeline_data["test_features_scaled"]
        self.test_labels = self.pipeline_data["test_labels"]
        self.current_k = self.pipeline_data["best_k"]

        # Thiết lập bố cục chính (Sidebar trái + Content phải)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.setup_sidebar()
        self.setup_content_tabs()

        # Hiển thị mặc định Tab 1
        self.select_tab("tab1")

    def load_or_init_pipeline(self):
        """Nạp pipeline từ models/knn_pipeline.pkl hoặc tự tạo nếu chưa có."""
        model_path = os.path.join(os.path.dirname(__file__), "models", "knn_pipeline.pkl")
        if os.path.exists(model_path):
            try:
                data = joblib.load(model_path)
                if data["train_features_scaled"].shape[1] == 12:
                    return data
            except Exception as e:
                print(f"[Warning] Cần cập nhật mô hình mới ({e})...")

        # Huấn luyện lại mô hình nếu chưa có
        from train_and_test import main as run_train
        run_train()
        return joblib.load(model_path)

    # =========================================================================
    # SIDEBAR GIAO DIỆN
    # =========================================================================
    def setup_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(
            self, width=220, corner_radius=0, fg_color=COLOR_SIDEBAR, border_width=1, border_color=COLOR_CARD_BORDER
        )
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        # Logo & Tiêu đề ứng dụng
        logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="🧠 WordPair AI",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLOR_TEXT_MAIN
        )
        logo_label.grid(row=0, column=0, padx=20, pady=(24, 2), sticky="w")

        sub_logo = ctk.CTkLabel(
            self.sidebar_frame,
            text="k-NN Semantic Classifier (12D)",
            font=ctk.CTkFont(size=11),
            text_color=COLOR_TEXT_MUTED
        )
        sub_logo.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")

        # Nút điều hướng Tab 1
        self.btn_nav_tab1 = ctk.CTkButton(
            self.sidebar_frame,
            text="🔍  Phân loại Cặp từ",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="transparent",
            text_color=COLOR_TEXT_MAIN,
            hover_color=COLOR_CARD,
            anchor="w",
            height=38,
            corner_radius=6,
            command=lambda: self.select_tab("tab1")
        )
        self.btn_nav_tab1.grid(row=2, column=0, padx=12, pady=4, sticky="ew")

        # Nút điều hướng Tab 2
        self.btn_nav_tab2 = ctk.CTkButton(
            self.sidebar_frame,
            text="📖  Tra cứu 1 Từ",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="transparent",
            text_color=COLOR_TEXT_MAIN,
            hover_color=COLOR_CARD,
            anchor="w",
            height=38,
            corner_radius=6,
            command=lambda: self.select_tab("tab2")
        )
        self.btn_nav_tab2.grid(row=3, column=0, padx=12, pady=4, sticky="ew")

        # Nút điều hướng Tab 3
        self.btn_nav_tab3 = ctk.CTkButton(
            self.sidebar_frame,
            text="⚙️  Mô hình & Benchmark",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="transparent",
            text_color=COLOR_TEXT_MAIN,
            hover_color=COLOR_CARD,
            anchor="w",
            height=38,
            corner_radius=6,
            command=lambda: self.select_tab("tab3")
        )
        self.btn_nav_tab3.grid(row=4, column=0, padx=12, pady=4, sticky="ew")

        # Khung trạng thái mô hình góc dưới Sidebar
        self.status_card = ctk.CTkFrame(self.sidebar_frame, fg_color=COLOR_CARD, corner_radius=8, border_width=1, border_color=COLOR_CARD_BORDER)
        self.status_card.grid(row=7, column=0, padx=14, pady=16, sticky="ew")

        status_dot = ctk.CTkLabel(
            self.status_card,
            text="● Model Active (12D)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLOR_SYNONYM_LIGHT
        )
        status_dot.pack(anchor="w", padx=10, pady=(8, 2))

        self.lbl_sidebar_k = ctk.CTkLabel(
            self.status_card,
            text=f"Thuật toán: k-NN (k = {self.current_k})\nĐộ chính xác: {self.pipeline_data['metrics']['accuracy'] * 100:.1f}%\nTập huấn luyện: {len(self.train_pairs)} cặp",
            font=ctk.CTkFont(size=10),
            text_color=COLOR_TEXT_MUTED,
            justify="left"
        )
        self.lbl_sidebar_k.pack(anchor="w", padx=10, pady=(0, 8))

    def select_tab(self, tab_id):
        """Chuyển đổi giao diện giữa 3 tab chức năng."""
        self.btn_nav_tab1.configure(fg_color=COLOR_CARD if tab_id == "tab1" else "transparent")
        self.btn_nav_tab2.configure(fg_color=COLOR_CARD if tab_id == "tab2" else "transparent")
        self.btn_nav_tab3.configure(fg_color=COLOR_CARD if tab_id == "tab3" else "transparent")

        self.tab1_frame.grid_forget()
        self.tab2_frame.grid_forget()
        self.tab3_frame.grid_forget()

        if tab_id == "tab1":
            self.tab1_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        elif tab_id == "tab2":
            self.tab2_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        elif tab_id == "tab3":
            self.tab3_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
            self.refresh_plot_image()

    # =========================================================================
    # CÁC TAB NỘI DUNG CHÍNH
    # =========================================================================
    def setup_content_tabs(self):
        # 1. TAB 1: PHÂN LOẠI CẶP TỪ
        self.tab1_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.build_tab1()

        # 2. TAB 2: TRA CỨU 1 TỪ
        self.tab2_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.build_tab2()

        # 3. TAB 3: MÔ HÌNH & BENCHMARK
        self.tab3_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.build_tab3()

    # =========================================================================
    # TAB 1: PHÂN LOẠI CẶP TỪ (PAIR CLASSIFIER)
    # =========================================================================
    def build_tab1(self):
        self.tab1_frame.grid_rowconfigure(3, weight=1)
        self.tab1_frame.grid_columnconfigure(0, weight=1)

        # Tiêu đề Tab 1
        header_frame = ctk.CTkFrame(self.tab1_frame, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        title = ctk.CTkLabel(
            header_frame,
            text="🔍 Phân loại Cặp từ (Pair Classifier)",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLOR_TEXT_MAIN
        )
        title.pack(anchor="w")

        desc = ctk.CTkLabel(
            header_frame,
            text="Dự đoán Đồng nghĩa, Trái nghĩa hoặc Không liên quan bằng không gian 12 đặc trưng ngữ nghĩa & hình thái.",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_TEXT_MUTED
        )
        desc.pack(anchor="w", pady=(2, 0))

        # Khung nhập liệu (Word 1, Swap, Word 2, Run Button)
        input_card = ctk.CTkFrame(self.tab1_frame, fg_color=COLOR_SIDEBAR, corner_radius=10, border_width=1, border_color=COLOR_CARD_BORDER)
        input_card.grid(row=1, column=0, sticky="ew", pady=(0, 10), padx=2)

        input_row = ctk.CTkFrame(input_card, fg_color="transparent")
        input_row.pack(fill="x", padx=16, pady=12)

        self.entry_w1 = ctk.CTkEntry(
            input_row,
            placeholder_text="Từ thứ nhất (VD: tall)",
            height=40,
            font=ctk.CTkFont(size=14),
            fg_color=COLOR_CARD,
            border_color=COLOR_CARD_BORDER
        )
        self.entry_w1.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.entry_w1.bind("<Return>", lambda event: self.classify_pair())

        btn_swap = ctk.CTkButton(
            input_row,
            text="⇄",
            width=36,
            height=40,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=COLOR_CARD,
            hover_color=COLOR_CARD_BORDER,
            text_color=COLOR_TEXT_MAIN,
            command=self.swap_words
        )
        btn_swap.pack(side="left", padx=2)

        self.entry_w2 = ctk.CTkEntry(
            input_row,
            placeholder_text="Từ thứ hai (VD: short)",
            height=40,
            font=ctk.CTkFont(size=14),
            fg_color=COLOR_CARD,
            border_color=COLOR_CARD_BORDER
        )
        self.entry_w2.pack(side="left", fill="x", expand=True, padx=(8, 12))
        self.entry_w2.bind("<Return>", lambda event: self.classify_pair())

        btn_run = ctk.CTkButton(
            input_row,
            text="🚀  PHÂN LOẠI (RUN k-NN)",
            width=180,
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLOR_ACCENT_BLUE,
            hover_color=COLOR_ACCENT_HOVER,
            command=self.classify_pair
        )
        btn_run.pack(side="right")

        # Hàng nút mẫu nhanh (Quick Samples)
        samples_frame = ctk.CTkFrame(input_card, fg_color="transparent")
        samples_frame.pack(fill="x", padx=16, pady=(0, 12))

        lbl_sample = ctk.CTkLabel(samples_frame, text="Thử cặp mẫu nhanh:", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_TEXT_MUTED)
        lbl_sample.pack(side="left", padx=(0, 8))

        samples = [
            ("tall", "short"),
            ("tall", "high"),
            ("hot", "cold"),
            ("happy", "glad"),
            ("brave", "courageous"),
            ("unhappy", "happy"),
            ("careful", "careless"),
            ("apple", "car")
        ]

        for w1, w2 in samples:
            btn_s = ctk.CTkButton(
                samples_frame,
                text=f"{w1} - {w2}",
                height=26,
                font=ctk.CTkFont(size=11),
                fg_color=COLOR_CARD,
                hover_color=COLOR_CARD_BORDER,
                text_color=COLOR_TEXT_MAIN,
                command=lambda a=w1, b=w2: self.fill_and_classify(a, b)
            )
            btn_s.pack(side="left", padx=3)

        # Khu vực Kết quả (2 cột)
        results_grid = ctk.CTkFrame(self.tab1_frame, fg_color="transparent")
        results_grid.grid(row=2, column=0, sticky="nsew", pady=2)
        results_grid.grid_columnconfigure(0, weight=5)
        results_grid.grid_columnconfigure(1, weight=6)

        # CỘT TRÁI: Nhãn kết quả & Độ tự tin
        self.card_prediction = ctk.CTkFrame(results_grid, fg_color=COLOR_SIDEBAR, corner_radius=10, border_width=1, border_color=COLOR_CARD_BORDER)
        self.card_prediction.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=2)

        lbl_pred_title = ctk.CTkLabel(self.card_prediction, text="KẾT QUẢ DỰ ĐOÁN", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_MUTED)
        lbl_pred_title.pack(anchor="w", padx=16, pady=(14, 8))

        self.lbl_main_badge = ctk.CTkLabel(
            self.card_prediction,
            text="CHƯA CÓ KẾT QUẢ",
            font=ctk.CTkFont(size=17, weight="bold"),
            fg_color=COLOR_CARD,
            text_color=COLOR_TEXT_MUTED,
            corner_radius=8,
            height=46
        )
        self.lbl_main_badge.pack(fill="x", padx=16, pady=4)

        conf_container = ctk.CTkFrame(self.card_prediction, fg_color="transparent")
        conf_container.pack(fill="x", padx=16, pady=12)

        # 1. Đồng nghĩa
        self.lbl_prob_syn = ctk.CTkLabel(conf_container, text="Đồng nghĩa (Synonym): 0.0%", font=ctk.CTkFont(size=11), text_color=COLOR_TEXT_MAIN)
        self.lbl_prob_syn.pack(anchor="w", pady=(2, 2))
        self.bar_prob_syn = ctk.CTkProgressBar(conf_container, height=10, fg_color=COLOR_CARD, progress_color=COLOR_SYNONYM_LIGHT)
        self.bar_prob_syn.pack(fill="x", pady=(0, 8))
        self.bar_prob_syn.set(0)

        # 2. Trái nghĩa
        self.lbl_prob_ant = ctk.CTkLabel(conf_container, text="Trái nghĩa (Antonym): 0.0%", font=ctk.CTkFont(size=11), text_color=COLOR_TEXT_MAIN)
        self.lbl_prob_ant.pack(anchor="w", pady=(2, 2))
        self.bar_prob_ant = ctk.CTkProgressBar(conf_container, height=10, fg_color=COLOR_CARD, progress_color=COLOR_ANTONYM_LIGHT)
        self.bar_prob_ant.pack(fill="x", pady=(0, 8))
        self.bar_prob_ant.set(0)

        # 3. Không liên quan
        self.lbl_prob_unrel = ctk.CTkLabel(conf_container, text="Không liên quan (Unrelated): 0.0%", font=ctk.CTkFont(size=11), text_color=COLOR_TEXT_MAIN)
        self.lbl_prob_unrel.pack(anchor="w", pady=(2, 2))
        self.bar_prob_unrel = ctk.CTkProgressBar(conf_container, height=10, fg_color=COLOR_CARD, progress_color=COLOR_UNRELATED_LIGHT)
        self.bar_prob_unrel.pack(fill="x", pady=(0, 8))
        self.bar_prob_unrel.set(0)

        # CỘT PHẢI: Bảng k Hàng xóm gần nhất
        self.card_neighbors = ctk.CTkFrame(results_grid, fg_color=COLOR_SIDEBAR, corner_radius=10, border_width=1, border_color=COLOR_CARD_BORDER)
        self.card_neighbors.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=2)

        lbl_nb_title = ctk.CTkLabel(
            self.card_neighbors,
            text="🎯 k HÀNG XÓM GẦN NHẤT (k-NEAREST NEIGHBORS)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_TEXT_MUTED
        )
        lbl_nb_title.pack(anchor="w", padx=16, pady=(14, 4))

        lbl_nb_sub = ctk.CTkLabel(
            self.card_neighbors,
            text="Minh chứng cách k-NN đo khoảng cách vector 12 chiều và bỏ phiếu:",
            font=ctk.CTkFont(size=11),
            text_color=COLOR_TEXT_MUTED
        )
        lbl_nb_sub.pack(anchor="w", padx=16, pady=(0, 8))

        self.scroll_neighbors = ctk.CTkScrollableFrame(
            self.card_neighbors,
            fg_color=COLOR_CARD,
            height=210,
            corner_radius=8
        )
        self.scroll_neighbors.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        self.placeholder_nb = ctk.CTkLabel(
            self.scroll_neighbors,
            text="Nhập cặp từ và bấm 'PHÂN LOẠI' để xem danh sách láng giềng gần nhất.",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_TEXT_MUTED
        )
        self.placeholder_nb.pack(pady=40)

    def swap_words(self):
        w1 = self.entry_w1.get()
        w2 = self.entry_w2.get()
        self.entry_w1.delete(0, "end")
        self.entry_w2.delete(0, "end")
        self.entry_w1.insert(0, w2)
        self.entry_w2.insert(0, w1)

    def fill_and_classify(self, w1, w2):
        self.entry_w1.delete(0, "end")
        self.entry_w2.delete(0, "end")
        self.entry_w1.insert(0, w1)
        self.entry_w2.insert(0, w2)
        self.classify_pair()

    def classify_pair(self):
        w1 = self.entry_w1.get().strip().lower()
        w2 = self.entry_w2.get().strip().lower()

        # Kiểm tra tính hợp lệ bằng classify_word_pair
        res = classify_word_pair(w1, w2, model=self.model, scaler=self.scaler, extractor=self.extractor)

        if res["status"] in ("error", "not_found"):
            self.lbl_main_badge.configure(
                text=f"⚠️  {res['message']}",
                fg_color=COLOR_ANTONYM,
                text_color="#ffffff"
            )
            return

        pred_label = res["pred_label"]
        probs = res["probabilities"]

        # Cập nhật thẻ kết quả lớn
        if pred_label == 1:
            self.lbl_main_badge.configure(
                text=f"🟢  ĐỒNG NGHĨA (SYNONYM) — {probs[1]*100:.1f}%",
                fg_color=COLOR_SYNONYM,
                text_color="#ffffff"
            )
        elif pred_label == 0:
            self.lbl_main_badge.configure(
                text=f"🔴  TRÁI NGHĨA (ANTONYM) — {probs[0]*100:.1f}%",
                fg_color=COLOR_ANTONYM,
                text_color="#ffffff"
            )
        else:
            self.lbl_main_badge.configure(
                text=f"⚪  KHÔNG LIÊN QUAN (UNRELATED) — {probs[2]*100:.1f}%",
                fg_color=COLOR_UNRELATED,
                text_color="#c9d1d9"
            )

        # Cập nhật 3 thanh tiến trình
        self.lbl_prob_syn.configure(text=f"Đồng nghĩa (Synonym): {probs[1]*100:.1f}%")
        self.bar_prob_syn.set(probs[1])

        self.lbl_prob_ant.configure(text=f"Trái nghĩa (Antonym): {probs[0]*100:.1f}%")
        self.bar_prob_ant.set(probs[0])

        self.lbl_prob_unrel.configure(text=f"Không liên quan (Unrelated): {probs[2]*100:.1f}%")
        self.bar_prob_unrel.set(probs[2])

        # Lấy k láng giềng gần nhất
        k_neighbors = min(self.current_k, len(self.train_pairs))
        scaled_vector = self.scaler.transform([res["feature_vector"]])
        distances, indices = self.model.kneighbors(scaled_vector, n_neighbors=k_neighbors)

        for widget in self.scroll_neighbors.winfo_children():
            widget.destroy()

        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            nb_pair = self.train_pairs[idx]
            nb_lbl = self.train_labels[idx]
            vote_weight = 1.0 / (dist + 1e-5)

            item_row = ctk.CTkFrame(self.scroll_neighbors, fg_color=COLOR_SIDEBAR, height=36, corner_radius=6)
            item_row.pack(fill="x", pady=2, padx=2)

            lbl_rank = ctk.CTkLabel(item_row, text=f"#{rank}", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_ACCENT_BLUE, width=32)
            lbl_rank.pack(side="left", padx=(8, 4))

            lbl_words = ctk.CTkLabel(item_row, text=f"({nb_pair[0]}, {nb_pair[1]})", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_MAIN)
            lbl_words.pack(side="left", padx=6)

            if nb_lbl == 1:
                badge_text, badge_color = "Đồng nghĩa", COLOR_SYNONYM
            elif nb_lbl == 0:
                badge_text, badge_color = "Trái nghĩa", COLOR_ANTONYM
            else:
                badge_text, badge_color = "Không liên quan", COLOR_UNRELATED

            lbl_badge = ctk.CTkLabel(item_row, text=f" {badge_text} ", font=ctk.CTkFont(size=10, weight="bold"), fg_color=badge_color, text_color="#ffffff", corner_radius=4)
            lbl_badge.pack(side="left", padx=8)

            dist_text = f"dist: {dist:.3f} | vote: {vote_weight:.1f}"
            lbl_dist = ctk.CTkLabel(item_row, text=dist_text, font=ctk.CTkFont(size=11), text_color=COLOR_TEXT_MUTED)
            lbl_dist.pack(side="right", padx=(4, 10))

    # =========================================================================
    # TAB 2: TRA CỨU 1 TỪ (WORD EXPLORER) - BỘ LỌC NGƯỠNG TỰ TIN >= 70%
    # =========================================================================
    def build_tab2(self):
        self.tab2_frame.grid_rowconfigure(2, weight=1)
        self.tab2_frame.grid_columnconfigure(0, weight=1)
        self.tab2_frame.grid_columnconfigure(1, weight=1)

        # Header Tab 2
        header = ctk.CTkFrame(self.tab2_frame, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        title = ctk.CTkLabel(header, text="📖 Tra cứu 1 Từ (Word Explorer)", font=ctk.CTkFont(size=20, weight="bold"), text_color=COLOR_TEXT_MAIN)
        title.pack(anchor="w")

        desc = ctk.CTkLabel(
            header,
            text="Quét nghĩa thông dụng trong WordNet, phân loại thành 2 cột Đồng nghĩa & Trái nghĩa với bộ lọc ngưỡng tin cậy >= 70%.",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_TEXT_MUTED
        )
        desc.pack(anchor="w", pady=(2, 0))

        # Thanh nhập từ cần khám phá
        explore_input_card = ctk.CTkFrame(self.tab2_frame, fg_color=COLOR_SIDEBAR, corner_radius=10, border_width=1, border_color=COLOR_CARD_BORDER)
        explore_input_card.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))

        input_row = ctk.CTkFrame(explore_input_card, fg_color="transparent")
        input_row.pack(fill="x", padx=16, pady=12)

        self.entry_explorer = ctk.CTkEntry(
            input_row,
            placeholder_text="Nhập 1 từ tiếng Anh (VD: tall, happy, fast, strong, rich, cold)...",
            height=40,
            font=ctk.CTkFont(size=14),
            fg_color=COLOR_CARD,
            border_color=COLOR_CARD_BORDER
        )
        self.entry_explorer.pack(side="left", fill="x", expand=True, padx=(0, 12))
        self.entry_explorer.bind("<Return>", lambda event: self.run_word_explorer())

        btn_explore = ctk.CTkButton(
            input_row,
            text="🔍  QUÉT TỪ LIÊN QUAN",
            width=200,
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLOR_ACCENT_BLUE,
            hover_color=COLOR_ACCENT_HOVER,
            command=self.run_word_explorer
        )
        btn_explore.pack(side="right")

        # Nút từ gợi ý nhanh
        quick_words = ["tall", "happy", "fast", "strong", "rich", "cold", "apple", "table"]
        quick_frame = ctk.CTkFrame(explore_input_card, fg_color="transparent")
        quick_frame.pack(fill="x", padx=16, pady=(0, 10))

        lbl_qw = ctk.CTkLabel(quick_frame, text="Từ mẫu nhanh:", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_TEXT_MUTED)
        lbl_qw.pack(side="left", padx=(0, 8))

        for qw in quick_words:
            btn_qw = ctk.CTkButton(
                quick_frame,
                text=qw,
                height=24,
                width=55,
                font=ctk.CTkFont(size=11),
                fg_color=COLOR_CARD,
                hover_color=COLOR_CARD_BORDER,
                command=lambda w=qw: self.fill_explorer_and_run(w)
            )
            btn_qw.pack(side="left", padx=3)

        # 2 CỘT KẾT QUẢ: ĐỒNG NGHĨA (TRÁI) VÀ TRÁI NGHĨA (PHẢI)
        # Cột Đồng nghĩa
        self.card_syn_col = ctk.CTkFrame(self.tab2_frame, fg_color=COLOR_SIDEBAR, corner_radius=10, border_width=1, border_color=COLOR_CARD_BORDER)
        self.card_syn_col.grid(row=2, column=0, sticky="nsew", padx=(0, 8), pady=2)
        self.card_syn_col.grid_rowconfigure(1, weight=1)
        self.card_syn_col.grid_columnconfigure(0, weight=1)

        self.lbl_syn_col_header = ctk.CTkLabel(
            self.card_syn_col,
            text="🟢  CỘT TỪ ĐỒNG NGHĨA (SYNONYMS) — (0 từ)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_SYNONYM_LIGHT
        )
        self.lbl_syn_col_header.grid(row=0, column=0, sticky="w", padx=16, pady=(14, 8))

        self.scroll_syn_col = ctk.CTkScrollableFrame(self.card_syn_col, fg_color=COLOR_CARD, corner_radius=8)
        self.scroll_syn_col.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

        # Cột Trái nghĩa
        self.card_ant_col = ctk.CTkFrame(self.tab2_frame, fg_color=COLOR_SIDEBAR, corner_radius=10, border_width=1, border_color=COLOR_CARD_BORDER)
        self.card_ant_col.grid(row=2, column=1, sticky="nsew", padx=(8, 0), pady=2)
        self.card_ant_col.grid_rowconfigure(1, weight=1)
        self.card_ant_col.grid_columnconfigure(0, weight=1)

        self.lbl_ant_col_header = ctk.CTkLabel(
            self.card_ant_col,
            text="🔴  CỘT TỪ TRÁI NGHĨA (ANTONYMS) — (0 từ)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_ANTONYM_LIGHT
        )
        self.lbl_ant_col_header.grid(row=0, column=0, sticky="w", padx=16, pady=(14, 8))

        self.scroll_ant_col = ctk.CTkScrollableFrame(self.card_ant_col, fg_color=COLOR_CARD, corner_radius=8)
        self.scroll_ant_col.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

    def fill_explorer_and_run(self, word):
        self.entry_explorer.delete(0, "end")
        self.entry_explorer.insert(0, word)
        self.run_word_explorer()

    def run_word_explorer(self):
        target = self.entry_explorer.get().strip().lower()

        # Xóa kết quả cũ
        for widget in self.scroll_syn_col.winfo_children():
            widget.destroy()
        for widget in self.scroll_ant_col.winfo_children():
            widget.destroy()

        # Gọi logic nghiệp vụ dùng chung từ word_explorer.py
        result = explore_single_word(
            target,
            model=self.model,
            scaler=self.scaler,
            extractor=self.extractor,
            min_confidence=70.0
        )

        if result["status"] == "error":
            lbl_err1 = ctk.CTkLabel(self.scroll_syn_col, text=result["message"], font=ctk.CTkFont(size=12), text_color="#f85149", justify="center")
            lbl_err1.pack(pady=40)
            lbl_err2 = ctk.CTkLabel(self.scroll_ant_col, text=result["message"], font=ctk.CTkFont(size=12), text_color="#f85149", justify="center")
            lbl_err2.pack(pady=40)
            self.lbl_syn_col_header.configure(text="🟢  CỘT TỪ ĐỒNG NGHĨA (SYNONYMS) — (0 từ)")
            self.lbl_ant_col_header.configure(text="🔴  CỘT TỪ TRÁI NGHĨA (ANTONYMS) — (0 từ)")
            return

        if result["status"] == "not_found":
            lbl_w1 = ctk.CTkLabel(self.scroll_syn_col, text=result["message"], font=ctk.CTkFont(size=12), text_color="#f85149", justify="center")
            lbl_w1.pack(pady=40)
            lbl_w2 = ctk.CTkLabel(self.scroll_ant_col, text=result["message"], font=ctk.CTkFont(size=12), text_color="#f85149", justify="center")
            lbl_w2.pack(pady=40)
            self.lbl_syn_col_header.configure(text="🟢  CỘT TỪ ĐỒNG NGHĨA (SYNONYMS) — (0 từ)")
            self.lbl_ant_col_header.configure(text="🔴  CỘT TỪ TRÁI NGHĨA (ANTONYMS) — (0 từ)")
            return

        synonyms_found = result["synonyms"]
        antonyms_found = result["antonyms"]

        # CẬP NHẬT CỘT ĐỒNG NGHĨA
        self.lbl_syn_col_header.configure(text=f"🟢  CỘT TỪ ĐỒNG NGHĨA (SYNONYMS) — ({len(synonyms_found)} từ)")

        if not synonyms_found:
            msg_empty_syn = "Không có từ đồng nghĩa phù hợp đạt ngưỡng tin cậy >= 70%."
            lbl_empty = ctk.CTkLabel(self.scroll_syn_col, text=msg_empty_syn, font=ctk.CTkFont(size=12), text_color=COLOR_TEXT_MUTED, justify="center")
            lbl_empty.pack(pady=40)
        else:
            for cand, conf in synonyms_found:
                row = ctk.CTkFrame(self.scroll_syn_col, fg_color=COLOR_SIDEBAR, height=36, corner_radius=6)
                row.pack(fill="x", pady=2, padx=2)

                lbl_w = ctk.CTkLabel(row, text=cand, font=ctk.CTkFont(size=13, weight="bold"), text_color=COLOR_TEXT_MAIN)
                lbl_w.pack(side="left", padx=12)

                btn_test = ctk.CTkButton(
                    row,
                    text="Thử cặp ->",
                    width=65,
                    height=24,
                    font=ctk.CTkFont(size=10),
                    fg_color=COLOR_CARD,
                    hover_color=COLOR_CARD_BORDER,
                    command=lambda c=cand: self.jump_to_pair_classifier(target, c)
                )
                btn_test.pack(side="right", padx=(4, 8))

                lbl_conf = ctk.CTkLabel(row, text=f"{conf:.1f}%", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_SYNONYM_LIGHT)
                lbl_conf.pack(side="right", padx=6)

        # CẬP NHẬT CỘT TRÁI NGHĨA
        self.lbl_ant_col_header.configure(text=f"🔴  CỘT TỪ TRÁI NGHĨA (ANTONYMS) — ({len(antonyms_found)} từ)")

        if not antonyms_found:
            msg_empty_ant = "Không có từ trái nghĩa phù hợp cho từ này\n(đặc biệt đối với danh từ chỉ thực thể/sự vật cụ thể)."
            lbl_empty = ctk.CTkLabel(self.scroll_ant_col, text=msg_empty_ant, font=ctk.CTkFont(size=12), text_color=COLOR_TEXT_MUTED, justify="center")
            lbl_empty.pack(pady=40)
        else:
            for cand, conf in antonyms_found:
                row = ctk.CTkFrame(self.scroll_ant_col, fg_color=COLOR_SIDEBAR, height=36, corner_radius=6)
                row.pack(fill="x", pady=2, padx=2)

                lbl_w = ctk.CTkLabel(row, text=cand, font=ctk.CTkFont(size=13, weight="bold"), text_color=COLOR_TEXT_MAIN)
                lbl_w.pack(side="left", padx=12)

                btn_test = ctk.CTkButton(
                    row,
                    text="Thử cặp ->",
                    width=65,
                    height=24,
                    font=ctk.CTkFont(size=10),
                    fg_color=COLOR_CARD,
                    hover_color=COLOR_CARD_BORDER,
                    command=lambda c=cand: self.jump_to_pair_classifier(target, c)
                )
                btn_test.pack(side="right", padx=(4, 8))

                lbl_conf = ctk.CTkLabel(row, text=f"{conf:.1f}%", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_ANTONYM_LIGHT)
                lbl_conf.pack(side="right", padx=6)

    def jump_to_pair_classifier(self, w1, w2):
        self.select_tab("tab1")
        self.fill_and_classify(w1, w2)

    # =========================================================================
    # TAB 3: MÔ HÌNH & BENCHMARK (MODEL BENCHMARK)
    # =========================================================================
    def build_tab3(self):
        self.tab3_frame.grid_rowconfigure(3, weight=1)
        self.tab3_frame.grid_columnconfigure(0, weight=1)

        # Header Tab 3
        header = ctk.CTkFrame(self.tab3_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        title = ctk.CTkLabel(header, text="⚙️ Huấn luyện & Thông số Mô hình (Model Benchmark)", font=ctk.CTkFont(size=20, weight="bold"), text_color=COLOR_TEXT_MAIN)
        title.pack(anchor="w")

        desc = ctk.CTkLabel(
            header,
            text="Tùy chỉnh tham số k-NN trên không gian 12 đặc trưng cô đọng, huấn luyện lại siêu tốc (< 0.03s) và xem biểu đồ báo cáo.",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_TEXT_MUTED
        )
        desc.pack(anchor="w", pady=(2, 0))

        # 4 Thẻ chỉ số đánh giá (KPI Cards)
        kpi_frame = ctk.CTkFrame(self.tab3_frame, fg_color="transparent")
        kpi_frame.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        for col in range(4):
            kpi_frame.grid_columnconfigure(col, weight=1)

        metrics = self.pipeline_data["metrics"]

        self.kpi_acc = self.create_kpi_card(kpi_frame, 0, "🎯  Accuracy (Độ chính xác)", f"{metrics['accuracy']*100:.2f}%", COLOR_ACCENT_BLUE)
        self.kpi_f1 = self.create_kpi_card(kpi_frame, 1, "⚖️  Macro F1-Score", f"{metrics['f1']*100:.2f}%", COLOR_SYNONYM_LIGHT)
        self.kpi_prec = self.create_kpi_card(kpi_frame, 2, "🔍  Macro Precision", f"{metrics['precision']*100:.2f}%", "#a371f7")
        self.kpi_rec = self.create_kpi_card(kpi_frame, 3, "📡  Macro Recall", f"{metrics['recall']*100:.2f}%", "#e3b341")

        # Khung điều khiển huấn luyện lại
        control_card = ctk.CTkFrame(self.tab3_frame, fg_color=COLOR_SIDEBAR, corner_radius=10, border_width=1, border_color=COLOR_CARD_BORDER)
        control_card.grid(row=2, column=0, sticky="ew", pady=(0, 12))

        ctrl_row = ctk.CTkFrame(control_card, fg_color="transparent")
        ctrl_row.pack(fill="x", padx=16, pady=12)

        # 1. Slider k
        k_frame = ctk.CTkFrame(ctrl_row, fg_color="transparent")
        k_frame.pack(side="left", padx=(0, 16))

        self.lbl_k_slider = ctk.CTkLabel(k_frame, text=f"Số láng giềng k: {self.current_k}", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_MAIN)
        self.lbl_k_slider.pack(anchor="w")

        self.slider_k = ctk.CTkSlider(
            k_frame,
            from_=1,
            to=21,
            number_of_steps=20,
            width=180,
            command=self.on_slider_k_changed
        )
        self.slider_k.set(self.current_k)
        self.slider_k.pack(pady=(4, 0))

        # 2. Metric dropdown
        metric_frame = ctk.CTkFrame(ctrl_row, fg_color="transparent")
        metric_frame.pack(side="left", padx=12)

        lbl_m = ctk.CTkLabel(metric_frame, text="Metric khoảng cách:", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_MAIN)
        lbl_m.pack(anchor="w")

        self.opt_metric = ctk.CTkOptionMenu(
            metric_frame,
            values=["euclidean", "manhattan", "cosine"],
            width=130,
            fg_color=COLOR_CARD,
            button_color=COLOR_CARD_BORDER,
            text_color=COLOR_TEXT_MAIN
        )
        self.opt_metric.set("euclidean")
        self.opt_metric.pack(pady=(4, 0))

        # 3. Weights dropdown
        weight_frame = ctk.CTkFrame(ctrl_row, fg_color="transparent")
        weight_frame.pack(side="left", padx=12)

        lbl_w = ctk.CTkLabel(weight_frame, text="Trọng số láng giềng:", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_MAIN)
        lbl_w.pack(anchor="w")

        self.opt_weight = ctk.CTkOptionMenu(
            weight_frame,
            values=["distance", "uniform"],
            width=120,
            fg_color=COLOR_CARD,
            button_color=COLOR_CARD_BORDER,
            text_color=COLOR_TEXT_MAIN
        )
        self.opt_weight.set("distance")
        self.opt_weight.pack(pady=(4, 0))

        # 4. Button Huấn luyện lại
        btn_retrain = ctk.CTkButton(
            ctrl_row,
            text="⚡  HUẤN LUYỆN LẠI",
            width=160,
            height=36,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=COLOR_ACCENT_BLUE,
            hover_color=COLOR_ACCENT_HOVER,
            command=self.retrain_model
        )
        btn_retrain.pack(side="right", pady=(10, 0))

        # Khung Xem Đồ thị Báo cáo
        plot_card = ctk.CTkFrame(self.tab3_frame, fg_color=COLOR_SIDEBAR, corner_radius=10, border_width=1, border_color=COLOR_CARD_BORDER)
        plot_card.grid(row=3, column=0, sticky="nsew", pady=2)
        plot_card.grid_rowconfigure(1, weight=1)
        plot_card.grid_columnconfigure(0, weight=1)

        plot_top = ctk.CTkFrame(plot_card, fg_color="transparent")
        plot_top.grid(row=0, column=0, sticky="ew", padx=16, pady=(10, 6))

        self.seg_plot = ctk.CTkSegmentedButton(
            plot_top,
            values=["Biểu đồ Tối ưu k (k_optimization.png)", "Ma trận Nhầm lẫn (confusion_matrix.png)"],
            command=lambda v: self.refresh_plot_image()
        )
        self.seg_plot.set("Biểu đồ Tối ưu k (k_optimization.png)")
        self.seg_plot.pack(side="left")

        btn_open_folder = ctk.CTkButton(
            plot_top,
            text="📂  Mở Thư mục Plots",
            width=140,
            height=28,
            font=ctk.CTkFont(size=11),
            fg_color=COLOR_CARD,
            hover_color=COLOR_CARD_BORDER,
            command=self.open_plots_folder
        )
        btn_open_folder.pack(side="right")

        self.plot_display = ctk.CTkLabel(plot_card, text="", corner_radius=8)
        self.plot_display.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 12))

    def create_kpi_card(self, parent, col, title, value, color):
        card = ctk.CTkFrame(parent, fg_color=COLOR_SIDEBAR, corner_radius=10, border_width=1, border_color=COLOR_CARD_BORDER)
        card.grid(row=0, column=col, sticky="ew", padx=4)

        t = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_TEXT_MUTED)
        t.pack(anchor="w", padx=14, pady=(10, 2))

        val_lbl = ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=20, weight="bold"), text_color=color)
        val_lbl.pack(anchor="w", padx=14, pady=(0, 10))
        return val_lbl

    def on_slider_k_changed(self, val):
        self.current_k = int(round(val))
        self.lbl_k_slider.configure(text=f"Số láng giềng k: {self.current_k}")

    def retrain_model(self):
        """Huấn luyện lại k-NN trên không gian 12 đặc trưng trong ~0.02 giây."""
        k = self.current_k
        metric = self.opt_metric.get()
        weight = self.opt_weight.get()

        new_clf = KNeighborsClassifier(n_neighbors=k, weights=weight, metric=metric)
        new_clf.fit(self.train_features_scaled, self.train_labels)

        # Đánh giá trên tập test
        y_pred = new_clf.predict(self.test_features_scaled)
        acc = accuracy_score(self.test_labels, y_pred)
        f1 = f1_score(self.test_labels, y_pred, average="macro")
        prec = precision_score(self.test_labels, y_pred, average="macro")
        rec = recall_score(self.test_labels, y_pred, average="macro")

        # Cập nhật model hoạt động
        self.model = new_clf
        self.kpi_acc.configure(text=f"{acc*100:.2f}%")
        self.kpi_f1.configure(text=f"{f1*100:.2f}%")
        self.kpi_prec.configure(text=f"{prec*100:.2f}%")
        self.kpi_rec.configure(text=f"{rec*100:.2f}%")

        self.lbl_sidebar_k.configure(
            text=f"Thuật toán: k-NN (k = {k})\nĐộ chính xác: {acc * 100:.1f}%\nTập huấn luyện: {len(self.train_pairs)} cặp"
        )

    def refresh_plot_image(self):
        """Hiển thị ảnh đồ thị tương ứng được chọn."""
        choice = self.seg_plot.get()
        if "k_optimization" in choice:
            img_path = os.path.join(os.path.dirname(__file__), "plots", "k_optimization.png")
        else:
            img_path = os.path.join(os.path.dirname(__file__), "plots", "confusion_matrix.png")

        if os.path.exists(img_path):
            try:
                pil_img = Image.open(img_path)
                w, h = 480, 220
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(w, h))
                self.plot_display.configure(image=ctk_img, text="")
            except Exception as e:
                self.plot_display.configure(image=None, text=f"Lỗi nạp ảnh: {e}")
        else:
            self.plot_display.configure(image=None, text="Chưa tìm thấy tệp ảnh biểu đồ trong thư mục plots/")

    def open_plots_folder(self):
        plots_dir = os.path.join(os.path.dirname(__file__), "plots")
        os.makedirs(plots_dir, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(plots_dir)
        else:
            subprocess.Popen(["xdg-open", plots_dir])


if __name__ == "__main__":
    app = KNNWordPairApp()
    app.mainloop()
