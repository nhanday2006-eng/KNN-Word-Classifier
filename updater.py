"""
updater.py
Module tự động cập nhật và ghi đè source code từ GitHub Repository.
Sử dụng 100% thư viện chuẩn của Python:
- urllib.request, zipfile, shutil, os, subprocess, tempfile, sys.

TÍCH HỢP CHỐT AN TOÀN (SAFETY GUARD) CHO MÁY DEVELOPER:
1. Kiểm tra file version.txt: Tuyệt đối KHÔNG ghi đè nếu phiên bản trên GitHub
   thấp hơn hoặc bằng phiên bản trên máy hiện tại (tránh kéo code cũ đè code mới).
2. Kiểm tra Git status: Nếu máy đang có thay đổi local chưa commit/push,
   chặn cập nhật ngay lập tức để bảo vệ mã nguồn của lập trình viên.
"""

import os
import re
import sys
import shutil
import tempfile
import zipfile
import subprocess
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_REPO_URL = "https://github.com/nhanday2006-eng/KNN-Word-Classifier"
CONFIG_FILE = "repo_config.txt"
VERSION_FILE = "version.txt"
OVERWRITE_EXTENSIONS = {
    ".py", ".pkl", ".csv", ".png", ".md", ".bat", ".sh", ".txt", ".gitignore", ".json"
}


def parse_version_tuple(v_str: str) -> tuple:
    """Chuyển chuỗi version 'v2.0' hoặc '2.1.0' thành tuple so sánh được (2, 0)."""
    if not v_str:
        return (0, 0)
    cleaned = v_str.strip().lstrip("vV")
    parts = re.findall(r"\d+", cleaned)
    if not parts:
        return (0, 0)
    return tuple(int(p) for p in parts)


def get_local_version(base_dir=None) -> str:
    """Đọc phiên bản hiện tại từ file version.txt trên máy."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    v_path = os.path.join(base_dir, VERSION_FILE)
    if os.path.exists(v_path):
        try:
            with open(v_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return content
        except Exception:
            pass
    return "v2.0"


def get_repo_url(base_dir=None) -> str:
    """Đọc URL repository GitHub từ file repo_config.txt."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, CONFIG_FILE)

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        return line
        except Exception:
            pass
    return DEFAULT_REPO_URL


def get_remote_version_raw(repo_url: str) -> str:
    """
    Thử đọc file version.txt trực tiếp từ GitHub qua raw URL.
    Ví dụ: https://raw.githubusercontent.com/owner/repo/main/version.txt
    """
    clean_url = repo_url.strip().rstrip("/")
    if clean_url.endswith(".git"):
        clean_url = clean_url[:-4]

    # Phân tích owner và repo
    match = re.search(r"github\.com/([^/]+)/([^/]+)", clean_url)
    if not match:
        return ""

    owner, repo = match.group(1), match.group(2)
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{VERSION_FILE}"

    try:
        req = urllib.request.Request(
            raw_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Antigravity-Updater/2.0"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            content = resp.read().decode("utf-8").strip()
            return content
    except Exception:
        # Nếu repo chưa có file version.txt (bản cũ trên GitHub)
        return ""


def update_from_github(repo_url=None, target_dir=None, log_fn=None) -> dict:
    """
    Tự động cập nhật an toàn từ GitHub:
    - Kiểm tra chốt an toàn phiên bản và thay đổi local trước khi ghi đè.
    - Tuyệt đối KHÔNG ghi đè nếu phiên bản trên GitHub cũ hơn hoặc bằng máy hiện tại.
    """
    def log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(f"[Updater] {msg}")

    if target_dir is None:
        target_dir = os.path.dirname(os.path.abspath(__file__))

    if not repo_url:
        repo_url = get_repo_url(target_dir)

    repo_url = repo_url.strip().rstrip("/")
    if repo_url.endswith(".git"):
        repo_url = repo_url[:-4]

    local_ver_str = get_local_version(target_dir)
    local_ver_tuple = parse_version_tuple(local_ver_str)
    log(f"Phiên bản hiện tại trên máy: {local_ver_str}")

    git_dir = os.path.join(target_dir, ".git")

    # =========================================================================
    # CHỐT AN TOÀN 1: KIỂM TRA THAY ĐỔI LOCAL NẾU DÙNG GIT
    # =========================================================================
    if os.path.isdir(git_dir):
        try:
            status_res = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=target_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            if status_res.returncode == 0 and status_res.stdout.strip():
                warn_msg = (
                    "Phát hiện mã nguồn trên máy bạn đang có các thay đổi local chưa commit/push lên GitHub!\n"
                    "Chốt an toàn đã CHẶN cập nhật để tránh mất code mới của bạn.\n"
                    "Vui lòng commit và push code lên GitHub trước khi chạy Update."
                )
                log(warn_msg)
                return {"status": "local_changes", "message": warn_msg}
        except Exception:
            pass

    # =========================================================================
    # CHỐT AN TOÀN 2: KIỂM TRA PHIÊN BẢN TRÊN GITHUB TRƯỚC KHI TẢI
    # =========================================================================
    log("Đang kiểm tra phiên bản trên GitHub...")
    remote_ver_str = get_remote_version_raw(repo_url)

    if not remote_ver_str:
        log("GitHub chưa có file version.txt (bản trên GitHub là bản cũ hơn máy bạn).")
        warn_msg = (
            f"Phiên bản trên máy hiện tại ({local_ver_str}) là phiên bản mới nhất!\n"
            f"Trên GitHub repository chưa có file {VERSION_FILE} (bản trên GitHub là bản cũ hơn).\n"
            f"Chốt an toàn đã NGĂN cập nhật để không làm mất code mới trên máy của bạn."
        )
        return {"status": "up_to_date", "message": warn_msg}

    remote_ver_tuple = parse_version_tuple(remote_ver_str)
    log(f"Phiên bản phát hiện trên GitHub: {remote_ver_str}")

    if remote_ver_tuple <= local_ver_tuple:
        msg = (
            f"Ứng dụng của bạn đã ở phiên bản mới nhất ({local_ver_str})!\n"
            f"Bản trên GitHub hiện tại là: {remote_ver_str}.\n"
            f"Không cần cập nhật."
        )
        log(msg)
        return {"status": "up_to_date", "message": msg}

    # =========================================================================
    # NẾU GITHUB CÓ BẢN MỚI HƠN THẬT SỰ (remote_ver > local_ver): TIẾN HÀNH CẬP NHẬT
    # =========================================================================
    log(f"Phát hiện bản mới {remote_ver_str} > {local_ver_str}! Bắt đầu cập nhật...")

    # Cách 1: Git CLI
    if os.path.isdir(git_dir):
        log("Đang cập nhật bằng Git CLI...")
        try:
            fetch_res = subprocess.run(["git", "fetch", "--all"], cwd=target_dir, capture_output=True, text=True, timeout=30)
            if fetch_res.returncode == 0:
                reset_res = subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=target_dir, capture_output=True, text=True, timeout=30)
                if reset_res.returncode == 0:
                    msg = f"Đã cập nhật thành công lên phiên bản {remote_ver_str} qua Git!"
                    log(msg)
                    return {"status": "success", "method": "git", "message": msg}
        except Exception as e:
            log(f"Git CLI lỗi ({e}), chuyển sang phương thức tải ZIP...")

    # Cách 2: Tải file ZIP
    zip_url = f"{repo_url}/archive/refs/heads/main.zip"
    log(f"Đang tải gói ZIP từ: {zip_url} ...")

    temp_dir = tempfile.mkdtemp(prefix="knn_update_")
    zip_path = os.path.join(temp_dir, "repo_main.zip")

    try:
        req = urllib.request.Request(
            zip_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Antigravity-Updater/2.0"}
        )
        with urllib.request.urlopen(req, timeout=40) as response, open(zip_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)

        log("Đã tải xong ZIP. Đang tiến hành giải nén...")
        extract_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        extracted_entries = os.listdir(extract_dir)
        if not extracted_entries:
            raise RuntimeError("File ZIP rỗng.")

        source_root = os.path.join(extract_dir, extracted_entries[0])
        if not os.path.isdir(source_root):
            source_root = extract_dir

        # Kiểm tra thêm version bên trong file zip trước khi ghi đè
        zip_ver_path = os.path.join(source_root, VERSION_FILE)
        if os.path.exists(zip_ver_path):
            with open(zip_ver_path, "r", encoding="utf-8") as f:
                zip_ver_tuple = parse_version_tuple(f.read().strip())
                if zip_ver_tuple <= local_ver_tuple:
                    msg = f"Gói ZIP có phiên bản ({zip_ver_tuple}) không mới hơn máy ({local_ver_tuple}). Hủy ghi đè!"
                    return {"status": "up_to_date", "message": msg}

        # Ghi đè file
        updated_count = 0
        for root, dirs, files in os.walk(source_root):
            rel_path = os.path.relpath(root, source_root)
            dest_dir = os.path.normpath(os.path.join(target_dir, rel_path))
            os.makedirs(dest_dir, exist_ok=True)

            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in OVERWRITE_EXTENSIONS or f in ("Start_Tool.bat", "start_tool.sh", ".gitignore"):
                    src_file = os.path.join(root, f)
                    dest_file = os.path.join(dest_dir, f)
                    shutil.copy2(src_file, dest_file)
                    updated_count += 1

        msg = f"Đã cập nhật thành công lên bản {remote_ver_str} ({updated_count} tệp được ghi đè)!"
        log(msg)
        return {
            "status": "success",
            "method": "zip",
            "message": msg,
            "files_updated": updated_count
        }

    except Exception as e:
        err_msg = f"Lỗi khi cập nhật từ GitHub: {e}"
        log(err_msg)
        return {
            "status": "error",
            "method": "zip",
            "message": err_msg
        }
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    print("=" * 60)
    print(" CÔNG CỤ TỰ ĐỘNG CẬP NHẬT TỪ GITHUB (CLI)")
    print("=" * 60)
    res = update_from_github()
    print(f"\nKết quả: {res['status'].upper()} — {res['message']}")
