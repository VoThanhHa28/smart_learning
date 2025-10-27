import os, re

def slugify_filename(file_path: str, course_id: str) -> str:
    """
    Tạo slug ngắn, an toàn, loại bỏ UUID hoặc ký tự đặc biệt.
    Ví dụ: uploads/P1_e27f72ae_thinkpython2.pdf → P1_thinkpython2
    """
    base = os.path.basename(file_path)
    name, _ = os.path.splitext(base)
    # loại UUID nếu có
    name = re.sub(r"[a-f0-9]{8}_[a-f0-9]{4}_[a-f0-9]{4}_[a-f0-9]{4}_[a-f0-9]{12}_?", "", name)
    # chỉ giữ ký tự an toàn
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    return f"{course_id}_{name}".strip("_")
