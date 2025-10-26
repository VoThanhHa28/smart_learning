# app/services/lms/lms_api.py
# -*- coding: utf-8 -*-
import logging

def get_current_homework(course_id: str) -> dict | None:
    """
    Giả lập (Mock) việc gọi API/DB để lấy bài tập về nhà.
    Trong thực tế, bạn sẽ dùng requests, sqlalchemy, v.v.
    """
    logging.info(f"[LMS Tool] Đang lấy bài tập cho course_id='{course_id}'")
    
    # --- Logic giả lập ---
    # (Bạn có thể thay course_id này bằng ID thật để test)
    if course_id == "your_real_course_id_for_testing": 
        return {
            "title": "Bài trắc nghiệm Chương 2: Kiểu dữ liệu",
            "due_date": "Thứ Sáu, 23:59"
        }
    # ---------------------
    
    logging.warning(f"[LMS Tool] Không tìm thấy bài tập cho course_id='{course_id}'")
    return None