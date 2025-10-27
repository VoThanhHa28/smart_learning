import os, asyncio, time
from concurrent.futures import ThreadPoolExecutor
# 1. Import State và timing
from ..rag_state import State, _update_timing
from ..rag_utils import build_context
# 2. Import build_unified_block (CACHED_PROMPT không cần nữa)
from ..prompts.prompt_utils import build_unified_block
# 3. Bỏ import sync_stream_generate
# from ...infrastructure.llm.llm_utils import sync_stream_generate

async def generate(state: State) -> State: # <-- Đổi tên thành generate_simple
    """
    Node tạo sinh cho câu hỏi đơn giản (luồng retrieve -> generate).
    CHỈ chuẩn bị prompt, KHÔNG gọi LLM.
    """
    t_start = time.perf_counter()
    print("➡️  [Node] generate_simple: Bắt đầu chuẩn bị prompt...")

    # --- Kiểm tra dừng sớm (GIỮ NGUYÊN) ---
    # Nếu state đã có answer (từ node analyze chẳng hạn), không làm gì cả
    if state.get("answer"):
        print("  [generate_simple] ⏭️ Bỏ qua vì đã có câu trả lời.")
        # Cập nhật timing (thời gian = 0)
        return _update_timing(state, "generate_simple", t_start)
    # --- Kết thúc kiểm tra ---

    docs_in = state.get("context", []) # Context lấy từ node retrieve trước đó
    if not docs_in:
        print("  [generate_simple] ⚠️ Không có context từ retrieve, không thể tạo prompt.")
        # Đặt câu trả lời lỗi và dừng
        answer = "Xin lỗi, mình không tìm thấy thông tin để trả lời câu hỏi này."
        new_state = {**state, "answer": answer, "status": "stop"}
        return _update_timing(new_state, "generate_simple", t_start)

    # --- Logic chọn context (GIỮ NGUYÊN) ---
    # (Copy logic chọn main_pick, supp_pick, build context từ file cũ)
    docs_in_sorted = docs_in # Giả sử retrieve đã sắp xếp
    course_id = str(state.get("course_id", "")).lower()
    same_course = [d for d in docs_in_sorted if str(d.metadata.get("course_id","")).lower() == course_id]
    other = [d for d in docs_in_sorted if d not in same_course]

    # Không cần is_definition nữa vì analyze V2 không cung cấp intent chi tiết
    limit_main = 5 # Số lượng context chính cố định
    limit_supp = 2 # Số lượng context phụ cố định

    main_pick = (same_course or docs_in_sorted)[:limit_main]
    supp_pick = other[:limit_supp]

    main_context = build_context(main_pick) if main_pick else "Không có ngữ cảnh chính."
    supp_context = build_context(supp_pick) if supp_pick else ""
    context = main_context + ("\n\n## Ngữ cảnh bổ sung\n" + supp_context if supp_context else "")
    # --- Kết thúc logic context ---

    # --- Lấy thông tin khác từ state ---
    analyze_meta = state.get("analyze_meta", {}) or {}
    question = state.get("question_user") or state.get("question") or ""
    # Lấy topic từ analyze_meta (nếu có, cho câu hỏi đơn)
    topic = analyze_meta.get("topic") or "" # Hoặc lấy từ subq đầu tiên nếu có
    if not topic and analyze_meta.get("sub_questions_with_intent"):
         topic = analyze_meta["sub_questions_with_intent"][0].get("topic", "")

    # Lấy doc_source từ context đầu tiên (nếu có)
    m0 = (main_pick[0].metadata if main_pick else (docs_in_sorted[0].metadata if docs_in_sorted else {})) or {}
    doc_source = m0.get("doc_source", "Tài liệu học tập")
    # --- Kết thúc lấy thông tin ---

    # --- Xây dựng Prompt (Dùng prompt V3 đã sửa) ---
    # Bỏ soft_hints vì analyze V2 không cung cấp intent chi tiết
    soft_hints = "Tập trung trả lời đầy đủ và chính xác câu hỏi dựa trên ngữ cảnh, đặc biệt chú ý trích dẫn nguồn [D#-p#] nếu có."

    # Tạo prompt text
    prompt_text = build_unified_block(
        subject=state.get("subject", "General"),
        context=context or "Không có ngữ cảnh phù hợp.",
        question=question, # Câu hỏi gốc từ user
        topic=topic,
        doc_source=doc_source,
        soft_hints=soft_hints
    )
    # --- Kết thúc xây dựng Prompt ---

    print(f"  [generate_simple] ✅ Chuẩn bị xong prompt (len={len(prompt_text)} chars).")

    # --- LƯU PROMPT VÀO STATE, KHÔNG GỌI LLM ---
    # Lưu prompt vào state để bước sau (trong chain.py) có thể lấy và stream
    new_state = {
        **state,
        "final_prompt_text": prompt_text, # Key mới để lưu prompt
        "final_context_docs": main_pick + supp_pick # Lưu cả docs đã dùng cho prompt
        # KHÔNG CÓ "answer" ở đây
    }
    # --- Kết thúc lưu state ---

    # Cập nhật timing và trả về state đã cập nhật
    new_state = _update_timing(new_state, "generate_simple", t_start)
    return new_state