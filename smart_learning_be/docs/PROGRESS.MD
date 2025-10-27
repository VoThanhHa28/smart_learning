# Báo cáo tiến độ 10/2
## Đã làm được
- Triển khai pipeline đọc PDF → chunk → embed → lưu Milvus.
- Hỗ trợ hybrid retrieve (BM25 (đã tắt) + dense) và reranker để tăng chất lượng.
- Có threshold lọc context kém, log debug để theo dõi pipeline.

## Hạn chế
- Text còn noise (TOC, dấu chấm lẻ, heading chưa chuẩn).
- Tốc độ parse ~3–4 phút cho 243 trang → chưa tối ưu.

## Kế hoạch tiếp theo
- Chuyển sang Docling (layout model + GPU + CPU đa nhân).
- Giữ heading/TOC ở metadata, chỉ embed body + code.
- Tối ưu tốc độ bằng batch insert + song song parse, giảm OCR thừa.
