from .docling_pipeline import DoclingPipeline

_pipeline = DoclingPipeline(
    lang="vi",
    enable_ocr=False,
    enable_code=False,
    enable_formula=False,
    enable_table=False,
    fast_mode=True  # ✅ bật enrichment thật
)

async def ingest_file(file_path: str, course_id: str, subject: str):
    result = await _pipeline.process_file(file_path, course_id=course_id, subject=subject)
    return {
        "course_id": course_id,
        "subject": subject,
        "chunks_indexed": result["chunks_indexed"],
        "enrichment_file": result["enrichment_file"]
    }

async def ingest_many(files: list[str], course_id: str, subject: str):
    results = await _pipeline.process_many(files, course_id=course_id, subject=subject)
    return results
