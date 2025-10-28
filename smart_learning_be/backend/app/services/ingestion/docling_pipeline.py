# services/docling_pipeline.py
import os
import json
import asyncio
from typing import List, Dict, Any

# --- Imports Docling ---
from docling.document_converter import DocumentConverter
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.pipeline.simple_pipeline import SimplePipeline
from docling_core.types import DoclingDocument
from docling.chunking import HybridChunker
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.pipeline_options import PdfPipelineOptions, PaginatedPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption

# --- Imports Utils ---
try:
    from ..utils.common_utils import slugify_filename
    from ..utils.pdf_utils import is_text_based_pdf
    from ..utils.toc_hybrid import extract_toc_hybrid, update_toc_index_safe
except ImportError:
    from app.services.utils.common_utils import slugify_filename # Fallback
    from app.services.utils.pdf_utils import is_text_based_pdf
    from app.services.utils.toc_hybrid import extract_toc_hybrid, update_toc_index_safe

# --- Imports Langchain & Dependencies ---
from langchain_core.documents import Document
try:
    from app.services.ingestion.vectorstore import upsert_chunks
    from app.infrastructure.llm.embedding import get_embeddings, async_embed_docs
except ImportError:
    from ...rag.retrieval.vectorstore import upsert_chunks # Fallback
    from ...infrastructure.llm.embedding import get_embeddings, async_embed_docs # Fallback

# --- Imports Khác ---
from langsmith import traceable
import requests
import time
import logging
import re
import torch
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
try:
    from transformers import AutoTokenizer
except ImportError:
    logging.warning("transformers library not found. Chunking might be affected.")
    AutoTokenizer = None
import asyncio
from concurrent.futures import ThreadPoolExecutor
import math
import fitz

# --- Cấu hình ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("docling").setLevel(logging.WARNING)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    logging.info("CUDA available. Pytorch performance settings applied.")
else:
     logging.info("CUDA not available. Running on CPU.")

# --- Helper Functions ---
@traceable(name="OCR PDF")
def call_ocr_service(file_path: str) -> str:
    ocr_service_url = os.getenv("OCR_SERVICE_URL", "http://ocr:9000/ocr")
    logging.info(f"OCR: Calling service at {ocr_service_url} for file: {file_path}")
    try:
        with open(file_path, "rb") as f:
            res = requests.post(ocr_service_url, files={"file": f}, timeout=60)
        res.raise_for_status()
        ocr_result = res.json()
        logging.info(f"OCR: Successful for {file_path}")
        return ocr_result.get("text", "")
    except requests.exceptions.RequestException as e:
        logging.error(f"OCR: Request failed for {file_path}: {e}")
        raise RuntimeError(f"OCR service failed: {e}") from e
    except Exception as e:
        logging.error(f"OCR: Error processing for {file_path}: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected OCR error: {e}") from e

@traceable(name="Milvus Upsert Wrapper")
async def upsert_docs_wrapper(docs):
    """Hàm wrapper để trace Langsmith cho upsert_chunks."""
    return await upsert_chunks(docs) # Gọi hàm từ vectorstore.py

@traceable(name="Chunking with UserID")
def hybrid_chunk_docs(dl_doc: DoclingDocument, subject: str, course_id: str, file_path: str, user_id: str): # <-- NHẬN user_id
    """
    Chunks DoclingDocument -> List[Langchain Document], GẮN user_id vào metadata.
    """
    if AutoTokenizer is None:
        raise ImportError("transformers library is required for hybrid chunking.")

    # --- 1. Tokenizer ---
    EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
    try:
        hf_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    except Exception as e:
        logging.error(f"Chunking: Failed to load tokenizer '{EMBED_MODEL}': {e}. Falling back.")
        default_tokenizer_name = "bert-base-uncased"
        try:
             hf_tokenizer = AutoTokenizer.from_pretrained(default_tokenizer_name)
             logging.warning(f"Chunking: Using fallback tokenizer: {default_tokenizer_name}")
        except Exception as fallback_e:
             logging.error(f"Chunking: Failed to load fallback tokenizer '{default_tokenizer_name}': {fallback_e}")
             raise RuntimeError("Could not load any tokenizer.") from fallback_e

    max_len = getattr(hf_tokenizer, "model_max_length", 512)
    if not isinstance(max_len, int) or max_len <= 0 or max_len > 8192: max_len = 512
    safe_max_tokens = int(max_len * 0.9)
    tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer, max_tokens=safe_max_tokens)
    logging.info(f"Chunking: Using tokenizer '{hf_tokenizer.name_or_path}', max_tokens={safe_max_tokens}")

    # --- 2. Chunker ---
    chunker = HybridChunker(
        tokenizer=tokenizer, merge_peers=True, split_by_sentence=True, min_chunk_tokens=50
    )

    # --- 3. Chunking & Cleaning ---
    try:
        chunks = list(chunker.chunk(dl_doc))
        logging.info(f"Chunking: Initial chunk count: {len(chunks)}")
    except Exception as e:
        logging.error(f"Chunking: Error during initial chunking: {e}", exc_info=True)
        return []

    seen_content = set()
    unique_docs: List[Document] = []

    for i, ch in enumerate(chunks):
        content = getattr(ch, "text", "").strip()
        if not content: continue
        content_hash = hash(content)
        if content_hash in seen_content: continue
        seen_content.add(content_hash)

        page_num = getattr(ch.meta, "page", i + 1)
        heading_text = getattr(ch, "heading", None)
        headings_list = getattr(ch.meta, "headings", [])
        heading_lvl = getattr(ch, "heading_level", None)
        content_typ = ch.type.name.lower() if hasattr(ch, "type") else "text"

        total_chunks_count = len(chunks)
        if i < total_chunks_count * 0.2: section_pos = "start"
        elif i > total_chunks_count * 0.8: section_pos = "end"
        else: section_pos = "middle"

        # === TẠO METADATA (GẮN user_id) ===
        meta = {
            "subject": subject,
            "course_id": course_id,
            "user_id": user_id, # <-- GÁN user_id VÀO METADATA
            "chunk_id": f"{course_id}_{i}",
            "page": page_num,
            "heading": heading_text or "",
            "headings": headings_list if isinstance(headings_list, list) else [],
            "heading_level": heading_lvl,
            "section_position": section_pos,
            "content_type": content_typ,
            "doc_source": slugify_filename(file_path, course_id)
        }
        # ==================================

        cleaned_content = re.sub(r"\s+", " ", content).strip()
        unique_docs.append(Document(page_content=cleaned_content, metadata=meta))

    logging.info(f"Chunking: Produced {len(unique_docs)} unique text documents.")

    # --- 4. Thêm Code/Formula (GẮN user_id) ---
    base_meta_special = {
        "subject": subject, "course_id": course_id, "user_id": user_id, # <-- GÁN user_id
        "doc_source": os.path.basename(file_path),
    }

    code_count = 0
    for c in getattr(dl_doc, "codes", []):
        code_text = getattr(c, "text", "").strip()
        if code_text:
            unique_docs.append(Document(page_content=code_text, metadata={**base_meta_special, "content_type": "code"}))
            code_count += 1

    formula_count = 0
    for f in getattr(dl_doc, "formulas", []):
         formula_text = str(f).strip()
         if formula_text:
            unique_docs.append(Document(page_content=formula_text, metadata={**base_meta_special, "content_type": "formula"}))
            formula_count += 1

    logging.info(f"Chunking: Added {code_count} codes, {formula_count} formulas.")
    logging.info(f"Chunking: ✅ Total documents prepared: {len(unique_docs)}")
    return unique_docs

# --- Hàm convert PDF song song (Giữ nguyên) ---
# def fast_convert_pdf(file_path, opts, max_workers=4): ... (Code giữ nguyên)

# === CLASS PIPELINE CHÍNH ===
class DoclingPipeline:
    def __init__(self,
                 lang: str = "vi",
                 enable_ocr: bool = False, # Sẽ dùng call_ocr_service
                 enable_formula: bool = False,
                 enable_code: bool = False,
                 enable_table: bool = False,
                 fast_mode: bool = True):

        self.lang = lang
        self.enable_ocr = enable_ocr
        self.enable_formula = enable_formula
        self.enable_code = enable_code
        self.enable_table = enable_table
        self.fast_mode = fast_mode

        if fast_mode:
            self.enable_code = False
            self.enable_formula = False
            self.enable_table = False
            logging.info("DoclingPipeline initialized in FAST mode (Docling enrichment disabled).")
        else:
             logging.info("DoclingPipeline initialized in FULL mode.")


    def _select_pipeline(self, file_path: str, ext: str) -> DoclingDocument:
        """Chọn và chạy pipeline Docling phù hợp dựa trên loại file."""
        logging.info(f"Pipeline: Selecting Docling pipeline for: {file_path} (ext: {ext})")

        # --- PDF Pipeline ---
        if ext == ".pdf":
            os.environ["DOCLING_READING_ORDER_MODE"] = "simple"
            opts = PdfPipelineOptions()
            opts.do_code_enrichment = self.enable_code
            opts.do_table_structure = self.enable_table
            opts.do_formula_enrichment = self.enable_formula
            opts.do_ocr = False # Tắt OCR của Docling
            logging.info(f"Pipeline: Using PdfPipelineOptions: Code={opts.do_code_enrichment}, Table={opts.do_table_structure}, Formula={opts.do_formula_enrichment}")

            pdf_format_option = PdfFormatOption(backend=DoclingParseV2DocumentBackend, pipeline_options=opts)
            converter = DocumentConverter(format_options={InputFormat.PDF: pdf_format_option})

            try:
                logging.info("Pipeline: Running standard Docling conversion for PDF...")
                result = converter.convert(file_path)
                if not result or not result.document:
                     raise RuntimeError("Docling converter returned empty result.")
                return result.document
            except Exception as e:
                logging.error(f"Pipeline: Docling PDF conversion failed for {file_path}: {e}", exc_info=True)
                raise RuntimeError(f"Docling PDF processing failed: {e}") from e

        # --- Các định dạng khác ---
        elif ext in [".docx", ".pptx", ".html", ".htm"]:
            logging.info(f"Pipeline: Using SimplePipeline for {ext} file.")
            pag_opts = PaginatedPipelineOptions()
            # Cấu hình enrichment nếu SimplePipeline hỗ trợ

            input_format = InputFormat(ext.strip('.'))
            format_option = PdfFormatOption(pipeline_cls=SimplePipeline, pipeline_options=pag_opts) # Key vẫn là PDF?
            converter = DocumentConverter(format_options={input_format: format_option})

            try:
                logging.info(f"Pipeline: Running Docling conversion for {ext}...")
                result = converter.convert(file_path)
                if not result or not result.document:
                     raise RuntimeError("Docling converter returned empty result.")
                return result.document
            except Exception as e:
                logging.error(f"Pipeline: Docling {ext} conversion failed for {file_path}: {e}", exc_info=True)
                raise RuntimeError(f"Docling {ext} processing failed: {e}") from e

        # --- Loại file không hỗ trợ ---
        else:
            logging.error(f"Pipeline: Unsupported file type: {ext}")
            raise ValueError(f"❌ Unsupported file type: {ext}")



    # === HÀM XỬ LÝ CHÍNH (ĐÃ SỬA ĐỂ NHẬN user_id) ===
    @traceable(name="Process Single File")
    async def process_file(self, file_path: str, course_id: str = "default", subject: str = "general", user_id: str = "default_user"): # <-- NHẬN user_id
        """
        Xử lý hoàn chỉnh một file: Parse -> OCR (nếu cần) -> TOC -> Chunk (gắn user_id) -> Upsert.
        """
        ext = os.path.splitext(file_path)[1].lower()
        base_filename = os.path.basename(file_path)
        logging.info(f"🚀 Pipeline: Starting processing file: {base_filename}, User: {user_id}, Course: {course_id}")
        start_total_time = time.time()

        # 1. Parse bằng Docling
        t_parse_start = time.time()
        try:
            dl_doc: DoclingDocument = self._select_pipeline(file_path, ext)
        except Exception as e:
             logging.error(f"Pipeline: CRITICAL - Docling parsing failed for {base_filename}. Aborting. Error: {e}")
             raise RuntimeError(f"Docling parsing failed: {e}") from e
        t_parse_end = time.time()
        logging.info(f"   Pipeline: Docling parsing completed in {t_parse_end - t_parse_start:.2f}s")

        # 2. OCR Fallback (nếu là PDF ảnh và bật OCR)
        ocr_text = None
        t_ocr_start = time.time()
        if ext == ".pdf" and self.enable_ocr and not is_text_based_pdf(file_path):
            logging.info(f"   Pipeline: Image PDF detected. Attempting OCR fallback for {base_filename}...")
            try:
                loop = asyncio.get_event_loop()
                ocr_text = await loop.run_in_executor(None, call_ocr_service, file_path)
                logging.info(f"   Pipeline: ✅ OCR fallback successful for {base_filename}")
            except Exception as e:
                logging.warning(f"   Pipeline: ⚠️ OCR fallback failed for {base_filename}: {e}. Proceeding.")
        t_ocr_end = time.time()
        if ocr_text: logging.info(f"   Pipeline: OCR completed in {t_ocr_end - t_ocr_start:.2f}s")

        # 3. Trích xuất TOC (Mục lục)
        t_toc_start = time.time()
        toc_data = None # Khởi tạo
        toc_key = None
        try:
            toc_data = extract_toc_hybrid(file_path, ocr_text=ocr_text)
            if toc_data:
                logging.info(f"   Pipeline: TOC extracted ({len(toc_data)} items) for {base_filename}.")
                toc_key = update_toc_index_safe(file_path, course_id, subject, toc_data)
                logging.info(f"   Pipeline: TOC saved/updated for key: {toc_key}")
            else:
                logging.info(f"   Pipeline: No TOC found for {base_filename}.")
        except Exception as e:
            logging.warning(f"   Pipeline: ⚠️ Error extracting/saving TOC for {base_filename}: {e}")
        t_toc_end = time.time()
        logging.info(f"   Pipeline: TOC extraction completed in {t_toc_end - t_toc_start:.2f}s")

        # 4. Chunking (TRUYỀN user_id VÀO)
        t_chunk_start = time.time()
        docs: List[Document] = [] # Khởi tạo list rỗng
        try:
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(
                None,
                hybrid_chunk_docs,
                dl_doc, subject, course_id, file_path, user_id # <-- TRUYỀN user_id
            )
        except Exception as e:
            logging.error(f"Pipeline: CRITICAL - Chunking failed for {base_filename}. Aborting. Error: {e}", exc_info=True)
            raise RuntimeError(f"Chunking process failed: {e}") from e
        t_chunk_end = time.time()
        if not docs:
            logging.warning(f"   Pipeline: ⚠️ Chunking resulted in 0 documents for {base_filename}. Skipping upsert.")
            return {"course_id": course_id, "subject": subject, "chunks_indexed": 0, "enrichment_file": ""}
        logging.info(f"   Pipeline: Chunking completed: {len(docs)} docs in {t_chunk_end - t_chunk_start:.2f}s")

        # 5. Upsert vào Milvus (hàm này trong vectorstore.py sẽ tự embed)
        t_upsert_start = time.time()
        n_indexed = 0 # Khởi tạo
        try:
            # upsert_docs_wrapper gọi upsert_chunks từ vectorstore.py
            n_indexed = await upsert_docs_wrapper(docs)
        except Exception as e:
            logging.error(f"Pipeline: CRITICAL - Milvus upsert failed for {base_filename}. Error: {e}", exc_info=True)
            raise RuntimeError(f"Database upsert failed: {e}") from e
        t_upsert_end = time.time()
        logging.info(f"   Pipeline: Upsert completed: {n_indexed} docs in {t_upsert_end - t_upsert_start:.2f}s")

        # Dọn dẹp GPU cache
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logging.debug("   Pipeline: GPU memory cache cleared.")
            except Exception as gpu_e:
                 logging.warning(f"   Pipeline: Could not clear GPU cache: {gpu_e}")

        # Lưu enrichment (TOC)
        enrichment_file_path = ""
        try:
            enrichment_data = { "toc": toc_data if toc_data else [] }
            enrichment_filename = f"{slugify_filename(file_path, course_id)}.enrichment.json"
            enrichment_dir = "data/enrichments"
            os.makedirs(enrichment_dir, exist_ok=True)
            enrichment_file_path = os.path.join(enrichment_dir, enrichment_filename)
            with open(enrichment_file_path, "w", encoding="utf-8") as f:
                json.dump(enrichment_data, f, ensure_ascii=False, indent=2)
            logging.info(f"   Pipeline: Enrichment data saved to {enrichment_file_path}")
        except Exception as e:
            logging.warning(f"   Pipeline: Failed to save enrichment file for {base_filename}: {e}")
            enrichment_file_path = ""

        # Hoàn tất
        end_total_time = time.time()
        total_duration = end_total_time - start_total_time
        logging.info(f"🏁 Pipeline: Finished processing file {base_filename} in {total_duration:.2f}s")
        logging.info(f"   [PROFILE] Parse: {t_parse_end-t_parse_start:.2f}s | OCR: {t_ocr_end-t_ocr_start:.2f}s | TOC: {t_toc_end-t_toc_start:.2f}s | Chunk: {t_chunk_end-t_chunk_start:.2f}s | Upsert: {t_upsert_end-t_upsert_start:.2f}s")

        return {
            "course_id": course_id,
            "subject": subject,
            "chunks_indexed": n_indexed,
            "enrichment_file": enrichment_file_path
        }

    # ============================================
    # 🚀 Xử lý nhiều file song song (ĐÃ SỬA)
    # ============================================
    @traceable(name="Process Batch Files")
    async def process_many(self, files: list[str], course_id: str, subject: str, user_id: str): # <-- NHẬN user_id
        """
        Xử lý nhiều file song song bằng asyncio.gather.
        Truyền user_id vào từng task process_file.
        """
        logging.info(f"🚀 Pipeline Batch: Starting for {len(files)} files. User: {user_id}, Course: {course_id}")

        # TRUYỀN user_id vào từng task
        tasks = [
            self.process_file(f, course_id=course_id, subject=subject, user_id=user_id)
            for f in files
            ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        successful_count = 0
        failed_count = 0
        for file_path, result_or_exception in zip(files, results):
            base_filename = os.path.basename(file_path)
            if isinstance(result_or_exception, Exception):
                failed_count += 1
                error_message = str(result_or_exception)
                logging.error(f"❌ Pipeline Batch: Failed processing file '{base_filename}': {error_message}")
                processed_results.append({"file": base_filename, "chunks_indexed": 0, "error": error_message})
            elif isinstance(result_or_exception, dict):
                successful_count += 1
                logging.info(f"✅ Pipeline Batch: Success file '{base_filename}'. Chunks: {result_or_exception.get('chunks_indexed', 0)}")
                result_or_exception["file"] = base_filename # Thêm tên file vào kết quả
                processed_results.append(result_or_exception)
            else:
                 failed_count += 1
                 logging.error(f"❓ Pipeline Batch: Unknown result type for file '{base_filename}': {type(result_or_exception)}")
                 processed_results.append({"file": base_filename, "chunks_indexed": 0, "error": "Unknown result type"})

        logging.info(f"🏁 Pipeline Batch: Finished. Success: {successful_count}, Failed: {failed_count}.")
        return processed_results