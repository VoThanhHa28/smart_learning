# services/docling_pipeline.py
import os
import json
import asyncio
from typing import List, Dict, Any

from docling.document_converter import DocumentConverter
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.pipeline.simple_pipeline import SimplePipeline
from docling_core.types import DoclingDocument
from docling.chunking import HybridChunker   # ✅ dùng Docling HybridChunker
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend

from docling.datamodel.pipeline_options import PdfPipelineOptions, PaginatedPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from .utils.common_utils import slugify_filename


from .utils.pdf_utils import is_text_based_pdf
from .utils.docling_utils import wrap_ocr_text_as_docling
from .utils.text_cleaner import clean_text, remove_headers_footers   # ✅ thay vì process_text_chunks
from app.services.utils.toc_regex import extract_toc, update_toc_index

from langchain_core.documents import Document
from app.services.embedding import get_embeddings, async_embed_docs
from .vectorstore import upsert_chunks
from langsmith import traceable
import requests
import time
import logging
logging.getLogger("docling").setLevel(logging.ERROR)
import re
import torch
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

import asyncio

logging.getLogger("docling").setLevel(logging.ERROR)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()


app_state = {"tagger_running": False}

@traceable(name="OCR PDF")

def call_ocr_service(file_path: str) -> str:
    with open(file_path, "rb") as f:
        res = requests.post("http://ocr:9000/ocr", files={"file": f})
    res.raise_for_status()
    return res.json()["text"]


@traceable(name="Embedding")
def embed_docs(docs):
    embeddings = get_embeddings()
    vectors = embeddings.embed_documents([d.page_content for d in docs])
    return vectors


@traceable(name="Milvus Upsert")
async def upsert_docs(docs):
    return await upsert_chunks(docs)

@traceable(name="Chunking")
def hybrid_chunk_docs(dl_doc: DoclingDocument, subject: str, course_id: str, file_path: str):
    """
    ✅ Chunk nâng cao:
    - Giữ heading + ngữ nghĩa logic
    - Gắn metadata chặt chẽ
    - Chuẩn hóa cho embedding & retrieval
    """
    from langchain_core.documents import Document

    # --- 1️⃣ Khởi tokenizer đồng bộ với model embedding ---
    EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
    hf_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)

    # ✅ Xác định max_tokens an toàn (ưu tiên model config, fallback 512)
    max_len = getattr(hf_tokenizer, "model_max_length", None)
    if not max_len or max_len > 100_000 or max_len == -1:
        max_len = 384  # fallback mặc định

    tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer, max_tokens=max_len)

    logging.info(f"[Tokenizer] Using {EMBED_MODEL}, max_tokens={max_len}")
    
    # --- 2️⃣ Cấu hình chunker ---
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,   # hợp nhất các đoạn nhỏ liền kề
    )

    # --- 3️⃣ Tạo Document với text + metadata enrich ---
    chunks = list(chunker.chunk(dl_doc))  # ✅ ép iterator → list
    
    seen = set()
    unique_chunks = []
    for ch in chunks:
        text = ch.text.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique_chunks.append(ch)

    chunks = unique_chunks
    print(f"🧹 Dedup: {len(chunks)} unique chunks retained.")

    total_chunks = len(chunks)
    docs = []

    for i, ch in enumerate(chunks):
        # --- Xác định vị trí (section_position) ---
        if i < total_chunks * 0.2:
            section_pos = "start"
        elif i > total_chunks * 0.8:
            section_pos = "end"
        else:
            section_pos = "middle"
        

        # --- Lấy thông tin heading ---
        heading_level = getattr(ch, "heading_level", None)
        heading_text = getattr(ch, "heading", None)
        headings = getattr(ch.meta, "headings", None)

        # --- Xác định loại nội dung ---
        content_type = ch.type.name.lower() if hasattr(ch, "type") else "text"

        # --- Tag heuristic tạm ---
        txt = ch.text.lower()
        tags = []
        if "ví dụ" in txt or "example" in txt: tags.append("example")
        if "định nghĩa" in txt or "definition" in txt: tags.append("definition")
        if "bài tập" in txt or "exercise" in txt: tags.append("problem")

        

        # --- Metadata chính ---
                # --- Metadata chính (đơn giản, không semantic tag) ---
        meta = {
            "subject": subject,
            "course_id": course_id,
            "chunk_id": f"{course_id}_{i}",
            "page": getattr(ch.meta, "page", None),
            "heading": heading_text,
            "headings": headings,
            "heading_level": heading_level,
            "section_position": section_pos,
            "content_type": content_type,
            "vector_scope_id": f"{course_id}_{heading_text or i}",
            "doc_source": slugify_filename(file_path, course_id)
        }

        # --- Làm sạch text ---
        ctx_text = re.sub(r"\s+", " ", getattr(ch, "text", "").strip())

        docs.append(Document(page_content=ctx_text, metadata=meta))

    logging.info(f"🧩 Chunking semantic-aware xong {len(docs)} đoạn.")

    # ✅ Thêm enrichment code & formula sau cùng (không lặp)
    for c in getattr(dl_doc, "codes", []):
        docs.append(Document(
            page_content=c.text.strip(),
            metadata={
                "subject": subject,
                "course_id": course_id,
                "content_type": "code",
                "doc_source": os.path.basename(file_path),
            },
        ))

    for f in getattr(dl_doc, "formulas", []):
        docs.append(Document(
            page_content=str(f),
            metadata={
                "subject": subject,
                "course_id": course_id,
                "content_type": "formula",
                "doc_source": os.path.basename(file_path),
            },
        ))

    logging.info(f"🧩 Tổng cộng {len(docs)} đoạn (gồm text + code + formula).")
    return docs




# ==========================================================
# ⚡ Fast Docling converter song song theo cụm trang
# ==========================================================
from concurrent.futures import ThreadPoolExecutor
import math

import math
from concurrent.futures import ThreadPoolExecutor
import fitz  # import PyMuPDF

def fast_convert_pdf(file_path, opts, max_workers=4):
    # opts.output_dir = None
    # opts.save_intermediate = False
    os.environ["DOCLING_SAVE_TMP"] = "false"
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(
            backend=DoclingParseV2DocumentBackend,
            pipeline_options=opts
        )}
    )

    # Dò số trang bằng PyMuPDF
    pdf = fitz.open(file_path)
    n_pages = pdf.page_count
    pdf.close()

    step = math.ceil(n_pages / max_workers)
    def _convert_chunk(start_page, end_page):
        # page_range param là tuple (start, end)
        result = converter.convert(file_path, page_range=(start_page + 1, end_page + 1))
        return result.document

    docs = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for i in range(0, n_pages, step):
            start = i
            end = min(i + step - 1, n_pages - 1)
            futures.append(ex.submit(_convert_chunk, start, end))
        for f in futures:
            docs.append(f.result())

    # Merge elements
    merged = docs[0]
    for d in docs[1:]:
        merged.content.elements.extend(d.content.elements)
    return merged



class DoclingPipeline:
    def __init__(self,
                 lang: str = "vi",
                 enable_ocr: bool = False,
                 enable_formula: bool = False,
                 enable_code: bool = False,
                 enable_table: bool = False,
                 fast_mode: bool = True):

        self.lang = lang
        self.enable_ocr = enable_ocr
        self.enable_formula = enable_formula   # ✅ thêm dòng này
        self.enable_code = enable_code         # ✅ thêm dòng này
        self.enable_table = enable_table       # ✅ thêm dòng này
        self.fast_mode = fast_mode

        if fast_mode:
            self.enable_code = False
            self.enable_formula = False
            self.enable_table = False


    def _select_pipeline(self, file_path: str, ext: str) -> DoclingDocument:
        if ext == ".pdf":
            # ⚙️ Bật chế độ đọc bố cục đơn giản để tránh lỗi KeyError
            os.environ["DOCLING_READING_ORDER_MODE"] = "simple"
            opts = PdfPipelineOptions()
            # Cấu hình enrichment phù hợp với fast_mode
            opts.do_code_enrichment = False if self.fast_mode else self.enable_code
            opts.do_table_structure = False if self.fast_mode else self.enable_table
            opts.do_formula_enrichment = False if self.fast_mode else self.enable_formula
            opts.do_ocr = False  # OCR riêng rồi


            # ✅ Dùng StandardPdfPipeline nhưng ở chế độ "nhẹ" (lite)
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        backend=DoclingParseV2DocumentBackend,
                        pipeline_options=opts
                    )
                }
            )

            # Nếu fast_mode: convert trực tiếp (đơn giản, không song song)
            if self.fast_mode:
                return converter.convert(file_path).document
            else:
                # Dạng đầy đủ: convert song song nhiều cụm trang
                return fast_convert_pdf(file_path, opts, max_workers=4)

        elif ext in [".docx", ".pptx", ".html", ".htm"]:
            pag_opts = PaginatedPipelineOptions()
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=SimplePipeline,
                        pipeline_options=pag_opts
                    )
                }
            )
            return converter.convert(file_path).document

        else:
            raise ValueError(f"❌ Unsupported file type: {ext}")


    

    async def process_file(self, file_path: str, course_id: str = "default", subject: str = "general"):
        ext = os.path.splitext(file_path)[1].lower()
        logging.info(f"📂 Bắt đầu xử lý file: {file_path}, subject={subject}, course={course_id}")

        t0 = time.time()
        dl_doc: DoclingDocument = self._select_pipeline(file_path, ext)
        logging.info(f"✅ Parse xong Docling trong {time.time()-t0:.2f}s")

        # 🧭 Detect TOC
        from .utils.pdf_utils import is_text_based_pdf

        if is_text_based_pdf(file_path):
            toc_data = extract_toc(file_path)
        else:
            # Nếu scan hoặc Docling không có text, fallback OCR
            try:
                ocr_text = call_ocr_service(file_path)
                toc_data = extract_toc(file_path, ocr_text=ocr_text)
            except Exception as e:
                logging.warning(f"⚠️ OCR fallback failed: {e}")
                toc_data = []
        
        logging.info(f"📖 First TOC preview: {toc_data[:5]}")

        toc_key = update_toc_index(file_path, course_id, subject, toc_data)
        logging.info(f"📚 TOC extracted for {toc_key} ({len(toc_data)} mục)")

        # Enrichment
        t1 = time.time()
        enrichments = {
            "formulas": [str(f) for f in getattr(dl_doc, "formulas", [])],
            "codes": [c.text for c in getattr(dl_doc, "codes", [])],
            "tables": [t.dict() for t in getattr(dl_doc, "tables", [])]
        }
        logging.info(f"📑 Extract xong enrichments trong {time.time()-t1:.2f}s")

        # Chunk
        t2 = time.time()
        docs = hybrid_chunk_docs(dl_doc, subject=subject, course_id=course_id, file_path=file_path)
        logging.info(f"🔪 Chunk xong {len(docs)} docs trong {time.time()-t2:.2f}s")

        # Embed
        t3 = time.time()
        embeddings = get_embeddings()
        try:
            vectors = await async_embed_docs(embeddings, [d.page_content for d in docs])
        except Exception as e:
            logging.error(f"❌ Embedding failed: {e}")
            vectors = [[0.0] * 768 for _ in docs]

        logging.info(f"⚡ Embed xong {len(docs)} docs trong {time.time()-t3:.2f}s")

        # Gắn embedding vào từng Document
        for doc, vec in zip(docs, vectors):
            doc.metadata["embedding"] = vec  # hoặc numpy → list để serialize
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logging.info("🧹 Đã dọn dẹp bộ nhớ GPU.")
        
        # Upsert
        t4 = time.time()
        n_indexed = await upsert_docs(docs)
        

        # ✅ Lưu enrichment ra file JSON để tra cứu TOC / code / formula sau
        enrichment_data = {
            "toc": toc_data,
            "formulas": [],
            "codes": [],
            "tables": [],
        }
        save_path = f"{file_path}.enrichment.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(enrichment_data, f, ensure_ascii=False, indent=2)
        logging.info(f"💾 Lưu xong enrichment tại {save_path}")
     
    
        logging.info(f"📥 Upsert xong {n_indexed} docs trong {time.time()-t4:.2f}s")

        total_time = time.time()-t0
        logging.info(f"🏁 Hoàn tất xử lý file {file_path} trong {total_time:.2f}s")
        logging.info(f"[PROFILE] parse={time.time()-t0:.2f}s | enrich={time.time()-t1:.2f}s | chunk={time.time()-t2:.2f}s | embed={time.time()-t3:.2f}s | upsert={time.time()-t4:.2f}s | total={time.time()-t0:.2f}s")

        return {
            "course_id": course_id,
            "subject": subject,
            "chunks_indexed": n_indexed,
            "enrichment_file": f"{file_path}.enrichment.json"
        }

    # ============================================
    # 🚀 Xử lý nhiều file song song (tối ưu async)
    # ============================================
    async def process_many(self, files: list[str], course_id: str, subject: str):
        """
        ✅ Chạy song song nhiều file cùng lúc bằng asyncio.gather.
        - Mỗi file gọi process_file riêng (có upsert async bên trong)
        - Tự chia tải GPU/CPU hợp lý
        - Trả về list kết quả cho từng file
        """
        tasks = [self.process_file(f, course_id=course_id, subject=subject) for f in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Ghi log lỗi rõ ràng nếu file nào fail
        clean_results = []
        for f, r in zip(files, results):
            if isinstance(r, Exception):
                print(f"❌ [DoclingPipeline] Lỗi xử lý file {f}: {r}")
                clean_results.append({"file": f, "chunks_indexed": 0, "error": str(r)})
            else:
                clean_results.append(r)

        print(f"✅ Đã xử lý xong {len(clean_results)} file (song song).")
        return clean_results

import asyncio

async def async_embed_docs(embeddings, texts, batch_size=64):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embeddings.embed_documents, texts)