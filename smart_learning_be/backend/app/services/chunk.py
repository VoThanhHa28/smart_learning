from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .sentence_splitter import split_sentences

def split_to_docs(chunks, subject, course_id=None, chunk_size=1000, overlap=120):
    """
    chunks: list[dict] từ process_text_chunks
        mỗi dict có: page, text, is_text, is_code?, is_formula?
    """
    docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        page_num = chunk.get("page")

        if chunk.get("is_code") or chunk.get("is_formula"):
            # --- Giữ nguyên block ---
            meta = {
                "page": page_num,
                "subject": subject,
                "chunk_id": f"{page_num}_{i}",
                "is_text": False,
                "is_code": chunk.get("is_code", False),
                "is_formula": chunk.get("is_formula", False),
            }
            if course_id:
                meta["course_id"] = course_id
            docs.append(Document(page_content=text, metadata=meta))

        else:
            # --- Text: NLP split + Recursive ---
            sentences = split_sentences(text)
            for j, sent in enumerate(sentences):
                for k, sub_chunk in enumerate(splitter.split_text(sent)):
                    meta = {
                        "page": page_num,
                        "subject": subject,
                        "doc_id": f"{page_num}_{i}_{j}_{k}",
                        "is_text": True,
                        "is_code": False,
                        "is_formula": False,
                    }
                    if course_id:
                        meta["course_id"] = course_id
                    docs.append(Document(page_content=sub_chunk, metadata=meta))


    return docs
