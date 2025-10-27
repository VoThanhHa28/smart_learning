import fitz  # PyMuPDF

def is_text_based_pdf(file_path: str) -> bool:
    """Check PDF có text layer hay không"""
    with fitz.open(file_path) as doc:
        for page in doc:
            if page.get_text("text").strip():
                return True
    return False
