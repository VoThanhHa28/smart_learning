# sentence_splitter.py
import nltk
from typing import List

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab")

def split_sentences(text: str, lang: str = "english") -> List[str]:
    """
    Tách text thành câu dùng NLTK (ngữ nghĩa).
    """
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text, language=lang)
