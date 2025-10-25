import re
from cleantext import clean

def clean_extracted_text(text: str) -> str:
    """Clean text using clean-text + regex"""
    text = clean(
        text,
        fix_unicode=True,
        to_ascii=False,
        lower=False,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_line_breaks=False,
    )
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)  # remove page numbers
    text = text.replace("\u200b", "").replace("�", "")
    return text.strip()
