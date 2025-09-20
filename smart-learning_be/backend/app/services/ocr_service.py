from paddleocr import PaddleOCR

# Khởi tạo 1 lần (multilang có tiếng Việt)
ocr_model = PaddleOCR(use_angle_cls=True, lang="vi")  # hoặc "en" + "vi" nếu muốn hỗ trợ nhiều

def ocr_image_bytes(img_bytes: bytes) -> str:
    import cv2
    import numpy as np
    import io
    from PIL import Image

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = np.array(img)

    results = ocr_model.ocr(img)
    texts = []
    for line in results[0]:   # results là list, mỗi page có 1 list
        if isinstance(line[1], (list, tuple)):
            txt = line[1][0]
        else:
            txt = str(line[1])
        texts.append(txt.strip())
    return "\n".join(texts)
