import easyocr
import numpy as np
from PIL import Image
import logging

log = logging.getLogger("rag")

_reader = None

def get_reader():
    global _reader
    if _reader is None:
        try:
            log.info("[OCR] Initializing EasyOCR...")
            _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            log.info("[OCR] ✅ EasyOCR ready")
        except Exception as e:
            log.error(f"[OCR] ❌ Failed to initialize: {e}")
            raise
    return _reader

def extract_text_from_image(image) -> str:
    """
    Extract text from image using EasyOCR.
    """
    try:
        reader = get_reader()
        
        if isinstance(image, str):
            # File path
            results = reader.readtext(image, detail=0, paragraph=True)
        elif isinstance(image, Image.Image):
            # PIL Image
            img_array = np.array(image)
            results = reader.readtext(img_array, detail=0, paragraph=True)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        text = "\n".join(results)
        log.info(f"[OCR] Extracted {len(text)} characters, {len(results)} blocks")
        return text
        
    except Exception as e:
        log.error(f"[OCR] ❌ Failed: {e}")
        return ""