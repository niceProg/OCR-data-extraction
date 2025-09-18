from typing import Dict, Any, Tuple
from PIL import Image, ImageOps, ImageFilter
import io
import pytesseract


def preprocess_image_bytes(image_bytes: bytes) -> bytes:
    """
    Basic preprocessing to improve OCR:
    - Open via PIL, convert to grayscale, auto-contrast, optional sharpening.
    Returns PNG bytes (RGB) ready for pytesseract.
    """
    img = Image.open(io.BytesIO(image_bytes))
    # Ensure RGB or L
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    # convert to grayscale for many OCR tasks
    gray = ImageOps.grayscale(img)
    # auto-contrast
    gray = ImageOps.autocontrast(gray, cutoff=1)
    # slight sharpen
    gray = gray.filter(ImageFilter.SHARPEN)
    buf = io.BytesIO()
    gray.save(buf, format="PNG")
    return buf.getvalue()


def run_tesseract_ocr(
    image_bytes: bytes, lang: str | None = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Run pytesseract OCR on given image bytes.
    Returns (extracted_text, meta)
    """
    # Preprocess
    try:
        pre_bytes = preprocess_image_bytes(image_bytes)
        img = Image.open(io.BytesIO(pre_bytes))
    except Exception:
        # fallback: try opening original
        img = Image.open(io.BytesIO(image_bytes))

    # Full text
    text = (
        pytesseract.image_to_string(img, lang=lang)
        if lang
        else pytesseract.image_to_string(img)
    )

    # Detailed data for confidence and words
    meta: Dict[str, Any] = {}
    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        confs = []
        for c in data.get("conf", []):
            try:
                ci = float(c)
                if ci >= 0:
                    confs.append(ci)
            except Exception:
                pass
        avg_conf = (sum(confs) / len(confs)) if confs else None
        meta["avg_confidence"] = avg_conf
        meta["word_count"] = len([w for w in data.get("text", []) if w and w.strip()])
    except Exception:
        meta["avg_confidence"] = None
        meta["word_count"] = len(text.split())

    return text, meta
