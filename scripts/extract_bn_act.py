# extract_bn_act.py
# Run: python extract_bn_act.py
# Needs: pip install pymupdf requests
import base64
import hashlib
import json
import re
import time
from pathlib import Path

import fitz       # PyMuPDF
import requests

# ---------------- CONFIG ----------------
PDF_PATH = Path(r"E:\TaxAssist\data_root\raw_pdfs\income_tax_act_2023_bn\Income_tax_act_2023.pdf")
OUT_DIR = Path(r"E:\TaxAssist\data_root\extracted_txt")
CACHE_DIR = OUT_DIR / "_ocr_cache_income_tax_act_2023_bn"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:12b"
DPI = 300
MIN_BN_RATIO = 0.5        # text layer-e Bangla char 50% er kom hole bad dhorbo
MIN_CHARS = 50            # er kom char hole bad dhorbo
FORCE_OCR = False         # True dile shob page-i OCR hobe
MAX_RETRIES = 3
REQUEST_TIMEOUT = 180     # sec
NUM_PREDICT = 3000        # output cap, repetition loop thamate

OCR_PROMPT = (
    "এই পৃষ্ঠার ছবি থেকে সব লেখা হুবহু বাংলা ইউনিকোডে লেখো। "
    "কোনো ব্যাখ্যা, অনুবাদ বা সারাংশ দেবে না। শুধু পৃষ্ঠায় যা লেখা আছে তাই দেবে। "
    "ধারা, উপধারা, দফা ও সংখ্যা যেভাবে আছে সেভাবেই রাখবে। "
    "টেবিল থাকলে সারি অনুযায়ী লেখো। পৃষ্ঠা নম্বর বা হেডার-ফুটার থাকলে সেটাও লেখো।"
)


# ---------------- CHECKS ----------------
def bn_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum("\u0980" <= c <= "\u09FF" for c in chars) / len(chars)


def looks_like_bijoy(text: str) -> bool:
    """Bijoy/SutonnyMJ legacy text Latin symbol diye bhora thake, Bangla block-e na."""
    chars = [c for c in text if not c.isspace()]
    if len(chars) < 30:
        return False
    odd = sum(c in "‡†…¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿×÷" for c in chars)
    return odd / len(chars) > 0.03 and bn_ratio(text) < 0.2


def is_bad_text(text: str) -> bool:
    if len(text.strip()) < MIN_CHARS:
        return True
    if "\ufffd" in text:
        return True
    if looks_like_bijoy(text):
        return True
    return bn_ratio(text) < MIN_BN_RATIO


def has_repetition_loop(text: str) -> bool:
    """Ek-i line 6+ bar bar ba ekta choto phrase onek bar repeat hole loop."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) >= 6:
        top = max(set(lines), key=lines.count)
        if lines.count(top) >= 6 and len(top) > 3:
            return True
    return bool(re.search(r"(.{4,40}?)\1{8,}", text))


# ---------------- OCR ----------------
def render_png(page: fitz.Page) -> bytes:
    return page.get_pixmap(dpi=DPI).tobytes("png")


def ocr_png(png: bytes) -> str:
    payload = {
        "model": MODEL,
        "prompt": OCR_PROMPT,
        "images": [base64.b64encode(png).decode()],
        "stream": False,
        "options": {"temperature": 0, "num_predict": NUM_PREDICT},
    }
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            text = r.json().get("response", "").strip()
            if has_repetition_loop(text):
                # loop hole prothom ordhek rekhe baki ta felo
                text = text[: len(text) // 2].rstrip()
                print("    warn: repetition loop detected, output trimmed")
            return text
        except Exception as e:
            last_err = e
            print(f"    OCR attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"OCR failed after {MAX_RETRIES} attempts: {last_err}")


def ocr_with_cache(png: bytes, page_no: int) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(png).hexdigest()[:16]
    f = CACHE_DIR / f"p{page_no:04d}_{key}.txt"
    if f.exists():
        return f.read_text(encoding="utf-8")
    text = ocr_png(png)
    f.write_text(text, encoding="utf-8")
    return text


# ---------------- MAIN ----------------
def extract_pdf(pdf_path: Path):
    doc = fitz.open(pdf_path)
    total = len(doc)
    pages = []
    n_text = n_ocr = n_empty = 0

    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        source = "text"

        if FORCE_OCR or is_bad_text(text):
            t0 = time.time()
            try:
                text = ocr_with_cache(render_png(page), i)
                source = "ocr"
                n_ocr += 1
                print(f"page {i}/{total}: OCR {len(text)} chars, {time.time() - t0:.0f}s")
            except Exception as e:
                text, source = "", "failed"
                n_empty += 1
                print(f"page {i}/{total}: FAILED ({e})")
        else:
            n_text += 1
            if i % 25 == 0:
                print(f"page {i}/{total}: text layer ok")

        pages.append({
            "page": i,
            "source": source,
            "bn_ratio": round(bn_ratio(text), 3),
            "n_chars": len(text),
            "text": text,
        })

    print(f"\nDone: {total} pages | text layer: {n_text} | OCR: {n_ocr} | failed: {n_empty}")
    return pages


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = extract_pdf(PDF_PATH)

    stem = PDF_PATH.stem
    (OUT_DIR / f"{stem}_pages.json").write_text(
        json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    with open(OUT_DIR / f"{stem}.txt", "w", encoding="utf-8") as f:
        f.write(f"# source: {PDF_PATH.name}\n# language: bn\n\n")
        for p in pages:
            if p["text"]:
                f.write(f"\n[[PAGE {p['page']}]]\n{p['text']}\n")

    # sanity check
    low = [p["page"] for p in pages if p["n_chars"] > 0 and p["bn_ratio"] < 0.5]
    empty = [p["page"] for p in pages if p["n_chars"] == 0]
    print("Low Bangla ratio pages:", low[:30])
    print("Empty pages:", empty[:30])
    print("Output:", OUT_DIR)


if __name__ == "__main__":
    main()