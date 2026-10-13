"""
Bangladesh Income Tax RAG - preprocessing v5
Follows the paper's "Data Preprocessing" section; retrieval stays plain (simple) RAG.
OCR engine = Ollama gemma3:12b (instead of the paper's Tesseract).

Pipeline
  1. download PDFs                                   (stage: download)
  2. per PDF -> text layer, or OCR for scanned / legacy-font pages  (stage: extract)
       - page rendered at 300 DPI with pypdfium2
       - Bangla-oriented image cleanup: median filter, CLAHE, normalization,
         adaptive thresholding, small morphological operations
       - OCR through Ollama
       - clean: Unicode NFC, repeated headers/footers, page numbers, repeated artifacts
       - saved as UTF-8 .txt with a metadata header  -> data_root/data/text/<category>/<doc>.txt
         (you may open and hand-edit these files like the paper did; existing .txt are NOT overwritten)
  3. .txt -> sentence-based chunks (Latin + Bangla delimiters) with controlled overlap
       + metadata for traceability                   -> data_root/data/legal/*.json   (stage: chunk)
  4. build_index.py -> bge-m3 embeddings -> ChromaDB

Install:
    pip install pymupdf pypdfium2 opencv-python-headless numpy pillow requests beautifulsoup4

Run:
    python -u "e:\\TaxAssist\\scripts\\tax_rag_preprocess_v5.py"
    python -u "...\\tax_rag_preprocess_v5.py" --only chunk          # re-chunk after editing .txt files
    python -u "...\\tax_rag_preprocess_v5.py" --skip-download       # PDFs already downloaded
    python -u "...\\tax_rag_preprocess_v5.py" --force-extract       # rebuild .txt (overwrites manual edits)
"""
import os, re, io, sys, json, time, zlib, bisect, base64, shutil, hashlib, argparse, unicodedata
from datetime import datetime
from collections import Counter
from urllib.parse import urljoin, unquote

import requests
import fitz                      # PyMuPDF: text layer
import pypdfium2 as pdfium       # page rendering for OCR (as in the paper)
import numpy as np
import cv2
from bs4 import BeautifulSoup
from PIL import Image

# Windows console can't print Bangla by default -> force utf-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
# script is in E:\TaxAssist\scripts -> data goes to E:\TaxAssist\data_root
ROOT = os.environ.get("TAX_RAG_ROOT") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_root")

RAW = os.path.join(ROOT, "raw_pdfs")
TEXT_DIR = os.path.join(ROOT, "data", "text")
OUT = os.path.join(ROOT, "data", "legal")
REMOVED = os.path.join(RAW, "_removed")
OCR_CACHE = os.path.join(RAW, "_ocr_cache")
DEBUG_DIR = os.path.join(RAW, "_ocr_debug")
BASE = "https://nbr.gov.bd/"
HDR = {"User-Agent": "Mozilla/5.0"}

# ---- OCR through Ollama ----
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:12b"
OCR_TIMEOUT = 900
OCR_MAX_TOKENS = 1200      # per band. A blank/looping band used to run to 2048 tokens (~35 s wasted)
OCR_PROMPT = (
    "You are an OCR engine. Transcribe ALL text in this image exactly as written. "
    "The text is Bangla and/or English. Keep every number, date, section number and symbol exactly. "
    "Keep line breaks. Write tables row by row with ' | ' between cells. "
    "Do NOT translate, summarize, explain or add any comment. Output only the transcribed text. "
    "If the image has no text, output nothing."
)

# ---- page image pipeline (paper: 300 DPI + Bangla cleanup) ----
OCR_DPI = 300
OCR_BANDS = 2              # Gemma 3 squeezes every image to ~896x896, so keep bands roughly square
IMG_PREPROCESS = True      # False = send the plain rendered page (test both on a few pages!)
MEDIAN_K = 3               # median filter kernel (odd)
CLAHE_CLIP = 2.0
CLAHE_TILE = 8
THRESH_BLOCK = 35          # adaptive threshold neighbourhood (odd)
THRESH_C = 11
MORPH_K = 2                # small morphological kernel
DEBUG_SAVE_PAGES = 3       # save the first N preprocessed pages of each scanned PDF to _ocr_debug

# ---- chunking (sentence-based, controlled overlap) ----
CHUNK_MAX_CHARS = 1000
CHUNK_OVERLAP_CHARS = 200
OVERLAP_MAX_SENTENCES = 2
SECTION_FLUSH_MIN = 250        # start a new chunk at a section heading once the current chunk has this many chars
DROP_FOREIGN_TOKENS = True     # OCR noise: drop words containing Greek/Cyrillic/Hebrew/Arabic/Devanagari/CJK letters

OCR_CACHE_TAG = (f"p{OCR_DPI}_{'prep' if IMG_PREPROCESS else 'raw'}_b{OCR_BANDS}_"
                 f"{OLLAMA_MODEL.replace(':', '-')}")

FOLDERS = ["income_tax_act_2023_en", "income_tax_act_2023_bn", "finance_act_amendments",
           "income_tax_rules", "relevant_sros", "nbr_guidelines"]
for d in FOLDERS:
    os.makedirs(os.path.join(RAW, d), exist_ok=True)
    os.makedirs(os.path.join(TEXT_DIR, d), exist_ok=True)
for d in [OUT, REMOVED, OCR_CACHE, DEBUG_DIR,
          os.path.join(ROOT, "data", "investment"), os.path.join(ROOT, "data", "evaluation")]:
    os.makedirs(d, exist_ok=True)

# direct PDF links -> (folder, saved filename, language)
DIRECT = {
    "https://nbr.gov.bd/uploads/acts/Income_Tax_act.pdf":                 # Bangla gazette (scanned / legacy font -> OCR)
        ("income_tax_act_2023_bn", "Income_tax_act_2023.pdf", "bn"),
    "https://nbr.gov.bd/uploads/acts/Income_tax_act_2023.pdf":            # Authentic English text (published Oct 2025)
        ("income_tax_act_2023_en", "Income_Tax_Act_2023_English.pdf", "en"),
    "https://nbr.gov.bd/uploads/acts/finance_act-23.pdf":
        ("finance_act_amendments", "Finance_Act_2023.pdf", "en"),
    "https://nbr.gov.bd/uploads/acts/Finance_Act-2024.pdf":
        ("finance_act_amendments", "Finance_Act_2024.pdf", "en"),
    "https://nbr.gov.bd/uploads/acts/5722-Ordinance-28-Law-02_June.pdf":
        ("finance_act_amendments", "Ordinance_28_2025_02_June.pdf", "en"),
    "https://nbr.gov.bd/uploads/acts/5936-Ordinance-33-Law-26_June_2025.pdf":
        ("finance_act_amendments", "Finance_Amendment_2025_Ordinance_33.pdf", "en"),
    "https://nbr.gov.bd/uploads/acts/Finance_Act_2026.pdf":
        ("finance_act_amendments", "Finance_Act_2026.pdf", "en"),
    "https://nbr.gov.bd/uploads/rules/With_holding_2026.pdf":
        ("income_tax_rules", "Withholding_Tax_Rules_2026.pdf", "en"),
}

# listing pages -> scrape PDF links -> (folder, filename prefix, language)
PAGES = {
    "https://nbr.gov.bd/regulations/sros/income-tax-sros/160/": ("relevant_sros", "SRO", "bn"),
    "https://nbr.gov.bd/regulations/rules/income-tax-rules/ban": ("income_tax_rules", "Rules_bn", "bn"),
    "https://nbr.gov.bd/taxtypes/income-tax/income-tax-paripatra/ban": ("nbr_guidelines", "Paripatra", "bn"),
}

# output json  <-  raw folder
OUTPUT_MAP = {
    "income_tax_act_2023_en.json": "income_tax_act_2023_en",
    "income_tax_act_2023_bn.json": "income_tax_act_2023_bn",
    "finance_act_amendments.json": "finance_act_amendments",
    "income_tax_rules.json": "income_tax_rules",
    "relevant_sros.json": "relevant_sros",
    "nbr_guidelines.json": "nbr_guidelines",
}

JUNK_LINK_PARTS = ("public-notice", "Final_Report_for_Print")
CLEAN_STALE = True   # move stale scraped PDFs (from earlier runs) into raw_pdfs/_removed


# ----------------------------------------------------------------------------
# DOWNLOAD
# ----------------------------------------------------------------------------
def safe_name(s, max_len=120):
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", unquote(s)).strip(" .")
    if not s.lower().endswith(".pdf"):
        s += ".pdf"
    if len(s) > max_len:
        s = hashlib.md5(s.encode("utf-8")).hexdigest()[:10] + "_" + s[-(max_len - 12):]
    return s


def fetch(url, path, retries=3):
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return True
    for i in range(retries):
        try:
            r = requests.get(url, headers=HDR, timeout=90, verify=False)  # NBR cert is sometimes broken
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                with open(path, "wb") as f:
                    f.write(r.content)
                return True
            print("  not a PDF / status", r.status_code, url)
            if r.status_code == 200:      # got an HTML page, retrying won't help
                return False
        except Exception as e:
            print("  retry", i + 1, e)
        time.sleep(2)
    return False


def scrape_pdf_links(page):
    try:
        html = requests.get(page, headers=HDR, timeout=60, verify=False).text
    except Exception as e:
        print("FAIL page", page, e)
        return set()
    soup = BeautifulSoup(html, "html.parser")
    return {
        urljoin(BASE, a["href"].strip())
        for a in soup.find_all("a", href=True)
        if a["href"].lower().strip().endswith(".pdf")
        and not any(j in a["href"] for j in JUNK_LINK_PARTS)
    }


def download_all():
    requests.packages.urllib3.disable_warnings()
    meta = {}

    for url, (folder, name, lang) in DIRECT.items():
        p = os.path.join(RAW, folder, name)
        ok = fetch(url, p)
        print("OK  " if ok else "FAIL", name)
        if ok:
            meta[p] = {"url": url, "lang": lang}

    # links that appear on 2+ listing pages are NBR sidebar/footer junk
    page_links = {page: scrape_pdf_links(page) for page in PAGES}
    count = Counter(u for s in page_links.values() for u in s)
    sitewide = {u for u, c in count.items() if c > 1}
    print("sitewide links skipped:", len(sitewide))

    for page, (folder, prefix, lang) in PAGES.items():
        links = sorted(page_links[page] - sitewide - set(DIRECT))
        print(f"{page} -> {len(links)} pdf links (after filtering)")
        wanted = set()
        for u in links:
            name = f"{prefix}_{safe_name(os.path.basename(u))}"
            p = os.path.join(RAW, folder, name)
            wanted.add(name)
            if fetch(u, p):
                meta[p] = {"url": u, "lang": lang}

        if CLEAN_STALE:
            d = os.path.join(RAW, folder)
            for fn in os.listdir(d):
                if fn.startswith(prefix + "_") and fn not in wanted:
                    dest = os.path.join(REMOVED, f"{folder}__{fn}")
                    if os.path.exists(dest):
                        os.remove(dest)
                    shutil.move(os.path.join(d, fn), dest)
                    print("  moved stale ->", fn)

    with open(os.path.join(RAW, "_sources.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return meta


def load_sources():
    p = os.path.join(RAW, "_sources.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ----------------------------------------------------------------------------
# OCR: page image pipeline + Ollama
# ----------------------------------------------------------------------------
_http = requests.Session()
_http.trust_env = False          # ignore system proxy for localhost
HAVE_OCR = False


def check_ollama():
    try:
        r = _http.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        names = [m.get("name", "") for m in r.json().get("models", [])]
        if any(n == OLLAMA_MODEL or n.split(":")[0] == OLLAMA_MODEL for n in names):
            print(f"Ollama OK, model {OLLAMA_MODEL} found -> OCR enabled")
            return True
        print(f"[warn] Ollama is running but model '{OLLAMA_MODEL}' not found. Installed: {names}")
        print(f"       run:  ollama pull {OLLAMA_MODEL}")
    except Exception as e:
        print("[warn] Cannot reach Ollama at", OLLAMA_URL, "->", e)
        print("       Start the Ollama app (or `ollama serve`). OCR is disabled for this run.")
    return False


def render_page(pdf, idx):
    """300 DPI render with pypdfium2 -> PIL RGB"""
    bitmap = pdf[idx].render(scale=OCR_DPI / 72.0)
    return bitmap.to_pil().convert("RGB")


def preprocess_for_bangla(img):
    """median filter -> CLAHE -> normalization -> adaptive threshold -> small morphology"""
    g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    g = cv2.medianBlur(g, MEDIAN_K)
    g = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE)).apply(g)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    g = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                              THRESH_BLOCK, THRESH_C)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_K, MORPH_K))
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, k)    # black text on white: closes tiny white gaps inside glyph strokes
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, k)   # removes tiny black specks (noise)
    return Image.fromarray(g).convert("RGB")


def split_bands(img, n):
    """cut image into n horizontal bands; cut lines are placed in white gaps between text lines"""
    w, h = img.size
    if n <= 1 or h < 600:
        return [img]
    rows = list(img.convert("L").resize((1, h), Image.BOX).getdata())   # mean brightness per row
    cuts = [0]
    win = int(h * 0.06)
    for k in range(1, n):
        target = int(h * k / n)
        lo = max(cuts[-1] + 1, target - win)
        hi = min(h - 1, target + win)
        cuts.append(max(range(lo, hi + 1), key=lambda y: rows[y]))
    cuts.append(h)
    return [img.crop((0, a, w, b)) for a, b in zip(cuts, cuts[1:]) if b - a > 20]


def collapse_repeats(text, max_repeat=2):
    """small models sometimes loop on one line -> keep at most max_repeat consecutive copies"""
    out, prev, run = [], None, 0
    for line in text.split("\n"):
        if line.strip() and line == prev:
            run += 1
            if run >= max_repeat:
                continue
        else:
            run = 0
        out.append(line)
        prev = line
    return "\n".join(out)


CHATTER_RX = re.compile(r"^\s*(here is|here's|here are|the (text|image)|sure|okay|transcription|```)", re.I)

# short "there is no text / I cannot read this" answers the model gives for blank or picture-only pages
NO_TEXT_RX = re.compile(r"^\W*(there (is|are) no|no (visible |readable |discernible )?text|"
                        r"(the|this) image (is|contains|does|shows|appears)|i (cannot|can't|am unable|'m unable)|"
                        r"sorry|unable to|blank|empty|\[?no text)", re.I)
OCR_FLAGS = []   # (file, page, flag) collected during a run, also written to raw_pdfs/_ocr_flags.log


def truncate_loops(text, min_len=15, max_seen=3):
    """the model sometimes repeats a few lines until the token limit -> keep text up to the 3rd copy of a line"""
    seen, out = Counter(), []
    for ln in text.split("\n"):
        key = ln.strip()
        if len(key) >= min_len:
            seen[key] += 1
            if seen[key] >= max_seen:
                break
        out.append(ln)
    return "\n".join(out)


def is_degenerate(text):
    """highly repetitive text (compresses to <10%) = runaway generation, not a real page"""
    if len(text) < 2000:
        return False
    raw = text.encode("utf-8")
    return len(zlib.compress(raw)) / len(raw) < 0.10


def postprocess_ocr(text):
    """returns (clean_text, flag). flag: '' ok | 'loop-truncated' | 'no-text' | 'degenerate'"""
    t = truncate_loops(collapse_repeats(text.strip())).strip()
    flag = ""
    if len(t) < 120 and NO_TEXT_RX.match(t):
        return "", "no-text"
    if is_degenerate(t):
        return "", "degenerate"
    if len(text) - len(t) > 500:
        flag = "loop-truncated"
    return t, flag


def band_is_blank(img, ink_fraction=0.0005):
    """True if (almost) no dark pixels -> do not ask the model, it would only make up text or emit whitespace"""
    g = np.asarray(img.convert("L"))
    return float((g < 128).mean()) < ink_fraction


def ollama_ocr_image(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    payload = {
        "model": OLLAMA_MODEL, "stream": False, "keep_alive": "30m",
        "options": {"temperature": 0, "num_predict": OCR_MAX_TOKENS},
        "messages": [{"role": "user", "content": OCR_PROMPT,
                      "images": [base64.b64encode(buf.getvalue()).decode("ascii")]}],
    }
    r = _http.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=OCR_TIMEOUT)
    r.raise_for_status()
    text = r.json()["message"]["content"]
    return "\n".join(l for l in text.split("\n") if not CHATTER_RX.match(l))


def ocr_pdf_page(pdf, idx, md5, debug_count):
    """render -> (Bangla cleanup) -> bands -> Ollama. returns (text, failed_requests)"""
    img = render_page(pdf, idx)
    if IMG_PREPROCESS:
        img = preprocess_for_bangla(img)
    if debug_count < DEBUG_SAVE_PAGES:
        img.save(os.path.join(DEBUG_DIR, f"{md5[:8]}_p{idx + 1}.png"))
    parts, failed = [], 0
    for band in split_bands(img, OCR_BANDS):
        if band_is_blank(band):
            continue
        try:
            parts.append(ollama_ocr_image(band).strip())
        except Exception as e:
            failed += 1
            print("    OCR request failed:", e)
    return collapse_repeats("\n".join(p for p in parts if p)), failed


# Legacy-font Bangla (SutonnyMJ / Bijoy / broken ToUnicode) extracts as symbols, private-use or control chars,
# or as ASCII gibberish without real words. Both cases are treated as "no usable text layer".
BAD_RX = re.compile(r"[\u0080-\u024F\u02C6\u02DC\u2020-\u2022\u2030\u0152\u0153\uE000-\uF8FF\uFFFD"
                    r"\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
STOPWORDS = {"the", "of", "and", "to", "in", "a", "is", "that", "for", "shall", "be", "by", "or", "as", "on",
             "with", "any", "this", "are", "under", "which", "an", "from", "at", "it", "not", "such", "may",
             "has", "have", "been", "other", "all", "if", "its", "than", "their", "was", "were", "no"}


def is_garbled(t):
    t2 = re.sub(r"\s", "", t)
    if not t2:
        return False
    if len(BAD_RX.findall(t2)) > 0.03 * len(t2):
        return True
    bn = len(re.findall(r"[\u0980-\u09FF]", t2))
    if bn < 0.1 * len(t2) and len(t2) > 200:
        words = re.findall(r"[A-Za-z]{2,}", t)
        if len(words) >= 30:                     # looks "English" but has no English function words
            stop = sum(w.lower() in STOPWORDS for w in words) / len(words)
            if stop < 0.06:
                return True
    return False


def page_needs_ocr(t):
    t = t.strip()
    if len(t) < 40:
        return True
    alnum = len(re.findall(r"[\u0980-\u09FFA-Za-z0-9]", t))   # digits count: rate tables are mostly numbers
    return alnum < 0.3 * len(t) or is_garbled(t)


# ----------------------------------------------------------------------------
# EXTRACT + CLEAN  (PDF -> .txt with metadata header)
# ----------------------------------------------------------------------------
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def clean(t):
    t = unicodedata.normalize("NFC", t)
    for ch in ("\u00ad", "\u200b", "\u200c", "\u200d", "\ufeff"):
        t = t.replace(ch, "")
    t = re.sub(r"-\n(?=[a-z])", "", t)                    # de-hyphenate English line breaks
    t = re.sub(r"[._\-=~*]{4,}", " ", t)                  # dot leaders / ruler lines
    t = re.sub(r"([^\d\s])\1{5,}", r"\1", t)              # long runs of one repeated symbol (OCR artifact)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def read_pdf(path, md5):
    """returns [(page_no, text, used_ocr)]"""
    cache_path = os.path.join(OCR_CACHE, f"{md5}__{OCR_CACHE_TAG}.json")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)

    doc = fitz.open(path)
    n_pages = len(doc)
    pdf, dbg = None, 0
    pages = []
    need, from_cache, failed_req, unusable = 0, 0, 0, 0
    for i, pg in enumerate(doc, 1):
        t = pg.get_text("text")
        used = False
        if page_needs_ocr(t):
            need += 1
            key = str(i)
            if key in cache:
                t, flag = postprocess_ocr(cache[key])
                used = bool(t)
                from_cache += 1
                if flag:
                    OCR_FLAGS.append((os.path.basename(path), i, flag))
            else:
                ocr_text = ""
                if HAVE_OCR:
                    if pdf is None:
                        pdf = pdfium.PdfDocument(path)
                    ts = time.time()
                    ocr_text, failed = ocr_pdf_page(pdf, i - 1, md5, dbg)
                    dbg += 1
                    failed_req += failed
                    print(f"    OCR page {i}/{n_pages}: {len(ocr_text)} chars, {time.time() - ts:.0f}s")
                if ocr_text:                              # cache the RAW model output (filters run on read)
                    cache[key] = ocr_text
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cache, f, ensure_ascii=False)
                    t, flag = postprocess_ocr(ocr_text)
                    used = bool(t)
                    if flag:
                        OCR_FLAGS.append((os.path.basename(path), i, flag))
                        print(f"    [flag] page {i}: {flag}")
                elif is_garbled(t):                       # garbage text is worse than no text
                    t = ""
                    unusable += 1
        pages.append((i, t, used))
    doc.close()
    if pdf is not None:
        pdf.close()
    if need:
        print(f"  OCR: {need} page(s) needed it, {from_cache} from cache, {failed_req} failed request(s), "
              f"{unusable} unusable page(s) left empty")

    # drop repeating headers/footers (line on >40% of pages) and bare page numbers
    cnt = Counter(l.strip() for _, t, _ in pages for l in set(t.split("\n")) if 3 < len(l.strip()) < 90)
    common = {l for l, c in cnt.items() if len(pages) > 5 and c > 0.4 * len(pages)}
    out = []
    for i, t, used in pages:
        lines = [l for l in t.split("\n")
                 if l.strip() not in common and not re.fullmatch(r"\s*[\d০-৯]{1,4}\s*", l)]
        out.append((i, clean("\n".join(lines)), used))
    return out


def write_txt(path_txt, doc_id, folder, lang, url, src_file, pages):
    ocr_pages = [str(p) for p, _, used in pages if used]
    head = ["---",
            f"doc_id: {doc_id}",
            f"category: {folder}",
            f"language: {lang}",
            f"source_url: {url or ''}",
            f"source_file: {src_file}",
            f"n_pages: {len(pages)}",
            f"ocr_pages: {','.join(ocr_pages)}",
            f"extracted_at: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "---"]
    body = "\n".join(f"[[page {p}]]\n{t}\n" for p, t, _ in pages)
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(head) + "\n" + body)


def plan_pdfs():
    """{folder: [(path, md5, stem)]}; byte-identical PDFs are kept only once"""
    plan, seen = {}, {}
    for folder in OUTPUT_MAP.values():
        d = os.path.join(RAW, folder)
        plan[folder] = []
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith(".pdf"):
                continue
            p = os.path.join(d, fn)
            h = hashlib.md5()
            with open(p, "rb") as f:
                for blk in iter(lambda: f.read(1 << 20), b""):
                    h.update(blk)
            md5 = h.hexdigest()
            if md5 in seen:
                print("skip duplicate:", fn, "==", os.path.basename(seen[md5]))
                continue
            seen[md5] = p
            plan[folder].append((p, md5, os.path.splitext(fn)[0]))
    return plan


def detect_lang(pages, default):
    """language of the extracted text (NBR 'English' acts are often the Bangla gazette)"""
    txt = " ".join(t for _, t, _ in pages)
    bn = len(re.findall(r"[\u0980-\u09FF]", txt))
    en = len(re.findall(r"[A-Za-z]", txt))
    if bn + en < 200:
        return default
    return "bn" if bn / (bn + en) > 0.3 else "en"


def extract_all(meta, plan, force=False):
    for folder, items in plan.items():
        for p, md5, stem in items:
            tp = os.path.join(TEXT_DIR, folder, stem + ".txt")
            if os.path.exists(tp) and not force:
                print("txt exists, skipped:", folder, "/", stem)
                continue
            info = meta.get(p, {})
            lang = info.get("lang", "bn" if folder.endswith("_bn") else "en")
            print("extracting", folder, "/", stem)
            try:
                pages = read_pdf(p, md5)
                det = detect_lang(pages, lang)
                if det != lang:
                    print(f"  [lang] {stem}: expected {lang}, text is {det} -> saved as {det}")
                write_txt(tp, stem, folder, det, info.get("url"), os.path.basename(p), pages)
            except Exception as e:
                print("  ERROR", stem, e)
        # txt files that no longer have a PDF (stale / duplicate) are reported, not deleted
        stems = {s for _, _, s in items}
        for fn in os.listdir(os.path.join(TEXT_DIR, folder)):
            if fn.endswith(".txt") and fn[:-4] not in stems:
                print(f"  [note] orphan txt (no matching PDF, ignored when chunking): {folder}/{fn}")


# ----------------------------------------------------------------------------
# .txt -> sentence-based chunks
# ----------------------------------------------------------------------------
PAGE_MARK = re.compile(r"^\[\[page (\d+)\]\]\s*$", re.M)
SEC_EN = re.compile(r"^(\d{1,3}[A-Z]{0,2})\.\s+([A-Z][^\n]{2,160}?)(?:\.\s?[—–-]|\s?[—–-]\s?\(1\)|$)", re.M)
SEC_BN = re.compile(r"^([০-৯]{1,3}[ক-হ]?)\s*[।.]\s+([^\n]{2,160}?)(?:[।.]\s?[—–-]|$)", re.M)

# a line starting like this begins a new sentence / clause: "(a) ", "12. ", "১২। ", "• "
MARKER = re.compile(r"^(?:[(\[]?[\dক-হa-zA-Z০-৯]{1,4}[)\].।]\s|[•\-–—]\s)")
ABBR = {"no", "nos", "sl", "dr", "mr", "mrs", "ms", "rs", "tk", "sec", "ss", "art", "vol", "ltd", "co",
        "inc", "cf", "viz", "etc", "vs", "para", "cl", "sub", "dt", "st", "govt", "dept", "m/s", "u/s", "r/w"}
BOUND = re.compile(r'(?P<p>[।?!]+|\.)["”’)\]]*[ \t]+|\n+')


def read_txt(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    hdr = {}
    m = re.match(r"\A---\n(.*?)\n---\n", raw, re.S)
    if m:
        for line in m.group(1).split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                hdr[k.strip()] = v.strip()
        raw = raw[m.end():]
    parts = PAGE_MARK.split(raw)        # [before, num, text, num, text, ...]
    if len(parts) == 1:                 # page markers were removed by hand
        return hdr, [(None, raw.strip())]
    pages = [(int(parts[i]), parts[i + 1].strip()) for i in range(1, len(parts), 2)]
    if parts[0].strip():
        pages.insert(0, (None, parts[0].strip()))
    return hdr, pages


def reflow(text):
    """join hard-wrapped lines of one paragraph; keep a newline only where a sentence/clause really starts"""
    out = []
    for para in re.split(r"\n\s*\n", text):
        cur = ""
        for ln in (l.strip() for l in para.split("\n")):
            if not ln:
                continue
            if not cur:
                cur = ln
            elif MARKER.match(ln) or "|" in ln or re.search(r"[।.?!]$", cur):
                cur += "\n" + ln
            else:
                cur += " " + ln
        if cur:
            out.append(cur)
    return "\n".join(out)


def split_sentences(text):
    """multilingual splitter: Bangla danda, ?, ! and '.' (with abbreviation / list-marker guards).
    returns [(start_offset, sentence)]"""
    res, start = [], 0
    for m in BOUND.finditer(text):
        end = m.end()
        p = m.group("p")
        ok = True
        if p == ".":
            pre = text[start:m.start()]
            toks = pre.split()
            tok = toks[-1].strip("([{\"“‘").lower() if toks else ""
            nxt = text[end:end + 1]
            if tok in ABBR or len(tok) == 1:
                ok = False
            elif len(toks) == 1 and re.fullmatch(r"[\dক-হ০-৯]{1,3}[a-zA-Z]?", tok):
                ok = False                                   # "1." / "12A." list or section number
            elif nxt and nxt.isascii() and nxt.islower():
                ok = False                                   # "... vs. the ..." style continuation
        if ok:
            s = text[start:m.start() if p is None else end].strip()
            if s:
                res.append((start, s))
            start = end
    tail = text[start:].strip()
    if tail:
        res.append((start, tail))
    return res


def hard_split(s, limit):
    """very long legal sentence -> pieces <= limit, cut at ; or : first, then at spaces"""
    if len(s) <= limit:
        return [s]
    pieces, cur = [], ""
    for seg in re.split(r"(?<=[;:])\s+", s):
        while len(seg) > limit:
            cut = seg.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            if cur:
                pieces.append(cur.strip())
                cur = ""
            pieces.append(seg[:cut].strip())
            seg = seg[cut:].strip()
        if cur and len(cur) + len(seg) + 1 > limit:
            pieces.append(cur.strip())
            cur = ""
        cur += seg + " "
    if cur.strip():
        pieces.append(cur.strip())
    return pieces


FOREIGN_RX = re.compile(r"[\u0370-\u052F\u0590-\u06FF\u0900-\u0963\u0966-\u097F\u0A00-\u0AFF\u0E00-\u0E7F"
                        r"\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]")   # U+0964/0965 (danda) is shared with Bangla
_foreign_removed = 0


def drop_foreign_tokens(text):
    """remove words that contain letters of scripts that cannot occur in Bangla/English legal text"""
    global _foreign_removed
    out = []
    for ln in text.split("\n"):
        toks = ln.split(" ")
        keep = [t for t in toks if not FOREIGN_RX.search(t)]
        _foreign_removed += len(toks) - len(keep)
        out.append(" ".join(keep))
    return "\n".join(out)


def chunk_pages(pages, doc_id, lang, url, folder):
    global _foreign_removed
    _foreign_removed = 0
    # 1) reflow each page, then join pages (a sentence may continue over a page break, but a page that
    #    starts with a section heading always starts a new line: footnotes at the page bottom have no full stop)
    rx = SEC_BN if lang == "bn" else SEC_EN
    full, page_starts, page_nums = "", [], []
    for pno, t in pages:
        if DROP_FOREIGN_TOKENS:
            t = drop_foreign_tokens(t)
        t = reflow(t)
        if not t:
            continue
        if full:
            full += "\n" if (re.search(r"[।.?!:;]$", full) or rx.match(t)) else " "
        page_starts.append(len(full))
        page_nums.append(pno)
        full += t

    # 2) sentences (over-long ones are split further)
    sents = []
    for off, sent in split_sentences(full):
        for piece in hard_split(sent, CHUNK_MAX_CHARS * 2):
            sents.append((off, piece))

    # 3) section headings: at a line start OR at the start of a sentence (headings glued to the previous
    #    text by the PDF extractor), then a monotonic filter drops OCR noise such as a stray "721."
    raw = {m.start(): (m.group(1).translate(BN_DIGITS), m.group(2).strip()) for m in rx.finditer(full)}
    for off, sent in sents:
        m = rx.match(sent)
        if m and off not in raw:
            raw[off] = (m.group(1).translate(BN_DIGITS), m.group(2).strip())
    heads, last = [], 0
    for off in sorted(raw):
        num, title = raw[off]
        mm = re.match(r"\d+", num)
        if not mm:
            continue
        n = int(mm.group())
        if (last < n <= last + 40) or n == 1:          # n == 1: new sequence (table of contents -> body, schedules)
            heads.append((off, num, title))
            last = n
    head_pos = [h[0] for h in heads]
    head_offs = set(head_pos)

    def page_of(off):
        i = bisect.bisect_right(page_starts, off) - 1
        return page_nums[max(i, 0)] if page_nums else None

    def sec_of(off):
        i = bisect.bisect_right(head_pos, off) - 1
        return (heads[i][1], heads[i][2]) if i >= 0 else (None, None)

    # 4) sentence-based chunks with controlled overlap; a new section starts a new chunk (no overlap across it)
    groups, cur, cur_len, fresh, prev_off = [], [], 0, 0, None
    for sent in sents:
        L = len(sent[1]) + 1
        new_section = sent[0] in head_offs and sent[0] != prev_off
        if fresh and (cur_len + L > CHUNK_MAX_CHARS or (new_section and cur_len >= SECTION_FLUSH_MIN)):
            groups.append(cur)
            if new_section:
                cur, cur_len, fresh = [], 0, 0
            else:
                keep, n = [], 0
                for x in reversed(cur):
                    if len(keep) >= OVERLAP_MAX_SENTENCES or n + len(x[1]) > CHUNK_OVERLAP_CHARS:
                        break
                    keep.insert(0, x)
                    n += len(x[1]) + 1
                cur, cur_len, fresh = keep, n, 0
        cur.append(sent)
        cur_len += L
        fresh += 1
        prev_off = sent[0]
    if fresh:
        groups.append(cur)

    out = []
    for j, g in enumerate(groups):
        text = " ".join(x[1] for x in g)
        sec_no, title = sec_of(g[0][0])
        i0, i1 = bisect.bisect_left(head_pos, g[0][0]), bisect.bisect_right(head_pos, g[-1][0])
        nums = [sec_no] if sec_no else []
        nums += [h[1] for h in heads[i0:i1] if h[1] not in nums]
        cid = hashlib.md5(f"{doc_id}|{j}|{text[:60]}".encode("utf-8")).hexdigest()[:12]
        out.append(dict(id=cid, doc_id=doc_id, category=folder, language=lang,
                        section_no=sec_no, sections=",".join(nums), title=title, part=j,
                        page_start=page_of(g[0][0]), page_end=page_of(g[-1][0]),
                        n_sentences=len(g), source_url=url, text=text))
    return out


def chunk_all(plan):
    for out_name, folder in OUTPUT_MAP.items():
        allc = []
        for p, md5, stem in plan[folder]:
            tp = os.path.join(TEXT_DIR, folder, stem + ".txt")
            if not os.path.exists(tp):
                print("missing txt (run the extract stage):", folder, "/", stem)
                continue
            hdr, pages = read_txt(tp)
            lang = hdr.get("language") or ("bn" if folder.endswith("_bn") else "en")
            chunks = chunk_pages(pages, stem, lang, hdr.get("source_url") or None, folder)
            extra = f"  (OCR noise: {_foreign_removed} foreign-script words removed)" if _foreign_removed else ""
            print(f"chunked {folder} / {stem}: {len(chunks)} chunks{extra}")
            allc += chunks
        with open(os.path.join(OUT, out_name), "w", encoding="utf-8") as f:
            json.dump(allc, f, ensure_ascii=False, indent=1)
        print(f"==> {out_name}: {len(allc)} chunks\n")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    global HAVE_OCR
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["download", "extract", "chunk"], help="run a single stage")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--force-extract", action="store_true", help="rebuild .txt files (overwrites manual edits)")
    args = ap.parse_args()

    stages = {"download", "extract", "chunk"} if not args.only else {args.only}
    if args.skip_download:
        stages.discard("download")

    print("ROOT =", ROOT)
    meta = download_all() if "download" in stages else load_sources()
    plan = plan_pdfs()
    if "extract" in stages:
        HAVE_OCR = check_ollama()
        extract_all(meta, plan, force=args.force_extract)
    if "chunk" in stages:
        chunk_all(plan)
    if OCR_FLAGS:
        with open(os.path.join(RAW, "_ocr_flags.log"), "w", encoding="utf-8") as f:
            for fn, pg, flag in OCR_FLAGS:
                f.write(f"{fn}\tpage {pg}\t{flag}\n")
        print(f"{len(OCR_FLAGS)} OCR page(s) flagged -> {os.path.join(RAW, '_ocr_flags.log')}")
        for k, v in Counter(fl for _, _, fl in OCR_FLAGS).items():
            print(f"   {k}: {v}")
    print("done.")


if __name__ == "__main__":
    main()