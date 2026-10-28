import json
import os
import pickle
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from collections import Counter
from tqdm import tqdm
import torch
import gc

# ============================================
# 1. PATH CONFIGURATION
# ============================================

input_json_path = r"E:\TaxAssist\data\Knowledge_Dataset\income_tax_act_2023.json"
output_dir = r"E:\TaxAssist\data\Database"
output_pkl_path = os.path.join(output_dir, "tax_data_parent_child.pkl")
output_faiss_path = os.path.join(output_dir, "tax_index.faiss")
output_meta_path = os.path.join(output_dir, "tax_metadata.pkl")

os.makedirs(output_dir, exist_ok=True)

# ============================================
# 2. HELPER FUNCTIONS
# ============================================

def extract_section_number(text):
    if not text:
        return ""
    match = re.search(r'(?:section|sec|ধারা)?\s*(\d+[a-z]?)', str(text), re.IGNORECASE)
    return match.group(1) if match else ""

def extract_topic_from_text(text, title=""):
    combined = (title + " " + text).lower()
    topic_keywords = {
        'salary': ['বেতন', 'salary', 'wages', 'remuneration', 'চাকরি', 'employment'],
        'rent': ['ভাড়া', 'rent', 'lease', 'house property'],
        'business': ['ব্যবসা', 'business', 'trade', 'commerce', 'profit'],
        'agriculture': ['কৃষি', 'agriculture', 'farming', 'cultivation'],
        'capital_gain': ['capital gain', 'মূলধন লাভ', 'share', 'stock'],
        'investment': ['বিনিয়োগ', 'investment', 'savings', 'provident fund', 'pension'],
        'interest': ['সুদ', 'interest', 'fixed deposit', 'FDR'],
        'foreign': ['foreign', 'বিদেশ', 'abroad', 'remittance'],
        'freelance': ['freelance', 'ফ্রিল্যান্স', 'online', 'gig'],
        'tax_procedure': ['return', 'রিটার্ন', 'TIN', 'assessment', 'audit', 'notice', 'appeal']
    }
    for topic, keywords in topic_keywords.items():
        if any(kw in combined for kw in keywords):
            return topic
    return 'general'

def extract_keywords(text, top_n=10):
    words = re.findall(r'[\w]+', str(text).lower())
    stopwords = {'আর', 'ও', 'এবং', 'থেকে', 'করতে', 'হতে', 'যে', 'তা', 'এই', 'সে', 'তারা',
                 'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'have'}
    words = [w for w in words if w not in stopwords and len(w) > 2]
    freq = Counter(words)
    return [w for w, _ in freq.most_common(top_n)]

# ============================================
# 3. LOAD DATA
# ============================================

print("Loading JSON dataset...")
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"Loaded {len(data)} records")

# ============================================
# 4. CREATE ENRICHED DOCUMENTS
# ============================================

print("Creating enriched documents...")
documents = []
enriched_texts = []
metadata_list = []

for item in tqdm(data, desc="Processing"):
    part = item.get("part", "")
    chapter = item.get("chapter", "")
    section = item.get("section", "")
    title = item.get("title", "")
    text = item.get("text", "")
    page = item.get("page", "")
    schedule = item.get("schedule", "")
    
    section_num = extract_section_number(section)
    topic = extract_topic_from_text(text, title)
    keywords = extract_keywords(text, top_n=10)
    year_match = re.search(r'\b(19|20)\d{2}\b', str(text))
    year = year_match.group(0) if year_match else ""
    
    doc = {
        'parent_sec': section_num,
        'parent_title': title,
        'parent_part': part,
        'parent_chapter': chapter,
        'child_text': text,
        'section': section_num,
        'title': title,
        'part': part,
        'chapter': chapter,
        'page': page,
        'schedule': schedule,
        'topic': topic,
        'year': year,
        'keywords': keywords,
        'metadata': item
    }
    documents.append(doc)
    
    metadata_list.append({
        'section': section_num,
        'topic': topic,
        'year': year,
        'part': part,
        'chapter': chapter,
        'schedule': schedule,
        'doc_id': len(documents) - 1
    })
    
    enriched_text = f"""
    [SECTION: {section_num}] 
    [TITLE: {title}] 
    [PART: {part}] 
    [CHAPTER: {chapter}] 
    [TOPIC: {topic}] 
    [YEAR: {year}]
    TEXT: {text}
    """
    enriched_texts.append(enriched_text.strip())

print(f"Created {len(documents)} enriched documents")

# ============================================
# 5. GPU-OPTIMIZED EMBEDDING GENERATION
# ============================================

print("Loading BGE-M3 model with GPU optimization...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")

model = SentenceTransformer("BAAI/bge-m3", device=device)

if device == "cuda":
    model.half()
    print("Model converted to FP16 (half precision) for GPU memory efficiency")

if device == "cuda":
    torch.cuda.empty_cache()
    gc.collect()

print("Generating embeddings...")
BATCH_SIZE = 8
embeddings = model.encode(
    enriched_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
)

if device == "cuda":
    torch.cuda.empty_cache()
    gc.collect()

print(f"Embeddings shape: {embeddings.shape}")

# ============================================
# 6. SAVE TO PKL
# ============================================

print("Saving PKL files...")
pkl_data = {
    "documents": documents,
    "embeddings": embeddings,
    "enriched_texts": enriched_texts,
    "metadata": metadata_list
}

with open(output_pkl_path, "wb") as f:
    pickle.dump(pkl_data, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"✅ PKL saved: {output_pkl_path}")

with open(output_meta_path, "wb") as f:
    pickle.dump(metadata_list, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f" Metadata saved: {output_meta_path}")
print("Creating FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings.astype('float32'))
print(f"FAISS index created with {index.ntotal} vectors")

faiss.write_index(index, output_faiss_path)
print(f" FAISS index saved: {output_faiss_path}")
print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)
print(f"Total documents: {len(documents)}")
print(f"Embedding dimension: {dimension}")
print(f"FAISS index size: {index.ntotal}")
print(f"Topics: {set([d['topic'] for d in documents])}")
print("=" * 70)
print(" All files created successfully!")