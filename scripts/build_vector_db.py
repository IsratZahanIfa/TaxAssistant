import os
import sys
import pickle
import json
import gc
from pathlib import Path
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
db_dir = project_root / "taxdatabase"

EMBEDDING_MODEL = "BAAI/bge-m3"

def load_source_data():
    pkl_file = db_dir / "tax_data_parent_child.pkl"
    pkl_fallback = db_dir / "tax_data.pkl"
    json_file = db_dir / "tax_data_with_vectors.json"

    if pkl_file.exists():
        print(f"Loading data from {pkl_file.name}...")
        with open(pkl_file, "rb") as f:
            return pickle.load(f)
    elif pkl_fallback.exists():
        print(f"Loading data from {pkl_fallback.name}...")
        with open(pkl_fallback, "rb") as f:
            return pickle.load(f)
    elif json_file.exists():
        print(f"Loading data from {json_file.name}...")
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise FileNotFoundError("No source data (.pkl or .json) found in taxdatabase folder.")

def convert_to_documents(raw_data):
    documents = []
    items = raw_data if isinstance(raw_data, list) else list(raw_data.values())

    for idx, item in enumerate(items):
        if isinstance(item, str):
            text = item
            metadata = {"law_number": [str(idx + 1)]}
        elif isinstance(item, dict):
            text = (
                item.get("page_content") 
                or item.get("text") 
                or item.get("content") 
                or str(item)
            )
            metadata = item.get("metadata", {})
            if "law_number" not in metadata:
                law_num = item.get("law_number") or item.get("section") or str(idx + 1)
                metadata["law_number"] = [law_num] if not isinstance(law_num, list) else law_num
        else:
            text = str(item)
            metadata = {"law_number": [str(idx + 1)]}

        if text.strip():
            documents.append(Document(page_content=text, metadata=metadata))

    print(f"✓ Converted {len(documents)} valid items into LangChain Documents.")
    return documents

def create_faiss_db():
    print("=" * 80)
    print("🚀 BUILDING FAISS VECTOR DATABASE (LOW MEMORY OPTIMIZED)")
    print("=" * 80)

    raw_data = load_source_data()
    docs = convert_to_documents(raw_data)

    if not docs:
        raise ValueError("No valid text documents found!")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Batch size 16 করা হলো VRAM OOM ঠেকানোর জন্য
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )

    print("Embedding texts and creating FAISS index...")
    
    batch_size = 64  # প্রতি প্রসেসিং ব্যাচে ৬৪টি ডকুমেন্ট
    vector_store = None
    
    total_batches = (len(docs) + batch_size - 1) // batch_size
    
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        current_batch = i // batch_size + 1
        print(f"  -> Processing batch {current_batch}/{total_batches} ({len(batch)} items)...")
        
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            vector_store.add_documents(batch)

        # GPU Cache খালি করার জন্য
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    vector_store.save_local(str(db_dir))
    
    print("\n" + "=" * 80)
    print(f"✅ FAISS Database created successfully!")
    print(f"📁 Files saved in: {db_dir}")
    print("=" * 80)

if __name__ == "__main__":
    create_faiss_db()