import re
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional

RAW_DATA_PATH = "data/Knowledge_Dataset/income_tax_act_2023.json"
OUTPUT_JSON_PATH = "taxdatabase/tax_data_with_vectors.json"
OUTPUT_PARENT_CHILD_PATH = "taxdatabase/tax_data_parent_child.pkl"
CHUNK_SIZE = 512  

def extract_metadata_from_text(text: str, section_num: Optional[str] = None) -> Dict[str, Any]:
    """
    টেক্সট থেকে মেটাডেটা এক্সট্র্যাক্ট করে।
    যেমন: Chapter নাম, মূল শিরোনাম, গুরুত্বপূর্ণ কীওয়ার্ড।
    """
    metadata = {
        "section": section_num,
        "chapter": "Unknown",
        "title": "",
        "keywords": [],
        "parent_section": None,
        "has_sub_sections": False
    }

    chapter_match = re.search(r'(CHAPTER|Chapter)\s+([A-Z0-9\-]+)', text, re.IGNORECASE)
    if chapter_match:
        metadata["chapter"] = chapter_match.group(0).strip()

    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line and len(line) > 5:
       
            if re.match(r'^(\d+)[\.\s]+', line):
                metadata["title"] = line
                break
            else:
                metadata["title"] = line[:200]  
                break

    if re.search(r'\(\d+\)', text):
        metadata["has_sub_sections"] = True
        
   
    parent_match = re.search(r'^(\d+)\.', text)
    if parent_match:
        metadata["parent_section"] = parent_match.group(1)

    tax_keywords = ["Income", "Tax", "Deduction", "Rebate", "Investment", "Allowance", 
                    "Exemption", "Return", "Assessment", "Penalty", "Appeal", "Refund"]
    found_keywords = []
    for kw in tax_keywords:
        if kw.lower() in text.lower():
            found_keywords.append(kw)
  
    if section_num:
        found_keywords.append(f"Section_{section_num}")
    
    metadata["keywords"] = list(set(found_keywords))  

    return metadata


def chunk_and_annotate(raw_data: List[Dict]) -> List[Dict]:
    """
    Raw ডেটা থেকে প্রতিটি ডকুমেন্ট প্রসেস করে মেটাডেটা যোগ করে।
    """
    enriched_docs = []
    parent_child_map = {}  

    for idx, item in enumerate(raw_data):
       
        section_num = item.get("section") or item.get("id") or str(idx+1)
        raw_text = item.get("text") or item.get("content") or item.get("body") or ""

        if not raw_text:
            continue

        meta = extract_metadata_from_text(raw_text, section_num)

    
        chunks = []
        if meta["has_sub_sections"] and len(raw_text) > CHUNK_SIZE:
         
            sub_sections = re.split(r'(\(\d+\))', raw_text)
            current_chunk = ""
            for part in sub_sections:
                current_chunk += part
                if len(current_chunk) > CHUNK_SIZE or re.search(r'\(\d+\)\s*$', part):
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = ""
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        else:
          
            for i in range(0, len(raw_text), CHUNK_SIZE):
                chunks.append(raw_text[i:i+CHUNK_SIZE])

   
        for i, chunk_text in enumerate(chunks):
            doc_id = f"{section_num}_{i}"
            enriched_doc = {
                "id": doc_id,
                "text": chunk_text,
                "metadata": meta.copy(),
                "chunk_index": i
            }
            enriched_docs.append(enriched_doc)

            if meta["parent_section"]:
                parent_id = f"{meta['parent_section']}_0"
                if parent_id not in parent_child_map:
                    parent_child_map[parent_id] = []
                parent_child_map[parent_id].append(doc_id)

    return enriched_docs, parent_child_map


def save_database(enriched_docs: List[Dict], parent_child_map: Dict):
    """
    তৈরি ডেটাবেস JSON এবং PKL ফাইল হিসেবে সেভ করে।
    """
 
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched_docs, f, ensure_ascii=False, indent=2)
    print(f"✅ Enriched data saved to: {OUTPUT_JSON_PATH}")

    with open(OUTPUT_PARENT_CHILD_PATH, "wb") as f:
        pickle.dump(parent_child_map, f)
    print(f"✅ Parent-Child map saved to: {OUTPUT_PARENT_CHILD_PATH}")

    print(f"\n📊 Summary:")
    print(f"   Total Chunks Created: {len(enriched_docs)}")
    print(f"   Total Parent-Child Groups: {len(parent_child_map)}")


if __name__ == "__main__":
    print("🚀 Starting Metadata Extraction & Index Building...")

    if not Path(RAW_DATA_PATH).exists():
        print(f"❌ Error: File not found at {RAW_DATA_PATH}")
        print("   Please check the path or run PDF extraction first.")
        exit(1)
    
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    print(f"✅ Loaded {len(raw_data)} raw documents.")
    
    enriched_docs, parent_child_map = chunk_and_annotate(raw_data)
    
    save_database(enriched_docs, parent_child_map)
    
    print("\n✨ Done! Now you can run your retrieval scripts.")