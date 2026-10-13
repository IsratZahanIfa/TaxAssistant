import pickle
import re

def create_parent_child_index(pkl_input_path, pkl_output_path):
    print("Parent-Child Chunking শুরু হচ্ছে...")
    with open(pkl_input_path, "rb") as f:
        raw_docs = pickle.load(f)

    hierarchical_docs = []
    
    for doc in raw_docs:
        sec = str(doc.get('section', '')).strip()
        title = str(doc.get('title', '')).strip()
        full_text = str(doc.get('text', '')).strip()
        
        parent_context = f"ACT SECTION {sec}: {title}"
        sub_chunks = re.split(r'(\(\d+\)|\([a-z]\))', full_text)
        
        if len(sub_chunks) > 1:
            current_child = ""
            for token in sub_chunks:
                if re.match(r'(\(\d+\)|\([a-z]\))', token):
                    if current_child.strip():
                        hierarchical_docs.append({
                            "parent_sec": sec,
                            "parent_title": title,
                            "child_text": f"{parent_context} -> {current_child.strip()}",
                            "full_doc": doc
                        })
                    current_child = token
                else:
                    current_child += " " + token
            if current_child.strip():
                hierarchical_docs.append({
                    "parent_sec": sec,
                    "parent_title": title,
                    "child_text": f"{parent_context} -> {current_child.strip()}",
                    "full_doc": doc
                })
        else:
            hierarchical_docs.append({
                "parent_sec": sec,
                "parent_title": title,
                "child_text": f"{parent_context} -> {full_text}",
                "full_doc": doc
            })

    with open(pkl_output_path, "wb") as f:
        pickle.dump(hierarchical_docs, f)
    
    print(f"ইনডেক্সিং সফল হয়েছে! মোট চ্যাঙ্ক তৈরি হয়েছে: {len(hierarchical_docs)}")

if __name__ == '__main__':
    create_parent_child_index(
        r"E:\TaxAssist\taxdatabase\tax_data.pkl", 
        r"E:\TaxAssist\taxdatabase\tax_data_parent_child.pkl"
    )
    