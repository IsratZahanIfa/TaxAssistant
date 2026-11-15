from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import torch

db_dir = Path("E:/TaxAssist/taxdatabase")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={
        "normalize_embeddings": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
)

db = FAISS.load_local(
    str(db_dir),
    embeddings,
    index_name="tax_data",
    allow_dangerous_deserialization=True
)

print("Total documents:", len(db.docstore._dict))

for i, doc in enumerate(db.docstore._dict.values()):
    print("\nDOCUMENT", i + 1)
    print("METADATA:", doc.metadata)
    print("TEXT:", doc.page_content[:500])

    if i >= 9:
        break
