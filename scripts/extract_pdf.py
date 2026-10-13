import fitz
from pathlib import Path

pdf_path = Path("data/Raw_PDF/Income_tax_act_2023.pdf")
output_path = Path("data/Clean_Text/Income_Tax_Act_2023_raw.txt")

if not pdf_path.exists():
    print("ERROR: PDF not found!")
    raise SystemExit

doc = fitz.open(pdf_path)

print("PDF opened successfully!")
print("Total pages:", len(doc))

with open(output_path, "w", encoding="utf-8") as f:
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text()

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write(f"PAGE {page_number}\n")
        f.write("=" * 80 + "\n")
        f.write(text)

doc.close()

print("\nExtraction completed successfully!")
print("Saved to:", output_path)