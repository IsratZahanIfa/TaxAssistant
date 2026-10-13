from pathlib import Path
from unicodeconverter import converter

input_file = Path("data/Clean_Text/Income_Tax_Act_2023_raw.txt")

print("=" * 70)
print("TaxAssist - Bijoy to Unicode Test")
print("=" * 70)

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read(5000)

converted = converter.convert_bijoy_to_unicode(text)

print("\n" + "=" * 70)
print("CONVERTED TEXT")
print("=" * 70)
print(converted[:3000])