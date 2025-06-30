import os
import json

# === CONFIG: adjust this to your folder structure ===
output_folder = "//Users/samierraarora/IndicPhotoOCR/IndicPhotoOCR/IndicPhotoOCR2/PAKKA_OCR_OUTPUTS"

expected_counts = {}

# === Loop through each JSON file and ask user ===
for filename in sorted(os.listdir(output_folder)):
    if filename.lower().endswith(".json"):
        while True:
            try:
                expected = int(input(f"How many words should {filename} have? "))
                break
            except ValueError:
                print("Please enter a valid integer.")
        
        expected_counts[filename] = expected

# === Save to JSON ===
with open("expected_counts.json", "w", encoding="utf-8") as f:
    json.dump(expected_counts, f, ensure_ascii=False, indent=4)

print("\n✅ All expected counts saved to expected_counts.json")

