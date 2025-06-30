from IndicPhotoOCR.ocr import OCR

# === CONFIG ===
image_path = "/Users/samierraarora/IndicPhotoOCR/IndicPhotoOCR/IndicPhotoOCR2/IndicPhotoOCR/script_identification/pic_testing/image copy 4.png"
# === RUN OCR ===
ocr_system = OCR(verbose=True, identifier_lang="auto",device="cpu")  # for CPU, or add device="cpu"
results = ocr_system.ocr(image_path)

# === PRINT RESULTS LIKE YOU WANTED ===
#for idx, result in enumerate(results):
    #print(f"#{idx} Recognized word: {result['text']} (Language: {result['lang']})")
print (results)

import json

# Assuming `results` is your final grouped list like:
# [['ब्लल्यू', 'बेरी', 'सहकारी'], ['गृहरचना', 'संस्था', 'मर्यादित'], ['blue', 'berry', 'cooperative'], ['housing', 'society', 'limited']]

# Choose your JSON output file name
output_file = "ocr_output.json"

# Write to JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"✅ OCR results saved to {output_file}")

