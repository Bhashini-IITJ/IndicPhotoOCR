from IndicPhotoOCR.ocr import OCR
import os
import json

# === CONFIG ===
image_folder = "/Users/samierraarora/IndicPhotoOCR/IndicPhotoOCR/IndicPhotoOCR2/my_photos"   # folder with your 54 images
output_folder = "/Users/samierraarora/IndicPhotoOCR/IndicPhotoOCR/IndicPhotoOCR2/PAKKA_OCR_OUTPUTS"  # where JSON files will be saved
os.makedirs(output_folder, exist_ok=True)

# === RUN OCR ===
ocr_system = OCR(verbose=True, identifier_lang="auto", device="cpu")

# Loop through all images
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_folder, filename)
        print(f"\n🚀 Processing {image_path}")
        
        # Run OCR
        results = ocr_system.ocr(image_path)
        print(results)
        
        # Save JSON named after the image
        output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"✅ OCR results saved to {output_file}")
