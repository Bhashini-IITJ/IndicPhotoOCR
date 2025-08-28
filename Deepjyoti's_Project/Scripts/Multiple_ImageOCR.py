import os
import json
from IndicPhotoOCR.ocr import OCR

# Initialize OCR system
ocr_system = OCR(verbose=True, identifier_lang="auto", device="cpu")

# Input and output paths
input_folder = "C:/Users/Deepjyoti Roy/OneDrive/Desktop/IndicPhotoOCR-Mine/Dataset/Raw_Images"
output_dir = "C:/Users/Deepjyoti Roy/OneDrive/Desktop/IndicPhotoOCR-Mine/Outputs/Batch_OCR_outputs"
os.makedirs(output_dir, exist_ok=True)

# Output file path
combined_output_path = os.path.join(output_dir, "combined_ocr_output.json")

# Supported image extensions
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# Combined result dictionary for JSON
combined_results = {}

# Process each image
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(valid_extensions):
        continue

    image_path = os.path.join(input_folder, filename)
    image_id = os.path.splitext(filename)[0]
    print(f"\nProcessing: {filename}")

    try:
        result = ocr_system.ocr(image_path)
        combined_results[image_id] = result
    except Exception as e:
        print(f" Failed on {filename}: {e}")
        continue

# Save combined JSON output
with open(combined_output_path, "w", encoding="utf-8") as f:
    json.dump(combined_results, f, ensure_ascii=False, indent=4)

print(f"\n✅ OCR completed.")
print(f"📄 Combined JSON saved at: {combined_output_path}")
