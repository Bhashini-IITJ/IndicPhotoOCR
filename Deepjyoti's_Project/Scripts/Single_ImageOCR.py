import os
import json
from IndicPhotoOCR.ocr import OCR

# Initialize OCR system
ocr_system = OCR(verbose=True, identifier_lang="auto", device="cpu")

# Input image path
image_path = "C:/Users/Deepjyoti Roy/OneDrive/Desktop/IndicPhotoOCR-Mine/Dataset/Raw_Images/img71.jpg"

# Run OCR
words = ocr_system.ocr(image_path)

# Create output directory if it doesn't exist
output_dir = os.path.join("C:/Users/Deepjyoti Roy/OneDrive/Desktop/IndicPhotoOCR-Mine/Outputs/Single_OCR_outputs")
os.makedirs(output_dir, exist_ok=True)

# Define output file name based on image name
image_name = os.path.splitext(os.path.basename(image_path))[0]
output_file = os.path.join(output_dir, f"{image_name}_ocr_output.json")

# Save the OCR result to JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(words, f, ensure_ascii=False, indent=4)

# Also print to console (optional)
print(words)
