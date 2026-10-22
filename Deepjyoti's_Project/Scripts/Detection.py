import os
import json
import cv2
from IndicPhotoOCR.ocr import OCR
from matplotlib import pyplot as plt

# Initialize OCR system
ocr_system = OCR(device="cpu", verbose=True)

# Input and output directories
input_folder = "C:/Users/Deepjyoti Roy/OneDrive/Desktop/IndicPhotoOCR-Mine/Dataset/Raw_Images"
output_folder = "C:/Users/Deepjyoti Roy/OneDrive/Desktop/IndicPhotoOCR-Mine/Outputs/Detection_outputs"
results_folder = os.path.join(output_folder, "Results")

# Ensure all directories exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)

# Output file path for JSON
json_output_path = os.path.join(results_folder, "detection_results.json")

# Supported image formats
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# Master detection dictionary for JSON output
all_detections = {}

# Process each image
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(valid_extensions):
        continue

    image_path = os.path.join(input_folder, filename)
    image_id = os.path.splitext(filename)[0]
    print(f"\n🔍 Detecting in: {filename}")

    try:
        # Perform detection
        detections = ocr_system.detect(image_path)

        if not detections:
            print(f"⚠️ No detections found for {filename}")
            continue

        # Save visualized detection image
        output_image_path = os.path.join(output_folder, f"detection_{filename}")
        ocr_system.visualize_detection(image_path, detections, save_path=output_image_path)
        print(f"✅ Saved visualization to: {output_image_path}")

        # Save detection results in dictionary
        all_detections[image_id] = detections

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")

# Save all detection results to JSON
with open(json_output_path, "w", encoding="utf-8") as json_file:
    json.dump(all_detections, json_file, ensure_ascii=False, indent=4)

print(f"\n✅ All detections saved:\n- JSON: {json_output_path}")
