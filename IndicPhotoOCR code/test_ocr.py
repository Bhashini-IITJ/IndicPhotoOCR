from IndicPhotoOCR.ocr import OCR
import json
import os
import cv2

# Initialize OCR system
ocr_system = OCR(verbose=True, identifier_lang="auto", device="cpu")

# Paths
json_files = [
    r"C:\Users\udayk\IndicPhotoOCR\test_data1.json",
    r"C:\Users\udayk\IndicPhotoOCR\test_data2.json",
    r"C:\Users\udayk\IndicPhotoOCR\test_data3.json",
    r"C:\Users\udayk\IndicPhotoOCR\test_data4.json",
    r"C:\Users\udayk\IndicPhotoOCR\test_data5.json"
]
image_directory = r"C:\Users\udayk\Downloads"

# Filename mapping to strip size suffix
def clean_filename(locality_id):
    parts = locality_id.split('.')
    if len(parts) > 1 and parts[-1].isdigit():  # Check if the last part is a size
        return '.'.join(parts[:-1]) + '.jpg'
    return locality_id

# Preprocess image with memory optimization
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_REDUCED_GRAYSCALE_2)
        if img is None:
            print(f"Warning: Failed to load image: {image_path}")
            return image_path, False
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cv2.imwrite(image_path, img)
        return image_path, True
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return image_path, False

# Perform OCR on an image
def perform_ocr(image_path):
    try:
        image_path, preprocessed = preprocess_image(image_path)
        if not preprocessed:
            return ""
        result = ocr_system.ocr(image_path)
        if isinstance(result, list):
            return " ".join(str(item) for item in result if item)
        elif isinstance(result, dict) and "text" in result:
            return " ".join(str(item) for item in result["text"] if item)
        else:
            return str(result)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return ""

# Normalize text
def normalize_text(text):
    return ''.join(c for c in text.lower() if c.isalnum() or c.isspace()).strip()

# Calculate WRR
def calculate_wrr(ocr_text, json_transcriptions):
    if not json_transcriptions:
        return 0.0
    ocr_words = set(normalize_text(ocr_text).split())
    json_words = set(normalize_text(" ".join(json_transcriptions)).split())
    correct_words = len(ocr_words & json_words)
    total_words = len(json_words)
    return (correct_words / total_words) * 100 if total_words > 0 else 0.0

# Process each JSON file
for json_file in json_files:
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"\nProcessing file: {json_file}")
            
            if not isinstance(data, dict):
                print(f"Error: {json_file} does not contain a valid dictionary structure.")
                continue
            
            for locality_id, locality_data in data.items():
                if locality_data is None:
                    print(f"Warning: Locality ID {locality_id} data is None in {json_file}. Skipping...")
                    continue
                
                filename = clean_filename(locality_id)
                size = locality_data.get("size", 0)
                regions = locality_data.get("regions", [])
                
                if not regions:
                    print(f"Warning: No regions found for Locality ID {locality_id} in {json_file}. Skipping...")
                    continue
                
                transcriptions = []
                for region in regions:
                    if isinstance(region, dict) and "region_attributes" in region:
                        transcription = region["region_attributes"].get("Transcription", "").strip()
                        if transcription:
                            transcriptions.append(transcription)
                    else:
                        print(f"Warning: Invalid region data for Locality ID {locality_id} in {json_file}. Skipping region...")
                
                if not transcriptions:
                    print(f"Warning: No valid transcriptions found for Locality ID {locality_id} in {json_file}. Skipping...")
                    continue
                
                image_path = os.path.join(image_directory, filename)
                if not os.path.exists(image_path):
                    print(f"Warning: Image {filename} not found for Locality ID {locality_id}. WRR set to 0%.")
                    ocr_text = ""
                    wrr = 0.0
                else:
                    ocr_text = perform_ocr(image_path)
                    wrr = calculate_wrr(ocr_text, transcriptions)
                
                unique_transcriptions = len(set(transcriptions))
                total_transcriptions = len(transcriptions)
                
                print(f"Locality ID: {locality_id}")
                print(f"Filename: {filename}, Size: {size} bytes")
                print(f"Total Transcriptions: {total_transcriptions}")
                print(f"Unique Transcriptions: {unique_transcriptions}")
                print(f"OCR Text: {ocr_text}")
                print(f"Transcriptions: {transcriptions}")
                print(f"Word Recognition Rate (WRR): {wrr:.2f}%")
                print("-" * 50)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {json_file}: {e}")
        except Exception as e:
            print(f"Unexpected error processing {json_file}: {e}")
    else:
        print(f"Warning: File {json_file} not found. Skipping...")

# Summary
total_files = len(json_files)
print(f"\nSummary: Processed {total_files} JSON files.")