import json
import os

# List of specific JSON file paths
json_files = [
    r"C:\Users\udayk\IndicPhotoOCR\test_data1.json",
    r"C:\Users\udayk\IndicPhotoOCR\test_data2.json",
    r"C:\Users\udayk\IndicPhotoOCR\test_data3.json",
    r"C:\Users\udayk\IndicPhotoOCR\test_data4.json",
    r"C:\Users\udayk\IndicPhotoOCR\test_data5.json"
]

# Optional: Dictionary for ground truth
ground_truth = {
    "locality_21.jpeg9426": {"firstcry": 1, ".com": 1},
    "locality_22.jpeg10412": {"CHANDA": 1, "MAMA": 1},
    # Add more as needed
}

# Function to calculate Word Recognition Rate (WRR)
def calculate_wrr(recognized_text, ground_truth_text):
    if not ground_truth_text:
        return 0.0
    recognized_words = set(recognized_text.lower().split())
    ground_truth_words = set(ground_truth_text.lower().split())
    correct_words = len(recognized_words & ground_truth_words)
    total_words = len(ground_truth_words)
    return (correct_words / total_words) * 100 if total_words > 0 else 0.0

# Process each JSON file
for json_file in json_files:
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:  # Explicitly use utf-8 encoding
                data = json.load(f)
            
            print(f"\nProcessing file: {json_file}")
            
            for locality_id, locality_data in data.items():
                filename = locality_data["filename"]
                size = locality_data["size"]
                transcriptions = []
                
                for region in locality_data["regions"]:
                    transcription = region["region_attributes"]["Transcription"]
                    transcriptions.append(transcription)
                
                unique_transcriptions = len(set(transcriptions))
                total_transcriptions = len(transcriptions)
                
                wrr = 0.0
                if locality_id in ground_truth:
                    wrr = calculate_wrr(" ".join(transcriptions), " ".join(ground_truth[locality_id].keys()))
                
                print(f"Locality ID: {locality_id}")
                print(f"Filename: {filename}, Size: {size} bytes")
                print(f"Total Transcriptions: {total_transcriptions}")
                print(f"Unique Transcriptions: {unique_transcriptions}")
                print(f"Word Recognition Rate (WRR): {wrr:.2f}%")
                print(f"Transcriptions: {transcriptions}")
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