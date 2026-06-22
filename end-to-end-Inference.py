import os
import json
from tqdm import tqdm
from IndicPhotoOCR.ocr import OCR 
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="End-to-end OCR inference with GPU/CPU fallback",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
    Examples:
    python end-to-end-Inference.py --path /path/to/images --output results.json --exceptions errors.log --retries 3
    python end-to-end-Inference.py -p ./images -o predictions.json -e exceptions.log -r 2
        """
    )

parser.add_argument(
    '--path', '-p',
    type=str,
    required=True,
    help='Path to directory containing images'
)
parser.add_argument(
    '--output', '-o',
    type=str,
    default='IndicPhotoOCR_predictions.json',
    help='Output JSON file for predictions (default: IndicPhotoOCR_predictions.json)'
)
parser.add_argument(
    '--exceptions', '-e',
    type=str,
    default='exceptions.log',
    help='Log file for exceptions (default: exceptions.log)'
)
parser.add_argument(
    '--retries', '-r',
    type=int,
    default=2,
    help='Maximum number of retry attempts (default: 2) if GPU runs into OOM'
)

args = parser.parse_args()

# Configuration
path = args.path
output_json_file = args.output
exception_file = args.exceptions
MAX_RETRIES = args.retries

# Validate path exists
if not os.path.isdir(path):
    print(f"Error: Image directory not found: {path}")
    exit(1)
# Initialize OCR Instances
print("Initializing primary GPU OCR instance...")
ocr_gpu = OCR(identifier_lang="auto", device="cuda", verbose=False, detector="east")

print("Initializing fallback CPU OCR instance for retries...")
ocr_cpu = OCR(identifier_lang="auto", device="cpu", verbose=False, detector="east")

results = {}
exception_images = []


# Helper function to process an individual image

def process_single_image(image_name, ocr_instance):
    image_path = os.path.join(path, image_name)
    try:
        # Expecting list of (list of texts, bbox)
        detected_words = ocr_instance.ocr(image_path, batch_size=32 if ocr_instance.device == "cuda" else 0)  
        
        polygon_dict = {}
        polygon_idx = 1

        for text_list in detected_words:
            for text in text_list:
                polygon_dict[f"polygon_{polygon_idx}"] = {
                    "text": text
                }
                polygon_idx += 1

        results[image_name] = polygon_dict
        return True  
    except Exception as e:
        return False  


# Pass 1: Full Dataset Scan (Using GPU)

all_images = os.listdir(path)
print(f"\n--- Starting Pass 1: Processing {len(all_images)} images on GPU ---")

for image in tqdm(all_images):
    success = process_single_image(image, ocr_gpu)
    if not success:
        exception_images.append(image)


# Pass 2+: Rerun logic for Exceptions Only (Using CPU)

retry_count = 1
while exception_images and retry_count <= MAX_RETRIES:
    print(f"\n--- Starting Retry Pass {retry_count}: Re-processing {len(exception_images)} exceptions on CPU ---")
    
    still_failing = []
    for image in tqdm(exception_images):
        # Forwarding the image to the CPU OCR instance
        success = process_single_image(image, ocr_cpu)
        if not success:
            still_failing.append(image)
            
    # Update exception tracker with the remaining failures
    exception_images = still_failing
    retry_count += 1


# File IO Saving

print(f"\n--- Processing Completed. Writing output logs ---")

# Save combined successful results to JSON
with open(output_json_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

# Save final persistent exceptions (if any left after CPU passes)
with open(exception_file, 'w') as f:
    for img in exception_images:
        f.write(img + '\n')

print(f"Final saved results count: {len(results)}")
print(f"Final persistent exceptions count: {len(exception_images)}")