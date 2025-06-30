import os
import json

# === CONFIG ===
output_folder = "PAKKA_OCR_OUTPUTS"
expected_counts_file = "expected_counts.json"

# === Load expected counts ===
with open(expected_counts_file, "r", encoding="utf-8") as f:
    expected_counts = json.load(f)

# === Stats variables ===
total_files = 0
total_recognized = 0
total_expected = 0

# === Process each JSON ===
for filename in sorted(os.listdir(output_folder)):
    if filename.lower().endswith(".json") and filename in expected_counts:
        json_path = os.path.join(output_folder, filename)
        
        # Load OCR results
        with open(json_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        recognized_count = sum(len(group) for group in results)
        expected_count = expected_counts[filename]
        
        total_files += 1
        total_recognized += recognized_count
        total_expected += expected_count
        
        # Print individual image result
        print(f"{filename}: {recognized_count} / {expected_count} words recognized")

# === Overall summary ===
print("\n📊 Overall Recognition Rate:")
print(f"Total files processed: {total_files}")
print(f"Total words recognized: {total_recognized}")
print(f"Total words expected: {total_expected}")

if total_expected > 0:
    overall_rate = (total_recognized / total_expected) * 100
    print(f"✅ Average recognition rate: {overall_rate:.2f}%")
else:
    print("⚠️ No valid expected counts to compute accuracy.")
