import time
import os
import gc
import torch
from IndicPhotoOCR.ocr import OCR

image_dir = '/DATA1/ocrteam/anik/splitonBSTD/4/12C_images'
all_images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
test_images = all_images[:5]  # Test 5 images

print(f"Benchmarking on {len(test_images)} images:")
for img in test_images:
    print(f" - {img}")

print("\n--- PHASE 1: Initialization & Warmup (Downloads models and loads them to GPU) ---")
ocr = OCR(device="cuda:0", identifier_lang='auto', verbose=False)

for img in test_images:
    ocr.ocr(img, batch_size=0)

print("\n--- PHASE 2: Sequential Benchmarking (batch_size=0) ---")
start = time.time()
for img in test_images:
    ocr.ocr(img, batch_size=0)
seq_time = time.time() - start
print(f"Sequential took {seq_time:.2f}s")
torch.cuda.empty_cache()
gc.collect()

print("\n--- PHASE 3: Batched Benchmarking (batch_size=32) ---")
start = time.time()
for img in test_images:
    ocr.ocr(img, batch_size=32)
bat_time = time.time() - start
print(f"Batched took {bat_time:.2f}s")
torch.cuda.empty_cache()
gc.collect()

print("\n========================================")
print("BENCHMARK RESULTS")
print("========================================")
print(f"Images tested:   {len(test_images)}")
print(f"Sequential Time: {seq_time:.2f} seconds")
print(f"Batched Time:    {bat_time:.2f} seconds")
print(f"Speedup:         {seq_time / bat_time:.2f}x")
print("========================================\n")
