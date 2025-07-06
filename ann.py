from IndicPhotoOCR.ocr import OCR

# Initialize OCR system
ocr_system = OCR(verbose=True, identifier_lang="auto", device="cpu")
words = ocr_sysrem.ocr("C:\Users\udayk\IndicPhotoOCR\test_images\cropped_image\image_141_10.jpg")
print(words)