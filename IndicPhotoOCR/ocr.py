import sys
import os
import torch
import cv2
import numpy as np

from PIL import Image
from IndicPhotoOCR.script_identification.vit.vit_infer import VIT_identifier
from IndicPhotoOCR.recognition.parseq_recogniser import PARseqrecogniser
from IndicPhotoOCR.detection.textbpn.textbpnpp_detector import TextBPNpp_detector
from IndicPhotoOCR.utils.helper import detect_para


class OCR:
    def __init__(self, device='cuda:0', identifier_lang='hindi', verbose=False):
        self.device = device
        self.verbose = verbose
        self.detector = TextBPNpp_detector(device=self.device)
        self.recogniser = PARseqrecogniser()
        self.identifier = VIT_identifier()
        self.indentifier_lang = identifier_lang

    def detect(self, image_path):
        self.detections = self.detector.detect(image_path)
        return self.detections['detections']

    def identify(self, cropped_path):
        return self.identifier.identify(cropped_path, self.indentifier_lang, self.device)

    def crop_and_identify_script(self, image, bbox):
        points = np.array(bbox, np.int32)

        # Create a mask to extract the polygonal region
        mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        # Apply the mask to extract the region
        cropped = cv2.bitwise_and(image, image, mask=mask)

        # Find bounding rectangle to crop the region
        x, y, w, h = cv2.boundingRect(points)
        cropped_bbox = cropped[y:y+h, x:x+w]

        if x > x + w or y > y + h:
            raise ValueError(f"Invalid crop region from bbox: {bbox}")

        os.makedirs("IndicPhotoOCR/script_identification/images", exist_ok=True)
        cropped_path = f'IndicPhotoOCR/script_identification/images/temp_crop_{x}_{y}.jpg'
        cv2.imwrite(cropped_path, cropped_bbox)

        if self.verbose:
            print("Identifying script for the cropped area...")
        script_lang = self.identify(cropped_path)
        return script_lang, cropped_path

    def recognise(self, cropped_image_path, script_lang):
        if self.verbose:
            print("Recognizing text in detected area...")

        # Normalize language name
        script_lang = script_lang.lower()

        # 🔥 Robust mapping to handle script vs language differences
        script_name_map = {
            "devanagari": "hindi",
            "marathi": "hindi",
            "latin": "english"
            # add more mappings here if needed
        }
        mapped_script = script_name_map.get(script_lang, script_lang)
        if mapped_script != script_lang and self.verbose:
            print(f"⚠️ Mapping script '{script_lang}' to '{mapped_script}' for recognizer.")
        script_lang = mapped_script

        recognized_text = self.recogniser.recognise(
            script_lang, cropped_image_path, script_lang, self.verbose, self.device
        )
        return recognized_text

    def ocr(self, image_path):
        recognized_texts = {}
        image = cv2.imread(image_path)
        detections = self.detect(image_path)

        for id, bbox in enumerate(detections):
            try:
                script_lang, cropped_path = self.crop_and_identify_script(image, bbox)

                x1 = min(pt[0] for pt in bbox)
                y1 = min(pt[1] for pt in bbox)
                x2 = max(pt[0] for pt in bbox)
                y2 = max(pt[1] for pt in bbox)

                if script_lang:
                    recognized_text = self.recognise(cropped_path, script_lang)
                    recognized_texts[f"img_{id}"] = {
                        "txt": recognized_text,
                        "bbox": [x1, y1, x2, y2],
                        "script": script_lang
                    }

                    if self.verbose:
                        print(f"✅ Polygon img_{id} ({script_lang}): {recognized_text}")

            except Exception as e:
                print(f"⚠️ Skipping polygon img_{id} due to error: {e}")

        return detect_para(recognized_texts)


if __name__ == '__main__':
    sample_image_path = 'test_images/image_88.jpg'
    ocr = OCR(device="cpu", identifier_lang='auto', verbose=True)
    recognised_words = ocr.ocr(sample_image_path)
    print(recognised_words)


