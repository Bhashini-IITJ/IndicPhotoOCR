import sys
import os
import torch
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile


from IndicPhotoOCR.detection.east.east_detector import EASTdetector
# from IndicPhotoOCR.script_identification.CLIP_identifier import CLIPidentifier
from IndicPhotoOCR.script_identification.vit.vit_infer import VIT_identifier
from IndicPhotoOCR.recognition.parseq_recogniser import PARseqrecogniser
import IndicPhotoOCR.detection.east.east_config as cfg
from IndicPhotoOCR.detection.textbpn.textbpnpp_detector import TextBPNpp_detector

from IndicPhotoOCR.utils.helper import detect_para


class OCR:
    """
    Optical Character Recognition (OCR) pipeline for text detection, script identification,
    and text recognition.

    Args:
        device (str): Device to use for inference ('cuda:0' or 'cpu').
        identifier_lang (str): Default script identifier model to use.
            Valid options: ['hindi', 'bengali', 'tamil', 'telugu', 'malayalam', 'kannada',
                            'gujarati', 'marathi', 'punjabi', 'odia', 'assamese', 'urdu', 'meitei']
        verbose (bool): Whether to print detailed processing information.
    """
    def __init__(self, device='cuda:0', identifier_lang='hindi', verbose=False, detector='textbpn'):
        # self.detect_model_checkpoint = detect_model_checkpoint
        # Original device string (e.g. 'cuda', 'cuda:0', or 'cpu')
        self.device = device
        # Torch device object for PyTorch models
        try:
            self.torch_device = torch.device(device)
        except Exception:
            # Fallback: default to cpu
            self.torch_device = torch.device('cpu')
        # Transformers pipeline expects an int (GPU index) or -1 for CPU
        if isinstance(device, str) and 'cuda' in device:
            # pick GPU 0 by default when user passes 'cuda'
            self.pipeline_device = 0
        elif isinstance(device, str) and device.startswith('cuda:'):
            try:
                self.pipeline_device = int(device.split(':', 1)[1])
            except Exception:
                self.pipeline_device = 0
        else:
            self.pipeline_device = -1
        self.verbose = verbose
        self.detector_name = detector.lower()
        # self.image_path = image_path
        # detectors expect a string device (they call torch.device internally)
        if self.detector_name == "east":
            self.detector = EASTdetector(device=str(self.torch_device))
        elif self.detector_name in {"textbpn", "textbpnpp"}:
            self.detector = TextBPNpp_detector(device=str(self.torch_device))
        else:
            raise ValueError("detector must be one of: 'east', 'textbpn', 'textbpnpp'")
        self.recogniser = PARseqrecogniser()
        # self.identifier = CLIPidentifier()
        self.identifier = VIT_identifier()
        self.indentifier_lang = identifier_lang
        # expose devices for downstream calls: pipeline (int) and torch (torch.device)
        self._pipeline_device = self.pipeline_device
        self._torch_device = self.torch_device

    # def detect(self, image_path, detect_model_checkpoint=cfg.checkpoint):
    #     """Run the detection model to get bounding boxes of text areas."""

    #     if self.verbose:
    #         print("Running text detection...")
    #     detections = self.detector.detect(image_path, detect_model_checkpoint, self.device)
    #     # print(detections)
    #     return detections['detections']
    def detect(self, image_path):
        """
        Detect text regions in the input image.
        
        Args:
            image_path (str): Path to the image file.
        
        Returns:
            list: Detected text bounding boxes.
        """
        self.detections = self.detector.detect(image_path)
        return self.detections['detections']

    def visualize_detection(self, image_path, detections, save_path=None, show=False):
        """
        Visualize and optionally save the detected text bounding boxes on an image.
        
        Args:
            image_path (str): Path to the image file.
            detections (list): List of bounding boxes.
            save_path (str, optional): Path to save the output image.
            show (bool): Whether to display the image.
        """
        # Default save path if none is provided
        default_save_path = "test.png"
        path_to_save = save_path if save_path is not None else default_save_path

        # Get the directory part of the path
        directory = os.path.dirname(path_to_save)
        
        # Check if the directory exists, and create it if it doesn’t
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

        # Read the image and draw bounding boxes
        image = cv2.imread(image_path)
        for box in detections:
            # Convert list of points to a numpy array with int type
            points = np.array(box, np.int32)

            # Compute the top-left and bottom-right corners of the bounding box
            x_min = np.min(points[:, 0])
            y_min = np.min(points[:, 1])
            x_max = np.max(points[:, 0])
            y_max = np.max(points[:, 1])

            # Draw the rectangle
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=3)

        # Show the image if 'show' is True
        if show:
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()

        # Save the annotated image
        cv2.imwrite(path_to_save, image)
        if self.verbose:
            print(f"Image saved at: {path_to_save}")
        
    def identify(self, cropped_path):
        return self.identifier.identify(cropped_path, self.indentifier_lang, self._pipeline_device)
        
    def crop_bbox(self, image, bbox):
        h_img, w_img = image.shape[:2]
        points = np.array(bbox, np.int32)

        # Clamp polygon points to image bounds to avoid negative coordinates
        points[:, 0] = np.clip(points[:, 0], 0, w_img - 1)
        points[:, 1] = np.clip(points[:, 1], 0, h_img - 1)

        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        cropped = cv2.bitwise_and(image, image, mask=mask)
        x, y, w, h = cv2.boundingRect(points)

        # If bounding rect is empty, return None so callers can skip it
        if w <= 0 or h <= 0:
            if self.verbose:
                print("Warning: empty crop for bbox:", bbox)
            return None

        cropped_bbox = cropped[y:y+h, x:x+w]
        if cropped_bbox.size == 0:
            if self.verbose:
                print("Warning: cropped image is empty for bbox:", bbox)
            return None

        fd, cropped_path = tempfile.mkstemp(suffix=".jpg", prefix=f"crop_{x}_{y}_")
        os.close(fd)
        success = cv2.imwrite(cropped_path, cropped_bbox)
        if not success:
            if self.verbose:
                print("Failed to write cropped image to", cropped_path)
            try:
                os.remove(cropped_path)
            except Exception:
                pass
            return None

        return cropped_path

    def crop_and_identify_script(self, image, bbox):
        cropped_path = self.crop_bbox(image, bbox)
        if cropped_path is None:
            if self.verbose:
                print("Skipping bbox due to empty crop:", bbox)
            return None, None

        if self.verbose:
            print("Identifying script for the cropped area...")
        script_lang = self.identifier.identify(cropped_path, "auto", self._pipeline_device)
        return script_lang, cropped_path

    def recognise(self, cropped_image_path, script_lang, return_confidence=False):
        """
        Recognize text in a cropped image using the identified script model.
        
        Args:
            cropped_image_path (str): Path to the cropped image.
            script_lang (str): Identified script language.
        
        Returns:
            str or tuple: Recognized text. If return_confidence is True, returns (text, confidence).
        """
        """Recognize text in a cropped image area using the identified script."""
        if self.verbose:
            print("Recognizing text in detected area...")
        result = self.recogniser.recognise(script_lang, cropped_image_path, script_lang, self.verbose, self._torch_device, return_confidence=return_confidence)
        # print(recognized_text)
        return result

    def ocr(self, image_path, batch_size=0):
        """
        Perform end-to-end OCR: detect text, identify script, and recognize text.
        
        Args:
            image_path (str): Path to the input image.
            batch_size (int): Size of batches for script identification and recognition models. If 0, uses sequential execution.
        
        Returns:
            dict: Recognized text with corresponding bounding boxes.
        """
        recognized_texts = {}
        recognized_words = []
        image = cv2.imread(image_path)
        
        # Run detection
        detections = self.detect(image_path)

        if batch_size > 0:
            cropped_paths = []
            for id, bbox in enumerate(detections):
                cropped_path = self.crop_bbox(image, bbox)
                cropped_paths.append(cropped_path)
                
            if self.verbose:
                print(f"Identifying script languages in batch (size={len(cropped_paths)})...")
                
            if len(cropped_paths) > 0:
                script_langs = self.identifier.identify_batch(cropped_paths, "auto", self._pipeline_device, batch_size=batch_size)
                
                langs_to_crops = {}
                for id, (lang, path) in enumerate(zip(script_langs, cropped_paths)):
                    if lang not in langs_to_crops:
                        langs_to_crops[lang] = []
                    langs_to_crops[lang].append((id, path))
                    
                for lang, items in langs_to_crops.items():
                    paths = [item[1] for item in items]
                    ids = [item[0] for item in items]
                    if self.verbose:
                        print(f"Recognizing {len(paths)} {lang} crops in batch...")
                    results = self.recogniser.recognise_batch(lang, paths, lang, self.verbose, self._torch_device, return_confidence=True, batch_size=batch_size)
                    
                    for (id, (text, conf)) in zip(ids, results):
                        bbox = detections[id]
                        x1 = min([bbox[i][0] for i in range(len(bbox))])
                        y1 = min([bbox[i][1] for i in range(len(bbox))])
                        x2 = max([bbox[i][0] for i in range(len(bbox))])
                        y2 = max([bbox[i][1] for i in range(len(bbox))])
                        
                        recognized_texts[f"img_{id}"] = {"txt": text, "bbox": [x1, y1, x2, y2], "confidence": conf}
            
            for path in cropped_paths:
                if os.path.exists(path):
                    os.remove(path)
                    
            return detect_para(recognized_texts)

        # Original Sequential Execution
        for id, bbox in enumerate(detections):
            # Identify the script and crop the image to this region
            script_lang, cropped_path = self.crop_and_identify_script(image, bbox)

            # Calculate bounding box coordinates
            x1 = min([bbox[i][0] for i in range(len(bbox))])
            y1 = min([bbox[i][1] for i in range(len(bbox))])
            x2 = max([bbox[i][0] for i in range(len(bbox))])
            y2 = max([bbox[i][1] for i in range(len(bbox))])

            if script_lang:
                recognized_text, confidence = self.recognise(cropped_path, script_lang, return_confidence=True)
                recognized_texts[f"img_{id}"] = {"txt": recognized_text, "bbox": [x1, y1, x2, y2], "confidence": confidence}

            # Clean up the temporary crop file now that we are done with it
            if os.path.exists(cropped_path):
                os.remove(cropped_path)

        return detect_para(recognized_texts)
        # return recognized_words

if __name__ == '__main__':
    # detect_model_checkpoint = 'bharatSTR/East/tmp/epoch_990_checkpoint.pth.tar'
    sample_image_path = 'test_images/image_88.jpg'
    cropped_image_path = 'test_images/cropped_image/image_141_0.jpg'

    ocr = OCR(device="cuda", identifier_lang='auto', verbose=False)

    # detections = ocr.detect(sample_image_path)
    # print(detections)

    # ocr.visualize_detection(sample_image_path, detections)

    # recognition = ocr.recognise(cropped_image_path, "hindi")
    # print(recognition)

    recognised_words = ocr.ocr(sample_image_path)
    print(recognised_words)
