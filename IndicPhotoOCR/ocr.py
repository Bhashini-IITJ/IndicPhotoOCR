import sys
import os
import torch
from PIL import Image
import cv2
import numpy as np


# from IndicPhotoOCR.detection.east_detector import EASTdetector
# from IndicPhotoOCR.script_identification.CLIP_identifier import CLIPidentifier
from IndicPhotoOCR.script_identification.vit.vit_infer import VIT_identifier
from IndicPhotoOCR.recognition.parseq_recogniser import PARseqrecogniser
import IndicPhotoOCR.detection.east_config as cfg
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
    def __init__(self, device='cuda:0', identifier_lang='hindi', verbose=False):
        # self.detect_model_checkpoint = detect_model_checkpoint
        self.device = device
        self.verbose = verbose
        # self.image_path = image_path
        # self.detector = EASTdetector()
        self.detector = TextBPNpp_detector(device=self.device)
        self.recogniser = PARseqrecogniser()
        # self.identifier = CLIPidentifier()
        self.identifier = VIT_identifier()
        self.indentifier_lang = identifier_lang

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
        return self.identifier.identify(cropped_path, self.indentifier_lang, self.device)
        
    def crop_and_identify_script(self, image, bbox):
        """
        Crop a text area from the image and identify its script language.

        Args:
            image (PIL.Image): The full image.
            bbox (list): List of four corner points, each a [x, y] pair.

        Returns:
            str: Identified script language.
            str: Path to the cropped image.
        """
        # # Extract x and y coordinates from the four corner points
        # x_coords = [point[0] for point in bbox]
        # y_coords = [point[1] for point in bbox]

        # # Get the bounding box coordinates (min and max)
        # x_min, y_min = min(x_coords), min(y_coords)
        # x_max, y_max = max(x_coords), max(y_coords)

        # # Crop the image based on the bounding box
        # cropped_image = image.crop((x_min, y_min, x_max, y_max))
        
        # Convert list to numpy array
        points = np.array(bbox, np.int32)

        # Create a mask to extract the polygonal region
        mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        # Apply the mask to extract the region
        cropped = cv2.bitwise_and(image, image, mask=mask)

        # Find bounding rectangle to crop the region
        x, y, w, h = cv2.boundingRect(points)
        cropped_bbox = cropped[y:y+h, x:x+w]

        root_image_dir = "IndicPhotoOCR/script_identification"
        os.makedirs(f"{root_image_dir}/images", exist_ok=True)
        # Temporarily save the cropped image to pass to the script model
        cropped_path = f'{root_image_dir}/images/temp_crop_{x}_{y}.jpg'
        # # Convert RGBA to RGB so as to save them
        # if cropped_image.mode == 'RGBA':
        #     cropped_image = cropped_image.convert('RGB')
        #     cropped_image.save(cropped_path)
        # else:
        #     cropped_image.save(cropped_path)
        cv2.imwrite(cropped_path, cropped_bbox)

        # Predict script language, here we assume "hindi" as the model name
        if self.verbose:
            print("Identifying script for the cropped area...")
        script_lang = self.identify(cropped_path)  # Use "hindi" as the model name
        # print(script_lang)

        # Clean up temporary file
        # os.remove(cropped_path)

        return script_lang, cropped_path

    def recognise(self, cropped_image_path, script_lang):
        """
        Recognize text in a cropped image using the identified script model.
        
        Args:
            cropped_image_path (str): Path to the cropped image.
            script_lang (str): Identified script language.
        
        Returns:
            str: Recognized text.
        """
        """Recognize text in a cropped image area using the identified script."""
        if self.verbose:
            print("Recognizing text in detected area...")
        recognized_text = self.recogniser.recognise(script_lang, cropped_image_path, script_lang, self.verbose, self.device)
        # print(recognized_text)
        return recognized_text

    def ocr(self, image_path):
        """
        Perform end-to-end OCR: detect text, identify script, and recognize text.
        
        Args:
            image_path (str): Path to the input image.
        
        Returns:
            dict: Recognized text with corresponding bounding boxes.
        """
        recognized_texts = {}
        recognized_words = []
        image = cv2.imread(image_path)
        
        # Run detection
        detections = self.detect(image_path)

        # Process each detected text area
        # for bbox in detections:
            # # Crop and identify script language
            # script_lang, cropped_path = self.crop_and_identify_script(image, bbox)

            # # Check if the script language is valid
            # if script_lang:

            #     # Recognize text
            #     recognized_word = self.recognise(cropped_path, script_lang)
            #     recognized_words.append(recognized_word)

            #     if self.verbose:
            #         print(f"Recognized word: {recognized_word}")


        for id, bbox in enumerate(detections):
            # Identify the script and crop the image to this region
            script_lang, cropped_path = self.crop_and_identify_script(image, bbox)

            # Calculate bounding box coordinates
            x1 = min([bbox[i][0] for i in range(len(bbox))])
            y1 = min([bbox[i][1] for i in range(len(bbox))])
            x2 = max([bbox[i][0] for i in range(len(bbox))])
            y2 = max([bbox[i][1] for i in range(len(bbox))])

            if script_lang:
                recognized_text = self.recognise(cropped_path, script_lang)
                recognized_texts[f"img_{id}"] = {"txt": recognized_text, "bbox": [x1, y1, x2, y2]}

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