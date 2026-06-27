import os
import torch
import cv2
import numpy as np
import time
import warnings


import IndicPhotoOCR.detection.east.east_config as cfg
from IndicPhotoOCR.detection.east.east_utils import ModelManager
from IndicPhotoOCR.detection.east.east_model import East
import IndicPhotoOCR.detection.east.east_utils as utils

# Suppress warnings
warnings.filterwarnings("ignore")

class EASTdetector:
    def __init__(self, model_name="east", model_path=None, device="cpu"):
        self.device = torch.device(device)
        self.model_path = model_path or cfg.checkpoint
        self.model_manager = ModelManager()
        self.model_manager.ensure_model(model_name)
        self.model = self._load_model()

    @staticmethod
    def _strip_module_prefix(state_dict):
        return {
            key.replace("module.", "", 1) if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }

    def _load_model(self):
        model = East()
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )
        state_dict = self._strip_module_prefix(checkpoint["state_dict"])
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def detect(self, image_path):
        # Load image
        im = cv2.imread(image_path)
        if im is None:
            raise ValueError(f"Failed to read the image at {image_path}")

        # Resize image and convert to tensor format
        im_resized, (ratio_h, ratio_w) = utils.resize_image(im)
        im_resized = im_resized.astype(np.float32).transpose(2, 0, 1)
        im_tensor = torch.from_numpy(im_resized).unsqueeze(0).to(self.device)

        # Inference
        timer = {'net': 0, 'restore': 0, 'nms': 0}
        start = time.time()
        with torch.no_grad():
            score, geometry = self.model(im_tensor)
        timer['net'] = time.time() - start

        # Process output
        score = score.permute(0, 2, 3, 1).data.cpu().numpy()
        geometry = geometry.permute(0, 2, 3, 1).data.cpu().numpy()
        
        # Detect boxes
        boxes, timer = utils.detect(
            score_map=score, geo_map=geometry, timer=timer,
            score_map_thresh=cfg.score_map_thresh, box_thresh=cfg.box_thresh,
            nms_thres=cfg.nms_thres
        )
        bbox_result_dict = {'detections': []}

        # Parse detected boxes and adjust coordinates
        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h
            for box in boxes:
                box = utils.sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                bbox_result_dict['detections'].append([
                    [int(coord[0]), int(coord[1])] for coord in box
                ])

        return bbox_result_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Text detection using EAST model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on, e.g., "cpu" or "cuda"')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to the model checkpoint file')
    args = parser.parse_args()

    # Run prediction and get results as dictionary
    east = EASTdetector(model_path=args.model_checkpoint, device=args.device)
    detection_result = east.detect(args.image_path)
    print(detection_result)
