import torch
import cv2
import numpy as np
from IndicPhotoOCR.detection.textbpn.network.textnet import TextNet
from IndicPhotoOCR.detection.textbpn.cfglib.config import config as cfg
import warnings
import os
import requests
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

model_info = {
    "textbpnpp": {
        "path": "models/TextBPN_resnet50_300.pth",
        "url" : "https://github.com/Bhashini-IITJ/SceneTextDetection/releases/download/TextBPN%2B%2B/TextBPN_resnet50_300.pth",
    },
    "textbpnpp_deformable": {
        "path":"models/TextBPN_deformable_resnet50_300.pth",
        "url": "https://github.com/Bhashini-IITJ/SceneTextDetection/releases/download/TextBPN%2B%2B/TextBPN_deformable_resnet50_300.pth",
    },
    "textbpn_resnet18" : {
        "path":"models/TextBPN_resnet18_300.pth",
        "url": "https://github.com/Bhashini-IITJ/SceneTextDetection/releases/download/TextBPN%2B%2B/TextBPN_resnet18_300.pth",

    }
}
        # Ensure model file exists; download directly if not
def ensure_model(model_name):
    """
    Ensure that the specified model is available locally. If not, download it.

    Args:
        model_name (str): Name of the model (e.g., "textbpnpp").

    Returns:
        str: Path to the model file.
    """
    model_path = model_info[model_name]["path"]
    url = model_info[model_name]["url"]
    root_model_dir = "IndicPhotoOCR/detection/textbpn"
    model_path = os.path.join(root_model_dir, model_path)
    
    if not os.path.exists(model_path):
        print(f"Model not found locally. Downloading {model_name} from {url}...")
        
        # Start the download with a progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        os.makedirs(f"{root_model_dir}/models", exist_ok=True)
        
        with open(model_path, "wb") as f, tqdm(
                desc=model_name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                bar.update(len(data))

        print(f"Downloaded model for {model_name}.")
        
    return model_path

class TextBPNpp_detector:
    def __init__(self, model_name="textbpnpp", backbone="resnet50", device="cpu"):
        """
        Initialize the TextBPN++ model for text detection.

        Args:
            model_name (str): Name of the model (default: "textbpnpp").
            backbone (str): Backbone network architecture (default: "resnet50").
            device (str): Device for model inference, "cpu" or "cuda" (default: "cpu").
        """
        self.model_path = ensure_model(model_name)
        self.device = torch.device(device)
        self.model = TextNet(is_training=False, backbone=backbone)
        self.model.load_model(self.model_path)
        self.model.eval()
        self.model.to(self.device)

    @staticmethod
    def to_device(tensor, device):
        """
        Move a tensor to the specified device.

        Args:
            tensor (torch.Tensor): Tensor to be moved.
            device (str): Target device ("cpu" or "cuda").

        Returns:
            torch.Tensor: Tensor moved to the specified device.
        """
        return tensor.to(device, non_blocking=True)

    @staticmethod
    def pad_image(image, stride=32):
        """
        Pad an image to ensure its dimensions are multiples of a given stride.

        Args:
            image (np.ndarray): Input image as a NumPy array.
            stride (int): Stride value for padding (default: 32).

        Returns:
            tuple: Padded image and original dimensions before padding.
        """
        h, w = image.shape[:2]
        new_h = (h + stride - 1) // stride * stride
        new_w = (w + stride - 1) // stride * stride
        padded_image = cv2.copyMakeBorder(
            image, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return padded_image, (h, w)

    @staticmethod
    def rescale_result(image, bbox_contours, original_height, original_width):
        """
        Rescale bounding box coordinates back to the original image dimensions.

        Args:
            image (np.ndarray): Processed image after resizing.
            bbox_contours (list): List of bounding box contours.
            original_height (int): Original image height before resizing.
            original_width (int): Original image width before resizing.

        Returns:
            list: Rescaled bounding box contours.
        """
        contours = []
        for cont in bbox_contours:
            cont[:, 0] = (cont[:, 0] * original_width / image.shape[1]).astype(int)
            cont[:, 1] = (cont[:, 1] * original_height / image.shape[0]).astype(int)
            contours.append(cont)
        return contours

    def detect(self, image_path):
        """
        Detect text regions in the input image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: Detected text bounding boxes in the format:
                  {"detections": [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]}
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read the image at {image_path}")

        padded_image, original_size = self.pad_image(image)
        padded_tensor = (
            torch.from_numpy(padded_image).permute(2, 0, 1).float() / 255.0
        ).unsqueeze(0)  # Convert to tensor and add batch dimension

        cfg.test_size = [padded_image.shape[0], padded_image.shape[1]]

        input_dict = {"img": self.to_device(padded_tensor, self.device)}
        with torch.no_grad():
            output_dict = self.model(input_dict, padded_image.shape)

        contours = output_dict["py_preds"][-1].int().cpu().numpy()
        contours = self.rescale_result(image, contours, *original_size)

        bbox_result_dict = {"detections": []}
        for contour in contours:
            # x_min, y_min = np.min(contour, axis=0)
            # x_max, y_max = np.max(contour, axis=0)
            # bbox_result_dict["detections"].append([x_min, y_min, x_max, y_max])
            bbox_result_dict["detections"].append(contour.tolist())

        return bbox_result_dict

    def visualize_detections(self, image_path, bbox_result_dict, output_path="output.png"):
        """
        Visualize and save detected text bounding boxes on an image.

        Args:
            image_path (str): Path to the input image.
            bbox_result_dict (dict): Dictionary containing detected text bounding boxes.
                                     Format: {"detections": [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]}.
            output_path (str): Path to save the visualized image (default: "output.png").

        Returns:
            None
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read the image at {image_path}")
        
        # Draw each detection
        for bbox in bbox_result_dict.get("detections", []):
            points = np.array(bbox, dtype=np.int32)  # Convert to numpy array
            cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Display or save the visualized image
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Visualization saved to {output_path}")
        else:
            cv2.imshow("Detections", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Text detection using EAST model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on, e.g., "cpu" or "cuda"')
    parser.add_argument('--model_name', type=str, required=True, help='Path to the model checkpoint file')
    args = parser.parse_args()



    # model_path = "/DATA1/ocrteam/anik/git/IndicPhotoOCR/IndicPhotoOCR/detection/textbpn/models/TextBPN_resnet50_300.pth"
    # image_path = "/DATA1/ocrteam/anik/splitonBSTD/detection/D/image_542.jpg"

    detector = TextBPNpp_detector(args.model_name, device="cpu")
    result = detector.detect(args.image_path)
    print(result)
    # detector.visualize_detections(image_path, result)

    # python -m IndicPhotoOCR.detection.textbpn.textbpnpp_detector \
    # --image_path /DATA1/ocrteam/anik/splitonBSTD/detection/D/image_542.jpg \
    # --model_name textbpnpp