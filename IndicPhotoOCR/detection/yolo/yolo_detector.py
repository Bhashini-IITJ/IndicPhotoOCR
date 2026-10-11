"""YOLO11n text detector fine-tuned on the BSTD training split.

The checkpoint is an Ultralytics YOLO11n text model fine-tuned on the official
BSTD train split only (4,737 train / 526 held-out validation images; no test
images). It predicts axis-aligned word boxes, returned here as 4-point
polygons so they plug into the existing OCR crop -> script-ID -> recognition
pipeline unchanged.

Default inference settings (imgsz=1280, conf=0.25, iou=0.7) are the ones used
for the reported BSTD end-to-end evaluation.
"""
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

model_info = {
    "yolo11n_bstd": {
        "path": "models",
        "filename": "yolo11n_bstd_finetuned.pt",
        "url": "https://github.com/Asadalk/IndicPhotoOCR/releases/download/yolo11n-bstd-v1.0/yolo11n_bstd_finetuned.pt",
        "sha256": "cb41a7f120ff086180e4db5b36001cd9ba6984b129076a6cf697c634aa3489db",
    }
}


def _verify_checksum(path, expected):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")


def download_model_from_release(model_name, save_path):
    """Download once, verifying existing and newly downloaded weights."""
    if model_name not in model_info:
        raise ValueError(f"Model '{model_name}' not found in model_info")
    model_data = model_info[model_name]
    save_path = Path(save_path).expanduser()
    if save_path.exists():
        _verify_checksum(save_path, model_data["sha256"])
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Download to a temporary file so interrupted downloads aren't cached as models.
    fd, temporary = tempfile.mkstemp(prefix=save_path.name + ".", suffix=".tmp",
                                     dir=save_path.parent)
    os.close(fd)
    try:
        print(f"Downloading model from {model_data['url']}...")
        with tqdm(desc="YOLO11n-BSTD", unit="B", unit_scale=True,
                  unit_divisor=1024) as progress:
            def report_progress(block_count, block_size, total_size):
                if total_size > 0:
                    progress.total = total_size
                downloaded = block_count * block_size
                if total_size > 0:
                    downloaded = min(downloaded, total_size)
                progress.update(max(0, downloaded - progress.n))

            urllib.request.urlretrieve(model_data["url"], temporary,
                                       reporthook=report_progress)
        print("Download complete. Verifying checkpoint SHA-256...")
        _verify_checksum(temporary, model_data["sha256"])
        os.replace(temporary, save_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


class YOLOdetector:
    def __init__(self, device="cuda:0", checkpoint_path=None, imgsz=1280,
                 conf=0.25, iou=0.7, model_name="yolo11n_bstd"):
        """Load the fine-tuned YOLO11n BSTD text detector.

        Args:
            device (str): 'cuda', 'cuda:N' or 'cpu'.
            checkpoint_path (str): Optional local checkpoint. If omitted, the
                verified release asset is downloaded and cached beside this module.
            imgsz (int): Inference image size (long side).
            conf (float): Box confidence threshold.
            iou (float): NMS IoU threshold.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "The YOLO detector needs the 'ultralytics' package: pip install ultralytics==8.3.227"
            ) from exc
        if checkpoint_path is None:
            checkpoint_path = (Path(__file__).resolve().parent /
                               model_info[model_name]["path"] /
                               model_info[model_name]["filename"])
            download_model_from_release(model_name, checkpoint_path)
        self.device = str(device)
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.model = YOLO(str(checkpoint_path))

    def detect(self, image_path):
        """Detect text boxes in an image file.

        Returns:
            dict: {"detections": [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]], ...],
                   "scores": [float, ...]}
        """
        if not os.path.isfile(image_path):
            raise ValueError(f"Failed to read the image at {image_path}")
        result = self.model.predict(image_path, device=self.device, imgsz=self.imgsz,
                                    conf=self.conf, iou=self.iou, verbose=False)[0]
        detections, scores = [], []
        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.detach().cpu().numpy().tolist()
            scores = [float(s) for s in result.boxes.conf.detach().cpu().numpy().tolist()]
            for x1, y1, x2, y2 in boxes:
                detections.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        return {"detections": detections, "scores": scores}
