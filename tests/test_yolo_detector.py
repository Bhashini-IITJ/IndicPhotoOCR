"""
Tests for IndicPhotoOCR.detection.yolo – fine-tuned YOLO11n text detectors.

Unit tests replace the `ultralytics` package with a fake module, so they need
no GPU, no network and no checkpoint. The integration test (--run-integration)
downloads/verifies the release checkpoint and runs it on a repo test image.
"""
import hashlib
import sys
import types

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class _Tensor:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values


def _fake_result(xyxy, conf):
    boxes = MagicMock()
    boxes.xyxy = _Tensor(xyxy)
    boxes.conf = _Tensor(conf)
    boxes.__len__ = lambda self: len(conf)
    result = MagicMock()
    result.boxes = boxes
    return result


@pytest.fixture
def fake_ultralytics(monkeypatch):
    """Install a fake `ultralytics` module whose YOLO() returns a MagicMock model."""
    model = MagicMock()
    module = types.ModuleType("ultralytics")
    module.YOLO = MagicMock(return_value=model)
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return module, model


@pytest.fixture
def local_checkpoint(tmp_path):
    path = tmp_path / "yolo.pt"
    path.write_bytes(b"fake-weights")
    return str(path)


class TestYOLOdetector:
    def test_uses_local_checkpoint_without_download(self, fake_ultralytics, local_checkpoint):
        module, _ = fake_ultralytics
        from IndicPhotoOCR.detection.yolo import yolo_detector
        with patch.object(yolo_detector, "download_model_from_release") as download:
            det = yolo_detector.YOLOdetector(device="cpu", checkpoint_path=local_checkpoint)
        download.assert_not_called()
        module.YOLO.assert_called_once_with(local_checkpoint)
        assert (det.imgsz, det.conf, det.iou) == (1920, 0.25, 0.7)

    def test_default_checkpoint_is_downloaded_and_verified(self, fake_ultralytics):
        from IndicPhotoOCR.detection.yolo import yolo_detector
        with patch.object(yolo_detector, "download_model_from_release") as download:
            det = yolo_detector.YOLOdetector(device="cpu")
        name, path = download.call_args[0]
        assert name == "yolo11n_bstd_mist_1920"
        assert str(path).endswith("detection/yolo/models/yolo11n_bstd_mist_1920.pt")
        assert det.imgsz == 1920

    def test_previous_bstd_model_keeps_1280(self, fake_ultralytics):
        from IndicPhotoOCR.detection.yolo import yolo_detector
        with patch.object(yolo_detector, "download_model_from_release") as download:
            det = yolo_detector.YOLOdetector(device="cpu", model_name="yolo11n_bstd")
        name, path = download.call_args[0]
        assert name == "yolo11n_bstd"
        assert str(path).endswith("detection/yolo/models/yolo11n_bstd_finetuned.pt")
        assert det.imgsz == 1280

    def test_explicit_imgsz_overrides_model_default(self, fake_ultralytics, local_checkpoint):
        from IndicPhotoOCR.detection.yolo.yolo_detector import YOLOdetector
        assert YOLOdetector(device="cpu", checkpoint_path=local_checkpoint, imgsz=640).imgsz == 640

    def test_unknown_model_name_rejected(self, fake_ultralytics):
        from IndicPhotoOCR.detection.yolo.yolo_detector import YOLOdetector
        with pytest.raises(ValueError, match="not found"):
            YOLOdetector(device="cpu", model_name="nope")

    def test_detect_returns_four_point_polygons_and_scores(self, fake_ultralytics, local_checkpoint,
                                                           synthetic_scene_image):
        _, model = fake_ultralytics
        model.predict.return_value = [_fake_result([[10, 20, 80, 40], [5, 5, 15, 9]], [0.9, 0.3])]
        from IndicPhotoOCR.detection.yolo.yolo_detector import YOLOdetector
        det = YOLOdetector(device="cpu", checkpoint_path=local_checkpoint)
        out = det.detect(synthetic_scene_image)
        assert out["detections"] == [[[10, 20], [80, 20], [80, 40], [10, 40]],
                                     [[5, 5], [15, 5], [15, 9], [5, 9]]]
        assert out["scores"] == pytest.approx([0.9, 0.3])
        _, kwargs = model.predict.call_args
        assert kwargs == {"device": "cpu", "imgsz": 1920, "conf": 0.25, "iou": 0.7, "verbose": False}

    def test_detect_handles_no_boxes(self, fake_ultralytics, local_checkpoint, synthetic_scene_image):
        _, model = fake_ultralytics
        model.predict.return_value = [_fake_result(np.zeros((0, 4)), [])]
        from IndicPhotoOCR.detection.yolo.yolo_detector import YOLOdetector
        out = YOLOdetector(device="cpu", checkpoint_path=local_checkpoint).detect(synthetic_scene_image)
        assert out == {"detections": [], "scores": []}

    def test_missing_image_raises(self, fake_ultralytics, local_checkpoint, tmp_path):
        from IndicPhotoOCR.detection.yolo.yolo_detector import YOLOdetector
        det = YOLOdetector(device="cpu", checkpoint_path=local_checkpoint)
        with pytest.raises(ValueError, match="Failed to read"):
            det.detect(str(tmp_path / "missing.jpg"))

    def test_missing_ultralytics_gives_install_hint(self, monkeypatch, local_checkpoint):
        monkeypatch.setitem(sys.modules, "ultralytics", None)
        from IndicPhotoOCR.detection.yolo.yolo_detector import YOLOdetector
        with pytest.raises(ImportError, match="ultralytics==8.3.227"):
            YOLOdetector(device="cpu", checkpoint_path=local_checkpoint)


class TestChecksum:
    def test_existing_file_with_wrong_hash_rejected(self, tmp_path, monkeypatch):
        from IndicPhotoOCR.detection.yolo import yolo_detector
        path = tmp_path / "w.pt"
        path.write_bytes(b"tampered")
        monkeypatch.setitem(yolo_detector.model_info, "t", {"url": "x", "sha256": "0" * 64})
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            yolo_detector.download_model_from_release("t", path)

    def test_existing_file_with_right_hash_accepted(self, tmp_path, monkeypatch):
        from IndicPhotoOCR.detection.yolo import yolo_detector
        path = tmp_path / "w.pt"
        path.write_bytes(b"weights")
        monkeypatch.setitem(yolo_detector.model_info, "t",
                            {"url": "x", "sha256": hashlib.sha256(b"weights").hexdigest()})
        yolo_detector.download_model_from_release("t", path)  # no download, no error

    def test_unknown_model_rejected(self, tmp_path):
        from IndicPhotoOCR.detection.yolo import yolo_detector
        with pytest.raises(ValueError, match="not found"):
            yolo_detector.download_model_from_release("nope", tmp_path / "w.pt")


@pytest.mark.integration
class TestYOLOIntegration:
    def test_real_model_detects_text(self, repo_scene_image):
        torch = pytest.importorskip("torch")
        from IndicPhotoOCR.detection.yolo.yolo_detector import YOLOdetector
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        out = YOLOdetector(device=device).detect(repo_scene_image)
        assert len(out["detections"]) > 0
        assert len(out["detections"]) == len(out["scores"])
        assert all(len(p) == 4 for p in out["detections"])
