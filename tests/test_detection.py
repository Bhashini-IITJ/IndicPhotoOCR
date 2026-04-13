"""
Tests for IndicPhotoOCR.detection – TextBPN++ detector.

Unit tests cover only the pure static/helper methods (no GPU, no weights).
Integration tests (marked `integration`) run the full model and require
the checkpoint to be present or downloadable.

Run unit tests only:
    pytest tests/test_detection.py

Run everything including integration:
    pytest tests/test_detection.py --run-integration
"""
import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, ANY


# ---------------------------------------------------------------------------
# Static-method unit tests (no model, no GPU)
# ---------------------------------------------------------------------------

class TestPadImage:
    """TextBPNpp_detector.pad_image is a static method – import directly."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from IndicPhotoOCR.detection.textbpn.textbpnpp_detector import TextBPNpp_detector
        self.pad_image = TextBPNpp_detector.pad_image

    def test_already_aligned_image_unchanged(self):
        img = np.zeros((64, 128, 3), dtype=np.uint8)
        padded, original_size = self.pad_image(img, stride=32)
        assert padded.shape[:2] == (64, 128)
        assert original_size == (64, 128)

    def test_unaligned_height_padded(self):
        img = np.zeros((65, 128, 3), dtype=np.uint8)
        padded, original_size = self.pad_image(img, stride=32)
        assert padded.shape[0] == 96  # ceil(65/32)*32
        assert original_size == (65, 128)

    def test_unaligned_width_padded(self):
        img = np.zeros((64, 100, 3), dtype=np.uint8)
        padded, original_size = self.pad_image(img, stride=32)
        assert padded.shape[1] == 128  # ceil(100/32)*32
        assert original_size == (64, 100)

    def test_both_dimensions_unaligned(self):
        img = np.zeros((33, 33, 3), dtype=np.uint8)
        padded, original_size = self.pad_image(img, stride=32)
        assert padded.shape[:2] == (64, 64)
        assert original_size == (33, 33)

    def test_padding_region_is_black(self):
        img = np.ones((32, 32, 3), dtype=np.uint8) * 200
        padded, _ = self.pad_image(img, stride=64)
        # Bottom pad (rows 32-63) should be zero
        assert np.all(padded[32:, :, :] == 0)
        # Right pad (cols 32-63) should be zero
        assert np.all(padded[:, 32:, :] == 0)

    def test_stride_1_is_noop(self):
        img = np.zeros((37, 53, 3), dtype=np.uint8)
        padded, original_size = self.pad_image(img, stride=1)
        assert padded.shape[:2] == (37, 53)

    def test_returns_original_size_tuple(self):
        img = np.zeros((50, 70, 3), dtype=np.uint8)
        _, original_size = self.pad_image(img, stride=32)
        assert isinstance(original_size, tuple)
        assert len(original_size) == 2


class TestRescaleResult:
    """TextBPNpp_detector.rescale_result rescales contour coords."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from IndicPhotoOCR.detection.textbpn.textbpnpp_detector import TextBPNpp_detector
        self.rescale = TextBPNpp_detector.rescale_result

    def _make_image(self, h, w):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def _make_contour(self, points):
        return np.array(points, dtype=np.int32)

    def test_identity_rescale(self):
        """If model image == original image, contours unchanged."""
        image = self._make_image(100, 200)
        cont = self._make_contour([[10, 20], [50, 80]])
        result = self.rescale(image, [cont], original_height=100, original_width=200)
        np.testing.assert_array_equal(result[0], cont)

    def test_downscale_factor_2(self):
        """Model ran on half the image → coords should double."""
        # model image is 50×100, original is 100×200
        image = self._make_image(50, 100)
        cont = self._make_contour([[10, 5], [20, 10]])
        result = self.rescale(image, [cont], original_height=100, original_width=200)
        assert result[0][0][0] == 20   # x: 10 * (200/100)
        assert result[0][0][1] == 10   # y:  5 * (100/50)

    def test_multiple_contours(self):
        image = self._make_image(100, 100)
        conts = [
            self._make_contour([[10, 20]]),
            self._make_contour([[30, 40]]),
        ]
        result = self.rescale(image, conts, original_height=100, original_width=100)
        assert len(result) == 2

    def test_empty_contours_list(self):
        image = self._make_image(100, 100)
        result = self.rescale(image, [], original_height=100, original_width=100)
        assert result == []


# ---------------------------------------------------------------------------
# ensure_model unit tests (network mocked)
# ---------------------------------------------------------------------------

class TestEnsureModel:
    """ensure_model() should download when file missing, skip when present."""

    def test_skips_download_when_model_exists(self, tmp_path):
        from IndicPhotoOCR.detection.textbpn import textbpnpp_detector as mod

        model_path = tmp_path / "model.pth"
        model_path.write_bytes(b"\x00" * 10)   # pretend it already exists

        original_info = mod.model_info.copy()
        mod.model_info["_test"] = {
            "path": "model.pth",
            "url": "http://fake.invalid/model.pth",
        }
        try:
            with patch("IndicPhotoOCR.detection.textbpn.textbpnpp_detector.requests.get") as mock_get, \
                 patch("IndicPhotoOCR.detection.textbpn.textbpnpp_detector.os.path.exists",
                        return_value=True):
                result = mod.ensure_model("_test")
                mock_get.assert_not_called()
        finally:
            mod.model_info.pop("_test", None)

    def test_triggers_download_when_missing(self, tmp_path):
        from IndicPhotoOCR.detection.textbpn import textbpnpp_detector as mod

        fake_response = MagicMock()
        fake_response.headers = {"content-length": "10"}
        fake_response.iter_content.return_value = [b"\x00" * 10]

        with patch("IndicPhotoOCR.detection.textbpn.textbpnpp_detector.requests.get",
                   return_value=fake_response) as mock_get, \
             patch("IndicPhotoOCR.detection.textbpn.textbpnpp_detector.os.path.exists",
                    return_value=False), \
             patch("IndicPhotoOCR.detection.textbpn.textbpnpp_detector.os.makedirs"), \
             patch("builtins.open", new_callable=MagicMock), \
             patch("IndicPhotoOCR.detection.textbpn.textbpnpp_detector.os.rename"):
            mod.ensure_model("textbpnpp")
            mock_get.assert_called_once()


# ---------------------------------------------------------------------------
# TextBPNpp_detector.__init__ (constructor) – mocked
# ---------------------------------------------------------------------------

class TestTextBPNppDetectorInit:
    @patch("IndicPhotoOCR.detection.textbpn.textbpnpp_detector.ensure_model",
           return_value="/fake/model.pth")
    @patch("IndicPhotoOCR.detection.textbpn.textbpnpp_detector.TextNet")
    def test_constructor_calls_ensure_model(self, MockNet, mock_ensure):
        from IndicPhotoOCR.detection.textbpn.textbpnpp_detector import TextBPNpp_detector
        MockNet.return_value.load_model = MagicMock()
        MockNet.return_value.eval = MagicMock(return_value=MagicMock())
        MockNet.return_value.to = MagicMock(return_value=MagicMock())
        TextBPNpp_detector(device="cpu")
        mock_ensure.assert_called_once_with("textbpnpp")

    @patch("IndicPhotoOCR.detection.textbpn.textbpnpp_detector.ensure_model",
           return_value="/fake/model.pth")
    @patch("IndicPhotoOCR.detection.textbpn.textbpnpp_detector.TextNet")
    def test_constructor_sets_device(self, MockNet, mock_ensure):
        from IndicPhotoOCR.detection.textbpn.textbpnpp_detector import TextBPNpp_detector
        import torch
        MockNet.return_value.load_model = MagicMock()
        MockNet.return_value.eval = MagicMock(return_value=MagicMock())
        MockNet.return_value.to = MagicMock(return_value=MagicMock())
        det = TextBPNpp_detector(device="cpu")
        assert det.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# detect() return-format contract – mocked
# ---------------------------------------------------------------------------

class TestTextBPNppDetectorDetect:
    """Test the detect() method output structure with a mocked forward pass."""

    def _make_detector(self):
        with patch("IndicPhotoOCR.detection.textbpn.textbpnpp_detector.ensure_model",
                   return_value="/fake/model.pth"), \
             patch("IndicPhotoOCR.detection.textbpn.textbpnpp_detector.TextNet") as MockNet:
            instance = MockNet.return_value
            instance.load_model = MagicMock()
            instance.eval = MagicMock(return_value=instance)
            instance.to = MagicMock(return_value=instance)
            from IndicPhotoOCR.detection.textbpn.textbpnpp_detector import TextBPNpp_detector
            return TextBPNpp_detector(device="cpu"), instance

    def test_detect_raises_on_missing_image(self, tmp_path):
        det, _ = self._make_detector()
        with pytest.raises((ValueError, Exception)):
            det.detect(str(tmp_path / "nonexistent.jpg"))

    def test_detect_returns_dict_with_detections_key(self, synthetic_scene_image):
        import torch

        det, mock_net = self._make_detector()

        # Fake two 4-point contours
        fake_contours = torch.zeros((2, 4, 2), dtype=torch.int32)
        fake_output = {"py_preds": [None, fake_contours]}
        mock_net.side_effect = None
        mock_net.__call__ = MagicMock(return_value=fake_output)
        mock_net.return_value = fake_output

        with patch.object(det.model, "__call__", return_value=fake_output):
            result = det.detect(synthetic_scene_image)

        assert isinstance(result, dict)
        assert "detections" in result
        assert isinstance(result["detections"], list)

    def test_detect_each_bbox_is_list(self, synthetic_scene_image):
        import torch

        det, mock_net = self._make_detector()
        fake_contours = torch.zeros((3, 4, 2), dtype=torch.int32)
        fake_output = {"py_preds": [None, fake_contours]}

        with patch.object(det.model, "__call__", return_value=fake_output):
            result = det.detect(synthetic_scene_image)

        for bbox in result["detections"]:
            assert isinstance(bbox, list)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestTextBPNppIntegration:
    """Requires model weights (downloaded automatically on first run)."""

    def test_detect_real_image(self, repo_scene_image):
        from IndicPhotoOCR.detection.textbpn.textbpnpp_detector import TextBPNpp_detector
        det = TextBPNpp_detector(device="cpu")
        result = det.detect(repo_scene_image)
        assert "detections" in result
        assert len(result["detections"]) > 0

    def test_detect_result_bboxes_within_image_bounds(self, repo_scene_image):
        import cv2
        from IndicPhotoOCR.detection.textbpn.textbpnpp_detector import TextBPNpp_detector
        det = TextBPNpp_detector(device="cpu")
        result = det.detect(repo_scene_image)
        img = cv2.imread(repo_scene_image)
        h, w = img.shape[:2]
        for bbox in result["detections"]:
            flat = np.array(bbox).flatten()
            xs = flat[0::2]
            ys = flat[1::2]
            assert np.all(xs >= 0) and np.all(xs <= w), f"x out of bounds: {xs}"
            assert np.all(ys >= 0) and np.all(ys <= h), f"y out of bounds: {ys}"
