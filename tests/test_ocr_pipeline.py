"""
Tests for IndicPhotoOCR.ocr – the OCR orchestration class.

All three sub-modules (detector, script-identifier, recogniser) are mocked
so that unit tests run with zero GPU or network access.

Integration tests (--run-integration) exercise the real pipeline.
"""
import os
import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_fake_detector(detections=None):
    """Return a mock detector whose detect() returns fixed bboxes."""
    if detections is None:
        # A single rectangular bbox (4 corner points)
        detections = [[[10, 20], [80, 20], [80, 40], [10, 40]]]
    mock = MagicMock()
    mock.detect.return_value = {"detections": detections}
    return mock


def _make_fake_identifier(label="hindi"):
    mock = MagicMock()
    mock.identify.return_value = label
    return mock


def _make_fake_recogniser(text="टेस्ट"):
    mock = MagicMock()
    mock.recognise.return_value = text
    return mock


def _build_ocr(detector=None, identifier=None, recogniser=None, **kwargs):
    """Construct an OCR instance with all sub-modules replaced by fakes."""
    with patch("IndicPhotoOCR.ocr.TextBPNpp_detector", return_value=detector or _make_fake_detector()), \
         patch("IndicPhotoOCR.ocr.VIT_identifier",     return_value=identifier or _make_fake_identifier()), \
         patch("IndicPhotoOCR.ocr.PARseqrecogniser",   return_value=recogniser or _make_fake_recogniser()):
        from IndicPhotoOCR.ocr import OCR
        return OCR(device="cpu", **kwargs)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestOCRConstructor:
    def test_default_device_is_cuda(self):
        """Default device param should be 'cuda:0' (even if unavailable)."""
        with patch("IndicPhotoOCR.ocr.TextBPNpp_detector"), \
             patch("IndicPhotoOCR.ocr.VIT_identifier"), \
             patch("IndicPhotoOCR.ocr.PARseqrecogniser"):
            from IndicPhotoOCR.ocr import OCR
            ocr = OCR.__new__(OCR)
            # Manually check the docstring / arg default via inspect
            import inspect
            sig = inspect.signature(OCR.__init__)
            assert sig.parameters["device"].default == "cuda:0"

    def test_verbose_stored(self):
        ocr = _build_ocr(verbose=True)
        assert ocr.verbose is True

    def test_identifier_lang_stored(self):
        ocr = _build_ocr(identifier_lang="tamil")
        assert ocr.indentifier_lang == "tamil"

    def test_submodules_instantiated(self):
        with patch("IndicPhotoOCR.ocr.TextBPNpp_detector") as MockDet, \
             patch("IndicPhotoOCR.ocr.VIT_identifier") as MockVit, \
             patch("IndicPhotoOCR.ocr.PARseqrecogniser") as MockRec:
            from IndicPhotoOCR.ocr import OCR
            OCR(device="cpu")
            MockDet.assert_called_once()
            MockVit.assert_called_once()
            MockRec.assert_called_once()


# ---------------------------------------------------------------------------
# detect()
# ---------------------------------------------------------------------------

class TestOCRDetect:
    def test_detect_delegates_to_detector(self, synthetic_scene_image):
        fake_det = _make_fake_detector([[[0, 0], [10, 0], [10, 5], [0, 5]]])
        ocr = _build_ocr(detector=fake_det)
        result = ocr.detect(synthetic_scene_image)
        fake_det.detect.assert_called_once_with(synthetic_scene_image)

    def test_detect_returns_list(self, synthetic_scene_image):
        ocr = _build_ocr()
        result = ocr.detect(synthetic_scene_image)
        assert isinstance(result, list)

    def test_detect_returns_detector_bboxes(self, synthetic_scene_image):
        bboxes = [[[1, 2], [3, 2], [3, 4], [1, 4]]]
        ocr = _build_ocr(detector=_make_fake_detector(bboxes))
        assert ocr.detect(synthetic_scene_image) == bboxes

    def test_detect_empty_image_returns_empty_list(self, synthetic_scene_image):
        ocr = _build_ocr(detector=_make_fake_detector([]))
        assert ocr.detect(synthetic_scene_image) == []


# ---------------------------------------------------------------------------
# identify()
# ---------------------------------------------------------------------------

class TestOCRIdentify:
    def test_identify_returns_string(self, synthetic_crop_image):
        ocr = _build_ocr()
        result = ocr.identify(synthetic_crop_image)
        assert isinstance(result, str)

    def test_identify_passes_lang_to_identifier(self, synthetic_crop_image):
        fake_id = _make_fake_identifier("bengali")
        ocr = _build_ocr(identifier=fake_id, identifier_lang="bengali")
        ocr.identify(synthetic_crop_image)
        fake_id.identify.assert_called_once_with(
            synthetic_crop_image, "bengali", "cpu"
        )


# ---------------------------------------------------------------------------
# crop_and_identify_script()
# ---------------------------------------------------------------------------

class TestCropAndIdentifyScript:
    def _make_image_array(self, h=200, w=300):
        return (np.random.rand(h, w, 3) * 255).astype(np.uint8)

    def test_returns_two_tuple(self):
        fake_id = _make_fake_identifier("hindi")
        ocr = _build_ocr(identifier=fake_id)
        img = self._make_image_array()
        bbox = [[10, 20], [80, 20], [80, 50], [10, 50]]
        lang, path = ocr.crop_and_identify_script(img, bbox)
        assert isinstance(lang, str)
        assert isinstance(path, str)

    def test_script_lang_matches_identifier_output(self):
        fake_id = _make_fake_identifier("tamil")
        ocr = _build_ocr(identifier=fake_id)
        img = self._make_image_array()
        bbox = [[10, 20], [80, 20], [80, 50], [10, 50]]
        lang, _ = ocr.crop_and_identify_script(img, bbox)
        assert lang == "tamil"

    def test_cropped_image_path_is_string(self):
        """crop_and_identify_script returns a non-empty string path."""
        ocr = _build_ocr()
        img = self._make_image_array()
        bbox = [[5, 5], [50, 5], [50, 30], [5, 30]]
        _, path = ocr.crop_and_identify_script(img, bbox)
        assert isinstance(path, str) and len(path) > 0

    def test_degenerate_bbox_does_not_crash(self):
        """A zero-area bbox should not raise an exception."""
        ocr = _build_ocr()
        img = self._make_image_array()
        bbox = [[10, 10], [10, 10], [10, 10], [10, 10]]
        try:
            ocr.crop_and_identify_script(img, bbox)
        except Exception as e:
            pytest.fail(f"crop_and_identify_script raised unexpectedly: {e}")


# ---------------------------------------------------------------------------
# recognise()
# ---------------------------------------------------------------------------

class TestOCRRecognise:
    def test_recognise_returns_string(self, synthetic_crop_image):
        ocr = _build_ocr()
        result = ocr.recognise(synthetic_crop_image, "hindi")
        assert isinstance(result, str)

    def test_recognise_delegates_to_recogniser(self, synthetic_crop_image):
        fake_rec = _make_fake_recogniser("मण्डी")
        ocr = _build_ocr(recogniser=fake_rec)
        result = ocr.recognise(synthetic_crop_image, "hindi")
        assert result == "मण्डी"
        fake_rec.recognise.assert_called_once()

    def test_recognise_passes_language_and_image(self, synthetic_crop_image):
        fake_rec = _make_fake_recogniser("road")
        ocr = _build_ocr(recogniser=fake_rec)
        ocr.recognise(synthetic_crop_image, "english")
        args = fake_rec.recognise.call_args
        # language and image path must appear somewhere in the call
        all_args = list(args.args) + list(args.kwargs.values())
        assert synthetic_crop_image in all_args
        assert "english" in all_args


# ---------------------------------------------------------------------------
# visualize_detection()
# ---------------------------------------------------------------------------

class TestVisualizeDetection:
    def test_creates_output_file(self, tmp_path, synthetic_scene_image):
        ocr = _build_ocr()
        out = str(tmp_path / "out.png")
        bboxes = [[[10, 20], [80, 20], [80, 40], [10, 40]]]
        ocr.visualize_detection(synthetic_scene_image, bboxes, save_path=out)
        assert os.path.exists(out)

    def test_creates_parent_directory(self, tmp_path, synthetic_scene_image):
        ocr = _build_ocr()
        out = str(tmp_path / "subdir" / "out.png")
        ocr.visualize_detection(synthetic_scene_image, [], save_path=out)
        assert os.path.exists(out)

    def test_default_save_path_is_test_png(self, synthetic_scene_image, monkeypatch, tmp_path):
        """When save_path=None it writes to 'test.png' relative to CWD."""
        monkeypatch.chdir(tmp_path)
        ocr = _build_ocr()
        ocr.visualize_detection(synthetic_scene_image, [])
        assert os.path.exists(tmp_path / "test.png")

    def test_saved_image_is_readable(self, tmp_path, synthetic_scene_image):
        ocr = _build_ocr()
        out = str(tmp_path / "vis.png")
        ocr.visualize_detection(synthetic_scene_image, [[[5, 5], [50, 5], [50, 30], [5, 30]]], save_path=out)
        loaded = cv2.imread(out)
        assert loaded is not None
        assert loaded.shape[2] == 3


# ---------------------------------------------------------------------------
# ocr() – end-to-end with all sub-modules mocked
# ---------------------------------------------------------------------------

class TestOCREndToEnd:
    def test_ocr_returns_list(self, synthetic_scene_image):
        ocr = _build_ocr()
        result = ocr.ocr(synthetic_scene_image)
        assert isinstance(result, list)

    def test_ocr_empty_detections_returns_empty_list(self, synthetic_scene_image):
        ocr = _build_ocr(detector=_make_fake_detector([]))
        result = ocr.ocr(synthetic_scene_image)
        assert result == []

    def test_ocr_single_detection_returns_one_line(self, synthetic_scene_image):
        bboxes = [[[10, 20], [80, 20], [80, 40], [10, 40]]]
        ocr = _build_ocr(
            detector=_make_fake_detector(bboxes),
            identifier=_make_fake_identifier("hindi"),
            recogniser=_make_fake_recogniser("नमस्ते"),
        )
        result = ocr.ocr(synthetic_scene_image)
        assert isinstance(result, list)
        flat = [w for line in result for w in line]
        assert "नमस्ते" in flat

    def test_ocr_multiple_detections_all_words_present(self, synthetic_scene_image):
        bboxes = [
            [[10, 20], [80, 20], [80, 40], [10, 40]],
            [[90, 20], [160, 20], [160, 40], [90, 40]],
        ]
        fake_rec = MagicMock()
        fake_rec.recognise.side_effect = ["word1", "word2"]

        ocr = _build_ocr(
            detector=_make_fake_detector(bboxes),
            identifier=_make_fake_identifier("hindi"),
            recogniser=fake_rec,
        )
        result = ocr.ocr(synthetic_scene_image)
        flat = [w for line in result for w in line]
        assert set(flat) == {"word1", "word2"}

    def test_ocr_detection_called_once(self, synthetic_scene_image):
        fake_det = _make_fake_detector([])
        ocr = _build_ocr(detector=fake_det)
        ocr.ocr(synthetic_scene_image)
        fake_det.detect.assert_called_once_with(synthetic_scene_image)

    def test_ocr_skips_word_when_script_unknown(self, synthetic_scene_image):
        """If identify() returns falsy, the word should be silently skipped."""
        bboxes = [[[10, 20], [80, 20], [80, 40], [10, 40]]]
        fake_id = _make_fake_identifier(None)  # falsy return
        fake_rec = _make_fake_recogniser("should_not_appear")
        ocr = _build_ocr(
            detector=_make_fake_detector(bboxes),
            identifier=fake_id,
            recogniser=fake_rec,
        )
        result = ocr.ocr(synthetic_scene_image)
        flat = [w for line in result for w in line]
        assert "should_not_appear" not in flat


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestOCRIntegration:
    def test_full_pipeline_real_image(self, repo_scene_image):
        from IndicPhotoOCR.ocr import OCR
        ocr = OCR(device="cpu", identifier_lang="auto", verbose=False)
        result = ocr.ocr(repo_scene_image)
        assert isinstance(result, list)
        assert len(result) > 0
        for line in result:
            assert isinstance(line, list)
            for word in line:
                assert isinstance(word, str)

    def test_detect_then_visualize(self, repo_scene_image, tmp_path):
        from IndicPhotoOCR.ocr import OCR
        ocr = OCR(device="cpu", verbose=False)
        detections = ocr.detect(repo_scene_image)
        assert len(detections) > 0
        out = str(tmp_path / "vis.png")
        ocr.visualize_detection(repo_scene_image, detections, save_path=out)
        assert os.path.exists(out)
