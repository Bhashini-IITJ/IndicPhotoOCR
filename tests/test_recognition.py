"""
Tests for IndicPhotoOCR.recognition – PARseqrecogniser.

Unit tests mock all model loading and torch.hub calls.
Integration tests (--run-integration) run real inference.
"""
import os
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, call
from PIL import Image


# ---------------------------------------------------------------------------
# model_info registry
# ---------------------------------------------------------------------------

class TestPARseqModelInfo:
    @pytest.fixture(autouse=True)
    def _import(self):
        from IndicPhotoOCR.recognition.parseq_recogniser import model_info
        self.model_info = model_info

    EXPECTED_LANGUAGES = [
        "assamese", "bengali", "hindi", "gujarati", "kannada",
        "malayalam", "marathi", "odia", "punjabi", "tamil", "telugu",
    ]

    def test_all_indic_languages_present(self):
        for lang in self.EXPECTED_LANGUAGES:
            assert lang in self.model_info, f"Missing language: {lang}"

    def test_each_entry_has_path_and_url(self):
        for lang, info in self.model_info.items():
            assert "path" in info, f"{lang}: missing 'path'"
            assert "url"  in info, f"{lang}: missing 'url'"

    def test_paths_end_with_ckpt(self):
        for lang, info in self.model_info.items():
            assert info["path"].endswith(".ckpt"), \
                f"{lang}: checkpoint path should end with .ckpt"

    def test_urls_start_with_https(self):
        for lang, info in self.model_info.items():
            assert info["url"].startswith("https://"), \
                f"{lang}: URL should use HTTPS"


# ---------------------------------------------------------------------------
# ensure_model – download / cache
# ---------------------------------------------------------------------------

class TestPARseqEnsureModel:
    @pytest.fixture(autouse=True)
    def _recogniser(self):
        from IndicPhotoOCR.recognition.parseq_recogniser import PARseqrecogniser
        self.rec = PARseqrecogniser()

    def test_skips_download_when_file_exists(self):
        with patch("IndicPhotoOCR.recognition.parseq_recogniser.os.path.exists",
                   return_value=True), \
             patch("IndicPhotoOCR.recognition.parseq_recogniser.requests.get") as mock_get:
            self.rec.ensure_model("hindi")
            mock_get.assert_not_called()

    def test_downloads_when_file_missing(self):
        fake_response = MagicMock()
        fake_response.headers = {"content-length": "10"}
        fake_response.iter_content.return_value = [b"\x00" * 10]

        with patch("IndicPhotoOCR.recognition.parseq_recogniser.os.path.exists",
                   return_value=False), \
             patch("IndicPhotoOCR.recognition.parseq_recogniser.requests.get",
                   return_value=fake_response) as mock_get, \
             patch("IndicPhotoOCR.recognition.parseq_recogniser.os.makedirs"), \
             patch("builtins.open", new_callable=MagicMock):
            self.rec.ensure_model("hindi")
            mock_get.assert_called_once()

    def test_returns_string_path(self):
        with patch("IndicPhotoOCR.recognition.parseq_recogniser.os.path.exists",
                   return_value=True):
            path = self.rec.ensure_model("hindi")
        assert isinstance(path, str)


# ---------------------------------------------------------------------------
# get_transform
# ---------------------------------------------------------------------------

class TestGetTransform:
    @pytest.fixture(autouse=True)
    def _recogniser(self):
        from IndicPhotoOCR.recognition.parseq_recogniser import PARseqrecogniser
        self.rec = PARseqrecogniser()

    def test_returns_callable(self):
        t = self.rec.get_transform((32, 128))
        assert callable(t)

    def test_transform_output_tensor_shape(self):
        t = self.rec.get_transform((32, 128))
        img = Image.new("RGB", (200, 60))
        result = t(img)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 32, 128)

    def test_transform_output_value_range(self):
        """Normalise(0.5, 0.5) maps [0,1] → [-1,1]."""
        t = self.rec.get_transform((32, 128))
        img = Image.new("RGB", (200, 60), (255, 255, 255))
        result = t(img)
        assert float(result.min()) >= -1.05   # small float tolerance
        assert float(result.max()) <= 1.05

    def test_transform_with_rotation(self):
        """Rotation kwarg should not raise."""
        t = self.rec.get_transform((32, 128), rotation=90)
        img = Image.new("RGB", (200, 60))
        result = t(img)
        assert isinstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# get_model_output – contract (model mocked)
# ---------------------------------------------------------------------------

class TestGetModelOutput:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from IndicPhotoOCR.recognition.parseq_recogniser import PARseqrecogniser
        self.rec = PARseqrecogniser()

    def _make_mock_model(self, text_out="hello"):
        mock_model = MagicMock()
        mock_model.hparams.img_size = (32, 128)

        # Build plausible logits shape (1, seq_len, vocab)
        seq_len = len(text_out) + 1
        logits = torch.randn(1, seq_len, 96)
        mock_model.return_value = logits

        # softmax → decode → charset_adapter chain
        mock_model.tokenizer.decode.return_value = ([text_out], [torch.ones(seq_len)])
        mock_model.charset_adapter.return_value = text_out
        return mock_model

    def test_returns_string(self, synthetic_crop_image):
        model = self._make_mock_model("test")
        result = self.rec.get_model_output("cpu", model, synthetic_crop_image)
        assert isinstance(result, str)

    def test_returns_model_output_text(self, synthetic_crop_image):
        model = self._make_mock_model("नमस्ते")
        result = self.rec.get_model_output("cpu", model, synthetic_crop_image)
        assert result == "नमस्ते"

    def test_raises_on_nonexistent_image(self, tmp_path):
        model = self._make_mock_model("x")
        with pytest.raises(Exception):
            self.rec.get_model_output("cpu", model, str(tmp_path / "ghost.jpg"))


# ---------------------------------------------------------------------------
# recognise() – top-level API contract (all model I/O mocked)
# ---------------------------------------------------------------------------

class TestRecognise:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from IndicPhotoOCR.recognition.parseq_recogniser import PARseqrecogniser
        self.rec = PARseqrecogniser()

    def test_recognise_indic_uses_ensure_model(self, synthetic_crop_image):
        with patch.object(self.rec, "ensure_model", return_value="/fake/hindi.ckpt") as m_ensure, \
             patch.object(self.rec, "load_model") as m_load, \
             patch.object(self.rec, "get_model_output", return_value="टेस्ट"):
            result = self.rec.recognise("hindi", synthetic_crop_image, "hindi", False, "cpu")
            m_ensure.assert_called_once_with("hindi")
            assert result == "टेस्ट"

    def test_recognise_english_uses_torch_hub(self, synthetic_crop_image):
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        with patch("IndicPhotoOCR.recognition.parseq_recogniser.torch.hub.load",
                   return_value=mock_model) as m_hub, \
             patch.object(self.rec, "get_model_output", return_value="hello"):
            result = self.rec.recognise("english", synthetic_crop_image, "english", False, "cpu")
            m_hub.assert_called_once()
            assert result == "hello"

    def test_recognise_returns_string(self, synthetic_crop_image):
        with patch.object(self.rec, "ensure_model", return_value="/fake/m.ckpt"), \
             patch.object(self.rec, "load_model", return_value=MagicMock()), \
             patch.object(self.rec, "get_model_output", return_value="word"):
            result = self.rec.recognise("hindi", synthetic_crop_image, "hindi", False, "cpu")
        assert isinstance(result, str)

    @pytest.mark.parametrize("lang", [
        "hindi", "bengali", "tamil", "telugu", "gujarati",
        "kannada", "malayalam", "marathi", "odia", "punjabi", "assamese",
    ])
    def test_recognise_all_indic_languages_routed_correctly(self, synthetic_crop_image, lang):
        """All Indic languages should NOT go through torch.hub."""
        with patch.object(self.rec, "ensure_model", return_value="/fake/m.ckpt"), \
             patch.object(self.rec, "load_model", return_value=MagicMock()), \
             patch.object(self.rec, "get_model_output", return_value="word"), \
             patch("IndicPhotoOCR.recognition.parseq_recogniser.torch.hub.load") as m_hub:
            self.rec.recognise(lang, synthetic_crop_image, lang, False, "cpu")
            m_hub.assert_not_called()


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPARseqIntegration:
    def test_recognise_hindi_crop(self, repo_crop_image):
        from IndicPhotoOCR.recognition.parseq_recogniser import PARseqrecogniser
        rec = PARseqrecogniser()
        result = rec.recognise("hindi", repo_crop_image, "hindi", False, "cpu")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_recognise_english_crop(self, repo_crop_image):
        from IndicPhotoOCR.recognition.parseq_recogniser import PARseqrecogniser
        rec = PARseqrecogniser()
        result = rec.recognise("english", repo_crop_image, "english", False, "cpu")
        assert isinstance(result, str)
