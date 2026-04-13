"""
Tests for IndicPhotoOCR.script_identification – VIT_identifier.

Unit tests mock all network calls and model loading.
Integration tests (--run-integration) run real inference.
"""
import os
import zipfile
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call
from PIL import Image


# ---------------------------------------------------------------------------
# ensure_model – download / cache logic
# ---------------------------------------------------------------------------

class TestVITEnsureModel:
    """Verify model is only downloaded when not already present."""

    def _make_identifier(self):
        # Import inside to avoid top-level side effects (processor auto-download)
        with patch("IndicPhotoOCR.script_identification.vit.vit_infer.AutoImageProcessor.from_pretrained"):
            from IndicPhotoOCR.script_identification.vit.vit_infer import VIT_identifier
            return VIT_identifier()

    def test_skips_download_when_folder_exists(self, tmp_path):
        ident = self._make_identifier()
        fake_path = tmp_path / "models" / "hindienglish"
        fake_path.mkdir(parents=True)

        with patch("IndicPhotoOCR.script_identification.vit.vit_infer.os.path.exists",
                   return_value=True), \
             patch("IndicPhotoOCR.script_identification.vit.vit_infer.requests.get") as mock_get:
            ident.ensure_model("hindi")
            mock_get.assert_not_called()

    def test_triggers_download_when_folder_absent(self, tmp_path):
        ident = self._make_identifier()

        fake_zip_buf = _create_fake_zip()
        fake_response = MagicMock()
        fake_response.iter_content.return_value = [fake_zip_buf]

        with patch("IndicPhotoOCR.script_identification.vit.vit_infer.os.path.exists",
                   return_value=False), \
             patch("IndicPhotoOCR.script_identification.vit.vit_infer.requests.get",
                   return_value=fake_response) as mock_get, \
             patch("IndicPhotoOCR.script_identification.vit.vit_infer.os.makedirs"), \
             patch("IndicPhotoOCR.script_identification.vit.vit_infer.zipfile.ZipFile"), \
             patch("IndicPhotoOCR.script_identification.vit.vit_infer.os.remove"), \
             patch("builtins.open", new_callable=MagicMock), \
             patch("IndicPhotoOCR.script_identification.vit.vit_infer.os.rename"):
            ident.ensure_model("hindi")
            mock_get.assert_called_once()

    def test_ensure_model_returns_string_path(self):
        ident = self._make_identifier()
        with patch("IndicPhotoOCR.script_identification.vit.vit_infer.os.path.exists",
                   return_value=True):
            path = ident.ensure_model("hindi")
        assert isinstance(path, str)
        assert len(path) > 0


def _create_fake_zip():
    """Return bytes of a minimal valid ZIP file."""
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("dummy.txt", "placeholder")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# model_info registry
# ---------------------------------------------------------------------------

class TestVITModelInfo:
    """Verify the model registry contains expected keys and structure."""

    @pytest.fixture(autouse=True)
    def _import(self):
        with patch("IndicPhotoOCR.script_identification.vit.vit_infer.AutoImageProcessor.from_pretrained"):
            from IndicPhotoOCR.script_identification.vit.vit_infer import model_info
            self.model_info = model_info

    EXPECTED_LANGUAGES = [
        "hindi", "assamese", "bengali", "gujarati", "kannada",
        "malayalam", "marathi", "meitei", "odia", "punjabi",
        "tamil", "telugu", "auto", "10C",
    ]

    def test_all_languages_present(self):
        for lang in self.EXPECTED_LANGUAGES:
            assert lang in self.model_info, f"Missing language: {lang}"

    def test_each_entry_has_required_keys(self):
        for lang, info in self.model_info.items():
            assert "path" in info, f"{lang}: missing 'path'"
            assert "url" in info, f"{lang}: missing 'url'"
            assert "subcategories" in info, f"{lang}: missing 'subcategories'"

    def test_subcategories_are_non_empty_lists(self):
        for lang, info in self.model_info.items():
            assert isinstance(info["subcategories"], list), f"{lang}: subcategories not a list"
            assert len(info["subcategories"]) >= 2, f"{lang}: need at least 2 subcategories"

    def test_auto_model_has_12_subcategories(self):
        assert len(self.model_info["auto"]["subcategories"]) == 12

    def test_english_is_always_a_subcategory(self):
        """Every model should be able to handle English."""
        for lang, info in self.model_info.items():
            assert "english" in info["subcategories"], \
                f"{lang}: 'english' missing from subcategories"

    def test_urls_start_with_https(self):
        for lang, info in self.model_info.items():
            assert info["url"].startswith("https://"), \
                f"{lang}: URL should use HTTPS"


# ---------------------------------------------------------------------------
# identify() – output contract (fully mocked)
# ---------------------------------------------------------------------------

class TestVITIdentify:
    def _make_identifier(self):
        with patch("IndicPhotoOCR.script_identification.vit.vit_infer.AutoImageProcessor.from_pretrained"):
            from IndicPhotoOCR.script_identification.vit.vit_infer import VIT_identifier
            return VIT_identifier()

    def test_identify_returns_string(self, synthetic_crop_image):
        ident = self._make_identifier()

        mock_pipeline = MagicMock(return_value=[
            {"label": "hindi", "score": 0.95},
            {"label": "english", "score": 0.05},
        ])

        with patch.object(ident, "ensure_model", return_value="/fake/model"), \
             patch("IndicPhotoOCR.script_identification.vit.vit_infer.ViTForImageClassification.from_pretrained",
                   return_value=MagicMock()), \
             patch("IndicPhotoOCR.script_identification.vit.vit_infer.pipeline",
                   return_value=mock_pipeline):
            result = ident.identify(synthetic_crop_image, "hindi", "cpu")

        assert isinstance(result, str)

    def test_identify_returns_highest_score_label(self, synthetic_crop_image):
        ident = self._make_identifier()

        mock_pipeline = MagicMock(return_value=[
            {"label": "hindi",   "score": 0.10},
            {"label": "english", "score": 0.80},
            {"label": "tamil",   "score": 0.10},
        ])

        with patch.object(ident, "ensure_model", return_value="/fake/model"), \
             patch("IndicPhotoOCR.script_identification.vit.vit_infer.ViTForImageClassification.from_pretrained",
                   return_value=MagicMock()), \
             patch("IndicPhotoOCR.script_identification.vit.vit_infer.pipeline",
                   return_value=mock_pipeline):
            result = ident.identify(synthetic_crop_image, "hindi", "cpu")

        assert result == "english"

    def test_identify_calls_ensure_model(self, synthetic_crop_image):
        ident = self._make_identifier()

        mock_pipeline = MagicMock(return_value=[{"label": "hindi", "score": 1.0}])

        with patch.object(ident, "ensure_model", return_value="/fake/model") as mock_ensure, \
             patch("IndicPhotoOCR.script_identification.vit.vit_infer.ViTForImageClassification.from_pretrained",
                   return_value=MagicMock()), \
             patch("IndicPhotoOCR.script_identification.vit.vit_infer.pipeline",
                   return_value=mock_pipeline):
            ident.identify(synthetic_crop_image, "hindi", "cpu")

        mock_ensure.assert_called_once_with("hindi")

    @pytest.mark.parametrize("ext", [".png", ".jpg", ".jpeg"])
    def test_identify_accepts_valid_extensions(self, tmp_path, ext):
        ident = self._make_identifier()
        img_path = tmp_path / f"crop{ext}"
        Image.new("RGB", (64, 32), (200, 100, 50)).save(str(img_path))

        mock_pipeline = MagicMock(return_value=[{"label": "hindi", "score": 1.0}])

        with patch.object(ident, "ensure_model", return_value="/fake/model"), \
             patch("IndicPhotoOCR.script_identification.vit.vit_infer.ViTForImageClassification.from_pretrained",
                   return_value=MagicMock()), \
             patch("IndicPhotoOCR.script_identification.vit.vit_infer.pipeline",
                   return_value=mock_pipeline):
            result = ident.identify(str(img_path), "hindi", "cpu")

        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# predict_batch() – output file contract
# ---------------------------------------------------------------------------

class TestVITBatchPredict:
    def _make_identifier(self):
        with patch("IndicPhotoOCR.script_identification.vit.vit_infer.AutoImageProcessor.from_pretrained"):
            from IndicPhotoOCR.script_identification.vit.vit_infer import VIT_identifier
            return VIT_identifier()

    def test_batch_creates_csv(self, tmp_path):
        ident = self._make_identifier()

        # Create 3 fake images
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(3):
            Image.new("RGB", (64, 32)).save(str(img_dir / f"img_{i}.jpg"))

        csv_out = str(tmp_path / "out.csv")
        mock_pipeline = MagicMock(return_value=[{"label": "Hindi", "score": 1.0}])

        with patch.object(ident, "ensure_model", return_value="/fake/model"), \
             patch("IndicPhotoOCR.script_identification.vit.vit_infer.ViTForImageClassification.from_pretrained",
                   return_value=MagicMock()), \
             patch("IndicPhotoOCR.script_identification.vit.vit_infer.pipeline",
                   return_value=mock_pipeline):
            result_path = ident.predict_batch(str(img_dir), "hindi",
                                              time_show=False, output_csv=csv_out)

        assert os.path.exists(result_path)
        import csv
        with open(result_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        for row in rows:
            assert "Filepath" in row
            assert "Language" in row


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestVITIntegration:
    def test_identify_real_crop(self, repo_crop_image):
        with patch("IndicPhotoOCR.script_identification.vit.vit_infer.AutoImageProcessor.from_pretrained"):
            from IndicPhotoOCR.script_identification.vit.vit_infer import VIT_identifier
        ident = VIT_identifier()
        result = ident.identify(repo_crop_image, "hindi", "cpu")
        assert isinstance(result, str)
        assert result in ["hindi", "english"]
