"""Offline unit tests for the EfficientNet identifier and backend selection."""
import hashlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from IndicPhotoOCR.script_identification.efficientnet import efficientnet_infer as module


CLASSES = ["hindi", "english", "assamese", "bengali", "gujarati", "kannada",
           "malayalam", "marathi", "odia", "punjabi", "tamil", "telugu"]


@pytest.fixture
def fake_checkpoint():
    return {"architecture": "efficientnet_v2_l", "classes": CLASSES.copy(),
            "image_size": 288, "model_state_dict": {}}


@pytest.fixture
def identifier(fake_checkpoint):
    model = MagicMock()
    model.classifier = [None, MagicMock(in_features=1280)]
    model.side_effect = lambda batch: torch.tensor([[0., 10.] + [0.] * 10]).repeat(len(batch), 1)
    with patch.object(module, "download_model_from_release"), \
         patch.object(module.torch, "load", return_value=fake_checkpoint), \
         patch.object(module, "efficientnet_v2_l", return_value=model):
        result = module.EfficientNetIdentifier("/fake/weights.pth", device="cpu")
    return result


def test_strict_loading_and_eval(identifier):
    identifier.model.load_state_dict.assert_called_once_with({}, strict=True)
    identifier.model.to.assert_called_once_with(torch.device("cpu"))
    identifier.model.eval.assert_called_once()
    assert identifier.classes == CLASSES


def test_preprocessing(identifier):
    tensor = identifier.transform(Image.new("RGB", (50, 20), (255, 255, 255)))
    assert tensor.shape == (3, 288, 288)
    expected = (torch.ones(3) - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
    torch.testing.assert_close(tensor[:, 0, 0], expected)
    assert identifier.transform.transforms[0].antialias is True


def test_single_batch_and_top_k(identifier, synthetic_crop_image):
    assert identifier.identify(synthetic_crop_image) == "english"
    assert identifier.identify_batch([synthetic_crop_image] * 3, batch_size=2) == ["english"] * 3
    assert identifier.identify_batch([]) == []
    assert identifier.identify_top_k(synthetic_crop_image, top_k=1)[0][0] == "english"
    assert identifier.identify_batch_top_k([synthetic_crop_image], top_k=1)[0][0][0] == "english"


@pytest.mark.parametrize("override", [{"classes": list(reversed(CLASSES))}, {"image_size": 480}])
def test_rejects_incompatible_arguments(fake_checkpoint, override):
    with patch.object(module, "download_model_from_release"), \
         patch.object(module.torch, "load", return_value=fake_checkpoint):
        with pytest.raises(ValueError):
            module.EfficientNetIdentifier("/fake/weights.pth", device="cpu", **override)


def test_rejects_wrong_architecture(fake_checkpoint):
    fake_checkpoint["architecture"] = "efficientnet_v2_m"
    with patch.object(module, "download_model_from_release"), \
         patch.object(module.torch, "load", return_value=fake_checkpoint):
        with pytest.raises(ValueError, match="V2-L"):
            module.EfficientNetIdentifier("/fake/weights.pth", device="cpu")


def test_default_model_location(identifier, fake_checkpoint):
    with patch.object(module, "download_model_from_release") as download, \
         patch.object(module.torch, "load", return_value=fake_checkpoint), \
         patch.object(module, "efficientnet_v2_l", return_value=identifier.model):
        module.EfficientNetIdentifier(device="cpu")
    expected = Path(module.__file__).resolve().parent / "models/efficientnet/efficientnetv2_l_script_id.pth"
    download.assert_called_once_with("efficientnet_v2_l", expected)


def test_existing_checksum_and_no_download(tmp_path):
    path = tmp_path / "weights.pth"
    path.write_bytes(b"test weights")
    info = dict(module.model_info["efficientnet_v2_l"], sha256=hashlib.sha256(b"test weights").hexdigest())
    with patch.dict(module.model_info, {"efficientnet_v2_l": info}), \
         patch.object(module.urllib.request, "urlretrieve") as retrieve:
        module.download_model_from_release("efficientnet_v2_l", path)
        retrieve.assert_not_called()
        path.write_bytes(b"corrupted")
        with pytest.raises(ValueError, match="SHA-256"):
            module.download_model_from_release("efficientnet_v2_l", path)


@pytest.mark.parametrize("corrupted", [False, True])
def test_download_progress_checksum_and_cleanup(tmp_path, corrupted):
    payload = b"test weights"
    path = tmp_path / "weights.pth"
    info = dict(module.model_info["efficientnet_v2_l"], sha256=hashlib.sha256(payload).hexdigest())
    bar = MagicMock(n=0)
    def update(amount):
        bar.n += amount
    bar.update.side_effect = update
    context = MagicMock()
    context.__enter__.return_value = bar
    def retrieve(url, temporary, reporthook):
        Path(temporary).write_bytes(b"bad" if corrupted else payload)
        for count in range(4):
            reporthook(count, 4, len(payload))
    with patch.dict(module.model_info, {"efficientnet_v2_l": info}), \
         patch.object(module, "tqdm", return_value=context), \
         patch.object(module.urllib.request, "urlretrieve", side_effect=retrieve):
        if corrupted:
            with pytest.raises(ValueError, match="SHA-256"):
                module.download_model_from_release("efficientnet_v2_l", path)
            assert not path.exists()
        else:
            module.download_model_from_release("efficientnet_v2_l", path)
            assert path.read_bytes() == payload
        assert bar.n == len(payload)
        assert bar.total == len(payload)
        assert not list(tmp_path.glob("*.tmp"))


def test_optional_vit_backend_without_downloads():
    from IndicPhotoOCR.ocr import OCR
    name = "IndicPhotoOCR.script_identification.vit.vit_infer"
    fake = types.ModuleType(name)
    fake.VIT_identifier = MagicMock()
    with patch.dict(sys.modules, {name: fake}), \
         patch("IndicPhotoOCR.ocr.TextBPNpp_detector"), \
         patch("IndicPhotoOCR.ocr.PARseqrecogniser"), \
         patch("IndicPhotoOCR.ocr.EfficientNetIdentifier") as efficient:
        ocr = OCR(device="cpu", identifier_type="vit")
    fake.VIT_identifier.assert_called_once()
    efficient.assert_not_called()
    assert ocr._pipeline_device == -1


def test_invalid_backend_rejected():
    from IndicPhotoOCR.ocr import OCR
    with pytest.raises(ValueError, match="identifier_type"):
        OCR(device="cpu", identifier_type="invalid")


@pytest.mark.parametrize("bad_index", [0, 1, 2])
def test_failed_crop_keeps_original_position(identifier, synthetic_crop_image, tmp_path, bad_index):
    paths = [synthetic_crop_image] * 3
    paths[bad_index] = str(tmp_path / "missing.jpg")
    expected = ["english"] * 3
    expected[bad_index] = "hindi"
    assert identifier.identify_batch(paths, batch_size=2) == expected
    top_k = identifier.identify_batch_top_k(paths, top_k=1, batch_size=2)
    assert [item[0][0] for item in top_k] == expected


def test_all_failed_crops_keep_length(identifier, tmp_path):
    paths = [str(tmp_path / "missing.jpg")] * 3
    assert identifier.identify_batch(paths, batch_size=2) == ["hindi"] * 3
    assert identifier.identify_batch_top_k(paths, batch_size=2) == [[("hindi", 1.0)]] * 3


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "cuda:1"])
def test_detector_settings_applied_before_loading(device):
    from IndicPhotoOCR.ocr import OCR
    from IndicPhotoOCR.detection.textbpn.cfglib.config import config
    def construct(**kwargs):
        assert config.device == torch.device(device)
        assert config.cuda == (torch.device(device).type == "cuda")
        assert config.exp_name == "MLT2019"
        return MagicMock()
    with patch.dict(config, dict(config)), \
         patch("IndicPhotoOCR.ocr.TextBPNpp_detector", side_effect=construct), \
         patch("IndicPhotoOCR.ocr.PARseqrecogniser"), \
         patch("IndicPhotoOCR.ocr.EfficientNetIdentifier"):
        OCR(device=device)
