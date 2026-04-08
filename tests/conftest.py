"""
Shared fixtures for IndicPhotoOCR tests.

All fixtures that create real files use tmp_path (pytest built-in) so they
are cleaned up automatically after each test.  No model weights are
downloaded during unit tests – heavy integration tests are gated behind
the --run-integration CLI flag.
"""
import os
import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Pytest configuration
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require downloaded model weights.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as needing real model weights (skipped by default).",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="Pass --run-integration to run.")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------

def _make_rgb_array(h: int = 64, w: int = 128, color=(128, 64, 32)) -> np.ndarray:
    """Return an H×W×3 uint8 numpy array filled with *color* (BGR order)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color[::-1]  # PIL is RGB, cv2 is BGR – store as BGR
    return img


def _make_pil_image(h: int = 64, w: int = 128, color=(200, 100, 50)) -> Image.Image:
    img = Image.new("RGB", (w, h), color=color)
    return img


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_scene_image(tmp_path) -> str:
    """A small synthetic scene image saved to a temp file. Returns path."""
    path = tmp_path / "scene.jpg"
    img = _make_pil_image(h=480, w=640, color=(180, 180, 180))
    img.save(str(path))
    return str(path)


@pytest.fixture()
def synthetic_crop_image(tmp_path) -> str:
    """A tiny synthetic word-crop image saved to a temp file. Returns path."""
    path = tmp_path / "crop.jpg"
    img = _make_pil_image(h=32, w=100, color=(230, 220, 210))
    img.save(str(path))
    return str(path)


@pytest.fixture()
def repo_scene_image() -> str:
    """Real test image shipped with the repository (no model required)."""
    base = os.path.join(os.path.dirname(__file__), "..", "test_images")
    path = os.path.join(base, "image_141.jpg")
    if not os.path.exists(path):
        pytest.skip("Repo test image not found – run from repo root.")
    return path


@pytest.fixture()
def repo_crop_image() -> str:
    """Real cropped-word image shipped with the repository."""
    base = os.path.join(os.path.dirname(__file__), "..", "test_images", "cropped_image")
    path = os.path.join(base, "image_141_0.jpg")
    if not os.path.exists(path):
        pytest.skip("Repo crop image not found – run from repo root.")
    return path


@pytest.fixture()
def sample_bbox_dict():
    """A typical `recognized_texts` dict as returned by ocr.ocr() internals."""
    return {
        "img_0": {"txt": "hello",   "bbox": [10,  10,  80,  30]},
        "img_1": {"txt": "world",   "bbox": [90,  12,  160, 32]},
        "img_2": {"txt": "नमस्ते", "bbox": [10,  50,  100, 70]},
        "img_3": {"txt": "दुनिया", "bbox": [110, 52,  200, 72]},
    }


@pytest.fixture()
def single_word_bbox_dict():
    return {"img_0": {"txt": "only", "bbox": [5, 5, 50, 20]}}
