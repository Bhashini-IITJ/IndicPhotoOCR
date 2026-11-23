# Changelog

## [2.0.0] - 2026-11-23
IndicPhotoOCR 2.0.

### Added
- YOLO11n text detector (`detector="yolo"`, the new default). Two models, chosen with `OCR(detector_model=...)`, each run at the size it was trained at:
  - `yolo11n_bstd_mist_1920` (default): fine-tuned on BSTD train + MIST training frames at 1920 px; most robust on camera and video frames (MIST detection F1 0.74 vs 0.43 for TextBPN++)
  - `yolo11n_bstd_1920`: fine-tuned on BSTD train only at 1920 px; highest BSTD test recall (0.59)
  - Weights download from the Bhashini-IITJ/SceneTextDetection `YOLO` release with SHA-256 verification; `detector_checkpoint` accepts a local file
- EfficientNetV2-L script identifier (default; ViT stays selectable), weights from the `efficientnetv2-l-v1.0` release with SHA-256 verification
- `--detector` option for `end-to-end-Inference.py` (`yolo` default; `textbpn` and `east` selectable)
- Unit tests for the YOLO detector, detector selection and the EfficientNet identifier

### Changed
- End-to-end on the BSTD test set (official metric): recall 0.48 -> 0.58 with the default model; detection about 11x faster
- `ultralytics==8.3.227` added to the package requirements (Ultralytics is AGPL-3.0)
- Dockerfile installs the package from the build context instead of cloning a fixed repository

### Fixed
- `make run` mounted an empty path; it now mounts the checkout at `/workspace/IndicPhotoOCR`
- `make build` no longer requires BuildKit/buildx; new `make test` target; image name overridable with `IMAGE=`
- `end-to-end-Inference.py` skips sub-directories of the image folder instead of logging them as failed images

## [1.3.1] - 2025-17-25
### Added
- Seperate function for script identification
- Added device opts for recognition

## [1.3.0] - 2025-17-25
### Added
- ViT models for script detection
- Word order based on horizontal and vertical lines

## [1.2.0] - 2024-11-24
### Added
- Textbpn++ detection model added

## [1.1.0] - 2024-11-24
### Added
- Updated package naming convention

## [1.0.3] - 2024-11-06
### Added
- Python package requirements sorted with setup.py

## [1.0.2] - 2024-11-01
### Added
- Added language support for 10 additional models in the recognition module.

## [1.0.1] - 2024-10-28
### Added
- Added support for detecting polygonal bounding boxes in `visualize_detection`.
- Introduced the `show` argument in `visualize_detection` to control image display.

