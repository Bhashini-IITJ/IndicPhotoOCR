# Changelog

## [1.4.0] - 2026-10-11
### Added
- YOLO11n text detector fine-tuned on the BSTD train split (`detector="yolo"`), downloaded from a release asset with SHA-256 verification
- `--detector` option for `end-to-end-Inference.py` (default `yolo`; `textbpn` and `east` selectable)
- `detector_checkpoint` argument for a local YOLO checkpoint
- Unit tests for the YOLO detector and detector selection

### Changed
- Default detector of `OCR()` and `end-to-end-Inference.py` is now the fine-tuned YOLO11n (was TextBPN++ / EAST)
- Dockerfile installs the package from the build context instead of cloning a fixed repository
- `ultralytics==8.3.227` added to the package requirements

### Fixed
- `make run` mounted an empty path (`$(pwd)` in Makefile); it now mounts the checkout at `/workspace/IndicPhotoOCR`
- `make build` no longer requires BuildKit/buildx (plain `docker build`); new `make test` target; image name overridable with `IMAGE=`
- `end-to-end-Inference.py` skips sub-directories of the image folder instead of logging them as failed images
- README: Docker instructions and the clone URL of this repository

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

