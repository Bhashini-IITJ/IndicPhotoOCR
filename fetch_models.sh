#!/usr/bin/env bash
# Downloads the 14 ONNX models (~324 MB zip, ~442 MB unpacked) into the install-time asset pack.
# They are kept out of Git so that cloning IndicPhotoOCR stays small; manifest.json (thresholds, charsets) is in Git.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
VERSION="2.0.0"
URL="${MODELS_URL:-https://github.com/Bhashini-IITJ/IndicPhotoOCR/releases/download/android-v${VERSION}/indicphotoocr-android-models-${VERSION}.zip}"
SHA256="9a127a422b8201944ab3728c412518a36de63cb97feffd88500cb2a440789aa0"
DEST="$HERE/android/model_pack/src/main/assets/models"

if [ "$(ls "$DEST"/*.onnx 2>/dev/null | wc -l)" -ge 14 ] && [ "$1" != "--force" ]; then
  echo "Models already present in $DEST (use --force to download again)."; exit 0
fi
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
echo "Downloading models $VERSION ..."
curl -fL --retry 3 -o "$TMP/models.zip" "$URL"
echo "$SHA256  $TMP/models.zip" | sha256sum -c -
python3 -c "import zipfile,sys;zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$TMP/models.zip" "$DEST"
echo "Models ready: $(ls "$DEST"/*.onnx | wc -l) files in $DEST"
