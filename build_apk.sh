#!/usr/bin/env bash
# Builds the app the way Google Play wants it, plus one installable APK for testing:
#   1. android/app/build/outputs/bundle/release/app-release.aab   -> upload this to Google Play
#   2. IndicPhotoOCR2.apk (next to this script)                    -> single APK for sideloading on a phone
# The models are in an install-time asset pack (:model_pack), so a plain `gradlew assembleRelease` APK would NOT
# contain them; the installable APK is made from the bundle with bundletool (--mode=universal).
# Test builds are signed with the debug key (~/.android/debug.keystore, created by the first Gradle build).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -x "$HERE/toolchain/jdk/bin/java" ] && export JAVA_HOME="$HERE/toolchain/jdk"
[ -d "$HERE/toolchain/sdk" ] && export ANDROID_HOME="$HERE/toolchain/sdk"
JAVA="${JAVA_HOME:+$JAVA_HOME/bin/}java"
BUNDLETOOL="$HERE/toolchain/bundletool.jar"

bash "$HERE/fetch_models.sh"   # the models are not in Git; downloads them once
cd "$HERE/android"; chmod +x gradlew
./gradlew :app:bundleRelease "$@"
AAB="$HERE/android/app/build/outputs/bundle/release/app-release.aab"

if [ ! -f "$BUNDLETOOL" ]; then
  mkdir -p "$HERE/toolchain"
  curl -sSL -o "$BUNDLETOOL" https://github.com/google/bundletool/releases/download/1.18.3/bundletool-all-1.18.3.jar
fi
KS="$HOME/.android/debug.keystore"
OUT="$(mktemp -d)"
"$JAVA" -jar "$BUNDLETOOL" build-apks --mode=universal --bundle="$AAB" --output="$OUT/app.apks" \
  --ks="$KS" --ks-pass=pass:android --ks-key-alias=androiddebugkey --key-pass=pass:android
(cd "$OUT" && python3 -c "import zipfile;zipfile.ZipFile('app.apks').extract('universal.apk','.')")
mv "$OUT/universal.apk" "$HERE/IndicPhotoOCR2.apk"; rm -rf "$OUT"
ls -la "$AAB" "$HERE/IndicPhotoOCR2.apk"
