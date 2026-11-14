#!/usr/bin/env bash
# Installs everything needed to build the app into ./toolchain (no root, nothing system-wide), Linux x86_64:
# JDK 17, Android SDK (platform 36, build-tools 36.0.0) and bundletool. Then writes android/local.properties.
# On macOS/Windows use Android Studio instead (see README.md).
# Usage: bash setup_toolchain.sh        (needs internet once; ~1.5 GB download)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
T="$HERE/toolchain"; mkdir -p "$T"; cd "$T"

if [ ! -x jdk/bin/java ]; then
  echo "[1/4] JDK 17 (Eclipse Temurin)"
  curl -sSL -o jdk.tar.gz "https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse"
  mkdir -p jdk && tar -xzf jdk.tar.gz -C jdk --strip-components=1 && rm jdk.tar.gz
fi
export JAVA_HOME="$T/jdk"

if [ ! -d sdk/platforms/android-36 ]; then
  echo "[2/4] Android SDK (platform 36, build-tools 36.0.0)"
  if [ ! -x sdk/cmdline-tools/latest/bin/sdkmanager ]; then
    curl -sSL -o cmdline.zip "https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip"
    mkdir -p sdk/cmdline-tools && python3 -c "import zipfile;zipfile.ZipFile('cmdline.zip').extractall('sdk/cmdline-tools')"
    mv sdk/cmdline-tools/cmdline-tools sdk/cmdline-tools/latest && rm cmdline.zip
    chmod +x sdk/cmdline-tools/latest/bin/*
  fi
  yes | sdk/cmdline-tools/latest/bin/sdkmanager --sdk_root="$T/sdk" --licenses > /dev/null
  sdk/cmdline-tools/latest/bin/sdkmanager --sdk_root="$T/sdk" "platform-tools" "platforms;android-36" "build-tools;36.0.0"
fi

if [ ! -f bundletool.jar ]; then
  echo "[3/4] bundletool 1.18.3"
  curl -sSL -o bundletool.jar https://github.com/google/bundletool/releases/download/1.18.3/bundletool-all-1.18.3.jar
fi

echo "[4/4] android/local.properties"
echo "sdk.dir=$T/sdk" > "$HERE/android/local.properties"
echo "Done. Build with:  bash build_apk.sh"
