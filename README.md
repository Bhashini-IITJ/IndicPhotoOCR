<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="store/logo_indicphotoocr2_dark.png">
    <img src="store/logo_indicphotoocr2.png" width="520" alt="IndicPhotoOCR 2.0">
  </picture>
</p>

<p align="center">Read text in photos in English and 11 Indian languages, fully offline, on Android.</p>

<p align="center">
  <a href="https://github.com/Asadalk/IndicPhotoOCR-Android/releases/latest"><img src="https://img.shields.io/github/v/release/Asadalk/IndicPhotoOCR-Android?label=Download%20APK&logo=android&logoColor=white&color=3DDC84&style=for-the-badge" alt="Download the APK"></a>
  <a href="https://apps.obtainium.imranr.dev/redirect?r=obtainium://add/https://github.com/Asadalk/IndicPhotoOCR-Android"><img src="https://img.shields.io/badge/Obtainium-Get%20it-6750A4?style=for-the-badge" alt="Get it on Obtainium"></a>
  <img src="https://img.shields.io/badge/Google%20Play-coming%20soon-414141?style=for-the-badge&logo=googleplay&logoColor=white" alt="Google Play: coming soon">
</p>

IndicPhotoOCR 2.0 finds and reads the text in photos of signboards, shop fronts, notices, labels and pages, in **English, Hindi, Bengali, Marathi, Gujarati, Punjabi, Odia, Assamese, Tamil, Telugu, Kannada and Malayalam**.

- **Offline and private.** Everything runs on the phone. The app has no internet permission: no account, no ads, no tracking, nothing collected ([privacy policy](https://asadalk.github.io/IndicPhotoOCR-Android/privacy.html)).
- **Simple.** Take a photo, pick one from the gallery, or share an image to the app. Found words are boxed on the photo, and the text can be copied or shared.
- **Fast where it can be.** Uses the phone's GPU/NPU for detection when that is faster than the CPU.

Developed by people at [VL2G](https://vl2g.github.io/), [IIT Jodhpur](https://www.iitj.ac.in/). Supported by [Bhashini](https://bhashini.gov.in/).

## Install

| Where | How | Updates |
|---|---|---|
| **Google Play** | Coming soon: in closed testing now; the public listing (`com.vl2g.indicphotoocr`) opens after the test. The link will be added here. | automatic |
| **GitHub Releases** | Download `IndicPhotoOCR-<version>.apk` from the [latest release](https://github.com/Asadalk/IndicPhotoOCR-Android/releases/latest), open it on the phone and allow "Install unknown apps". | manual |
| **[Obtainium](https://obtainium.imranr.dev)** | Tap the Obtainium badge above on the phone, or in Obtainium add the URL `https://github.com/Asadalk/IndicPhotoOCR-Android`. | from GitHub Releases |

The release APK is the same Google-signed app as on Google Play, so you can switch between Play, GitHub and Obtainium without uninstalling.

**Requirements:** Android 8.0 or newer, a 64-bit ARM phone (almost all phones since 2017), about 500 MB free space; 4 GB RAM or more recommended. The download is large (~335 MB from Play, ~465 MB as one APK) because the 14 recognition models are inside the app; that is what lets it work offline.

**F-Droid:** not available. F-Droid builds every app from source on its own servers, and this app depends on prebuilt ONNX Runtime binaries and ~420 MB of trained models, which do not fit its rules. Obtainium is the open-source alternative.

## Credits

<table>
  <tr>
    <td align="center" width="50%"><b>Developed at</b></td>
    <td align="center" width="50%"><b>Supported by</b></td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://www.iitj.ac.in/"><picture>
        <source media="(prefers-color-scheme: dark)" srcset="store/logo_iitj_dark.png">
        <img src="store/logo_iitj.png" height="110" alt="IIT Jodhpur">
      </picture></a><br>
      <a href="https://vl2g.github.io/">Vision, Language and Learning Group (VL2G)</a><br>IIT Jodhpur
    </td>
    <td align="center">
      <a href="https://bhashini.gov.in/"><picture>
        <source media="(prefers-color-scheme: dark)" srcset="store/logo_bhashini_dark.png">
        <img src="store/logo_bhashini.png" height="110" alt="Bhashini">
      </picture></a><br>
      <a href="https://bhashini.gov.in/">Bhashini</a><br>Digital India Bhashini Division, MeitY
    </td>
  </tr>
</table>

IndicPhotoOCR 2.0 builds on [IndicPhotoOCR](https://github.com/Bhashini-IITJ/IndicPhotoOCR) and the Bharat Scene Text Dataset (BSTD) from IIT Jodhpur.

---

# For developers

This is the **`android` branch** of [IndicPhotoOCR](https://github.com/Bhashini-IITJ/IndicPhotoOCR): the complete source code of the IndicPhotoOCR 2.0 Android app, kept separate from the Python package on `main`. It has the app source, the build scripts, the Play Store graphics and the docs; the 14 ONNX models (~324 MB) are downloaded by `fetch_models.sh` from the release [`android-v2.0.0`](https://github.com/Bhashini-IITJ/IndicPhotoOCR/releases/tag/android-v2.0.0), so cloning IndicPhotoOCR stays small.

**Licence of this branch: AGPL-3.0** (see `LICENSE`; it is required because the text detector was trained with Ultralytics YOLO). The rest of IndicPhotoOCR stays MIT. Other components and their licences are listed in the app under *Open-source licences* and in `docs/LICENCE_AUDIT.md`. APK releases for users are published at [Asadalk/IndicPhotoOCR-Android](https://github.com/Asadalk/IndicPhotoOCR-Android/releases), which also hosts the privacy policy page.

## 1. Quick start (Linux x86_64)

```bash
git clone -b android --single-branch https://github.com/Bhashini-IITJ/IndicPhotoOCR.git IndicPhotoOCR-Android && cd IndicPhotoOCR-Android
bash setup_toolchain.sh     # once: JDK 17 + Android SDK 36 + bundletool into ./toolchain (~1.5 GB, needs internet)
bash build_apk.sh           # first run: downloads the models (fetch_models.sh) and Gradle 8.13
```

Outputs:
- `IndicPhotoOCR2.apk`: one installable APK for testing. Copy it to the phone and install it (allow "install unknown apps").
- `android/app/build/outputs/bundle/release/app-release.aab`: the app bundle for Google Play.

**Important:** the models are in a separate install-time asset pack (`android/model_pack`, needed for Google Play's 200 MB limit). Because of that, a plain `./gradlew assembleRelease` APK does **not** contain the models, and the app will say "Could not load the models". Always make test APKs with `build_apk.sh`, which runs bundletool `--mode=universal`.

The first build downloads Gradle and the Maven dependencies, so it needs internet and takes ~5–10 min. Later builds take ~1–2 min. Building needs ~4 GB RAM (`android/gradle.properties` sets `-Xmx4g`).

### Android Studio (Windows / macOS / Linux)
Open the `android/` folder in Android Studio (Narwhal 2025.1 or newer, for Android Gradle Plugin 8.11). Let it sync. Studio will create `local.properties` with your SDK path itself. Then use **Build → Generate App Bundles or APKs → Generate Bundles**, or press Run with a phone connected over USB (developer options → USB debugging); Run deploys the asset pack too. To make a single APK from the bundle, use bundletool as in `build_apk.sh`. Use JDK 17 (Settings → Build Tools → Gradle → Gradle JDK).

### Installing straight to a connected phone
```bash
toolchain/sdk/platform-tools/adb install -r IndicPhotoOCR2.apk
```

---

## 2. What is in the repository

```
IndicPhotoOCR-Android/
├── README.md                  this file
├── setup_toolchain.sh         installs JDK 17 + Android SDK locally (Linux)
├── fetch_models.sh            downloads the 14 ONNX models (release asset, SHA-256 checked) into model_pack
├── build_apk.sh               builds the .aab (for Play) and one installable test APK (runs fetch_models.sh first)
├── android/                   the Gradle project (open this in Android Studio)
│   ├── settings.gradle.kts    modules :app, :ocrcore, :verify, :model_pack
│   ├── build.gradle.kts       plugin versions (AGP 8.11.2, Kotlin 2.0.21)
│   ├── gradlew, gradle/       Gradle 8.13 wrapper
│   ├── model_pack/            install-time asset pack: src/main/assets/models/ = manifest.json (in Git) + the 14 ONNX models (fetch_models.sh)
│   ├── app/                   the Android app (UI)
│   │   ├── build.gradle.kts   versionCode/versionName, minSdk 26, targetSdk 36, arm64 only, ONNX Runtime 1.22.0
│   │   └── src/main/
│   │       ├── kotlin/com/iitj/indicphotoocr/app/
│   │       │   ├── MainActivity.kt   camera/gallery, progress, drawing boxes, result card, side menu
│   │       │   ├── InfoActivity.kt   About Us / Terms of Service / Privacy Policy / Open-source licences screens
│   │       │   ├── ModelStore.kt     loads models straight from the APK; saves the accelerator choice
│   │       │   └── ReadingOrder.kt   puts recognised words into lines (top-to-bottom, left-to-right)
│   │       ├── res/                  layouts, strings (all app text), colours, icons, app logo
│   │       ├── assets/licenses/      NOTICE.txt + full licence texts shown in "Open-source licences"
│   │       └── AndroidManifest.xml
│   ├── ocrcore/               pure-Kotlin OCR engine shared by the app and :verify
│   │   └── src/main/kotlin/com/iitj/indicphotoocr/core/
│   │       ├── OcrEngine.kt   detection → script ID → reading → overlap pruning; accelerator tuning
│   │       └── RgbImage.kt    image resize/crop/normalise helpers (match the Python reference exactly)
│   └── verify/                runs the same engine on a desktop/server JVM to compare with Python
├── tools/                     Python helpers used to make the models and check the app (section 6)
├── store/                     Play Store icon (512×512), feature graphic (1024×500), listing text, README wordmark
└── docs/
    ├── PLAY_STORE_GUIDE.md    how to publish on Google Play, costs, checklist
    ├── LICENCE_AUDIT.md       licences of every model/library and how each is met
    ├── privacy.html, index.html  privacy policy page (GitHub Pages)
    └── CHANGELOG.md           what changed in each test build
```

---

## 3. How the app works

1. **Photo in.** From the camera (system camera app), the gallery, or "Share" from another app. It is decoded with EXIF rotation applied, long side ≤ 4096 px, and shown at once.
2. **Detection.** `yolo.onnx` (YOLO11n trained on BSTD + MIST at 1920 px) runs on a letterboxed 1920-px image. Boxes with conf ≥ 0.25 are kept, then NMS at IoU 0.7, ≤ 300 boxes. The photo is redrawn with white boxes.
3. **Script ID.** `script_id.onnx` (EfficientNet-B0, 224 px, distilled from our EfficientNetV2-L) assigns each box one of 12 scripts. This is used only to pick the reader; nothing is shown to the user (7.3).
4. **Reading.** Each word crop is read by the PARSeq model of its script (`parseq_<lang>.onnx`, int8).
5. **Overlap pruning.** Duplicate, fragment and phrase boxes of the same text are removed: base rule + D1 (cross-script pairs) + D2 (per-script confidence percentile, table in `manifest.json → "prune"`). D3 (re-reading jittered crops) is in the code but off, because it is too slow on phones. Select it with `-PpruneMode=d123`; the default is `d12`.
6. **Result.** The final boxes are drawn and the words are put into reading order. The text card has Copy and Share buttons.

**Loading:** the models are memory-mapped straight from the installed asset-pack APK (stored uncompressed: `noCompress "onnx"`; if a file ever arrives compressed, `ModelStore` reads it into memory instead). Hindi and English load and warm up when the app opens. The other 10 readers preload in the background on phones with ≥ 5.5 GB RAM; otherwise they load on first use, with at most 3 kept.

**Accelerators:** on first launch after an install/update, `OcrEngine.tuneAccelerators` tries NNAPI (GPU/NPU, fp32 and fp16) and XNNPACK for detection and script ID on a synthetic test image. One is kept only if it gives the same boxes/scripts as the CPU and is ≥ 10 % faster. The readers always run on the CPU. About Us shows what is in use.

---

## 4. Models (`android/model_pack/src/main/assets/models/`)

| File | What | Size |
|---|---|---|
| `yolo.onnx` | text detector, YOLO11n BSTD+MIST 1920, fp32 | ~11 MB |
| `script_id.onnx` | 12-class script identifier, EfficientNet-B0, fp32 | ~16 MB |
| `parseq_<lang>.onnx` × 12 | PARSeq readers, int8 (MatMul only; the patch-embed Conv stays fp32) | ~33 MB each |
| `manifest.json` | class list, charsets, input sizes, normalisation, detector thresholds, pruning table | — |

The models are not in Git: run `bash fetch_models.sh` (build_apk.sh does it for you). To replace a model, put the new ONNX file under the same name and update `manifest.json` if its classes, charset or input size changed. The tools in section 6 show how the current files were made (they use the paths of the original server, so edit the paths at the top).

Accuracy of this exact pipeline (BSTD, official metric, average of 10 languages, P / R / F1 / WRR / CRR):
- validation (526 images): 59.26 / 70.80 / 62.33 / 45.94 / 57.69
- test (1,319 images, one run): 53.28 / 57.30 / 53.57 / 40.93 / 57.91

---

## 5. Common changes

| Want to… | Edit |
|---|---|
| Change any text in the app (tagline, About, Terms, Privacy) | `app/src/main/res/values/strings.xml` |
| Change the source-code link or licence list | `strings.xml → source_url`, `app/src/main/assets/licenses/NOTICE.txt` |
| Change colours | `app/src/main/res/values/colors.xml` |
| Change the screen layout | `app/src/main/res/layout/activity_main.xml`, `activity_info.xml`, `drawer_header.xml`, `menu/drawer.xml` |
| Change how boxes are drawn | `MainActivity.kt → drawPlainBoxes` |
| Change the icon | `python tools/make_icon.py` (needs `pip install fonttools pillow` and the Lohit Devanagari + DejaVu fonts), or replace `res/drawable/ic_launcher_foreground.xml` |
| Change the app logo / credit line | `res/drawable-nodpi/logo_indicphotoocr.png`; credit text = `strings.xml → credit_line, supported_by` (no institute/Bhashini logos, by decision) |
| Change detection threshold / NMS | `model_pack/src/main/assets/models/manifest.json → yolo.conf / yolo.iou` |
| Change pruning | `OcrEngine.kt → prune / pruneD12 / pruneD3`; keep `tools/dedup_boxes.py` in sync |
| App ID (permanent on Play) | `app/build.gradle.kts → applicationId` = `com.vl2g.indicphotoocr`; do not change after the first upload |
| **Make a new version** | `app/build.gradle.kts`: raise `versionCode` by 1 and set `versionName` |
| App name on the home screen | `app/build.gradle.kts → manifestPlaceholders["appLabel"]` |

---

## 6. Python tools (`tools/`)

These ran on the lab server (conda env `indicphotoocr`). They need the IndicPhotoOCR repo and the BSTD dataset, which are not in this repository.

| Script | Purpose |
|---|---|
| `export_onnx.py` | exports YOLO, the script identifier and the PARSeq readers to ONNX |
| `compress_models.py` | makes the fp16/int8 versions |
| `make_bundle.py`, `make_bundle_all.py` | assembles the models folder (canonical names + manifest); copy the result to `android/model_pack/src/main/assets/models/` |
| `reference_pipeline.py`, `reference_pipeline_all.py` | Python copy of the app pipeline; scores BSTD val with the official metric |
| `dedup_boxes.py` | Python copy of the app's overlap pruning (base rule) |
| `make_icon.py` | generates the app icon and the Play Store graphics |
| `make_logo2.py` | makes the "IndicPhotoOCR 2.0" wordmark: the original IndicPhotoOCR logo + "2.0" in Kalam Light (SIL OFL 1.1, Google Fonts) |

### Checking the Kotlin engine against Python (`:verify`)
```bash
cd android && ./gradlew :verify:installDist
PRUNE_MODE=d12 verify/build/install/verify/bin/verify ../path/to/models - <folder of .png images> out.json 8
```
This runs the exact app engine on a computer, so its output can be compared word by word with `reference_pipeline_all.py`.

---

## 7. Signing and release

- **Without a key**, release builds are signed with the debug key. That is fine for testing on your own phone, but such an APK cannot update the official app (different signature): uninstall the official app first.
- **Official builds** are signed with the upload key, read from `~/.indicphotoocr-signing.properties` (`storeFile`, `storePassword`, `keyAlias`, `keyPassword`; another path can be given in `INDICPHOTOOCR_SIGNING`). The key and its password are never committed. Google Play re-signs the app with its own app-signing key (Play App Signing).
- **Release steps:** raise `versionCode` and set `versionName` in `app/build.gradle.kts`, run `build_apk.sh`, upload `app-release.aab` to Play. For GitHub, download the *signed, universal APK* of that version from Play Console (App bundle explorer → Downloads) and attach it to a GitHub release named `v<versionName>`, so the GitHub APK has the same signature as the Play app.
- **If the models changed:** zip the 14 `.onnx` files, attach the zip to a new release `android-v<version>` in Bhashini-IITJ/IndicPhotoOCR (untick "Set as the latest release" so the Python package stays the latest), and update `VERSION` and `SHA256` in `fetch_models.sh`.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `SDK location not found` | run `setup_toolchain.sh`, or create `android/local.properties` with `sdk.dir=/path/to/Android/Sdk` |
| `Unsupported class file major version` / Kotlin JVM target errors | use **JDK 17** (`JAVA_HOME`) |
| Gradle out of memory | raise `-Xmx` in `android/gradle.properties` |
| `INSTALL_FAILED_UPDATE_INCOMPATIBLE` on the phone | an older build with another signing key is installed: uninstall it first |
| `INSTALL_FAILED_NO_MATCHING_ABIS` | the APK is arm64 only; emulators need `x86_64` added to `abiFilters` (+ ~40 MB) |
| App says "Could not load the models" | the APK was built with `assembleRelease` (no asset pack): use `build_apk.sh`; or the models are missing: run `bash fetch_models.sh` |
| Phone with < 4 GB RAM is slow or closes | expected: the models need ~1–1.5 GB; the readers are loaded lazily on such phones |

Source: https://github.com/Bhashini-IITJ/IndicPhotoOCR/tree/android (AGPL-3.0). VL2G, IIT Jodhpur.
