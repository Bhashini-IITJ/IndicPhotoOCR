# IndicPhotoOCR 2.0 app: licence audit

Updated 7 Nov 2026 (build 7.5). This is an engineering audit, not legal advice. Publishing: personal Play developer account (sir's decision, 7 Nov 2026), no institute or Bhashini logos, text credit only.

## What the status colours mean

| Status | Meaning |
|---|---|
| **RED** | **Blocks the Play Store release.** Publishing before it is solved risks a licence violation or a rejection by Google. |
| **AMBER** | **Not a blocker, but unconfirmed or legally grey.** Confirm it, or keep doing the safe practice listed, before release. |
| **GREEN** | **Fine.** The licence allows our use, and the required credit/notice is already in the app (side menu → Open-source licences). |

## Summary

| # | Component | Licence | Status | What to do |
|---|---|---|---|---|
| 1 | **YOLO11n text detector** in the app (`yolo.onnx`, fine-tuned with Ultralytics YOLO on BSTD + MIST) | **AGPL-3.0**: Ultralytics' position is that models trained with their code are AGPL-3.0 | **GREEN** | Done: source public since 7 Nov 2026. From 14 Nov 2026 (Anik's request) the official source is the **`android` branch of https://github.com/Bhashini-IITJ/IndicPhotoOCR** (AGPL-3.0 `LICENSE` in the branch; `main` stays MIT); the models are a release asset (`android-v2.0.0`). The app links there from the next build (2.0.0 links to Asadalk/IndicPhotoOCR-Android, which stays public as a mirror with the APK releases). Push every future release's source to the branch. |
| 2 | IIT Jodhpur and Bhashini logos | Trademarks of IIT Jodhpur and of the Digital India Bhashini Division (MeitY) | **GREEN** | Removed in 7.5 (sir: avoid logos; "Supported by Bhashini" is fine). The app now shows only text: "Created at VL2G, IIT Jodhpur · Supported by Bhashini". Terms item 7 says neither publishes the app. Keep logos out of screenshots and store graphics too. **GitHub README:** logos allowed (14 Nov 2026: sir's "no logo" was only about Google Play rejecting it), shown in the README Credits section with links (`store/logo_iitj(_dark).png`, `store/logo_bhashini(_dark).png`; dark versions made by `tools/make_dark_logos.py`). |
| 3 | Starting checkpoint of the detector: `yolo11n-text-best.pt` = Hugging Face **`RoyRud1902/yolo11n-text`** (`best.pt`, byte-identical, SHA-256 `ce26dca3…2a20`). Our detector copied only its backbone, layers 0–16; the head was trained fresh. | **Apache-2.0** (model card). Its training data **DonkeySmall/Yolo-Text-Detection** is **MIT**. | **GREEN** | Source and licence are confirmed. Apache-2.0 asks for credit: add it to the in-app licence notice (already added to `NOTICE.txt`, ships in the next build). It was itself trained with Ultralytics, so it falls under item 1 too, which open-sourcing solves. |
| 4 | MIST training data (used for the detector and for the script-ID student) | Not stated in our copy | **GREEN** | Cleared: sir is in direct contact with MIST's author, so its use is not an issue (user, 7 Nov 2026). |
| 5 | BSTD (Bharat Scene Text Dataset), used to train all models. **IIT Jodhpur's own dataset** (Bhashini Team, IITJ), so the publisher of the app is the dataset's owner. | Annotations: IITJ's, released as Apache-2.0. Photos: Wikimedia Commons contributors, **CC BY-SA 4.0** | **GREEN** | Nothing to ask anyone. The photos themselves belong to Wikimedia contributors, so keep the credit that is already in the app (7.4). IITJ already releases BSTD-trained models publicly (IndicPhotoOCR), so the app does the same thing. |
| 6 | PARSeq readers (12 `parseq_*.onnx`), trained by IndicPhotoOCR | IndicPhotoOCR: **MIT** (Bhashini Team, IIT Jodhpur); PARSeq architecture/code: **Apache-2.0** | **GREEN** | Notices are in the app (7.4). |
| 7 | Script-ID student (EfficientNet-B0, ImageNet start weights from torchvision, distilled from our EfficientNetV2-L) | torchvision: **BSD-3-Clause** | **GREEN** | Notice is in the app (7.4). |
| 8 | ONNX Runtime 1.22.0 (Android) | **MIT** | **GREEN** | Notice is in the app (7.4). |
| 9 | AndroidX (core, appcompat, activity, exifinterface), Material Components, Kotlin stdlib | **Apache-2.0** | **GREEN** | Notice is in the app (7.4). |
| 10 | App icon (from 7.3): letter shapes from the Lohit Devanagari (अ) and DejaVu Sans Bold (A) fonts | SIL OFL 1.1 / Bitstream Vera licence | **GREEN** | Both licences allow use in a logo. No in-app notice needed; this file is the record. |
| 10b | README wordmark "IndicPhotoOCR 2.0" (`store/logo_indicphotoocr2*.png`, 14 Nov 2026): the original IndicPhotoOCR logo + "2.0" set in Kalam Light | Original logo: IndicPhotoOCR repo, **MIT** (Bhashini Team, IIT Jodhpur). Kalam: **SIL OFL 1.1** | **GREEN** | Both allow this use. The font file is not shipped, only the rendered image. |
| 11 | Our own app code (Kotlin, tools) | Ours (VL2G, IIT Jodhpur) | — | Release it under **AGPL-3.0** together with the app (simplest way to satisfy item 1). |

**Nothing blocks the release: every item is GREEN.** Remaining steps are Play Console work (see `PLAY_STORE_GUIDE.md`).

## What the plan is

1. **Open-source the app under AGPL-3.0**: done. Official source: `https://github.com/Bhashini-IITJ/IndicPhotoOCR/tree/android`; mirror and APK releases: `https://github.com/Asadalk/IndicPhotoOCR-Android`.
   - Push the contents of `IndicPhotoOCR2-app-source-7.5.zip`, with a `LICENSE` file holding the AGPL-3.0 text at the repo root (`assets/licenses/AGPL-3.0.txt` is that text).
   - The 14 ONNX models are each under GitHub's 100 MB file limit (largest ~35 MB, ~420 MB in total), so they can be pushed normally; attaching them to a GitHub Release instead keeps the repo small.
   - All other parts (MIT, Apache-2.0, BSD-3, OFL) are compatible with AGPL-3.0.
2. Logos: done (removed in 7.5).
3. MIST: done (cleared by sir).

### Free / non-profit does not remove the AGPL duties
AGPL-3.0 has no exception for free or non-commercial apps. Its duties start when the software is distributed, and a Play Store release is distribution. So the source must be public even though the app is free.

### Where the licence goes
- **Main place:** the `LICENSE` file in the public GitHub repo.
- **In the app (done in 7.4):** side menu → Open-source licences. It shows what each component is, its licence and the source-code link, then the full AGPL-3.0, MIT, BSD-3 and Apache-2.0 texts. These ship inside the APK (`assets/licenses/`).
- **Optional:** one line in the Play Store description: "Open source (AGPL-3.0): <repo link>".

## Facts checked for this audit

- **Checkpoint source** (handoff log, 7 Nov 2026): `best.pt` from `huggingface.co/RoyRud1902/yolo11n-text` is byte-identical to our `faisal_storage/weights_backup/yolo11n-text-best.pt` (SHA-256 `ce26dca363a67fe96c09cc51bed5d09de887239ad5f8a1bd71ee216c2f602a20`). The Hugging Face API lists that model's licence as `apache-2.0`, and the dataset `DonkeySmall/Yolo-Text-Detection` as `mit`.
- **Layers copied from the checkpoint** (handoff, 15 Oct 2026): only layers 0–16 (288 tensors) were copied from it; the detection head was trained from scratch.
- **Permissions** (from the built 7.4 APK): no INTERNET, CAMERA or storage permission. Photos come through the system camera app or photo picker. Play "Data safety" answer: **no data collected, no data shared**.
- **IndicPhotoOCR licence:** `IndicPhotoOCR/LICENSE` is MIT, "Copyright (c) 2024 Bhashini Team@IIT Jodhpur".
- **BSTD licence:** BSTD is IIT Jodhpur's own dataset (confirmed by the user, 7 Nov 2026). `IndicPhotoOCR/BharatSceneTextDataset/LICENSE` is Apache-2.0; its README says all images are from Wikimedia Commons under `cc-by-sa-4.0`.
- **Ultralytics:** models trained with Ultralytics are AGPL-3.0, and closed/commercial use needs an Enterprise licence (https://www.ultralytics.com/license).
