# Publishing IndicPhotoOCR 2.0 on Google Play: step by step

Written 7 Nov 2026, for a first-time publisher. Google changes these rules often, so check each step against Play Console when you get there.

## 0. What it costs (the "bill" sir asked for)

| Item | Cost | Notes |
|---|---|---|
| Google Play developer registration (personal account) | **US$25, once** (about ₹2,200; the exact amount depends on the exchange rate; paid by card) | The only Google fee. No yearly fee, nothing more for a free app. |
| Ultralytics Enterprise licence | Not needed | The app is open-sourced under AGPL-3.0 instead (see `LICENCE_AUDIT.md`). |
| Hosting the privacy policy page | Free | GitHub Pages in the app's repo. |

**Total: US$25, one time.**

## 1. Account type: personal (decided)

Sir's decision (7 Nov 2026): a **personal developer account**, to avoid administrative work at IIT Jodhpur. No logos in the app; the text "Supported by Bhashini" is fine.

What a personal account means:
- **Needs:** a government ID (PAN/Aadhaar/passport) matching the account holder's name, a card for the US$25, and a phone number. No D-U-N-S number.
- **Before going public:** a **closed test with at least 12 testers who stay opted in for 14 days in a row** (rule for personal accounts created after 13 Nov 2023). Only then can you apply for production access. Collect 12+ Gmail IDs from the group and friends early.
- **Shown on Play:** the developer name you enter and a contact email. Choose a developer name that does not suggest an institute publishes the app (for example your own name, or "IndicPhotoOCR"); the app itself credits "VL2G, IIT Jodhpur" in text only.
- **Logos:** none in the app (removed in 7.5), and keep them out of screenshots and store graphics.

## 2. Create the developer account

1. Go to https://play.google.com/console/signup, sign in with the Google account that will own the app, choose **Personal**, and pay US$25.
2. Complete identity verification (ID + address); it takes a few days.
3. Verify a real Android phone (the Play Console app on the phone does this).

## 3. Technical state of the app (updated 7 Nov 2026, build 7.5)

**Already done:**
- **App bundle (.aab):** `build_apk.sh` makes `app-release.aab` for Play plus a single test APK.
- **Size:** the 14 models are in an **install-time asset pack** (`android/model_pack`). Download sizes: base ~14 MB (limit 200 MB), model pack ~335 MB (limit 1.5 GB per pack, 4 GB total). Still one install, fully offline.
- **Target API 36 (Android 16):** compileSdk/targetSdk 36, AGP 8.11.2, Gradle 8.13.
- **16 KB page-size support:** ONNX Runtime 1.22.0.
- **Terms of Service, Privacy Policy and Open-source licences** screens with real text.
- **App ID (permanent): `com.vl2g.indicphotoocr`** (sir, 9 Nov 2026; replaces 7.5's `io.github.asadalk.indicphotoocr`). Each ID change installs as a new app next to older test builds; uninstall those. Never change it after the first Play upload.
- **No logos**; credit line "Created at VL2G, IIT Jodhpur · Supported by Bhashini" (7.5).
- **Source link** in the app: 2.0.0 → `https://github.com/Asadalk/IndicPhotoOCR-Android`; next build → `https://github.com/Bhashini-IITJ/IndicPhotoOCR/tree/android` (official source from 14 Nov 2026).

**Still to do:**
1. ~~Push the source~~: done. Official: `android` branch of Bhashini-IITJ/IndicPhotoOCR (models as release asset `android-v<version>`); APK releases: Asadalk/IndicPhotoOCR-Android. Push each new release's source to the branch.
2. **Upload key + Play App Signing.** Make the key on the machine that will do releases:
   `keytool -genkeypair -v -keystore upload.jks -keyalg RSA -keysize 4096 -validity 10000 -alias upload`
   - Keep `upload.jks` and its password safe, never commit them, and keep a backup.
   - Add a `signingConfigs.release` that reads them in `app/build.gradle.kts` (I can do this when the key exists).
   - In Play Console, accept **Play App Signing**: Google keeps the final app-signing key.
3. `versionName` without "alpha" (e.g. `2.0.0`) for the public release.

## 4. Create the app in Play Console

Play Console → **Create app**:
- App name: `IndicPhotoOCR 2.0` (17 of max 30 characters; same as the name under the app icon)
- Default language: English (India)
- App, Free
- Accept the declarations

Then complete **every item under "App content"**:

| Item | What to answer |
|---|---|
| Privacy policy | URL of a public page with the app's privacy text (section 6). Mandatory even for offline apps. |
| App access | "All functionality is available without special access" (no login). |
| Ads | No ads. |
| Content rating | Fill in the IARC questionnaire (category: Utility/Productivity, no violence etc.). Expect "Everyone / 3+". |
| Target audience | 13+ or 18+ (choosing under-13 brings the Families policy; avoid unless needed). |
| Data safety | **No data collected, no data shared.** The app has no internet permission, and photos are processed on the phone only. |
| Government app | **No.** It is not published on behalf of a government body; it only says "Supported by Bhashini". |
| Financial features, health, news | None. |

## 5. Store listing (Grow → Store presence → Main store listing)

Prepared in `store/` (graphics generated by `tools/make_icon.py`; all listing text, ready to paste, in `store/listing.txt`):
- **App icon** 512×512 PNG: `store/icon_512.png` ✔
- **Feature graphic** 1024×500 PNG: `store/feature_graphic_1024x500.png` ✔
- **Phone screenshots:** at least 2 (recommended 4–8), 16:9 or 9:16, min 320 px side. Take them on the Pixel 9 with real photos: start screen, a sign with boxes, the text result, the side menu.

Draft text:

- **Short description** (≤80 chars): `Read text in photos in English and 11 Indian languages, fully offline.`
- **Full description** (≤4000 chars):

  > IndicPhotoOCR reads text from photos of signs, boards, shop fronts and pages, in English and 11 Indian languages: Hindi, Bengali, Marathi, Tamil, Telugu, Kannada, Malayalam, Gujarati, Punjabi, Odia and Assamese.
  >
  > • Works completely offline. Your photos never leave your phone.
  > • Take a photo or pick one from your gallery, or share an image to the app.
  > • See every word that was found, then copy or share the text.
  > • Uses the phone's GPU/NPU when available.
  >
  > Created by people at the Vision, Language and Learning Group (VL2G), IIT Jodhpur. Supported by Bhashini.

  (Sir approved the "Supported by Bhashini" line.)
- Category: **Tools** (or Productivity). Contact email: required, and shown publicly.

## 6. Privacy policy and Terms of Service

Both texts are written and already in the app (7.4): `app/src/main/res/values/strings.xml → privacy_body, terms_body`. Summary of the privacy text:
- Everything runs on the phone, and the app has no internet permission.
- Nothing is collected: no photos, text, location, identifiers, analytics or ads.
- A camera photo is kept only in the app's private temporary folder and is replaced by the next photo.
- The only setting stored is which processor is fastest.

What is left:
- **Host the same privacy text at a public URL** (GitHub Pages in the app's repo is free, or a page on iitj.ac.in) and give that URL in Play Console.
- Show both texts to sir once. If the institute wants a named contact email instead of "the repository's issues page", change the last point of each text.

## 7. Testing tracks, then production

1. **Internal testing** (up to 100 testers, available in minutes): upload the .aab, add the group's Gmail IDs, and install from the Play link. This is where we check that the asset pack works.
2. **Closed testing (required for our personal account):** at least 12 testers opted in for 14 continuous days. Then apply for production access in Play Console (Google asks a few questions about the test).
3. **Production**:
   - Countries: India (or all).
   - Release notes: "First release".
   - Submit for review. The first review usually takes a few days and can take up to ~7.
4. Updates later: raise `versionCode`, build a new .aab with the same upload key, and upload to the same app.

## 8. Checklist before pressing "Publish"

- [x] Source public at `Asadalk/IndicPhotoOCR-Android` with the AGPL-3.0 `LICENSE` (7 Nov 2026)
- [x] Logos removed, "Supported by Bhashini" text only (7.5)
- [x] Detector starting checkpoint source and licence confirmed (RoyRud1902/yolo11n-text, Apache-2.0)
- [x] MIST use cleared by sir
- [x] Open-source licences screen, real Terms and Privacy text in the app; privacy page `docs/privacy.html` pushed; [x] GitHub Pages on → URL https://asadalk.github.io/IndicPhotoOCR-Android/privacy.html
- [x] Final app ID `com.vl2g.indicphotoocr` (first build with it: 7.6)
- [x] .aab with install-time model pack, targetSdk 36; [x] signed with the upload key (2.0.0, versionCode 14)
- [x] Personal developer account created and verified (developer name "faisalyst")
- [x] Closed test live (13 Nov 2026; installed from Play and checked: models load, text reads)
- [ ] 12+ testers opted in for 14 days in a row, on different phones (low-end too); then Apply for production
- [x] Screenshots (no logos), descriptions, content rating, data safety done (text: `store/listing.txt`)
- [x] versionName without "alpha" (`2.0.0`)

## Sources

- Target API 36 from 31 Aug 2026: https://developer.android.com/google/play/requirements/target-sdk
- Size limits (200 MB base, 1.5 GB per asset pack, 4 GB total): https://support.google.com/googleplay/android-developer/answer/9859372 and https://developer.android.com/guide/app-bundle/faq
- 12 testers / 14 days for new personal accounts: https://support.google.com/googleplay/android-developer/answer/14151465
- Ultralytics licensing: https://www.ultralytics.com/license
