import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.iitj.indicphotoocr.app"
    compileSdk = 36
    defaultConfig {
        applicationId = "com.vl2g.indicphotoocr"  // permanent Play ID (sir, 9 Nov 2026); code package stays com.iitj...
        minSdk = 26          // Android 8.0+ (7.x and older: ~2% of active devices, Dec 2025)
        targetSdk = 36           // Google Play: new apps and updates must target API 36 from 31 Aug 2026
        versionCode = 14
        versionName = "2.0.0"
        // 64-bit ARM phones only: x86_64 is for emulators, and 32-bit-only phones (Android Go, 1-2 GB RAM) cannot hold the models
        ndk { abiFilters += listOf("arm64-v8a") }
        // overlap pruning: -PpruneMode=d12 (default) or d123 (adds D3 re-reads; installs next to the d12 app)
        val pruneMode = (project.findProperty("pruneMode") as String?) ?: "d12"
        buildConfigField("String", "PRUNE_MODE", "\"$pruneMode\"")
        if (pruneMode == "d123") applicationIdSuffix = ".d3"
        manifestPlaceholders["appLabel"] = if (pruneMode == "d123") "IndicPhotoOCR 2.0 D3" else "IndicPhotoOCR 2.0"
    }
    // Play upload key: read from ~/.indicphotoocr-signing.properties (storeFile, storePassword, keyAlias, keyPassword),
    // or the file named by INDICPHOTOOCR_SIGNING. The key and password never go into Git. Without the file, release
    // builds are signed with the debug key (fine for sideloading, rejected by Google Play).
    val signingFile = file(System.getenv("INDICPHOTOOCR_SIGNING") ?: "${System.getProperty("user.home")}/.indicphotoocr-signing.properties")
    val uploadSigning = if (signingFile.isFile) Properties().apply { signingFile.inputStream().use { load(it) } } else null
    signingConfigs {
        if (uploadSigning != null) create("upload") {
            storeFile = file(uploadSigning.getProperty("storeFile"))
            storePassword = uploadSigning.getProperty("storePassword")
            keyAlias = uploadSigning.getProperty("keyAlias")
            keyPassword = uploadSigning.getProperty("keyPassword")
        }
    }
    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName(if (uploadSigning != null) "upload" else "debug")
        }
    }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
    androidResources { noCompress += listOf("onnx") }
    // The 14 models (~420 MB) live in an install-time asset pack: Google Play allows at most 200 MB for the base
    // module, but up to 1.5 GB per asset pack. Install-time packs are installed together with the app (still one
    // install, fully offline) and their files are read through the same AssetManager as the app's own assets.
    assetPacks += listOf(":model_pack")  // models are copied out once; storing them uncompressed is faster
    buildFeatures { viewBinding = true; buildConfig = true }
}

dependencies {
    implementation(project(":ocrcore"))
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.exifinterface:exifinterface:1.3.7")
    implementation("com.google.android.material:material:1.12.0")
}
