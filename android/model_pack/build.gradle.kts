// Install-time asset pack with the app's ONNX models (assets/models). See app/build.gradle.kts -> assetPacks.
plugins { id("com.android.asset-pack") }

assetPack {
    packName.set("model_pack")
    dynamicDelivery { deliveryType.set("install-time") }
}
