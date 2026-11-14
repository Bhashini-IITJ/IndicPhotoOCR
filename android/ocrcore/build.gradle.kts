// Platform-independent OCR pipeline (port of mobile_app/tools/reference_pipeline.py).
// ONNX Runtime and org.json are provided by the consumer: onnxruntime-android + Android's org.json
// in the app, onnxruntime (desktop) + org.json in :verify.
plugins { id("org.jetbrains.kotlin.jvm") }
kotlin { jvmToolchain(17) }
dependencies {
    compileOnly("com.microsoft.onnxruntime:onnxruntime:1.22.0")
    compileOnly("org.json:json:20240303")
}
