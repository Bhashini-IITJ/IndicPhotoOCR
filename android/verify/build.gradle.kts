// Runs the exact Kotlin pipeline on the server for comparison with the Python reference.
plugins { id("org.jetbrains.kotlin.jvm"); application }
kotlin { jvmToolchain(17) }
dependencies {
    implementation(project(":ocrcore"))
    implementation("com.microsoft.onnxruntime:onnxruntime:1.22.0")
    implementation("org.json:json:20240303")
}
application { mainClass.set("com.iitj.indicphotoocr.verify.VerifyKt") }
