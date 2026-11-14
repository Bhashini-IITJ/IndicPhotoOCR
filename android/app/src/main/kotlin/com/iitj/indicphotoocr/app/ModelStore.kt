package com.iitj.indicphotoocr.app

import android.app.ActivityManager
import android.content.Context
import com.iitj.indicphotoocr.core.ModelSource
import com.iitj.indicphotoocr.core.OcrEngine
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

/**
 * Builds the engine on the models shipped with the app (install-time asset pack :model_pack, read through the
 * normal AssetManager). The .onnx files are stored uncompressed (noCompress), so each is memory-mapped straight
 * from the installed APK: nothing is copied to app storage and the models take no extra space on the phone.
 */
object ModelStore {
    private const val ASSET_DIR = "models"

    private class AssetModelSource(private val ctx: Context) : ModelSource {
        override fun manifest() = ctx.assets.open("$ASSET_DIR/manifest.json").bufferedReader().use { it.readText() }
        override fun model(file: String): ByteBuffer = try {
            ctx.assets.openFd("$ASSET_DIR/$file").use { fd ->
                FileInputStream(fd.fileDescriptor).channel.use { it.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength) }
            }
        } catch (e: java.io.IOException) {
            // openFd works only on files stored uncompressed (the normal case, also in the install-time asset pack).
            // If a model ever arrives compressed, read it into memory instead of failing.
            val bytes = ctx.assets.open("$ASSET_DIR/$file").use { it.readBytes() }
            ByteBuffer.allocateDirect(bytes.size).put(bytes).apply { flip() }
        }
    }

    /** Removes the model copies that test builds up to 2.0.0-alpha3 wrote to app storage. */
    fun deleteOldCopies(ctx: Context) {
        File(ctx.filesDir, ASSET_DIR).takeIf { it.exists() }?.deleteRecursively()
    }

    fun createEngine(ctx: Context, providers: Map<String, String> = emptyMap()): OcrEngine {
        val app = JSONObject(AssetModelSource(ctx).manifest()).optJSONObject("app") ?: JSONObject()
        val threads = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
        return OcrEngine(AssetModelSource(ctx), threads = threads, siMode = app.optString("si_mode", "model"), providers = providers)
            .apply { pruneMode = BuildConfig.PRUNE_MODE }
    }

    // Accelerator choice (see OcrEngine.tuneAccelerators), measured once per app version and phone.
    private fun accelKey() = "accel_${BuildConfig.VERSION_CODE}_${android.os.Build.FINGERPRINT}"
    fun savedAccelerators(ctx: Context): Map<String, String>? =
        ctx.getSharedPreferences("engine", Context.MODE_PRIVATE).getString(accelKey(), null)
            ?.let { JSONObject(it) }?.let { o -> o.keys().asSequence().associateWith { o.getString(it) } }
    fun saveAccelerators(ctx: Context, choice: Map<String, String>) =
        ctx.getSharedPreferences("engine", Context.MODE_PRIVATE).edit().putString(accelKey(), JSONObject(choice).toString()).apply()

    /** Whether to load all readers in the background after start (about 80 MB of memory each): phones with 6 GB+. */
    fun canPreloadAll(ctx: Context): Boolean {
        val mi = ActivityManager.MemoryInfo().also { (ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(it) }
        return !mi.lowMemory && mi.totalMem >= 5_500L * 1024 * 1024
    }
}
