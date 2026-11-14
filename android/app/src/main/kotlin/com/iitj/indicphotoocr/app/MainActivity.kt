package com.iitj.indicphotoocr.app

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.core.view.GravityCompat
import androidx.exifinterface.media.ExifInterface
import com.iitj.indicphotoocr.app.databinding.ActivityMainBinding
import com.iitj.indicphotoocr.core.OcrCancelled
import com.iitj.indicphotoocr.core.OcrEngine
import com.iitj.indicphotoocr.core.OcrResult
import com.iitj.indicphotoocr.core.OcrStages
import com.iitj.indicphotoocr.core.RgbImage
import java.io.File
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {
    private lateinit var ui: ActivityMainBinding
    private val worker = Executors.newSingleThreadExecutor()   // models: load, then one photo at a time
    private val decoder = Executors.newSingleThreadExecutor()  // opens the photo at once, even while models load
    @Volatile private var engine: OcrEngine? = null
    @Volatile private var loadFailed: String? = null
    private var captureUri: Uri? = null
    private var lastText = ""
    @Volatile private var cancelRequested = false

    private val takePicture = registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        if (ok) captureUri?.let { process(it) }
    }
    private val pickImage = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let { process(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ui = ActivityMainBinding.inflate(layoutInflater)
        setContentView(ui.root)
        ui.menu.setOnClickListener { ui.drawer.openDrawer(GravityCompat.START) }
        ui.nav.setNavigationItemSelectedListener { item ->
            ui.drawer.closeDrawer(GravityCompat.START)
            when (item.itemId) {
                R.id.nav_about -> InfoActivity.open(this, InfoActivity.ABOUT)
                R.id.nav_terms -> InfoActivity.open(this, InfoActivity.TERMS)
                R.id.nav_privacy -> InfoActivity.open(this, InfoActivity.PRIVACY)
                R.id.nav_licences -> InfoActivity.open(this, InfoActivity.LICENCES)
            }
            true
        }
        ui.camera.setOnClickListener {
            val f = File(cacheDir, "captures").apply { mkdirs() }.let { File(it, "photo.jpg") }
            captureUri = FileProvider.getUriForFile(this, "$packageName.fileprovider", f)
            takePicture.launch(captureUri!!)
        }
        ui.gallery.setOnClickListener {
            pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        ui.cancel.setOnClickListener { cancelRequested = true; ui.progressLabel.text = "Cancelling…" }
        ui.copy.setOnClickListener {
            (getSystemService(CLIPBOARD_SERVICE) as ClipboardManager).setPrimaryClip(ClipData.newPlainText("text", lastText))
            Toast.makeText(this, "Copied", Toast.LENGTH_SHORT).show()
        }
        ui.share.setOnClickListener {
            val send = Intent(Intent.ACTION_SEND).setType("text/plain").putExtra(Intent.EXTRA_TEXT, lastText)
            startActivity(Intent.createChooser(send, "Share text"))
        }
        load()
        intent?.takeIf { it.action == Intent.ACTION_SEND }?.let { sharedUri(it) }?.let { process(it) }
    }

    /**
     * Loads and warms up the models on the worker thread as soon as the app opens, while the user is still taking
     * or choosing a photo; a photo picked meanwhile waits on the same worker. Then, on a low-priority thread, the
     * other languages' readers are loaded and, the first time on this phone, the GPU/NPU is tried for detection
     * and script ID (kept only if it is faster and gives the same result as the CPU).
     */
    private fun load() {
        worker.execute {
            try {
                engine?.close(); engine = null
                ModelStore.deleteOldCopies(this)
                val saved = ModelStore.savedAccelerators(this)
                val e = ModelStore.createEngine(this, saved.orEmpty())
                e.warmUp()
                engine = e
                Thread({
                    android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND)
                    if (ModelStore.canPreloadAll(this)) runCatching { e.preloadReaders() }
                    if (saved == null) runCatching { ModelStore.saveAccelerators(this, e.tuneAccelerators(sampleImage())) }
                }, "background-setup").start()
            } catch (t: Throwable) {
                loadFailed = t.message
                runOnUiThread { showError("Could not load the models: ${t.message}") }
            }
        }
    }

    /** A synthetic photo with printed words, used only to compare accelerators with the CPU. */
    private fun sampleImage(): RgbImage {
        val bmp = Bitmap.createBitmap(960, 1280, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmp); c.drawColor(Color.rgb(236, 232, 222))
        val p = Paint().apply { isAntiAlias = true; color = Color.rgb(30, 30, 40); isFakeBoldText = true }
        val lines = listOf("WATER PURIFIER" to 72f, "नमस्ते भारत" to 80f, "Indian Institute" to 64f, "ঢাকা কলকাতা" to 70f, "SERVICE 2026" to 76f)
        for ((i, l) in lines.withIndex()) { p.textSize = l.second; c.drawText(l.first, 70f, 220f + i * 210f, p) }
        val px = IntArray(bmp.width * bmp.height); bmp.getPixels(px, 0, bmp.width, 0, 0, bmp.width, bmp.height)
        return RgbImage.fromArgb(bmp.width, bmp.height, px)
    }

    @Suppress("DEPRECATION")
    private fun sharedUri(i: Intent): Uri? =
        if (Build.VERSION.SDK_INT >= 33) i.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java) else i.getParcelableExtra(Intent.EXTRA_STREAM)

    private fun showError(msg: String) { ui.status.text = msg; ui.status.visibility = View.VISIBLE }

    /** Overall progress of one photo: detection up to 30%, script ID up to 45%, reading up to 98%, done 100%. */
    private fun setProgress(label: String, percent: Int) {
        ui.progressLabel.text = label
        ui.progress.setProgressCompat(percent.coerceIn(0, 100), true)
        ui.progressPercent.text = "${percent.coerceIn(0, 100)}%"
    }

    /**
     * The photo is shown as soon as it is opened, then redrawn with boxes after detection, and once more at the end
     * without the overlapping boxes that were removed. Boxes and text are not coloured by language (supervisor, 7.3).
     */
    private fun process(uri: Uri) {
        cancelRequested = false
        ui.status.visibility = View.GONE
        ui.camera.visibility = View.GONE; ui.gallery.visibility = View.GONE; ui.cancel.visibility = View.VISIBLE
        ui.progressCard.visibility = View.VISIBLE
        ui.progress.setProgressCompat(0, false)
        setProgress("Opening photo", 0)
        decoder.execute {
            val bmp = try { decodeOriented(uri, MAX_SIDE) } catch (t: Throwable) {
                runOnUiThread { endRun(); showError("Could not open the photo: ${t.message}") }; return@execute
            }
            runOnUiThread {
                ui.image.setImageBitmap(bmp)
                ui.emptyCard.visibility = View.GONE; ui.imageCard.visibility = View.VISIBLE; ui.resultCard.visibility = View.GONE
                setProgress(if (engine == null) "Getting ready" else "Finding text", 3)
            }
            worker.execute { read(bmp) }
        }
    }

    private fun read(bmp: Bitmap) {
        try {
            val e = engine ?: throw IllegalStateException(loadFailed ?: "models not loaded")
            if (cancelRequested) throw OcrCancelled()
            runOnUiThread { setProgress("Finding text", 5) }
            val px = IntArray(bmp.width * bmp.height)
            bmp.getPixels(px, 0, bmp.width, 0, 0, bmp.width, bmp.height)
            val r = e.run(RgbImage.fromArgb(bmp.width, bmp.height, px), { stage, done, total ->
                val (label, pct) = when (stage) {
                    "detect" -> "Finding text" to 5
                    "script" -> "Analysing text" to 30 + 15 * done / max(1, total)
                    else -> "Reading text" to 45 + (53.0 * done / max(1, total)).roundToInt()
                }
                runOnUiThread { if (!cancelRequested) setProgress(label, pct) }
            }, { cancelRequested }, object : OcrStages {
                override fun detected(boxes: List<FloatArray>) {
                    val shown = drawPlainBoxes(bmp, boxes)
                    runOnUiThread { ui.image.setImageBitmap(shown); if (!cancelRequested) setProgress("Analysing text", 30) }
                }
                override fun identified(boxes: List<FloatArray>, scripts: List<String>) {
                    runOnUiThread { if (!cancelRequested) setProgress("Reading text", 45) }
                }
            })
            val shown = drawPlainBoxes(bmp, r.words.map { it.box })
            runOnUiThread { show(shown, r) }
        } catch (c: OcrCancelled) {
            runOnUiThread { ui.image.setImageBitmap(bmp) }
        } catch (t: Throwable) {
            runOnUiThread { showError("Could not read this photo: ${t.message}") }
        } finally {
            runOnUiThread { endRun() }
        }
    }

    private fun endRun() {
        ui.progressCard.visibility = View.GONE
        ui.cancel.visibility = View.GONE; ui.camera.visibility = View.VISIBLE; ui.gallery.visibility = View.VISIBLE
    }

    private fun show(bmp: Bitmap, r: OcrResult) {
        setProgress("Done", 100)
        ui.image.setImageBitmap(bmp)
        ui.resultCard.visibility = View.VISIBLE
        val lines = ReadingOrder.lines(r.words.filter { !it.text.isNullOrBlank() })
        lastText = lines.joinToString("\n") { l -> l.joinToString(" ") { it.text!! } }
        ui.result.text = if (lastText.isEmpty()) "No text found." else lastText
        ui.copy.isEnabled = lastText.isNotEmpty(); ui.share.isEnabled = lastText.isNotEmpty()
    }

    private fun strokeFor(b: Bitmap) = max(3f, max(b.width, b.height) / 400f)

    /** All boxes, one style: white with a dark edge, visible on any photo. */
    private fun drawPlainBoxes(src: Bitmap, boxes: List<FloatArray>): Bitmap {
        val out = src.copy(Bitmap.Config.ARGB_8888, true); val c = Canvas(out); val w = strokeFor(out)
        val edge = Paint().apply { style = Paint.Style.STROKE; strokeWidth = w * 1.8f; color = Color.argb(150, 20, 20, 30); isAntiAlias = true }
        val line = Paint().apply { style = Paint.Style.STROKE; strokeWidth = w; color = Color.WHITE; isAntiAlias = true }
        for (b in boxes) { c.drawRect(b[0], b[1], b[2], b[3], edge); c.drawRect(b[0], b[1], b[2], b[3], line) }
        return out
    }

    /** Decode with EXIF orientation applied (like cv2.imread), long side limited to [maxSide]. */
    private fun decodeOriented(uri: Uri, maxSide: Int): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it, null, bounds) }
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxSide) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample; inPreferredConfig = Bitmap.Config.ARGB_8888 }
        var bmp = contentResolver.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it, null, opts) }!!
        val orientation = contentResolver.openInputStream(uri)!!.use { ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL) }
        val m = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> { m.postRotate(90f); m.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_TRANSVERSE -> { m.postRotate(270f); m.postScale(-1f, 1f) }
        }
        if (!m.isIdentity) bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
        return bmp
    }

    override fun onDestroy() {
        super.onDestroy(); decoder.shutdown(); worker.execute { engine?.close() }; worker.shutdown()
    }

    companion object {
        const val MAX_SIDE = 4096
    }
}
