package com.iitj.indicphotoocr.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.util.EnumSet
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/** One recognised word: box in original-image pixels (x1, y1, x2, y2). */
data class Word(val box: FloatArray, val score: Float, val script: String, val text: String?, val confidence: Float?,
                /** lowest per-character probability of the reading (used by the D3 overlap rule) */
                val minProb: Float? = null)

/** One reading of a crop: text, mean max-probability up to EOS, lowest max-probability up to EOS. */
data class Read(val text: String, val conf: Float, val minp: Float)

data class OcrResult(val words: List<Word>, val millis: Map<String, Long>)

/** Thrown when [OcrEngine.run]'s cancel check returns true. */
class OcrCancelled : RuntimeException("cancelled")

/** Progress report: stage ("detect", "script", "read"), items done, items total. */
fun interface OcrProgress { fun update(stage: String, done: Int, total: Int) }

/** Intermediate results for live display: the boxes after detection, then their scripts after script ID. */
interface OcrStages {
    fun detected(boxes: List<FloatArray>) {}
    fun identified(boxes: List<FloatArray>, scripts: List<String>) {}
}

/** Where the engine reads manifest.json and the .onnx files from. */
interface ModelSource {
    fun manifest(): String
    /** The bytes of [file]; a direct (e.g. memory-mapped) buffer, so nothing is copied into the Java heap. */
    fun model(file: String): ByteBuffer
}

/** Models in a plain directory, memory-mapped. */
class DirModelSource(private val dir: File) : ModelSource {
    override fun manifest() = File(dir, "manifest.json").readText()
    override fun model(file: String): ByteBuffer = java.io.RandomAccessFile(File(dir, file), "r").use { f ->
        f.channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, 0, f.length())
    }
}

/**
 * IndicPhotoOCR 2.0 offline pipeline: YOLO11n -> script ID -> PARSeq.
 * A line-by-line port of mobile_app/tools/reference_pipeline.py (reference_pipeline_all.py for
 * bundles with all readers); keep them in sync.
 *
 * Runs on ONNX Runtime's CPU execution provider by default so it works on any device. Each script
 * is read by its own PARSeq if the bundle has one (Marathi falls back to the Hindi reader); words in
 * a script without a reader are returned with text = null. Readers are loaded on first use, or all
 * at once with [preloadReaders].
 *
 * @param source   manifest.json and the .onnx files
 * @param suffix   model variant suffix ("", "_fp16", "_int8")
 * @param configure optional hook to add an accelerator to the session options; any failure there
 *                  falls back to the plain CPU session.
 */
class OcrEngine(
    private val source: ModelSource,
    private val suffix: String = "",
    private val threads: Int = 4,
    private val detectSize: Int? = null,
    private val configure: ((OrtSession.SessionOptions) -> Unit)? = null,
    /** "model": EfficientNetV2-L picks the recogniser; "dual": no script-ID model, read with both, keep the more confident. */
    val siMode: String = "model",
    /** Execution provider per model ("detector", "script_id" -> "cpu" | "xnnpack" | "nnapi" | "nnapi_fp16"), e.g. a
     *  choice saved from an earlier [tuneAccelerators]; any provider that fails to load falls back to the CPU. */
    providers: Map<String, String> = emptyMap(),
) : AutoCloseable {
    constructor(modelDir: File, suffix: String = "", threads: Int = 4, detectSize: Int? = null,
                configure: ((OrtSession.SessionOptions) -> Unit)? = null, siMode: String = "model") :
        this(DirModelSource(modelDir), suffix, threads, detectSize, configure, siMode)

    private val env = OrtEnvironment.getEnvironment()
    private val manifest = JSONObject(source.manifest())
    private val y = manifest.getJSONObject("yolo")
    private val sidCfg = manifest.getJSONObject("script_id")
    /** Script-ID labels. */
    val classes = sidCfg.getJSONArray("classes").let { a -> List(a.length()) { a.getString(it) } }
    private val recCfg = manifest.getJSONObject("parseq")

    /** [lean]: no weight pre-packing and no memory arena, about 77 instead of 244 MB per PARSeq reader, ~35% slower. */
    private fun open(name: String, lean: Boolean = false): OrtSession {
        val bytes = source.model(name.replace(".onnx", "$suffix.onnx"))
        fun options() = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            if (lean) { setCPUArenaAllocator(false); addConfigEntry("session.disable_prepacking", "1") }
        }
        if (configure != null) {
            try {
                return env.createSession(bytes.duplicate(), options().also { configure(it) })
            } catch (_: Throwable) { /* accelerator unavailable or unsupported: fall back to CPU */ }
        }
        return env.createSession(bytes.duplicate(), options())
    }

    /**
     * Opens [name] on execution provider [ep]. NNAPI (the phone's GPU/NPU through Android's neural-network API) only
     * accelerates fixed shapes, so [fixed] pins the model's free dimensions for it. Throws if [ep] is unavailable.
     */
    private fun openOn(name: String, ep: String, fixed: Map<String, Long>): OrtSession {
        val o = OrtSession.SessionOptions()
        when (ep) {
            "cpu" -> o.setIntraOpNumThreads(threads)
            "xnnpack" -> {
                o.setIntraOpNumThreads(1); o.addConfigEntry("session.intra_op.allow_spinning", "0")
                o.addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
            }
            "nnapi", "nnapi_fp16" -> {
                o.setIntraOpNumThreads(threads)
                for ((k, v) in fixed) o.setSymbolicDimensionValue(k, v)
                o.addNnapi(if (ep == "nnapi_fp16") EnumSet.of(NNAPIFlags.USE_FP16) else EnumSet.noneOf(NNAPIFlags::class.java))
            }
            else -> throw IllegalArgumentException(ep)
        }
        return env.createSession(source.model(name.replace(".onnx", "$suffix.onnx")).duplicate(), o)
    }

    private val detSize get() = detectSize ?: y.getInt("imgsz")
    private val sidFixed = mapOf("batch" to SID_BATCH.toLong())
    private fun shapeDims(hw: Pair<Int, Int>) = mapOf("batch" to 1L, "height" to hw.first.toLong(), "width" to hw.second.toLong())

    private fun openPreferred(name: String, ep: String?, fixed: Map<String, Long>): Pair<OrtSession, String> {
        if (ep != null && ep != "cpu") runCatching { return openOn(name, ep, fixed) to ep }
        return (if (configure != null) open(name) else openOn(name, "cpu", fixed)) to "cpu"
    }

    /** Detector with dynamic input size, on the CPU (or XNNPACK); also the fallback when NNAPI fails. */
    @Volatile private var yolo: OrtSession
    @Volatile private var yoloProvider = "cpu"
    /**
     * Detector on NNAPI, if chosen. NNAPI needs a fixed input size, and padding every photo to one size changes the
     * detections (94% of boxes matched on test images), so a session is built for each letterboxed photo size
     * actually seen (photos from one camera share a few sizes; the last [NNAPI_SHAPES] are kept).
     */
    @Volatile private var detNnapi: String? = null
    private val nnapiDet = LinkedHashMap<Pair<Int, Int>, OrtSession>()
    @Volatile private var sid: OrtSession?
    /** Execution provider in use for the detector / the script identifier. */
    val detectorProvider get() = detNnapi ?: yoloProvider
    @Volatile var scriptProvider = "cpu"; private set
    init {
        val want = providers["detector"]
        if (want != null && want.startsWith("nnapi")) detNnapi = want
        openPreferred(y.getString("file"), want.takeIf { it == "xnnpack" }, emptyMap()).let { (s, ep) -> yolo = s; yoloProvider = ep }
        sid = if (siMode == "model") openPreferred(sidCfg.getString("file"), providers["script_id"], sidFixed)
            .let { (s, ep) -> scriptProvider = ep; s } else null
    }

    private fun nnapiDetector(ep: String, hw: Pair<Int, Int>): OrtSession = synchronized(nnapiDet) {
        nnapiDet[hw] ?: openOn(y.getString("file"), ep, shapeDims(hw)).also {
            nnapiDet[hw] = it
            while (nnapiDet.size > NNAPI_SHAPES) nnapiDet.remove(nnapiDet.keys.first())?.close()
        }
    }
    /** Held for a whole [run]; [tuneAccelerators] swaps sessions only under it. */
    private val runLock = Any()
    /** Script-ID label -> reader name, for the scripts this bundle can read. */
    val readerFor: Map<String, String> = classes.mapNotNull { c ->
        when {
            recCfg.has(c) -> c to c
            c == "marathi" && recCfg.has("hindi") -> c to "hindi"  // both Devanagari
            else -> null
        }
    }.toMap()
    // Hindi and English (most photos) are loaded up front with the fast settings and always kept.
    // Other readers use the lean settings and are loaded on first use or by [preloadReaders]; without
    // preloading, at most [maxExtraReaders] of them stay loaded (least recently used is closed).
    private val mainReaders = setOf("hindi", "english")
    private val rec = HashMap<String, OrtSession>()
    private val loadLocks = recCfg.keys().asSequence().associateWith { Any() }
    private val lastUsed = LinkedHashMap<String, Unit>(16, 0.75f, true)
    @Volatile private var closed = false
    var maxExtraReaders = 3

    private fun reader(lang: String): OrtSession {
        synchronized(rec) { rec[lang]?.let { lastUsed[lang] = Unit; return it } }
        synchronized(loadLocks.getValue(lang)) {
            synchronized(rec) { rec[lang]?.let { return it } }
            val s = open(recCfg.getJSONObject(lang).getString("file"), lean = lang !in mainReaders)
            synchronized(rec) {
                if (closed) { s.close(); throw OcrCancelled() }
                rec[lang] = s; lastUsed[lang] = Unit
                if (!preloaded) {
                    val extra = lastUsed.keys.filter { it !in mainReaders }
                    for (old in extra.dropLast(maxExtraReaders)) { rec.remove(old)?.close(); lastUsed.remove(old) }
                }
            }
            return s
        }
    }
    @Volatile private var preloaded = false

    init { for (l in mainReaders) if (recCfg.has(l)) reader(l) }

    /**
     * Tries the phone's GPU/NPU (NNAPI, fp32 and fp16) and XNNPACK for the detector and the script identifier on
     * [sample] (an image with text). A provider is kept only if it gives the same boxes (IoU >= 0.85 each) or the
     * same scripts as the CPU and is at least 10% faster (for NNAPI the detector is timed at the sample's own input size). Runs while the engine is idle or between photos; returns
     * the providers in use. The readers stay on the CPU (their int8 dynamic-quantised layers have no NNAPI kernels).
     */
    fun tuneAccelerators(sample: RgbImage, candidates: List<String> = listOf("nnapi", "nnapi_fp16", "xnnpack")): Map<String, String> {
        fun timeMs(f: () -> Unit): Double { f(); val t = System.nanoTime(); f(); f(); return (System.nanoTime() - t) / 2e6 }
        val cpuYolo = yolo  // tuning starts from the CPU detector (no saved choice)
        val ref = detectWith(sample) { cpuYolo }.first
        val hw = inputShape(sample)
        var best: Triple<OrtSession, String, Double>? = Triple(cpuYolo, "cpu", timeMs { detectWith(sample) { cpuYolo } })
        for (ep in candidates) {
            if (closed) return mapOf("detector" to detectorProvider, "script_id" to scriptProvider)
            val s = runCatching { openOn(y.getString("file"), ep, if (ep.startsWith("nnapi")) shapeDims(hw) else emptyMap()) }.getOrNull() ?: continue
            val ok = runCatching { sameBoxes(ref, detectWith(sample) { s }.first) }.getOrDefault(false)
            val t = if (ok) runCatching { timeMs { detectWith(sample) { s } } }.getOrNull() else null
            if (t != null && t < 0.9 * best!!.third) { if (best.second != "cpu") best.first.close(); best = Triple(s, ep, t) } else s.close()
        }
        val crops = ref.mapNotNull { crop(sample, it) }.ifEmpty { listOf(dummyCrop) }.let { c -> List(SID_BATCH) { c[it % c.size] } }
        var bestSid: Triple<OrtSession, String, Double>? = null
        if (sid != null) {
            val cpuSid = synchronized(runLock) { sid.takeIf { scriptProvider == "cpu" } } ?: openOn(sidCfg.getString("file"), "cpu", sidFixed)
            val refS = scriptIdWith(crops, cpuSid, false, report = false)
            bestSid = Triple(cpuSid, "cpu", timeMs { scriptIdWith(crops, cpuSid, false, report = false) })
            for (ep in candidates) {
                if (closed) break
                val s = runCatching { openOn(sidCfg.getString("file"), ep, sidFixed) }.getOrNull() ?: continue
                val ok = runCatching { scriptIdWith(crops, s, ep.startsWith("nnapi"), report = false) == refS }.getOrDefault(false)
                val t = if (ok) runCatching { timeMs { scriptIdWith(crops, s, ep.startsWith("nnapi"), report = false) } }.getOrNull() else null
                if (t != null && t < 0.9 * bestSid!!.third) { if (bestSid.second != "cpu") bestSid.first.close(); bestSid = Triple(s, ep, t) } else s.close()
            }
        }
        synchronized(runLock) {
            if (closed) return mapOf("detector" to detectorProvider, "script_id" to scriptProvider)
            best!!.let { (s, ep, _) ->
                when {
                    s === yolo -> {}
                    ep.startsWith("nnapi") -> { synchronized(nnapiDet) { nnapiDet[hw] = s }; detNnapi = ep }
                    else -> { val old = yolo; yolo = s; yoloProvider = ep; old.close() }
                }
            }
            bestSid?.let { (s, ep, _) -> if (s !== sid) { val old = sid; sid = s; scriptProvider = ep; old?.close() } }
        }
        return mapOf("detector" to detectorProvider, "script_id" to scriptProvider)
    }

    private fun sameBoxes(a: List<FloatArray>, b: List<FloatArray>): Boolean {
        if (a.size != b.size) return false
        val used = BooleanArray(b.size)
        return a.all { p ->
            val j = b.indices.filter { !used[it] }.maxByOrNull { q -> iou(p, b[q]) } ?: return@all false
            if (iou(p, b[j]) < 0.85f) false else { used[j] = true; true }
        }
    }
    private fun iou(a: FloatArray, b: FloatArray): Float {
        val i = inter(a, b); val u = area(a) + area(b) - i; return if (u > 0) i / u else 0f
    }

    /** Loads every reader and keeps them all (call from a background thread, on phones with enough memory). */
    fun preloadReaders() {
        preloaded = true
        for (l in recCfg.keys()) { if (closed) return; reader(l); warmReader(l) }
    }

    private val dummyCrop = RgbImage(100, 32)
    private fun warmReader(lang: String) { recogniseWith(reader(lang), lang, listOf(dummyCrop)) }

    /**
     * Runs a small blank image through the detector, the script ID and the Hindi/English readers once,
     * so ONNX Runtime's one-time setup (memory planning, kernel selection) happens before the first photo.
     */
    fun warmUp() {
        detect(RgbImage(960, 1280))
        if (sid != null) scriptId(listOf(dummyCrop))
        for (l in mainReaders) if (recCfg.has(l)) warmReader(l)
    }

    // ------------------------------------------------------------------ 1. detection
    private fun pyRound(v: Double): Int = Math.rint(v).toInt()  // Python round(): half to even

    fun detect(img: RgbImage): Pair<List<FloatArray>, List<Float>> {
        val ep = detNnapi ?: return detectWith(img) { yolo }
        return try { detectWith(img) { hw -> nnapiDetector(ep, hw) } }
        catch (c: OcrCancelled) { throw c }
        catch (_: Throwable) { detNnapi = null; detectWith(img) { yolo } }  // this phone's NNAPI failed: CPU from now on
    }

    /** Letterboxed input size (height, width) for [img], as [detectWith] builds it. */
    private fun inputShape(img: RgbImage): Pair<Int, Int> {
        val s = detSize; val st = y.getInt("stride")
        val r = min(s.toDouble() / img.height, s.toDouble() / img.width)
        val nw = pyRound(img.width * r); val nh = pyRound(img.height * r)
        val dw = ((s - nw) % st) / 2.0; val dh = ((s - nh) % st) / 2.0
        return (nh + pyRound(dh - 0.1) + pyRound(dh + 0.1)) to (nw + pyRound(dw - 0.1) + pyRound(dw + 0.1))
    }

    /** [sessionFor] gives the detector session for the letterboxed input size (height, width). */
    private fun detectWith(img: RgbImage, sessionFor: (Pair<Int, Int>) -> OrtSession): Pair<List<FloatArray>, List<Float>> {
        val s = detSize; val st = y.getInt("stride")
        val h = img.height; val w = img.width
        val r = min(s.toDouble() / h, s.toDouble() / w)
        val nw = pyRound(w * r); val nh = pyRound(h * r)
        val dw = ((s - nw) % st) / 2.0; val dh = ((s - nh) % st) / 2.0
        val small = if (nw != w || nh != h) img.resize(nw, nh, Interp.LINEAR) else img
        val top = pyRound(dh - 0.1); val bottom = pyRound(dh + 0.1); val left = pyRound(dw - 0.1); val right = pyRound(dw + 0.1)
        val H = nh + top + bottom; val W = nw + left + right
        val yolo = sessionFor(H to W)
        val plane = H * W
        val x = FloatArray(3 * plane) { 114f / 255f }
        for (yy in 0 until nh) for (xx in 0 until nw) {
            val o = (yy + top) * W + (xx + left)
            x[o] = small[xx, yy, 0] / 255f; x[plane + o] = small[xx, yy, 1] / 255f; x[2 * plane + o] = small[xx, yy, 2] / 255f
        }
        val out = OnnxTensor.createTensor(env, FloatBuffer.wrap(x), longArrayOf(1, 3, H.toLong(), W.toLong())).use { t ->
            yolo.run(mapOf(yolo.inputNames.first() to t)).use { res ->
                val o = res.get(0) as OnnxTensor
                val n = o.info.shape[2].toInt(); val fb = o.floatBuffer
                Array(5) { c -> FloatArray(n) { fb.get(c * n + it) } }
            }
        }
        val conf = y.getDouble("conf").toFloat()
        val boxes = ArrayList<FloatArray>(); val scores = ArrayList<Float>()
        for (i in out[0].indices) if (out[4][i] >= conf) {
            val cx = out[0][i]; val cy = out[1][i]; val bw = out[2][i]; val bh = out[3][i]
            boxes += floatArrayOf(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2); scores += out[4][i]
        }
        val keep = nms(boxes, scores, y.getDouble("iou").toFloat()).take(y.getInt("max_det"))
        val rr = r.toFloat()
        return keep.map { i ->
            val b = boxes[i]
            floatArrayOf(((b[0] - left) / rr).coerceIn(0f, w.toFloat()), ((b[1] - top) / rr).coerceIn(0f, h.toFloat()),
                         ((b[2] - left) / rr).coerceIn(0f, w.toFloat()), ((b[3] - top) / rr).coerceIn(0f, h.toFloat()))
        } to keep.map { scores[it] }
    }

    private fun nms(b: List<FloatArray>, s: List<Float>, thr: Float): List<Int> {
        val order = s.indices.sortedWith(compareByDescending<Int> { s[it] })  // stable, like np.argsort(-s, kind="stable")
        val area = FloatArray(b.size) { (b[it][2] - b[it][0]) * (b[it][3] - b[it][1]) }
        val removed = BooleanArray(b.size); val keep = ArrayList<Int>()
        for ((pos, i) in order.withIndex()) {
            if (removed[i]) continue
            keep += i
            for (q in pos + 1 until order.size) {
                val j = order[q]; if (removed[j]) continue
                val iw = max(0f, min(b[i][2], b[j][2]) - max(b[i][0], b[j][0]))
                val ih = max(0f, min(b[i][3], b[j][3]) - max(b[i][1], b[j][1]))
                val inter = iw * ih
                if (inter / (area[i] + area[j] - inter + 1e-9f) > thr) removed[j] = true
            }
        }
        return keep
    }

    // ------------------------------------------------------------------ 2. crop (as OCR.crop_bbox)
    private fun crop(img: RgbImage, b: FloatArray): RgbImage? {
        val x1 = b[0].toInt().coerceIn(0, img.width - 1); val x2 = b[2].toInt().coerceIn(0, img.width - 1)
        val y1 = b[1].toInt().coerceIn(0, img.height - 1); val y2 = b[3].toInt().coerceIn(0, img.height - 1)
        if (x2 < x1 || y2 < y1) return null
        return img.crop(x1, y1, x2, y2)
    }

    // ------------------------------------------------------------------ 3. script identification
    private var progress: OcrProgress? = null
    private var cancelled: () -> Boolean = { false }
    private fun check() { if (cancelled()) throw OcrCancelled() }

    private fun scriptId(crops: List<RgbImage>): List<String> {
        if (!scriptProvider.startsWith("nnapi")) return scriptIdWith(crops, sid!!, false)
        return try { scriptIdWith(crops, sid!!, true) }
        catch (c: OcrCancelled) { throw c }
        catch (_: Throwable) {  // this phone's NNAPI failed: CPU from now on
            val old = sid; sid = openOn(sidCfg.getString("file"), "cpu", emptyMap()); scriptProvider = "cpu"; old?.close()
            scriptIdWith(crops, sid!!, false)
        }
    }

    /** [padBatch]: always feed [SID_BATCH] crops (the fixed shape NNAPI sessions are built for), padding with zeros. */
    private fun scriptIdWith(crops: List<RgbImage>, session: OrtSession, padBatch: Boolean, report: Boolean = true): List<String> {
        val size = sidCfg.getInt("size")
        val mean = sidCfg.getJSONArray("mean").let { a -> FloatArray(3) { a.getDouble(it).toFloat() } }
        val std = sidCfg.getJSONArray("std").let { a -> FloatArray(3) { a.getDouble(it).toFloat() } }
        val out = ArrayList<String>()
        for (start in crops.indices step 16) {
            if (report) { check(); progress?.update("script", start, crops.size) }
            val batch = crops.subList(start, min(crops.size, start + 16))
            var x = toTensorInput(batch.map { it.resizeForModel(size, size, Interp.LINEAR) }, size, size) { v, c -> (v / 255f - mean[c]) / std[c] }
            val n = if (padBatch) SID_BATCH else batch.size
            if (padBatch) x = x.copyOf(SID_BATCH * 3 * size * size)
            OnnxTensor.createTensor(env, FloatBuffer.wrap(x), longArrayOf(n.toLong(), 3, size.toLong(), size.toLong())).use { t ->
                val sess = session
                sess.run(mapOf(sess.inputNames.first() to t)).use { res ->
                    val fb = (res.get(0) as OnnxTensor).floatBuffer; val k = classes.size
                    for (bi in batch.indices) {
                        var best = 0
                        for (c in 1 until k) if (fb.get(bi * k + c) > fb.get(bi * k + best)) best = c
                        out += classes[best]
                    }
                }
            }
        }
        return out
    }

    private inline fun toTensorInput(imgs: List<RgbImage>, w: Int, h: Int, norm: (Float, Int) -> Float): FloatArray {
        val plane = w * h; val x = FloatArray(imgs.size * 3 * plane)
        for ((n, im) in imgs.withIndex()) {
            val base = n * 3 * plane
            for (p in 0 until plane) for (c in 0 until 3) x[base + c * plane + p] = norm((im.data[p * 3 + c].toInt() and 0xFF).toFloat(), c)
        }
        return x
    }

    // ------------------------------------------------------------------ 4. recognition
    private fun recognise(lang: String, crops: List<RgbImage>): List<Read> =
        recogniseWith(reader(lang), lang, crops) { n -> check(); readDone += n; progress?.update("read", readDone, readTotal) }

    private fun recogniseWith(sess: OrtSession, lang: String, crops: List<RgbImage>, onBatch: (Int) -> Unit = {}): List<Read> {
        val cfg = recCfg.getJSONObject(lang)
        val hw = cfg.getJSONArray("img_size"); val H = hw.getInt(0); val W = hw.getInt(1)
        val itos = cfg.getJSONArray("itos").let { a -> List(a.length()) { a.getString(it) } }
        val eos = cfg.getInt("eos_id"); val charset = cfg.getString("charset_test")
        val allowed = charset.toSet(); val lower = charset == charset.lowercase()
        val res = ArrayList<Read>()
        for (start in crops.indices step 32) {
            onBatch(min(32, crops.size - start))
            val batch = crops.subList(start, min(crops.size, start + 32))
            val x = toTensorInput(batch.map { it.resizeForModel(W, H, Interp.CUBIC) }, W, H) { v, _ -> (v / 255f - 0.5f) / 0.5f }
            OnnxTensor.createTensor(env, FloatBuffer.wrap(x), longArrayOf(batch.size.toLong(), 3, H.toLong(), W.toLong())).use { t ->
                sess.run(mapOf(sess.inputNames.first() to t)).use { r ->
                    val o = r.get(0) as OnnxTensor; val shape = o.info.shape
                    val T = shape[1].toInt(); val C = shape[2].toInt(); val fb = o.floatBuffer
                    for (bi in batch.indices) {
                        val ids = IntArray(T); val pr = FloatArray(T)
                        for (tt in 0 until T) {
                            val off = (bi * T + tt) * C
                            var best = 0; var mx = fb.get(off)
                            for (c in 1 until C) { val v = fb.get(off + c); if (v > mx) { mx = v; best = c } }
                            var sum = 0.0
                            for (c in 0 until C) sum += exp((fb.get(off + c) - mx).toDouble())
                            ids[tt] = best; pr[tt] = (1.0 / sum).toFloat()  // softmax max-prob
                        }
                        val n = ids.indexOfFirst { it == eos }.let { if (it < 0) T else it }
                        var txt = buildString { for (tt in 0 until n) append(itos[ids[tt]]) }
                        if (lower) txt = txt.lowercase()
                        txt = txt.filter { it in allowed }
                        val m = min(n + 1, T); var cs = 0f; var mp = 1f
                        for (tt in 0 until m) { cs += pr[tt]; mp = min(mp, pr[tt]) }
                        res += Read(txt, cs / m, mp)
                    }
                }
            }
        }
        return res
    }

    // ------------------------------------------------------------------ full pipeline
    private var readDone = 0
    private var readTotal = 0

    /**
     * Full pipeline. [progress] is called between batches; [cancelled] is polled between batches and
     * aborts the run with [OcrCancelled].
     */
    fun run(img: RgbImage, progress: OcrProgress? = null, cancelled: () -> Boolean = { false }, stages: OcrStages? = null): OcrResult =
        synchronized(runLock) {
            this.progress = progress; this.cancelled = cancelled; this.stages = stages
            try {
                if (siMode == "dual") runDual(img) else runModel(img)
            } finally { this.progress = null; this.cancelled = { false }; this.stages = null }
        }
    private var stages: OcrStages? = null

    /** Dual mode (reference_pipeline.run_dual): Hindi and English PARSeq on every word, keep the higher confidence. */
    private fun runDual(img: RgbImage): OcrResult {
        val ms = LinkedHashMap<String, Long>()
        progress?.update("detect", 0, 1)
        var t0 = System.nanoTime()
        val (boxes, scores) = detect(img); ms["detect"] = (System.nanoTime() - t0) / 1_000_000
        check()
        val crops = ArrayList<RgbImage>(); val keep = ArrayList<Int>()
        for ((j, b) in boxes.withIndex()) crop(img, b)?.let { crops += it; keep += j }
        ms["script_id"] = 0
        t0 = System.nanoTime()
        readDone = 0; readTotal = 2 * crops.size
        val hi = if (crops.isEmpty()) emptyList() else recognise("hindi", crops)
        val en = if (crops.isEmpty()) emptyList() else recognise("english", crops)
        ms["recognise"] = (System.nanoTime() - t0) / 1_000_000
        val out = keep.mapIndexed { q, j ->
            if (hi[q].conf >= en[q].conf) Word(boxes[j], scores[j], "hindi", hi[q].text, hi[q].conf, hi[q].minp)
            else Word(boxes[j], scores[j], "english", en[q].text, en[q].conf, en[q].minp)
        }
        return OcrResult(out, ms)
    }

    private fun runModel(img: RgbImage): OcrResult {
        val ms = LinkedHashMap<String, Long>()
        progress?.update("detect", 0, 1)
        var t0 = System.nanoTime()
        val (boxes, scores) = detect(img); ms["detect"] = (System.nanoTime() - t0) / 1_000_000
        check()
        val crops = ArrayList<RgbImage>(); val keep = ArrayList<Int>()
        for ((j, b) in boxes.withIndex()) crop(img, b)?.let { crops += it; keep += j }
        stages?.detected(keep.map { boxes[it] })
        t0 = System.nanoTime(); val langs = if (crops.isEmpty()) emptyList() else scriptId(crops); ms["script_id"] = (System.nanoTime() - t0) / 1_000_000
        stages?.identified(keep.map { boxes[it] }, langs)
        t0 = System.nanoTime()
        val words = HashMap<Int, Read>()
        readDone = 0; readTotal = langs.count { readerFor[it] != null }
        for (lang in readerFor.values.distinct()) {
            val qs = langs.indices.filter { readerFor[langs[it]] == lang }
            if (qs.isNotEmpty()) recognise(lang, qs.map { crops[it] }).forEachIndexed { k, w -> words[qs[k]] = w }
        }
        ms["recognise"] = (System.nanoTime() - t0) / 1_000_000
        val out = keep.mapIndexed { q, j -> Word(boxes[j], scores[j], langs[q], words[q]?.text, words[q]?.conf, words[q]?.minp) }
        if (!pruneOverlaps) return OcrResult(out, ms)
        t0 = System.nanoTime()
        var kept = prune(out)
        if (pruneMode != "test6") kept = pruneD12(kept, confP20)
        if (pruneMode == "d123") kept = pruneD3(img, kept)
        ms["prune"] = (System.nanoTime() - t0) / 1_000_000
        return OcrResult(kept, ms)
    }

    /** Remove overlapping detections of the same text (see [prune], [pruneD12], [pruneD3]). */
    var pruneOverlaps = true
    /** "test6": [prune] only; "d12": + [pruneD12]; "d123": + [pruneD3] (extra re-reads of conflicting boxes). */
    var pruneMode = "d12"
    /** Per-script 20th-percentile reader confidence (manifest prune.conf_p20, frozen from validation). */
    private val confP20: Map<String, Float> = manifest.optJSONObject("prune")?.optJSONObject("conf_p20")
        ?.let { o -> o.keys().asSequence().associateWith { o.getDouble(it).toFloat() } } ?: emptyMap()

    /**
     * D3 (faisal_storage/results/overlap_ship_20261030/ship_rules.py, keep in sync): every box still in an overlap
     * pair is read again from 8 shifted crops (per script one batch of the 4 J4 shifts, then one of the 4 J8 shifts).
     * stab = share of re-reads equal to its reading. In a pair, the less stable box x is dropped if
     * stab(other) - stab(x) >= 0.625 and score(x) * minp(x) < score(other) * minp(other).
     */
    private fun pruneD3(img: RgbImage, words: List<Word>): List<Word> {
        val alive = words.indices.filter { !words[it].text.isNullOrBlank() }
        val conflict = overlapPairs(words, alive).flatMap { listOf(it.first, it.second) }.toSortedSet()
        if (conflict.isEmpty()) return words
        val reads = HashMap<Int, ArrayList<Read?>>()
        val w = img.width.toFloat(); val h = img.height.toFloat()
        for (lang in classes) {
            val cj = conflict.filter { words[it].script == lang }
            val rd = readerFor[lang] ?: continue
            if (cj.isEmpty()) continue
            for (shifts in listOf(J4, J8)) {
                val cr = ArrayList<RgbImage?>(); val own = ArrayList<Int>()
                for (q in cj) {
                    val b = words[q].box; val bh = b[3] - b[1]
                    for (s in shifts) {
                        val jb = floatArrayOf((b[0] + s[0] * bh).coerceIn(0f, w), (b[1] + s[1] * bh).coerceIn(0f, h),
                                              (b[2] + s[2] * bh).coerceIn(0f, w), (b[3] + s[3] * bh).coerceIn(0f, h))
                        cr += crop(img, jb); own += q
                    }
                }
                val good = cr.indices.filter { i -> cr[i]?.let { it.width > 1 && it.height > 1 } == true }
                val got = good.zip(recogniseWith(reader(rd), rd, good.map { cr[it]!! })).toMap()
                for ((i, q) in own.withIndex()) reads.getOrPut(q) { ArrayList() } += got[i]
            }
        }
        fun stab(q: Int): Float? {
            val r = reads[q]?.filterNotNull().orEmpty(); if (r.isEmpty()) return null
            val t0 = norm(words[q].text); return r.count { norm(it.text) == t0 }.toFloat() / r.size
        }
        val drop = BooleanArray(words.size)
        for ((b, s) in overlapPairs(words, alive.filter { it in conflict })) {
            if (drop[b] || drop[s]) continue
            val sb = stab(b) ?: continue; val ss = stab(s) ?: continue
            val g = { q: Int -> words[q].score * (words[q].minProb ?: 0f) }
            if (ss - sb >= D3_GAP - 1e-6f && g(b) < g(s)) drop[b] = true
            else if (sb - ss >= D3_GAP - 1e-6f && g(s) < g(b)) drop[s] = true
        }
        return words.filterIndexed { i, _ -> !drop[i] }
    }

    companion object {
        /** Fixed script-ID batch for NNAPI sessions (the CPU path also uses batches of 16, the last one shorter). */
        private const val SID_BATCH = 16
        /** NNAPI detector sessions kept, one per photo size. */
        private const val NNAPI_SHAPES = 3
        private const val STACK = 1.6f; private const val D3_GAP = 0.625f
        /** D3 crop shifts, in units of the box height: dx1, dy1, dx2, dy2. */
        private val J4 = listOf(floatArrayOf(-.15f, -.1f, .15f, .1f), floatArrayOf(.06f, .04f, -.06f, -.04f),
                                floatArrayOf(-.1f, 0f, .02f, .08f), floatArrayOf(-.02f, -.08f, .1f, 0f))
        private val J8 = listOf(floatArrayOf(-.05f, -.05f, .05f, .05f), floatArrayOf(.08f, 0f, .08f, 0f),
                                floatArrayOf(-.08f, 0f, -.08f, 0f), floatArrayOf(.03f, -.15f, -.03f, .15f))

        /** Overlap pairs (big, small) among [alive] (in order): one box >= 0.7 inside the other, or IoU >= 0.5. */
        private fun overlapPairs(words: List<Word>, alive: List<Int>): List<Pair<Int, Int>> {
            val out = ArrayList<Pair<Int, Int>>()
            for ((x, i) in alive.withIndex()) for (j in alive.subList(x + 1, alive.size)) {
                val a = words[i].box; val b = words[j].box; val I = inter(a, b)
                if (I <= 0) continue
                val m = min(area(a), area(b)); val u = area(a) + area(b) - I
                if ((m > 0 && I / m >= 0.7f) || (u > 0 && I / u >= 0.5f)) out += if (area(a) >= area(b)) i to j else j to i
            }
            return out
        }

        /**
         * D1 + D2 (ship_rules.py, keep in sync), after [prune]: for each overlap pair that is not stacked
         * (h(big) < 1.6 h(small)), lo = the box with lower score * conf^4; drop lo if the other box's confidence
         * is >= its own and (the two scripts differ [D1] or conf(lo) <= [confP20] of its script [D2]).
         */
        fun pruneD12(words: List<Word>, confP20: Map<String, Float>): List<Word> {
            val alive = words.indices.filter { !words[it].text.isNullOrBlank() }
            val cf = { q: Int -> words[q].confidence ?: 0f }
            val q4 = { q: Int -> words[q].score * cf(q).let { it * it * it * it } }
            val drop = BooleanArray(words.size)
            for ((b, s) in overlapPairs(words, alive)) {
                if (drop[b] || drop[s]) continue
                if (words[b].box[3] - words[b].box[1] >= STACK * (words[s].box[3] - words[s].box[1])) continue
                val (lo, hi) = if (q4(b) < q4(s)) b to s else s to b
                if (cf(hi) >= cf(lo) && (words[b].script != words[s].script ||
                        cf(lo) <= (confP20[words[lo].script] ?: Float.NEGATIVE_INFINITY))) drop[lo] = true
            }
            return words.filterIndexed { i, _ -> !drop[i] }
        }

        private const val SAME_OVERLAP = 0.5f; private const val INSIDE = 0.7f
        private const val PART_OVERLAP = 0.3f; private const val COVER = 0.6f; private const val SPAN = 0.75f

        private fun norm(t: String?) = t.orEmpty().lowercase().filterNot { it.isWhitespace() }
        private fun area(b: FloatArray) = max(0f, b[2] - b[0]) * max(0f, b[3] - b[1])
        private fun inter(a: FloatArray, b: FloatArray) =
            max(0f, min(a[2], b[2]) - max(a[0], b[0])) * max(0f, min(a[3], b[3]) - max(a[1], b[1]))

        /**
         * Port of mobile_app/tools/dedup_boxes.py (keep in sync). Uses only the boxes, detector scores and reader confidences:
         * 1. two boxes on the same place (>= 0.5 of the smaller) with the same text: keep the higher score;
         * 2. boxes inside a bigger box B (>= 0.7 of their area), biggest B first; non-overlapping ones are its parts:
         *    a) English B: 2+ parts spanning >= 75% of its width make it a phrase box over separate words; it is
         *       dropped if every part beats it on reader confidence or on detector score;
         *    b) otherwise only parts whose text is part of B's text count: 2+ covering >= 60% of its text -> drop B
         *       if the reader is more confident in every part; else they are fragments, each dropped if the
         *       reader is at least as confident in B.
         * Boxes that do not overlap are never removed.
         */
        fun prune(words: List<Word>): List<Word> {
            val n = words.size; val txt = words.map { norm(it.text) }; val ar = words.map { area(it.box) }
            val cf = words.map { it.confidence ?: 0f }; val sc = words.map { it.score }; val B = words.map { it.box }
            val drop = BooleanArray(n)
            for (i in 0 until n) for (j in i + 1 until n) {
                if (drop[i] || drop[j] || txt[i].isEmpty() || txt[i] != txt[j]) continue
                val m = min(ar[i], ar[j])
                if (m > 0 && inter(B[i], B[j]) / m >= SAME_OVERLAP) drop[if (sc[j] < sc[i]) j else i] = true
            }
            for (b in (0 until n).sortedByDescending { ar[it] }) {
                if (drop[b] || txt[b].isEmpty()) continue
                val inside = (0 until n).filter { i ->
                    i != b && !drop[i] && txt[i].isNotEmpty() && ar[i] > 0 && ar[i] < ar[b] && inter(B[i], B[b]) / ar[i] >= INSIDE
                }
                val inner = inside.filter { txt[it] != txt[b] && txt[b].contains(txt[it]) }
                val phrase = words[b].script == "english"
                val parts = ArrayList<Int>()
                for (i in (if (phrase) inside else inner).sortedByDescending { ar[it] })
                    if (parts.all { p -> inter(B[i], B[p]) / min(ar[i], ar[p]) < PART_OVERLAP }) parts += i
                if (phrase) {
                    val joined = parts.size >= 2 &&
                        (parts.maxOf { B[it][2] } - parts.minOf { B[it][0] }) / max(1e-6f, B[b][2] - B[b][0]) >= SPAN
                    if (joined) {
                        if (parts.minOf { cf[it] } > cf[b] || parts.minOf { sc[it] } > sc[b]) drop[b] = true
                        continue
                    }
                } else if (parts.size >= 2 && parts.sumOf { txt[it].length } >= COVER * txt[b].length) {
                    if (parts.minOf { cf[it] } > cf[b]) drop[b] = true
                    continue
                }
                inner.forEach { if (cf[b] >= cf[it]) drop[it] = true }
            }
            return words.filterIndexed { i, _ -> !drop[i] }
        }
    }

    override fun close() {
        closed = true
        synchronized(runLock) { yolo.close(); sid?.close(); synchronized(nnapiDet) { nnapiDet.values.forEach { it.close() }; nnapiDet.clear() } }; synchronized(rec) { rec.values.forEach { it.close() }; rec.clear() }
    }
}
