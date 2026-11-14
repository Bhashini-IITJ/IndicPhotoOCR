package com.iitj.indicphotoocr.core

import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * 8-bit RGB image, interleaved (r, g, b) row-major. Platform independent so the same pipeline runs
 * on Android (from a Bitmap) and on the JVM (for verification against the Python reference).
 *
 * The resize functions follow OpenCV's conventions (pixel centres at +0.5, replicated borders), which
 * the Python reference pipeline (mobile_app/tools/reference_pipeline.py) uses.
 */
class RgbImage(val width: Int, val height: Int, val data: ByteArray = ByteArray(width * height * 3)) {
    init {
        require(data.size == width * height * 3) { "data size ${data.size} != ${width}x${height}x3" }
    }

    operator fun get(x: Int, y: Int, c: Int): Int = data[(y * width + x) * 3 + c].toInt() and 0xFF

    /** Rows [y1, y2] and columns [x1, x2], inclusive. */
    fun crop(x1: Int, y1: Int, x2: Int, y2: Int): RgbImage {
        val w = x2 - x1 + 1
        val h = y2 - y1 + 1
        val out = ByteArray(w * h * 3)
        for (y in 0 until h) System.arraycopy(data, ((y1 + y) * width + x1) * 3, out, y * w * 3, w * 3)
        return RgbImage(w, h, out)
    }

    companion object {
        /** From ARGB ints (Android Bitmap.getPixels / Java BufferedImage.getRGB). */
        fun fromArgb(width: Int, height: Int, argb: IntArray): RgbImage {
            val d = ByteArray(width * height * 3)
            for (i in 0 until width * height) {
                val p = argb[i]
                d[i * 3] = (p shr 16 and 0xFF).toByte(); d[i * 3 + 1] = (p shr 8 and 0xFF).toByte(); d[i * 3 + 2] = (p and 0xFF).toByte()
            }
            return RgbImage(width, height, d)
        }
    }
}

enum class Interp { LINEAR, CUBIC, AREA }

/** One axis of a separable resize: for each output index, source indices and weights. */
private class AxisWeights(val idx: Array<IntArray>, val w: Array<FloatArray>)

private fun clampI(v: Int, lo: Int, hi: Int) = if (v < lo) lo else if (v > hi) hi else v

private fun linearAxis(src: Int, dst: Int): AxisWeights {
    val s = src.toDouble() / dst
    val idx = Array(dst) { IntArray(2) }; val w = Array(dst) { FloatArray(2) }
    for (d in 0 until dst) {
        var f = (d + 0.5) * s - 0.5
        if (f < 0) f = 0.0
        val i0 = floor(f).toInt(); val t = (f - i0).toFloat()
        idx[d][0] = clampI(i0, 0, src - 1); idx[d][1] = clampI(i0 + 1, 0, src - 1)
        w[d][0] = 1f - t; w[d][1] = t
    }
    return AxisWeights(idx, w)
}

private fun cubicAxis(src: Int, dst: Int): AxisWeights {
    val a = -0.75
    fun k(x: Double): Double {
        val ax = Math.abs(x)
        return when {
            ax <= 1 -> ((a + 2) * ax - (a + 3)) * ax * ax + 1
            ax < 2 -> ((a * ax - 5 * a) * ax + 8 * a) * ax - 4 * a
            else -> 0.0
        }
    }
    val s = src.toDouble() / dst
    val idx = Array(dst) { IntArray(4) }; val w = Array(dst) { FloatArray(4) }
    for (d in 0 until dst) {
        val f = (d + 0.5) * s - 0.5
        val i0 = floor(f).toInt(); val t = f - i0
        for (j in 0 until 4) {
            idx[d][j] = clampI(i0 - 1 + j, 0, src - 1)
            w[d][j] = k(t - (j - 1)).toFloat()
        }
    }
    return AxisWeights(idx, w)
}

/** Box filter with fractional coverage (OpenCV INTER_AREA when shrinking). */
private fun areaAxis(src: Int, dst: Int): AxisWeights {
    val s = src.toDouble() / dst
    val idx = arrayOfNulls<IntArray>(dst); val w = arrayOfNulls<FloatArray>(dst)
    for (d in 0 until dst) {
        val a = d * s; val b = min((d + 1) * s, src.toDouble())
        val i0 = floor(a).toInt(); val i1 = min(src - 1, kotlin.math.ceil(b).toInt() - 1)
        val ii = IntArray(i1 - i0 + 1); val ww = FloatArray(i1 - i0 + 1)
        for (i in i0..i1) {
            val cover = min(b, (i + 1).toDouble()) - max(a, i.toDouble())
            ii[i - i0] = i; ww[i - i0] = (cover / s).toFloat()
        }
        idx[d] = ii; w[d] = ww
    }
    @Suppress("UNCHECKED_CAST")
    return AxisWeights(idx as Array<IntArray>, w as Array<FloatArray>)
}

private fun axis(src: Int, dst: Int, interp: Interp) = when {
    interp == Interp.AREA && dst < src -> areaAxis(src, dst)
    interp == Interp.CUBIC -> cubicAxis(src, dst)
    else -> linearAxis(src, dst)
}

/**
 * Separable resize to (dw, dh). AREA shrinks by box filtering (and falls back to LINEAR on an
 * axis that is enlarged, as OpenCV does); results are rounded and clamped to 0..255.
 */
fun RgbImage.resize(dw: Int, dh: Int, interp: Interp): RgbImage {
    if (dw == width && dh == height) return this
    val ax = axis(width, dw, interp); val ay = axis(height, dh, interp)
    val tmp = FloatArray(height * dw * 3)
    for (y in 0 until height) {
        val row = y * width * 3
        for (x in 0 until dw) {
            val ii = ax.idx[x]; val ww = ax.w[x]
            var r = 0f; var g = 0f; var b = 0f
            for (j in ii.indices) {
                val p = row + ii[j] * 3; val wt = ww[j]
                r += (data[p].toInt() and 0xFF) * wt; g += (data[p + 1].toInt() and 0xFF) * wt; b += (data[p + 2].toInt() and 0xFF) * wt
            }
            val o = (y * dw + x) * 3
            tmp[o] = r; tmp[o + 1] = g; tmp[o + 2] = b
        }
    }
    val out = ByteArray(dw * dh * 3)
    for (y in 0 until dh) {
        val ii = ay.idx[y]; val ww = ay.w[y]
        for (x in 0 until dw) {
            var r = 0f; var g = 0f; var b = 0f
            for (j in ii.indices) {
                val p = (ii[j] * dw + x) * 3; val wt = ww[j]
                r += tmp[p] * wt; g += tmp[p + 1] * wt; b += tmp[p + 2] * wt
            }
            val o = (y * dw + x) * 3
            out[o] = clampI(r.roundToInt(), 0, 255).toByte(); out[o + 1] = clampI(g.roundToInt(), 0, 255).toByte(); out[o + 2] = clampI(b.roundToInt(), 0, 255).toByte()
        }
    }
    return RgbImage(dw, dh, out)
}

/**
 * The pipeline's crop resize (reference_pipeline.resize): one axis at a time, width first, each pass
 * rounded to 8-bit; area-average on an axis that shrinks, [up] on an axis that grows.
 */
fun RgbImage.resizeForModel(dw: Int, dh: Int, up: Interp): RgbImage {
    var im = this
    if (im.width != dw) im = im.resize(dw, im.height, if (im.width > dw) Interp.AREA else up)
    if (im.height != dh) im = im.resize(dw, dh, if (im.height > dh) Interp.AREA else up)
    return im
}
