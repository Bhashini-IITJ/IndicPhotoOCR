package com.iitj.indicphotoocr.verify

import com.iitj.indicphotoocr.core.OcrEngine
import com.iitj.indicphotoocr.core.RgbImage
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import javax.imageio.ImageIO

/** args: modelDir suffix imageDir(*.png, already EXIF-oriented) out.json [threads]; env PRUNE_MODE = test6 | d12 | d123 */
fun main(args: Array<String>) {
    val (modelDir, suffix, imageDir, outPath) = args
    val threads = args.getOrNull(4)?.toInt() ?: 8
    val engine = OcrEngine(File(modelDir), if (suffix == "-") "" else suffix, threads)
    System.getenv("PRUNE_MODE")?.let { engine.pruneMode = it }
    System.getenv("TUNE")?.let { cands ->  // e.g. TUNE=cpu,xnnpack,nnapi: run the accelerator check on the first image
        val f = File(args[2]).listFiles { x -> x.name.endsWith(".png") }!!.minByOrNull { it.name }!!
        val bi = ImageIO.read(f)
        val t0 = System.nanoTime()
        val choice = engine.tuneAccelerators(RgbImage.fromArgb(bi.width, bi.height, bi.getRGB(0, 0, bi.width, bi.height, null, 0, bi.width)), cands.split(","))
        println("tune ${f.name}: $choice in ${(System.nanoTime() - t0) / 1_000_000} ms")
    }
    val out = JSONObject()
    val files = File(imageDir).listFiles { f -> f.name.endsWith(".png") }!!.sortedBy { it.name }
    for ((i, f) in files.withIndex()) {
        val bi = ImageIO.read(f)
        val argb = bi.getRGB(0, 0, bi.width, bi.height, null, 0, bi.width)
        val r = engine.run(RgbImage.fromArgb(bi.width, bi.height, argb))
        val words = JSONArray()
        for (w in r.words) words.put(JSONObject().put("box", JSONArray(w.box.map { it.toDouble() })).put("score", w.score.toDouble())
            .put("si", w.script).put("text", w.text ?: JSONObject.NULL).put("conf", w.confidence?.toDouble() ?: JSONObject.NULL))
        out.put(f.nameWithoutExtension, JSONObject().put("words", words).put("ms", JSONObject(r.millis)))
        println("${i + 1}/${files.size} ${f.nameWithoutExtension} words=${r.words.size} ms=${r.millis}")
    }
    File(outPath).writeText(out.toString())
    engine.close()
}
