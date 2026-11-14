package com.iitj.indicphotoocr.app

import com.iitj.indicphotoocr.core.Word

/** Groups words into lines (top to bottom), each line left to right. */
object ReadingOrder {
    fun lines(words: List<Word>): List<List<Word>> {
        if (words.isEmpty()) return emptyList()
        val sorted = words.sortedBy { (it.box[1] + it.box[3]) / 2 }
        val lines = ArrayList<MutableList<Word>>()
        for (w in sorted) {
            val cy = (w.box[1] + w.box[3]) / 2; val h = w.box[3] - w.box[1]
            val line = lines.lastOrNull()
            val ly = line?.let { l -> l.map { (it.box[1] + it.box[3]) / 2 }.average().toFloat() }
            val lh = line?.let { l -> l.map { it.box[3] - it.box[1] }.average().toFloat() }
            if (line != null && Math.abs(cy - ly!!) < 0.5f * kotlin.math.max(h, lh!!)) line += w else lines += mutableListOf(w)
        }
        return lines.map { l -> l.sortedBy { it.box[0] } }
    }
}
