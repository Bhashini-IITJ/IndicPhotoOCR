package com.iitj.indicphotoocr.app

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import com.iitj.indicphotoocr.app.databinding.ActivityInfoBinding

/** One text page opened from the side menu: About Us, Terms of Service, Privacy Policy or Open-source licences. */
class InfoActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val ui = ActivityInfoBinding.inflate(layoutInflater)
        setContentView(ui.root)
        val page = intent.getStringExtra(PAGE) ?: ABOUT
        val (title, body) = when (page) {
            TERMS -> R.string.terms_title to R.string.terms_body
            PRIVACY -> R.string.privacy_title to R.string.privacy_body
            LICENCES -> R.string.licences_title to 0
            else -> R.string.about_us_title to R.string.about_us_body
        }
        ui.title.setText(title)
        if (page == LICENCES) {
            ui.body.textSize = 13f
            ui.body.text = licencesText()
        } else {
            ui.body.setText(body)
        }
        if (page == ABOUT) ui.logos.visibility = View.VISIBLE
        ui.back.setOnClickListener { finish() }
    }

    /** assets/licenses: NOTICE.txt (what each component is, under which licence) followed by the full licence texts. */
    private fun licencesText(): String {
        fun read(name: String) = assets.open("licenses/$name").bufferedReader().use { it.readText() }
        val sb = StringBuilder(read("NOTICE.txt").replace("{SOURCE_URL}", getString(R.string.source_url)))
        for ((heading, file) in LICENCE_FILES) sb.append("\n\n━━━━━━━━━━\n").append(heading).append("\n━━━━━━━━━━\n\n").append(read(file).trim())
        return sb.toString()
    }

    companion object {
        private const val PAGE = "page"
        const val ABOUT = "about"; const val TERMS = "terms"; const val PRIVACY = "privacy"; const val LICENCES = "licences"
        private val LICENCE_FILES = listOf(
            "GNU Affero General Public License v3.0 (this app; Ultralytics YOLO)" to "AGPL-3.0.txt",
            "MIT License (IndicPhotoOCR)" to "MIT-IndicPhotoOCR.txt",
            "MIT License (ONNX Runtime)" to "MIT-onnxruntime.txt",
            "BSD 3-Clause License (torchvision)" to "BSD-3-torchvision.txt",
            "Apache License 2.0 (PARSeq, yolo11n-text, AndroidX, Material Components, Kotlin, BSTD annotations)" to "Apache-2.0.txt",
        )
        fun open(ctx: Context, page: String) = ctx.startActivity(Intent(ctx, InfoActivity::class.java).putExtra(PAGE, page))
    }
}
