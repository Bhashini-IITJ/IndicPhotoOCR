pluginManagement {
    repositories { google(); mavenCentral(); gradlePluginPortal() }
}
dependencyResolutionManagement {
    repositories { google(); mavenCentral() }
}
rootProject.name = "IndicPhotoOCR2"
include(":ocrcore", ":verify", ":app", ":model_pack")
