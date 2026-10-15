"""
Tesseract OCR Full Scene Evaluation Script.

Runs Tesseract OCR on scene text images and evaluates predictions against
ground truth using scriptwise PRF (Precision, Recall, F1) and WER/WRR metrics.

Usage:
    python evaluate_tesseract_full_scene.py -t /path/to/test_dir -g /path/to/gt.json -o /path/to/output_dir
    python evaluate_tesseract_full_scene.py -t /path/to/test_dir -g /path/to/gt.json -o /path/to/output_dir --use_crops
"""

import os
import json
import argparse
import re
import string
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm
import jiwer
import pytesseract

# Tesseract binary: use TESSERACT_CMD env variable or default to system PATH
tesseract_binary = os.environ.get("TESSERACT_CMD", "tesseract")
pytesseract.pytesseract.tesseract_cmd = tesseract_binary

# --- CONFIGURATION ---

TESSERACT_LANG_MAP = {
    "hindi": "hin+eng",
    "marathi": "mar+eng",
    "bengali": "ben+eng",
    "assamese": "asm+eng",
    "gujarati": "guj+eng",
    "kannada": "kan+eng",
    "malayalam": "mal+eng",
    "odia": "ori+eng",
    "punjabi": "pan+eng",
    "tamil": "tam+eng",
    "telugu": "tel+eng",
    "urdu": "urd+eng",
}

TARGET_LANGUAGES = {
    "hindi", "english", "bengali", "assamese", "gujarati", "kannada",
    "malayalam", "marathi", "odia", "punjabi", "tamil", "telugu", "urdu",
}

SCRIPT_MAPPING = {
    "assamese": "bengali",
    "marathi": "hindi",
}

PUNC_RE = re.compile(f"[{re.escape(string.punctuation + '।॥')}]")


# --- UTILS ---

def clean_text(text):
    """Removes punctuation, strips whitespace, and lowercases."""
    if not text:
        return ""
    return PUNC_RE.sub("", str(text)).strip().lower()


def normalize_filename(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def word_level_prf(gt_words, pred_words):
    """Multiset Counter based PRF evaluation preserving word frequencies."""
    gt_counter = Counter(gt_words)
    pred_counter = Counter(pred_words)

    tp = sum((gt_counter & pred_counter).values())
    fp = sum((pred_counter - gt_counter).values())
    fn = sum((gt_counter - pred_counter).values())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1_score": f1}


def load_gt_data(gt_path):
    """Loads GT annotations for crop mode AND language detection."""
    gt_boxes_map = {}
    gt_image_langs = {}

    if not gt_path or not os.path.exists(gt_path):
        return gt_boxes_map, gt_image_langs

    with open(gt_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        if isinstance(raw_data, list):
            for item in raw_data:
                img_key = os.path.splitext(os.path.basename(item["image"]))[0]
                gt_boxes_map[img_key] = item.get("bboxes", [])
                lang = item.get("language", "english").lower()
                gt_image_langs.setdefault(img_key, set()).add(lang)
        elif isinstance(raw_data, dict):
            for img_key, val in raw_data.items():
                boxes = []
                langs = set()
                for ann in val.get("annotations", {}).values():
                    coords = ann.get("coordinates", ann.get("polygon", []))
                    if coords:
                        if isinstance(coords[0], (list, tuple)):
                            xs = [pt[0] for pt in coords]
                            ys = [pt[1] for pt in coords]
                            flat = [min(xs), min(ys), max(xs), min(ys), max(xs), max(ys), min(xs), max(ys)]
                        else:
                            flat = coords
                        boxes.append(flat)
                    lang = ann.get("script_language", "english").lower()
                    langs.add(lang)
                gt_boxes_map[img_key] = boxes
                gt_image_langs[img_key] = langs

    return gt_boxes_map, gt_image_langs


def get_test_samples(base_dir, gt_image_langs):
    """Scans the test directory for image files and determines languages."""
    samples = []
    image_dir = os.path.join(base_dir, "images")
    if not os.path.exists(image_dir):
        image_dir = base_dir

    for file in sorted(os.listdir(image_dir)):
        ext = os.path.splitext(file)[1].lower()
        if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue

        img_key = os.path.splitext(file)[0]

        # Try filename-based language (works for IndicSTR12: "assamese_000000.jpg")
        fname_lang = file.split("_")[0].lower()
        if fname_lang in TESSERACT_LANG_MAP or fname_lang == "english":
            langs = {fname_lang}
        elif img_key in gt_image_langs:
            # Fall back to GT annotations (BSTD: "A_image_2839.jpg")
            langs = gt_image_langs[img_key]
        else:
            langs = {"english"}

        samples.append((os.path.join(image_dir, file), langs))

    return samples


def load_gt_json(gt_path):
    """Loads ground truth data for evaluation."""
    with open(gt_path, "r", encoding="utf-8") as f:
        raw_gt = json.load(f)

    gt_json = {}
    if isinstance(raw_gt, list):
        for sample in raw_gt:
            image_key = os.path.splitext(sample["image"])[0]
            annotations = {}
            texts = sample.get("text", [])
            if isinstance(texts, str):
                texts = [texts]
            boxes = sample.get("bboxes", [])
            if not isinstance(boxes, list):
                boxes = []

            for idx, txt in enumerate(texts):
                poly = boxes[idx] if idx < len(boxes) else []
                annotations[f"polygon_{idx+1}"] = {
                    "text": txt,
                    "polygon": poly,
                    "script_language": sample.get("language", "UNK")
                }
            gt_json[image_key] = {"annotations": annotations}
    elif isinstance(raw_gt, dict):
        gt_json = raw_gt

    return gt_json


# --- INFERENCE ---

def process_single_image(sample, gt_boxes_map, psm, use_crops):
    """Processes a single image with Tesseract OCR."""
    img_path, langs = sample
    img_filename = os.path.basename(img_path)
    img_key = os.path.splitext(img_filename)[0]
    langs_str = "+".join(sorted(langs))

    try:
        # Build combined Tesseract language string from all GT languages
        tess_parts = set()
        for lang in langs:
            tl = TESSERACT_LANG_MAP.get(lang, None)
            if tl:
                for part in tl.split("+"):
                    tess_parts.add(part)
            elif lang == "english":
                tess_parts.add("eng")
        if not tess_parts:
            tess_parts = {"eng"}
        tess_lang = "+".join(sorted(tess_parts))

        image = Image.open(img_path).convert("RGB")
        psm_config = f"--psm {psm}"

        annotations = {}

        if use_crops and img_key in gt_boxes_map and len(gt_boxes_map[img_key]) > 0:
            bboxes = gt_boxes_map[img_key]
            for idx, bbox in enumerate(bboxes):
                if len(bbox) >= 8:
                    xs = bbox[0::2]
                    ys = bbox[1::2]
                    crop_box = (
                        max(0, min(xs)),
                        max(0, min(ys)),
                        min(image.width, max(xs)),
                        min(image.height, max(ys))
                    )
                    if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
                        crop_img = image.crop(crop_box)
                        rec_text = pytesseract.image_to_string(crop_img, lang=tess_lang, config=psm_config, timeout=60).strip()
                    else:
                        rec_text = ""
                else:
                    rec_text = ""

                annotations[f"polygon_{idx+1}"] = {
                    "text": rec_text,
                    "confidence": 1.0,
                    "script_language": langs_str,
                }
        else:
            rec_text = pytesseract.image_to_string(
                image,
                lang=tess_lang,
                config=psm_config,
                timeout=60
            )
            annotations["polygon_1"] = {
                "text": rec_text,
                "confidence": 1.0,
                "script_language": langs_str,
            }

        return img_filename, annotations

    except Exception as e:
        return img_filename, {
            "polygon_1": {
                "text": "",
                "confidence": 0.0,
                "script_language": langs_str,
            }
        }


def run_tesseract_inference(test_samples, gt_boxes_map, args):
    """Runs Tesseract OCR inference on test images."""
    predictions_json = {}

    print(f"🚀 Running Tesseract OCR Inference (PSM={args.psm}, crop_mode={args.use_crops}, workers={args.num_workers})...")

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_single_image, s, gt_boxes_map, args.psm, args.use_crops): s
                for s in test_samples
            }
            for future in tqdm(as_completed(futures), total=len(test_samples), desc="Parallel Tesseract OCR"):
                fname, annos = future.result()
                predictions_json[fname] = annos
    else:
        for s in tqdm(test_samples, desc="Tesseract OCR"):
            fname, annos = process_single_image(s, gt_boxes_map, args.psm, args.use_crops)
            predictions_json[fname] = annos

    return predictions_json


# --- EVALUATION ---

def evaluate_predictions(predictions_json, gt_json, output_dir):
    """Runs scriptwise PRF and WER/WRR evaluation."""
    print("\n" + "=" * 60)
    print("📊 RUNNING TEXT RECOGNITION EVALUATION (Tesseract)")
    print("=" * 60)

    scriptwise_gt = defaultdict(lambda: defaultdict(list))

    for pred_filename in predictions_json:
        image_key = normalize_filename(pred_filename)
        if image_key not in gt_json:
            continue

        for ann in gt_json[image_key]["annotations"].values():
            lang = ann.get("script_language", "UNK")
            if lang not in TARGET_LANGUAGES:
                continue
            lang = SCRIPT_MAPPING.get(lang, lang)

            txt = ann.get("text", "")
            if isinstance(txt, list):
                for item in txt:
                    for word in str(item).split():
                        w_clean = clean_text(word)
                        if w_clean and w_clean.upper() != "UNK":
                            scriptwise_gt[image_key][lang].append(w_clean)
            else:
                for word in str(txt).split():
                    w_clean = clean_text(word)
                    if w_clean and w_clean.upper() != "UNK":
                        scriptwise_gt[image_key][lang].append(w_clean)

    pred_cleaned = {}
    for filename, annotations in predictions_json.items():
        key = normalize_filename(filename)
        words = []
        for v in annotations.values():
            raw_t = v.get("text", "")
            for w in raw_t.split():
                cw = clean_text(w)
                if cw:
                    words.append(cw)
        pred_cleaned[key] = words

    # --- PRF Metrics ---
    language_counts = defaultdict(int)
    language_p, language_r, language_f = defaultdict(float), defaultdict(float), defaultdict(float)

    for image_key, lang_dict in scriptwise_gt.items():
        pred_texts = pred_cleaned.get(image_key, [])
        for lang, gt_texts in lang_dict.items():
            result = word_level_prf(list(gt_texts), pred_texts)
            language_counts[lang] += 1
            language_p[lang] += result["precision"]
            language_r[lang] += result["recall"]
            language_f[lang] += result["f1_score"]

    prf_report = ["\n" + "=" * 60, "🏆 SCRIPTWISE PRF (Precision, Recall, F1) EVALUATION (Tesseract)", "=" * 60, "\nLanguage, Precision, Recall, F1-score"]
    for lang in sorted(language_counts):
        total = language_counts[lang]
        prf_report.append(f"{lang}, {language_p[lang]/total:.4f}, {language_r[lang]/total:.4f}, {language_f[lang]/total:.4f}")

    if language_counts:
        num_langs = len(language_counts)
        macro_p = sum(language_p[l]/language_counts[l] for l in language_counts) / num_langs
        macro_r = sum(language_r[l]/language_counts[l] for l in language_counts) / num_langs
        macro_f = sum(language_f[l]/language_counts[l] for l in language_counts) / num_langs
        prf_report.extend([f"Average Precision: {macro_p:.4f}", f"Average Recall: {macro_r:.4f}", f"Average F1-score: {macro_f:.4f}"])

    prf_text = "\n".join(prf_report)
    print(prf_text)
    with open(os.path.join(output_dir, "report_prf.txt"), "w") as f:
        f.write(prf_text)

    # --- WER / WRR Metrics ---
    lang_stats = defaultdict(lambda: {"total_wer": 0, "count": 0, "total_words": 0})
    for img_key, lang_dict in scriptwise_gt.items():
        pred_texts = pred_cleaned.get(img_key, [])
        for lang, gt_list in lang_dict.items():
            # Maintain natural word sequence without sorting
            gt_str = " ".join(gt_list)
            pred_str = " ".join(pred_texts)
            wer = min(1.0, jiwer.wer(gt_str, pred_str)) if gt_str else 0.0
            lang_stats[lang]["total_wer"] += wer
            lang_stats[lang]["count"] += 1
            lang_stats[lang]["total_words"] += len(gt_list)

    wer_report = ["\n" + "=" * 60, "🏆 SCRIPTWISE WER & WRR EVALUATION (Tesseract)", "=" * 60, f"\n{'Language':<12} | {'WER':<6} | {'WRR':<8} | {'GT Words':<10}", "-" * 45]
    for lang in sorted(lang_stats.keys()):
        s = lang_stats[lang]
        avg_wer = s["total_wer"] / s["count"] if s["count"] > 0 else 0.0
        wrr = (1 - avg_wer) * 100
        wer_report.append(f"{lang.capitalize():<12} | {avg_wer:.4f}   | {wrr:.4f}% | {s['total_words']:<10}")

    if lang_stats:
        num_langs = len(lang_stats)
        macro_wer = sum(s['total_wer']/s['count'] for s in lang_stats.values() if s['count'] > 0) / num_langs
        wer_report.extend([f"\nOverall Average WER: {macro_wer:.4f}", f"Overall Average WRR: {(1 - macro_wer) * 100:.4f}%"])

    wer_text = "\n".join(wer_report)
    print(wer_text)
    with open(os.path.join(output_dir, "report_wer_wrr.txt"), "w") as f:
        f.write(wer_text)


# --- MAIN ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tesseract OCR Full Scene Evaluation Script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python evaluate_tesseract_full_scene.py -t /path/to/test_dir -g /path/to/gt.json -o /path/to/output_dir
    python evaluate_tesseract_full_scene.py -t /path/to/test_dir -g /path/to/gt.json -o /path/to/output_dir --use_crops --psm 6
        """
    )

    parser.add_argument(
        "--test_dir", "-t",
        required=True,
        type=str,
        help="Path to test images directory"
    )
    parser.add_argument(
        "--gt_path", "-g",
        required=True,
        type=str,
        help="Path to ground truth JSON file"
    )
    parser.add_argument(
        "--output_dir", "-o",
        required=True,
        type=str,
        help="Path to directory for output predictions and reports"
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=11,
        help="Tesseract Page Segmentation Mode (PSM). Default 11 (sparse text). Use 6 for crops."
    )
    parser.add_argument(
        "--use_crops",
        action="store_true",
        help="If enabled, crop bounding boxes from GT before running Tesseract OCR."
    )
    parser.add_argument(
        "--num_workers", "-w",
        type=int,
        default=16,
        help="Number of parallel CPU worker processes for Tesseract (default: 16)."
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load GT data for language detection and crop mode
    gt_boxes_map, gt_image_langs = load_gt_data(args.gt_path)

    print("🔍 Scanning Test Data...")
    test_samples = get_test_samples(args.test_dir, gt_image_langs)
    print(f"🎯 Found {len(test_samples)} test images.")

    # Run inference
    predictions_json = run_tesseract_inference(test_samples, gt_boxes_map, args)

    # Save predictions
    output_json_path = os.path.join(args.output_dir, "output_predictions.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(predictions_json, f, ensure_ascii=False, indent=4)
    print(f"💾 Saved predictions to {output_json_path}")

    # Run evaluation if GT is available
    if args.gt_path and os.path.exists(args.gt_path):
        gt_json = load_gt_json(args.gt_path)
        evaluate_predictions(predictions_json, gt_json, args.output_dir)

    print("\n💾 Evaluation completed for Tesseract!")
