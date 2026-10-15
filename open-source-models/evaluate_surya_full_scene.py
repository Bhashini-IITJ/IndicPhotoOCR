"""
Surya OCR Full Scene Evaluation Script.

Runs Surya OCR (detection + recognition) on scene text images and evaluates
predictions against ground truth using scriptwise PRF (Precision, Recall, F1)
and WER/WRR metrics.

Usage:
    python evaluate_surya_full_scene.py -t /path/to/test_dir -g /path/to/gt.json -o /path/to/output_dir
    python evaluate_surya_full_scene.py -t /path/to/test_dir -g /path/to/gt.json -o /path/to/output_dir --use_crops
"""

import os
import json
import argparse
import re
import string
from collections import defaultdict, Counter
from PIL import Image
from tqdm import tqdm
import jiwer
import torch

from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor, FoundationPredictor

# --- CONFIGURATION ---

SURYA_LANG_MAP = {
    "hindi": "hi",
    "marathi": "mr",
    "bengali": "bn",
    "assamese": "as",
    "gujarati": "gu",
    "kannada": "kn",
    "malayalam": "ml",
    "odia": "or",
    "punjabi": "pa",
    "tamil": "ta",
    "telugu": "te",
    "urdu": "ur"
}

TARGET_LANGUAGES = {
    "hindi", "english", "bengali", "assamese", "gujarati", "kannada",
    "malayalam", "marathi", "odia", "punjabi", "tamil", "telugu", "urdu",
}

SCRIPT_MAPPING = {
    "assamese": "bengali",
    "marathi": "hindi"
}

PUNC_RE = re.compile(f'[{re.escape(string.punctuation + "।॥")}]')


# --- UTILS ---

def clean_text(text):
    """Removes punctuation, strips whitespace, and lowercases."""
    if not text:
        return ""
    return PUNC_RE.sub('', str(text)).strip().lower()


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
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1_score": f1}


def get_test_samples(base_dir):
    """Scans the test directory for image files and extracts language from filename."""
    samples = []
    image_dir = os.path.join(base_dir, "images")
    if not os.path.exists(image_dir):
        image_dir = base_dir

    for file in sorted(os.listdir(image_dir)):
        ext = os.path.splitext(file)[1].lower()
        if ext in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}:
            lang = file.split("_")[0].lower()
            samples.append((os.path.join(image_dir, file), lang))
    return samples


def load_gt_boxes(gt_path):
    """Loads ground truth bounding boxes for crop mode."""
    gt_boxes_map = {}
    if not gt_path or not os.path.exists(gt_path):
        return gt_boxes_map

    with open(gt_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
        if isinstance(raw, list):
            for item in raw:
                key = os.path.splitext(os.path.basename(item["image"]))[0]
                gt_boxes_map[key] = item.get("bboxes", [])
        elif isinstance(raw, dict):
            for key, val in raw.items():
                boxes = []
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
                gt_boxes_map[key] = boxes
    return gt_boxes_map


def load_gt_json(gt_path):
    """Loads ground truth data for evaluation."""
    with open(gt_path, 'r', encoding='utf-8') as f:
        raw_gt = json.load(f)

    gt_json = {}
    if isinstance(raw_gt, list):
        for sample in raw_gt:
            image_key = os.path.splitext(os.path.basename(sample["image"]))[0]
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

def run_surya_inference(test_samples, gt_boxes_map, args):
    """Runs Surya OCR inference on test images."""
    print("⚙️ Loading Surya OCR Predictors...")
    det_predictor = DetectionPredictor()
    foundation_predictor = FoundationPredictor()
    rec_predictor = RecognitionPredictor(foundation_predictor)

    predictions_json = {}
    batch_size = args.batch_size

    print(f"🚀 Running Surya OCR Inference (crop_mode={args.use_crops})...")

    if args.use_crops:
        for img_path, script_lang in tqdm(test_samples, desc="Surya Crop OCR"):
            img_filename = os.path.basename(img_path)
            img_key = os.path.splitext(img_filename)[0]
            try:
                img = Image.open(img_path).convert("RGB")
                lang = SURYA_LANG_MAP.get(script_lang, "en")
                bboxes = gt_boxes_map.get(img_key, [])
                annotations = {}
                if bboxes:
                    crop_imgs = []
                    valid_indices = []
                    for idx, bbox in enumerate(bboxes):
                        if len(bbox) >= 8:
                            xs = bbox[0::2]
                            ys = bbox[1::2]
                            crop_box = (max(0, min(xs)), max(0, min(ys)), min(img.width, max(xs)), min(img.height, max(ys)))
                            if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
                                crop_imgs.append(img.crop(crop_box))
                                valid_indices.append(idx)

                    if crop_imgs:
                        rec_results = rec_predictor(crop_imgs, [lang]*len(crop_imgs))
                        for idx, rec in zip(valid_indices, rec_results):
                            rec_text = getattr(rec, "text", "")
                            annotations[f"polygon_{idx+1}"] = {
                                "text": rec_text,
                                "confidence": 1.0,
                                "script_language": script_lang
                            }
                predictions_json[img_filename] = annotations
            except Exception as e:
                print(f"Error processing crops for {img_filename}: {e}")
                predictions_json[img_filename] = {
                    "polygon_1": {"text": "", "confidence": 0.0, "script_language": script_lang}
                }
    else:
        for i in tqdm(range(0, len(test_samples), batch_size), desc="Surya OCR Batches"):
            batch_samples = test_samples[i:i+batch_size]
            batch_images = []
            batch_langs = []
            batch_meta = []

            for img_path, script_lang in batch_samples:
                try:
                    img = Image.open(img_path).convert("RGB")
                    lang = SURYA_LANG_MAP.get(script_lang, "en")
                    batch_images.append(img)
                    batch_langs.append(lang)
                    batch_meta.append((os.path.basename(img_path), script_lang))
                except Exception as e:
                    print(f"Error opening {img_path}: {e}")

            if not batch_images:
                continue

            try:
                results = rec_predictor(batch_images, det_predictor=det_predictor)
                for (img_filename, script_lang), page_ocr in zip(batch_meta, results):
                    text_lines = []
                    if hasattr(page_ocr, "text_lines"):
                        for line in page_ocr.text_lines:
                            text_lines.append(getattr(line, "text", ""))

                    full_text = " ".join(text_lines)
                    predictions_json[img_filename] = {
                        "polygon_1": {
                            "text": full_text,
                            "confidence": 1.0,
                            "script_language": script_lang
                        }
                    }
            except Exception as e:
                print(f"Error running Surya OCR batch: {e}")
                for (img_filename, script_lang) in batch_meta:
                    predictions_json[img_filename] = {
                        "polygon_1": {
                            "text": "",
                            "confidence": 0.0,
                            "script_language": script_lang
                        }
                    }

    return predictions_json


# --- EVALUATION ---

def evaluate_predictions(predictions_json, gt_json, output_dir):
    """Runs scriptwise PRF and WER/WRR evaluation."""
    print("\n" + "=" * 60)
    print("📊 RUNNING TEXT RECOGNITION EVALUATION (Surya OCR)")
    print("=" * 60)

    scriptwise_gt = defaultdict(lambda: defaultdict(list))
    for pred_filename in predictions_json:
        image_key = normalize_filename(pred_filename)
        if image_key not in gt_json:
            continue
        for ann in gt_json[image_key].get("annotations", {}).values():
            txt = ann.get("text", "")
            lang = ann.get("script_language", "UNK")
            if lang not in TARGET_LANGUAGES:
                continue
            lang = SCRIPT_MAPPING.get(lang, lang)

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

    prf_report = ["\n" + "=" * 60, "🏆 SCRIPTWISE PRF (Precision, Recall, F1) EVALUATION (Surya OCR)", "=" * 60, "\nLanguage, Precision, Recall, F1-score"]
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

    wer_report = ["\n" + "=" * 60, "🏆 SCRIPTWISE WER & WRR EVALUATION (Surya OCR)", "=" * 60, f"\n{'Language':<12} | {'WER':<6} | {'WRR':<8} | {'GT Words':<10}", "-" * 45]
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
        description="Surya OCR Full Scene Evaluation Script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python evaluate_surya_full_scene.py -t /path/to/test_dir -g /path/to/gt.json -o /path/to/output_dir
    python evaluate_surya_full_scene.py -t /path/to/test_dir -g /path/to/gt.json -o /path/to/output_dir --use_crops
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
        "--batch_size", "-b",
        type=int,
        default=16,
        help="Batch size for Surya OCR inference (default: 16)"
    )
    parser.add_argument(
        "--use_crops",
        action="store_true",
        help="If enabled, crop bounding boxes from GT before running Surya OCR."
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("🔍 Scanning Test Data...")
    test_samples = get_test_samples(args.test_dir)
    print(f"🎯 Found {len(test_samples)} test images.")

    # Load GT bounding boxes for crop mode
    gt_boxes_map = load_gt_boxes(args.gt_path)

    # Run inference
    predictions_json = run_surya_inference(test_samples, gt_boxes_map, args)

    # Save predictions
    output_json_path = os.path.join(args.output_dir, "output_predictions.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(predictions_json, f, ensure_ascii=False, indent=4)
    print(f"💾 Saved predictions to {output_json_path}")

    # Run evaluation if GT is available
    if args.gt_path and os.path.exists(args.gt_path):
        gt_json = load_gt_json(args.gt_path)
        evaluate_predictions(predictions_json, gt_json, args.output_dir)

    print("\n💾 Evaluation completed for Surya OCR!")
