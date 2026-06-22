import json
import os
import re
import string
from collections import defaultdict
import jiwer
import argparse

# --- CONFIGURATION ---
TARGET_LANGUAGES = {
    "hindi", "english", "bengali", "gujarati", "kannada",
    "malayalam", "marathi", "odia", "punjabi", "tamil", "telugu"
}

# Mapping for scripts that share Unicode blocks or evaluation categories
SCRIPT_MAPPING = {
    "assamese": "bengali",
    "marathi": "hindi"
}

UNICODE_RANGES = {
    "english": (0x0000, 0x007F),
    "bengali": (0x0980, 0x09FF),
    "gujarati": (0x0A80, 0x0AFF),
    "hindi": (0x0900, 0x097F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
    "odia": (0x0B00, 0x0B7F),
    "punjabi": (0x0A00, 0x0A7F),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F)
}

PUNC_RE = re.compile(f'[{re.escape(string.punctuation)}]')

# --- METRICS CALCULATION ---

def calculate_prf_metrics(ground_truth_words, predicted_words):
    """
    Calculates Precision, Recall, and F1 Score based on set intersection.
    """
    gt_set = set(ground_truth_words)
    pred_set = set(predicted_words)

    true_positives = len(gt_set.intersection(pred_set))
    false_positives = len(pred_set.difference(gt_set))
    false_negatives = len(gt_set.difference(pred_set))

    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0.0

    try:
        recall = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        recall = 0.0

    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

# --- UTILS ---

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_text(text):
    """Removes punctuation, strips whitespace, and lowercases."""
    if not text:
        return ""
    text = PUNC_RE.sub('', text)
    return text.strip().lower()

def normalize_filename(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def detect_language(word):
    if not word:
        return "unknown"
    first_char_code = ord(word[0])
    for lang, (start, end) in UNICODE_RANGES.items():
        if start <= first_char_code <= end:
            return SCRIPT_MAPPING.get(lang, lang)
    return "unknown"

# --- CORE LOGIC ---

def process_data(gt_json, pred_json):
    """Filters and cleans GT and Pred data in one pass."""
    pred_keys = {normalize_filename(k) for k in pred_json.keys()}
    scriptwise_gt = defaultdict(lambda: defaultdict(list))
    pred_cleaned = {}

    # Clean Prediction Data
    for filename, annotations in pred_json.items():
        key = normalize_filename(filename)
        pred_cleaned[key] = [clean_text(v.get("text", "")) for v in annotations.values() if v.get("text")]

    # Clean Ground Truth Data (Only for matched files)
    for key, gt_entry in gt_json.items():
        if key not in pred_keys:
            continue
        
        for ann in gt_entry.get("annotations", {}).values():
            text = clean_text(ann.get("text", ""))
            lang = ann.get("script_language", "UNK").lower()
            lang = SCRIPT_MAPPING.get(lang, lang)

            if not text or text == "unk" or lang not in TARGET_LANGUAGES:
                continue
            
            scriptwise_gt[key][lang].append(text)
    
    return scriptwise_gt, pred_cleaned

def calculate_metrics(scriptwise_gt, pred_cleaned):
    """
    Calculates both PRF metrics and WER for each language.
    """
    lang_stats = defaultdict(lambda: {
        "total_precision": 0,
        "total_recall": 0,
        "total_f1": 0,
        "total_wer": 0,
        "count": 0,
        "total_words": 0
    })

    for img_key, lang_dict in scriptwise_gt.items():
        # Group predictions by detected language
        pred_texts = pred_cleaned.get(img_key, [])
        pred_by_lang = defaultdict(list)
        for word in pred_texts:
            pred_by_lang[detect_language(word)].append(word)

        for lang, gt_list in lang_dict.items():
            pred_list = pred_by_lang[lang]
            
            # Calculate PRF metrics
            prf = calculate_prf_metrics(gt_list, pred_list)
            
            # Calculate Bag of Words WER
            gt_str = " ".join(sorted(gt_list))
            pred_str = " ".join(sorted(pred_list))
            
            # Calculate WER (capped at 1.0)
            wer = min(1.0, jiwer.wer(gt_str, pred_str)) if gt_str else 0
            
            # Accumulate stats
            lang_stats[lang]["total_precision"] += prf["precision"]
            lang_stats[lang]["total_recall"] += prf["recall"]
            lang_stats[lang]["total_f1"] += prf["f1_score"]
            lang_stats[lang]["total_wer"] += wer
            lang_stats[lang]["count"] += 1
            lang_stats[lang]["total_words"] += len(gt_list)

    print_results(lang_stats)

def print_results(stats):
    """
    Prints combined evaluation results: Precision, Recall, F1 Score, and WRR for each language.
    WRR (Word Recognition Rate) = 1 - WER
    """
    print(f"\n{'Language':<12} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<9} | {'WRR':<6} | {'GT Words':<10}")
    print("-" * 75)
    
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_wrr = 0
    total_words = 0
    lang_count = 0
    
    for lang in sorted(stats.keys()):
        s = stats[lang]
        avg_precision = s["total_precision"] / s["count"] if s["count"] > 0 else 0
        avg_recall = s["total_recall"] / s["count"] if s["count"] > 0 else 0
        avg_f1 = s["total_f1"] / s["count"] if s["count"] > 0 else 0
        avg_wer = s["total_wer"] / s["count"] if s["count"] > 0 else 0
        avg_wrr = 1 - avg_wer
        
        print(f"{lang.capitalize():<12} | {avg_precision:<10.2f} | {avg_recall:<8.2f} | {avg_f1:<9.2f} | {avg_wrr:<6.2f} | {s['total_words']:<10}")
        
        total_precision += avg_precision
        total_recall += avg_recall
        total_f1 += avg_f1
        total_wrr += avg_wrr
        total_words += s['total_words']
        lang_count += 1
    
    print("-" * 75)
    avg_precision_all = total_precision / lang_count if lang_count > 0 else 0
    avg_recall_all = total_recall / lang_count if lang_count > 0 else 0
    avg_f1_all = total_f1 / lang_count if lang_count > 0 else 0
    avg_wrr_all = total_wrr / lang_count if lang_count > 0 else 0
    print(f"{'Average':<12} | {avg_precision_all:<10.2f} | {avg_recall_all:<8.2f} | {avg_f1_all:<9.2f} | {avg_wrr_all:<6.2f} | {total_words:<10}")

# --- MAIN ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End OCR Evaluation Script Script.")
    
    # Adding command line arguments
    parser.add_argument(
        "--gt_path", "-g", 
        required=True, 
        type=str, 
        help="Path to the ground truth BSTD JSON file."
    )
    parser.add_argument(
        "--pred_path", "-p", 
        required=True, 
        type=str, 
        help="Path to the prediction JSON file using the end-to-end-Inference.py"
    )
    
    args = parser.parse_args()

    # Load data using parsed arguments
    gt_data = load_json(args.gt_path)
    pred_data = load_json(args.pred_path)

    gt_filtered, pred_filtered = process_data(gt_data, pred_data)
    calculate_metrics(gt_filtered, pred_filtered)