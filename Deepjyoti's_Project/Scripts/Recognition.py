import os
import json
import glob
import csv
from collections import defaultdict
import tkinter as tk
from tkinter import filedialog, messagebox


def load_ground_truth(gt_path):
    """
    Load ground truth texts and their script languages from a JSON file.
    Returns a list of tuples: (text, script_language).
    """
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    base = os.path.splitext(os.path.basename(gt_path))[0]
    ann_key = 'AnnotationDataset_' + base
    texts = []
    ann = data.get(ann_key, {}).get('annotations', {})
    for key, poly in sorted(ann.items()):
        texts.append((poly['text'], poly.get('script_language', 'Unknown')))
    return texts


def load_ocr_output(ocr_path):
    """
    Load OCR output tokens from a JSON file.
    Flattens nested lists into a single list of tokens.
    """
    with open(ocr_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [token for line in data for token in line]


def evaluate_folders(gt_folder, ocr_folder, csv_path):
    """
    Compare all ground truth and OCR outputs to compute:
      - Word Recognition Rate (WRR) per file
      - Language-wise WRR across all files
      - Overall WRR
    Saves results in the specified CSV path.
    """
    file_results = []
    lang_counts = defaultdict(lambda: {'total': 0, 'recognized': 0})
    overall = {'total': 0, 'recognized': 0}

    gt_files = glob.glob(os.path.join(gt_folder, '*.json'))
    for gt_path in gt_files:
        base = os.path.splitext(os.path.basename(gt_path))[0]
        ocr_path = os.path.join(ocr_folder, f"{base}_ocr_output.json")
        if not os.path.exists(ocr_path):
            continue

        gt_entries = load_ground_truth(gt_path)
        ocr_tokens = set(load_ocr_output(ocr_path))

        total = len(gt_entries)
        recognized = sum(1 for text, _ in gt_entries if text in ocr_tokens)

        # Update language and overall counts
        for text, lang in gt_entries:
            lang_counts[lang]['total'] += 1
            overall['total'] += 1
            if text in ocr_tokens:
                lang_counts[lang]['recognized'] += 1
                overall['recognized'] += 1

        wrr = recognized / total if total else 0
        file_results.append({
            'level': 'file',
            'name': base,
            'total_words': total,
            'recognized_words': recognized,
            'wrr': f"{wrr:.4f}"
        })

    # Add language-wise results
    for lang, counts in lang_counts.items():
        total = counts['total']
        rec = counts['recognized']
        wrr = rec / total if total else 0
        file_results.append({
            'level': 'language',
            'name': lang,
            'total_words': total,
            'recognized_words': rec,
            'wrr': f"{wrr:.4f}"
        })

    # Add overall result
    overall_wrr = overall['recognized'] / overall['total'] if overall['total'] else 0
    file_results.append({
        'level': 'overall',
        'name': 'all',
        'total_words': overall['total'],
        'recognized_words': overall['recognized'],
        'wrr': f"{overall_wrr:.4f}"
    })

    # Write to CSV
    fieldnames = ['level', 'name', 'total_words', 'recognized_words', 'wrr']
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in file_results:
            writer.writerow(row)


def browse_folder(entry):
    folder = filedialog.askdirectory()
    if folder:
        entry.delete(0, tk.END)
        entry.insert(0, folder)


def browse_file(entry):
    file = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV files', '*.csv')])
    if file:
        entry.delete(0, tk.END)
        entry.insert(0, file)


def run_evaluation(gt_entry, ocr_entry, csv_entry):
    gt_folder = gt_entry.get()
    ocr_folder = ocr_entry.get()
    csv_path = csv_entry.get()

    if not (os.path.isdir(gt_folder) and os.path.isdir(ocr_folder)):
        messagebox.showerror('Error', 'Please select valid folders for ground truth and OCR output.')
        return
    if not csv_path:
        messagebox.showerror('Error', 'Please specify a path for the output CSV file.')
        return

    try:
        evaluate_folders(gt_folder, ocr_folder, csv_path)
        messagebox.showinfo('Success', f'WRR evaluation saved to {csv_path}')
    except Exception as e:
        messagebox.showerror('Error', str(e))


def build_gui():
    root = tk.Tk()
    root.title('OCR WRR Evaluation')
    root.geometry('600x200')

    # Ground truth folder
    tk.Label(root, text='Ground Truth Folder:').grid(row=0, column=0, sticky='e', padx=5, pady=5)
    gt_entry = tk.Entry(root, width=50)
    gt_entry.grid(row=0, column=1, padx=5)
    tk.Button(root, text='Browse', command=lambda: browse_folder(gt_entry)).grid(row=0, column=2, padx=5)

    # OCR output folder
    tk.Label(root, text='OCR Output Folder:').grid(row=1, column=0, sticky='e', padx=5, pady=5)
    ocr_entry = tk.Entry(root, width=50)
    ocr_entry.grid(row=1, column=1, padx=5)
    tk.Button(root, text='Browse', command=lambda: browse_folder(ocr_entry)).grid(row=1, column=2, padx=5)

    # CSV output file
    tk.Label(root, text='Output CSV File:').grid(row=2, column=0, sticky='e', padx=5, pady=5)
    csv_entry = tk.Entry(root, width=50)
    csv_entry.grid(row=2, column=1, padx=5)
    tk.Button(root, text='Browse', command=lambda: browse_file(csv_entry)).grid(row=2, column=2, padx=5)

    # Run button
    tk.Button(root, text='Run Evaluation', width=20,
              command=lambda: run_evaluation(gt_entry, ocr_entry, csv_entry)).grid(row=3, column=1, pady=20)

    root.mainloop()


if __name__ == '__main__':
    build_gui()
