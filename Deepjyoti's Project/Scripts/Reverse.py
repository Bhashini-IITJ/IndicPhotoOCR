import os
import json
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np

def draw_polygons_on_image(image, polygons):
    for i, poly in enumerate(polygons):
        points = np.array(poly['coordinates'], np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        # Get the center of the polygon to put the index number
        moments = cv2.moments(points)
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            cv2.putText(image, str(i), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return image

def process_images(image_dir, json_dir, output_dir):
    processed = 0
    for image_name in os.listdir(image_dir):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(image_dir, image_name)
        json_filename = os.path.splitext(image_name)[0] + ".json"
        json_path = os.path.join(json_dir, json_filename)

        if not os.path.exists(json_path):
            print(f"⚠️ No matching JSON for: {image_name}")
            continue

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                root_key = next(iter(data))
                annotations = data[root_key]['annotations']
                polygons = [v for v in annotations.values()]
        except Exception as e:
            print(f"❌ Error reading {json_filename}: {e}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ Could not load image: {img_path}")
            continue

        drawn_image = draw_polygons_on_image(image, polygons)

        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, drawn_image)
        print(f"✅ Saved: {output_path}")
        processed += 1

    messagebox.showinfo("Done", f"Processed {processed} images.")

def run_gui():
    root = tk.Tk()
    root.withdraw()

    img_dir = filedialog.askdirectory(title="Select folder containing images")
    if not img_dir:
        messagebox.showerror("Error", "No image folder selected.")
        return

    json_dir = filedialog.askdirectory(title="Select folder containing JSON annotation files")
    if not json_dir:
        messagebox.showerror("Error", "No JSON folder selected.")
        return

    out_dir = filedialog.askdirectory(title="Select output folder to save visualized images")
    if not out_dir:
        messagebox.showerror("Error", "No output folder selected.")
        return

    process_images(img_dir, json_dir, out_dir)

if __name__ == "__main__":
    run_gui()
