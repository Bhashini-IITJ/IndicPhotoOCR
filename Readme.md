# Internship Contribution - IndicPhotoOCR at IIT Jodhpur 🇮🇳

This repository contains my contributions to the IndicPhotoOCR project as part of my internship at IIT Jodhpur. It includes image collections, annotations, OCR processing scripts, and evaluation results for scene text recognition in Indian scripts.

## 🗂 Project Structure

internship_iit_jodhpur/
├── Locality_Pics/ # Raw images collected from my locality
├── AnnotatedPics/ # Annotated versions of the images
├── Json_Data/ # JSON files with annotations (e.g., LOC_JSON_PICS.json, test_data*.json)
├── Outputs/ # Generated files like wrr_evaluation.csv
├── test_ocr.py # Script for OCR processing
├── evaluate_wrr.py # Script for WRR evaluation
└── README.md # This file

## 📝 Workflow Summary

1. *Image Collection*  
   Scene images were collected locally (e.g., shop boards, banners) containing multilingual text.

2. *Annotation*  
   Annotated using a tool (e.g., Label Studio or similar). Polygons and labels are saved in Json_Data/.

3. *OCR Pipeline*  
   Utilized a custom OCR script (`test_ocr.py`) to:
   - Detect text regions
   - Recognize text (script detection to be refined)

4. *Results*  
   OCR output is intended to be saved in Outputs/ as `ocr_combined_output_poly.json`. The evaluation CSV (`wrr_evaluation.csv`) will also be generated here.

5. *Evaluation*  
   Compared OCR predictions with ground-truth annotations using `evaluate_wrr.py` to compute the Word Recognition Rate (WRR) and save results in `wrr_evaluation.csv`.

## 🔁 How to Re-run OCR and Evaluation

1. Place new images in Locality_Pics/.
2. Update `test_ocr.py` with the correct image directory path.
3. Run the OCR script:
   ```bash
   python test_ocr.py

Example OCR Output Format (JSON)

{
  "locality_01pic.jpg": [
    {
      "text": "मा",
      "script_language": "Hindi"
    }
  ]
}