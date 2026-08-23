<p align="center">
  <img src="./static/pics/IndicPhotoOCR_LOGO.png" alt="IndicPhotoOCR Logo" width="45%">
  <h3 align="center">
A Comprehensive Toolkit for Scene Text Recognition in Indian Languages
  </h3>
</p>
<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2511.23071-B31B1B?style=flat-square&logo=arxiv&logoColor=white)](https://www.arxiv.org/pdf/2511.23071)
[![Hugging Face](https://img.shields.io/badge/Demo-Hugging%20Face-FF6C00?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/Bhashini-IITJ/IndicPhotoOCR)
[![Colab](https://img.shields.io/badge/Notebook-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1BILXjUF2kKKrzUJ_evubgLHl2busPiH2)
[![Documentation](https://img.shields.io/badge/Docs-ReadTheDocs-007BFF?style=flat-square&logo=readthedocs&logoColor=white)](https://bhashini-iitj.github.io/IndicPhotoOCR/)
[![Project Page](https://img.shields.io/badge/Project-Page-6C63FF?style=flat-square&logo=academia&logoColor=white)](https://vl2g.github.io/projects/IndicPhotoOCR/)

[![Open Source](https://img.shields.io/badge/Bhashini-Open%20Source-FF6C00?style=flat-square)](https://bhashini.gov.in/)
[![Stars](https://img.shields.io/github/stars/Bhashini-IITJ/IndicPhotoOCR?style=flat-square&color=blue)](https://github.com/Bhashini-IITJ/IndicPhotoOCR/stargazers)
[![Forks](https://img.shields.io/github/forks/Bhashini-IITJ/IndicPhotoOCR?style=flat-square&color=blue)](https://github.com/Bhashini-IITJ/IndicPhotoOCR/network/members)
[![License](https://img.shields.io/github/license/Bhashini-IITJ/IndicPhotoOCR?style=flat-square&color=green)](LICENSE)

</div>
<hr style="width: 100%; border: 1px solid #000;">

Welcome to **IndicPhotoOCR**! ⚡ We've built an scene text recognition toolkit designed for detecting, identifying, and recognizing text across **11 Indian languages** (plus English). 

**Supported Languages:** Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, and English. (with Urdu and Meitei in the pipeline!)



![](static/pics/visualizeIndicPhotoOCR.png)
<hr style="width: 100%; border: 1px solid #000;">


<hr style="width: 100%; border: 1px solid #000;">

## 📅 Updates Timeline
<b>[August 2026]:</b> Oral presentation [ICDAR 2026](https://icdar2026.org/) in Vienna, Austria.</br> 
<b>[April 2026]:</b> Accepted at International Journal on Document Analysis and Recognition (IJDAR) 20026.</br>
<b>[August 2025]:</b> [Project page](https://vl2g.github.io/projects/IndicPhotoOCR/) created.</br>
<b>[April 2025]:</b> [Documentation page](https://bhashini-iitj.github.io/IndicPhotoOCR/) created using Sphnix.</br>
<b>[March 2025]:</b> Support for [Huggingface Demo](https://huggingface.co/spaces/Bhashini-IITJ/IndicPhotoOCR) extened to 12 languages.</br>
<b>[Feburary 2025]:</b> Added option to choose between tri-lingual and 12 class script identifiction models.</br>
<b>[Feburary 2025]:</b> Added recoginition models for Malayalam and Kannada.</br>
<b>[January 2025]:</b> Added ViT based script identification models.</br>
<b>[January 2025]:</b> Demo available in [huggingface space](https://huggingface.co/spaces/Bhashini-IITJ/IndicPhotoOCR).</br>
Currently demo supports scene images containing bi-lingual Hindi and English text.  
<b>[December 2024]:</b> Detection Module: TextBPN++ added.\
<b>[November 2024]:</b> Code available at [Google Colab](https://colab.research.google.com/drive/1BILXjUF2kKKrzUJ_evubgLHl2busPiH2?usp=sharing).\
<b>[November 2024]:</b> Added support for 10 languages in the recognition module.</br>
<b>[September 2024]:</b> Repository created.

<hr style="width: 100%; border: 1px solid #000;">

## 📦 Quick Installation

We recommend creating a virtual environment before installing:
```bash
conda create -n indicphotoocr python=3.10 -y
conda activate indicphotoocr

git clone https://github.com/Bhashini-IITJ/IndicPhotoOCR.git
cd IndicPhotoOCR
pip install -e .
```

<hr style="width: 100%; border: 1px solid #000;">

## 💡 How to Use

Using `IndicPhotoOCR` is incredibly simple. You can execute the entire End-to-End Scene Text Recognition pipeline (Detection ➡️ Identification ➡️ Recognition) with just three lines of Python!

### 💥 End-to-End Pipeline (Fastest Method)
```python
from IndicPhotoOCR.ocr import OCR

# Initialize the OCR Engine
ocr_system = OCR(verbose=False, identifier_lang="auto", device="cuda:0")

# Boom! Run the whole pipeline natively
results = ocr_system.ocr("test_images/image_141.jpg")

# The output is a structured list of lines (paragraphs), where each line is a list of words sequentially ordered left-to-right!
# Example Output:
# [
#    ["राजीव", "चौक", "मेट्रो", "स्टेशन"],   <-- Line 1
#    ["Rajiv", "Chowk", "Metro", "Station"]  <-- Line 2
# ]
fast_results = ocr_system.ocr("test_images/image_141.jpg", batch_size=32)
```

### Inference and Evaluation on BSTD
```python
# run the following script while providing path to directory of images
python end-to-end-Inference.py --path </path/to/images>
# by default it will create indicPhotoOCR_predictions.json

# use the following script to reproduce the results provided in the IJDAR version
python end-to-end-Evaluation.py -g <path/to/bstd/json> -p indicPhotoOCR.json
# bstd json is in the first section named as BSTD_17.57.json in repository
```

### 🤖 Open Source Models Evaluation (Surya OCR & Tesseract OCR)
Evaluate open-source baseline models (Surya OCR and Tesseract OCR) on scene text datasets:

```bash
# Evaluate Surya OCR model
python open_source_models/evaluate_surya_full_scene.py \
    -t <path/to/images> \
    -g <path/to/gt.json> \
    -o <path/to/output_dir>

# Evaluate Tesseract OCR model
python open_source_models/evaluate_tesseract_full_scene.py \
    -t <path/to/images> \
    -g <path/to/gt.json> \
    -o <path/to/output_dir>

# Optional: Run in crop mode (crop GT bounding boxes before recognition)
python open_source_models/evaluate_surya_full_scene.py -t <path/to/images> -g <path/to/gt.json> -o <path/to/output_dir> --use_crops
```



### 🎯 Modular Execution (Advanced)
If you do not want to run the entire pipeline at once, you can hook into individual modules manually:

<details>
<summary><b>1. Text Detection Module</b></summary>
Extract coordinates of all bounding boxes containing text in an image.

```python
from IndicPhotoOCR.ocr import OCR

ocr_system = OCR(verbose=True, device="cuda:0")

# Get raw bounding box detections
detections = ocr_system.detect("test_images/image_141.jpg")

# Optional: Visualize and save the detected bounding boxes
ocr_system.visualize_detection("test_images/image_141.jpg", detections)
# Saves an image with boxes drawn over it
```
</details>


<details>
<summary><b>2. Script Identification Module</b></summary>
Take a single, cropped image of a word and predict what language it is written in.

```python
from IndicPhotoOCR.ocr import OCR

ocr_system = OCR(verbose=True, identifier_lang="auto", device="cuda:0")

# Identify script of a cropped word
lang = ocr_system.identify("test_images/cropped_word.jpg")
print(lang)
# Output: 'hindi'
```
</details>


<details>
<summary><b>3. Text Recognition Module</b></summary>
Extract the literal text string from a cropped word image (and optionally get its confidence score).

```python
from IndicPhotoOCR.ocr import OCR

ocr_system = OCR(verbose=True, device="cuda:0")

# Recognize text (old behavior, returns string)
text = ocr_system.recognise("test_images/cropped_word.jpg", "hindi")

# Recognize text WITH Confidence Score (new behavior)
text, conf_score = ocr_system.recognise("test_images/cropped_word.jpg", "hindi", return_confidence=True)
print(f"Recognized: {text} | Certainty: {conf_score * 100:.2f}%")
```
</details>

<hr style="width: 100%; border: 1px solid #000;">

## 📚 Related Datasets & Citations
- **Bharat Scene Text Dataset** - [BSTD](https://github.com/Bhashini-IITJ/BharatSceneTextDataset)

> 🎉 **Our paper has been officially accepted in IJDAR** (International Journal on Document Analysis and Recognition)!

If you use IndicPhotoOCR in your research, please cite us:
```bibtex
@article{De2026,
  author    = {De, Anik and Penamakuri, Abhirama Subramanyam and Yadav, Rajeev and Rathore, Aditya and Shah, Harshiv and Sharma, Devesh and Agarwal, Sagar and Kumar, Pravin and Mishra, Anand},
  title     = {Bharat scene text: a novel comprehensive dataset and benchmark for indian language scene text understanding},
  journal   = {International Journal on Document Analysis and Recognition (IJDAR)},
  year      = {2026},
  issn      = {1433-2825},
  doi       = {10.1007/s10032-026-00583-9},
  url       = {https://doi.org/10.1007/s10032-026-00583-9}
}
```

## 🤝 Project Contributors
| <img src="https://github.com/anikde/anikde.github.io/blob/main/Anik_New_2.jpg" width="100" style="border-radius:15px;"> |
|:---------------------------------:|
| **[Anik De](https://www.linkedin.com/in/anik-de/)** - Tech Lead & Main Contributor |

| <img src="https://abhiram4572.github.io/images/personal.jpeg" width="100" style="border-radius:15px;"> | <img src="https://github.com/Bhashini-IITJ/SceneTextDetection/releases/download/Photos/Aditya_Rathor.jpeg" width="100" style="border-radius:15px;"> | <img src="https://github.com/Bhashini-IITJ/SceneTextDetection/releases/download/Photos/harshiv.jpg" width="100" style="border-radius:15px;"> |
|:---:|:---:|:---:|
| [Abhirama](https://abhiram4572.github.io/) | [Aditya Rathore](https://www.linkedin.com/in/aditya-rathor-87829324b/) | [Harshiv Shah](https://www.linkedin.com/in/harshivshah27/) | 

| <img src="https://github.com/Bhashini-IITJ/SceneTextDetection/releases/download/Photos/sagar_agrawal.png" width="100" style="border-radius:15px;"> | <img src="https://github.com/Bhashini-IITJ/SceneTextDetection/releases/download/Photos/rajeev.png" width="100" style="border-radius:15px;"> | <img src="https://github.com/Bhashini-IITJ/SceneTextDetection/releases/download/Photos/pravin.JPG" width="100" style="border-radius:15px;"> |
|:---:|:---:|:---:|
| [Sagar Agarwal](https://www.linkedin.com/in/sagar-agrawal-4a0b94106/) | [Rajeev Yadav](https://www.linkedin.com/in/rajeev-yadav/) | [Pravin Kumar](https://www.linkedin.com/in/prvnkmr9060/) |

| <img src="https://anandmishra22.github.io/files/Mishra_oct22.png" width="100" style="border-radius:15px;"> |
|:---------------------------------:|
| **[Anand Mishra](https://anandmishra22.github.io/)** - Project Investigator |

## 🙏 Acknowledgements
- **Text Recognition**: [PARseq](https://github.com/baudm/parseq)
- **Text Detection**: TextBPN++ [Original Repository](https://github.com/GXYM/TextBPN-Plus-Plus)
- **EAST Re-implementation**: [EAST Repository](https://github.com/foamliu/EAST)
- **National Language Translation Mission**: [Bhashini](https://bhashini.gov.in/)

## 📬 Contact us
For any queries, please contact us at:
- **[Anik De](mailto:anekde@gmail.com)**
