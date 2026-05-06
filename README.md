<p align="center">
  <img src="./static/pics/IndicPhotoOCR_LOGO.png" alt="IndicPhotoOCR Logo" width="45%">
  <h3 align="center">
A Comprehensive Toolkit for Scene Text Recognition in Indian Languages
  </h3>
</p>
<div align="center">

[![Open Source](https://img.shields.io/badge/Open%20Source-Bhashini-FF6C00)](https://bhashini.gov.in/)
![Visitor Count](https://visitor-badge.laobi.icu/badge?page_id=Bhashini-IITJ.IndicPhotoOCR)
![GitHub Repo stars](https://img.shields.io/github/stars/Bhashini-IITJ/IndicPhotoOCR?style=social)
![GitHub forks](https://img.shields.io/github/forks/Bhashini-IITJ/IndicPhotoOCR?style=social)
[![arXiv](https://img.shields.io/badge/arXiv-2401.01234-b31b1b.svg?style=flat-square)](https://www.arxiv.org/pdf/2511.23071)

[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Demo-FF6C00?logo=Huggingface&logoColor=white)](https://huggingface.co/spaces/Bhashini-IITJ/IndicPhotoOCR)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BILXjUF2kKKrzUJ_evubgLHl2busPiH2?usp=sharing#scrollTo=3v76fsYYzVvz)
[![Documentation](https://img.shields.io/badge/Documentation-Click%20Here-007BFF?style=for-the-badge&logo=ReadTheDocs&logoColor=white)](https://bhashini-iitj.github.io/IndicPhotoOCR/)
</div>
<hr style="width: 100%; border: 1px solid #000;">

Welcome to **IndicPhotoOCR**! ⚡ We've built an fast, robust, and comprehensive scene text recognition toolkit designed for detecting, identifying, and recognizing text across **11 Indian languages** (plus English). 

**Supported Languages:** Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, and English. (with Urdu and Meitei in the pipeline!)

It is expertly crafted to handle the unique scripts and complex structures of Indian languages. And with our latest **upgrades**, it runs **faster** natively, with built-in support for **Batch Inference** and precision **Confidence Scoring**! 🔥

![](static/pics/visualizeIndicPhotoOCR.png)
<hr style="width: 100%; border: 1px solid #000;">

## ✨ What's New & Exciting?
- ⚡ **Batch Inference Engine**: Got an image with hundreds of words? Just pass `batch_size=32` into the core engine to process bounding boxes concurrently, slashing execution times even further.
- 🎯 **Confidence Scores**: Our public APIs now optionally expose exact neural network confidence probabilities so you can reliably filter out low-certainty predictions or calculate metrics.
- 🛡️ **Atomic & Self-Contained**: Auto-downloads models safely without corrupting, and uses system-agnostic absolute paths so you can run it from anywhere.

<hr style="width: 100%; border: 1px solid #000;">

## 📅 Updates Timeline
<b>[April 2026]:</b> Added Batch Inference Engine, Model Caching, and Neural Confidence Scores resulting in better speedup.</br>
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
conda create -n indicphotoocr python=3.9 -y
conda activate indicphotoocr

git clone https://github.com/Bhashini-IITJ/IndicPhotoOCR.git
cd IndicPhotoOCR
chmod +x setup.sh
./setup.sh
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

# 🔥 PRO-TIP: Process very large images with thousands of words concurrently!
fast_results = ocr_system.ocr("test_images/image_141.jpg", batch_size=32)
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
@misc{ipo,
  author = {Anik De et al.},
  title      = {{I}ndic{P}hoto{O}CR: A comprehensive toolkit for {I}ndian language scene text understanding},
  howpublished = {\url{https://github.com/Bhashini-IITJ/IndicPhotoOCR/}},
  year         = 2024,
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
