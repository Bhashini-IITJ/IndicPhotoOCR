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
<b>[October 2026]:</b> Default text detector is now YOLO11n fine-tuned on BSTD train (TextBPN++ and EAST remain selectable).</br>
<b>[September 2026]:</b> Improved script identification with EfficientNetV2-L, boosting overall WRR by 2%.</br>
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

git clone https://github.com/Asadalk/IndicPhotoOCR.git
cd IndicPhotoOCR
pip install -e .
```

### 🐳 Docker

Requirements: an NVIDIA GPU, Docker, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Prefix the commands with `sudo` if your user is not in the `docker` group.

```bash
git clone https://github.com/Asadalk/IndicPhotoOCR.git
cd IndicPhotoOCR

make build   # docker build -t indicphotoocr:latest .
make test    # run the test suite inside the container
make run     # interactive shell; this checkout is mounted at /workspace/IndicPhotoOCR
```

Inside the container (`make run` opens a shell in `/workspace/IndicPhotoOCR`):

```bash
python end-to-end-Inference.py -p test_images -o predictions.json
python end-to-end-Evaluation.py -g <path/to/BSTD_v17.57.json> -p predictions.json
```

Model weights download on first use (about 5 GB) into the mounted checkout,
so later runs reuse them. To use another image name (for example on a shared
server), pass it to every target: `make build IMAGE=myname`, `make test IMAGE=myname`,
`make run IMAGE=myname`. Without `make`, the equivalent commands are:

```bash
docker build -t indicphotoocr:latest .
docker run --gpus all --rm -it --shm-size=16g -v "$PWD":/workspace/IndicPhotoOCR -w /workspace/IndicPhotoOCR indicphotoocr:latest
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

### Script identification backends

This test branch uses **EfficientNetV2-L by default**. Detection and recognition
models are unchanged. The identifier uses 288×288 RGB crops, bilinear resizing
with antialiasing, and ImageNet normalization. Class order comes from the
checkpoint: Hindi, English, Assamese, Bengali, Gujarati, Kannada, Malayalam,
Marathi, Odia, Punjabi, Tamil, and Telugu. It always predicts across these 12
classes; `identifier_lang` does not restrict EfficientNet predictions.

```python
# Default: EfficientNetV2-L (weights download automatically if missing)
ocr_system = OCR(device="cuda:0", identifier_lang="auto")

# Optional original ViT backend
vit_system = OCR(device="cuda:0", identifier_lang="auto", identifier_type="vit")

# Optional local EfficientNet checkpoint (verified against the release checksum)
local_system = OCR(device="cuda:0", identifier_checkpoint="/path/to/efficientnetv2_l_script_id.pth")
```

The default EfficientNet checkpoint is stored relative to the installed module:
`IndicPhotoOCR/script_identification/efficientnet/models/efficientnet/efficientnetv2_l_script_id.pth`.
This directory must be writable for automatic downloads. An explicit
`identifier_checkpoint` can point to another writable location.

The [test checkpoint release](https://github.com/dikshant-sharma05/IndicPhotoOCR/releases/tag/efficientnetv2-l-v1.0-test)
hosts the inference-only weights (471,575,150 bytes). Downloads show progress,
use a temporary file, and verify SHA-256 before installation. Existing weights
are also verified. Expected SHA-256:

```text
74b6f6e8f7f77b0d2dc1c6f0684bac718ac2b82247bcc393dad804fa4ea15709
```

Detection and required recognition weights download separately when missing.
The first prediction can include recognizer loading/downloads; subsequent
predictions reuse models within the same `OCR` instance. The smaller checkpoint
removes training state, not model weights, and does not reduce inference computation.

### Text detection backends

The default detector is **YOLO11n fine-tuned on the BSTD training split**
(4,737 train / 526 held-out validation images from the official `train` split;
no test images). It predicts axis-aligned word boxes at `imgsz=1280`,
`conf=0.25`, `iou=0.7`, returned as 4-point polygons to the unchanged crop →
script identification → recognition pipeline. TextBPN++ and EAST are still
available.

```python
# Default: fine-tuned YOLO11n (weights download automatically if missing)
ocr_system = OCR(device="cuda:0", identifier_lang="auto")

# Previous default detector
textbpn_system = OCR(device="cuda:0", identifier_lang="auto", detector="textbpn")

# Optional local YOLO checkpoint (verified only when downloaded by the package)
local_yolo = OCR(device="cuda:0", detector_checkpoint="/path/to/yolo11n_bstd_finetuned.pt")
```

The checkpoint is cached at
`IndicPhotoOCR/detection/yolo/models/yolo11n_bstd_finetuned.pt` and is
downloaded from the [YOLO release](https://github.com/Asadalk/IndicPhotoOCR/releases/tag/yolo11n-bstd-v1.0) (5,544,339 bytes) with the
same temporary-file and SHA-256 verification as the EfficientNet weights:

```text
cb41a7f120ff086180e4db5b36001cd9ba6984b129076a6cf697c634aa3489db
```

The YOLO detector requires `ultralytics==8.3.227` (installed with the
package). Note that Ultralytics is licensed under **AGPL-3.0**.

**End-to-end results on the 1,319 BSTD test images** (EfficientNetV2-L script
identification + PARSeq for both rows; scored with `end-to-end-Evaluation.py`,
averaged per image, per language, then over the 10 language groups):

| Detector | Precision | Recall | F1 | WRR |
|---|---:|---:|---:|---:|
| TextBPN++ (`--detector textbpn`) | 0.49 | 0.48 | 0.47 | 0.38 |
| **YOLO11n fine-tuned (default)** | **0.54** | **0.57** | **0.54** | **0.42** |

The YOLO row is the output of `python end-to-end-Inference.py -p <test images>`
on this branch (0 failed images). For TextBPN++, `J_image_241` and
`K_image_128` fail (GPU out of memory, and the CPU retry needs more than 100 GB
RAM), so they are omitted as `end-to-end-Inference.py` omits failed images.

| Language | Bengali* | English | Gujarati | Devanagari* | Kannada | Malayalam | Odia | Punjabi | Tamil | Telugu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n recall | 0.62 | 0.58 | 0.51 | 0.58 | 0.43 | 0.52 | 0.62 | 0.61 | 0.73 | 0.49 |
| YOLO11n WRR | 0.47 | 0.46 | 0.41 | 0.51 | 0.29 | 0.34 | 0.40 | 0.50 | 0.51 | 0.29 |

\*Assamese is merged into Bengali and Marathi into Devanagari (Hindi), as in the evaluator.

### Automated tests

Run from the repository root in an environment with the package dependencies
and pytest installed:

```bash
python -m pytest tests/
```

Unit tests mock downloads and heavy models; no GPU or checkpoint downloads are
needed. Existing integration tests are skipped unless `--run-integration` is
passed. A real end-to-end validation on this test branch completed 20 images
twice without errors and with identical outputs between passes. Exported versus
original V2-L weights also produced identical script predictions on 439 crops.
These checks do not establish that EfficientNet is more accurate than ViT.

### Inference and Evaluation on BSTD
```python
# run the following script while providing path to directory of images
python end-to-end-Inference.py --path </path/to/images>
# by default it will create indicPhotoOCR_predictions.json
# the default detector is the fine-tuned YOLO11n; use --detector textbpn or --detector east to compare
# image file names must match the BSTD keys (e.g. A_image_109.jpg) for the evaluator

# Historical IJDAR results use the original model setup, not the new default.
# Select the original ViT backend in the inference script to reproduce that setup.
python end-to-end-Evaluation.py -g <path/to/bstd/json> -p indicPhotoOCR.json
# bstd json is in the first section named as BSTD_17.57.json in repository
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
- **Text Detection**: YOLO11n via [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0), fine-tuned on BSTD train
- **Text Detection**: TextBPN++ [Original Repository](https://github.com/GXYM/TextBPN-Plus-Plus)
- **EAST Re-implementation**: [EAST Repository](https://github.com/foamliu/EAST)
- **National Language Translation Mission**: [Bhashini](https://bhashini.gov.in/)

## 📬 Contact us
For any queries, please contact us at:
- **[Anik De](mailto:anekde@gmail.com)**
