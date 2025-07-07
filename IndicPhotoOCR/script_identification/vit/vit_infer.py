from transformers import AutoImageProcessor,ViTForImageClassification,pipeline
from PIL import Image
from datasets import DatasetDict,Dataset,ClassLabel
import torchvision.transforms as transforms
import numpy as np
import csv
import os
import argparse
import requests
from tqdm import tqdm
import zipfile
import time
import glob
from IndicPhotoOCR.script_identification.vit.config import infer_config as config

model_info = {
    "hindi": {
        "path": "models/hindienglish",
        "url" : "https://github.com/Bhashini-IITJ/ScriptIdentification/releases/download/Vit_Models/hindienglish.zip",
        "subcategories": ["hindi", "english"]
    },
    "assamese": {
        "path": "models/hindienglishassamese",
        "url": "https://github.com/Bhashini-IITJ/ScriptIdentification/releases/download/Vit_Models/hindienglishassamese.zip",
        "subcategories": ["hindi", "english", "assamese"]
    },
    "bengali": {
        "path": "models/hindienglishbengali",
        "url" : "https://github.com/Bhashini-IITJ/ScriptIdentification/releases/download/Vit_Models/hindienglishbengali.zip",
        "subcategories": ["hindi", "english", "bengali"]
    },
    "gujarati": {
        "path": "models/hindienglishgujarati",
        "url" : "https://github.com/Bhashini-IITJ/ScriptIdentification/releases/download/Vit_Models/hindienglishgujarati.zip",
        "subcategories": ["hindi", "english", "gujarati"]
    },
    "kannada": {
        "path": "models/hindienglishkannada",
        "url" : "https://github.com/Bhashini-IITJ/ScriptIdentification/releases/download/Vit_Models/hindienglishkannada.zip",
        "subcategories": ["hindi", "english", "kannada"]
    },
    "malayalam": {
        "path": "models/hindienglishmalayalam",
        "url" : "https://github.com/Bhashini-IITJ/ScriptIdentification/releases/download/Vit_Models/hindienglishmalayalam.zip",
        "subcategories": ["hindi", "english", "malayalam"]
    },
    "marathi": {
        "path": "models/hindienglishmarathi",
        "url" : "https://github.com/Bhashini-IITJ/ScriptIdentification/releases/download/Vit_Models/hindienglishmarathi.zip",
        "subcategories": ["hindi", "english", "marathi"]
    },
    "meitei": {
        "path": "models/hindienglishmeitei",
        "url" : "https://github.com/Bhashini-IITJ/ScriptIdentification/releases/download/Vit_Models/hindienglishmeitei.zip",
        "subcategories": ["hindi", "english", "meitei"]
    },
    "odia": {
        "path": "models/hindienglishodia",
        "url" : "https://github.com/Bhashini-IITJ/ScriptIdentification/releases/download/Vit_Models/hindienglishodia.zip",
        "subcategories": ["hindi", "english", "odia"]
    },
    "punjabi": {
        "path": "models/hindienglishpunjabi",
        "url" : "https://github.com/Bhashini-IITJ/ScriptIdentification/releases/download/Vit_Models/hindienglishpunjabi.zip",
        "subcategories": ["hindi", "english", "punjabi"]
    },
    "tamil": {
        "path": "models/hindienglishtamil",
        "url" : "https://github.com/Bhashini-IITJ/ScriptIdentification/releases/download/Vit_Models/hindienglishtamil.zip",
        "subcategories": ["hindi", "english", "tamil"]
    },
    "telugu": {
        "path": "models/hindienglishtelugu",
        "url" : "https://github.com/Bhashini-IITJ/ScriptIdentification/releases/download/Vit_Models/hindienglishtelugu.zip",
        "subcategories": ["hindi", "english", "telugu"]
    },
    "auto": {
        "path": "models/12_classes",
        "url" : "https://github.com/Bhashini-IITJ/ScriptIdentification/releases/download/Vit_Models/12_classes.zip",
        "subcategories": ["hindi", "english", "assamese","bengali","gujarati","kannada","malayalam","marathi","odia","punjabi","tamil","telegu"]
    },
    "10C": {
        "path": "models/12_classes",
        "url" : "https://github.com/Bhashini-IITJ/ScriptIdentification/releases/download/Vit_Models/10_classes.zip",
        "subcategories": ["hindi", "english", "assamese","bengali","gujarati","kannada","malayalam","marathi","odia","punjabi","tamil","telegu"]
    },
    

}

pretrained_vit_model = config['pretrained_vit_model']
processor = AutoImageProcessor.from_pretrained(pretrained_vit_model,use_fast=True)


class VIT_identifier:
    def __init__(self):
        pass 

    def unzip_file(self, zip_path, extract_to):

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Extracted files to {extract_to}")




    def ensure_model(self, model_name):
        model_path = model_info[model_name]["path"]
        url = model_info[model_name]["url"]
        root_model_dir = "IndicPhotoOCR/script_identification/vit"
        model_path = os.path.join(root_model_dir, model_path)

        if not os.path.exists(model_path):
            print(f"Model not found locally. Downloading {model_name} from {url}...")

            response = requests.get(url, stream=True)
            zip_path = os.path.join(model_path, "temp_download.zip")

            os.makedirs(model_path, exist_ok=True)

            with open(zip_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_path)

            os.remove(zip_path)

            print(f"Downloaded and extracted to {model_path}")
        
        else:
            # print(f"Model folder already exists: {model_path}")
            pass
        
        return model_path





    def identify(self, image_path,model_name, device):
        model_path = self.ensure_model(model_name)

        vit = ViTForImageClassification.from_pretrained(model_path)
        model= pipeline('image-classification', model=vit, feature_extractor=processor,device=device)

        if image_path.endswith((".png", ".jpg", ".jpeg")):  

            image = Image.open(image_path)
            output = model(image)
            predicted_label = max(output, key=lambda x: x['score'])['label']
            
            # print(f"image_path: {image_path}, predicted_label: {predicted_label}\n")
        
        return predicted_label


    def predict_batch(self, image_dir,model_name,time_show,output_csv="prediction.csv"):
        model_path = self.ensure_model(model_name)
        vit = ViTForImageClassification.from_pretrained(model_path)
        model= pipeline('image-classification', model=vit, feature_extractor=processor,device=0)

        start_time = time.time()
        results=[]
        image_count=0
        for filename in os.listdir(image_dir):
            
            if filename.endswith((".png", ".jpg", ".jpeg")):  
                img_path = os.path.join(image_dir, filename)
                image = Image.open(img_path)
                

                output = model(image)
                predicted_label = max(output, key=lambda x: x['score'])['label'].capitalize()
                
                results.append({"Filepath": filename, "Language": predicted_label})
                image_count+=1
        
        elapsed_time = time.time() - start_time

        if time_show:
            print(f"Time taken to process {image_count} images: {elapsed_time:.2f} seconds")
        
        with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["Filepath", "Language"])
            writer.writeheader()
            writer.writerows(results)
        
        return output_csv


# if __name__ == "__main__":
#     # Argument parser for command line usage
#     parser = argparse.ArgumentParser(description="Image classification using CLIP fine-tuned model")
#     parser.add_argument("--image_path", type=str, help="Path to the input image")
#     parser.add_argument("--image_dir", type=str, help="Path to the input image directory")
#     parser.add_argument("--model_name", type=str, choices=model_info.keys(), help="Name of the model (e.g., hineng, hinengpun, hinengguj)")
#     parser.add_argument("--batch", action="store_true", help="Process images in batch mode if specified")
#     parser.add_argument("--time",type=bool, nargs="?", const=True, default=False, help="Prints the time required to process a batch of images")

#     args = parser.parse_args()


#     # Choose function based on the batch parameter
#     if args.batch:
#         if not args.image_dir:
#             print("Error: image_dir is required when batch is set to True.")
#         else:
#             result = predict_batch(args.image_dir, args.model_name, args.time)
#             print(result)
#     else:
#         if not args.image_path:
#             print("Error: image_path is required when batch is not set.")
#         else:
#             result = predict(args.image_path, args.model_name)
#             print(result)