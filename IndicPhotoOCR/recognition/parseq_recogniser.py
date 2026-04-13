import csv
# import fire
import json
import numpy as np
import os
# import pandas as pd
import sys
import torch
import requests

from dataclasses import dataclass
from PIL import Image
from nltk import edit_distance
from torchvision import transforms as T
from typing import Optional, Callable, Sequence, Tuple
from tqdm import tqdm


from IndicPhotoOCR.utils.strhub.data.module import SceneTextDataModule
from IndicPhotoOCR.utils.strhub.models.utils import load_from_checkpoint


model_info = {
    "assamese": {
        "path": "models/assamese.ckpt",
        "url" : "https://github.com/anikde/STocr/releases/download/V2.0.0/assamese.ckpt",
    },
    "bengali": {
        "path": "models/bengali.ckpt",
        "url" : "https://github.com/anikde/STocr/releases/download/V2.0.0/bengali.ckpt",
    },
    "hindi": {
        "path": "models/hindi.ckpt",
        "url" : "https://github.com/anikde/STocr/releases/download/V2.0.0/hindi.ckpt",
    },
    "gujarati": {
        "path": "models/gujarati.ckpt",
        "url" : "https://github.com/anikde/STocr/releases/download/V2.0.0/gujarati.ckpt",
    },
    "kannada": {
        "path": "models/kannada.ckpt",
        "url" : "https://github.com/anikde/STocr/releases/download/V2.0.0/kannada.ckpt",
    },
    "malayalam": {
        "path": "models/malayalam.ckpt",
        "url" : "https://github.com/anikde/STocr/releases/download/V2.0.0/malayalam.ckpt",
    },
    "marathi": {
        "path": "models/marathi.ckpt",
        "url" : "https://github.com/anikde/STocr/releases/download/V2.0.0/marathi.ckpt",
    },
    "odia": {
        "path": "models/odia.ckpt",
        "url" : "https://github.com/anikde/STocr/releases/download/V2.0.0/odia.ckpt",
    },
    "punjabi": {
        "path": "models/punjabi.ckpt",
        "url" : "https://github.com/anikde/STocr/releases/download/V2.0.0/punjabi.ckpt",
    },
    "tamil": {
        "path": "models/tamil.ckpt",
        "url" : "https://github.com/anikde/STocr/releases/download/V2.0.0/tamil.ckpt",
    },
    "telugu": {
        "path": "models/telugu.ckpt",
        "url" : "https://github.com/anikde/STocr/releases/download/V2.0.0/telugu.ckpt",
    }
}

class PARseqrecogniser:
    def __init__(self):
        self._model_cache = {}

    def get_transform(self, img_size: Tuple[int], augment: bool = False, rotation: int = 0):
        transforms = []
        if augment:
            from .augment import rand_augment_transform
            transforms.append(rand_augment_transform())
        if rotation:
            transforms.append(lambda img: img.rotate(rotation, expand=True))
        transforms.extend([
            T.Resize(img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
        return T.Compose(transforms)


    def load_model(self, device, checkpoint):
        model = load_from_checkpoint(checkpoint).eval().to(device)
        return model

    def get_model_output(self, device, model, image_path, return_confidence=False):
        hp = model.hparams
        transform = self.get_transform(hp.img_size, rotation=0)

        image_name = image_path.split("/")[-1]
        img = Image.open(image_path).convert('RGB')
        img = transform(img)
        logits = model(img.unsqueeze(0).to(device))
        probs = logits.softmax(-1)
        preds, probs = model.tokenizer.decode(probs)
        text = model.charset_adapter(preds[0])
        scores = probs[0].detach().cpu().numpy()
        confidence = float(scores.mean()) if len(scores) > 0 else 0.0

        if return_confidence:
            return text, confidence
        return text

    def get_model_output_batch(self, device, model, image_paths, return_confidence=False, batch_size=32):
        hp = model.hparams
        transform = self.get_transform(hp.img_size, rotation=0)

        results = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            imgs = []
            for path in batch_paths:
                img = Image.open(path).convert('RGB')
                imgs.append(transform(img))
            
            img_tensor = torch.stack(imgs).to(device)
            
            # Disable grad for inference speed!
            with torch.no_grad():
                logits = model(img_tensor)
                probs = logits.softmax(-1)
                preds, probs = model.tokenizer.decode(probs)
            
            for j in range(len(preds)):
                text = model.charset_adapter(preds[j])
                scores = probs[j].detach().cpu().numpy()
                confidence = float(scores.mean()) if len(scores) > 0 else 0.0
                
                if return_confidence:
                    results.append((text, confidence))
                else:
                    results.append(text)
        
        return results

        # Ensure model file exists; download directly if not
    def ensure_model(self, model_name):
        model_path = model_info[model_name]["path"]
        url = model_info[model_name]["url"]
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_path)
        
        if not os.path.exists(model_path):
            print(f"Model not found locally. Downloading {model_name} from {url}...")
            
            # Start the download with a progress bar
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            os.makedirs(os.path.join(current_dir, "models"), exist_ok=True)
            
            tmp_path = model_path + ".tmp"
            with open(tmp_path, "wb") as f, tqdm(
                    desc=model_name,
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    bar.update(len(data))
            
            os.rename(tmp_path, model_path)

            print(f"Downloaded model for {model_name}.")
            
        return model_path

    def bstr(self, checkpoint, language, image_dir, save_dir):
        """
        Runs the OCR model to process images and save the output as a JSON file.

        Args:
            checkpoint (str): Path to the model checkpoint file.
            language (str): Language code (e.g., 'hindi', 'english').
            image_dir (str): Directory containing the images to process.
            save_dir (str): Directory where the output JSON file will be saved.

        Example usage:
            python your_script.py --checkpoint /path/to/checkpoint.ckpt --language hindi --image_dir /path/to/images --save_dir /path/to/save
        """
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        if language != "english":
            model = self.load_model(device, checkpoint)
        else:
            model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval().to(device)

        parseq_dict = {}
        for image_path in tqdm(os.listdir(image_dir)):
            assert os.path.exists(os.path.join(image_dir, image_path)) == True, f"{image_path}"
            text = self.get_model_output(device, model, os.path.join(image_dir, image_path))
        
            filename = image_path.split('/')[-1]
            parseq_dict[filename] = text

        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/{language}_test.json", 'w') as json_file:
            json.dump(parseq_dict, json_file, indent=4, ensure_ascii=False)


    def bstr_onImage(self, checkpoint, language, image_path):
        """
        Runs the OCR model to process images and save the output as a JSON file.

        Args:
            checkpoint (str): Path to the model checkpoint file.
            language (str): Language code (e.g., 'hindi', 'english').
            image_dir (str): Directory containing the images to process.
            save_dir (str): Directory where the output JSON file will be saved.

        Example usage:
            python your_script.py --checkpoint /path/to/checkpoint.ckpt --language hindi --image_dir /path/to/images --save_dir /path/to/save
        """
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        if language != "english":
            model = self.load_model(device, checkpoint)
        else:
            model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval().to(device)

        text = self.get_model_output(device, model, image_path)
        
        return text


    def recognise(self, checkpoint: str, image_path: str, language: str, verbose: bool, device: str, return_confidence: bool = False):
        """
        Loads the desired model and returns the recognized word from the specified image.

        Args:
            checkpoint (str): Path to the model checkpoint file.
            language (str): Language code (e.g., 'hindi', 'english').
            image_path (str): Path to the image for which text recognition is needed.

        Returns:
            str or tuple: The recognized text from the image as a string. If return_confidence is True, returns (text, confidence) tuple.
        """
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if language not in self._model_cache:
            if language != "english":
                model_path = self.ensure_model(checkpoint)
                self._model_cache[language] = self.load_model(device, model_path)
            else:
                self._model_cache[language] = torch.hub.load('baudm/parseq', 'parseq', pretrained=True, verbose=verbose).eval().to(device)

        model = self._model_cache[language]

        result = self.get_model_output(device, model, image_path, return_confidence=return_confidence)
        
        return result

    def recognise_batch(self, checkpoint: str, image_paths: list, language: str, verbose: bool, device: str, return_confidence: bool = False, batch_size: int = 32) -> list:
        """
        Loads the desired model and returns recognized words for a batch of images.

        Args:
            checkpoint (str): Path to the model checkpoint file.
            image_paths (list): List of paths to the images.
            language (str): Language code.
            verbose (bool): Whether to print verbose output.
            device (str): Device to run inference on.
            return_confidence (bool): Whether to return (text, confidence) tuples.
            batch_size (int): Size of the image batch.

        Returns:
            list: List of recognized texts or (text, confidence) tuples.
        """
        if language not in self._model_cache:
            if language != "english":
                model_path = self.ensure_model(language)
                self._model_cache[language] = self.load_model(device, model_path)
            else:
                self._model_cache[language] = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval().to(device)

        model = self._model_cache[language]

        results = self.get_model_output_batch(device, model, image_paths, return_confidence=return_confidence, batch_size=batch_size)
        
        return results
# if __name__ == '__main__':
#     fire.Fire(main)