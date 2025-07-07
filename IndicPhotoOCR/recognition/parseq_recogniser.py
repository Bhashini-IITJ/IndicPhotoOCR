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
        pass

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

    def get_model_output(self, device, model, image_path):
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

        return text

        # Ensure model file exists; download directly if not
    def ensure_model(self, model_name):
        model_path = model_info[model_name]["path"]
        url = model_info[model_name]["url"]
        root_model_dir = "IndicPhotoOCR/recognition/"
        model_path = os.path.join(root_model_dir, model_path)
        
        if not os.path.exists(model_path):
            print(f"Model not found locally. Downloading {model_name} from {url}...")
            
            # Start the download with a progress bar
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            os.makedirs(f"{root_model_dir}/models", exist_ok=True)
            
            with open(model_path, "wb") as f, tqdm(
                    desc=model_name,
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    bar.update(len(data))

            print(f"Downloaded model for {model_name}.")
            
        return model_path

    def bstr(checkpoint, language, image_dir, save_dir):
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
            model = load_model(device, checkpoint)
        else:
            model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval().to(device)

        parseq_dict = {}
        for image_path in tqdm(os.listdir(image_dir)):
            assert os.path.exists(os.path.join(image_dir, image_path)) == True, f"{image_path}"
            text = get_model_output(device, model, os.path.join(image_dir, image_path), language=f"{language}")
        
            filename = image_path.split('/')[-1]
            parseq_dict[filename] = text

        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/{language}_test.json", 'w') as json_file:
            json.dump(parseq_dict, json_file, indent=4, ensure_ascii=False)


    def bstr_onImage(checkpoint, language, image_path):
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
            model = load_model(device, checkpoint)
        else:
            model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval().to(device)

        # parseq_dict = {}
        # for image_path in tqdm(os.listdir(image_dir)):
        #     assert os.path.exists(os.path.join(image_dir, image_path)) == True, f"{image_path}"
        text = get_model_output(device, model, image_path, language=f"{language}")
        
        return text


    def recognise(self, checkpoint: str, image_path: str, language: str, verbose: bool, device: str) -> str:
        """
        Loads the desired model and returns the recognized word from the specified image.

        Args:
            checkpoint (str): Path to the model checkpoint file.
            language (str): Language code (e.g., 'hindi', 'english').
            image_path (str): Path to the image for which text recognition is needed.

        Returns:
            str: The recognized text from the image.
        """
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if language != "english":
            model_path = self.ensure_model(checkpoint)
            model = self.load_model(device, model_path)
        else:
            model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True, verbose=verbose).eval().to(device)

        recognized_text = self.get_model_output(device, model, image_path)
        
        return recognized_text
# if __name__ == '__main__':
#     fire.Fire(main)