
import torch
import clip
from PIL import Image
from io import BytesIO
import os
import requests

# Model information dictionary containing model paths and language subcategories
model_info = {
    "hindi": {
        "path": "models/clip_finetuned_hindienglish_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglish_real.pth",
        "subcategories": ["hindi", "english"]
    },
    "hinengasm": {
        "path": "models/clip_finetuned_hindienglishassamese_real.pth",
        "url": "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishassamese_real.pth",
        "subcategories": ["hindi", "english", "assamese"]
    },
    "hinengben": {
        "path": "models/clip_finetuned_hindienglishbengali_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishbengali_real.pth",
        "subcategories": ["hindi", "english", "bengali"]
    },
    "hinengguj": {
        "path": "models/clip_finetuned_hindienglishgujarati_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishgujarati_real.pth",
        "subcategories": ["hindi", "english", "gujarati"]
    },
    "hinengkan": {
        "path": "models/clip_finetuned_hindienglishkannada_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishkannada_real.pth",
        "subcategories": ["hindi", "english", "kannada"]
    },
    "hinengmal": {
        "path": "models/clip_finetuned_hindienglishmalayalam_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishmalayalam_real.pth",
        "subcategories": ["hindi", "english", "malayalam"]
    },
    "hinengmar": {
        "path": "models/clip_finetuned_hindienglishmarathi_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishmarathi_real.pth",
        "subcategories": ["hindi", "english", "marathi"]
    },
    "hinengmei": {
        "path": "models/clip_finetuned_hindienglishmeitei_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishmeitei_real.pth",
        "subcategories": ["hindi", "english", "meitei"]
    },
    "hinengodi": {
        "path": "models/clip_finetuned_hindienglishodia_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishodia_real.pth",
        "subcategories": ["hindi", "english", "odia"]
    },
    "hinengpun": {
        "path": "models/clip_finetuned_hindienglishpunjabi_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishpunjabi_real.pth",
        "subcategories": ["hindi", "english", "punjabi"]
    },
    "hinengtam": {
        "path": "models/clip_finetuned_hindienglishtamil_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishtamil_real.pth",
        "subcategories": ["hindi", "english", "tamil"]
    },
    "hinengtel": {
        "path": "models/clip_finetuned_hindienglishtelugu_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishtelugu_real.pth",
        "subcategories": ["hindi", "english", "telugu"]
    },
    "hinengurd": {
        "path": "models/clip_finetuned_hindienglishurdu_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishurdu_real.pth",
        "subcategories": ["hindi", "english", "urdu"]
    },
    

}


# Set device to CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, preprocess = clip.load("ViT-B/32", device=device)

class CLIPFineTuner(torch.nn.Module):
    """
    Fine-tuning class for the CLIP model to adapt to specific tasks.
    
    Attributes:
        model (torch.nn.Module): The CLIP model to be fine-tuned.
        classifier (torch.nn.Linear): A linear classifier to map features to the desired number of classes.
    """
    def __init__(self, model, num_classes):
        """
        Initializes the fine-tuner with the CLIP model and classifier.

        Args:
            model (torch.nn.Module): The base CLIP model.
            num_classes (int): The number of target classes for classification.
        """
        super(CLIPFineTuner, self).__init__()
        self.model = model
        self.classifier = torch.nn.Linear(model.visual.output_dim, num_classes)

    def forward(self, x):
        """
        Forward pass for image classification.

        Args:
            x (torch.Tensor): Preprocessed input tensor for an image.

        Returns:
            torch.Tensor: Logits for each class.
        """
        with torch.no_grad():
            features = self.model.encode_image(x).float()  # Extract image features from CLIP model
        return self.classifier(features)  # Return class logits

class CLIPidentifier:
    def __init__(self):
        pass

    # Ensure model file exists; download directly if not
    def ensure_model(self, model_name):
        model_path = model_info[model_name]["path"]
        url = model_info[model_name]["url"]
        root_model_dir = "IndicPhotoOCR/script_identification/"
        model_path = os.path.join(root_model_dir, model_path)
        
        if not os.path.exists(model_path):
            print(f"Model not found locally. Downloading {model_name} from {url}...")
            response = requests.get(url, stream=True)
            os.makedirs(f"{root_model_dir}/models", exist_ok=True)
            with open(f"{model_path}", "wb") as f:
                f.write(response.content)
            print(f"Downloaded model for {model_name}.")
        
        return model_path

    # Prediction function to verify and load the model
    def identify(self, image_path, model_name):
        """
        Predicts the class of an input image using a fine-tuned CLIP model.

        Args:
            image_path (str): Path to the input image file.
            model_name (str): Name of the model (e.g., hineng, hinengpun, hinengguj) as specified in `model_info`.

        Returns:
            dict: Contains either `predicted_class` if successful or `error` if an exception occurs.

        Example usage:
            result = predict("sample_image.jpg", "hinengguj")
            print(result)  # Output might be {'predicted_class': 'hindi'}
        """
        try:
            # Validate model name and retrieve associated subcategories
            if model_name not in model_info:
                return {"error": "Invalid model name"}

            # Ensure the model file is downloaded and accessible
            model_path = self.ensure_model(model_name)


            subcategories = model_info[model_name]["subcategories"]
            num_classes = len(subcategories)

            # Load the fine-tuned model with the specified number of classes
            model_ft = CLIPFineTuner(clip_model, num_classes)
            model_ft.load_state_dict(torch.load(model_path, map_location=device))
            model_ft = model_ft.to(device)
            model_ft.eval()

            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            input_tensor = preprocess(image).unsqueeze(0).to(device)

            # Run the model and get the prediction
            outputs = model_ft(input_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = subcategories[predicted_idx.item()]

            return predicted_class

        except Exception as e:
            return {"error": str(e)}


# if __name__ == "__main__":
#     import argparse

#     # Argument parser for command line usage
#     parser = argparse.ArgumentParser(description="Image classification using CLIP fine-tuned model")
#     parser.add_argument("image_path", type=str, help="Path to the input image")
#     parser.add_argument("model_name", type=str, choices=model_info.keys(), help="Name of the model (e.g., hineng, hinengpun, hinengguj)")

#     args = parser.parse_args()

#     # Execute prediction with command line inputs
#     result = predict(args.image_path, args.model_name)
#     print(result)