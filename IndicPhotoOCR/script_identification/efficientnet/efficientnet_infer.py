import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_v2_l
import os
import urllib.request
import hashlib
import tempfile
from pathlib import Path
from tqdm import tqdm

model_info = {
    "efficientnet_v2_l": {
        "path": "models/efficientnet",
        "url": "https://github.com/dikshant-sharma05/IndicPhotoOCR/releases/download/efficientnetv2-l-v1.0-test/efficientnetv2_l_script_id.pth",
        "filename": "efficientnetv2_l_script_id.pth",
        "sha256": "74b6f6e8f7f77b0d2dc1c6f0684bac718ac2b82247bcc393dad804fa4ea15709",
    }
}


def _verify_checksum(path, expected):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")


def download_model_from_release(model_name, save_path):
    """Download once, verifying existing and newly downloaded weights."""
    if model_name not in model_info:
        raise ValueError(f"Model '{model_name}' not found in model_info")
    model_data = model_info[model_name]
    save_path = Path(save_path).expanduser()
    if save_path.exists():
        _verify_checksum(save_path, model_data["sha256"])
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Download to a temporary file so interrupted downloads aren't cached as models.
    fd, temporary = tempfile.mkstemp(prefix=save_path.name + ".", suffix=".tmp",
                                      dir=save_path.parent)
    os.close(fd)
    try:
        print(f"Downloading model from {model_data['url']}...")
        with tqdm(desc="EfficientNetV2-L", unit="B", unit_scale=True,
                  unit_divisor=1024) as progress:
            def report_progress(block_count, block_size, total_size):
                if total_size > 0:
                    progress.total = total_size
                downloaded = block_count * block_size
                if total_size > 0:
                    downloaded = min(downloaded, total_size)
                progress.update(max(0, downloaded - progress.n))

            urllib.request.urlretrieve(model_data["url"], temporary,
                                       reporthook=report_progress)
        print("Download complete. Verifying checkpoint SHA-256...")
        _verify_checksum(temporary, model_data["sha256"])
        os.replace(temporary, save_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


class EfficientNetIdentifier:
    def __init__(self, checkpoint_path=None, classes=None, image_size=None,
                 device='cuda:0', model_name="efficientnet_v2_l"):
        """Load the verified 12-class V2-L inference checkpoint.

        Optional classes/image_size are compatibility arguments and must match
        checkpoint metadata. lang_hint in prediction methods is retained for
        the existing OCR interface; this model always predicts all 12 classes.
        """
        self.device = torch.device(device)
        if checkpoint_path is None:
            # Match the other identifiers: store weights beside this module.
            checkpoint_path = (Path(__file__).resolve().parent /
                               model_info[model_name]["path"] /
                               model_info[model_name]["filename"])
        download_model_from_release(model_name, checkpoint_path)
        # Older PyTorch releases require a string filename when mmap=True.
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu",
                                weights_only=True, mmap=True)
        if checkpoint.get("architecture") != "efficientnet_v2_l":
            raise ValueError("Expected an EfficientNetV2-L checkpoint")
        self.classes = checkpoint["classes"]
        checkpoint_size = checkpoint["image_size"]
        if len(self.classes) != 12 or checkpoint_size != 288:
            raise ValueError("Expected the verified 12-class, 288px checkpoint")
        if classes is not None and list(classes) != self.classes:
            raise ValueError("Supplied classes differ from checkpoint class order")
        if image_size is not None and image_size != checkpoint_size:
            raise ValueError("Supplied image_size differs from checkpoint metadata")
        image_size = checkpoint_size

        self.model = efficientnet_v2_l()
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        del checkpoint
        self.model.to(self.device)
        self.model.eval()

        # Same RGB / resize / normalization as the verified V2-L runner.
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def identify(self, cropped_path, lang_hint="auto", device=None):
        target_device = device if device else self.device
        
        try:
            image = Image.open(cropped_path).convert('RGB')
            tensor = self.transform(image).unsqueeze(0).to(target_device)
            
            with torch.no_grad():
                outputs = self.model(tensor)
                _, predicted = torch.max(outputs, 1)
                
            return self.classes[predicted.item()]
        except Exception as e:
            print(f"Error identifying {cropped_path}: {e}")
            return None

    def identify_batch(self, cropped_paths, lang_hint="auto", device=None, batch_size=32):
        target_device = device if device else self.device
        results = []
        
        for i in range(0, len(cropped_paths), batch_size):
            batch_paths = cropped_paths[i:i + batch_size]
            batch_tensors = []
            valid_indices = []
            batch_results = [self.classes[0]] * len(batch_paths)
            
            for index, path in enumerate(batch_paths):
                try:
                    image = Image.open(path).convert('RGB')
                    batch_tensors.append(self.transform(image))
                    valid_indices.append(index)
                except Exception:
                    # Leave the fallback at this crop's original position.
                    pass
                    
            if not batch_tensors:
                results.extend(batch_results)
                continue
                
            input_batch = torch.stack(batch_tensors).to(target_device)
            
            with torch.no_grad():
                outputs = self.model(input_batch)
                _, predicted = torch.max(outputs, 1)
                
            # Map predictions to class names
            batch_langs = [self.classes[idx.item()] for idx in predicted]
            for index, label in zip(valid_indices, batch_langs):
                batch_results[index] = label
            results.extend(batch_results)
        return results

    def identify_top_k(self, cropped_path, top_k=3, lang_hint="auto", device=None):
        target_device = device if device else self.device
        
        try:
            image = Image.open(cropped_path).convert('RGB')
            tensor = self.transform(image).unsqueeze(0).to(target_device)
            
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.classes)), dim=1)
                
            top_classes = [self.classes[idx.item()] for idx in top_indices[0]]
            top_scores = [prob.item() for prob in top_probs[0]]
            return list(zip(top_classes, top_scores))
        except Exception as e:
            print(f"Error identifying top_k {cropped_path}: {e}")
            return [(self.classes[0], 1.0)]

    def identify_batch_top_k(self, cropped_paths, top_k=3, lang_hint="auto", device=None, batch_size=32):
        target_device = device if device else self.device
        results = []
        
        for i in range(0, len(cropped_paths), batch_size):
            batch_paths = cropped_paths[i:i + batch_size]
            batch_tensors = []
            valid_indices = []
            batch_results = [[(self.classes[0], 1.0)] for _ in batch_paths]
            
            for index, path in enumerate(batch_paths):
                try:
                    image = Image.open(path).convert('RGB')
                    batch_tensors.append(self.transform(image))
                    valid_indices.append(index)
                except Exception:
                    pass
                    
            if not batch_tensors:
                results.extend(batch_results)
                continue
                
            input_batch = torch.stack(batch_tensors).to(target_device)
            
            with torch.no_grad():
                outputs = self.model(input_batch)
                probs = torch.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.classes)), dim=1)
                
            for index, indices, scores in zip(valid_indices, top_indices, top_probs):
                item_top = [(self.classes[idx.item()], prob.item()) for idx, prob in zip(indices, scores)]
                batch_results[index] = item_top
            results.extend(batch_results)
        return results
