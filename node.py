import os
import torch
import folder_paths
import torchvision.transforms as T

from transformers import (
    AutoModelForImageClassification,
    ViTImageProcessor,
)


def load_falcon_model(checkpoint_path):
    model = AutoModelForImageClassification.from_pretrained(checkpoint_path)
    falcon_processor = ViTImageProcessor.from_pretrained(checkpoint_path)

    return model, falcon_processor


class FalconAISafetyChecker:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        paths = []
        for search_path in folder_paths.get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                for root, dirs, files in os.walk(search_path, followlinks=True):
                    if "model_index.json" in files or "config.json" in files:
                        paths.append(root)

        return {
            "required": {
                "model_path": (paths,),
                "image": ("IMAGE",),
                "score": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.001,
                        "display": "safety_threshold",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "run"

    CATEGORY = "FalconAISafetyChecker"

    def run(self, model_path, image, score):
        if isinstance(image, list):
            image = image[0]

        if isinstance(image, torch.Tensor):
            image = T.ToPILImage()(image.permute(2, 0, 1))

        model, falcon_processor = load_falcon_model(model_path)
        with torch.no_grad():
            inputs = falcon_processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_label = logits.argmax(-1).item()
            label = model.config.id2label[predicted_label]
            probabilities = logits.softmax(dim=-1)
            if label == "normal" and probabilities[0][0] > score:
                return label

        return label


NODE_CLASS_MAPPINGS = {"FalconAISafetyChecker": FalconAISafetyChecker}

NODE_DISPLAY_NAME_MAPPINGS = {"FalconAISafetyChecker": "FalconAI Safety Checker"}
