import os
import torch
import logging
import numpy as np

import folder_paths
import torchvision.transforms as T

from pathlib import Path
from typing import Tuple
from functools import partial
from torch import nn
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import (
    AutoModelForImageClassification,
    ViTImageProcessor,
    CLIPImageProcessor,
)

import comfy.model_management


logger = logging.getLogger(__name__)


def generate_black_image(image):
    if torch.is_tensor(image):
        return torch.zeros_like(image)
    else:
        try:
            np_image = np.asarray(image)
            return torch.zeros(np_image.shape)
        except Exception:
            pass
    return torch.zeros(image.shape)


# custom safety checker
concepts = [
    "sexual",
    "nude",
    "sex",
    "18+",
    "naked",
    "nsfw",
    "porn",
    "dick",
    "vagina",
    "naked person (approximation)",
    "explicit content",
    "uncensored",
    "fuck",
    "nipples",
    "nipples (approximation)",
    "naked breasts",
    "areola",
]
special_concepts = ["small girl (approximation)", "young child", "young girl"]
s_treshold = -0.028


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


@torch.no_grad()
def custom_forward(self, clip_input, images):

    global s_treshold

    pooled_output = self.vision_model(clip_input)[1]  # pooled_output
    image_embeds = self.visual_projection(pooled_output)

    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    special_cos_dist = (
        cosine_distance(image_embeds, self.special_care_embeds).cpu().float().numpy()
    )
    cos_dist = cosine_distance(image_embeds, self.concept_embeds).cpu().float().numpy()

    result = []
    batch_size = image_embeds.shape[0]
    for i in range(batch_size):
        result_img = {
            "special_scores": {},
            "special_care": [],
            "concept_scores": {},
            "bad_concepts": [],
        }

        # increase this value to create a stronger `nfsw` filter
        # at the cost of increasing the possibility of filtering benign images
        # s_treshold = -0.028

        for concept_idx in range(len(special_cos_dist[0])):
            concept_cos = special_cos_dist[i][concept_idx]
            concept_threshold = self.special_care_embeds_weights[concept_idx].item()
            result_img["special_scores"][concept_idx] = round(
                concept_cos - concept_threshold + s_treshold, 3
            )
            if result_img["special_scores"][concept_idx] > 0:
                result_img["special_care"].append(
                    {concept_idx, result_img["special_scores"][concept_idx]}
                )
                s_treshold = 0.01
                print("Special concept matched:", special_concepts[concept_idx])

        for concept_idx in range(len(cos_dist[0])):
            concept_cos = cos_dist[i][concept_idx]
            concept_threshold = self.concept_embeds_weights[concept_idx].item()
            result_img["concept_scores"][concept_idx] = round(
                concept_cos - concept_threshold + s_treshold, 3
            )
            if result_img["concept_scores"][concept_idx] > 0:
                result_img["bad_concepts"].append(concept_idx)
                print("NSFW concept found:", concepts[concept_idx])

        result.append(result_img)

    has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]

    for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
        if has_nsfw_concept:
            if torch.is_tensor(images) or torch.is_tensor(images[0]):
                images[idx] = torch.zeros_like(images[idx])  # black image
            else:
                images[idx] = np.zeros(images[idx].shape)  # black image

    if any(has_nsfw_concepts):
        logger.warning(
            "Potential NSFW content was detected in one or more images. A black image will be returned instead."
            " Try again with a different prompt and/or seed."
        )

    return images, has_nsfw_concepts


def load_falcons_model(
    checkpoint_path: str | Path,
) -> Tuple[AutoModelForImageClassification, ViTImageProcessor]:
    model = AutoModelForImageClassification.from_pretrained(checkpoint_path)
    falcons_processor = ViTImageProcessor.from_pretrained(checkpoint_path)

    return model, falcons_processor


def load_compvis_model(
    checkpoint_path: str | Path,
) -> Tuple[StableDiffusionSafetyChecker, CLIPImageProcessor]:
    model = StableDiffusionSafetyChecker.from_pretrained(
        checkpoint_path,
        torch_dtype=(
            torch.float16
            if comfy.model_management.get_torch_device().__str__() == "cuda"
            else torch.float32
        ),
    ).to(comfy.model_management.get_torch_device())
    clip_processor = ViTImageProcessor.from_pretrained(checkpoint_path)

    return model, clip_processor


class FalconsAISafetyChecker:

    @classmethod
    def INPUT_TYPES(s):
        models = []
        for search_path in folder_paths.get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                for root, dirs, files in os.walk(search_path, followlinks=True):
                    models.extend(
                    dir for dir in dirs
                    if "model_index.json" in os.listdir(os.path.join(root, dir)) or
                    "config.json" in os.listdir(os.path.join(root, dir))
                    )
                    break  # Only need the top-level directories

        return {
            "required": {
                "model_name": (models,),
                "image": ("IMAGE",),
                "safety_threshold": (
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

    RETURN_TYPES = ("STRING", "IMAGE")

    FUNCTION = "run"

    CATEGORY = "SafetyChecker"

    TITLE = "Falcons AI Safety Checker"

    def run(self, model_name, image, safety_threshold):
        model_path = f"{folder_paths.get_folder_paths('diffusers')[0]}/{model_name}"
        if (
            isinstance(image, list)
            or isinstance(image, torch.Tensor)
            and image.dim() == 4
        ):
            image = image[0]

        model, falcons_processor = load_falcons_model(model_path)
        with torch.no_grad():
            inputs = falcons_processor(
                images=T.ToPILImage()(image.permute(2, 0, 1)), return_tensors="pt"
            )
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_label = logits.argmax(-1).item()
            label = model.config.id2label[predicted_label]
            probabilities = logits.softmax(dim=-1)
            if label == "normal" and probabilities[0][0] >= safety_threshold:
                return (label, (image,))

            label = "nsfw" if label == "normal" else label

        return (label, (generate_black_image(image),))


class CompVisSafetyChecker:

    @classmethod
    def INPUT_TYPES(s):
        models = []
        for search_path in folder_paths.get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                for root, dirs, files in os.walk(search_path, followlinks=True):
                    models.extend(
                    dir for dir in dirs
                    if "model_index.json" in os.listdir(os.path.join(root, dir)) or
                    "config.json" in os.listdir(os.path.join(root, dir))
                    )
                    break  # Only need the top-level directories

        return {
            "required": {
                "model_name": (models,),
                "image": ("IMAGE",),
                "safety_threshold": (
                    "FLOAT",
                    {
                        "default": -0.028,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.001,
                        "round": 0.0001,
                        "display": "safety_threshold",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")

    FUNCTION = "run"

    CATEGORY = "SafetyChecker"

    TITLE = "CompVis Safety Checker"

    def run(self, model_name, image, safety_threshold):
        global s_treshold

        model_path = f"{folder_paths.get_folder_paths('diffusers')[0]}/{model_name}"
        if (
            isinstance(image, list)
            or isinstance(image, torch.Tensor)
            and image.dim() == 4
        ):
            image = image[0]

        if isinstance(image, torch.Tensor):
            image = image.permute(2, 0, 1).numpy()

        model, clip_processor = load_compvis_model(model_path)
        s_treshold = safety_threshold
        model.forward = partial(custom_forward, self=model)

        with torch.no_grad():
            inputs = clip_processor(image, return_tensors="pt").to(
                comfy.model_management.get_torch_device()
            )
            image, has_nsfw_concept = model(
                images=[image],
                clip_input=inputs.pixel_values.to(
                    torch.float16
                    if comfy.model_management.get_torch_device().__str__() == "cuda"
                    else torch.float32
                ),
            )

            image = torch.from_numpy(image[0]).permute(1, 2, 0)

            label = "normal" if not all(has_nsfw_concept) else "nsfw"

        return (label, (image,))


NODE_CLASS_MAPPINGS = {
    "FalconsAISafetyChecker": FalconsAISafetyChecker,
    "CompVisSafetyChecker": CompVisSafetyChecker,
}
