import torch
import contextlib
from comfy.clip_vision import clip_preprocess, Output
import comfy.model_management
from .sageAttention import (
    sageattn_context,
    save_attentions,
    get_yaml_parameters,
    orig_attentions,
    sageattention
)

class BlehCLIPVisionSage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_vision": ("CLIP_VISION",),
                "image": ("IMAGE",),
                "crop": (["center", "none"], {"default": "center"}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "yaml_parameters": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "YAML formatted parameters for SageAttention"
                }),
            },
        }

    RETURN_TYPES = ("CLIP_VISION_OUTPUT",)
    FUNCTION = "encode"
    CATEGORY = "hacks"

    def encode(self, clip_vision, image, crop, enabled, yaml_parameters=""):
        if enabled and not sageattention:
            raise RuntimeError("SageAttention required: pip install sageattention")

        # Convert parameters
        extra_params = get_yaml_parameters(yaml_parameters)
        crop_bool = crop == "center"

        # Ensure original attentions are preserved
        if enabled and not orig_attentions:
            orig_attentions.update(save_attentions())

        # Apply SageAttention during encoding
        with sageattn_context(
            enabled=enabled,
            orig_attentions=orig_attentions,
            **extra_params
        ):
            output = clip_vision.encode_image(image, crop=crop_bool)

        return (output,)

