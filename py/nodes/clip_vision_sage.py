from .sageAttention import sageattn_context, get_yaml_parameters
import comfy.ldm.modules.attention as comfyattn

class BlehCLIPVisionSage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_vision": ("CLIP_VISION",),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "yaml_parameters": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "YAML parameters for SageAttention (e.g., sageattn_allow_head_sizes: [72])"
                }),
            }
        }
    
    RETURN_TYPES = ("CLIP_VISION",)
    FUNCTION = "patch"
    CATEGORY = "hacks"

    def patch(self, clip_vision, enabled, yaml_parameters=""):
        if not enabled:
            return (clip_vision,)
        
        sage_kwargs = get_yaml_parameters(yaml_parameters)
        
        original_encode = clip_vision.encode_image
        
        def patched_encode_image(self, image, crop=True):
            with sageattn_context(enabled=True, **sage_kwargs):
                return original_encode(image, crop=crop)
        
        # Bind the patched method to the instance
        clip_vision.encode_image = patched_encode_image.__get__(clip_vision, type(clip_vision))
        
        return (clip_vision,)
